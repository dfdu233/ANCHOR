"""Processor-aware matched-source alignment plus a real wrong-center control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import encode_array, load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.infer_ce import _structure_metrics, decoded_label_index, resize_image
from corrected_sgta.methods import feddg_frequency_interpolation
from corrected_sgta.models_alignment import load_alignment_adapter
from corrected_sgta.protocol_v2 import (
    CACHE_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    ProtocolError,
    build_prompt,
    file_sha256,
    ground_truth_index,
    labels_for_sample,
    protocol_fingerprint,
    resolve_image,
    task_kind,
    validate_dataset,
)
from corrected_sgta.provenance_v2 import code_identity, model_identity
from corrected_sgta.source_bank_v2 import (
    cosine_distance,
    entries_for_modality,
    load_feature_centers,
    load_manifest,
    normalize_modality,
    sha256_file,
    verify_source_artifacts,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
ALIGNMENT_CACHE_VERSION = "sgta-alignment-evidence-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("hulu", "llava"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--visual-centers", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l-grid", type=float, nargs="+", default=(0.01, 0.03, 0.05, 0.1, 0.2))
    parser.add_argument("--source-ratio", type=float, default=0.0)
    parser.add_argument("--max-views", type=int, default=2)
    parser.add_argument("--min-relative-closure", type=float, default=0.0)
    parser.add_argument("--min-style-psnr", type=float, default=20.0)
    parser.add_argument("--min-edge-correlation", type=float, default=0.90)
    parser.add_argument("--decode-labels", action="store_true")
    parser.add_argument("--decode-max-new-tokens", type=int, default=8)
    parser.add_argument("--allow-unknown-modality", action="store_true")
    return parser.parse_args()


@lru_cache(maxsize=None)
def load_amplitude(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def sample_modality(sample: dict, dataset: Path) -> str | None:
    explicit = normalize_modality(sample.get("modality"))
    if explicit is not None:
        return explicit
    if "cxr" in dataset.name.lower():
        return "xray"
    return None


def safe_structure(metrics: dict, args: argparse.Namespace) -> bool:
    return (
        metrics.get("psnr") is not None
        and float(metrics["psnr"]) >= args.min_style_psnr
        and metrics.get("edge_correlation") is not None
        and float(metrics["edge_correlation"]) >= args.min_edge_correlation
    )


def choose_wrong_entry(target: dict, controls: list[dict], key: str) -> dict | None:
    """Choose a label-free, deterministic, preferably cross-modality control."""

    different_modality = [
        item
        for item in controls
        if item["source_id"] != target["source_id"]
        and normalize_modality(item.get("modality"))
        != normalize_modality(target.get("modality"))
    ]
    candidates = different_modality or [
        item for item in controls if item["source_id"] != target["source_id"]
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: hashlib.sha256(
            f"{key}:{target['source_id']}:{item['source_id']}".encode()
        ).hexdigest(),
    )


def select_alignment_views(
    image: Image.Image,
    matched_entries: list[dict],
    control_entries: list[dict],
    feature_centers: dict[str, np.ndarray],
    adapter,
    args: argparse.Namespace,
    control_key: str,
) -> tuple[list[dict], np.ndarray, list[dict]]:
    original_feature = adapter.visual_features([image])[0]
    targets = [item for item in matched_entries if item["source_id"] in feature_centers]
    wrong_by_target = {
        item["source_id"]: choose_wrong_entry(
            item,
            [control for control in control_entries if control["source_id"] in feature_centers],
            control_key,
        )
        for item in targets
    }

    needed_amplitudes = {item["source_id"]: item for item in targets}
    for item in wrong_by_target.values():
        if item is not None:
            needed_amplitudes[item["source_id"]] = item

    transformed_pool: dict[tuple[str, float], dict] = {}
    ordered_keys = []
    transformed_images = []
    for source_id, entry in sorted(needed_amplitudes.items()):
        for low_frequency_ratio in args.l_grid:
            transformed = feddg_frequency_interpolation(
                image,
                load_amplitude(entry["amplitude_file"]),
                low_frequency_ratio=float(low_frequency_ratio),
                source_ratio=float(args.source_ratio),
            )
            key = (source_id, float(low_frequency_ratio))
            transformed_pool[key] = {
                "entry": entry,
                "image": transformed,
                "structure": _structure_metrics(image, transformed),
            }
            ordered_keys.append(key)
            transformed_images.append(transformed)
    if transformed_images:
        visual = adapter.visual_features(transformed_images)
        for key, feature in zip(ordered_keys, visual):
            transformed_pool[key]["visual_feature"] = feature

    candidates = []
    for target in targets:
        target_id = target["source_id"]
        center = feature_centers[target_id]
        distance_before = cosine_distance(original_feature, center)
        wrong = wrong_by_target[target_id]
        for low_frequency_ratio in args.l_grid:
            ratio = float(low_frequency_ratio)
            matched = transformed_pool[(target_id, ratio)]
            distance_after = cosine_distance(matched["visual_feature"], center)
            item = {
                "source_id": target_id,
                "entry": target,
                "image": matched["image"],
                "visual_feature": matched["visual_feature"],
                "low_frequency_ratio": ratio,
                "source_ratio": float(args.source_ratio),
                "structure": matched["structure"],
                "safe": safe_structure(matched["structure"], args),
                "distance_before": distance_before,
                "distance_after": distance_after,
                "absolute_closure": distance_before - distance_after,
                "relative_closure": (distance_before - distance_after) / max(distance_before, 1e-12),
                "wrong_source_id": None,
                "wrong_entry": None,
                "wrong_image": None,
                "wrong_visual_feature": None,
                "wrong_structure": None,
                "wrong_safe": False,
                "wrong_distance_after": None,
                "wrong_relative_closure": None,
            }
            if wrong is not None:
                wrong_view = transformed_pool[(wrong["source_id"], ratio)]
                wrong_after = cosine_distance(wrong_view["visual_feature"], center)
                item.update(
                    {
                        "wrong_source_id": wrong["source_id"],
                        "wrong_entry": wrong,
                        "wrong_image": wrong_view["image"],
                        "wrong_visual_feature": wrong_view["visual_feature"],
                        "wrong_structure": wrong_view["structure"],
                        "wrong_safe": safe_structure(wrong_view["structure"], args),
                        "wrong_distance_after": wrong_after,
                        "wrong_relative_closure": (distance_before - wrong_after)
                        / max(distance_before, 1e-12),
                    }
                )
            candidates.append(item)

    selected = []
    for source_id in sorted({item["source_id"] for item in candidates}):
        safe = [item for item in candidates if item["source_id"] == source_id and item["safe"]]
        if not safe:
            continue
        best = min(safe, key=lambda item: (item["distance_after"], item["low_frequency_ratio"]))
        if best["relative_closure"] >= args.min_relative_closure:
            selected.append(best)
    selected.sort(key=lambda item: (-item["relative_closure"], item["source_id"]))
    return selected[: args.max_views], original_feature, candidates


def candidate_metadata(item: dict, selected: bool) -> dict:
    return {
        "source_id": item["source_id"],
        "modality": item["entry"]["modality"],
        "amplitude_file": item["entry"]["amplitude_file"],
        "amplitude_sha256": item["entry"]["amplitude_sha256"],
        "low_frequency_ratio": item["low_frequency_ratio"],
        "source_ratio": item["source_ratio"],
        "structure": item["structure"],
        "safe": item["safe"],
        "selected": selected,
        "visual_distance_before": item["distance_before"],
        "visual_distance_after": item["distance_after"],
        "absolute_closure": item["absolute_closure"],
        "relative_closure": item["relative_closure"],
        "wrong_source_id": item["wrong_source_id"],
        "wrong_modality": None if item["wrong_entry"] is None else item["wrong_entry"]["modality"],
        "wrong_amplitude_sha256": None
        if item["wrong_entry"] is None
        else item["wrong_entry"]["amplitude_sha256"],
        "wrong_structure": item["wrong_structure"],
        "wrong_safe": item["wrong_safe"],
        "wrong_distance_after": item["wrong_distance_after"],
        "wrong_relative_closure": item["wrong_relative_closure"],
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    source_bank_sha256 = sha256_file(args.source_bank)
    manifest = load_manifest(args.source_bank)
    verified_artifacts = verify_source_artifacts(manifest)
    visual_meta_path = args.visual_centers.with_suffix(args.visual_centers.suffix + ".meta.json")
    visual_meta_sha256 = sha256_file(visual_meta_path)
    center_meta, feature_centers = load_feature_centers(
        args.visual_centers,
        expected_model=args.model,
        expected_source_bank_sha256=source_bank_sha256,
    )
    project_root = Path(__file__).resolve().parents[1]
    config = {
        "alignment_cache_version": ALIGNMENT_CACHE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": args.model,
        "model_identity": model_identity(args.model),
        "code_identity": code_identity(project_root),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "seed": args.seed,
        "max_samples": args.max_samples,
        "subset_order": "sha256(seed:qid)",
        "max_image_side": args.max_image_side,
        "source_bank": str(args.source_bank.resolve()),
        "source_bank_sha256": source_bank_sha256,
        "verified_source_artifacts": verified_artifacts,
        "visual_centers": str(args.visual_centers.resolve()),
        "visual_centers_sha256": sha256_file(args.visual_centers),
        "visual_centers_meta_sha256": visual_meta_sha256,
        "l_grid": list(args.l_grid),
        "source_ratio": args.source_ratio,
        "max_views": args.max_views,
        "min_relative_closure": args.min_relative_closure,
        "min_style_psnr": args.min_style_psnr,
        "min_edge_correlation": args.min_edge_correlation,
        "allow_unknown_modality": args.allow_unknown_modality,
        "decode_labels": args.decode_labels,
        "decode_max_new_tokens": args.decode_max_new_tokens,
        "wrong_control": "deterministic cross-modality amplitude intervention; target center unchanged",
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "alignment_cache_version": ALIGNMENT_CACHE_VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "dataset_validation": validation,
        "source_bank": manifest,
        "visual_center_metadata": center_meta,
    }
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise RuntimeError(f"metadata mismatch; choose a new output: {args.output}")
    else:
        atomic_json(metadata_path, metadata)
    repair = repair_truncated_jsonl_tail(args.output)
    if repair["action"] != "none":
        print(f"cache tail repair: {repair}", flush=True)
    saved = load_successful_qids(args.output, fingerprint)

    target_rows = []
    for sample in rows:
        try:
            if task_kind(sample) == "open":
                continue
            labels_for_sample(sample)
            ground_truth_index(sample)
            if resolve_image(sample.get("img_name", "")) is None:
                continue
            target_rows.append(sample)
        except ProtocolError:
            continue
    target_rows.sort(
        key=lambda sample: hashlib.sha256(f"{args.seed}:{sample['qid']}".encode()).hexdigest()
    )
    if args.max_samples:
        target_rows = target_rows[: args.max_samples]
    eligible = [row for row in target_rows if str(row["qid"]) not in saved]
    print(
        f"alignment={ALIGNMENT_CACHE_VERSION} fingerprint={fingerprint[:12]} "
        f"eligible={len(eligible)} cached={len(saved)}",
        flush=True,
    )
    if not eligible:
        return

    adapter = load_alignment_adapter(args.model)
    errors = 0
    formal_controls = [
        item
        for item in manifest.get("entries", [])
        if item.get("formal") and item["source_id"] in feature_centers
    ]
    try:
        with args.output.open("a") as output:
            for sample in tqdm(eligible, desc=f"alignment {args.model}"):
                try:
                    image_path = resolve_image(sample.get("img_name", ""))
                    assert image_path is not None
                    with Image.open(image_path) as source:
                        image = resize_image(source, args.max_image_side)
                    modality = sample_modality(sample, args.dataset)
                    matched_entries = (
                        formal_controls
                        if modality is None and args.allow_unknown_modality
                        else entries_for_modality(manifest, modality, formal_only=True)
                        if modality is not None
                        else []
                    )
                    selected, original_visual, candidates = select_alignment_views(
                        image,
                        matched_entries,
                        formal_controls,
                        feature_centers,
                        adapter,
                        args,
                        control_key=f"{args.seed}:{sample['qid']}:{args.model}",
                    )
                    controls = [item for item in selected if item["wrong_image"] is not None and item["wrong_safe"]]
                    style_images = [image] + [item["image"] for item in selected] + [
                        item["wrong_image"] for item in controls
                    ]
                    style_names = ["original"] + [
                        f"matched_{item['source_id']}_l{item['low_frequency_ratio']:g}"
                        for item in selected
                    ] + [
                        f"wrong_{item['wrong_source_id']}_to_{item['source_id']}_l{item['low_frequency_ratio']:g}"
                        for item in controls
                    ]
                    style_roles = ["original"] + ["matched"] * len(selected) + ["wrong_control"] * len(controls)
                    labels = labels_for_sample(sample)
                    prompt = build_prompt(sample)
                    evidence = adapter.forward_ce(style_images, prompt, labels)
                    decoded_text = (
                        adapter.decode_ce(style_images, prompt, max_new_tokens=args.decode_max_new_tokens)
                        if args.decode_labels
                        else None
                    )
                    decoded_prediction = (
                        [decoded_label_index(text, labels, sample) for text in decoded_text]
                        if decoded_text is not None
                        else None
                    )
                    selected_ids = {id(item) for item in selected}
                    visual_matrix = np.stack(
                        [original_visual]
                        + [item["visual_feature"] for item in selected]
                        + [item["wrong_visual_feature"] for item in controls]
                    )
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "alignment_cache_version": ALIGNMENT_CACHE_VERSION,
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "qid": sample["qid"],
                        "img_name": sample.get("img_name", ""),
                        "modality": modality,
                        "question_type": task_kind(sample),
                        "labels": list(labels),
                        "gt_index": ground_truth_index(sample),
                        "style_names": style_names,
                        "style_roles": style_roles,
                        "style_target_source_ids": ["original"]
                        + [item["source_id"] for item in selected]
                        + [item["source_id"] for item in controls],
                        "style_amplitude_source_ids": ["original"]
                        + [item["source_id"] for item in selected]
                        + [item["wrong_source_id"] for item in controls],
                        "style_logits": [item.logits.tolist() for item in evidence],
                        "style_sequence_nll": [
                            None if item.sequence_nll is None else item.sequence_nll.tolist()
                            for item in evidence
                        ],
                        "style_language_features": encode_array(np.stack([item.features for item in evidence])),
                        "style_visual_features": encode_array(visual_matrix),
                        "alignment_candidates": [
                            candidate_metadata(item, id(item) in selected_ids) for item in candidates
                        ],
                        "fallback_to_original": len(selected) == 0,
                        "style_decoded_text": decoded_text,
                        "style_decoded_prediction": decoded_prediction,
                    }
                except Exception as exc:
                    errors += 1
                    traceback.print_exc()
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "alignment_cache_version": ALIGNMENT_CACHE_VERSION,
                        "fingerprint": fingerprint,
                        "status": "error",
                        "qid": sample.get("qid"),
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                    }
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        gc.collect()
                        torch.cuda.empty_cache()
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
                output.flush()
    finally:
        adapter.close()
    print(f"finished rows={len(eligible)} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
