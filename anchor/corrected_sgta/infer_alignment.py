"""Processor-aware source alignment with one cached VLM load per run."""

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

from corrected_sgta.cache import (
    encode_array,
    load_successful_qids,
    repair_truncated_jsonl_tail,
)
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
from corrected_sgta.source_bank import (
    cosine_distance,
    entries_for_modality,
    load_feature_centers,
    load_manifest,
    normalize_modality,
    sha256_file,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
ALIGNMENT_CACHE_VERSION = "sgta-alignment-evidence-v1"


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
    parser.add_argument(
        "--l-grid", type=float, nargs="+", default=(0.01, 0.03, 0.05, 0.1, 0.2)
    )
    parser.add_argument("--source-ratio", type=float, default=0.0)
    parser.add_argument("--max-views", type=int, default=2)
    parser.add_argument("--min-relative-closure", type=float, default=0.0)
    parser.add_argument("--min-style-psnr", type=float, default=20.0)
    parser.add_argument("--min-edge-correlation", type=float, default=0.90)
    parser.add_argument("--decode-labels", action="store_true")
    parser.add_argument("--decode-max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--allow-unknown-modality",
        action="store_true",
        help="route an unknown modality over all formal source entries",
    )
    return parser.parse_args()


@lru_cache(maxsize=None)
def load_amplitude(path: str) -> np.ndarray:
    return np.load(path)


def sample_modality(sample: dict, dataset: Path) -> str | None:
    explicit = normalize_modality(sample.get("modality"))
    if explicit is not None:
        return explicit
    if "cxr" in dataset.name.lower():
        return "xray"
    return None


def safe_structure(metrics: dict, args: argparse.Namespace) -> bool:
    psnr = metrics.get("psnr")
    edge = metrics.get("edge_correlation")
    return (
        psnr is not None
        and float(psnr) >= args.min_style_psnr
        and edge is not None
        and float(edge) >= args.min_edge_correlation
    )


def select_alignment_views(
    image: Image.Image,
    modality: str | None,
    entries: list[dict],
    feature_centers: dict[str, np.ndarray],
    adapter,
    args: argparse.Namespace,
) -> tuple[list[dict], np.ndarray, list[dict]]:
    original_feature = adapter.visual_features([image])[0]
    candidates = []
    for entry in entries:
        source_id = entry["source_id"]
        if source_id not in feature_centers:
            continue
        center = feature_centers[source_id]
        before = cosine_distance(original_feature, center)
        source_candidates = []
        for low_frequency_ratio in args.l_grid:
            transformed = feddg_frequency_interpolation(
                image,
                load_amplitude(entry["amplitude_file"]),
                low_frequency_ratio=float(low_frequency_ratio),
                source_ratio=float(args.source_ratio),
            )
            structure = _structure_metrics(image, transformed)
            source_candidates.append(
                {
                    "source_id": source_id,
                    "entry": entry,
                    "image": transformed,
                    "low_frequency_ratio": float(low_frequency_ratio),
                    "structure": structure,
                    "safe": safe_structure(structure, args),
                }
            )
        visual = adapter.visual_features([item["image"] for item in source_candidates])
        for item, feature in zip(source_candidates, visual):
            after = cosine_distance(feature, center)
            item["visual_feature"] = feature
            item["distance_before"] = before
            item["distance_after"] = after
            item["absolute_closure"] = before - after
            item["relative_closure"] = (before - after) / max(before, 1e-12)
            candidates.append(item)

    selected = []
    for source_id in sorted({item["source_id"] for item in candidates}):
        safe = [
            item
            for item in candidates
            if item["source_id"] == source_id and item["safe"]
        ]
        if not safe:
            continue
        best = min(
            safe,
            key=lambda item: (
                item["distance_after"],
                item["low_frequency_ratio"],
            ),
        )
        if best["relative_closure"] >= args.min_relative_closure:
            selected.append(best)
    selected.sort(
        key=lambda item: (-item["relative_closure"], item["source_id"])
    )
    selected = selected[: args.max_views]

    source_ids = sorted(feature_centers)
    for item in candidates:
        if len(source_ids) <= 1:
            item["shuffled_source_id"] = None
            item["shuffled_distance_after"] = None
            item["shuffled_relative_closure"] = None
            continue
        index = source_ids.index(item["source_id"])
        shuffled_id = source_ids[(index + 1) % len(source_ids)]
        shuffled_center = feature_centers[shuffled_id]
        shuffled_before = cosine_distance(original_feature, shuffled_center)
        shuffled_after = cosine_distance(item["visual_feature"], shuffled_center)
        item["shuffled_source_id"] = shuffled_id
        item["shuffled_distance_after"] = shuffled_after
        item["shuffled_relative_closure"] = (
            shuffled_before - shuffled_after
        ) / max(shuffled_before, 1e-12)
    return selected, original_feature, candidates


def candidate_metadata(item: dict, selected: bool) -> dict:
    return {
        "source_id": item["source_id"],
        "modality": item["entry"]["modality"],
        "amplitude_file": item["entry"]["amplitude_file"],
        "amplitude_sha256": item["entry"]["amplitude_sha256"],
        "low_frequency_ratio": item["low_frequency_ratio"],
        "source_ratio": item.get("source_ratio", 0.0),
        "structure": item["structure"],
        "safe": item["safe"],
        "selected": selected,
        "visual_distance_before": item["distance_before"],
        "visual_distance_after": item["distance_after"],
        "absolute_closure": item["absolute_closure"],
        "relative_closure": item["relative_closure"],
        "shuffled_source_id": item["shuffled_source_id"],
        "shuffled_distance_after": item["shuffled_distance_after"],
        "shuffled_relative_closure": item["shuffled_relative_closure"],
    }


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    manifest = load_manifest(args.source_bank)
    center_meta, feature_centers = load_feature_centers(
        args.visual_centers, expected_model=args.model
    )
    config = {
        "model": args.model,
        "dataset_sha256": file_sha256(args.dataset),
        "seed": args.seed,
        "max_image_side": args.max_image_side,
        "source_bank_sha256": sha256_file(args.source_bank),
        "visual_centers_sha256": sha256_file(args.visual_centers),
        "l_grid": list(args.l_grid),
        "source_ratio": args.source_ratio,
        "max_views": args.max_views,
        "min_relative_closure": args.min_relative_closure,
        "min_style_psnr": args.min_style_psnr,
        "min_edge_correlation": args.min_edge_correlation,
        "allow_unknown_modality": args.allow_unknown_modality,
        "decode_labels": args.decode_labels,
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
        metadata_path.write_text(json.dumps(metadata, indent=2))
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
        key=lambda sample: hashlib.sha256(
            f"{args.seed}:{sample['qid']}".encode()
        ).hexdigest()
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
    try:
        with args.output.open("a") as output:
            for sample in tqdm(eligible, desc=f"alignment {args.model}"):
                try:
                    image_path = resolve_image(sample.get("img_name", ""))
                    assert image_path is not None
                    with Image.open(image_path) as source:
                        image = resize_image(source, args.max_image_side)
                    modality = sample_modality(sample, args.dataset)
                    if modality is None and not args.allow_unknown_modality:
                        entries = []
                    else:
                        entries = entries_for_modality(
                            manifest, modality, formal_only=True
                        )
                    selected, original_visual, candidates = select_alignment_views(
                        image,
                        modality,
                        entries,
                        feature_centers,
                        adapter,
                        args,
                    )
                    style_images = [image] + [item["image"] for item in selected]
                    style_names = ["original"] + [
                        f"aligned_{item['source_id']}_l{item['low_frequency_ratio']:g}"
                        for item in selected
                    ]
                    labels = labels_for_sample(sample)
                    evidence = adapter.forward_ce(
                        style_images, build_prompt(sample), labels
                    )
                    decoded_text = (
                        adapter.decode_ce(
                            style_images,
                            build_prompt(sample),
                            max_new_tokens=args.decode_max_new_tokens,
                        )
                        if args.decode_labels
                        else None
                    )
                    decoded_prediction = (
                        [
                            decoded_label_index(text, labels, sample)
                            for text in decoded_text
                        ]
                        if decoded_text is not None
                        else None
                    )
                    selected_ids = {id(item) for item in selected}
                    visual_matrix = np.stack(
                        [original_visual]
                        + [item["visual_feature"] for item in selected]
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
                        "style_source_ids": ["original"]
                        + [item["source_id"] for item in selected],
                        "style_logits": [item.logits.tolist() for item in evidence],
                        "style_sequence_nll": [
                            None
                            if item.sequence_nll is None
                            else item.sequence_nll.tolist()
                            for item in evidence
                        ],
                        "style_language_features": encode_array(
                            np.stack([item.features for item in evidence])
                        ),
                        "style_visual_features": encode_array(visual_matrix),
                        "alignment_candidates": [
                            candidate_metadata(item, id(item) in selected_ids)
                            for item in candidates
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
