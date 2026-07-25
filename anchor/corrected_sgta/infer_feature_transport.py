"""Label-free single-step visual-token transport toward a source-domain center."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import encode_array, load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.infer_ce import decoded_label_index, resize_image
from corrected_sgta.models_transport import load_transport_adapter, transported_mean
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
from corrected_sgta.provenance_release2 import center_code_identity
from corrected_sgta.source_bank_v2 import (
    cosine_distance,
    entries_for_modality,
    load_feature_centers,
    load_manifest,
    sha256_file,
)
from corrected_sgta.source_bank_v3 import verify_source_artifacts
from corrected_sgta.transport_provenance import model_identity, transport_code_identity


ImageFile.LOAD_TRUNCATED_IMAGES = True
TRANSPORT_CACHE_VERSION = "sgta-feature-transport-v1"


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
    parser.add_argument("--beta-grid", type=float, nargs="+", default=(0.1, 0.2, 0.3, 0.4))
    parser.add_argument("--target-relative-closure", type=float, default=0.20)
    parser.add_argument("--decode-max-new-tokens", type=int, default=8)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    manifest = load_manifest(args.source_bank)
    source_bank_sha256 = sha256_file(args.source_bank)
    verified = verify_source_artifacts(manifest)
    center_meta, centers = load_feature_centers(
        args.visual_centers,
        expected_model=args.model,
        expected_source_bank_sha256=source_bank_sha256,
    )
    project_root = Path(__file__).resolve().parents[1]
    if center_meta.get("model_identity") != model_identity(args.model):
        raise RuntimeError("visual center/current model identity mismatch")
    if center_meta.get("code_identity") != center_code_identity(project_root):
        raise RuntimeError("visual center/current encoder code identity mismatch")
    config = {
        "transport_cache_version": TRANSPORT_CACHE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": args.model,
        "model_identity": model_identity(args.model),
        "code_identity": transport_code_identity(project_root),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "source_bank": str(args.source_bank.resolve()),
        "source_bank_sha256": source_bank_sha256,
        "verified_source_artifacts": verified,
        "visual_centers": str(args.visual_centers.resolve()),
        "visual_centers_sha256": sha256_file(args.visual_centers),
        "visual_centers_meta_sha256": sha256_file(
            args.visual_centers.with_suffix(args.visual_centers.suffix + ".meta.json")
        ),
        "max_samples": args.max_samples,
        "max_image_side": args.max_image_side,
        "seed": args.seed,
        "subset_order": "sha256(seed:qid)",
        "beta_grid": list(args.beta_grid),
        "target_relative_closure": args.target_relative_closure,
        "source_selection": "nearest same-modality visual source center",
        "beta_selection": "smallest beta reaching target closure; no task labels",
        "wrong_control": "other same-modality source center at identical beta",
        "decode_max_new_tokens": args.decode_max_new_tokens,
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "transport_cache_version": TRANSPORT_CACHE_VERSION,
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
    repair_truncated_jsonl_tail(args.output)
    saved = load_successful_qids(args.output, fingerprint)
    target_rows = []
    for sample in rows:
        try:
            if task_kind(sample) == "open":
                continue
            labels_for_sample(sample)
            ground_truth_index(sample)
            if resolve_image(sample.get("img_name", "")) is not None:
                target_rows.append(sample)
        except ProtocolError:
            continue
    target_rows.sort(key=lambda row: hashlib.sha256(f"{args.seed}:{row['qid']}".encode()).hexdigest())
    if args.max_samples:
        target_rows = target_rows[: args.max_samples]
    eligible = [row for row in target_rows if str(row["qid"]) not in saved]
    print(f"transport={TRANSPORT_CACHE_VERSION} fingerprint={fingerprint[:12]} eligible={len(eligible)}", flush=True)
    if not eligible:
        return

    entries = [
        entry
        for entry in entries_for_modality(manifest, "xray", formal_only=True)
        if entry["source_id"] in centers
    ]
    if len(entries) < 2:
        raise RuntimeError("feature transport requires at least two formal X-ray sources")
    adapter = load_transport_adapter(args.model)
    errors = 0
    try:
        with args.output.open("a") as output:
            for sample in tqdm(eligible, desc=f"feature transport {args.model}"):
                try:
                    with Image.open(resolve_image(sample.get("img_name", ""))) as source:
                        image = resize_image(source, args.max_image_side)
                    original_visual = adapter.visual_features([image])[0]
                    target_entry = min(
                        entries,
                        key=lambda entry: cosine_distance(original_visual, centers[entry["source_id"]]),
                    )
                    wrong_entry = min(
                        [entry for entry in entries if entry["source_id"] != target_entry["source_id"]],
                        key=lambda entry: entry["source_id"],
                    )
                    target_center = centers[target_entry["source_id"]]
                    wrong_center = centers[wrong_entry["source_id"]]
                    distance_before = cosine_distance(original_visual, target_center)
                    chosen = None
                    for beta in sorted(args.beta_grid):
                        matched_visual = transported_mean(original_visual, target_center, beta)
                        after = cosine_distance(matched_visual, target_center)
                        relative = (distance_before - after) / max(distance_before, 1e-12)
                        if relative >= args.target_relative_closure:
                            chosen = (float(beta), matched_visual, after, relative)
                            break
                    fallback = chosen is None
                    if fallback:
                        style_names = ["original"]
                        style_roles = ["original"]
                        visuals = [original_visual]
                        labels = labels_for_sample(sample)
                        prompt = build_prompt(sample)
                        evidence = adapter.forward_ce([image], prompt, labels)
                        decoded_text = adapter.decode_ce([image], prompt, max_new_tokens=args.decode_max_new_tokens)
                        candidate = None
                    else:
                        beta, matched_visual, matched_after, matched_relative = chosen
                        wrong_visual = transported_mean(original_visual, wrong_center, beta)
                        wrong_after = cosine_distance(wrong_visual, target_center)
                        wrong_relative = (distance_before - wrong_after) / max(distance_before, 1e-12)
                        labels = labels_for_sample(sample)
                        prompt = build_prompt(sample)
                        original_evidence = adapter.forward_ce([image], prompt, labels)[0]
                        matched_evidence = adapter.forward_ce_transport(image, prompt, labels, target_center, beta)
                        wrong_evidence = adapter.forward_ce_transport(image, prompt, labels, wrong_center, beta)
                        evidence = [original_evidence, matched_evidence, wrong_evidence]
                        decoded_text = [
                            adapter.decode_ce([image], prompt, max_new_tokens=args.decode_max_new_tokens)[0],
                            adapter.decode_ce_transport(image, prompt, target_center, beta, args.decode_max_new_tokens),
                            adapter.decode_ce_transport(image, prompt, wrong_center, beta, args.decode_max_new_tokens),
                        ]
                        style_names = [
                            "original",
                            f"matched_feature_{target_entry['source_id']}_b{beta:g}",
                            f"wrong_feature_{wrong_entry['source_id']}_to_{target_entry['source_id']}_b{beta:g}",
                        ]
                        style_roles = ["original", "matched", "wrong_control"]
                        visuals = [original_visual, matched_visual, wrong_visual]
                        identity_structure = {
                            "psnr": 100.0,
                            "edge_correlation": 1.0,
                            "ssim": 1.0,
                            "central_local_contrast_correlation": 1.0,
                            "central_gradient_magnitude_ratio": 1.0,
                            "scope": "pixel-identical feature-space intervention",
                        }
                        candidate = {
                            "source_id": target_entry["source_id"],
                            "modality": "xray",
                            "beta": beta,
                            "structure": identity_structure,
                            "safe": True,
                            "selected": True,
                            "visual_distance_before": distance_before,
                            "visual_distance_after": matched_after,
                            "absolute_closure": distance_before - matched_after,
                            "relative_closure": matched_relative,
                            "wrong_source_id": wrong_entry["source_id"],
                            "wrong_structure": identity_structure,
                            "wrong_safe": True,
                            "wrong_distance_after": wrong_after,
                            "wrong_relative_closure": wrong_relative,
                        }
                    decoded_prediction = [
                        decoded_label_index(text, labels, sample) for text in decoded_text
                    ]
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "alignment_cache_version": TRANSPORT_CACHE_VERSION,
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "qid": sample["qid"],
                        "img_name": sample.get("img_name", ""),
                        "modality": "xray",
                        "question_type": task_kind(sample),
                        "labels": list(labels),
                        "gt_index": ground_truth_index(sample),
                        "style_names": style_names,
                        "style_roles": style_roles,
                        "style_target_source_ids": ["original"]
                        + ([] if fallback else [target_entry["source_id"], target_entry["source_id"]]),
                        "style_amplitude_source_ids": ["original"]
                        + ([] if fallback else [target_entry["source_id"], wrong_entry["source_id"]]),
                        "style_logits": [item.logits.tolist() for item in evidence],
                        "style_sequence_nll": [item.sequence_nll.tolist() for item in evidence],
                        "style_language_features": encode_array(np.stack([item.features for item in evidence])),
                        "style_visual_features": encode_array(np.stack(visuals)),
                        "alignment_candidates": [] if candidate is None else [candidate],
                        "fallback_to_original": fallback,
                        "style_decoded_text": decoded_text,
                        "style_decoded_prediction": decoded_prediction,
                    }
                except Exception as exc:
                    errors += 1
                    traceback.print_exc()
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "alignment_cache_version": TRANSPORT_CACHE_VERSION,
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
