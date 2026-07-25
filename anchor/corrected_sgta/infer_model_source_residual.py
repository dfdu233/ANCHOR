"""Fixed-beta projected-visual-token residual toward the PubMed source center."""

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
from corrected_sgta.model_source_residual_provenance import model_source_residual_identity
from corrected_sgta.models_transport import load_transport_adapter, transported_mean
from corrected_sgta.protocol_v2 import (
    CACHE_SCHEMA_VERSION, PROTOCOL_VERSION, ProtocolError, build_prompt,
    file_sha256, ground_truth_index, labels_for_sample, protocol_fingerprint,
    resolve_image, task_kind, validate_dataset,
)
from corrected_sgta.provenance_release2 import center_code_identity
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import cosine_distance, load_feature_centers, load_manifest, sha256_file
from corrected_sgta.source_bank_v3 import verify_source_artifacts


ImageFile.LOAD_TRUNCATED_IMAGES = True
CACHE_VERSION = "sgta-model-source-visual-residual-v1"
SOURCE_ID = "pubmedvision_xray_formal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("hulu", "llava"))
    parser.add_argument("--domain", required=True, choices=("iu", "mimic"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--visual-centers", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--decode-max-new-tokens", type=int, default=8)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2)); temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.beta != 0.5:
        raise RuntimeError("V5 freezes beta exactly at 0.5")
    rows = json.loads(args.dataset.read_text()); validation = validate_dataset(rows)
    manifest = load_manifest(args.source_bank); source_bank_sha256 = sha256_file(args.source_bank)
    verified = verify_source_artifacts(manifest)
    center_meta, centers = load_feature_centers(
        args.visual_centers, expected_model=args.model,
        expected_source_bank_sha256=source_bank_sha256,
    )
    project_root = Path(__file__).resolve().parents[1]
    if center_meta.get("model_identity") != model_identity(args.model):
        raise RuntimeError("visual center/current model identity mismatch")
    if center_meta.get("code_identity") != center_code_identity(project_root):
        raise RuntimeError("visual center/current encoder identity mismatch")
    control_id = "iuxray_xray_leaksafe" if args.domain == "iu" else "mimic_cxr_leaksafe"
    if SOURCE_ID not in centers or control_id not in centers:
        raise RuntimeError("required model-source/control visual center missing")
    config = {
        "transport_cache_version": CACHE_VERSION,
        "protocol_version": PROTOCOL_VERSION, "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": args.model, "model_identity": model_identity(args.model),
        "code_identity": model_source_residual_identity(project_root),
        "domain": args.domain, "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset), "max_samples": args.max_samples,
        "seed": args.seed, "subset_order": "sha256(seed:qid)",
        "max_image_side": args.max_image_side, "beta": args.beta,
        "operator": "uniform projected-token mean residual; within-image token residuals invariant",
        "matched_source_id": SOURCE_ID, "control_source_id": control_id,
        "source_bank": str(args.source_bank.resolve()), "source_bank_sha256": source_bank_sha256,
        "verified_source_artifacts": verified,
        "visual_centers": str(args.visual_centers.resolve()),
        "visual_centers_sha256": sha256_file(args.visual_centers),
        "visual_centers_meta_sha256": sha256_file(args.visual_centers.with_suffix(args.visual_centers.suffix + ".meta.json")),
        "decode_max_new_tokens": args.decode_max_new_tokens,
        "model_source_claim_scope": {
            "llava": "training-adjacent PMC proxy; exact PMC-15M membership not claimed",
            "hulu": "public medical multimodal proxy; training sample manifest unavailable",
        },
    }
    fingerprint = protocol_fingerprint(config); args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "protocol_version": PROTOCOL_VERSION, "cache_schema_version": CACHE_SCHEMA_VERSION,
        "transport_cache_version": CACHE_VERSION, "fingerprint": fingerprint,
        "config": config, "dataset_validation": validation, "source_bank": manifest,
        "visual_center_metadata": center_meta,
    }
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError(f"metadata mismatch; choose a new output: {args.output}")
    else:
        atomic_json(metadata_path, metadata)
    repair_truncated_jsonl_tail(args.output); saved = load_successful_qids(args.output, fingerprint)
    target_rows = []
    for sample in rows:
        try:
            if task_kind(sample) == "open": continue
            labels_for_sample(sample); ground_truth_index(sample)
            if resolve_image(sample.get("img_name", "")) is not None: target_rows.append(sample)
        except ProtocolError:
            continue
    target_rows.sort(key=lambda row: hashlib.sha256(f"{args.seed}:{row['qid']}".encode()).hexdigest())
    if args.max_samples: target_rows = target_rows[:args.max_samples]
    eligible = [row for row in target_rows if str(row["qid"]) not in saved]
    print(f"transport={CACHE_VERSION} fingerprint={fingerprint[:12]} eligible={len(eligible)}", flush=True)
    if not eligible: return

    source_center = centers[SOURCE_ID]; control_center = centers[control_id]
    adapter = load_transport_adapter(args.model); errors = 0
    identity_structure = {
        "psnr": None, "edge_correlation": 1.0, "ssim": 1.0,
        "central_local_contrast_correlation": 1.0,
        "central_gradient_magnitude_ratio": 1.0,
        "scope": "processor input pixels are identical across roles",
    }
    try:
        with args.output.open("a") as output:
            for sample in tqdm(eligible, desc=f"model-source residual {args.model}:{args.domain}"):
                try:
                    image_path = resolve_image(sample.get("img_name", "")); assert image_path is not None
                    with Image.open(image_path) as source: image = resize_image(source, args.max_image_side)
                    original_visual = adapter.visual_features([image])[0]
                    matched_visual = transported_mean(original_visual, source_center, args.beta)
                    control_visual = transported_mean(original_visual, control_center, args.beta)
                    before = cosine_distance(original_visual, source_center)
                    matched_after = cosine_distance(matched_visual, source_center)
                    control_after = cosine_distance(control_visual, source_center)
                    matched_relative = (before - matched_after) / max(before, 1e-12)
                    control_relative = (before - control_after) / max(before, 1e-12)
                    labels = labels_for_sample(sample); prompt = build_prompt(sample)
                    original_evidence = adapter.forward_ce([image], prompt, labels)[0]
                    matched_evidence = adapter.forward_ce_transport(image, prompt, labels, source_center, args.beta)
                    control_evidence = adapter.forward_ce_transport(image, prompt, labels, control_center, args.beta)
                    evidence = [original_evidence, matched_evidence, control_evidence]
                    decoded_text = [
                        adapter.decode_ce([image], prompt, max_new_tokens=args.decode_max_new_tokens)[0],
                        adapter.decode_ce_transport(image, prompt, source_center, args.beta, args.decode_max_new_tokens),
                        adapter.decode_ce_transport(image, prompt, control_center, args.beta, args.decode_max_new_tokens),
                    ]
                    candidate = {
                        "source_id": SOURCE_ID, "modality": "xray", "beta": args.beta,
                        "structure": identity_structure, "safe": True, "selected": True,
                        "visual_distance_before": before, "visual_distance_after": matched_after,
                        "absolute_closure": before - matched_after, "relative_closure": matched_relative,
                        "wrong_source_id": control_id, "wrong_structure": identity_structure,
                        "wrong_safe": True, "wrong_distance_after": control_after,
                        "wrong_relative_closure": control_relative,
                    }
                    row = {
                        "protocol_version": PROTOCOL_VERSION, "alignment_cache_version": CACHE_VERSION,
                        "fingerprint": fingerprint, "status": "ok", "qid": sample["qid"],
                        "img_name": sample.get("img_name", ""), "modality": "xray",
                        "question_type": task_kind(sample), "labels": list(labels),
                        "gt_index": ground_truth_index(sample),
                        "style_names": ["original", f"matched_{SOURCE_ID}_b0.5", f"control_{control_id}_b0.5"],
                        "style_roles": ["original", "matched", "wrong_control"],
                        "style_target_source_ids": ["original", SOURCE_ID, SOURCE_ID],
                        "style_amplitude_source_ids": ["original", SOURCE_ID, control_id],
                        "style_logits": [item.logits.tolist() for item in evidence],
                        "style_sequence_nll": [None if item.sequence_nll is None else item.sequence_nll.tolist() for item in evidence],
                        "style_language_features": encode_array(np.stack([item.features for item in evidence])),
                        "style_visual_features": encode_array(np.stack([original_visual, matched_visual, control_visual])),
                        "alignment_candidates": [candidate], "fallback_to_original": False,
                        "style_decoded_text": decoded_text,
                        "style_decoded_prediction": [decoded_label_index(text, labels, sample) for text in decoded_text],
                    }
                except Exception as exc:
                    errors += 1; traceback.print_exc()
                    row = {"protocol_version": PROTOCOL_VERSION, "alignment_cache_version": CACHE_VERSION,
                           "fingerprint": fingerprint, "status": "error", "qid": sample.get("qid"),
                           "error": f"{type(exc).__name__}: {exc}"[:500]}
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        gc.collect(); torch.cuda.empty_cache()
                output.write(json.dumps(row, separators=(",", ":")) + "\n"); output.flush()
    finally:
        adapter.close()
    print(f"finished rows={len(eligible)} errors={errors}", flush=True)


if __name__ == "__main__":
    main()

