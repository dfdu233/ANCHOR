"""Single-view source-manifold alignment pilot for LLaVA-Med finite-label tasks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import encode_array, load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.infer_ce import decoded_label_index, resize_image
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
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
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import normalize_modality, sha256_file


ImageFile.LOAD_TRUNCATED_IMAGES = True
CACHE_VERSION = "sgta-exact-source-manifold-view-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifold-stats", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=("ct", "mri"))
    parser.add_argument("--max-per-question-type", type=int, default=16)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--decode-max-new-tokens", type=int, default=8)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def source_manifold_shift(
    adapter: LlavaMedAlignmentAdapter,
    source_mean: np.ndarray,
    source_basis: np.ndarray,
    beta: float,
):
    original_encode = adapter.model.encode_images

    def shifted_encode(*args, **kwargs):
        features = original_encode(*args, **kwargs)
        mean = torch.as_tensor(source_mean, device=features.device, dtype=features.dtype)
        basis = torch.as_tensor(source_basis, device=features.device, dtype=features.dtype)
        if features.ndim == 3:
            current = features.mean(dim=1, keepdim=True)
            mean = mean.view(1, 1, -1)
        elif features.ndim == 2:
            current = features.mean(dim=0, keepdim=True)
            mean = mean.view(1, -1)
        else:
            raise RuntimeError(f"unexpected projected feature shape: {tuple(features.shape)}")
        delta = current - mean
        tangent = (delta @ basis) @ basis.T
        aligned_mean = mean + tangent
        return features + float(beta) * (aligned_mean - current)

    adapter.model.encode_images = shifted_encode
    try:
        yield
    finally:
        adapter.model.encode_images = original_encode


def aligned_mean(value: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    delta = value - mean
    return mean + (delta @ basis) @ basis.T


def main() -> None:
    args = parse_args()
    if args.beta != 1.0:
        raise RuntimeError("the falsification pilot freezes beta exactly at 1.0")
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    stats_meta_path = args.manifold_stats.with_suffix(args.manifold_stats.suffix + ".meta.json")
    stats_meta = json.loads(stats_meta_path.read_text())
    if stats_meta["output_sha256"] != sha256_file(args.manifold_stats):
        raise RuntimeError("manifold-stat artifact hash mismatch")
    if stats_meta["config"]["model_identity"] != model_identity("llava"):
        raise RuntimeError("manifold-stat/current model identity mismatch")
    with np.load(args.manifold_stats) as stats:
        mean = np.asarray(stats[f"{args.modality}_mean"], dtype=np.float32)
        basis = np.asarray(stats[f"{args.modality}_basis"], dtype=np.float32)
    if mean.shape != (4096,) or basis.shape[0] != 4096:
        raise RuntimeError(f"invalid manifold shapes: {mean.shape}, {basis.shape}")

    config = {
        "cache_version": CACHE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": "llava",
        "model_identity": model_identity("llava"),
        "code_sha256": sha256_file(Path(__file__)),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "modality": args.modality,
        "max_per_question_type": args.max_per_question_type,
        "question_types": ["binary", "multichoice"],
        "seed": args.seed,
        "subset_order": "within question type, sha256(seed:qid)",
        "max_image_side": args.max_image_side,
        "beta": args.beta,
        "operator": (
            "remove only the source-affine-subspace normal component of the "
            "per-image mean projected visual token; preserve pixels, tangent "
            "coordinates, and all within-image token residuals"
        ),
        "manifold_rank": int(basis.shape[1]),
        "manifold_stats": str(args.manifold_stats.resolve()),
        "manifold_stats_sha256": sha256_file(args.manifold_stats),
        "manifold_stats_meta_sha256": sha256_file(stats_meta_path),
        "decode_max_new_tokens": args.decode_max_new_tokens,
        "probe_scope": "one beta=1 source-manifold falsification pilot; no tuning",
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "fingerprint": fingerprint,
        "config": config,
        "dataset_validation": validation,
        "manifold_stats_metadata": stats_meta,
    }
    if meta_path.exists():
        if json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError(f"metadata mismatch; choose a new output: {args.output}")
    else:
        atomic_json(meta_path, metadata)

    repair_truncated_jsonl_tail(args.output)
    saved = load_successful_qids(args.output, fingerprint)
    grouped = {"binary": [], "multichoice": []}
    for sample in rows:
        try:
            kind = task_kind(sample)
            if kind not in grouped or normalize_modality(sample.get("modality")) != args.modality:
                continue
            labels_for_sample(sample)
            ground_truth_index(sample)
            if resolve_image(sample.get("img_name", "")) is not None:
                grouped[kind].append(sample)
        except ProtocolError:
            continue
    selected = []
    for kind in ("binary", "multichoice"):
        grouped[kind].sort(
            key=lambda row: hashlib.sha256(
                (str(args.seed) + ":" + str(row.get("qid"))).encode()
            ).hexdigest()
        )
        selected.extend(grouped[kind][: args.max_per_question_type])
    eligible = [row for row in selected if str(row["qid"]) not in saved]
    print(
        f"view={CACHE_VERSION} modality={args.modality} fingerprint={fingerprint[:12]} "
        f"selected={len(selected)} eligible={len(eligible)}",
        flush=True,
    )
    if not eligible:
        return

    adapter = LlavaMedAlignmentAdapter()
    errors = 0
    try:
        with args.output.open("a", encoding="utf-8") as output:
            for sample in tqdm(eligible, desc=f"exact manifold {args.modality}"):
                try:
                    image_path = resolve_image(sample.get("img_name", ""))
                    assert image_path is not None
                    with Image.open(image_path) as source:
                        image = resize_image(source, args.max_image_side)
                    labels = labels_for_sample(sample)
                    prompt = build_prompt(sample)
                    original = adapter.forward_ce([image], prompt, labels)[0]
                    original_text = adapter.decode_ce(
                        [image], prompt, max_new_tokens=args.decode_max_new_tokens
                    )[0]
                    with source_manifold_shift(adapter, mean, basis, args.beta):
                        shifted = adapter.forward_ce([image], prompt, labels)[0]
                        shifted_text = adapter.decode_ce(
                            [image], prompt, max_new_tokens=args.decode_max_new_tokens
                        )[0]
                    original_visual = adapter.visual_features([image])[0]
                    projected = aligned_mean(original_visual, mean, basis)
                    shifted_visual = original_visual + args.beta * (projected - original_visual)
                    before = float(np.linalg.norm(original_visual - projected))
                    after_projected = aligned_mean(shifted_visual, mean, basis)
                    after = float(np.linalg.norm(shifted_visual - after_projected))
                    decoded = [original_text, shifted_text]
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "alignment_cache_version": CACHE_VERSION,
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "qid": sample["qid"],
                        "img_name": sample.get("img_name", ""),
                        "modality": args.modality,
                        "question_type": task_kind(sample),
                        "labels": list(labels),
                        "gt_index": ground_truth_index(sample),
                        "style_names": ["original", "exact_source_manifold_b1"],
                        "style_logits": [original.logits.tolist(), shifted.logits.tolist()],
                        "style_sequence_nll": [
                            original.sequence_nll.tolist(), shifted.sequence_nll.tolist()
                        ],
                        "style_visual_features": encode_array(
                            np.stack([original_visual, shifted_visual])
                        ),
                        "style_decoded_text": decoded,
                        "style_decoded_prediction": [
                            decoded_label_index(text, labels, sample) for text in decoded
                        ],
                        "alignment": {
                            "source_modality": args.modality,
                            "beta": args.beta,
                            "rank": int(basis.shape[1]),
                            "normal_residual_before": before,
                            "normal_residual_after": after,
                            "relative_normal_closure": (before - after) / max(before, 1e-12),
                            "pixel_identity": True,
                            "token_residual_identity": True,
                            "tangent_coordinate_identity": True,
                        },
                    }
                    image.close()
                except Exception as exc:
                    errors += 1
                    traceback.print_exc()
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "alignment_cache_version": CACHE_VERSION,
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
