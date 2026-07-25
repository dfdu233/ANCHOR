"""Fixed-beta source-mean activation shift for LLaVA-Med CE probes."""

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

from corrected_sgta.cache import (
    encode_array,
    load_successful_qids,
    repair_truncated_jsonl_tail,
)
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
from corrected_sgta.source_bank_v2 import sha256_file


ImageFile.LOAD_TRUNCATED_IMAGES = True
CACHE_VERSION = "sgta-exact-source-activation-shift-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-stats", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--decode-max-new-tokens", type=int, default=8)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def code_identity(project_root: Path) -> dict:
    names = (
        "corrected_sgta/infer_exact_activation_shift.py",
        "corrected_sgta/models_alignment.py",
        "corrected_sgta/models_surface.py",
        "corrected_sgta/infer_ce.py",
        "corrected_sgta/cache.py",
        "corrected_sgta/protocol_v2.py",
        "corrected_sgta/provenance_v3.py",
    )
    return {name: sha256_file(project_root / name) for name in names}


@contextmanager
def source_mean_shift(
    adapter: LlavaMedAlignmentAdapter,
    source_mean: np.ndarray,
    beta: float,
):
    original_encode = adapter.model.encode_images
    source = np.asarray(source_mean, dtype=np.float32)

    def shifted_encode(*args, **kwargs):
        features = original_encode(*args, **kwargs)
        target = torch.as_tensor(
            source, device=features.device, dtype=features.dtype
        )
        if features.ndim == 3:
            current = features.mean(dim=1, keepdim=True)
            target = target.view(1, 1, -1)
        elif features.ndim == 2:
            current = features.mean(dim=0, keepdim=True)
            target = target.view(1, -1)
        else:
            raise RuntimeError(
                f"unexpected projected feature shape: {tuple(features.shape)}"
            )
        return features + float(beta) * (target - current)

    adapter.model.encode_images = shifted_encode
    try:
        yield
    finally:
        adapter.model.encode_images = original_encode


def main() -> None:
    args = parse_args()
    if args.beta != 1.0:
        raise RuntimeError("the falsification pilot freezes beta exactly at 1.0")
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    stats_meta_path = args.activation_stats.with_suffix(
        args.activation_stats.suffix + ".meta.json"
    )
    stats_meta = json.loads(stats_meta_path.read_text())
    if stats_meta["output_sha256"] != sha256_file(args.activation_stats):
        raise RuntimeError("activation-stat artifact hash mismatch")
    if stats_meta["model_identity"] != model_identity("llava"):
        raise RuntimeError("activation-stat/current model identity mismatch")
    with np.load(args.activation_stats) as stats:
        source_mean = np.asarray(stats["projected_mean"], dtype=np.float32)
    if source_mean.shape != (4096,) or not np.isfinite(source_mean).all():
        raise RuntimeError(f"invalid source mean: {source_mean.shape}")

    project_root = Path(__file__).resolve().parents[1]
    config = {
        "cache_version": CACHE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": "llava",
        "model_identity": model_identity("llava"),
        "code_identity": code_identity(project_root),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "max_samples": args.max_samples,
        "seed": args.seed,
        "subset_order": "sha256(seed:qid)",
        "max_image_side": args.max_image_side,
        "beta": args.beta,
        "operator": (
            "uniform raw source-mean shift of projected visual tokens; "
            "all within-image pairwise token residuals invariant"
        ),
        "activation_stats": str(args.activation_stats.resolve()),
        "activation_stats_sha256": sha256_file(args.activation_stats),
        "activation_stats_meta_sha256": sha256_file(stats_meta_path),
        "decode_max_new_tokens": args.decode_max_new_tokens,
        "probe_scope": (
            "single preregistered FOA-style beta=1 falsification pilot; no tuning"
        ),
        "model_source_claim_scope": (
            "exact released LLaVA-Med alignment-stage source membership; "
            "caption-filtered CXR candidates"
        ),
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "dataset_validation": validation,
        "activation_stats_metadata": stats_meta,
    }
    if meta_path.exists():
        if json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError(f"metadata mismatch; choose a new output: {args.output}")
    else:
        atomic_json(meta_path, metadata)

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
    target_rows.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row['qid']}".encode()
        ).hexdigest()
    )
    if args.max_samples:
        target_rows = target_rows[: args.max_samples]
    eligible = [row for row in target_rows if str(row["qid"]) not in saved]
    print(
        f"shift={CACHE_VERSION} fingerprint={fingerprint[:12]} "
        f"eligible={len(eligible)}",
        flush=True,
    )
    if not eligible:
        return

    adapter = LlavaMedAlignmentAdapter()
    errors = 0
    try:
        with args.output.open("a", encoding="utf-8") as output:
            for sample in tqdm(eligible, desc="exact activation shift llava"):
                try:
                    image_path = resolve_image(sample.get("img_name", ""))
                    assert image_path is not None
                    with Image.open(image_path) as source:
                        image = resize_image(source, args.max_image_side)
                    labels = labels_for_sample(sample)
                    prompt = build_prompt(sample)
                    original = adapter.forward_ce([image], prompt, labels)[0]
                    original_text = adapter.decode_ce(
                        [image],
                        prompt,
                        max_new_tokens=args.decode_max_new_tokens,
                    )[0]
                    with source_mean_shift(adapter, source_mean, args.beta):
                        shifted = adapter.forward_ce([image], prompt, labels)[0]
                        shifted_text = adapter.decode_ce(
                            [image],
                            prompt,
                            max_new_tokens=args.decode_max_new_tokens,
                        )[0]
                    original_visual = adapter.visual_features([image])[0]
                    shifted_visual = (
                        original_visual
                        + args.beta * (source_mean - original_visual)
                    )
                    before = float(np.linalg.norm(original_visual - source_mean))
                    after = float(np.linalg.norm(shifted_visual - source_mean))
                    candidate = {
                        "source_id": "llava_alignment_cxr_exact",
                        "modality": "xray",
                        "shift_beta": args.beta,
                        "structure": {
                            "pixel_identity": True,
                            "token_residual_identity": True,
                            "scope": (
                                "input pixels are identical; projected token "
                                "pairwise residuals are invariant by construction"
                            ),
                        },
                        "safe": True,
                        "selected": True,
                        "visual_euclidean_before": before,
                        "visual_euclidean_after": after,
                        "relative_closure": (
                            (before - after) / max(before, 1e-12)
                        ),
                    }
                    evidence = [original, shifted]
                    decoded = [original_text, shifted_text]
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "alignment_cache_version": CACHE_VERSION,
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "qid": sample["qid"],
                        "img_name": sample.get("img_name", ""),
                        "modality": "xray",
                        "question_type": task_kind(sample),
                        "labels": list(labels),
                        "gt_index": ground_truth_index(sample),
                        "style_names": ["original", "exact_source_activation_b1"],
                        "style_roles": ["original", "matched"],
                        "style_target_source_ids": [
                            "original",
                            "llava_alignment_cxr_exact",
                        ],
                        "style_logits": [item.logits.tolist() for item in evidence],
                        "style_sequence_nll": [
                            item.sequence_nll.tolist() for item in evidence
                        ],
                        "style_language_features": encode_array(
                            np.stack([item.features for item in evidence])
                        ),
                        "style_visual_features": encode_array(
                            np.stack([original_visual, shifted_visual])
                        ),
                        "alignment_candidates": [candidate],
                        "fallback_to_original": False,
                        "style_decoded_text": decoded,
                        "style_decoded_prediction": [
                            decoded_label_index(text, labels, sample)
                            for text in decoded
                        ],
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
