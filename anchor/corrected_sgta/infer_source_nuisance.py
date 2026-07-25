"""Evaluate one-view source nuisance projection on LLaVA-Med CE."""

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
from PIL import Image
from tqdm import tqdm

from corrected_sgta.cache import load_successful_qids, repair_truncated_jsonl_tail
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
from corrected_sgta.source_nuisance import remove_nuisance


CACHE_VERSION = "source-nuisance-projection-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--subspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-type", choices=("binary", "multichoice", "all"), default="binary")
    parser.add_argument("--decode-max-new-tokens", type=int, default=8)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def nuisance_projection(
    adapter: LlavaMedAlignmentAdapter,
    source_mean: np.ndarray,
    basis: np.ndarray,
):
    original_encode = adapter.model.encode_images
    mean_tensor = torch.as_tensor(source_mean)
    basis_tensor = torch.as_tensor(basis)

    def hooked_encode(*args, **kwargs):
        features = original_encode(*args, **kwargs)
        return remove_nuisance(features, mean_tensor, basis_tensor)

    adapter.model.encode_images = hooked_encode
    try:
        yield
    finally:
        adapter.model.encode_images = original_encode


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    meta_path = args.subspace.with_suffix(args.subspace.suffix + ".meta.json")
    subspace_meta = json.loads(meta_path.read_text())
    if subspace_meta["output_sha256"] != sha256_file(args.subspace):
        raise RuntimeError("subspace hash mismatch")
    if subspace_meta["model_identity"] != model_identity("llava"):
        raise RuntimeError("subspace/model identity mismatch")
    with np.load(args.subspace) as data:
        source_mean = np.asarray(data["source_mean"], dtype=np.float32)
        basis = np.asarray(data["nuisance_basis"], dtype=np.float32)
    if source_mean.shape != (4096,) or basis.ndim != 2 or basis.shape[0] != 4096:
        raise RuntimeError("invalid nuisance artifact shapes")

    config = {
        "cache_version": CACHE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": "llava",
        "model_identity": model_identity("llava"),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "subspace": str(args.subspace.resolve()),
        "subspace_sha256": sha256_file(args.subspace),
        "subspace_meta_sha256": sha256_file(meta_path),
        "rank": int(basis.shape[1]),
        "operator": "z - U U^T(mean_tokens(z)-source_mean), broadcast to tokens",
        "max_samples": args.max_samples,
        "max_image_side": args.max_image_side,
        "seed": args.seed,
        "subset_order": "sha256(seed:qid)",
        "question_type": args.question_type,
        "decode_max_new_tokens": args.decode_max_new_tokens,
        "probe_scope": "one preregistered rank and unit-strength pilot; no tuning",
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_meta = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "dataset_validation": validation,
        "subspace_metadata": subspace_meta,
    }
    if output_meta.exists():
        if json.loads(output_meta.read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError("metadata mismatch; choose a new output")
    else:
        atomic_json(output_meta, metadata)

    repair_truncated_jsonl_tail(args.output)
    saved = load_successful_qids(args.output, fingerprint)
    eligible = []
    for row in rows:
        try:
            kind = task_kind(row)
            if kind == "open" or (args.question_type != "all" and kind != args.question_type):
                continue
            labels_for_sample(row)
            ground_truth_index(row)
            if resolve_image(row.get("img_name", "")) is not None:
                eligible.append(row)
        except ProtocolError:
            continue
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row['qid']}".encode()
        ).hexdigest()
    )
    if args.max_samples:
        eligible = eligible[: args.max_samples]
    eligible = [row for row in eligible if str(row["qid"]) not in saved]
    print(
        f"snp fingerprint={fingerprint[:12]} eligible={len(eligible)} "
        f"rank={basis.shape[1]}",
        flush=True,
    )
    if not eligible:
        return

    adapter = LlavaMedAlignmentAdapter()
    errors = 0
    try:
        with args.output.open("a") as output:
            for sample in tqdm(eligible, desc="source nuisance llava"):
                try:
                    path = resolve_image(sample.get("img_name", ""))
                    assert path is not None
                    with Image.open(path) as source:
                        image = resize_image(source, args.max_image_side)
                    labels = labels_for_sample(sample)
                    prompt = build_prompt(sample)
                    original = adapter.forward_ce([image], prompt, labels)[0]
                    original_text = adapter.decode_ce(
                        [image], prompt, args.decode_max_new_tokens
                    )[0]
                    with nuisance_projection(adapter, source_mean, basis):
                        projected = adapter.forward_ce([image], prompt, labels)[0]
                        projected_text = adapter.decode_ce(
                            [image], prompt, args.decode_max_new_tokens
                        )[0]
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "cache_version": CACHE_VERSION,
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "qid": sample["qid"],
                        "img_name": sample.get("img_name", ""),
                        "question_type": task_kind(sample),
                        "labels": list(labels),
                        "gt_index": ground_truth_index(sample),
                        "style_names": ["original", "source_nuisance_projection"],
                        "style_logits": [
                            original.logits.tolist(),
                            projected.logits.tolist(),
                        ],
                        "style_sequence_nll": [
                            original.sequence_nll.tolist(),
                            projected.sequence_nll.tolist(),
                        ],
                        "style_decoded_text": [original_text, projected_text],
                        "style_decoded_prediction": [
                            decoded_label_index(original_text, labels, sample),
                            decoded_label_index(projected_text, labels, sample),
                        ],
                    }
                    image.close()
                except Exception as exc:
                    errors += 1
                    traceback.print_exc()
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "cache_version": CACHE_VERSION,
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
    print(f"finished={len(eligible)} errors={errors}", flush=True)


if __name__ == "__main__":
    main()

