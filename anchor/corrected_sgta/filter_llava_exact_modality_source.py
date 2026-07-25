"""Filter exact LLaVA-Med alignment images into a modality-pure source index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.build_llava_exact_source_bank import (
    benchmark_target_hashes,
    canonical_rgb_sha256,
)
from corrected_sgta.protocol_v2 import file_sha256
from corrected_sgta.source_bank_v2 import sha256_file


VERSION = "llava-exact-modality-filter-v1"
MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
LABELS = (
    "a clinical computed tomography CT image",
    "a clinical magnetic resonance imaging MRI image",
    "a chest X-ray radiograph",
    "a pathology or microscopy image",
    "a chart diagram photograph or composite figure",
)
EXPECTED_INDEX = {"ct": 0, "mri": 1}
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-index", required=True, type=Path)
    parser.add_argument("--prepared-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=tuple(EXPECTED_INDEX))
    parser.add_argument("--cxr-benchmark", required=True, type=Path)
    parser.add_argument("--mm-benchmark", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--target-hash-cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-probability", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def target_hashes_with_cache(args: argparse.Namespace) -> tuple[set[str], dict, str | None]:
    datasets = [args.cxr_benchmark, args.mm_benchmark]
    expected = [
        {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for path in datasets
    ]
    if args.target_hash_cache and args.target_hash_cache.exists():
        payload = json.loads(args.target_hash_cache.read_text())
        if payload.get("datasets") != expected:
            raise RuntimeError("target hash cache/dataset mismatch")
        return (
            set(payload["canonical_rgb_hashes"]),
            payload["audit"],
            sha256_file(args.target_hash_cache),
        )
    hashes, audit = benchmark_target_hashes(datasets)
    cache_sha256 = None
    if args.target_hash_cache:
        args.target_hash_cache.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.target_hash_cache,
            {
                "version": "medheval-target-rgb-hashes-v1",
                "datasets": expected,
                "audit": audit,
                "canonical_rgb_hashes": sorted(hashes),
            },
        )
        cache_sha256 = sha256_file(args.target_hash_cache)
    return hashes, audit, cache_sha256


def load_model(cache_dir: Path, device: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir.resolve()))
    from open_clip import create_model_from_pretrained, get_tokenizer

    name = f"hf-hub:{MODEL_ID}"
    model, preprocess = create_model_from_pretrained(name, cache_dir=str(cache_dir))
    tokenizer = get_tokenizer(name, cache_dir=str(cache_dir))
    model = model.to(device).eval()
    tokens = tokenizer(
        [f"this is {label}" for label in LABELS], context_length=256
    ).to(device)
    return model, preprocess, tokens


@torch.inference_mode()
def classify(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    model, preprocess, tokens = load_model(args.cache_dir, args.device)
    results = []
    try:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="BiomedCLIP modality"):
            batch = rows[start : start + args.batch_size]
            tensors = []
            kept = []
            for row in batch:
                try:
                    with Image.open(row["local_path"]) as source:
                        tensors.append(preprocess(source.convert("RGB")))
                    kept.append(row)
                except OSError as exc:
                    results.append(
                        {**row, "filter_status": "unreadable", "error": str(exc)[:300]}
                    )
            if not kept:
                continue
            images = torch.stack(tensors).to(args.device)
            image_features, text_features, logit_scale = model(images, tokens)
            probabilities = (
                (logit_scale * image_features @ text_features.t())
                .softmax(dim=-1)
                .float()
                .cpu()
            )
            for row, values in zip(kept, probabilities):
                predicted = int(values.argmax())
                results.append(
                    {
                        **row,
                        "filter_status": "scored",
                        "biomedclip_probabilities": values.tolist(),
                        "biomedclip_prediction": predicted,
                        "biomedclip_label": LABELS[predicted],
                    }
                )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_probability <= 1.0:
        raise ValueError("min-probability must lie in [0, 1]")
    prepared_meta = json.loads(args.prepared_metadata.read_text())
    if prepared_meta["config"].get("modality") != args.modality:
        raise RuntimeError("prepared metadata/modality mismatch")
    rows = [
        json.loads(line)
        for line in args.prepared_index.read_text().splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: hashlib.sha256(row["pair_id"].encode()).hexdigest())
    if args.max_samples:
        rows = rows[: args.max_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_hashes, target_audit, target_hash_cache_sha256 = target_hashes_with_cache(args)
    index_path = args.output_dir / f"llava_alignment_{args.modality}_exact.jsonl"
    rejected_path = args.output_dir / "rejected.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    config = {
        "version": VERSION,
        "prepared_index": str(args.prepared_index.resolve()),
        "prepared_index_sha256": file_sha256(args.prepared_index),
        "prepared_metadata": str(args.prepared_metadata.resolve()),
        "prepared_metadata_sha256": file_sha256(args.prepared_metadata),
        "prepared_fingerprint": prepared_meta["fingerprint"],
        "modality": args.modality,
        "model_id": MODEL_ID,
        "model_weights": str(
            (Path("/root/autodl-tmp/BiomedCLIP") / "open_clip_pytorch_model.bin").resolve()
        ),
        "model_weights_sha256": sha256_file(
            Path("/root/autodl-tmp/BiomedCLIP/open_clip_pytorch_model.bin")
        ),
        "labels": list(LABELS),
        "prompt_template": "this is {label}",
        "acceptance": "expected modality is top-1 and probability >= threshold",
        "min_probability": args.min_probability,
        "max_samples": args.max_samples,
        "cxr_benchmark": str(args.cxr_benchmark.resolve()),
        "cxr_benchmark_sha256": file_sha256(args.cxr_benchmark),
        "mm_benchmark": str(args.mm_benchmark.resolve()),
        "mm_benchmark_sha256": file_sha256(args.mm_benchmark),
        "target_use": "content-hash exclusion only; no labels or target features used",
        "target_hash_cache": (
            None if args.target_hash_cache is None else str(args.target_hash_cache.resolve())
        ),
        "target_hash_cache_sha256": target_hash_cache_sha256,
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise RuntimeError(f"metadata mismatch; choose a new output: {args.output_dir}")

    scored = classify(rows, args)
    expected = EXPECTED_INDEX[args.modality]
    accepted = []
    rejected = []
    seen_rgb = set()
    counts = {
        "unreadable": 0,
        "wrong_top1": 0,
        "below_probability": 0,
        "target_overlap": 0,
        "source_duplicate": 0,
    }
    for row in scored:
        if row["filter_status"] != "scored":
            counts["unreadable"] += 1
            rejected.append({**row, "rejection_reason": "unreadable"})
            continue
        probability = float(row["biomedclip_probabilities"][expected])
        if row["biomedclip_prediction"] != expected:
            counts["wrong_top1"] += 1
            rejected.append({**row, "rejection_reason": "wrong_top1"})
            continue
        if probability < args.min_probability:
            counts["below_probability"] += 1
            rejected.append({**row, "rejection_reason": "below_probability"})
            continue
        path = Path(row["local_path"])
        rgb_hash = canonical_rgb_sha256(path)
        if rgb_hash in target_hashes:
            counts["target_overlap"] += 1
            rejected.append({**row, "rejection_reason": "target_overlap"})
            continue
        if rgb_hash in seen_rgb:
            counts["source_duplicate"] += 1
            rejected.append({**row, "rejection_reason": "source_duplicate"})
            continue
        seen_rgb.add(rgb_hash)
        accepted.append(
            {
                "kind": "path",
                "path": str(path.resolve()),
                "file_sha256": sha256_file(path),
                "canonical_rgb_sha256": rgb_hash,
                "pair_id": row["pair_id"],
                "source_url": row["source_url"],
                "caption": row["caption"],
                "modality": args.modality,
                "biomedclip_expected_probability": probability,
                "filter_fingerprint": fingerprint,
            }
        )
    index_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in accepted)
    )
    rejected_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rejected)
    )
    payload = {
        "fingerprint": fingerprint,
        "config": config,
        "target_exclusion": target_audit,
        "n_scored": len(scored),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "rejection_counts": counts,
        "accepted_index": str(index_path.resolve()),
        "accepted_index_sha256": sha256_file(index_path),
        "rejected_index": str(rejected_path.resolve()),
        "rejected_index_sha256": sha256_file(rejected_path),
    }
    atomic_json(metadata_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
