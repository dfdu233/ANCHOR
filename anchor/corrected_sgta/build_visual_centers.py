"""Build model-specific visual-encoder centers for a source bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from corrected_sgta.models_alignment import load_alignment_adapter
from corrected_sgta.source_bank import (
    load_descriptor_image,
    load_index,
    load_manifest,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("hulu", "llava"))
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-per-source", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ordered_descriptors(values: list[dict], seed: int) -> list[dict]:
    return sorted(
        values,
        key=lambda item: hashlib.sha256(
            f"{seed}:{json.dumps(item, sort_keys=True)}".encode()
        ).hexdigest(),
    )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12, None)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.source_bank)
    adapter = load_alignment_adapter(args.model)
    arrays = {}
    entries = []
    try:
        for entry in manifest.get("entries", []):
            if not entry.get("formal") or not entry.get("image_index"):
                continue
            descriptors = ordered_descriptors(
                load_index(Path(entry["image_index"])), args.seed
            )[: args.max_per_source]
            vectors = []
            for start in tqdm(
                range(0, len(descriptors), args.batch_size),
                desc=f"{args.model}:{entry['source_id']}",
            ):
                batch_descriptors = descriptors[start : start + args.batch_size]
                images = [load_descriptor_image(item) for item in batch_descriptors]
                vectors.append(adapter.visual_features(images))
            matrix = normalize_rows(np.concatenate(vectors, axis=0))
            center = normalize_rows(matrix.mean(axis=0, keepdims=True))[0]
            key = f"center_{len(entries)}"
            arrays[key] = center.astype(np.float32)
            entries.append(
                {
                    "source_id": entry["source_id"],
                    "modality": entry["modality"],
                    "array_key": key,
                    "n_images": int(matrix.shape[0]),
                    "dimension": int(matrix.shape[1]),
                    "image_index_sha256": entry["image_index_sha256"],
                }
            )
    finally:
        adapter.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "version": "sgta-visual-source-centers-v1",
        "model": args.model,
        "source_bank": str(args.source_bank.resolve()),
        "source_bank_sha256": sha256_file(args.source_bank),
        "max_per_source": args.max_per_source,
        "seed": args.seed,
        "pooling": "mean projected visual tokens; per-image L2 normalization; normalized mean",
        "entries": entries,
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
