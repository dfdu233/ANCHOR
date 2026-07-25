"""Build model/processor-specific visual centers with strict provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models_alignment import load_alignment_adapter
from corrected_sgta.provenance_v2 import code_identity, model_identity
from corrected_sgta.source_bank_v2 import (
    load_descriptor_image,
    load_index,
    load_manifest,
    sha256_file,
    verify_source_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("hulu", "llava"))
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-per-source", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-image-side", type=int, default=384)
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


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source_bank_sha256 = sha256_file(args.source_bank)
    manifest = load_manifest(args.source_bank)
    verified = verify_source_artifacts(manifest)
    adapter = load_alignment_adapter(args.model)
    arrays: dict[str, np.ndarray] = {}
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
                batch = descriptors[start : start + args.batch_size]
                images = [
                    resize_image(load_descriptor_image(item), args.max_image_side)
                    for item in batch
                ]
                vectors.append(adapter.visual_features(images))
            if not vectors:
                raise RuntimeError(f"no source images for {entry['source_id']}")
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
    temporary = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(args.output)
    project_root = Path(__file__).resolve().parents[1]
    metadata = {
        "version": "sgta-visual-source-centers-v2",
        "model": args.model,
        "model_identity": model_identity(args.model),
        "source_bank": str(args.source_bank.resolve()),
        "source_bank_sha256": source_bank_sha256,
        "verified_source_artifacts": verified,
        "max_per_source": args.max_per_source,
        "max_image_side": args.max_image_side,
        "seed": args.seed,
        "pooling": "processor-resized images; mean projected visual tokens; normalized mean",
        "code_identity": code_identity(project_root),
        "entries": entries,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
