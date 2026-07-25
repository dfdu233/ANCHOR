"""Build deterministic local Fourier-density indices from Source Bank images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from corrected_sgta.qls_tr import fit_pca_index, spectral_descriptor
from corrected_sgta.source_bank_v2 import (
    load_descriptor_image,
    load_index,
    load_manifest,
    sha256_file,
    verify_source_artifacts,
)


VERSION = "sgta-qls-source-index-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-per-source", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.source_bank)
    verified = verify_source_artifacts(manifest)
    arrays: dict[str, np.ndarray] = {}
    entries = []
    for entry in manifest["entries"]:
        if not entry.get("formal") or not entry.get("image_index"):
            continue
        descriptors = load_index(Path(entry["image_index"]))
        descriptors.sort(
            key=lambda row: hashlib.sha256(
                f"{args.seed}:{row.get('canonical_rgb_sha256', row)}".encode()
            ).hexdigest()
        )
        descriptors = descriptors[: args.max_per_source]
        features = []
        retained = []
        for descriptor in tqdm(descriptors, desc=f"QLS index {entry['source_id']}"):
            try:
                image = load_descriptor_image(descriptor)
                features.append(spectral_descriptor(image))
                retained.append(descriptor)
            except Exception:
                continue
        matrix = np.stack(features)
        index = fit_pca_index(matrix)
        prefix = entry["source_id"]
        arrays[f"{prefix}__mean"] = index.mean
        arrays[f"{prefix}__components"] = index.components
        arrays[f"{prefix}__coordinates"] = index.coordinates
        retained_path = args.output.with_name(f"{args.output.stem}.{prefix}.jsonl")
        retained_path.parent.mkdir(parents=True, exist_ok=True)
        retained_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained)
        )
        entries.append(
            {
                "source_id": prefix,
                "modality": entry["modality"],
                "n": len(retained),
                "bandwidth": index.bandwidth,
                "median_nn_distance": index.median_nn_distance,
                "retained_index": str(retained_path.resolve()),
                "retained_index_sha256": sha256_file(retained_path),
                "rank": int(index.components.shape[0]),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "version": VERSION,
        "source_bank": str(args.source_bank.resolve()),
        "source_bank_sha256": sha256_file(args.source_bank),
        "verified_source_artifacts": verified,
        "max_per_source": args.max_per_source,
        "seed": args.seed,
        "descriptor": {
            "image_size": 128,
            "grid_size": 16,
            "normalization": "remove DC then unit L2",
        },
        "pca": {"variance": 0.90, "rank_cap": 16},
        "kde": {"neighbors": 8, "bandwidth": "median leave-one-out 8-NN"},
        "entries": entries,
        "artifact_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(json.dumps({"output": str(args.output), "entries": entries}, indent=2))


if __name__ == "__main__":
    main()
