#!/usr/bin/env python3
"""Build a source-only local Fourier index for Huatuo CXR projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corrected_sgta.local_source_projection import load_archived_source_image
from corrected_sgta.mosec import stable_sha256
from corrected_sgta.qls_tr import fit_pca_index, spectral_descriptor


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-source-images", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.source_index.read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("split") == "train"]
    rows.sort(
        key=lambda row: stable_sha256(f"{args.seed}:{row['id']}")
    )
    rows = rows[: args.max_source_images]
    retained = []
    features = []
    rgb_means = []
    rgb_stds = []
    errors = []
    for index, row in enumerate(rows):
        try:
            image = load_archived_source_image(row)
            features.append(spectral_descriptor(image))
            pixels = np.asarray(image, dtype=np.float32).reshape(-1, 3)
            rgb_means.append(pixels.mean(axis=0))
            rgb_stds.append(pixels.std(axis=0))
            retained.append(row)
        except Exception as exc:
            errors.append({"id": row.get("id"), "error": repr(exc)})
        if (index + 1) % args.progress_every == 0 or index + 1 == len(rows):
            print(
                json.dumps(
                    {
                        "progress": f"{index + 1}/{len(rows)}",
                        "retained": len(retained),
                        "errors": len(errors),
                    }
                ),
                flush=True,
            )
    if len(features) < 16:
        raise SystemExit("not enough readable source images")

    geometry = fit_pca_index(np.stack(features))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        mean=geometry.mean,
        components=geometry.components,
        coordinates=geometry.coordinates,
        rgb_mean=np.mean(rgb_means, axis=0).astype(np.float32),
        rgb_std=np.mean(rgb_stds, axis=0).astype(np.float32),
    )
    records_path = args.output.with_suffix(args.output.suffix + ".records.jsonl")
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained)
    )
    metadata = {
        "version": "huatuo-local-source-index-v1",
        "source_index": str(args.source_index.resolve()),
        "source_index_sha256": file_sha256(args.source_index),
        "records": str(records_path.resolve()),
        "records_sha256": file_sha256(records_path),
        "artifact_sha256": file_sha256(args.output),
        "target_data_accessed": False,
        "selection": {
            "split": "train",
            "order": "sha256(seed:id)",
            "seed": args.seed,
            "requested": args.max_source_images,
            "retained": len(retained),
            "errors": errors,
        },
        "descriptor": {
            "implementation": "corrected_sgta.qls_tr.spectral_descriptor",
            "image_size": 128,
            "grid_size": 16,
            "dc_removed": True,
            "unit_l2": True,
        },
        "pca": {
            "variance": 0.90,
            "rank_cap": 16,
            "rank": int(geometry.components.shape[0]),
        },
        "bandwidth": geometry.bandwidth,
        "median_nn_distance": geometry.median_nn_distance,
        "projection_defaults": {
            "neighbors": 8,
            "radius_fraction": 0.25,
        },
        "rgb_statistics": {
            "aggregation": "mean of per-image RGB mean/std after model-visible preprocessing",
            "mean": np.mean(rgb_means, axis=0).tolist(),
            "std": np.mean(rgb_stds, axis=0).tolist(),
        },
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
