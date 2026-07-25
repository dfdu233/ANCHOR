"""Deterministic structural/provenance audit for an SGTA source bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from corrected_sgta.source_bank import load_index, load_manifest, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.log1p(np.asarray(left, dtype=np.float64)).reshape(-1)
    b = np.log1p(np.asarray(right, dtype=np.float64)).reshape(-1)
    return float(1.0 - (a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.source_bank)
    formal = [entry for entry in manifest.get("entries", []) if entry.get("formal")]
    entries = []
    arrays = {}
    checks = []
    for entry in formal:
        amplitude_path = Path(entry["amplitude_file"])
        index_path = Path(entry["image_index"])
        amplitude = np.load(amplitude_path)
        index = load_index(index_path)
        record = {
            "source_id": entry["source_id"],
            "modality": entry["modality"],
            "n_manifest": entry["n_used"],
            "n_index": len(index),
            "shape": list(amplitude.shape),
            "dtype": str(amplitude.dtype),
            "finite": bool(np.isfinite(amplitude).all()),
            "nonnegative": bool((amplitude >= 0).all()),
            "amplitude_sha256_matches": sha256_file(amplitude_path)
            == entry["amplitude_sha256"],
            "index_sha256_matches": sha256_file(index_path)
            == entry["image_index_sha256"],
            "unique_index_paths": len({item.get("path") for item in index}),
        }
        record["pass"] = bool(
            record["n_manifest"] == record["n_index"]
            and record["n_index"] == record["unique_index_paths"]
            and record["shape"] == [manifest["target_size"], manifest["target_size"]]
            and record["finite"]
            and record["nonnegative"]
            and record["amplitude_sha256_matches"]
            and record["index_sha256_matches"]
        )
        entries.append(record)
        checks.append(record["pass"])
        arrays[entry["source_id"]] = amplitude

    pairwise = []
    source_ids = sorted(arrays)
    for left_index, left in enumerate(source_ids):
        for right in source_ids[left_index + 1 :]:
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "log_amplitude_cosine_distance": cosine_distance(
                        arrays[left], arrays[right]
                    ),
                }
            )
    report = {
        "version": "sgta-source-bank-audit-v1",
        "source_bank": str(args.source_bank),
        "source_bank_sha256": sha256_file(args.source_bank),
        "entries": entries,
        "pairwise": pairwise,
        "pass": bool(formal) and all(checks),
        "formal_source_count": len(formal),
        "legacy_diagnostic_count": len(manifest.get("entries", [])) - len(formal),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
