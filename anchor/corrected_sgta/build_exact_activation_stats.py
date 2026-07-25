"""Derive source-only projector activation statistics from a frozen feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from corrected_sgta.source_bank_v2 import sha256_file


VERSION = "llava-exact-source-activation-stats-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    analysis = json.loads(args.analysis.read_text())
    if sha256_file(args.features) != analysis["features_sha256"]:
        raise RuntimeError("feature cache hash does not match support analysis")
    with np.load(args.features) as cache:
        exact = np.asarray(cache["exact_projected"], dtype=np.float32)
    exact_ids = [str(item) for item in analysis["ids"]["exact"]]
    if exact.ndim != 2 or exact.shape[1] != 4096 or not np.isfinite(exact).all():
        raise RuntimeError(f"invalid exact projected features: {exact.shape}")
    if exact.shape[0] != len(exact_ids) or exact.shape[0] != analysis["n"]["exact"]:
        raise RuntimeError("exact feature/id/provenance count mismatch")
    source_mean = exact.mean(axis=0, dtype=np.float64).astype(np.float32)
    source_std = exact.std(axis=0, dtype=np.float64).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        projected_mean=source_mean,
        projected_std=source_std,
        n_source=np.asarray([exact.shape[0]], dtype=np.int64),
    )
    temporary.replace(args.output)
    ids_sha256 = hashlib.sha256(
        json.dumps(exact_ids, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = {
        "version": VERSION,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "features": str(args.features.resolve()),
        "features_sha256": analysis["features_sha256"],
        "support_analysis": str(args.analysis.resolve()),
        "support_analysis_sha256": sha256_file(args.analysis),
        "support_fingerprint": analysis["fingerprint"],
        "exact_index": analysis["config"]["exact_index"],
        "exact_index_sha256": analysis["config"]["exact_index_sha256"],
        "model_identity": analysis["config"]["model_identity"],
        "n_source": exact.shape[0],
        "dimension": exact.shape[1],
        "exact_ids_sha256": ids_sha256,
        "statistic": "mean/std of per-image mean projected visual tokens",
        "target_data_used": False,
        "derivation": (
            "Only exact_projected and exact_ids are read from the parent cache; "
            "proxy and target arrays are not used."
        ),
        "code_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(
        args.output.with_suffix(args.output.suffix + ".meta.json"), metadata
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
