"""Build source-only affine-manifold statistics from exact LLaVA features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from corrected_sgta.source_bank_v2 import sha256_file


VERSION = "llava-exact-source-manifold-stats-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--explained-variance", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.explained_variance <= 1.0:
        raise ValueError("explained-variance must lie in (0, 1]")
    analysis = json.loads(args.support_analysis.read_text())
    features_path = Path(analysis["features"])
    if analysis["features_sha256"] != sha256_file(features_path):
        raise RuntimeError("support feature hash mismatch")
    arrays = {}
    entries = {}
    with np.load(features_path) as features:
        for source in analysis["config"]["sources"]:
            modality = source["modality"]
            values = np.asarray(features[f"{modality}_source_projected"], dtype=np.float64)
            mean = values.mean(axis=0)
            centered = values - mean
            _, singular, vt = np.linalg.svd(centered, full_matrices=False)
            cumulative = np.cumsum(singular**2) / np.maximum((singular**2).sum(), 1e-12)
            rank = int(np.searchsorted(cumulative, args.explained_variance) + 1)
            arrays[f"{modality}_mean"] = mean.astype(np.float32)
            arrays[f"{modality}_basis"] = vt[:rank].T.astype(np.float32)
            entries[modality] = {
                "n_source": len(values),
                "dimension": values.shape[1],
                "rank": rank,
                "explained_variance": float(cumulative[rank - 1]),
                "source_index": source["index"],
                "source_index_sha256": source["index_sha256"],
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    config = {
        "version": VERSION,
        "support_analysis": str(args.support_analysis.resolve()),
        "support_analysis_sha256": sha256_file(args.support_analysis),
        "support_fingerprint": analysis["fingerprint"],
        "features": str(features_path.resolve()),
        "features_sha256": sha256_file(features_path),
        "explained_variance_threshold": args.explained_variance,
        "source_arrays_read": [f"{m}_source_projected" for m in entries],
        "target_arrays_read": [],
        "target_data_used": False,
        "entries": entries,
        "model_identity": analysis["config"]["model_identity"],
    }
    payload = {
        "fingerprint": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "config": config,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    meta = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
