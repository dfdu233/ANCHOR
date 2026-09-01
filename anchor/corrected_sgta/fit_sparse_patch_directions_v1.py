#!/usr/bin/env python3
"""Fit fixed diagonal-LDA finding directions from development global means."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


VERSION = "sparse-patch-direction-fit-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu"), required=True)
    parser.add_argument("--raw-visual", type=Path, required=True)
    parser.add_argument("--hidden-development", type=Path, required=True)
    parser.add_argument("--findings", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    raw_rows = [json.loads(line) for line in (args.raw_visual / "metadata.jsonl").read_text().splitlines()]
    raw = np.load(args.raw_visual / "features.npz")
    means = np.asarray(raw["pre_mean"], dtype=np.float64)
    index = {str(row["image_id"]): int(row["ordered_index"]) for row in raw_rows}
    if len(index) != len(raw_rows) or means.shape[0] != len(raw_rows):
        raise ValueError("raw visual image identity mismatch")
    hidden_rows = [json.loads(line) for line in (args.hidden_development / "metadata.jsonl").read_text().splitlines()]
    findings = tuple(args.findings)
    directions, diagnostics = [], {}
    for finding in findings:
        rows = [row for row in hidden_rows if row["finding"] == finding and int(row["positive_votes"]) in (0, 3)]
        labels = np.asarray([int(row["positive_votes"] == 3) for row in rows])
        features = np.stack([means[index[row["image_id"]]] for row in rows])
        if len(set(labels.tolist())) != 2 or min(np.bincount(labels)) < 30:
            raise ValueError(f"{finding}: insufficient clear development labels")
        positive, negative = features[labels == 1], features[labels == 0]
        pooled_variance = 0.5 * (positive.var(axis=0) + negative.var(axis=0))
        shrinkage = float(np.median(pooled_variance[pooled_variance > 0]))
        direction = (positive.mean(axis=0) - negative.mean(axis=0)) / (pooled_variance + shrinkage)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"{finding}: degenerate direction")
        direction /= norm
        scores = features @ direction
        auc = float(roc_auc_score(labels, scores))
        if auc < 0.5:
            direction *= -1
            scores *= -1
            auc = 1.0 - auc
        directions.append(direction.astype(np.float32))
        diagnostics[finding] = {
            "n": len(rows),
            "negative_n": int((labels == 0).sum()),
            "positive_n": int((labels == 1).sum()),
            "development_in_sample_global_mean_auroc": auc,
            "variance_shrinkage": shrinkage,
        }
    matrix = np.stack(directions)
    np.savez_compressed(
        args.output_dir / "directions.npz",
        directions=matrix,
        findings=np.asarray(findings),
    )
    receipt = {
        "version": VERSION,
        "status": "frozen",
        "model": args.model,
        "method": "diagonal LDA mean difference; variance denominator plus median-variance shrinkage; L2 normalized",
        "selection_scope": "development 0/3 and 3/3 claims only",
        "findings": list(findings),
        "dimension": int(matrix.shape[1]),
        "diagnostics": diagnostics,
        "inputs": {
            "raw_visual": str(args.raw_visual.resolve()),
            "raw_features_sha256": sha256(args.raw_visual / "features.npz"),
            "raw_metadata_sha256": sha256(args.raw_visual / "metadata.jsonl"),
            "hidden_development": str(args.hidden_development.resolve()),
            "hidden_metadata_sha256": sha256(args.hidden_development / "metadata.jsonl"),
        },
        "directions_sha256": sha256(args.output_dir / "directions.npz"),
        "command": " ".join(sys.argv),
        "source_sha256": sha256(Path(__file__)),
        "boundary": "This supervised linear direction is a mechanism probe, not an end-to-end mitigation method.",
    }
    atomic_json(args.output_dir / "receipt.json", receipt)
    print(json.dumps({"status": "frozen", "model": args.model, "findings": len(findings)}))


if __name__ == "__main__":
    main()
