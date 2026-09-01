#!/usr/bin/env python3
"""Analyze the strict VinDr delete-then-relocate causal law."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


VERSION = "vindr-lesion-relocation-analysis-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("status") != "ok" for row in rows):
        raise ValueError("raw input is empty or incomplete")
    original = np.asarray([row["scores"]["original"]["yes_minus_no"] for row in rows])
    deletion = np.asarray([row["scores"]["deletion"]["yes_minus_no"] for row in rows])
    relocation = np.asarray([row["scores"]["relocation"]["yes_minus_no"] for row in rows])
    delete_drop = original - deletion
    relocation_recovery = relocation - deletion
    relocation_overshoot = relocation - original
    joint = (delete_drop > 0) & (relocation_recovery > 0)
    admitted = original > 0
    rng = np.random.default_rng(20260806)
    boot = np.empty((args.draws, 4))
    for draw in range(args.draws):
        idx = rng.integers(0, len(rows), len(rows))
        boot[draw] = [
            delete_drop[idx].mean(),
            relocation_recovery[idx].mean(),
            relocation_overshoot[idx].mean(),
            joint[idx].mean(),
        ]

    def metric(values: np.ndarray, column: int) -> dict:
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "bootstrap_95_ci": [float(np.quantile(boot[:, column], 0.025)), float(np.quantile(boot[:, column], 0.975))],
            "positive_n": int((values > 0).sum()),
            "positive_rate": float((values > 0).mean()),
            "zero_rate": float((values == 0).mean()),
        }

    delete_ci = [
        float(np.quantile(boot[:, 0], 0.025)),
        float(np.quantile(boot[:, 0], 0.975)),
    ]
    recovery_ci = [
        float(np.quantile(boot[:, 1], 0.025)),
        float(np.quantile(boot[:, 1], 0.975)),
    ]
    admitted_directional = bool(
        admitted.any()
        and delete_drop[admitted].mean() > 0
        and relocation_recovery[admitted].mean() > 0
    )
    gate_pass = bool(
        delete_ci[0] > 0
        and recovery_ci[0] > 0
        and (delete_drop > 0).mean() >= 0.60
        and (relocation_recovery > 0).mean() >= 0.60
        and admitted_directional
    )
    result = {
        "version": VERSION,
        "status": "complete",
        "input": {"path": str(args.raw.resolve()), "sha256": sha256(args.raw)},
        "configuration": {
            "bootstrap_draws": args.draws,
            "bootstrap_seed": 20260806,
            "source_sha256": sha256(Path(__file__)),
            "command": " ".join(sys.argv),
        },
        "n": len(rows),
        "finding": sorted({row["finding"] for row in rows}),
        "directional_admission_rate_original_margin_positive": float((original > 0).mean()),
        "mean_scores": {
            "original": float(original.mean()),
            "deletion": float(deletion.mean()),
            "relocation": float(relocation.mean()),
        },
        "deletion_drop": metric(delete_drop, 0),
        "relocation_recovery": metric(relocation_recovery, 1),
        "relocation_minus_original": metric(relocation_overshoot, 2),
        "joint_causal_law": {
            "definition": "original>deletion and relocation>deletion",
            "n": int(joint.sum()),
            "rate": float(joint.mean()),
            "bootstrap_95_ci": [float(np.quantile(boot[:, 3], 0.025)), float(np.quantile(boot[:, 3], 0.975))],
        },
        "admitted_subset": {
            "definition": "original margin > 0",
            "n": int(admitted.sum()),
            "deletion_drop_mean": float(delete_drop[admitted].mean()) if admitted.any() else None,
            "relocation_recovery_mean": float(relocation_recovery[admitted].mean()) if admitted.any() else None,
            "both_mean_directions_positive": admitted_directional,
        },
        "decision": {
            "gate": (
                "deletion and relocation mean-effect CI lower bounds >0; each "
                "positive on >=60% of cases; admitted subset mean directions >0"
            ),
            "pass": gate_pass,
            "lesion_sensitivity_admitted": gate_pass,
            "advance_evidence_adapter": gate_pass,
            "reason": (
                "The frozen bidirectional causal law passed."
                if gate_pass
                else "At least one frozen bidirectional causal-law condition failed."
            ),
        },
        "boundary": "Focal consensus boxes and patient-internal contralateral tissue reduce, but do not eliminate, counterfactual editing artifacts.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
