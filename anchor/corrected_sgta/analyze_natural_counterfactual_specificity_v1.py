#!/usr/bin/env python3
"""Test whether natural-transition score movement is claim specific."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


VERSION = "natural-counterfactual-specificity-analysis-v1"


def patient_cluster_ci(rows: list[dict], draws: int, seed: int) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["patient_id"]].append(row["specificity_premium"])
    patients = sorted(grouped)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = rng.choice(patients, len(patients), replace=True)
        effects = [value for patient in sampled for value in grouped[str(patient)]]
        values.append(float(np.mean(effects)))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def sign_flip_p(rows: list[dict], draws: int, seed: int) -> float:
    grouped: dict[str, float] = defaultdict(float)
    for row in rows:
        grouped[row["patient_id"]] += row["specificity_premium"]
    values = np.asarray(list(grouped.values()))
    observed = abs(float(values.sum()))
    rng = np.random.default_rng(seed)
    exceed = sum(
        abs(float((values * rng.choice((-1.0, 1.0), size=len(values))).sum())) >= observed
        for _ in range(draws)
    )
    return float((exceed + 1) / (draws + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-raw", type=Path, required=True)
    parser.add_argument("--control-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    targets = [
        row
        for row in map(json.loads, args.target_raw.read_text().splitlines())
        if row.get("status") == "ok"
    ]
    controls = {
        row["pair_key"]: row
        for row in map(json.loads, args.control_raw.read_text().splitlines())
        if row.get("status") == "ok"
    }
    matched = []
    for row in targets:
        pair_key = f"{row['patient_id']}:{row['prior_study']}:{row['current_study']}"
        control = controls.get(pair_key)
        if control is None:
            continue
        target_delta = float(row["direction"]) * (
            float(row["scores"]["current"]["yes_minus_no"])
            - float(row["scores"]["prior"]["yes_minus_no"])
        )
        control_delta = float(row["direction"]) * (
            float(control["scores"]["current"]["yes_minus_no"])
            - float(control["scores"]["prior"]["yes_minus_no"])
        )
        matched.append(
            {
                "record_key": row["record_key"],
                "patient_id": row["patient_id"],
                "finding": row["finding"],
                "control_finding": control["control_finding"],
                "target_signed_delta": target_delta,
                "control_signed_delta": control_delta,
                "specificity_premium": target_delta - control_delta,
            }
        )
    if not matched:
        raise RuntimeError("no matched target/control rows")
    target_values = np.asarray([row["target_signed_delta"] for row in matched])
    control_values = np.asarray([row["control_signed_delta"] for row in matched])
    premium = target_values - control_values
    ci = patient_cluster_ci(matched, args.draws, args.seed)
    p_value = sign_flip_p(matched, args.draws, args.seed + 1)
    admitted = bool(ci[0] > 0 and p_value < 0.05)
    result = {
        "version": VERSION,
        "status": "claim_specific_signal" if admitted else "no_claim_specific_signal",
        "n": len(matched),
        "n_patients": len({row["patient_id"] for row in matched}),
        "mean_target_signed_delta": float(target_values.mean()),
        "mean_off_claim_signed_delta": float(control_values.mean()),
        "mean_specificity_premium": float(premium.mean()),
        "median_specificity_premium": float(np.median(premium)),
        "specificity_premium_95_ci": ci,
        "patient_cluster_sign_flip_two_sided_p": p_value,
        "premium_positive_rate": float((premium > 0).mean()),
        "decision": {
            "claim_specific_natural_signal_admitted": admitted,
            "authorize_evidence_head_training": False,
            "reason": (
                "A silver off-claim control can falsify generic time drift but cannot replace "
                "unanimous expert transition labels."
            ),
        },
        "boundary": (
            "The control finding is absent from the silver change list, not verified clinically stable."
        ),
        "matched_rows": matched,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "matched_rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
