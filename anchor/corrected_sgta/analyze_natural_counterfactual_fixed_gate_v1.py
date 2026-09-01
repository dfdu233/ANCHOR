#!/usr/bin/env python3
"""Evaluate a frozen layer/subset natural-transition replication gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


VERSION = "natural-counterfactual-fixed-gate-analysis-v1"


def cluster_ci(values: np.ndarray, groups: np.ndarray, draws: int, seed: int) -> list[float]:
    patients = np.unique(groups)
    buckets = [values[groups == patient] for patient in patients]
    rng = np.random.default_rng(seed)
    means = [
        float(np.concatenate([buckets[index] for index in rng.integers(0, len(buckets), len(buckets))]).mean())
        for _ in range(draws)
    ]
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def sign_flip_p(values: np.ndarray, groups: np.ndarray, draws: int, seed: int) -> float:
    patients = np.unique(groups)
    sums = np.asarray([values[groups == patient].sum() for patient in patients])
    observed = abs(float(sums.sum()))
    rng = np.random.default_rng(seed)
    exceed = sum(
        abs(float((sums * rng.choice((-1.0, 1.0), len(sums))).sum())) >= observed
        for _ in range(draws)
    )
    return float((exceed + 1) / (draws + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--preregistered-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    gate = json.loads(args.preregistered_gate.read_text())
    layer, final_layer = int(gate["layer"]), int(gate["final_layer"])
    subset = str(gate["subset"])
    if subset != "direction_name == resolved":
        raise ValueError(f"unsupported frozen subset: {subset}")
    pairs = [
        row
        for row in map(json.loads, args.raw.read_text().splitlines())
        if row.get("status") == "ok"
    ]
    rows = []
    for pair in pairs:
        controls = pair["control_findings"]
        for claim in pair["target_claims"]:
            if claim["direction_name"] != "resolved":
                continue
            values = {}
            for name, selected_layer in (("selected", layer), ("final", final_layer)):
                key = str(selected_layer)
                direction = float(claim["direction"])
                target = direction * (
                    float(pair["scores"]["current"][claim["finding"]][key])
                    - float(pair["scores"]["prior"][claim["finding"]][key])
                )
                control = np.mean(
                    [
                        direction
                        * (
                            float(pair["scores"]["current"][finding][key])
                            - float(pair["scores"]["prior"][finding][key])
                        )
                        for finding in controls
                    ]
                )
                values[name] = float(target - control)
            rows.append(
                {
                    "record_key": claim["record_key"],
                    "patient_id": pair["patient_id"],
                    "finding": claim["finding"],
                    **values,
                }
            )
    selected = np.asarray([row["selected"] for row in rows])
    final = np.asarray([row["final"] for row in rows])
    groups = np.asarray([row["patient_id"] for row in rows])
    difference = selected - final
    thresholds = gate["primary_gates"]
    selected_ci = cluster_ci(selected, groups, args.draws, args.seed)
    difference_ci = cluster_ci(difference, groups, args.draws, args.seed + 1)
    p_value = sign_flip_p(selected, groups, args.draws, args.seed + 2)
    checks = {
        "n_patients_at_least": len(np.unique(groups)) >= int(thresholds["n_patients_at_least"]),
        "layer24_mean_specificity_premium_at_least": float(selected.mean()) >= float(thresholds["layer24_mean_specificity_premium_at_least"]),
        "patient_bootstrap_95_ci_lower_above_zero": selected_ci[0] > 0,
        "patient_cluster_sign_flip_two_sided_p_below": p_value < float(thresholds["patient_cluster_sign_flip_two_sided_p_below"]),
        "layer24_minus_final_patient_bootstrap_95_ci_lower_above_zero": difference_ci[0] > 0,
        "row_positive_rate_at_least": float((selected > 0).mean()) >= float(thresholds["row_positive_rate_at_least"]),
    }
    by_finding = {}
    qualifying_positive = 0
    for finding in sorted({row["finding"] for row in rows}):
        values = np.asarray([row["selected"] for row in rows if row["finding"] == finding])
        by_finding[finding] = {"n": len(values), "mean": float(values.mean()), "positive_rate": float((values > 0).mean())}
        qualifying_positive += len(values) >= 10 and float(values.mean()) > 0
    secondary = qualifying_positive >= 2
    passed = all(checks.values()) and secondary
    result = {
        "version": VERSION,
        "status": "replicated" if passed else "replication_failed",
        "frozen_gate": gate,
        "n": len(rows),
        "n_patients": len(np.unique(groups)),
        "layer": layer,
        "mean_specificity_premium": float(selected.mean()),
        "median_specificity_premium": float(np.median(selected)),
        "positive_rate": float((selected > 0).mean()),
        "patient_bootstrap_95_ci": selected_ci,
        "patient_cluster_sign_flip_two_sided_p": p_value,
        "final_mean_specificity_premium": float(final.mean()),
        "layer_minus_final_mean": float(difference.mean()),
        "layer_minus_final_patient_bootstrap_95_ci": difference_ci,
        "primary_gate_checks": checks,
        "by_finding": by_finding,
        "secondary_heterogeneity_gate_passed": secondary,
        "decision": {
            "silver_longitudinal_layer_erasure_admitted": passed,
            "authorize_evidence_head_training": False,
            "reason": "A failed frozen replication closes the route; success would still require expert-label replication.",
        },
        "errors": sum(row.get("status") != "ok" for row in map(json.loads, args.raw.read_text().splitlines())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
