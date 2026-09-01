#!/usr/bin/env python3
"""Leakage-safe layer selection for claim-specific natural transition signals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


VERSION = "natural-counterfactual-layerwise-analysis-v1"


def cluster_ci(values: np.ndarray, groups: np.ndarray, draws: int, seed: int) -> list[float]:
    patients = np.unique(groups)
    grouped = [values[groups == patient] for patient in patients]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        sample = rng.integers(0, len(patients), size=len(patients))
        means.append(float(np.concatenate([grouped[index] for index in sample]).mean()))
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    pairs = [
        row
        for row in map(json.loads, args.raw.read_text().splitlines())
        if row.get("status") == "ok"
    ]
    if not pairs:
        raise RuntimeError("no successful pair rows")
    layer_sets = {
        tuple(sorted(map(int, next(iter(row["scores"]["current"].values())).keys())))
        for row in pairs
    }
    if len(layer_sets) != 1:
        raise RuntimeError("inconsistent decoder layer sets")
    layers = list(next(iter(layer_sets)))
    final_layer = max(layers)
    candidate_layers = [layer for layer in layers if 0 < layer < final_layer]
    expanded = []
    for pair in pairs:
        controls = pair["control_findings"]
        if not controls:
            continue
        for claim in pair["target_claims"]:
            premiums = []
            target_deltas = []
            control_deltas = []
            for layer in layers:
                key = str(layer)
                direction = float(claim["direction"])
                target_delta = direction * (
                    float(pair["scores"]["current"][claim["finding"]][key])
                    - float(pair["scores"]["prior"][claim["finding"]][key])
                )
                control_delta = float(
                    np.mean(
                        [
                            direction
                            * (
                                float(pair["scores"]["current"][finding][key])
                                - float(pair["scores"]["prior"][finding][key])
                            )
                            for finding in controls
                        ]
                    )
                )
                target_deltas.append(target_delta)
                control_deltas.append(control_delta)
                premiums.append(target_delta - control_delta)
            expanded.append(
                {
                    **claim,
                    "patient_id": pair["patient_id"],
                    "pair_key": pair["pair_key"],
                    "premiums": premiums,
                    "target_deltas": target_deltas,
                    "control_deltas": control_deltas,
                }
            )
    matrix = np.asarray([row["premiums"] for row in expanded], dtype=float)
    groups = np.asarray([row["patient_id"] for row in expanded])
    if len(np.unique(groups)) < 5:
        raise RuntimeError("fewer than five patient groups")
    layer_to_column = {layer: index for index, layer in enumerate(layers)}

    oof = np.full(len(expanded), np.nan)
    selected_folds = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(splitter.split(matrix, groups=groups)):
        train_means = {
            layer: float(matrix[train, layer_to_column[layer]].mean())
            for layer in candidate_layers
        }
        selected = max(candidate_layers, key=lambda layer: (train_means[layer], -layer))
        oof[test] = matrix[test, layer_to_column[selected]]
        selected_folds.append(
            {
                "fold": fold,
                "selected_layer": selected,
                "train_mean_premium": train_means[selected],
                "test_n": len(test),
                "test_patients": sorted(set(groups[test])),
                "test_mean_premium": float(oof[test].mean()),
            }
        )
    if np.isnan(oof).any():
        raise RuntimeError("incomplete out-of-fold layer scores")
    final = matrix[:, layer_to_column[final_layer]]
    improvement = oof - final
    oof_ci = cluster_ci(oof, groups, args.draws, args.seed)
    improvement_ci = cluster_ci(improvement, groups, args.draws, args.seed + 1)
    oof_p = sign_flip_p(oof, groups, args.draws, args.seed + 2)

    per_layer = {}
    for layer in layers:
        values = matrix[:, layer_to_column[layer]]
        per_layer[str(layer)] = {
            "mean_specificity_premium": float(values.mean()),
            "patient_cluster_95_ci": cluster_ci(
                values, groups, args.draws, args.seed + 1000 + layer
            ),
        }
    admitted = bool(
        float(oof.mean()) >= 0.125
        and oof_ci[0] > 0
        and improvement_ci[0] > 0
        and oof_p < 0.05
    )
    result = {
        "version": VERSION,
        "status": "recoverable_intermediate_specificity" if admitted else "no_recoverable_intermediate_specificity",
        "n": len(expanded),
        "n_pairs": len(pairs),
        "n_patients": len(np.unique(groups)),
        "layers": layers,
        "final_layer_index": final_layer,
        "cross_validation": {
            "protocol": "five-fold patient GroupKFold; layer selected by train mean specificity premium only",
            "selected_layer_counts": dict(sorted(Counter(row["selected_layer"] for row in selected_folds).items())),
            "folds": selected_folds,
            "oof_mean_specificity_premium": float(oof.mean()),
            "oof_specificity_premium_95_ci": oof_ci,
            "oof_patient_sign_flip_two_sided_p": oof_p,
            "oof_positive_rate": float((oof > 0).mean()),
        },
        "final_layer": {
            "index": final_layer,
            "mean_specificity_premium": float(final.mean()),
            "patient_cluster_95_ci": cluster_ci(final, groups, args.draws, args.seed + 3),
        },
        "selected_minus_final": {
            "mean": float(improvement.mean()),
            "patient_cluster_95_ci": improvement_ci,
        },
        "per_layer_descriptive": per_layer,
        "decision": {
            "intermediate_claim_specific_signal_admitted": admitted,
            "thresholds": {
                "oof_mean_at_least": 0.125,
                "oof_ci_lower_above_zero": True,
                "selected_minus_final_ci_lower_above_zero": True,
                "patient_sign_flip_p_below": 0.05,
            },
            "authorize_evidence_head_training": False,
            "reason": (
                "Silver labels can kill the mechanism but cannot authorize training without "
                "replication on unanimous expert progression labels."
            ),
        },
        "boundary": (
            "Layer selection is leakage-safe by patient, but labels and off-claim stability remain silver."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_layer_descriptive"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
