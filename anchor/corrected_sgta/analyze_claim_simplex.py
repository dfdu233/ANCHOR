#!/usr/bin/env python3
"""Analyze whether claim evidence occupies a plane rather than a signed line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from corrected_sgta.clinical_claims import epistemic_coordinates


VERSION = "claim-simplex-analysis-v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def auc(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def bootstrap(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
    seed: int,
    draws: int,
) -> dict[str, float | int | None]:
    estimate = statistic(rows)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sample = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        value = statistic(sample)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(values, 0.025)) if values else None,
        "ci_high": float(np.quantile(values, 0.975)) if values else None,
        "valid_draws": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [row for row in load_jsonl(args.input) if row.get("status") == "ok"]
    if not rows:
        raise ValueError("no successful probe rows")
    layers = sorted(
        {int(layer) for row in rows for layer in row["measurement"]["trajectory"]}
    )
    config_path = args.input.parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    tau = float(config.get("tau", 0.5))
    result: dict[str, Any] = {
        "version": VERSION,
        "input": str(args.input.resolve()),
        "n": len(rows),
        "legacy_tau": tau,
        "reference_ceiling": (
            "Report-derived grade-C labels; geometry is diagnostic and clinical claims "
            "require reader/expert reference replication."
        ),
        "per_layer": {},
    }
    for layer in layers:
        values = []
        for row in rows:
            trajectory = row["measurement"]["trajectory"][str(layer)]
            real = trajectory["real_logits"]
            null = trajectory["null_logits"]
            visual = epistemic_coordinates(real, null)
            real_coordinates = epistemic_coordinates(real)
            null_coordinates = epistemic_coordinates(null)
            predicted = max(trajectory["real_probabilities"], key=trajectory["real_probabilities"].get)
            values.append(
                {
                    "error": int(predicted != row["reader_state"]),
                    "visual_polarity": float(visual["polarity"]),
                    "visual_commitment": float(visual["commitment"]),
                    "real_polarity": float(real_coordinates["polarity"]),
                    "real_commitment": float(real_coordinates["commitment"]),
                    "null_commitment": float(null_coordinates["commitment"]),
                    "legacy_residual": float(
                        visual["commitment"]
                        - (abs(float(trajectory["signed_visual_evidence"])) - tau)
                    ),
                }
            )
        polarity = np.asarray([abs(row["visual_polarity"]) for row in values])
        commitment = np.asarray([row["visual_commitment"] for row in values])
        design = np.column_stack([np.ones(len(values)), polarity])
        coefficients = np.linalg.lstsq(design, commitment, rcond=None)[0]
        fitted = design @ coefficients
        denominator = float(np.sum((commitment - commitment.mean()) ** 2))
        r_squared = 1.0 - float(np.sum((commitment - fitted) ** 2)) / denominator if denominator else None
        labels = [row["error"] for row in values]

        def margin_auc(batch: list[dict[str, Any]]) -> float | None:
            return auc([row["error"] for row in batch], [-abs(row["real_polarity"]) for row in batch])

        def commitment_auc(batch: list[dict[str, Any]]) -> float | None:
            return auc([row["error"] for row in batch], [-row["visual_commitment"] for row in batch])

        def unearned_auc(batch: list[dict[str, Any]]) -> float | None:
            return auc(
                [row["error"] for row in batch],
                [row["real_commitment"] - row["visual_commitment"] for row in batch],
            )

        rho = spearmanr(polarity, commitment).statistic if len(values) > 1 else None
        result["per_layer"][str(layer)] = {
            "n_errors": sum(labels),
            "visual_commitment_mean": float(commitment.mean()),
            "visual_commitment_std": float(commitment.std()),
            "absolute_visual_polarity_mean": float(polarity.mean()),
            "spearman_abs_polarity_vs_commitment": float(rho) if np.isfinite(rho) else None,
            "linear_r_squared_commitment_from_abs_polarity": r_squared,
            "legacy_constraint_mean_absolute_residual": float(
                np.mean([abs(row["legacy_residual"]) for row in values])
            ),
            "error_auroc_low_real_margin": bootstrap(values, margin_auc, args.seed + layer, args.bootstrap_draws),
            "error_auroc_low_visual_commitment": bootstrap(values, commitment_auc, args.seed + 100 + layer, args.bootstrap_draws),
            "error_auroc_unearned_commitment": bootstrap(values, unearned_auc, args.seed + 200 + layer, args.bootstrap_draws),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
