"""Test whether style instability adds error signal beyond decision margin.

This is an offline, post-hoc diagnostic for outputs produced by
run_huatuo_style_phenomenon_confirm.py.  It does not establish that a style
view preserves clinical semantics, and it does not equate a VQA error with a
free-form clinical hallucination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


VERSION = "huatuo-style-boundary-analysis-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return pairwise AUROC with half credit for ties."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    wins = 0.0
    for value in positive:
        wins += float(np.sum(value > negative))
        wins += 0.5 * float(np.sum(value == negative))
    return wins / float(len(positive) * len(negative))


def bootstrap_auc_statistics(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    seed: int,
    n_bootstrap: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    names = list(features)
    observed = {name: auroc(features[name], labels) for name in names}
    draws: dict[str, list[float]] = {name: [] for name in names}
    n = len(labels)
    attempts = 0
    maximum_attempts = max(n_bootstrap * 20, 1000)
    while len(draws[names[0]]) < n_bootstrap and attempts < maximum_attempts:
        attempts += 1
        indices = rng.integers(0, n, size=n)
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) != 2:
            continue
        for name in names:
            draws[name].append(auroc(features[name][indices], sampled_labels))
    if len(draws[names[0]]) != n_bootstrap:
        raise RuntimeError("could not obtain enough valid bootstrap replicates")

    summaries = {}
    for name in names:
        values = np.asarray(draws[name], dtype=np.float64)
        summaries[name] = {
            "auroc": observed[name],
            "bootstrap_90ci": [float(v) for v in np.quantile(values, [0.05, 0.95])],
        }

    baseline = "negative_abs_margin"
    differences = {}
    baseline_draws = np.asarray(draws[baseline], dtype=np.float64)
    for name in names:
        if name == baseline:
            continue
        values = np.asarray(draws[name], dtype=np.float64) - baseline_draws
        differences[f"{name}_minus_{baseline}"] = {
            "observed": observed[name] - observed[baseline],
            "bootstrap_mean": float(values.mean()),
            "bootstrap_90ci": [float(v) for v in np.quantile(values, [0.05, 0.95])],
        }
    return summaries, differences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--relative-epsilon", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("status") == "ok"]
    if not rows:
        raise RuntimeError("input contains no successful rows")

    patients = [str(row["patient_id"]) for row in rows]
    if len(set(patients)) != len(patients):
        raise ValueError("patient IDs are not unique; use clustered bootstrap instead")

    style_names = [
        name
        for name in rows[0]["scores"]
        if name != "original"
    ]
    if not style_names:
        raise ValueError("no style views found")

    labels = []
    margins = []
    drift_matrix = []
    union_flips = []
    for row in rows:
        original = row["scores"]["original"]
        labels.append(
            int(str(original["prediction"]).lower() != str(row["ground_truth"]).lower())
        )
        original_margin = float(original["yes_minus_no"])
        margins.append(abs(original_margin))
        drifts = [
            abs(float(row["scores"][style]["yes_minus_no"]) - original_margin)
            for style in style_names
        ]
        drift_matrix.append(drifts)
        union_flips.append(
            int(
                any(
                    str(row["scores"][style]["prediction"]).lower()
                    != str(original["prediction"]).lower()
                    for style in style_names
                )
            )
        )

    error_labels = np.asarray(labels, dtype=np.int64)
    abs_margin = np.asarray(margins, dtype=np.float64)
    drift = np.asarray(drift_matrix, dtype=np.float64)
    mean_drift = drift.mean(axis=1)
    max_drift = drift.max(axis=1)
    relative_drift = mean_drift / (abs_margin + args.relative_epsilon)
    union_flip = np.asarray(union_flips, dtype=np.int64)

    features = {
        "negative_abs_margin": -abs_margin,
        "mean_abs_style_drift": mean_drift,
        "max_abs_style_drift": max_drift,
        "relative_style_drift": relative_drift,
    }
    aucs, differences = bootstrap_auc_statistics(
        features,
        error_labels,
        args.seed,
        args.n_bootstrap,
    )

    rho, rho_p = spearmanr(mean_drift, abs_margin)
    margin_median = float(np.median(abs_margin))
    low_margin = abs_margin <= margin_median
    high_margin = ~low_margin
    flipped = union_flip == 1
    unflipped = ~flipped

    payload = {
        "version": VERSION,
        "analysis_status": "post_hoc_mechanism_lead_only",
        "n": len(rows),
        "n_errors": int(error_labels.sum()),
        "style_names": style_names,
        "error_detection": aucs,
        "paired_auc_differences": differences,
        "margin_drift_association": {
            "spearman_rho": float(rho),
            "p_two_sided": float(rho_p),
        },
        "union_decision_flips": {
            "n": int(union_flip.sum()),
            "low_margin_n": int(union_flip[low_margin].sum()),
            "high_margin_n": int(union_flip[high_margin].sum()),
            "margin_median": margin_median,
            "error_rate_flipped": float(error_labels[flipped].mean()) if flipped.any() else None,
            "error_rate_unflipped": float(error_labels[unflipped].mean()) if unflipped.any() else None,
        },
        "provenance": {
            "input": str(args.input.resolve()),
            "input_sha256": file_sha256(args.input),
            "code_sha256": file_sha256(Path(__file__)),
            "command": sys.argv,
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "relative_epsilon": args.relative_epsilon,
        },
        "claim_boundary": (
            "This post-hoc analysis compares first-token binary VQA error signals "
            "for one model, one dataset, and three mild image transforms. It neither "
            "proves a universal boundary mechanism nor evaluates token-level report "
            "hallucinations or clinical semantic preservation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
