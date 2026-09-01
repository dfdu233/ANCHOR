#!/usr/bin/env python3
"""Test whether a worst-prior claim score improves truth ranking.

This is the predeclared mitigation screen paired with the prior-titration
probe. It compares the neutral-prompt margin with the minimum margin over
low/neutral/high stated priors. All uncertainty intervals resample images
within finding x reference-polarity strata. The method passes only if the
worst-prior score improves AUROC and does not reduce precision at fixed 50%
claim coverage in every supplied model.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.run_huatuo_vindr_commitment_probe import sha256_file
from corrected_sgta.run_slake_prior_titration_probe import polarity_margin


VERSION = "slake-prior-robustness-screen-v1"


def auc(labels: list[bool], scores: list[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        raise ValueError("AUROC requires both classes")
    comparisons = [
        float(p > n) + 0.5 * float(p == n) for p in positive for n in negative
    ]
    return float(np.mean(comparisons))


def row_scores(row: dict[str, Any]) -> dict[str, float]:
    margins = {
        prior: polarity_margin(score) for prior, score in row["scores"].items()
    }
    return {
        "neutral": margins["neutral"],
        "worst_prior": min(margins.values()),
        "prior_mean": float(np.mean(list(margins.values()))),
    }


def precision_at_half(labels: list[bool], scores: list[float]) -> float:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    selected = order[: len(order) // 2]
    return float(np.mean([labels[index] for index in selected]))


def resample_stratified(
    rows: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    sampled = []
    keys = sorted({
        (str(row["finding"]), str(row["reference_polarity"])) for row in rows
    })
    for key in keys:
        stratum = [
            row for row in rows
            if (str(row["finding"]), str(row["reference_polarity"])) == key
        ]
        indices = rng.integers(0, len(stratum), len(stratum))
        sampled.extend(stratum[index] for index in indices)
    return sampled


def analyze_rows(
    rows: list[dict[str, Any]], seed: int, draws: int
) -> dict[str, Any]:
    rows = [row for row in rows if row.get("status") == "ok"]
    labels = [row["reference_polarity"] == "positive" for row in rows]
    scores = [row_scores(row) for row in rows]
    metrics = {
        method: {
            "auroc": auc(labels, [score[method] for score in scores]),
            "precision_at_50pct_claim_coverage": precision_at_half(
                labels, [score[method] for score in scores]
            ),
        }
        for method in ("neutral", "worst_prior", "prior_mean")
    }
    observed = metrics["worst_prior"]["auroc"] - metrics["neutral"]["auroc"]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        sample = resample_stratified(rows, rng)
        sample_labels = [
            row["reference_polarity"] == "positive" for row in sample
        ]
        sample_scores = [row_scores(row) for row in sample]
        deltas.append(
            auc(sample_labels, [score["worst_prior"] for score in sample_scores])
            - auc(sample_labels, [score["neutral"] for score in sample_scores])
        )
    delta = {
        "estimate": observed,
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }
    passed = (
        delta["ci_low"] > 0.0
        and metrics["worst_prior"]["precision_at_50pct_claim_coverage"]
        >= metrics["neutral"]["precision_at_50pct_claim_coverage"]
    )
    return {
        "n": len(rows),
        "metrics": metrics,
        "worst_prior_minus_neutral_auroc": delta,
        "screening_gate_passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    models = {}
    inputs = []
    for index, run_dir in enumerate(args.run_dirs):
        raw_path = run_dir / "raw.jsonl"
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        models[config["model"]] = analyze_rows(
            rows, args.seed + 100 * index, args.bootstrap_draws
        )
        inputs.append({
            "model": config["model"],
            "raw_path": str(raw_path.resolve()),
            "raw_sha256": sha256_file(raw_path),
            "config_sha256": sha256_file(config_path),
        })
    result = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": sha256_file(Path(__file__)),
        "inputs": inputs,
        "models": models,
        "all_models_passed": all(
            result["screening_gate_passed"] for result in models.values()
        ),
        "claim_ceiling": (
            "SLAKE labels test binary finding truth, not reader uncertainty or "
            "open-ended report hallucination. Failure prunes the proposed score; "
            "success would only authorize a larger OE test."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
