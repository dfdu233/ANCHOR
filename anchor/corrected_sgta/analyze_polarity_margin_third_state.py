#!/usr/bin/env python3
"""Test whether opposing clinical claims encode the missing third state.

This is a dev-first screening diagnostic.  It ignores the model's ``Maybe``
verbalizer and derives abstention only from the magnitude of the Yes--No
margin.  Holdout evaluation requires an explicitly frozen dev threshold.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.analyze_mimic_uncertainty_triplets import metrics, rank_auc


STATES = ("supported", "refuted", "undetermined")
VERSION = "polarity-margin-third-state-v1"


def polarity_margin(row: dict[str, Any]) -> float:
    logits = row["logits"]
    return float(logits["supported"] - logits["refuted"])


def threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    confusion: Counter[str] = Counter()
    for row in rows:
        margin = polarity_margin(row)
        # A zero threshold must exactly recover the supported/refuted argmax;
        # only a strictly interior margin enters the third state.
        if abs(margin) < threshold:
            prediction = "undetermined"
        else:
            prediction = "supported" if margin >= 0 else "refuted"
        confusion[f"{row['state']}->{prediction}"] += 1
    recall = {}
    for state in STATES:
        total = sum(confusion[f"{state}->{prediction}"] for prediction in STATES)
        recall[state] = confusion[f"{state}->{state}"] / total if total else None
    definite_n = sum(
        confusion[f"{truth}->{prediction}"]
        for truth in ("supported", "refuted")
        for prediction in STATES
    )
    definite_correct = confusion["supported->supported"] + confusion["refuted->refuted"]
    return {
        "n": len(rows),
        "threshold": threshold,
        "accuracy": sum(confusion[f"{state}->{state}"] for state in STATES) / len(rows),
        "macro_recall": float(np.mean([value for value in recall.values() if value is not None])),
        "recall": recall,
        "definite_accuracy": definite_correct / definite_n if definite_n else None,
        "confusion": confusion,
    }


def uncertainty_margin_auc(rows: list[dict[str, Any]]) -> float | None:
    uncertain = [-abs(polarity_margin(row)) for row in rows if row["state"] == "undetermined"]
    definite = [
        -abs(polarity_margin(row))
        for row in rows
        if row["state"] in {"supported", "refuted"}
    ]
    return rank_auc(uncertain, definite)


def select_threshold(
    rows: list[dict[str, Any]], baseline_definite_accuracy: float
) -> tuple[float, dict[str, Any], bool]:
    magnitudes = sorted({abs(polarity_margin(row)) for row in rows})
    candidates = [0.0, *magnitudes]
    scored = [(threshold, threshold_metrics(rows, threshold)) for threshold in candidates]
    feasible = [
        item
        for item in scored
        if baseline_definite_accuracy - item[1]["definite_accuracy"] <= 0.01
    ]
    pool = feasible or scored
    selected = max(
        pool,
        key=lambda item: (
            item[1]["macro_recall"],
            item[1]["accuracy"],
            -item[0],
        ),
    )
    return selected[0], selected[1], bool(feasible)


def cluster_bootstrap_auc(
    rows: list[dict[str, Any]], seed: int = 20260801, repetitions: int = 2000
) -> dict[str, Any]:
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_subject.setdefault(str(row["subject_id"]), []).append(row)
    subjects = sorted(by_subject)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        replicate = [row for subject in sampled for row in by_subject[str(subject)]]
        value = uncertainty_margin_auc(replicate)
        if value is not None:
            estimates.append(value)
    return {
        "unit": "MIMIC subject_id",
        "repetitions": repetitions,
        "seed": seed,
        "estimate": uncertainty_margin_auc(rows),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
    }


def finding_diagnostics(rows: list[dict[str, Any]], minimum_per_state: int = 5) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["finding"]), []).append(row)
    findings = {}
    qualified = []
    for finding, values in sorted(grouped.items()):
        counts = {state: sum(row["state"] == state for row in values) for state in STATES}
        auc = uncertainty_margin_auc(values)
        means = {
            state: float(
                np.mean(
                    [abs(polarity_margin(row)) for row in values if row["state"] == state]
                )
            )
            for state in STATES
            if counts[state]
        }
        eligible = all(counts[state] >= minimum_per_state for state in STATES)
        findings[finding] = {
            "counts": counts,
            "uncertainty_margin_auc": auc,
            "absolute_margin_mean": means,
            "eligible": eligible,
        }
        if eligible:
            qualified.append(bool(auc is not None and auc >= 0.60))
    return {
        "minimum_per_state": minimum_per_state,
        "findings": findings,
        "eligible_findings": len(qualified),
        "eligible_passing_auc_0_60": sum(qualified),
        "majority_pass": bool(qualified) and sum(qualified) > len(qualified) / 2,
    }


def load_rows(manifest_path: Path, scores_path: Path, split: str) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    source = {int(row["qid"]): row for row in manifest["rows"]}
    rows = []
    for line in scores_path.read_text().splitlines():
        if not line.strip():
            continue
        scored = json.loads(line)
        if scored.get("status") != "ok":
            continue
        meta = source[int(scored["question_id"])]
        if meta["split"] != split:
            continue
        rows.append(
            {
                **meta,
                "logits": {
                    key: float(value)
                    for key, value in scored["original"]["logits"].items()
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-threshold", type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = load_rows(args.manifest, args.scores, args.split)
    if not rows:
        raise ValueError("no rows for split")
    original = metrics(rows)
    auc = cluster_bootstrap_auc(rows)
    if args.split == "dev":
        threshold, projected, feasibility = select_threshold(
            rows, original["definite_accuracy"]
        )
    else:
        if args.frozen_threshold is None:
            raise ValueError("holdout requires --frozen-threshold selected on dev")
        threshold = args.frozen_threshold
        projected = threshold_metrics(rows, threshold)
        feasibility = True
    macro_gain = projected["macro_recall"] - original["macro_recall"]
    definite_drop = original["definite_accuracy"] - projected["definite_accuracy"]
    gate = {
        "passed": (
            auc["estimate"] is not None
            and auc["estimate"] >= 0.60
            and auc["ci95"][0] > 0.50
            and macro_gain >= 0.05
            and definite_drop <= 0.01
        ),
        "rule": (
            "patient-bootstrap margin-uncertainty AUROC >= .60 with CI lower > .50; "
            "macro recall gain >= .05; definite accuracy drop <= .01"
        ),
        "selection_had_no_harm_feasible_threshold": feasibility,
        "macro_recall_gain": macro_gain,
        "definite_accuracy_drop": definite_drop,
    }
    payload = {
        "version": VERSION,
        "evidence_ceiling": (
            "Patient-disjoint single-report linguistic uncertainty; not multi-reader visual ambiguity."
        ),
        "split": args.split,
        "original_three_logit_baseline": original,
        "polarity_margin_uncertainty_auc": auc,
        "finding_diagnostics": finding_diagnostics(rows),
        "selected_threshold": threshold,
        "margin_projection": projected,
        "screening_gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=dict) + "\n")
    print(json.dumps(payload, indent=2, default=dict))


if __name__ == "__main__":
    main()
