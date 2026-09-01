#!/usr/bin/env python3
"""Analyze the missing-third-state screen without opening the holdout early."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


STATES = ("supported", "refuted", "undetermined")
ANSWER_TO_STATE = {"yes": "supported", "no": "refuted", "maybe": "undetermined"}


def softmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - array.max()
    result = np.exp(shifted)
    return result / result.sum()


def rank_auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    for left in positive:
        for right in negative:
            wins += float(left > right) + 0.5 * float(left == right)
    return wins / (len(positive) * len(negative))


def metrics(rows: list[dict[str, Any]], uncertainty_bias: float = 0.0) -> dict[str, Any]:
    confusion = Counter()
    probabilities = []
    losses = []
    briers = []
    uncertainty_advantage = {state: [] for state in STATES}
    absolute_polarity_margin = {state: [] for state in STATES}
    draft_confusion = Counter()
    for row in rows:
        logits = row["logits"].copy()
        logits["undetermined"] += uncertainty_bias
        vector = [float(logits[state]) for state in STATES]
        probs = softmax(vector)
        truth = row["state"]
        pred = STATES[int(np.argmax(probs))]
        truth_index = STATES.index(truth)
        confusion[f"{truth}->{pred}"] += 1
        probabilities.append(probs)
        losses.append(-math.log(max(float(probs[truth_index]), 1e-12)))
        target = np.eye(3)[truth_index]
        briers.append(float(np.square(probs - target).sum()))
        uncertainty_advantage[truth].append(
            float(logits["undetermined"] - max(logits["supported"], logits["refuted"]))
        )
        absolute_polarity_margin[truth].append(
            abs(float(logits["supported"] - logits["refuted"]))
        )
        draft = row.get("draft_prediction")
        draft_confusion[f"{truth}->{ANSWER_TO_STATE.get(draft, 'invalid')}"] += 1
    recalls = {}
    for state in STATES:
        denominator = sum(confusion[f"{state}->{pred}"] for pred in STATES)
        recalls[state] = confusion[f"{state}->{state}"] / denominator if denominator else None
    definite_n = sum(
        count for key, count in confusion.items() if not key.startswith("undetermined->")
    )
    definite_correct = confusion["supported->supported"] + confusion["refuted->refuted"]
    draft_recalls = {}
    for state in STATES:
        denominator = sum(
            draft_confusion[f"{state}->{pred}"]
            for pred in (*STATES, "invalid")
        )
        draft_recalls[state] = (
            draft_confusion[f"{state}->{state}"] / denominator if denominator else None
        )
    return {
        "n": len(rows),
        "accuracy": sum(confusion[f"{s}->{s}"] for s in STATES) / len(rows),
        "macro_recall": float(np.mean([value for value in recalls.values() if value is not None])),
        "recall": recalls,
        "definite_accuracy": definite_correct / definite_n if definite_n else None,
        "uncertain_overcommitment_rate": 1.0 - (recalls["undetermined"] or 0.0),
        "nll": float(np.mean(losses)),
        "multiclass_brier": float(np.mean(briers)),
        "confusion": confusion,
        "uncertainty_advantage_mean": {
            key: float(np.mean(value)) if value else None
            for key, value in uncertainty_advantage.items()
        },
        "absolute_polarity_margin_mean": {
            key: float(np.mean(value)) if value else None
            for key, value in absolute_polarity_margin.items()
        },
        "uncertainty_advantage_auc": rank_auc(
            uncertainty_advantage["undetermined"],
            uncertainty_advantage["supported"] + uncertainty_advantage["refuted"],
        ),
        "draft_recall": draft_recalls,
        "draft_confusion": draft_confusion,
    }


def select_bias(rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    candidates = np.linspace(-8.0, 8.0, 321)
    scored = [(float(bias), metrics(rows, float(bias))) for bias in candidates]
    # Freeze one scalar using macro recall; Brier and magnitude break ties.
    return max(
        scored,
        key=lambda item: (
            item[1]["macro_recall"],
            -item[1]["multiclass_brier"],
            -abs(item[0]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-bias", type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    source = {int(row["qid"]): row for row in manifest["rows"]}
    score_rows = [json.loads(line) for line in args.scores.read_text().splitlines() if line.strip()]
    rows = []
    for scored in score_rows:
        if scored.get("status") != "ok":
            continue
        meta = source[int(scored["question_id"])]
        if meta["split"] != args.split:
            continue
        rows.append({
            **meta,
            "logits": {key: float(value) for key, value in scored["original"]["logits"].items()},
            "draft_prediction": (scored.get("draft") or {}).get("prediction"),
        })
    if not rows:
        raise ValueError("no rows for split")
    baseline = metrics(rows)
    if args.split == "dev":
        bias, calibrated = select_bias(rows)
    else:
        if args.frozen_bias is None:
            raise ValueError("holdout requires --frozen-bias selected on dev")
        bias, calibrated = args.frozen_bias, metrics(rows, args.frozen_bias)
    definite_drop = baseline["definite_accuracy"] - calibrated["definite_accuracy"]
    macro_gain = calibrated["macro_recall"] - baseline["macro_recall"]
    gate = {
        "passed": (
            baseline["uncertainty_advantage_auc"] is not None
            and baseline["uncertainty_advantage_auc"] >= 0.60
            and macro_gain >= 0.05
            and definite_drop <= 0.01
        ),
        "rule": (
            "uncertainty-advantage AUROC >= .60; scalar third-state bias gains "
            ">= .05 macro recall; definite accuracy drops <= .01"
        ),
        "macro_recall_gain": macro_gain,
        "definite_accuracy_drop": definite_drop,
    }
    payload = {
        "evidence_ceiling": (
            "Patient-disjoint, finding-matched single-report uncertainty screen; "
            "not multi-reader visual ambiguity truth."
        ),
        "split": args.split,
        "baseline": baseline,
        "selected_uncertainty_logit_bias": bias,
        "calibrated": calibrated,
        "screening_gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=dict) + "\n")
    print(json.dumps(payload, indent=2, default=dict))


if __name__ == "__main__":
    main()
