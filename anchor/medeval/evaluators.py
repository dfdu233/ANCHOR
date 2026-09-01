"""Task-separated evaluators; OE and reports cannot fall through to CE parsing."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable


LEADING_DECISION = re.compile(r"^\s*(?:answer\s*:\s*)?(yes|no)\b", re.IGNORECASE)


def parse_leading_binary(text: str) -> str | None:
    match = LEADING_DECISION.search(text)
    return match.group(1).lower() if match else None


def evaluate_ce_generation(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    per_sample = []
    for row in rows:
        truth = str(row["reference"]).strip().lower()
        if truth not in {"yes", "no"}:
            raise ValueError(f"CE reference must be yes/no, got {truth!r}")
        prediction = parse_leading_binary(str(row["prediction"]))
        valid = prediction is not None
        correct = valid and prediction == truth
        counts["n"] += 1
        counts["valid"] += int(valid)
        counts["correct"] += int(correct)
        if valid:
            counts[f"truth_{truth}_prediction_{prediction}"] += 1
        per_sample.append({
            "sample_id": str(row["sample_id"]),
            "reference": truth,
            "decision": prediction,
            "valid": valid,
            "correct": correct,
        })
    n = counts["n"]
    tp = counts["truth_yes_prediction_yes"]
    tn = counts["truth_no_prediction_no"]
    fp = counts["truth_no_prediction_yes"]
    fn = counts["truth_yes_prediction_no"]
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "task_kind": "ce_generation",
        "n": n,
        "accuracy": counts["correct"] / n if n else None,
        "valid_parse_rate": counts["valid"] / n if n else None,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None else None
        ),
        "predicted_positive_prevalence": (tp + fp) / n if n else None,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "per_sample": per_sample,
    }


def require_evaluator(task_kind: str, evaluator_kind: str) -> None:
    allowed = {
        "ce_decision": {"ce_decision"},
        "ce_generation": {"ce_generation"},
        "oe_vqa": {"oe_claims"},
        "report_generation": {"report_claims"},
    }
    if evaluator_kind not in allowed.get(task_kind, set()):
        raise ValueError(
            f"evaluator {evaluator_kind!r} is invalid for task {task_kind!r}; "
            f"allowed={sorted(allowed.get(task_kind, set()))}"
        )
