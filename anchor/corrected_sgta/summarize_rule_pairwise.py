#!/usr/bin/env python3
"""Summarize paired RULE decision-first flips from evaluation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def is_correct(row: dict) -> bool:
    stored = row.get("decision_first_correct")
    if stored is not None:
        return bool(stored)
    ground_truth = row.get("ground_truth_explicit")
    prediction = row.get("decision_first_prediction")
    return ground_truth is not None and prediction == ground_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    method = load(args.method)
    baseline_by_id = {str(row["question_id"]): row for row in baseline}
    method_by_id = {str(row["question_id"]): row for row in method}
    if baseline_by_id.keys() != method_by_id.keys():
        raise ValueError("baseline/method question ids differ")

    pairs = []
    for identifier, base_row in baseline_by_id.items():
        method_row = method_by_id[identifier]
        base_correct = is_correct(base_row)
        method_correct = is_correct(method_row)
        pairs.append(
            {
                "question_id": identifier,
                "baseline_correct": base_correct,
                "method_correct": method_correct,
                "baseline_prediction": base_row.get(
                    "decision_first_prediction"
                ),
                "method_prediction": method_row.get(
                    "decision_first_prediction"
                ),
                "ground_truth": base_row.get("ground_truth_explicit"),
            }
        )
    rescue = sum(
        not row["baseline_correct"] and row["method_correct"] for row in pairs
    )
    harm = sum(
        row["baseline_correct"] and not row["method_correct"] for row in pairs
    )
    payload = {
        "version": "rule-decision-first-paired-v1",
        "method": args.method_name,
        "n": len(pairs),
        "baseline_correct": sum(row["baseline_correct"] for row in pairs),
        "method_correct": sum(row["method_correct"] for row in pairs),
        "wrong_to_correct": rescue,
        "correct_to_wrong": harm,
        "net_rescue": rescue - harm,
        "decision_flips": sum(
            row["baseline_prediction"] != row["method_prediction"]
            for row in pairs
        ),
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "pairs"},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
