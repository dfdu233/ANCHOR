#!/usr/bin/env python3
"""Summarize feature-space SGTA validation/full JSON files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import binomtest
except Exception:  # pragma: no cover - scipy may be absent on lean envs.
    binomtest = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize_file(path: Path) -> dict:
    data = json.loads(path.read_text())
    audit = data.get("prediction_audit_test", [])
    rescues = sum((not row["baseline_correct"]) and row["selected_correct"] for row in audit)
    harmful = sum(row["baseline_correct"] and (not row["selected_correct"]) for row in audit)
    p_value = None
    if binomtest is not None and rescues + harmful > 0:
        p_value = float(binomtest(min(rescues, harmful), rescues + harmful, 0.5).pvalue)
    point = data["point_accuracy"]
    best_alpha0 = data.get("best_alpha0_scat_method_diagnostic_only")
    best_alpha_positive = data.get("best_alpha_positive_feature_method_diagnostic_only")
    alpha0_accuracy = point[best_alpha0]["accuracy"] if best_alpha0 else None
    alpha_positive_accuracy = point[best_alpha_positive]["accuracy"] if best_alpha_positive else None
    return {
        "source": str(path),
        "task": path.stem,
        "seed": data["split"]["seed"],
        "n_test": data["split"]["n_test"],
        "baseline_accuracy": data["gate"]["baseline_accuracy"],
        "selected_accuracy": data["gate"]["selected_accuracy"],
        "selected_delta": data["gate"]["selected_delta"],
        "selected_method": data["selected_by_calibration"],
        "selected_calibration_accuracy": data["calibration_accuracy"][data["selected_by_calibration"]],
        "rescues": rescues,
        "harmful": harmful,
        "mcnemar_binom_p": p_value,
        "best_alpha0_method": best_alpha0,
        "best_alpha0_accuracy": alpha0_accuracy,
        "best_alpha_positive_method": best_alpha_positive,
        "best_alpha_positive_accuracy": alpha_positive_accuracy,
        "alpha_positive_minus_alpha0": None
        if alpha0_accuracy is None or alpha_positive_accuracy is None
        else alpha_positive_accuracy - alpha0_accuracy,
        "oracle_accuracy": data["gate"].get("test_method_oracle_accuracy_diagnostic_only"),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for directory in args.input_dir:
        for path in sorted(directory.glob("*.json")):
            rows.append(summarize_file(path))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    aggregate = {}
    for task, items in sorted(grouped.items()):
        aggregate[task] = {
            "seeds": len(items),
            "mean_selected_delta": sum(row["selected_delta"] for row in items) / len(items),
            "min_selected_delta": min(row["selected_delta"] for row in items),
            "positive_splits": sum(row["selected_delta"] > 0 for row in items),
            "nonnegative_splits": sum(row["selected_delta"] >= 0 for row in items),
            "mean_rescues": sum(row["rescues"] for row in items) / len(items),
            "mean_harmful": sum(row["harmful"] for row in items) / len(items),
            "mean_alpha_positive_minus_alpha0": sum(row["alpha_positive_minus_alpha0"] for row in items) / len(items),
        }
    report = {"rows": rows, "aggregate": aggregate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(aggregate, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
