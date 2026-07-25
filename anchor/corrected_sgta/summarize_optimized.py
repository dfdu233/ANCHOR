#!/usr/bin/env python3
"""Aggregate repeated split seeds for optimized CE and OE reports."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce-dir", type=Path)
    parser.add_argument("--oe-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def statistics(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    return {
        "n": int(len(finite)),
        "mean": float(finite.mean()) if len(finite) else None,
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0 if len(finite) else None,
        "min": float(finite.min()) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
    }


def seed_from_name(path: Path) -> int:
    match = re.search(r"\.seed(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"missing seed suffix: {path}")
    return int(match.group(1))


def ce_group_name(path: Path) -> str:
    return re.sub(r"\.sgta_optimized\.seed\d+\.json$", "", path.name)


def oe_group_name(path: Path) -> str:
    return re.sub(r"\.confgen_optimized\.seed\d+\.json$", "", path.name)


def aggregate_ce(directory: Path | None) -> dict:
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    if directory is None or not directory.exists():
        return {}
    for path in sorted(directory.glob("*.sgta_optimized.seed*.json")):
        grouped[ce_group_name(path)].append((seed_from_name(path), json.loads(path.read_text())))
    output = {}
    for name, reports in grouped.items():
        methods = sorted(
            set.intersection(
                *(set(report["test_results"]) for _, report in reports)
            )
        )
        output[name] = {
            "seeds": [seed for seed, _ in reports],
            "methods": {
                method: {
                    metric: statistics(
                        [float(report["test_results"][method][metric]) for _, report in reports]
                    )
                    for metric in ("accuracy", "nll")
                }
                for method in methods
            },
            "test_delta_vs_baseline": statistics(
                [float(report["test_delta_vs_baseline"]) for _, report in reports]
            ),
            "selected_configurations": [
                report["selected_by_calibration_cv"] for _, report in reports
            ],
        }
    return output


def aggregate_oe(directory: Path | None) -> dict:
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    if directory is None or not directory.exists():
        return {}
    for path in sorted(directory.glob("*.confgen_optimized.seed*.json")):
        grouped[oe_group_name(path)].append((seed_from_name(path), json.loads(path.read_text())))
    output = {}
    scalar_metrics = (
        "empirical_coverage",
        "coverage_gap",
        "average_set_size",
        "average_unique_set_size",
        "empty_set_rate",
        "confidence_reduced_admissibility_rate",
    )
    for name, reports in grouped.items():
        common_methods = sorted(
            set.intersection(*(set(report["methods"]) for _, report in reports))
        )
        method_output = {}
        for method in common_methods:
            common_gammas = sorted(
                set.intersection(
                    *(set(report["methods"][method]["gamma"]) for _, report in reports)
                ),
                key=float,
            )
            gamma_output = {}
            for gamma in common_gammas:
                rows = [report["methods"][method]["gamma"][gamma] for _, report in reports]
                metrics = {
                    metric: statistics(
                        [float(row[metric]) for row in rows if row.get(metric) is not None]
                    )
                    for metric in scalar_metrics
                }
                reduced_keys = rows[0]["confidence_reduced_output_metrics"].keys()
                metrics["confidence_reduced_output_metrics"] = {
                    key: statistics(
                        [
                            float(row["confidence_reduced_output_metrics"][key])
                            for row in rows
                            if row["confidence_reduced_output_metrics"].get(key) is not None
                        ]
                    )
                    for key in reduced_keys
                }
                gamma_output[gamma] = metrics
            method_output[method] = gamma_output
        output[name] = {
            "seeds": [seed for seed, _ in reports],
            "methods": method_output,
        }
    return output


def main() -> None:
    args = parse_args()
    report = {
        "ce": aggregate_ce(args.ce_dir),
        "oe": aggregate_oe(args.oe_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
