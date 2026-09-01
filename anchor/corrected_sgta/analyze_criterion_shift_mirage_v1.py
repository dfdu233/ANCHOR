#!/usr/bin/env python3
"""Audit whether a method ranking is stable across frozen CE criteria.

The input files are ``evaluation_ce_v7.json`` artifacts produced by
``evaluate_medheval_answers.py``.  This script does not reparse generations or
change any benchmark decision rule.  It compares the already-frozen strict,
official-proxy, parseable-only, and RULE-compatible summaries, then uses the
per-question records for paired image-cluster bootstrap intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


VERSION = "criterion-shift-mirage-audit-v1"
CRITERIA = {
    "strict": ("decoded_strict", "accuracy_invalid_as_error", "correct"),
    "parseable_only": ("decoded_strict", "accuracy_parseable_only", None),
    "official_proxy": (
        "official_benchmark_proxy",
        "accuracy",
        "official_benchmark_correct",
    ),
    "rule_binary": ("rule_compatible_binary_diagnostic", "accuracy", None),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected METHOD=/path/to/evaluation_ce_v7.json")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("method name and path must be nonempty")
    return name.strip(), Path(raw_path)


def sign(value: float) -> int:
    return int(value > 0) - int(value < 0)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty list")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_cluster_bootstrap(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    field: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    qids = sorted(set(left) & set(right))
    if set(left) != set(right):
        raise ValueError("paired methods do not contain identical question ids")
    by_cluster: dict[str, list[float]] = {}
    for qid in qids:
        left_row, right_row = left[qid], right[qid]
        left_cluster = str(left_row["cluster_id"])
        right_cluster = str(right_row["cluster_id"])
        if left_cluster != right_cluster:
            raise ValueError(f"cluster mismatch for {qid}")
        delta = float(bool(left_row[field])) - float(bool(right_row[field]))
        by_cluster.setdefault(left_cluster, []).append(delta)
    clusters = sorted(by_cluster)
    observed_values = [value for cluster in clusters for value in by_cluster[cluster]]
    observed = sum(observed_values) / len(observed_values)
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(draws):
        selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values = [value for cluster in selected for value in by_cluster[cluster]]
        replicates.append(sum(values) / len(values))
    return {
        "estimate": observed,
        "ci95": [quantile(replicates, 0.025), quantile(replicates, 0.975)],
        "n_questions": len(qids),
        "n_clusters": len(clusters),
        "draws": draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.input) < 2:
        parser.error("at least two --input METHOD=PATH values are required")
    if len({name for name, _ in args.input}) != len(args.input):
        parser.error("method names must be unique")

    reports: dict[str, dict[str, Any]] = {}
    details: dict[str, dict[str, dict[str, Any]]] = {}
    inputs: dict[str, dict[str, str]] = {}
    for name, path in args.input:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reports[name] = payload
        rows = {str(row["question_id"]): row for row in payload.get("details", [])}
        if len(rows) != len(payload.get("details", [])):
            raise ValueError(f"{name}: duplicate question ids")
        details[name] = rows
        inputs[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}

    metrics: dict[str, dict[str, float]] = {}
    rankings: dict[str, list[str]] = {}
    for criterion, (section, metric, _) in CRITERIA.items():
        values = {name: float(report[section][metric]) for name, report in reports.items()}
        metrics[criterion] = values
        rankings[criterion] = sorted(values, key=lambda name: (-values[name], name))

    methods = sorted(reports)
    criterion_pair_flips: dict[str, Any] = {}
    criterion_names = list(CRITERIA)
    for index, left_criterion in enumerate(criterion_names):
        for right_criterion in criterion_names[index + 1 :]:
            flips = []
            for i, left_method in enumerate(methods):
                for right_method in methods[i + 1 :]:
                    left_order = sign(
                        metrics[left_criterion][left_method]
                        - metrics[left_criterion][right_method]
                    )
                    right_order = sign(
                        metrics[right_criterion][left_method]
                        - metrics[right_criterion][right_method]
                    )
                    if left_order * right_order < 0:
                        flips.append([left_method, right_method])
            key = f"{left_criterion}__vs__{right_criterion}"
            criterion_pair_flips[key] = {
                "flips": len(flips),
                "possible_pairs": len(methods) * (len(methods) - 1) // 2,
                "method_pairs": flips,
            }

    paired_intervals: dict[str, Any] = {}
    for i, left_method in enumerate(methods):
        for right_method in methods[i + 1 :]:
            pair_key = f"{left_method}__minus__{right_method}"
            paired_intervals[pair_key] = {}
            for criterion, (_, _, detail_field) in CRITERIA.items():
                if detail_field is None:
                    continue
                paired_intervals[pair_key][criterion] = paired_cluster_bootstrap(
                    details[left_method],
                    details[right_method],
                    detail_field,
                    args.bootstrap_draws,
                    args.seed,
                )

    result = {
        "version": VERSION,
        "status": "complete",
        "inputs": inputs,
        "metrics": metrics,
        "rankings": rankings,
        "criterion_pair_ranking_flips": criterion_pair_flips,
        "paired_cluster_bootstrap": paired_intervals,
        "interpretation_boundary": (
            "A rank reversal shows criterion dependence of the reported comparison. "
            "It does not identify which criterion is clinically correct or prove a model mechanism."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
