#!/usr/bin/env python3
"""Classify preregistered VinDr reader-clarity layer boundaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "vindr-reader-layer-boundary-classifier-v1"
STATES = {"early_erasure", "late_emergence", "layer_stable", "not_decodable", "indeterminate"}


def classify(row: dict[str, Any], margin: float = 0.05) -> str:
    counts = row["test_vote_bin_counts"]
    required_bins = row["direction_bins"]
    minimum = int(row.get("minimum_test_per_vote_bin", 10))
    if any(int(counts.get(name, 0)) < minimum for name in required_bins):
        return "indeterminate"
    delta = row["early_minus_final_auroc"]
    estimate, low, high = (float(delta[key]) for key in ("estimate", "ci_low", "ci_high"))
    controls = row["increment_over_strongest_control"]
    early_control_high = float(controls["early"]["ci_high"])
    final_control_high = float(controls["final"]["ci_high"])
    if (
        bool(row.get("powered_for_margin", False))
        and bool(row.get("all_preregistered_controls_present", False))
        and early_control_high <= 0
        and final_control_high <= 0
    ):
        return "not_decodable"
    if estimate >= margin and low > 0 and bool(row.get("causal_patch_passed", False)):
        return "early_erasure"
    if estimate <= -margin and high < 0:
        return "late_emergence"
    if low >= -margin and high <= margin:
        return "layer_stable"
    return "indeterminate"


def summarize(rows: list[dict[str, Any]], margin: float = 0.05) -> dict[str, Any]:
    classified = [{**row, "boundary_state": classify(row, margin)} for row in rows]
    by_model_finding: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in classified:
        direction = str(row["direction"])
        if direction not in {"negative", "positive"}:
            raise ValueError(f"unsupported direction: {direction}")
        key = (str(row["model_id"]), str(row["finding"]))
        if direction in by_model_finding[key]:
            raise ValueError(f"duplicate model/finding/direction: {key + (direction,)}")
        by_model_finding[key][direction] = row["boundary_state"]
    finding_rows = []
    for (model, finding), directions in sorted(by_model_finding.items()):
        complete = set(directions) == {"negative", "positive"}
        finding_rows.append({
            "model_id": model,
            "finding": finding,
            "directions": directions,
            "both_directions_complete": complete,
            "early_erasure_both_directions": complete and set(directions.values()) == {"early_erasure"},
        })
    model_gate = {}
    for model in sorted({row["model_id"] for row in finding_rows}):
        eligible = [row for row in finding_rows if row["model_id"] == model and row["both_directions_complete"]]
        passed = [row for row in eligible if row["early_erasure_both_directions"]]
        model_gate[model] = {
            "qualified_findings": len(eligible),
            "early_erasure_findings": len(passed),
            "strict_majority": bool(eligible) and len(passed) > len(eligible) / 2,
        }
    return {
        "protocol_version": VERSION,
        "effect_and_equivalence_margin_auroc": margin,
        "records": classified,
        "finding_direction_gate": finding_rows,
        "model_gate": model_gate,
        "method_branch_authorized": (
            {"huatuo", "hulu"} <= set(model_gate)
            and model_gate["huatuo"]["strict_majority"]
            and model_gate["hulu"]["strict_majority"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text())
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    result = summarize(rows, float(prereg["effect_margin_auroc"]))
    result["provenance"] = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "prereg": str(args.prereg.resolve()),
        "prereg_sha256": sha256_file(args.prereg),
    }
    atomic_write_json(args.output, result)
    print(json.dumps({"method_branch_authorized": result["method_branch_authorized"], "model_gate": result["model_gate"]}, indent=2))


if __name__ == "__main__":
    main()
