#!/usr/bin/env python3
"""Synthesize the two-model VinDr reader-unanimity confirmation boundary.

The summary is deliberately fail-closed: an observational Early-erasure result
requests a causal follow-up, but never authorizes a decoder or mitigation claim.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file


VERSION = "vindr-reader-residual-two-model-summary-v1"
DIRECTIONS = ("negative_0v1", "positive_2v3")
INTERPRETABLE = {
    "Early erasure",
    "Late emergence",
    "Layer-stable",
    "Not decodable",
}


def load_confirmation(path: Path) -> dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("status") != "complete":
        raise ValueError(f"non-complete confirmation: {path}")
    if set(row.get("results", {})) != set(DIRECTIONS):
        raise ValueError(f"confirmation lacks the two preregistered directions: {path}")
    return row


def synthesize(
    inputs: list[tuple[Path, dict[str, Any]]], min_finding_n: int = 100
) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("the mechanism-boundary paper gate requires at least two models")
    model_ids = [str(row.get("model_id")) for _, row in inputs]
    if len(set(model_ids)) != len(model_ids) or "None" in model_ids:
        raise ValueError("model identifiers must be present and unique")

    pooled = []
    finding_cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path, row in inputs:
        model = str(row["model_id"])
        for direction in DIRECTIONS:
            result = row["results"][direction]
            boundary = str(result.get("boundary"))
            pooled.append(
                {
                    "model_id": model,
                    "direction": direction,
                    "boundary": boundary,
                    "n": int(result["n"]),
                    "representation_controls_pass": bool(
                        result.get("representation_controls_pass", False)
                    ),
                }
            )
            for finding, value in result.get("finding_wise", {}).items():
                finding_cells.setdefault((direction, finding), []).append(
                    {
                        "model_id": model,
                        "n": int(value["n"]),
                        "boundary": str(value["boundary"]),
                    }
                )

    cell_rows = []
    for (direction, finding), values in sorted(finding_cells.items()):
        qualified = (
            len(values) == len(inputs)
            and all(value["n"] >= min_finding_n for value in values)
        )
        # Indeterminate cells stay in the denominator. Excluding them would
        # allow a few high-signal findings to manufacture an apparent majority.
        consistent = (
            qualified
            and all(value["boundary"] in INTERPRETABLE for value in values)
            and len({value["boundary"] for value in values}) == 1
        )
        cell_rows.append(
            {
                "direction": direction,
                "finding": finding,
                "models": values,
                "qualified": qualified,
                "consistent": consistent,
                "shared_boundary": values[0]["boundary"] if consistent else None,
            }
        )
    qualified = [row for row in cell_rows if row["qualified"]]
    consistent = [row for row in qualified if row["consistent"]]
    directions_represented = {row["direction"] for row in qualified}
    pooled_direction_consistency = []
    for direction in DIRECTIONS:
        values = [row["boundary"] for row in pooled if row["direction"] == direction]
        pooled_direction_consistency.append(
            {
                "direction": direction,
                "model_boundaries": values,
                "consistent": bool(
                    len(values) == len(inputs)
                    and all(value in INTERPRETABLE for value in values)
                    and len(set(values)) == 1
                ),
                "shared_boundary": values[0]
                if len(values) == len(inputs)
                and all(value in INTERPRETABLE for value in values)
                and len(set(values)) == 1
                else None,
            }
        )
    pooled_consistent = all(row["consistent"] for row in pooled_direction_consistency)
    # "Majority" is strict (> 50%), not a rounded half. Requiring at least four
    # qualified direction/finding cells prevents a tiny subset from passing.
    boundary_paper_gate = bool(
        len(qualified) >= 4
        and len(consistent) * 2 > len(qualified)
        and directions_represented == set(DIRECTIONS)
        and pooled_consistent
    )
    observational_early_erasure = all(
        bool(row.get("observational_gate_passed", False)) for _, row in inputs
    )
    counts = Counter(row["boundary"] for row in pooled)
    next_action = (
        "run_preregistered_causal_activation_patch_only"
        if observational_early_erasure
        else "do_not_build_decoder; consolidate_boundary_controls_and_replication"
    )
    return {
        "version": VERSION,
        "status": "complete",
        "models": model_ids,
        "pooled_direction_results": pooled,
        "pooled_boundary_counts": dict(sorted(counts.items())),
        "finding_direction_cells": cell_rows,
        "paper_gate": {
            "min_finding_n": min_finding_n,
            "qualified_cells": len(qualified),
            "consistent_cells": len(consistent),
            "strict_majority_consistent": boundary_paper_gate,
            "directions_represented": sorted(directions_represented),
            "pooled_direction_consistency": pooled_direction_consistency,
            "all_pooled_directions_cross_model_consistent": pooled_consistent,
            "interpretation": (
                "two-model mechanism-boundary evidence eligible for paper synthesis"
                if boundary_paper_gate
                else "insufficient cross-model/finding consistency for the mechanism-boundary claim"
            ),
        },
        "observational_early_erasure_all_models": observational_early_erasure,
        "method_authorized": False,
        "next_action": next_action,
        "method_authorization_reason": (
            "a causal activation patch is still required even when every observational gate passes"
        ),
        "provenance": {
            "inputs": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path, _ in inputs
            ],
            "code_sha256": sha256_file(Path(__file__)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-finding-n", type=int, default=100)
    args = parser.parse_args()
    if args.min_finding_n < 1:
        raise ValueError("min-finding-n must be positive")
    inputs = [(path, load_confirmation(path)) for path in args.input]
    output = synthesize(inputs, min_finding_n=args.min_finding_n)
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
