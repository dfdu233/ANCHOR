#!/usr/bin/env python3
"""Summarize the domain-orbit fatal audit without tuning on its labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_method(records: list[dict], method: str) -> dict:
    original = [row["methods"]["original"] for row in records]
    values = [row["methods"][method] for row in records]
    correct = [value["prediction"] == row["expected_state"] for row, value in zip(records, values)]
    original_correct = [value["prediction"] == row["expected_state"] for row, value in zip(records, original)]
    clear_indices = [i for i, row in enumerate(records) if row["positive_votes"] in (0, 3)]
    clear_aligned_margin_changes = []
    for index in clear_indices:
        sign = 1.0 if records[index]["positive_votes"] == 3 else -1.0
        clear_aligned_margin_changes.append(
            sign * (values[index]["polarity"] - original[index]["polarity"])
        )
    result = {
        "n": len(records),
        "accuracy_3state": mean(correct),
        "clear_0_or_3_accuracy": mean(correct[i] for i in clear_indices),
        "rescues_vs_original": sum((not base) and now for base, now in zip(original_correct, correct)),
        "harms_vs_original": sum(base and (not now) for base, now in zip(original_correct, correct)),
        "prediction_counts": dict(sorted(Counter(value["prediction"] for value in values).items())),
        "mean_polarity": mean(value["polarity"] for value in values),
        "mean_commitment": mean(value["commitment"] for value in values),
        "mean_clear_label_aligned_margin_change_vs_original": mean(clear_aligned_margin_changes),
    }
    for field in ("heldout_attenuation", "degeneration_ratio", "visual_displacement_fro"):
        present = [value[field] for value in values if field in value]
        if present:
            result[f"mean_{field}"] = mean(present)
    return result


def analyze(payload: dict, input_path: Path) -> dict:
    records = payload["records"]
    methods = records[0]["methods"].keys()
    cumulative = [
        mean(row["cumulative_explained_fraction"][index] for row in records)
        for index in range(len(records[0]["cumulative_explained_fraction"]))
    ]
    summaries = {method: summarize_method(records, method) for method in methods}
    primary = {}
    for rank in payload["ranks"]:
        doc = f"doc_r{rank}_a1"
        random = f"random_r{rank}_a1"
        matched_mean = f"mean_interp_r{rank}_a1"
        primary[f"rank_{rank}"] = {
            "doc": summaries[doc],
            "norm_matched_random": summaries[random],
            "displacement_matched_mean": summaries[matched_mean],
            "doc_accuracy_advantage_vs_random": summaries[doc]["accuracy_3state"]
            - summaries[random]["accuracy_3state"],
            "doc_accuracy_advantage_vs_mean": summaries[doc]["accuracy_3state"]
            - summaries[matched_mean]["accuracy_3state"],
            "heldout_attenuation_advantage_vs_random": summaries[doc]["mean_heldout_attenuation"]
            - summaries[random]["mean_heldout_attenuation"],
        }
    any_doc_rescue = any(summaries[f"doc_r{rank}_a1"]["rescues_vs_original"] > 0 for rank in payload["ranks"])
    any_doc_advantage = any(
        primary[f"rank_{rank}"]["doc_accuracy_advantage_vs_random"] > 0
        and primary[f"rank_{rank}"]["doc_accuracy_advantage_vs_mean"] > 0
        for rank in payload["ranks"]
    )
    return {
        "version": "domain-orbit-diagnostic-analysis-v1",
        "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "scientific_role": "fatal feasibility audit; estimates are descriptive at this pilot size",
        "n": len(records),
        "label_counts": dict(sorted(Counter(row["expected_state"] for row in records).items())),
        "mean_cumulative_orbit_variance_explained": cumulative,
        "all_methods": summaries,
        "primary_alpha_1_comparisons": primary,
        "gates": {
            "low_dimensional_orbit": cumulative[1] >= 0.80,
            "heldout_style_tangent_generalizes": all(
                summaries[f"doc_r{rank}_a1"]["mean_heldout_attenuation"] >= 0.50
                for rank in payload["ranks"]
            ),
            "any_doc_label_rescue": any_doc_rescue,
            "doc_beats_both_equal_displacement_controls": any_doc_advantage,
        },
        "decision": (
            "continue_mechanism_only_not_mitigation"
            if any_doc_rescue and any_doc_advantage
            else "stop_scaling_render_orbit_doc_as_hallucination_mitigation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    result = analyze(payload, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
