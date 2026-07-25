"""Analyze the QLS-TR n=16/32 identifiability gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def exact_mcnemar_two_sided(rescues: int, harmful: int) -> float:
    discordant = rescues + harmful
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(rescues, harmful) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def summarize(rows: list[dict], prediction_key: str) -> dict:
    original = np.asarray([row["original_prediction"] for row in rows])
    candidate = np.asarray([row[prediction_key] for row in rows])
    truth = np.asarray([row["gt_index"] for row in rows])
    original_correct = original == truth
    candidate_correct = candidate == truth
    rescues = int(np.sum(~original_correct & candidate_correct))
    harmful = int(np.sum(original_correct & ~candidate_correct))
    return {
        "n": len(rows),
        "original_accuracy": float(np.mean(original_correct)),
        "candidate_accuracy": float(np.mean(candidate_correct)),
        "delta_pp": float(100 * (np.mean(candidate_correct) - np.mean(original_correct))),
        "changed": int(np.sum(original != candidate)),
        "rescues": rescues,
        "harmful": harmful,
        "net": rescues - harmful,
        "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided(rescues, harmful),
    }


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("status") == "ok"]
    matched = summarize(rows, "matched_prediction")
    wrong = summarize(rows, "wrong_prediction")
    null = summarize(rows, "null_unconditioned_prediction")
    oracle_correct = []
    density_gains = []
    selected_density_gains = []
    true_margin_gains = []
    for row in rows:
        truth = int(row["gt_index"])
        candidates = np.asarray(row["matched_candidate_nll"], dtype=float)
        predictions = np.argmin(candidates, axis=1)
        oracle_correct.append(
            row["original_prediction"] == truth or bool(np.any(predictions == truth))
        )
        gains = row["matched_geometry"]["candidate_density_gain"]
        density_gains.extend(gains)
        selected = row["matched_selected_candidate"]
        if selected is not None:
            selected_density_gains.append(gains[selected])
            original_margin = -row["original_nll"][truth] + row["original_nll"][1 - truth]
            chosen = row["matched_nll"]
            chosen_margin = -chosen[truth] + chosen[1 - truth]
            true_margin_gains.append(chosen_margin - original_margin)
    baseline_accuracy = matched["original_accuracy"]
    oracle_accuracy = float(np.mean(oracle_correct)) if rows else float("nan")
    result = {
        "input": str(args.input.resolve()),
        "n": len(rows),
        "matched": matched,
        "wrong": wrong,
        "null_unconditioned": null,
        "style_oracle_accuracy": oracle_accuracy,
        "style_oracle_headroom_pp": 100 * (oracle_accuracy - baseline_accuracy),
        "mean_candidate_density_gain": float(np.mean(density_gains)),
        "mean_selected_density_gain": float(np.mean(selected_density_gains)) if selected_density_gains else None,
        "mean_selected_true_margin_gain": float(np.mean(true_margin_gains)) if true_margin_gains else None,
        "gate_n16": {
            "nonzero_flips": matched["changed"] > 0,
            "harms_not_greater_than_rescues": matched["harmful"] <= matched["rescues"],
            "matched_beats_wrong": matched["delta_pp"] > wrong["delta_pp"],
            "matched_beats_null": matched["delta_pp"] > null["delta_pp"],
        },
        "claim_gate": {
            "delta_strictly_gt_3pp": matched["delta_pp"] > 3.0,
            "paired_significant": matched["mcnemar_exact_two_sided_p"] < 0.05,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
