"""Analyze the decoder-visible source projection gate."""

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


def mcnemar(rescues: int, harms: int) -> float:
    total = rescues + harms
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(rescues, harms) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def summarize(rows: list[dict], prefix: str) -> dict:
    truth = np.asarray([row["gt_index"] for row in rows])
    original = np.asarray([row["original_prediction"] for row in rows])
    candidate = np.asarray([row[f"{prefix}_prediction"] for row in rows])
    original_ok = original == truth
    candidate_ok = candidate == truth
    rescues = int(np.sum(~original_ok & candidate_ok))
    harms = int(np.sum(original_ok & ~candidate_ok))
    density_up = [
        row[prefix]["kde_log_density_after"]
        > row[prefix]["kde_log_density_before"]
        for row in rows
    ]
    return {
        "n": len(rows),
        "original_accuracy": float(np.mean(original_ok)),
        "accuracy": float(np.mean(candidate_ok)),
        "delta_pp": float(
            100
            * (
                np.mean(candidate_ok.astype(np.float64))
                - np.mean(original_ok.astype(np.float64))
            )
        ),
        "changed": int(np.sum(candidate != original)),
        "rescues": rescues,
        "harms": harms,
        "mcnemar_p": mcnemar(rescues, harms),
        "density_ascent_rate": float(np.mean(density_up)),
        "positive_score_step_rate": float(
            np.mean([row[prefix]["source_ascent_inner_product"] >= -1e-8 for row in rows])
        ),
        "median_step_norm": float(
            np.median([row[prefix]["step_norm"] for row in rows])
        ),
    }


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("status") == "ok"]
    matched = summarize(rows, "matched")
    wrong = summarize(rows, "wrong")
    truth = np.asarray([row["gt_index"] for row in rows])
    original = np.asarray([row["original_prediction"] for row in rows])
    matched_pred = np.asarray([row["matched_prediction"] for row in rows])
    oracle = (original == truth) | (matched_pred == truth)
    result = {
        "input": str(args.input.resolve()),
        "n": len(rows),
        "matched": matched,
        "wrong": wrong,
        "matched_oracle_headroom_pp": float(
            100 * (np.mean(oracle) - np.mean(original == truth))
        ),
        "gate_n32": {
            "oracle_headroom_ge_6_25pp": bool(
                100 * (np.mean(oracle) - np.mean(original == truth)) >= 6.25
            ),
            "rescues_ge_harms": matched["rescues"] >= matched["harms"],
            "matched_beats_wrong": matched["delta_pp"] > wrong["delta_pp"],
            "density_ascent_ge_90pct": matched["density_ascent_rate"] >= 0.9,
        },
        "claim_gate": {
            "delta_ge_3pp": matched["delta_pp"] >= 3.0,
            "mcnemar_p_lt_0_05": matched["mcnemar_p"] < 0.05,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
