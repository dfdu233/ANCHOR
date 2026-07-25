#!/usr/bin/env python3
"""Evaluate a simple source-center majority vote with calibration-only filtering."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from corrected_sgta.cache import iter_successes
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION, deterministic_split


METHOD_VERSION = "source-center-consensus-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows-output", type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seeds", type=int, default=50)
    return parser.parse_args()


def load_records(cache: Path) -> tuple[dict, list[dict]]:
    metadata = json.loads(cache.with_suffix(cache.suffix + ".meta.json").read_text())
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported cache protocol")
    records = list(iter_successes(cache, metadata["fingerprint"]))
    if not records:
        raise RuntimeError("cache contains no successful records")
    return metadata, records


def add_consensus_fields(row: dict) -> dict:
    logits = np.asarray(row["style_logits"], dtype=np.float64)
    predictions = np.argmax(logits, axis=1)
    center_indices = [
        index
        for index, name in enumerate(row["style_names"])
        if name.startswith("feddg_")
    ]
    if not center_indices:
        raise RuntimeError(f"qid={row['qid']} has no source-center views")
    votes = Counter(int(predictions[index]) for index in center_indices)
    top_count = max(votes.values())
    winners = [label for label, count in votes.items() if count == top_count]
    baseline_prediction = int(predictions[0])
    consensus_prediction = winners[0] if len(winners) == 1 else baseline_prediction
    return {
        "qid": str(row["qid"]),
        "gt_index": int(row["gt_index"]),
        "baseline_prediction": baseline_prediction,
        "consensus_prediction": int(consensus_prediction),
        "support": int(top_count),
        "n_center_views": len(center_indices),
    }


def prediction(record: dict, min_support: int) -> int:
    if record["support"] < min_support:
        return record["baseline_prediction"]
    return record["consensus_prediction"]


def evaluate(records: list[dict], min_support: int) -> dict:
    baseline_correct = np.asarray(
        [row["baseline_prediction"] == row["gt_index"] for row in records],
        dtype=bool,
    )
    candidate_correct = np.asarray(
        [prediction(row, min_support) == row["gt_index"] for row in records],
        dtype=bool,
    )
    changed = np.asarray(
        [prediction(row, min_support) != row["baseline_prediction"] for row in records],
        dtype=bool,
    )
    rescue = np.logical_and(~baseline_correct, candidate_correct)
    harm = np.logical_and(baseline_correct, ~candidate_correct)
    n = len(records)
    return {
        "n": n,
        "min_support": min_support,
        "baseline_accuracy": float(baseline_correct.mean()),
        "candidate_accuracy": float(candidate_correct.mean()),
        "delta": float(candidate_correct.mean() - baseline_correct.mean()),
        "changed_count": int(changed.sum()),
        "rescue_count": int(rescue.sum()),
        "harmful_count": int(harm.sum()),
        "changed_rate": float(changed.mean()),
        "rescue_rate": float(rescue.mean()),
        "harmful_rate": float(harm.mean()),
    }


def select_support(calibration: list[dict]) -> tuple[int, dict[int, dict]]:
    n_views = calibration[0]["n_center_views"]
    minimum = n_views // 2 + 1
    scores = {support: evaluate(calibration, support) for support in range(minimum, n_views + 1)}
    selected = sorted(
        scores,
        key=lambda support: (
            -scores[support]["candidate_accuracy"],
            scores[support]["harmful_count"],
            -support,
        ),
    )[0]
    if scores[selected]["delta"] <= 0.0:
        selected = n_views + 1
    return selected, scores


def run_split(records: list[dict], seed: int, fraction: float) -> dict:
    calibration_qids, test_qids = deterministic_split(
        [row["qid"] for row in records], fraction, seed
    )
    by_qid = {row["qid"]: row for row in records}
    calibration = [by_qid[qid] for qid in calibration_qids]
    test = [by_qid[qid] for qid in test_qids]
    selected_support, calibration_grid = select_support(calibration)
    strict_majority = records[0]["n_center_views"] // 2 + 1
    return {
        "seed": seed,
        "calibration_qids": calibration_qids,
        "test_qids": test_qids,
        "selected_min_support": selected_support,
        "calibration_grid": calibration_grid,
        "calibration_selected": evaluate(calibration, selected_support),
        "test_selected": evaluate(test, selected_support),
        "test_strict_majority": evaluate(test, strict_majority),
    }


def summarize_splits(splits: list[dict]) -> dict:
    deltas = np.asarray([split["test_selected"]["delta"] for split in splits])
    return {
        "n_seeds": len(splits),
        "mean_test_delta": float(deltas.mean()),
        "std_test_delta": float(deltas.std()),
        "min_test_delta": float(deltas.min()),
        "max_test_delta": float(deltas.max()),
        "negative_seed_fraction": float((deltas < 0).mean()),
        "positive_seed_fraction": float((deltas > 0).mean()),
        "selected_support_counts": dict(
            sorted(Counter(split["selected_min_support"] for split in splits).items())
        ),
    }


def write_rows(path: Path, records: list[dict], split: dict) -> None:
    calibration_qids = set(split["calibration_qids"])
    selected_support = split["selected_min_support"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "qid",
        "split",
        "gt_index",
        "baseline_prediction",
        "consensus_prediction",
        "support",
        "n_center_views",
        "selected_min_support",
        "selected_prediction",
        "changed",
        "rescue",
        "harmful",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in records:
            selected_prediction = prediction(row, selected_support)
            baseline_correct = row["baseline_prediction"] == row["gt_index"]
            selected_correct = selected_prediction == row["gt_index"]
            writer.writerow(
                {
                    **row,
                    "split": "calibration" if row["qid"] in calibration_qids else "test",
                    "selected_min_support": selected_support,
                    "selected_prediction": selected_prediction,
                    "changed": int(selected_prediction != row["baseline_prediction"]),
                    "rescue": int(not baseline_correct and selected_correct),
                    "harmful": int(baseline_correct and not selected_correct),
                }
            )


def main() -> None:
    args = parse_args()
    metadata, raw_records = load_records(args.cache)
    records = [add_consensus_fields(row) for row in raw_records]
    primary = run_split(records, args.seed, args.calibration_fraction)
    splits = [
        run_split(records, seed, args.calibration_fraction)
        for seed in range(args.seed, args.seed + args.split_seeds)
    ]
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "method_version": METHOD_VERSION,
        "source_cache": str(args.cache),
        "fingerprint": metadata["fingerprint"],
        "method": (
            "Use the strict majority among source-center views; ties retain the "
            "original prediction. Select only the minimum vote support on the "
            "calibration split, falling back to the original if calibration gain is non-positive."
        ),
        "n_records": len(records),
        "n_center_views": records[0]["n_center_views"],
        "split": {
            "seed": args.seed,
            "calibration_fraction": args.calibration_fraction,
            "n_calibration": len(primary["calibration_qids"]),
            "n_test": len(primary["test_qids"]),
        },
        "primary": {key: value for key, value in primary.items() if not key.endswith("_qids")},
        "multi_seed_summary": summarize_splits(splits),
        "multi_seed_results": [
            {
                "seed": split["seed"],
                "selected_min_support": split["selected_min_support"],
                "calibration_selected": split["calibration_selected"],
                "test_selected": split["test_selected"],
            }
            for split in splits
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    if args.rows_output:
        write_rows(args.rows_output, records, primary)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
