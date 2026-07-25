#!/usr/bin/env python3
"""Calibration-safe, one-parameter source-view barycenter.

The fused posterior is the reverse-KL barycenter

    q_beta ∝ p_original ** (1-beta) * prod(p_view ** (beta / K)).

Thus beta=0 is an exact, clinically conservative fallback, while beta>0 moves
only along evidence shared by the source-guided views.  No test label is used
to select beta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from corrected_sgta.cache import iter_successes
from corrected_sgta.methods import softmax_np
from corrected_sgta.protocol_v2 import deterministic_split

VERSION = "anchor-source-barycenter-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4, 42))
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=(0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--min-cal-gain", type=float, default=0.005)
    parser.add_argument("--nll-tolerance", type=float, default=0.02)
    return parser.parse_args()


def barycenter(logits: np.ndarray, beta: float) -> np.ndarray:
    probabilities = np.clip(softmax_np(logits), 1e-12, 1.0)
    if len(probabilities) == 1 or beta <= 0:
        return probabilities[0]
    log_q = (1.0 - beta) * np.log(probabilities[0])
    log_q += beta * np.log(probabilities[1:]).mean(axis=0)
    log_q -= log_q.max()
    q = np.exp(log_q)
    return q / q.sum()


def evaluate(rows: list[dict], beta: float) -> dict:
    correct, nll, predictions = [], [], []
    for row in rows:
        q = barycenter(np.asarray(row["style_logits"], dtype=np.float64), beta)
        gt = int(row["gt_index"])
        pred = int(np.argmax(q))
        predictions.append(pred)
        correct.append(pred == gt)
        nll.append(-math.log(max(float(q[gt]), 1e-12)))
    return {
        "n": len(rows),
        "accuracy": float(np.mean(correct)) if rows else None,
        "nll": float(np.mean(nll)) if rows else None,
        "predictions": predictions,
    }


def flip_summary(base: list[int], other: list[int], rows: list[dict]) -> dict:
    gt = np.asarray([int(row["gt_index"]) for row in rows])
    base_array, other_array = np.asarray(base), np.asarray(other)
    return {
        "changed": int(np.sum(base_array != other_array)),
        "rescues": int(np.sum((base_array != gt) & (other_array == gt))),
        "harmful": int(np.sum((base_array == gt) & (other_array != gt))),
    }


def mean_js(row: dict) -> float:
    p = np.clip(softmax_np(np.asarray(row["style_logits"], dtype=np.float64)), 1e-12, 1.0)
    m = p.mean(axis=0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(m)), axis=1)))


def stable_hash(values: object) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    metadata_path = args.cache.with_suffix(args.cache.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    rows = list(iter_successes(args.cache, metadata["fingerprint"]))
    if not rows:
        raise RuntimeError("no successful cache rows")
    by_qid = {str(row["qid"]): row for row in rows}
    qids = sorted(by_qid)
    betas = sorted(set(float(value) for value in args.betas))
    if not betas or betas[0] != 0.0 or any(value < 0 or value > 1 for value in betas):
        raise ValueError("betas must include 0 and lie in [0, 1]")

    split_reports = []
    for seed in args.seeds:
        cal_qids, test_qids = deterministic_split(
            qids, args.calibration_fraction, int(seed)
        )
        calibration = [by_qid[qid] for qid in cal_qids]
        test = [by_qid[qid] for qid in test_qids]
        calibration_grid = {str(beta): evaluate(calibration, beta) for beta in betas}
        baseline_cal = calibration_grid["0.0"]
        feasible = []
        for beta in betas:
            metric = calibration_grid[str(beta)]
            if beta == 0 or (
                metric["accuracy"] >= baseline_cal["accuracy"] + args.min_cal_gain
                and metric["nll"] <= baseline_cal["nll"] + args.nll_tolerance
            ):
                feasible.append(beta)
        selected_beta = sorted(
            feasible,
            key=lambda beta: (
                -calibration_grid[str(beta)]["accuracy"],
                calibration_grid[str(beta)]["nll"],
                beta,
            ),
        )[0]
        baseline_test = evaluate(test, 0.0)
        selected_test = evaluate(test, selected_beta)
        split_reports.append(
            {
                "seed": int(seed),
                "n_calibration": len(calibration),
                "n_test": len(test),
                "selected_beta": selected_beta,
                "baseline_accuracy": baseline_test["accuracy"],
                "selected_accuracy": selected_test["accuracy"],
                "delta_pp": 100.0
                * (selected_test["accuracy"] - baseline_test["accuracy"]),
                "flips": flip_summary(
                    baseline_test["predictions"], selected_test["predictions"], test
                ),
                "calibration_grid": {
                    key: {k: v for k, v in value.items() if k != "predictions"}
                    for key, value in calibration_grid.items()
                },
                "split_fingerprint": stable_hash(
                    {"seed": seed, "calibration": cal_qids, "test": test_qids}
                ),
            }
        )

    original = evaluate(rows, 0.0)
    view_predictions = [
        np.argmax(np.asarray(row["style_logits"]), axis=1) for row in rows
    ]
    oracle_accuracy = float(
        np.mean(
            [
                int(row["gt_index"]) in predictions
                for row, predictions in zip(rows, view_predictions)
            ]
        )
    )
    deltas = np.asarray([row["delta_pp"] for row in split_reports])
    report = {
        "version": VERSION,
        "source_cache": str(args.cache),
        "source_fingerprint": metadata["fingerprint"],
        "settings": {
            "betas": betas,
            "calibration_fraction": args.calibration_fraction,
            "min_cal_gain": args.min_cal_gain,
            "nll_tolerance": args.nll_tolerance,
        },
        "n": len(rows),
        "diagnostics": {
            "original_accuracy": original["accuracy"],
            "style_oracle_accuracy": oracle_accuracy,
            "style_oracle_headroom_pp": 100.0
            * (oracle_accuracy - original["accuracy"]),
            "mean_view_js": float(np.mean([mean_js(row) for row in rows])),
            "prediction_disagreement_rate": float(
                np.mean([len(set(map(int, preds))) > 1 for preds in view_predictions])
            ),
        },
        "split_reports": split_reports,
        "aggregate": {
            "mean_delta_pp": float(deltas.mean()),
            "median_delta_pp": float(np.median(deltas)),
            "min_delta_pp": float(deltas.min()),
            "max_delta_pp": float(deltas.max()),
            "positive_splits": int(np.sum(deltas > 0)),
            "negative_splits": int(np.sum(deltas < 0)),
            "fallback_splits": int(
                np.sum([row["selected_beta"] == 0 for row in split_reports])
            ),
        },
        "method_note": (
            "A single calibrated beta controls the information-preservation/domain-alignment "
            "trade-off. Large cross-view disagreement is reported as sensitivity evidence, "
            "not assumed to be beneficial."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("diagnostics", "aggregate")}, indent=2))


if __name__ == "__main__":
    main()
