#!/usr/bin/env python3
"""Calibration-safe analysis of one source-aligned view plus original fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from corrected_sgta.cache import iter_successes
from corrected_sgta.methods import softmax_np
from corrected_sgta.protocol_v2 import deterministic_split

VERSION = "single-view-alignment-analysis-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interface", choices=("sequence", "surface"), default="sequence")
    parser.add_argument("--seeds", type=int, nargs="+", default=tuple(range(50)))
    parser.add_argument("--betas", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--min-cal-gain", type=float, default=0.005)
    parser.add_argument("--nll-tolerance", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def _style_posteriors(row: dict, interface: str) -> np.ndarray:
    if interface == "sequence":
        scores = -np.asarray(row["style_sequence_nll"], dtype=np.float64)
    else:
        scores = np.asarray(row["style_logits"], dtype=np.float64)
    return np.clip(softmax_np(scores, axis=-1), 1e-12, 1.0)


def fused_posterior(row: dict, style: str | None, beta: float, interface: str) -> np.ndarray:
    probabilities = _style_posteriors(row, interface)
    if beta <= 0 or style is None or style not in row["style_names"]:
        return probabilities[0]
    aligned = probabilities[row["style_names"].index(style)]
    log_q = (1.0 - beta) * np.log(probabilities[0]) + beta * np.log(aligned)
    log_q -= log_q.max()
    q = np.exp(log_q)
    return q / q.sum()


def evaluate(rows: list[dict], style: str | None, beta: float, interface: str) -> dict:
    correct, nll, predictions = [], [], []
    for row in rows:
        q = fused_posterior(row, style, beta, interface)
        gt = int(row["gt_index"])
        pred = int(np.argmax(q))
        correct.append(pred == gt)
        nll.append(-math.log(max(float(q[gt]), 1e-12)))
        predictions.append(pred)
    return {
        "n": len(rows),
        "accuracy": float(np.mean(correct)),
        "nll": float(np.mean(nll)),
        "correct": correct,
        "predictions": predictions,
    }


def paired_statistics(
    baseline: dict, method: dict, bootstrap_samples: int, seed: int = 2027
) -> dict:
    left = np.asarray(baseline["correct"], dtype=np.int8)
    right = np.asarray(method["correct"], dtype=np.int8)
    difference = 100.0 * (right - left)
    rescues = int(np.sum((left == 0) & (right == 1)))
    harmful = int(np.sum((left == 1) & (right == 0)))
    discordant = rescues + harmful
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(rescues, harmful) + 1))
        mcnemar_p = min(1.0, 2.0 * tail / (2**discordant))
    else:
        mcnemar_p = 1.0
    rng = np.random.default_rng(seed)
    n = len(difference)
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        samples[index] = difference[rng.integers(0, n, n)].mean()
    return {
        "delta_pp": float(difference.mean()),
        "bootstrap_95ci_pp": [float(x) for x in np.quantile(samples, (0.025, 0.975))],
        "rescues": rescues,
        "harmful": harmful,
        "mcnemar_exact_p": float(mcnemar_p),
    }


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    metadata = json.loads(
        args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text()
    )
    rows = list(iter_successes(args.cache, metadata["fingerprint"]))
    by_qid = {str(row["qid"]): row for row in rows}
    qids = sorted(by_qid)
    styles = sorted(
        {
            name
            for row in rows
            for name in row["style_names"]
            if name != "original" and not name.startswith("gamma_")
        }
    )
    betas = sorted(set(float(value) for value in args.betas))
    if not styles:
        raise RuntimeError("cache contains no source-aligned styles")
    if not betas or betas[0] != 0.0 or any(beta < 0 or beta > 1 for beta in betas):
        raise ValueError("betas must include 0 and lie in [0, 1]")

    baseline_full = evaluate(rows, None, 0.0, args.interface)
    fixed_reports = {}
    for style in styles:
        method = evaluate(rows, style, 1.0, args.interface)
        fixed_reports[style] = {
            "availability": int(sum(style in row["style_names"] for row in rows)),
            "baseline_accuracy": baseline_full["accuracy"],
            "accuracy": method["accuracy"],
            "nll": method["nll"],
            **paired_statistics(
                baseline_full, method, args.bootstrap_samples, seed=2027
            ),
        }

    split_reports = []
    for seed in args.seeds:
        calibration_qids, test_qids = deterministic_split(
            qids, args.calibration_fraction, int(seed)
        )
        calibration = [by_qid[qid] for qid in calibration_qids]
        test = [by_qid[qid] for qid in test_qids]
        baseline_cal = evaluate(calibration, None, 0.0, args.interface)
        candidates = []
        for style in styles:
            for beta in betas:
                metric = evaluate(calibration, style, beta, args.interface)
                if beta == 0 or (
                    metric["accuracy"]
                    >= baseline_cal["accuracy"] + args.min_cal_gain
                    and metric["nll"] <= baseline_cal["nll"] + args.nll_tolerance
                ):
                    candidates.append((style, beta, metric))
        style, beta, calibration_metric = sorted(
            candidates,
            key=lambda item: (
                -item[2]["accuracy"],
                item[2]["nll"],
                item[1],
                item[0],
            ),
        )[0]
        baseline_test = evaluate(test, None, 0.0, args.interface)
        method_test = evaluate(test, style, beta, args.interface)
        split_reports.append(
            {
                "seed": int(seed),
                "selected_style": style if beta > 0 else "original",
                "selected_beta": beta,
                "calibration_accuracy": calibration_metric["accuracy"],
                "baseline_test_accuracy": baseline_test["accuracy"],
                "test_accuracy": method_test["accuracy"],
                **paired_statistics(
                    baseline_test,
                    method_test,
                    args.bootstrap_samples,
                    seed=2027 + int(seed),
                ),
                "split_fingerprint": stable_hash(
                    {
                        "seed": seed,
                        "calibration": calibration_qids,
                        "test": test_qids,
                    }
                ),
            }
        )

    deltas = np.asarray([row["delta_pp"] for row in split_reports])
    report = {
        "version": VERSION,
        "source_cache": str(args.cache),
        "source_fingerprint": metadata["fingerprint"],
        "settings": {
            "interface": args.interface,
            "betas": betas,
            "calibration_fraction": args.calibration_fraction,
            "min_cal_gain": args.min_cal_gain,
            "nll_tolerance": args.nll_tolerance,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "n": len(rows),
        "baseline_accuracy": baseline_full["accuracy"],
        "fixed_reports": fixed_reports,
        "split_reports": split_reports,
        "aggregate": {
            "mean_delta_pp": float(deltas.mean()),
            "median_delta_pp": float(np.median(deltas)),
            "min_delta_pp": float(deltas.min()),
            "max_delta_pp": float(deltas.max()),
            "positive_splits": int(np.sum(deltas > 0)),
            "negative_splits": int(np.sum(deltas < 0)),
            "fallback_splits": int(
                sum(row["selected_beta"] == 0 for row in split_reports)
            ),
            "selected_styles": dict(
                Counter(row["selected_style"] for row in split_reports)
            ),
            "selected_betas": {
                str(key): value
                for key, value in Counter(
                    row["selected_beta"] for row in split_reports
                ).items()
            },
        },
        "method_note": (
            "Each rejected or unavailable aligned view deterministically falls back to "
            "the original posterior. Hyperparameters are selected on calibration labels "
            "only; fixed_reports are diagnostics and must not be used to tune on test."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
