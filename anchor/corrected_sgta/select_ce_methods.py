#!/usr/bin/env python3
"""Calibration-safe selector over CE label-free methods.

This is an offline analysis layer over an existing CE cache.  It does not rerun
models and it keeps the held-out test split evaluation-only.  Binary questions
may select transductive LAME/LATA; multichoice questions are restricted to
methods with a native fixed class space for that row.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from corrected_sgta.analyze_ce import apply_in_windows, sgta_probabilities
from corrected_sgta.cache import decode_array, iter_successes
from corrected_sgta.methods import entropy_weighted_fusion, softmax_np
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION, deterministic_split

METHOD_VERSION = "ce-method-selector-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sgta-gamma", type=float, default=0.35)
    parser.add_argument("--sgta-iterations", type=int, default=8)
    parser.add_argument("--transductive-window", type=int, default=256)
    parser.add_argument("--lata-gamma", type=float, default=1.0)
    parser.add_argument("--lata-knn", type=int, default=5)
    parser.add_argument("--min-accuracy-gain", type=float, default=0.015)
    parser.add_argument("--nll-tolerance", type=float, default=0.0)
    return parser.parse_args()


def load_records(cache: Path) -> tuple[dict, list[dict]]:
    metadata = json.loads(cache.with_suffix(cache.suffix + ".meta.json").read_text())
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported cache protocol")
    fingerprint = metadata["fingerprint"]
    records = list(iter_successes(cache, fingerprint))
    for row in records:
        row["qid"] = str(row["qid"])
        row["style_logits"] = np.asarray(row["style_logits"], dtype=np.float64)
        row["features"] = decode_array(row["style_features"]).astype(np.float32)
    return metadata, records


def build_probabilities(records: list[dict], args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    probabilities: dict[str, dict[str, np.ndarray]] = {
        "baseline": {},
        "feddg_center": {},
        "domain_center_mean": {},
        "tta_entropy": {},
        "sgta": {},
    }
    for row in records:
        qid = row["qid"]
        logits = row["style_logits"]
        feddg_indices = [
            index
            for index, style in enumerate(row["style_names"])
            if style == "feddg_center" or style.startswith("feddg_")
        ]
        inferred_feddg = logits[feddg_indices[0]] if feddg_indices else logits[0]
        domain_center_mean = logits[feddg_indices].mean(0) if feddg_indices else logits[0]
        method_logits = {
            "baseline": logits[0],
            "feddg_center": inferred_feddg,
            "domain_center_mean": domain_center_mean,
            "tta_entropy": entropy_weighted_fusion(logits),
        }
        for method, value in method_logits.items():
            probabilities[method][qid] = softmax_np(value)
        probabilities["sgta"][qid] = sgta_probabilities(
            logits, row["features"], args.sgta_gamma, args.sgta_iterations
        )
    probabilities["lame"] = apply_in_windows(records, args.transductive_window, "lame", args)
    probabilities["lata"] = apply_in_windows(records, args.transductive_window, "lata", args)
    return probabilities


def evaluate(method_probs: dict[str, np.ndarray], rows: list[dict]) -> dict:
    usable = [row for row in rows if row["qid"] in method_probs]
    correct, nll = [], []
    for row in usable:
        prob = np.asarray(method_probs[row["qid"]], dtype=np.float64)
        gt = int(row["gt_index"])
        correct.append(int(np.argmax(prob)) == gt)
        nll.append(-math.log(max(float(prob[gt]), 1e-12)))
    return {
        "n": len(usable),
        "accuracy": float(np.mean(correct)) if usable else None,
        "nll": float(np.mean(nll)) if usable else None,
    }


def choose_for_kind(
    kind: str,
    calibration: list[dict],
    probabilities: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> dict:
    if kind == "binary":
        candidates = ["baseline", "feddg_center", "domain_center_mean", "tta_entropy", "sgta", "lame", "lata"]
    else:
        candidates = ["baseline", "feddg_center", "domain_center_mean", "tta_entropy", "sgta"]
    rows = [row for row in calibration if row["question_type"] == kind]
    scores = {method: evaluate(probabilities[method], rows) for method in candidates}
    baseline = scores["baseline"]
    if not baseline["n"]:
        return {"kind": kind, "selected": "baseline", "scores": scores, "reason": "empty"}
    margin = max(float(args.min_accuracy_gain), 2.0 / max(1, int(baseline["n"])))
    selectable = []
    for method, score in scores.items():
        if score["accuracy"] is None:
            continue
        if method == "baseline" or (
            score["accuracy"] >= baseline["accuracy"] + margin
            and score["nll"] <= baseline["nll"] + float(args.nll_tolerance)
        ):
            selectable.append((method, score))
    selected, selected_score = sorted(
        selectable,
        key=lambda item: (-item[1]["accuracy"], item[1]["nll"], 0 if item[0] == "baseline" else 1, item[0]),
    )[0]
    return {
        "kind": kind,
        "selected": selected,
        "selected_score": selected_score,
        "baseline_score": baseline,
        "required_accuracy_margin": margin,
        "scores": scores,
        "selectable": [method for method, _ in selectable],
    }


def evaluate_selector(
    selections: dict[str, dict],
    test: list[dict],
    probabilities: dict[str, dict[str, np.ndarray]],
) -> dict:
    correct, nll, used = [], [], {}
    for row in test:
        method = selections[row["question_type"]]["selected"]
        if row["qid"] not in probabilities[method]:
            method = "baseline"
        prob = probabilities[method][row["qid"]]
        gt = int(row["gt_index"])
        correct.append(int(np.argmax(prob)) == gt)
        nll.append(-math.log(max(float(prob[gt]), 1e-12)))
        used[method] = used.get(method, 0) + 1
    return {
        "n": len(correct),
        "accuracy": float(np.mean(correct)) if correct else None,
        "nll": float(np.mean(nll)) if nll else None,
        "used_methods": used,
    }


def main() -> None:
    args = parse_args()
    metadata, records = load_records(args.cache)
    probabilities = build_probabilities(records, args)
    qids = [row["qid"] for row in records]
    calibration_qids, test_qids = deterministic_split(qids, args.calibration_fraction, args.seed)
    by_qid = {row["qid"]: row for row in records}
    calibration = [by_qid[qid] for qid in calibration_qids]
    test = [by_qid[qid] for qid in test_qids]
    selections = {
        kind: choose_for_kind(kind, calibration, probabilities, args)
        for kind in ("binary", "multichoice")
    }
    baseline = evaluate(probabilities["baseline"], test)
    selected = evaluate_selector(selections, test, probabilities)
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "method_version": METHOD_VERSION,
        "source_cache": str(args.cache),
        "fingerprint": metadata["fingerprint"],
        "split": {
            "seed": args.seed,
            "calibration_fraction": args.calibration_fraction,
            "n_calibration": len(calibration),
            "n_test": len(test),
        },
        "selections": selections,
        "test_results": {
            "baseline_original": baseline,
            "calibrated_method_selector": selected,
        },
        "test_delta_vs_baseline": None if baseline["accuracy"] is None or selected["accuracy"] is None else selected["accuracy"] - baseline["accuracy"],
        "scope": "Methods are selected on the calibration split only; test labels are evaluation-only. LAME/LATA are eligible only for binary Yes/No rows.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
