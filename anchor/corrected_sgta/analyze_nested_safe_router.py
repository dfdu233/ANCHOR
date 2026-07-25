"""Nested safe router for competence-native SGTA.

Protocol:
  - use the first half of calibration rows to choose a risk score and route
    budget;
  - use the second half of calibration rows only for safety acceptance;
  - if validation does not improve/non-degrade greedy without calibration
    harmful flips, fall back to greedy;
  - on target rows, route the same top-risk fraction within the unlabeled batch.

This keeps the method deliberately small: competence-native support estimation
plus one conformal-style rank budget.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-route-frac", type=float, default=0.5)
    parser.add_argument("--min-val-gain", type=float, default=0.0)
    parser.add_argument("--allow-val-harmful", type=int, default=0)
    parser.add_argument("--selection-objective", choices=["accuracy", "rescue_margin"], default="accuracy")
    return parser.parse_args()


def acc(values):
    return float(np.mean(values)) if values else None


def routed_qids_by_fraction(rows, score_key: str, frac: float) -> set[str]:
    if not rows or frac <= 0:
        return set()
    k = int(math.ceil(frac * len(rows)))
    k = min(max(k, 0), len(rows))
    ranked = sorted(
        rows,
        key=lambda r: (float(r["risk_scores"][score_key]), str(r["qid"])),
        reverse=True,
    )
    return {str(r["qid"]) for r in ranked[:k]}


def evaluate(rows, score_key: str | None, frac: float):
    if score_key is None or frac <= 0:
        routed = set()
    else:
        routed = routed_qids_by_fraction(rows, score_key, frac)
    correct = [
        bool(r["mitigation_correct"]) if str(r["qid"]) in routed else bool(r["greedy_correct"])
        for r in rows
    ]
    return {
        "n": len(rows),
        "route_frac": float(frac),
        "routed": len(routed),
        "coverage_mitigation": float(len(routed) / len(rows)) if rows else None,
        "accuracy": acc(correct),
        "greedy_accuracy": acc([bool(r["greedy_correct"]) for r in rows]),
        "mitigation_accuracy": acc([bool(r["mitigation_correct"]) for r in rows]),
        "rescues_kept": sum(bool(r["rescue"]) and str(r["qid"]) in routed for r in rows),
        "harmful_introduced": sum(bool(r["harmful"]) and str(r["qid"]) in routed for r in rows),
        "rescue_margin": sum(bool(r["rescue"]) and str(r["qid"]) in routed for r in rows)
        - sum(bool(r["harmful"]) and str(r["qid"]) in routed for r in rows),
        "routed_qids": sorted(routed),
    }


def candidate_objective(metrics, mode: str):
    if mode == "rescue_margin":
        return (
            int(metrics["rescue_margin"]),
            float(metrics["accuracy"] or 0.0),
            -int(metrics["routed"]),
        )
    return (
        float(metrics["accuracy"] or 0.0),
        int(metrics["rescue_margin"]),
        -int(metrics["routed"]),
    )


def main():
    args = parse_args()
    source = json.loads(args.analysis.read_text())
    rows = source["rows"]
    calibration = [r for r in rows if r.get("split") == "train"]
    target = [r for r in rows if r.get("split") == "test"]
    cut = len(calibration) // 2
    router_train = calibration[:cut]
    safety_cal = calibration[cut:]
    score_keys = sorted(rows[0]["risk_scores"]) if rows else []

    best = None
    per_key = {}
    for key in score_keys:
        valid_train = [r for r in router_train if r["risk_scores"].get(key) is not None]
        valid_cal = [r for r in safety_cal if r["risk_scores"].get(key) is not None]
        if not valid_train or not valid_cal:
            continue
        max_k = int(math.floor(args.max_route_frac * len(valid_train)))
        fracs = sorted({0.0} | {k / len(valid_train) for k in range(1, max_k + 1)})
        key_best = None
        for frac in fracs:
            train_metrics = evaluate(valid_train, key, frac)
            train_obj = candidate_objective(train_metrics, args.selection_objective)
            candidate = {
                "score_key": key,
                "route_frac": float(frac),
                "train_objective": list(train_obj),
                "router_train": train_metrics,
                "safety_cal": evaluate(valid_cal, key, frac),
            }
            if key_best is None or train_obj > tuple(key_best["train_objective"]):
                key_best = candidate
        if key_best is None:
            continue
        val = key_best["safety_cal"]
        val_gain = float(val["accuracy"] or 0.0) - float(val["greedy_accuracy"] or 0.0)
        key_best["accepted"] = (
            val["harmful_introduced"] <= args.allow_val_harmful
            and val_gain + 1e-12 >= args.min_val_gain
        )
        key_best["validation_gain"] = val_gain
        per_key[key] = key_best
        if key_best["accepted"]:
            # Final selection is driven by held-out safety calibration, not the
            # router-training objective.
            val_obj = (
                float(val["accuracy"] or 0.0),
                int(val["rescue_margin"]),
                -int(val["routed"]),
                float(key_best["router_train"]["accuracy"] or 0.0),
            )
            candidate_final = dict(key_best)
            candidate_final["validation_objective"] = list(val_obj)
            if best is None or val_obj > tuple(best["validation_objective"]):
                best = candidate_final

    if best is None:
        selected = {
            "accepted": False,
            "reason": "no candidate passed held-out safety calibration",
            "score_key": None,
            "route_frac": 0.0,
            "router_train": evaluate(router_train, None, 0.0),
            "safety_cal": evaluate(safety_cal, None, 0.0),
            "target": evaluate(target, None, 0.0),
            "all": evaluate(rows, None, 0.0),
        }
    else:
        selected = {
            **best,
            "target": evaluate(target, best["score_key"], best["route_frac"]),
            "all": evaluate(rows, best["score_key"], best["route_frac"]),
        }

    summary = {
        "version": "nested-safe-native-router-v1",
        "source_analysis": str(args.analysis.resolve()),
        "n": len(rows),
        "router_train_n": len(router_train),
        "safety_cal_n": len(safety_cal),
        "target_n": len(target),
        "max_route_frac": args.max_route_frac,
        "min_val_gain": args.min_val_gain,
        "allow_val_harmful": args.allow_val_harmful,
        "selection_objective": args.selection_objective,
        "greedy_accuracy": acc([bool(r["greedy_correct"]) for r in rows]),
        "mitigation_accuracy": acc([bool(r["mitigation_correct"]) for r in rows]),
        "rescues": sum(bool(r["rescue"]) for r in rows),
        "harmful": sum(bool(r["harmful"]) for r in rows),
    }
    out = {"summary": summary, "selected_nested_safe_router": selected, "per_key": per_key}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({"summary": summary, "selected_nested_safe_router": selected}, indent=2))


if __name__ == "__main__":
    main()
