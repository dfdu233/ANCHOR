"""Post-hoc safe router for competence-native SGTA analyses.

The earlier router used an absolute risk threshold learned on calibration.
That is brittle when the risk distribution shifts between calibration and a
new deployment batch.  This script evaluates a simpler batch-rank variant:

  1. choose a risk score and routed fraction on calibration;
  2. require non-inferiority to greedy and zero harmful flips on calibration;
  3. route only the same top-risk fraction in the unlabeled target batch.

The rule uses no target labels to decide which target examples are routed.
Target labels are used only here for retrospective evaluation.
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
    parser.add_argument("--min-train-gain", type=float, default=0.0)
    parser.add_argument("--allow-calibration-harmful", type=int, default=0)
    return parser.parse_args()


def acc(values):
    return float(np.mean(values)) if values else None


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    for value in pos:
        wins += float((value > neg).sum()) + 0.5 * float((value == neg).sum())
    return float(wins / (len(pos) * len(neg)))


def routed_qids_by_fraction(rows, score_key: str, frac: float) -> set[str]:
    if not rows or frac <= 0:
        return set()
    k = int(math.ceil(frac * len(rows)))
    k = min(max(k, 0), len(rows))
    if k == 0:
        return set()
    ranked = sorted(
        rows,
        key=lambda r: (float(r["risk_scores"][score_key]), str(r["qid"])),
        reverse=True,
    )
    return {str(r["qid"]) for r in ranked[:k]}


def evaluate(rows, score_key: str, frac: float):
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
        "routed_qids": sorted(routed),
    }


def main():
    args = parse_args()
    payload = json.loads(args.analysis.read_text())
    rows = payload["rows"]
    train = [r for r in rows if r.get("split") == "train"]
    test = [r for r in rows if r.get("split") == "test"]
    score_keys = sorted(rows[0]["risk_scores"]) if rows else []
    train_greedy = acc([bool(r["greedy_correct"]) for r in train]) or 0.0

    metrics = {}
    selected = None
    for key in score_keys:
        valid_train = [r for r in train if r["risk_scores"].get(key) is not None]
        valid_test = [r for r in test if r["risk_scores"].get(key) is not None]
        valid_all = [r for r in rows if r["risk_scores"].get(key) is not None]
        if not valid_train:
            continue
        max_k = int(math.floor(args.max_route_frac * len(valid_train)))
        candidate_fracs = sorted({0.0} | {k / len(valid_train) for k in range(1, max_k + 1)})
        key_metrics = {
            "auroc_greedy_error": auroc(
                [float(r["risk_scores"][key]) for r in valid_all],
                [not bool(r["greedy_correct"]) for r in valid_all],
            ),
            "auroc_mitigation_rescue": auroc(
                [float(r["risk_scores"][key]) for r in valid_all],
                [bool(r["rescue"]) for r in valid_all],
            ),
            "auroc_mitigation_harmful": auroc(
                [float(r["risk_scores"][key]) for r in valid_all],
                [bool(r["harmful"]) for r in valid_all],
            ),
        }
        best_for_key = None
        for frac in candidate_fracs:
            tr = evaluate(valid_train, key, frac)
            if tr["harmful_introduced"] > args.allow_calibration_harmful:
                continue
            if (tr["accuracy"] or 0.0) + 1e-12 < train_greedy + args.min_train_gain:
                continue
            # Conservative objective: first maximize calibration accuracy, then
            # keep more verified rescues, then use fewer routed examples.
            obj = (
                float(tr["accuracy"] or 0.0),
                int(tr["rescues_kept"]),
                -int(tr["routed"]),
                float(key_metrics["auroc_mitigation_rescue"] or -1.0),
            )
            candidate = {
                "score_key": key,
                "objective": list(obj),
                "train": tr,
                "test": evaluate(valid_test, key, frac),
                "all": evaluate(valid_all, key, frac),
            }
            if best_for_key is None or obj > tuple(best_for_key["objective"]):
                best_for_key = candidate
        if best_for_key is not None:
            key_metrics["safe_rank_router"] = best_for_key
            if selected is None or tuple(best_for_key["objective"]) > tuple(selected["objective"]):
                selected = best_for_key
        metrics[key] = key_metrics

    summary = {
        "version": "safe-native-rank-router-v1",
        "source_analysis": str(args.analysis.resolve()),
        "n": len(rows),
        "train_n": len(train),
        "test_n": len(test),
        "max_route_frac": args.max_route_frac,
        "min_train_gain": args.min_train_gain,
        "allow_calibration_harmful": args.allow_calibration_harmful,
        "greedy_accuracy": acc([bool(r["greedy_correct"]) for r in rows]),
        "mitigation_accuracy": acc([bool(r["mitigation_correct"]) for r in rows]),
        "rescues": sum(bool(r["rescue"]) for r in rows),
        "harmful": sum(bool(r["harmful"]) for r in rows),
    }
    out = {"summary": summary, "selected_safe_rank_router": selected, "risk_metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({"summary": summary, "selected_safe_rank_router": selected}, indent=2))


if __name__ == "__main__":
    main()
