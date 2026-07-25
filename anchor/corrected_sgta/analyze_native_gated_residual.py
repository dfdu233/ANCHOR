"""Conservative native-source residual calibration for cached native views.

The goal is deliberately small and paper-friendly: keep the original VLM
prediction unless native-aligned views provide a strong, source-consistent
counter-signal.  This script does not rerun the VLM; it reuses cached logits
from native_view_projection.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--alphas", type=float, nargs="*", default=(1.0, 2.0, 3.0, 4.0, 6.0, 8.0))
    parser.add_argument("--margin-thresholds", type=float, nargs="*", default=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5))
    parser.add_argument("--min-mean-closure", type=float, nargs="*", default=(0.0, 0.00002, 0.00005, 0.0001))
    parser.add_argument("--min-edge-corr", type=float, nargs="*", default=(0.85, 0.9, 0.95, 0.98))
    parser.add_argument("--min-psnr", type=float, nargs="*", default=(18.0, 25.0, 35.0, 45.0))
    parser.add_argument("--min-train-rescues", type=int, default=1)
    parser.add_argument("--max-train-harmful", type=int, default=0)
    parser.add_argument("--allow-unsafe-select", action="store_true", help="Allow selecting configs with train harmful flips.")
    return parser.parse_args()


def pred(logits) -> int:
    return int(np.argmax(np.asarray(logits, dtype=float)))


def margin(logits) -> float:
    z = np.sort(np.asarray(logits, dtype=float))
    if len(z) < 2:
        return 0.0
    return float(z[-1] - z[-2])


def entropy(logits) -> float:
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(np.maximum(p, 1e-12))))


def mean_logits(cands, idxs):
    return np.mean([np.asarray(cands[i]["logits"], dtype=float) for i in idxs], axis=0)


def finite_psnr(value) -> float:
    if value is None:
        return 99.0
    try:
        v = float(value)
    except Exception:
        return 0.0
    if math.isinf(v):
        return 99.0
    return v


def source_indices(row, top_k: int, min_edge: float, min_psnr: float, min_closure: float):
    cands = row.get("candidates", [])
    non_original = []
    for i, c in enumerate(cands[1:], start=1):
        st = c.get("structure") or {}
        if not c.get("structure_safe", False):
            continue
        if float(st.get("edge_correlation", 0.0)) < min_edge:
            continue
        if finite_psnr(st.get("psnr")) < min_psnr:
            continue
        if float(c.get("native_closure") or 0.0) < min_closure:
            continue
        non_original.append(i)
    if not non_original:
        return []
    non_original.sort(
        key=lambda i: (
            float(cands[i].get("native_closure") or -1e9),
            -float(cands[i].get("native_distance") or 1e9),
        ),
        reverse=True,
    )
    return non_original[:top_k]


def apply_rule(row, cfg):
    cands = row.get("candidates", [])
    if not cands:
        return None
    z0 = np.asarray(cands[0]["logits"], dtype=float)
    base = pred(z0)
    if margin(z0) > cfg["max_base_margin"]:
        return base, False, "base_confident"
    idxs = source_indices(row, cfg["top_k"], cfg["min_edge_corr"], cfg["min_psnr"], cfg["min_mean_closure"])
    if len(idxs) < cfg["min_views"]:
        return base, False, "insufficient_safe_source_views"
    source_preds = [pred(cands[i]["logits"]) for i in idxs]
    majority = max(set(source_preds), key=source_preds.count)
    agree = source_preds.count(majority) / len(source_preds)
    if agree < cfg["min_agree"]:
        return base, False, "source_views_disagree"
    zs = mean_logits(cands, idxs)
    z = z0 + cfg["alpha"] * (zs - z0)
    out = pred(z)
    if out == base:
        return base, False, "residual_no_change"
    if out != majority:
        return base, False, "residual_not_source_consistent"
    if margin(z) < cfg["min_new_margin"]:
        return base, False, "new_margin_too_small"
    return out, True, "native_residual_gate"


def evaluate(rows, cfg):
    details = []
    for row in rows:
        got = apply_rule(row, cfg)
        if got is None:
            continue
        p, routed, reason = got
        y = int(row["gt_index"])
        base = pred(row["candidates"][0]["logits"])
        details.append({
            "qid": row.get("qid"),
            "gt": y,
            "base": base,
            "pred": p,
            "base_correct": base == y,
            "correct": p == y,
            "routed": bool(routed),
            "reason": reason,
            "base_margin": margin(row["candidates"][0]["logits"]),
            "base_entropy": entropy(row["candidates"][0]["logits"]),
        })
    if not details:
        return {"n": 0, "accuracy": None, "base_accuracy": None, "rescues": 0, "harmful": 0, "changed": 0, "routed": 0, "rows": []}
    return {
        "n": len(details),
        "accuracy": float(np.mean([x["correct"] for x in details])),
        "base_accuracy": float(np.mean([x["base_correct"] for x in details])),
        "rescues": int(sum((not x["base_correct"]) and x["correct"] for x in details)),
        "harmful": int(sum(x["base_correct"] and (not x["correct"]) for x in details)),
        "changed": int(sum(x["base"] != x["pred"] for x in details)),
        "routed": int(sum(x["routed"] for x in details)),
        "rows": details,
    }


def summarize_result(result):
    return {k: v for k, v in result.items() if k != "rows"}


def cfg_sort_key(item):
    cfg, tr = item
    gain = (tr["accuracy"] or 0.0) - (tr["base_accuracy"] or 0.0)
    return (
        tr["harmful"] == 0,
        gain,
        tr["rescues"] - tr["harmful"],
        -tr["harmful"],
        -tr["routed"],
        -cfg["alpha"],
    )


def analyze(path: Path, args):
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    train = [r for r in rows if r.get("split") == "train"]
    test = [r for r in rows if r.get("split") == "test"]
    configs = []
    for alpha in args.alphas:
        for max_margin in args.margin_thresholds:
            for min_closure in args.min_mean_closure:
                for min_edge in args.min_edge_corr:
                    for min_psnr in args.min_psnr:
                        for min_views in (1, 2, min(args.top_k, 3)):
                            for min_agree in (0.67, 1.0):
                                cfg = {
                                    "alpha": float(alpha),
                                    "top_k": int(args.top_k),
                                    "max_base_margin": float(max_margin),
                                    "min_mean_closure": float(min_closure),
                                    "min_edge_corr": float(min_edge),
                                    "min_psnr": float(min_psnr),
                                    "min_views": int(min_views),
                                    "min_agree": float(min_agree),
                                    "min_new_margin": 0.05,
                                }
                                tr = evaluate(train, cfg)
                                configs.append((cfg, tr))
    eligible = []
    for cfg, tr in configs:
        if not args.allow_unsafe_select and tr["harmful"] > args.max_train_harmful:
            continue
        if tr["rescues"] < args.min_train_rescues:
            continue
        eligible.append((cfg, tr))
    if not eligible:
        selected = {
            "alpha": 0.0,
            "top_k": int(args.top_k),
            "max_base_margin": -1.0,
            "min_mean_closure": 999.0,
            "min_edge_corr": 1.0,
            "min_psnr": 999.0,
            "min_views": 99,
            "min_agree": 1.0,
            "min_new_margin": 999.0,
            "fallback_noop": True,
        }
        train_result = evaluate(train, selected)
    else:
        eligible.sort(key=cfg_sort_key, reverse=True)
        selected, train_result = eligible[0]
        selected = dict(selected)
        selected["fallback_noop"] = False
    test_result = evaluate(test, selected)
    overall_result = evaluate(rows, selected)
    # Diagnostic oracle: best gated residual on test, not a valid selected result.
    test_trials = [(cfg, evaluate(test, cfg)) for cfg, _ in configs]
    test_trials.sort(key=cfg_sort_key, reverse=True)
    best_test_cfg, best_test = test_trials[0]
    return {
        "path": str(path),
        "cache_summary": payload.get("summary", {}),
        "selected": selected,
        "train": summarize_result(train_result),
        "test": summarize_result(test_result),
        "overall": summarize_result(overall_result),
        "diagnostic_best_test": {"config": best_test_cfg, **summarize_result(best_test)},
        "selected_test_rows": [x for x in test_result["rows"] if x["base"] != x["pred"]],
    }


def main():
    args = parse_args()
    reports = [analyze(path, args) for path in args.paths]
    text = json.dumps(reports, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
