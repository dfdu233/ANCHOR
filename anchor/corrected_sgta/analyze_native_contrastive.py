"""Offline center-aware contrastive calibration over native-view caches.

This tests whether native-aligned views can help as residual evidence without
rerunning the VLM.  Given cached per-view logits, choose source-like and
anti-source views by native distance/closure, then calibrate a simple contrastive
logit rule on the train split and report locked test behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--alphas", type=float, nargs="*", default=(-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4,6,8))
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def acc(xs):
    return float(np.mean(xs)) if xs else None


def pred(logits):
    return int(np.argmax(np.asarray(logits, dtype=float)))


def mean_logits(candidates, indices):
    return np.mean([np.asarray(candidates[i]["logits"], dtype=float) for i in indices], axis=0)


def candidate_groups(row, top_k):
    cands = row.get("candidates", [])
    if not cands:
        return {"source": [0], "anti": [0], "stable": [0]}
    non_original = list(range(1, len(cands))) or [0]
    # Source-like = maximum native closure, then minimum native distance.
    def source_key(i):
        c = cands[i]
        closure = c.get("native_closure")
        dist = c.get("native_distance")
        return (float(closure if closure is not None else -1e9), -float(dist if dist is not None else 1e9))
    def anti_key(i):
        c = cands[i]
        closure = c.get("native_closure")
        dist = c.get("native_distance")
        return (float(closure if closure is not None else 1e9), -float(dist if dist is not None else -1e9))
    source = sorted(non_original, key=source_key, reverse=True)[:top_k]
    anti = sorted(non_original, key=anti_key)[:top_k]
    # Stable source = source-like views that agree with original prediction.
    base = pred(cands[0]["logits"])
    stable = [i for i in source if pred(cands[i]["logits"]) == base]
    if not stable:
        stable = source[:1]
    return {"source": source, "anti": anti, "stable": stable}


def logits_for(row, method, alpha, top_k):
    cands = row["candidates"]
    z0 = np.asarray(cands[0]["logits"], dtype=float)
    groups = candidate_groups(row, top_k)
    zs = mean_logits(cands, groups["source"])
    za = mean_logits(cands, groups["anti"])
    zstable = mean_logits(cands, groups["stable"])
    if method == "source_extrapolate":
        return z0 + alpha * (zs - z0)
    if method == "stable_extrapolate":
        return z0 + alpha * (zstable - z0)
    if method == "source_vs_anti":
        return z0 + alpha * (zs - za)
    if method == "anti_suppress":
        return z0 - alpha * (za - z0)
    if method == "source_only":
        return zs
    raise ValueError(method)


def evaluate(rows, method, alpha, top_k):
    out = []
    for row in rows:
        if not row.get("candidates"):
            continue
        y = int(row["gt_index"])
        base = pred(row["candidates"][0]["logits"])
        p = pred(logits_for(row, method, alpha, top_k))
        out.append({"qid": row["qid"], "base": base, "pred": p, "gt": y, "base_correct": base == y, "correct": p == y})
    return {
        "n": len(out),
        "accuracy": acc([x["correct"] for x in out]),
        "base_accuracy": acc([x["base_correct"] for x in out]),
        "rescues": sum((not x["base_correct"]) and x["correct"] for x in out),
        "harmful": sum(x["base_correct"] and (not x["correct"]) for x in out),
        "changed": sum(x["base"] != x["pred"] for x in out),
        "rows": out,
    }


def analyze(path: Path, alphas, top_k):
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    train = [r for r in rows if r.get("split") == "train"]
    test = [r for r in rows if r.get("split") == "test"]
    methods = ["source_extrapolate", "stable_extrapolate", "source_vs_anti", "anti_suppress", "source_only"]
    trials = []
    for method in methods:
        if method == "source_only":
            tr = evaluate(train, method, 0.0, top_k)
            trials.append((tr["accuracy"], -tr["harmful"], tr["rescues"], method, 0.0, tr))
        else:
            for alpha in alphas:
                tr = evaluate(train, method, float(alpha), top_k)
                trials.append((tr["accuracy"], -tr["harmful"], tr["rescues"], method, float(alpha), tr))
    trials.sort(reverse=True, key=lambda x: (x[0] if x[0] is not None else -1, x[1], x[2]))
    best_acc, _, _, best_method, best_alpha, train_result = trials[0]
    test_result = evaluate(test, best_method, best_alpha, top_k)
    overall_result = evaluate(rows, best_method, best_alpha, top_k)
    # Also report the best oracle over all calibrated formulas on test for diagnosis only.
    test_trials = []
    for method in methods:
        for alpha in ([0.0] if method == "source_only" else alphas):
            te = evaluate(test, method, float(alpha), top_k)
            test_trials.append((te["accuracy"], -te["harmful"], te["rescues"], method, float(alpha), te))
    test_trials.sort(reverse=True, key=lambda x: (x[0] if x[0] is not None else -1, x[1], x[2]))
    return {
        "path": str(path),
        "cache_summary": payload.get("summary", {}),
        "selected": {"method": best_method, "alpha": best_alpha, "top_k": top_k},
        "train": {k:v for k,v in train_result.items() if k != "rows"},
        "test": {k:v for k,v in test_result.items() if k != "rows"},
        "overall": {k:v for k,v in overall_result.items() if k != "rows"},
        "diagnostic_best_test": {"method": test_trials[0][3], "alpha": test_trials[0][4], **{k:v for k,v in test_trials[0][5].items() if k != "rows"}},
    }


def main():
    args = parse_args()
    reports = [analyze(path, args.alphas, args.top_k) for path in args.paths]
    print(json.dumps(reports, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
