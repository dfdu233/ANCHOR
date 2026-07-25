"""Analyze native multi-view alignment hypotheses and CP behavior.

This script is intentionally offline: it reuses cached native-view logits from
native_view_projection.py and tests whether (i) moving views closer to a native
support is associated with fewer errors, (ii) stronger alignment also damages
image evidence, and (iii) conformal prediction can absorb the resulting
trade-off by expanding the answer set on unstable samples.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--gammas", type=float, nargs="*", default=(0.8, 0.9, 0.95))
    p.add_argument("--max-views", type=int, default=4)
    p.add_argument("--min-edge", type=float, default=0.85)
    p.add_argument("--min-psnr", type=float, default=18.0)
    return p.parse_args()


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def pred(logits):
    return int(np.argmax(np.asarray(logits, dtype=np.float64)))


def acc(xs):
    return float(np.mean(xs)) if xs else None


def finite_psnr(v):
    if v is None:
        return 99.0
    try:
        x = float(v)
    except Exception:
        return 0.0
    return 99.0 if math.isinf(x) else x


def rankdata(values):
    vals = np.asarray(values, dtype=np.float64)
    order = np.argsort(vals)
    ranks = np.empty(len(vals), dtype=np.float64)
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[order[j]] == vals[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx == 0.0 or sy == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


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


def quantile_cp(scores, gamma):
    # Split conformal conservative finite-sample quantile.
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    if len(scores) == 0:
        return 1.0
    k = int(math.ceil((len(scores) + 1) * gamma)) - 1
    k = min(max(k, 0), len(scores) - 1)
    return float(scores[k])


def candidate_quality(c):
    st = c.get("structure") or {}
    edge = float(st.get("edge_correlation", 0.0))
    psnr = finite_psnr(st.get("psnr"))
    damage = (1.0 - edge) + 0.02 * max(0.0, 45.0 - psnr)
    closure = float(c.get("native_closure") or 0.0)
    dist = float(c.get("native_distance") or 0.0)
    return closure, dist, edge, psnr, damage


def safe_source_indices(row, max_views, min_edge, min_psnr):
    cands = row.get("candidates", [])
    idxs = []
    for i, c in enumerate(cands[1:], start=1):
        closure, dist, edge, psnr, _ = candidate_quality(c)
        if not c.get("structure_safe", False):
            continue
        if edge < min_edge or psnr < min_psnr:
            continue
        idxs.append((closure, -dist, i))
    idxs.sort(reverse=True)
    return [i for _, __, i in idxs[:max_views]]


def fused_probs(row, mode, max_views, min_edge, min_psnr):
    cands = row.get("candidates", [])
    if not cands:
        return None
    if mode == "original":
        return softmax(cands[0]["logits"])
    if mode == "safe_native_max":
        idxs = [0] + safe_source_indices(row, max_views, min_edge, min_psnr)
    elif mode == "all_native_max":
        idxs = list(range(min(len(cands), max_views + 1)))
    elif mode == "laplacian":
        probs = row.get("laplacian_fusion_probs")
        if probs is not None:
            return np.asarray(probs, dtype=np.float64)
        idxs = [0]
    else:
        raise ValueError(mode)
    ps = np.stack([softmax(cands[i]["logits"]) for i in idxs], axis=0)
    return np.max(ps, axis=0)


def cp_eval(rows_train, rows_test, mode, gamma, max_views, min_edge, min_psnr):
    train_scores = []
    for r in rows_train:
        p = fused_probs(r, mode, max_views, min_edge, min_psnr)
        if p is None:
            continue
        y = int(r["gt_index"])
        train_scores.append(1.0 - float(p[y]))
    q = quantile_cp(train_scores, gamma)
    details = []
    for r in rows_test:
        p = fused_probs(r, mode, max_views, min_edge, min_psnr)
        if p is None:
            continue
        y = int(r["gt_index"])
        included = [i for i, pi in enumerate(p) if 1.0 - float(pi) <= q]
        details.append({
            "qid": str(r.get("qid")),
            "covered": y in included,
            "set_size": len(included),
            "singleton": len(included) == 1,
            "abstain": len(included) != 1,
            "pred": int(np.argmax(p)),
            "gt": y,
        })
    return {
        "mode": mode,
        "gamma": float(gamma),
        "qhat": q,
        "n_train": len(train_scores),
        "n_test": len(details),
        "coverage": acc([d["covered"] for d in details]),
        "avg_set_size": float(np.mean([d["set_size"] for d in details])) if details else None,
        "singleton_rate": acc([d["singleton"] for d in details]),
        "singleton_accuracy": acc([d["pred"] == d["gt"] for d in details if d["singleton"]]),
        "abstain_rate": acc([d["abstain"] for d in details]),
    }


def analyze_path(path: Path, args):
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    train = [r for r in rows if r.get("split") == "train"]
    test = [r for r in rows if r.get("split") == "test"]
    pair_rows = []
    view_rows = []
    for r in rows:
        if not r.get("candidates"):
            continue
        y = int(r["gt_index"])
        base_pred = pred(r["candidates"][0]["logits"])
        base_correct = base_pred == y
        for i, c in enumerate(r["candidates"]):
            closure, dist, edge, psnr, damage = candidate_quality(c)
            p = pred(c["logits"])
            correct = p == y
            item = {
                "qid": str(r.get("qid")),
                "split": r.get("split"),
                "index": i,
                "name": c.get("name"),
                "family": c.get("family"),
                "closure": closure,
                "distance": dist,
                "edge": edge,
                "psnr": psnr,
                "damage": damage,
                "correct": correct,
                "base_correct": base_correct,
                "changed": p != base_pred,
                "rescue": (not base_correct) and correct,
                "harmful": base_correct and (not correct),
            }
            view_rows.append(item)
            if i > 0:
                pair_rows.append(item)
    non_orig = [v for v in view_rows if v["index"] > 0]
    changed = [v for v in non_orig if v["changed"]]
    by_family = {}
    for fam in sorted(set(v["family"] for v in non_orig)):
        xs = [v for v in non_orig if v["family"] == fam]
        by_family[fam] = summarize_views(xs)
    bins = alignment_bins(non_orig, "closure", 5)
    cp = []
    for gamma in args.gammas:
        for mode in ("original", "safe_native_max", "all_native_max", "laplacian"):
            cp.append(cp_eval(train, test, mode, gamma, args.max_views, args.min_edge, args.min_psnr))
    return {
        "path": str(path),
        "cache_summary": payload.get("summary", {}),
        "n_rows": len(rows),
        "n_non_original_views": len(non_orig),
        "overall_non_original": summarize_views(non_orig),
        "changed_views": summarize_views(changed),
        "by_family": by_family,
        "correlations": {
            "closure_vs_correct_spearman": spearman([v["closure"] for v in non_orig], [v["correct"] for v in non_orig]),
            "closure_vs_rescue_auroc": auroc([v["closure"] for v in non_orig], [v["rescue"] for v in non_orig]),
            "closure_vs_harmful_auroc": auroc([v["closure"] for v in non_orig], [v["harmful"] for v in non_orig]),
            "damage_vs_harmful_auroc": auroc([v["damage"] for v in non_orig], [v["harmful"] for v in non_orig]),
            "damage_vs_rescue_auroc": auroc([v["damage"] for v in non_orig], [v["rescue"] for v in non_orig]),
        },
        "alignment_bins_by_closure": bins,
        "cp": cp,
        "diagnosis": diagnose(non_orig, bins, cp),
    }


def summarize_views(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "accuracy": acc([x["correct"] for x in xs]),
        "base_accuracy": acc([x["base_correct"] for x in xs]),
        "changed_rate": acc([x["changed"] for x in xs]),
        "rescues": int(sum(x["rescue"] for x in xs)),
        "harmful": int(sum(x["harmful"] for x in xs)),
        "mean_closure": float(np.mean([x["closure"] for x in xs])),
        "mean_distance": float(np.mean([x["distance"] for x in xs])),
        "mean_damage": float(np.mean([x["damage"] for x in xs])),
        "mean_edge": float(np.mean([x["edge"] for x in xs])),
        "mean_psnr": float(np.mean([x["psnr"] for x in xs])),
    }


def alignment_bins(xs, key, n_bins):
    if not xs:
        return []
    vals = np.asarray([x[key] for x in xs], dtype=np.float64)
    qs = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        if i == n_bins - 1:
            members = [x for x in xs if lo <= x[key] <= hi]
        else:
            members = [x for x in xs if lo <= x[key] < hi]
        item = summarize_views(members)
        item["bin"] = i
        item["closure_range"] = [float(lo), float(hi)]
        out.append(item)
    return out


def diagnose(non_orig, bins, cp):
    total_rescue = sum(v["rescue"] for v in non_orig)
    total_harm = sum(v["harmful"] for v in non_orig)
    tradeoff = False
    if bins:
        damages = [b.get("mean_damage", 0.0) for b in bins if b.get("n", 0)]
        harms = [b.get("harmful", 0) for b in bins if b.get("n", 0)]
        if len(damages) >= 2 and damages[-1] > damages[0] and max(harms) > 0:
            tradeoff = True
    cp_gain = []
    for gamma in sorted(set(c["gamma"] for c in cp)):
        orig = next(c for c in cp if c["gamma"] == gamma and c["mode"] == "original")
        for mv in [c for c in cp if c["gamma"] == gamma and c["mode"] != "original"]:
            cp_gain.append({
                "gamma": gamma,
                "mode": mv["mode"],
                "coverage_delta": None if orig["coverage"] is None or mv["coverage"] is None else mv["coverage"] - orig["coverage"],
                "set_size_delta": None if orig["avg_set_size"] is None or mv["avg_set_size"] is None else mv["avg_set_size"] - orig["avg_set_size"],
            })
    return {
        "has_any_view_rescue": bool(total_rescue > 0),
        "has_any_view_harmful": bool(total_harm > 0),
        "tradeoff_observed": tradeoff,
        "point_prediction_supported": bool(total_rescue > total_harm),
        "cp_deltas_vs_original": cp_gain,
    }


def main():
    args = parse_args()
    reports = [analyze_path(p, args) for p in args.paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
