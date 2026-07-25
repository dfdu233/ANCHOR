"""Paired statistics for native-risk router analyses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True, type=Path)
    p.add_argument("--router", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", choices=("test", "all", "train"), default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=10000)
    return p.parse_args()


def mean_bool(xs):
    return float(np.mean(np.asarray(xs, dtype=np.float64))) if xs else None


def exact_mcnemar_p(a_correct, b_correct):
    # discordant counts: b correct / a wrong vs a correct / b wrong
    b_win = sum((not a) and b for a, b in zip(a_correct, b_correct))
    a_win = sum(a and (not b) for a, b in zip(a_correct, b_correct))
    n = b_win + a_win
    if n == 0:
        return {"b_win": b_win, "a_win": a_win, "discordant": n, "p_exact": 1.0}
    k = min(a_win, b_win)
    # two-sided exact binomial under p=0.5
    prob = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"b_win": b_win, "a_win": a_win, "discordant": n, "p_exact": float(min(1.0, 2.0 * prob))}


def bootstrap_delta(a, b, rng, n_boot):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    if n == 0:
        return None
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(np.mean(b[idx]) - np.mean(a[idx]))
    return {
        "mean_delta": float(np.mean(b) - np.mean(a)),
        "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
    }


def main():
    args = parse_args()
    analysis = json.loads(args.analysis.read_text())
    router_payload = json.loads(args.router.read_text())
    selected = router_payload.get("selected_safe_rank_router") or {}
    routed = set(map(str, ((selected.get(args.split) or {}).get("routed_qids") or [])))
    rows = [r for r in analysis["rows"] if args.split == "all" or r.get("split") == args.split]
    greedy = [bool(r["greedy_correct"]) for r in rows]
    mitigation = [bool(r["mitigation_correct"]) for r in rows]
    router = [
        bool(r["mitigation_correct"]) if str(r["qid"]) in routed else bool(r["greedy_correct"])
        for r in rows
    ]
    rng = np.random.default_rng(args.seed)
    out = {
        "analysis": str(args.analysis),
        "router": str(args.router),
        "split": args.split,
        "n": len(rows),
        "selected_score_key": selected.get("score_key"),
        "route_frac": (selected.get(args.split) or {}).get("route_frac"),
        "routed": len(routed),
        "accuracy": {
            "greedy": mean_bool(greedy),
            "mitigation": mean_bool(mitigation),
            "router": mean_bool(router),
        },
        "paired": {
            "router_vs_greedy": {
                "mcnemar": exact_mcnemar_p(greedy, router),
                "bootstrap_delta": bootstrap_delta(greedy, router, rng, args.bootstrap),
            },
            "router_vs_mitigation": {
                "mcnemar": exact_mcnemar_p(mitigation, router),
                "bootstrap_delta": bootstrap_delta(mitigation, router, rng, args.bootstrap),
            },
            "mitigation_vs_greedy": {
                "mcnemar": exact_mcnemar_p(greedy, mitigation),
                "bootstrap_delta": bootstrap_delta(greedy, mitigation, rng, args.bootstrap),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
