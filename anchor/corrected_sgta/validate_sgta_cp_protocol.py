"""Validate a compact SGTA-CP protocol across split seeds.

The goal is not to search for a best gate, but to stress-test a paper-friendly
fixed protocol:

  risk(x) = rank(view-JS) + rank(mean structural damage) + rank(original uncertainty)

The router split only chooses how much of the test distribution should be
treated as high risk.  Proper conformal thresholds are then fitted on a disjoint
calibration split, and results are reported on the locked test split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.analyze_risk_gated_multiview_cp import (
    cp_set,
    cp_quantile,
    eval_policy_with_probs,
    mean,
    multiview_probs,
    original_probs,
    row_features,
    stable_u01,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--gammas", nargs="*", type=float, default=(0.8, 0.9, 0.95))
    p.add_argument("--modes", nargs="*", choices=("max", "laplacian"), default=("max", "laplacian"))
    p.add_argument("--seeds", nargs="*", type=int, default=list(range(20)))
    p.add_argument("--high-risk-fracs", nargs="*", type=float, default=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    p.add_argument("--router-frac", type=float, default=0.5)
    p.add_argument("--max-views", type=int, default=4)
    p.add_argument("--min-edge", type=float, default=0.85)
    p.add_argument("--min-psnr", type=float, default=18.0)
    return p.parse_args()


def split_train(rows: list[dict[str, Any]], seed: int, router_frac: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [r for r in rows if r.get("split") == "train"]
    pairs = [(stable_u01(str(r.get("qid", r.get("img_name", i))), seed), r) for i, r in enumerate(train)]
    pairs.sort(key=lambda x: x[0])
    n_router = max(1, min(len(pairs) - 1, int(round(len(pairs) * router_frac)))) if len(pairs) >= 2 else len(pairs)
    return [r for _, r in pairs[:n_router]], [r for _, r in pairs[n_router:]]


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return ranks / float(len(values) - 1)


def fit_risk_scaler(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    feats = [row_features(r, args) for r in rows]
    names = ("view_js", "mean_damage", "orig_uncertainty")
    scaler = {}
    for n in names:
        vals = np.asarray([f[n] for f in feats], dtype=np.float64)
        scaler[n] = (float(np.min(vals)), float(np.max(vals)))
    return scaler


def risk_score(row: dict[str, Any], args: argparse.Namespace, scaler: dict[str, tuple[float, float]]) -> float:
    f = row_features(row, args)
    score = 0.0
    for n in ("view_js", "mean_damage", "orig_uncertainty"):
        lo, hi = scaler[n]
        if hi <= lo:
            z = 0.0
        else:
            z = (float(f[n]) - lo) / (hi - lo)
        score += min(1.0, max(0.0, z))
    return float(score / 3.0)


def choose_frac(router_rows: list[dict[str, Any]], gamma: float, args: argparse.Namespace, scaler: dict[str, tuple[float, float]]) -> dict[str, Any]:
    best = None
    risks = np.asarray([risk_score(r, args, scaler) for r in router_rows], dtype=np.float64)
    for frac in args.high_risk_fracs:
        thr = float(np.quantile(risks, 1.0 - frac))
        q_low = fit_q(router_rows, risks < thr, False, gamma, args)
        q_high = fit_q(router_rows, risks >= thr, True, gamma, args)
        m = eval_rows(router_rows, risks, thr, q_low, q_high, args)
        feasible = m["coverage"] is not None and m["coverage"] >= gamma
        # Conservative: satisfy coverage, then minimize size, then prefer singleton.
        score = (
            (0.0 if feasible else -1000.0)
            - float(m["avg_set_size"] or 9.0)
            + 0.25 * float(m["singleton_rate"] or 0.0)
        )
        item = {"frac": frac, "threshold": thr, "q_low_router": q_low, "q_high_router": q_high, "router": m, "score": score, "feasible": feasible}
        if best is None or score > best["score"]:
            best = item
    assert best is not None
    return best


def fit_q(rows: list[dict[str, Any]], mask: np.ndarray, high: bool, gamma: float, args: argparse.Namespace) -> float:
    scores = []
    for r, is_high in zip(rows, mask):
        if bool(is_high) != high:
            continue
        p = multiview_probs(r, args) if high else original_probs(r)
        scores.append(1.0 - float(p[int(r["gt_index"])]))
    return cp_quantile(scores, gamma)


def eval_rows(rows: list[dict[str, Any]], risks: np.ndarray, thr: float, q_low: float, q_high: float, args: argparse.Namespace) -> dict[str, Any]:
    details = []
    for r, risk in zip(rows, risks):
        high = bool(risk >= thr)
        p = multiview_probs(r, args) if high else original_probs(r)
        s = cp_set(p, q_high if high else q_low)
        y = int(r["gt_index"])
        pred = int(np.argmax(p))
        opred = int(np.argmax(original_probs(r)))
        details.append({
            "high": high,
            "covered": y in s,
            "set_size": len(s),
            "singleton": len(s) == 1,
            "singleton_correct": (pred == y) if len(s) == 1 else None,
            "point_correct": pred == y,
            "orig_correct": opred == y,
            "changed": pred != opred,
        })
    single = [d for d in details if d["singleton"]]
    changed = [d for d in details if d["changed"]]
    return {
        "n": len(details),
        "coverage": mean([d["covered"] for d in details]),
        "avg_set_size": mean([d["set_size"] for d in details]),
        "singleton_rate": mean([d["singleton"] for d in details]),
        "singleton_accuracy": mean([d["singleton_correct"] for d in single]),
        "high_risk_rate": mean([d["high"] for d in details]),
        "point_accuracy": mean([d["point_correct"] for d in details]),
        "orig_point_accuracy": mean([d["orig_correct"] for d in details]),
        "rescues": int(sum((not d["orig_correct"]) and d["point_correct"] for d in changed)),
        "harmful": int(sum(d["orig_correct"] and (not d["point_correct"]) for d in changed)),
    }


def summarize(vals: list[float]) -> dict[str, float | None]:
    xs = np.asarray([v for v in vals if v is not None and math.isfinite(v)], dtype=np.float64)
    if len(xs) == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {"mean": float(xs.mean()), "std": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0, "min": float(xs.min()), "max": float(xs.max())}


def analyze_path(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    test = [r for r in rows if r.get("split") == "test"]
    out = {"path": str(path), "cache_summary": payload.get("summary", {}), "modes": {}}
    for mode in args.modes:
        args.mode = mode
        mode_runs = []
        for seed in args.seeds:
            router, cal = split_train(rows, seed, args.router_frac)
            scaler = fit_risk_scaler(router, args)
            cal_risks = np.asarray([risk_score(r, args, scaler) for r in cal], dtype=np.float64)
            test_risks = np.asarray([risk_score(r, args, scaler) for r in test], dtype=np.float64)
            for gamma in args.gammas:
                chosen = choose_frac(router, gamma, args, scaler)
                thr = chosen["threshold"]
                q_low = fit_q(cal, cal_risks < thr, False, gamma, args)
                q_high = fit_q(cal, cal_risks >= thr, True, gamma, args)
                gated = eval_rows(test, test_risks, thr, q_low, q_high, args)
                orig = baseline_cp(cal, test, "original", gamma, args)
                mv = baseline_cp(cal, test, "multiview", gamma, args)
                mode_runs.append({
                    "seed": seed,
                    "gamma": gamma,
                    "chosen_frac": chosen["frac"],
                    "threshold": thr,
                    "q_low": q_low,
                    "q_high": q_high,
                    "gated": gated,
                    "original_cp": orig,
                    "multiview_cp": mv,
                    "delta_vs_original": {
                        "coverage": none_sub(gated["coverage"], orig["coverage"]),
                        "avg_set_size": none_sub(gated["avg_set_size"], orig["avg_set_size"]),
                        "singleton_rate": none_sub(gated["singleton_rate"], orig["singleton_rate"]),
                        "singleton_accuracy": none_sub(gated["singleton_accuracy"], orig["singleton_accuracy"]),
                    },
                    "delta_vs_multiview": {
                        "coverage": none_sub(gated["coverage"], mv["coverage"]),
                        "avg_set_size": none_sub(gated["avg_set_size"], mv["avg_set_size"]),
                    },
                })
        out["modes"][mode] = {"runs": mode_runs, "summary": summarize_runs(mode_runs, args.gammas)}
    return out


def none_sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def baseline_cp(cal: list[dict[str, Any]], test: list[dict[str, Any]], mode: str, gamma: float, args: argparse.Namespace) -> dict[str, Any]:
    scores = []
    for r in cal:
        p = original_probs(r) if mode == "original" else multiview_probs(r, args)
        scores.append(1.0 - float(p[int(r["gt_index"])]))
    q = cp_quantile(scores, gamma)
    return eval_policy_with_probs(test, mode, q, args)


def summarize_runs(runs: list[dict[str, Any]], gammas: list[float]) -> dict[str, Any]:
    out = {}
    for gamma in gammas:
        rs = [r for r in runs if float(r["gamma"]) == float(gamma)]
        out[str(gamma)] = {
            "chosen_frac": summarize([r["chosen_frac"] for r in rs]),
            "gated_coverage": summarize([r["gated"]["coverage"] for r in rs]),
            "original_coverage": summarize([r["original_cp"]["coverage"] for r in rs]),
            "multiview_coverage": summarize([r["multiview_cp"]["coverage"] for r in rs]),
            "delta_coverage_vs_original": summarize([r["delta_vs_original"]["coverage"] for r in rs]),
            "delta_size_vs_original": summarize([r["delta_vs_original"]["avg_set_size"] for r in rs]),
            "delta_coverage_vs_multiview": summarize([r["delta_vs_multiview"]["coverage"] for r in rs]),
            "gated_size": summarize([r["gated"]["avg_set_size"] for r in rs]),
            "original_size": summarize([r["original_cp"]["avg_set_size"] for r in rs]),
            "high_risk_rate": summarize([r["gated"]["high_risk_rate"] for r in rs]),
            "singleton_accuracy": summarize([r["gated"]["singleton_accuracy"] for r in rs]),
            "rescues": summarize([float(r["gated"]["rescues"]) for r in rs]),
            "harmful": summarize([float(r["gated"]["harmful"]) for r in rs]),
        }
    return out


def main() -> None:
    args = parse_args()
    results = [analyze_path(p, args) for p in args.paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    for res in results:
        print(f"\n== {res['path']} ==")
        for mode, md in res["modes"].items():
            print(f"mode={mode}")
            for gamma, s in md["summary"].items():
                dc = s["delta_coverage_vs_original"]
                ds = s["delta_size_vs_original"]
                gc = s["gated_coverage"]
                oc = s["original_coverage"]
                print(
                    f"  gamma={gamma}: gated_cov={100*(gc['mean'] or 0):.2f} vs orig={100*(oc['mean'] or 0):.2f} "
                    f"delta={100*(dc['mean'] or 0):+.2f}±{100*(dc['std'] or 0):.2f}pp "
                    f"size_delta={ds['mean']:+.3f} high={100*((s['high_risk_rate']['mean']) or 0):.1f}%"
                )


if __name__ == "__main__":
    main()
