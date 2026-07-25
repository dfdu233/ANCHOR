"""Risk-gated native multi-view conformal prediction.

This is an offline analyzer for cached outputs produced by
``corrected_sgta.native_view_projection``.  It tests a compact paper-friendly
algorithm:

  1. Generate a small set of native-aligned views.
  2. Use a label-free risk score to route each sample.
  3. Low-risk samples use original-image CP; high-risk samples use multi-view CP.

The router is selected on a router split, while conformal quantiles are fitted
on a disjoint calibration split.  This keeps the pilot close to the intended
protocol and prevents the gate from consuming the same labels used for qhat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--gammas", type=float, nargs="*", default=(0.8, 0.9, 0.95))
    p.add_argument("--mode", choices=("max", "laplacian"), default="max")
    p.add_argument("--max-views", type=int, default=4)
    p.add_argument("--min-edge", type=float, default=0.85)
    p.add_argument("--min-psnr", type=float, default=18.0)
    p.add_argument("--router-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-high-risk-frac", type=float, default=0.05)
    p.add_argument("--max-high-risk-frac", type=float, default=0.60)
    return p.parse_args()


def stable_u01(key: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(16**16 - 1)


def softmax(logits: list[float] | np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    return float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())


def js_divergence(ps: list[np.ndarray]) -> float:
    if len(ps) <= 1:
        return 0.0
    stack = np.stack(ps, axis=0)
    m = stack.mean(axis=0)
    return float(np.mean([np.sum(p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(np.clip(m, 1e-12, 1.0)))) for p in stack]))


def finite_psnr(v: Any) -> float:
    if v is None:
        return 99.0
    try:
        x = float(v)
    except Exception:
        return 0.0
    return 99.0 if math.isinf(x) else x


def candidate_quality(c: dict[str, Any]) -> tuple[float, float, float, float, float]:
    st = c.get("structure") or {}
    edge = float(st.get("edge_correlation", 0.0))
    psnr = finite_psnr(st.get("psnr"))
    damage = (1.0 - edge) + 0.02 * max(0.0, 45.0 - psnr)
    closure = float(c.get("native_closure") or 0.0)
    dist = float(c.get("native_distance") or 0.0)
    return closure, dist, edge, psnr, damage


def safe_indices(row: dict[str, Any], max_views: int, min_edge: float, min_psnr: float) -> list[int]:
    idxs: list[tuple[float, float, int]] = []
    for i, c in enumerate(row.get("candidates", [])[1:], start=1):
        closure, dist, edge, psnr, _ = candidate_quality(c)
        if not c.get("structure_safe", False):
            continue
        if edge < min_edge or psnr < min_psnr:
            continue
        idxs.append((closure, -dist, i))
    idxs.sort(reverse=True)
    return [i for _, __, i in idxs[:max_views]]


def original_probs(row: dict[str, Any]) -> np.ndarray:
    return softmax(row["candidates"][0]["logits"])


def multiview_probs(row: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    if args.mode == "laplacian" and row.get("laplacian_fusion_probs") is not None:
        return np.asarray(row["laplacian_fusion_probs"], dtype=np.float64)
    idxs = [0] + safe_indices(row, args.max_views, args.min_edge, args.min_psnr)
    ps = np.stack([softmax(row["candidates"][i]["logits"]) for i in idxs], axis=0)
    return np.max(ps, axis=0)


def cp_quantile(scores: list[float], gamma: float) -> float:
    if not scores:
        return 1.0
    xs = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(math.ceil((len(xs) + 1) * gamma)) - 1
    k = min(max(k, 0), len(xs) - 1)
    return float(xs[k])


def cp_set(p: np.ndarray, qhat: float) -> list[int]:
    return [int(i) for i, pi in enumerate(p) if 1.0 - float(pi) <= qhat]


@dataclass
class Gate:
    feature: str
    threshold: float
    high_if: str

    def is_high(self, feats: dict[str, float]) -> bool:
        value = feats[self.feature]
        if self.high_if == ">=":
            return value >= self.threshold
        return value <= self.threshold


def row_features(row: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    cands = row.get("candidates", [])
    op = original_probs(row)
    o_pred = int(np.argmax(op))
    idxs = [0] + safe_indices(row, args.max_views, args.min_edge, args.min_psnr)
    view_ps = [softmax(cands[i]["logits"]) for i in idxs]
    view_preds = [int(np.argmax(p)) for p in view_ps]
    qualities = [candidate_quality(cands[i]) for i in idxs[1:]]
    closures = [q[0] for q in qualities]
    damages = [q[4] for q in qualities]
    margin = float(np.sort(op)[-1] - np.sort(op)[-2]) if len(op) > 1 else float(op.max())
    mvp = multiview_probs(row, args)
    return {
        "orig_uncertainty": 1.0 - float(op.max()),
        "orig_entropy": entropy(op),
        "orig_margin_neg": -margin,
        "native_distance": float(row.get("original_distance") or 0.0),
        "view_js": js_divergence(view_ps),
        "disagree_frac": float(np.mean([p != o_pred for p in view_preds[1:]])) if len(view_preds) > 1 else 0.0,
        "max_closure": float(max(closures)) if closures else 0.0,
        "mean_closure": float(np.mean(closures)) if closures else 0.0,
        "max_damage": float(max(damages)) if damages else 0.0,
        "mean_damage": float(np.mean(damages)) if damages else 0.0,
        "mv_conf_gain": float(mvp.max() - op.max()),
        "accepted_view_count": float(len(idxs) - 1),
    }


def split_router_cal(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [r for r in rows if r.get("split") == "train"]
    pairs = []
    for r in train:
        key = str(r.get("qid", r.get("img_name", len(pairs))))
        pairs.append((stable_u01(key, args.seed), r))
    pairs.sort(key=lambda x: x[0])
    n_router = max(1, min(len(pairs) - 1, int(round(len(pairs) * args.router_frac)))) if len(pairs) >= 2 else len(pairs)
    return [r for _, r in pairs[:n_router]], [r for _, r in pairs[n_router:]]


def eval_policy(rows: list[dict[str, Any]], gate: Gate | None, q_low: float, q_high: float, args: argparse.Namespace) -> dict[str, Any]:
    details = []
    for r in rows:
        feats = row_features(r, args)
        high = bool(gate.is_high(feats)) if gate is not None else False
        p = multiview_probs(r, args) if high else original_probs(r)
        qhat = q_high if high else q_low
        s = cp_set(p, qhat)
        y = int(r["gt_index"])
        pred = int(np.argmax(p))
        orig_pred = int(np.argmax(original_probs(r)))
        details.append({
            "qid": str(r.get("qid")),
            "high_risk": high,
            "covered": y in s,
            "set_size": len(s),
            "singleton": len(s) == 1,
            "abstain": len(s) != 1,
            "pred": pred,
            "orig_pred": orig_pred,
            "gt": y,
            "point_correct": pred == y,
            "orig_correct": orig_pred == y,
            "singleton_correct": (pred == y) if len(s) == 1 else None,
        })
    singleton = [d for d in details if d["singleton"]]
    changed = [d for d in details if d["pred"] != d["orig_pred"]]
    return {
        "n": len(details),
        "coverage": mean([d["covered"] for d in details]),
        "avg_set_size": mean([d["set_size"] for d in details]),
        "singleton_rate": mean([d["singleton"] for d in details]),
        "abstain_rate": mean([d["abstain"] for d in details]),
        "singleton_accuracy": mean([d["singleton_correct"] for d in singleton]),
        "point_accuracy": mean([d["point_correct"] for d in details]),
        "orig_point_accuracy": mean([d["orig_correct"] for d in details]),
        "high_risk_rate": mean([d["high_risk"] for d in details]),
        "changed_count": len(changed),
        "rescues": int(sum((not d["orig_correct"]) and d["point_correct"] for d in changed)),
        "harmful": int(sum(d["orig_correct"] and (not d["point_correct"]) for d in changed)),
    }


def mean(xs: list[Any]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return float(np.mean(vals)) if vals else None


def fit_q(rows: list[dict[str, Any]], high: bool, gate: Gate | None, gamma: float, args: argparse.Namespace) -> float:
    scores = []
    for r in rows:
        feats = row_features(r, args)
        is_high = bool(gate.is_high(feats)) if gate is not None else False
        if is_high != high:
            continue
        p = multiview_probs(r, args) if high else original_probs(r)
        scores.append(1.0 - float(p[int(r["gt_index"])]))
    return cp_quantile(scores, gamma)


def global_cp(rows_cal: list[dict[str, Any]], rows_test: list[dict[str, Any]], mode: str, gamma: float, args: argparse.Namespace) -> dict[str, Any]:
    scores = []
    for r in rows_cal:
        p = original_probs(r) if mode == "original" else multiview_probs(r, args)
        scores.append(1.0 - float(p[int(r["gt_index"])]))
    q = cp_quantile(scores, gamma)
    old_mode = args.mode
    out = eval_policy_with_probs(rows_test, mode, q, args)
    out["qhat"] = q
    out["mode"] = mode
    args.mode = old_mode
    return out


def eval_policy_with_probs(rows: list[dict[str, Any]], mode: str, qhat: float, args: argparse.Namespace) -> dict[str, Any]:
    details = []
    for r in rows:
        p = original_probs(r) if mode == "original" else multiview_probs(r, args)
        s = cp_set(p, qhat)
        y = int(r["gt_index"])
        pred = int(np.argmax(p))
        details.append({
            "covered": y in s,
            "set_size": len(s),
            "singleton": len(s) == 1,
            "point_correct": pred == y,
            "singleton_correct": (pred == y) if len(s) == 1 else None,
        })
    singleton = [d for d in details if d["singleton"]]
    return {
        "n": len(details),
        "coverage": mean([d["covered"] for d in details]),
        "avg_set_size": mean([d["set_size"] for d in details]),
        "singleton_rate": mean([d["singleton"] for d in details]),
        "abstain_rate": mean([not d["singleton"] for d in details]),
        "singleton_accuracy": mean([d["singleton_correct"] for d in singleton]),
        "point_accuracy": mean([d["point_correct"] for d in details]),
    }


def select_gate(router_rows: list[dict[str, Any]], gamma: float, args: argparse.Namespace) -> dict[str, Any]:
    feats = [row_features(r, args) for r in router_rows]
    names = sorted(feats[0].keys()) if feats else []
    best = None
    candidates = []
    for name in names:
        values = np.asarray([f[name] for f in feats], dtype=np.float64)
        if not np.isfinite(values).all() or float(values.max() - values.min()) == 0.0:
            continue
        thresholds = np.unique(np.quantile(values, np.linspace(0.05, 0.95, 19)))
        for high_if in (">=", "<="):
            for t in thresholds:
                gate = Gate(name, float(t), high_if)
                high_rate = mean([gate.is_high(f) for f in feats]) or 0.0
                if high_rate < args.min_high_risk_frac or high_rate > args.max_high_risk_frac:
                    continue
                q_low = fit_q(router_rows, False, gate, gamma, args)
                q_high = fit_q(router_rows, True, gate, gamma, args)
                m = eval_policy(router_rows, gate, q_low, q_high, args)
                original_errors = [int(np.argmax(original_probs(r))) != int(r["gt_index"]) for r in router_rows]
                high_flags = [gate.is_high(f) for f in feats]
                err_recall = float(sum(e and h for e, h in zip(original_errors, high_flags)) / max(1, sum(original_errors)))
                err_precision = float(sum(e and h for e, h in zip(original_errors, high_flags)) / max(1, sum(high_flags)))
                feasible = (m["coverage"] is not None and m["coverage"] >= gamma)
                score = (
                    (0 if feasible else -1000)
                    - 2.0 * float(m["avg_set_size"] or 9.0)
                    + 1.0 * err_recall
                    + 0.5 * float(m["singleton_rate"] or 0.0)
                    - 0.5 * max(0.0, float(m["harmful"]) - float(m["rescues"]))
                )
                item = {
                    "gate": gate,
                    "router_metrics": m,
                    "router_error_recall": err_recall,
                    "router_error_precision": err_precision,
                    "q_low_router": q_low,
                    "q_high_router": q_high,
                    "score": score,
                    "feasible": feasible,
                }
                candidates.append(item)
                if best is None or score > best["score"]:
                    best = item
    if best is None:
        gate = Gate("orig_uncertainty", 0.0, ">=")
        best = {"gate": gate, "router_metrics": {}, "router_error_recall": None, "router_error_precision": None, "score": None, "feasible": False}
    gate = best["gate"]
    return {
        "feature": gate.feature,
        "threshold": gate.threshold,
        "high_if": gate.high_if,
        "router_metrics": best.get("router_metrics"),
        "router_error_recall": best.get("router_error_recall"),
        "router_error_precision": best.get("router_error_precision"),
        "router_score": best.get("score"),
        "router_feasible": best.get("feasible"),
        "num_gate_candidates": len(candidates),
    }


def analyze_one(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    router_rows, cal_rows = split_router_cal(rows, args)
    test_rows = [r for r in rows if r.get("split") == "test"]
    out: dict[str, Any] = {
        "path": str(path),
        "summary": payload.get("summary", {}),
        "n_total": len(rows),
        "n_router": len(router_rows),
        "n_cal": len(cal_rows),
        "n_test": len(test_rows),
        "mode": args.mode,
        "max_views": args.max_views,
        "gammas": {},
    }
    for gamma in args.gammas:
        gate_info = select_gate(router_rows, gamma, args)
        gate = Gate(gate_info["feature"], gate_info["threshold"], gate_info["high_if"])
        q_low = fit_q(cal_rows, False, gate, gamma, args)
        q_high = fit_q(cal_rows, True, gate, gamma, args)
        gated_cal = eval_policy(cal_rows, gate, q_low, q_high, args)
        gated_test = eval_policy(test_rows, gate, q_low, q_high, args)
        out["gammas"][str(gamma)] = {
            "gate": gate_info,
            "q_low": q_low,
            "q_high": q_high,
            "cal": gated_cal,
            "test": gated_test,
            "baselines": {
                "original_cp": global_cp(cal_rows, test_rows, "original", gamma, args),
                "multiview_cp": global_cp(cal_rows, test_rows, "multiview", gamma, args),
            },
        }
    return out


def main() -> None:
    args = parse_args()
    results = [analyze_one(p, args) for p in args.paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    for r in results:
        print(f"\n== {r['path']} ==")
        print(f"n router/cal/test = {r['n_router']}/{r['n_cal']}/{r['n_test']}  mode={r['mode']}")
        for gamma, g in r["gammas"].items():
            gate = g["gate"]
            test = g["test"]
            orig = g["baselines"]["original_cp"]
            mv = g["baselines"]["multiview_cp"]
            print(
                f"gamma={gamma} gate={gate['feature']} {gate['high_if']} {gate['threshold']:.6g} "
                f"q=({g['q_low']:.4f},{g['q_high']:.4f})"
            )
            print(
                "  gated  cov={:.2f} size={:.3f} single={:.2f} single_acc={} high={:.2f} rescue/harm={}/{}".format(
                    100 * (test["coverage"] or 0.0),
                    test["avg_set_size"] or 0.0,
                    100 * (test["singleton_rate"] or 0.0),
                    None if test["singleton_accuracy"] is None else round(100 * test["singleton_accuracy"], 2),
                    100 * (test["high_risk_rate"] or 0.0),
                    test["rescues"],
                    test["harmful"],
                )
            )
            print(
                "  origCP cov={:.2f} size={:.3f} single={:.2f}; mvCP cov={:.2f} size={:.3f} single={:.2f}".format(
                    100 * (orig["coverage"] or 0.0),
                    orig["avg_set_size"] or 0.0,
                    100 * (orig["singleton_rate"] or 0.0),
                    100 * (mv["coverage"] or 0.0),
                    mv["avg_set_size"] or 0.0,
                    100 * (mv["singleton_rate"] or 0.0),
                )
            )


if __name__ == "__main__":
    main()
