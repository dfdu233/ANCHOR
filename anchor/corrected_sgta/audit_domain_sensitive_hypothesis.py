"""Audit the domain-sensitive hallucination hypothesis.

This script turns the current SGTA-CP story into measurable evidence:

Domain-sensitive hallucination (operational):
  A sample/task is domain-sensitive if low-distortion native-aligned views cause
  non-trivial prediction-distribution instability and this instability predicts
  original errors or view-induced harmful changes.

Knowledge/semantic hallucination (operational contrast):
  Errors that are not affected by visual native perturbations: view predictions
  remain stable, no rescue/harmful trade-off exists, and SGTA-CP does not improve
  conformal coverage.  These failures are more likely due to language/medical
  concept priors, Yes/No interface bias, or calibration noise than visual DG.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.analyze_risk_gated_multiview_cp import (
    multiview_probs,
    original_probs,
    row_features,
)
from corrected_sgta.validate_sgta_cp_protocol import fit_risk_scaler, risk_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    p.add_argument("--fixed-risk", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("max", "laplacian"), default="max")
    p.add_argument("--gamma", type=str, default="0.9")
    p.add_argument("--max-views", type=int, default=4)
    p.add_argument("--min-edge", type=float, default=0.85)
    p.add_argument("--min-psnr", type=float, default=18.0)
    return p.parse_args()


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    for v in pos:
        wins += float((v > neg).sum()) + 0.5 * float((v == neg).sum())
    return float(wins / (len(pos) * len(neg)))


def mean(xs: list[float | bool]) -> float | None:
    if not xs:
        return None
    return float(np.mean([float(x) for x in xs]))


def get_fixed_summary(fixed: dict[str, Any], path: str, mode: str, gamma: str) -> dict[str, Any] | None:
    for res in fixed.get("results", []):
        if res.get("path") == path or Path(res.get("path", "")).name == Path(path).name:
            return res.get("modes", {}).get(mode, {}).get("summary", {}).get(gamma)
    return None


def summarize_task(name: str, path: Path, fixed: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    train = [r for r in rows if r.get("split") == "train"]
    test = [r for r in rows if r.get("split") == "test"]

    scaler = fit_risk_scaler(train, args)
    risks = [risk_score(r, args, scaler) for r in test]
    orig_wrong = []
    mv_changed = []
    harmful = []
    rescue = []
    js_vals = []
    damage_vals = []
    uncertainty_vals = []
    for r in test:
        y = int(r["gt_index"])
        op = original_probs(r)
        mp = multiview_probs(r, args)
        orig_pred = int(np.argmax(op))
        mv_pred = int(np.argmax(mp))
        f = row_features(r, args)
        orig_wrong.append(orig_pred != y)
        mv_changed.append(mv_pred != orig_pred)
        harmful.append(orig_pred == y and mv_pred != y)
        rescue.append(orig_pred != y and mv_pred == y)
        js_vals.append(float(f["view_js"]))
        damage_vals.append(float(f["mean_damage"]))
        uncertainty_vals.append(float(f["orig_uncertainty"]))

    fixed_summary = get_fixed_summary(fixed, str(path), args.mode, args.gamma)
    if fixed_summary is None:
        # path in fixed file is often relative; try suffix matching.
        for res in fixed.get("results", []):
            if str(path).endswith(res.get("path", "")) or res.get("path", "").endswith(str(path.name)):
                fixed_summary = res.get("modes", {}).get(args.mode, {}).get("summary", {}).get(args.gamma)
                break

    delta_cov = None
    delta_size = None
    if fixed_summary:
        dc = fixed_summary.get("delta_coverage_vs_original", {})
        ds = fixed_summary.get("delta_size_vs_original", {})
        delta_cov = dc.get("mean")
        delta_size = ds.get("mean")

    evidence = {
        "risk_predicts_original_error": auroc(risks, orig_wrong),
        "risk_predicts_view_change": auroc(risks, mv_changed),
        "risk_predicts_harmful_flip": auroc(risks, harmful),
        "view_js_predicts_original_error": auroc(js_vals, orig_wrong),
        "damage_predicts_harmful_flip": auroc(damage_vals, harmful),
        "original_uncertainty_predicts_error": auroc(uncertainty_vals, orig_wrong),
        "original_error_rate": mean(orig_wrong),
        "view_change_rate": mean(mv_changed),
        "rescue_rate": mean(rescue),
        "harmful_rate": mean(harmful),
        "mean_view_js": mean(js_vals),
        "mean_damage": mean(damage_vals),
        "sgta_cp_delta_coverage_vs_original": delta_cov,
        "sgta_cp_delta_size_vs_original": delta_size,
    }

    # Conservative rule: a task is supported only when risk predicts errors and
    # SGTA-CP gives positive coverage delta.  View changes/trade-off are strong
    # mechanistic evidence but may be rare for max/laplacian point predictions.
    supported = (
        (evidence["risk_predicts_original_error"] is not None and evidence["risk_predicts_original_error"] >= 0.65)
        and (delta_cov is not None and delta_cov > 0.0)
    )
    boundary = (
        (evidence["risk_predicts_original_error"] is not None and evidence["risk_predicts_original_error"] >= 0.60)
        and not supported
    )
    if supported:
        label = "domain_sensitive_supported"
    elif boundary:
        label = "risk_signal_without_method_gain"
    else:
        label = "not_domain_sensitive_under_current_probe"

    return {
        "name": name,
        "path": str(path),
        "n_train": len(train),
        "n_test": len(test),
        "mode": args.mode,
        "gamma": args.gamma,
        "evidence": evidence,
        "classification": label,
        "fixed_risk_summary": fixed_summary,
    }


def main() -> None:
    args = parse_args()
    fixed = json.loads(args.fixed_risk.read_text())
    results = [summarize_task(name, Path(path), fixed, args) for name, path in args.cache]
    out = {
        "hypothesis": {
            "domain_sensitive": "native-view instability predicts original errors and SGTA-CP improves coverage",
            "knowledge_or_semantic": "errors remain stable under native views or SGTA-CP does not improve coverage",
            "important_caveat": "This is an operational audit, not a proof that the true VLM training distribution center is recovered.",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in results:
        e = r["evidence"]
        print(
            f"{r['name']}: {r['classification']} | "
            f"risk-error AUROC={fmt(e['risk_predicts_original_error'])}, "
            f"view-change={fmt(e['risk_predicts_view_change'])}, "
            f"harmful={fmt(e['risk_predicts_harmful_flip'])}, "
            f"SGTA-CP Δcov={fmt(e['sgta_cp_delta_coverage_vs_original'], pct=True)}, "
            f"Δsize={fmt(e['sgta_cp_delta_size_vs_original'])}"
        )


def fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "NA"
    if pct:
        return f"{100*x:+.2f}pp"
    return f"{x:.3f}"


if __name__ == "__main__":
    main()
