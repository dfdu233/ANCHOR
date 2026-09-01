#!/usr/bin/env python3
"""Leakage-controlled phase-1 test for the currently available C3-Guard features.

This script deliberately evaluates only features that really exist in the frozen
128-case Huatuo style experiment.  It does not synthesize grounding, lesion-mask,
or HALP/source features.  Each row is one patient, and every reported risk score
is out-of-fold under repeated nested patient-level cross-validation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "c3-guard-available-features-v1"
TRANSFORMS = ("lf_vqarad_l0.01_sr0.8", "gamma_0.9", "gamma_1.1")


def binary_entropy_from_margin(margin: float) -> float:
    p = 1.0 / (1.0 + math.exp(-margin))
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 128:
        raise ValueError(f"expected frozen n=128, got {len(rows)}")
    if len({row["patient_id"] for row in rows}) != len(rows):
        raise ValueError("frozen set is not one-row-per-patient")
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError(f"non-ok row: {row.get('question_id')}")
        missing = [name for name in ("original", *TRANSFORMS) if name not in row["scores"]]
        if missing:
            raise ValueError(f"missing views for {row.get('question_id')}: {missing}")
    return rows


def build_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    margins = np.asarray([row["scores"]["original"]["yes_minus_no"] for row in rows])
    abs_margin = np.abs(margins)
    entropy = np.asarray([binary_entropy_from_margin(float(m)) for m in margins])
    transformed = np.asarray(
        [[row["scores"][name]["yes_minus_no"] for name in TRANSFORMS] for row in rows]
    )
    domain_variance = np.var(transformed, axis=1, ddof=1)
    y = np.asarray(
        [
            int(row["scores"]["original"]["prediction"].lower() != row["ground_truth"].lower())
            for row in rows
        ],
        dtype=int,
    )
    groups = np.asarray([row["patient_id"] for row in rows])
    # Risk features must not use the ground-truth polarity.  Confidence is
    # therefore represented by |Yes-No|, not by a truth-aligned signed margin.
    x_base = np.column_stack([abs_margin, entropy])
    x_domain = np.column_stack([abs_margin, entropy, domain_variance])
    audit = {
        "n": len(rows),
        "n_patients": int(len(set(groups))),
        "n_errors": int(y.sum()),
        "n_correct": int((1 - y).sum()),
        "error_prevalence": float(y.mean()),
        "ground_truth_yes": int(sum(row["ground_truth"].lower() == "yes" for row in rows)),
        "ground_truth_no": int(sum(row["ground_truth"].lower() == "no" for row in rows)),
        "feature_definitions": {
            "abs_margin": "absolute original-image FP32 Yes-minus-No logit margin",
            "binary_entropy": "entropy of sigmoid(original Yes-minus-No margin)",
            "domain_variance": "sample variance of Yes-minus-No margins over the three frozen transforms",
        },
        "entropy_abs_margin_correlation": float(np.corrcoef(abs_margin, entropy)[0, 1]),
        "domain_variance_summary": {
            "mean": float(domain_variance.mean()),
            "median": float(np.median(domain_variance)),
            "max": float(domain_variance.max()),
        },
    }
    return x_base, x_domain, y, groups, audit


def nested_oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int = 10,
    folds: int = 5,
) -> tuple[np.ndarray, list[dict]]:
    cs = (0.01, 0.1, 1.0, 10.0)
    accumulated = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=int)
    fold_records: list[dict] = []
    for repeat in range(repeats):
        outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=9100 + repeat)
        for outer_fold, (train, test) in enumerate(outer.split(x, y, groups)):
            inner = StratifiedGroupKFold(
                n_splits=4, shuffle=True, random_state=19100 + repeat * 10 + outer_fold
            )
            c_scores: dict[float, list[float]] = {c: [] for c in cs}
            for inner_train_local, valid_local in inner.split(
                x[train], y[train], groups[train]
            ):
                inner_train = train[inner_train_local]
                valid = train[valid_local]
                for c in cs:
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=c, class_weight="balanced", max_iter=2000),
                    )
                    model.fit(x[inner_train], y[inner_train])
                    pred = model.predict_proba(x[valid])[:, 1]
                    c_scores[c].append(float(average_precision_score(y[valid], pred)))
            best_c = max(cs, key=lambda c: (float(np.mean(c_scores[c])), -c))
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000),
            )
            model.fit(x[train], y[train])
            pred = model.predict_proba(x[test])[:, 1]
            accumulated[test] += pred
            counts[test] += 1
            fold_records.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "best_c": best_c,
                    "inner_mean_ap": {str(c): float(np.mean(v)) for c, v in c_scores.items()},
                }
            )
    if not np.all(counts == repeats):
        raise RuntimeError(f"OOF coverage error: {np.unique(counts, return_counts=True)}")
    return accumulated / counts, fold_records


def selective_metrics(y: np.ndarray, risk: np.ndarray, coverage: float = 0.70) -> dict:
    n_keep = int(math.ceil(coverage * len(y)))
    order = np.argsort(risk, kind="stable")
    keep = order[:n_keep]
    remove = order[n_keep:]
    base_risk = float(y.mean())
    retained_risk = float(y[keep].mean())
    corrections = int(y[remove].sum())
    harms = int((1 - y[remove]).sum())
    return {
        "requested_coverage": coverage,
        "realized_coverage": float(n_keep / len(y)),
        "retained_n": n_keep,
        "removed_n": int(len(remove)),
        "base_error_risk": base_risk,
        "retained_error_risk": retained_risk,
        "relative_risk_reduction": float((base_risk - retained_risk) / base_risk),
        "errors_removed_corrections": corrections,
        "correct_cases_removed_harms": harms,
        "correction_harm_ratio": float(corrections / harms) if harms else None,
    }


def bootstrap(
    y: np.ndarray,
    base: np.ndarray,
    domain: np.ndarray,
    *,
    draws: int = 5000,
) -> dict:
    rng = np.random.default_rng(20260806)
    ap_delta = []
    risk_delta = []
    n = len(y)
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        ap_delta.append(
            average_precision_score(y[idx], domain[idx])
            - average_precision_score(y[idx], base[idx])
        )
        b = selective_metrics(y[idx], base[idx])["retained_error_risk"]
        d = selective_metrics(y[idx], domain[idx])["retained_error_risk"]
        risk_delta.append(b - d)
    def ci(v: list[float]) -> dict:
        a = np.asarray(v)
        return {
            "mean": float(a.mean()),
            "lower_95": float(np.quantile(a, 0.025)),
            "upper_95": float(np.quantile(a, 0.975)),
        }
    return {"draws_used": len(ap_delta), "ap_delta": ci(ap_delta), "retained_risk_reduction": ci(risk_delta)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--source-score", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.raw)
    x_base, x_domain, y, groups, audit = build_arrays(rows)
    base_oof, base_folds = nested_oof(x_base, y, groups)
    domain_oof, domain_folds = nested_oof(x_domain, y, groups)
    source_record = None
    source_folds = None
    final_name = "margin_entropy_domain_stability"
    final_oof = domain_oof
    if args.source_score:
        source_payload = json.loads(args.source_score.read_text())
        score_map = {
            str(row["question_id"]): float(row["source_logit"])
            for row in source_payload["target_scores"]
        }
        expected_ids = [str(row["question_id"]) for row in rows]
        if set(score_map) != set(expected_ids):
            raise ValueError("external source-score IDs do not exactly match the frozen cohort")
        source_values = np.asarray([score_map[qid] for qid in expected_ids])
        x_source = np.column_stack([x_domain, source_values])
        source_oof, source_folds = nested_oof(x_source, y, groups)
        source_record = {
            "features": ["abs_margin", "binary_entropy", "domain_variance", "external_source_logit"],
            "oof_auprc": float(average_precision_score(y, source_oof)),
            "selective_metrics": selective_metrics(y, source_oof),
            "source_score_provenance": str(args.source_score.resolve()),
            "source_score_target_labels_read_during_fit": source_payload.get("target_outcome_labels_read"),
        }
        final_name = "margin_entropy_domain_stability_external_source"
        final_oof = source_oof
    base_ap = float(average_precision_score(y, base_oof))
    domain_ap = float(average_precision_score(y, domain_oof))
    final_ap = float(average_precision_score(y, final_oof))
    boot = bootstrap(y, base_oof, final_oof)
    base_sel = selective_metrics(y, base_oof)
    domain_sel = selective_metrics(y, domain_oof)
    final_sel = selective_metrics(y, final_oof)
    incremental_risk_reduction = (
        (base_sel["retained_error_risk"] - final_sel["retained_error_risk"])
        / base_sel["retained_error_risk"]
    )
    gates = {
        "ap_delta_at_least_0.05": final_ap - base_ap >= 0.05,
        "bootstrap_ap_delta_lower_above_zero": boot["ap_delta"]["lower_95"] > 0,
        "same_coverage_risk_reduction_vs_baseline_at_least_0.25": (
            incremental_risk_reduction >= 0.25
        ),
        "available_model_correction_harm_above_2": (
            final_sel["correction_harm_ratio"] is not None
            and final_sel["correction_harm_ratio"] > 2
        ),
        "worst_site_and_rare_claim_noninferiority": None,
    }
    result = {
        "version": VERSION,
        "status": "partial_available_feature_falsification",
        "audit": audit,
        "protocol": {
            "target": "original-answer error (not a full OE hallucination label)",
            "outer_cv": "10 repeats of 5-fold StratifiedGroupKFold by patient",
            "inner_cv": "4-fold StratifiedGroupKFold; C in [0.01, 0.1, 1, 10]",
            "learner": "standardized class-balanced logistic regression",
            "all_reported_predictions": "averaged out-of-fold risk only",
            "bootstrap": "5000 paired patient resamples of fixed OOF predictions",
            "coverage": 0.70,
        },
        "models": {
            "margin_entropy": {
                "features": ["abs_margin", "binary_entropy"],
                "oof_auprc": base_ap,
                "selective_metrics": base_sel,
            },
            "margin_entropy_domain_stability": {
                "features": ["abs_margin", "binary_entropy", "domain_variance"],
                "oof_auprc": domain_ap,
                "selective_metrics": domain_sel,
            },
            **({final_name: source_record} if source_record is not None else {}),
        },
        "comparison": {
            "final_available_model": final_name,
            "domain_only_oof_auprc_delta_vs_baseline": domain_ap - base_ap,
            "oof_auprc_delta": final_ap - base_ap,
            "same_coverage_relative_risk_reduction_vs_baseline": incremental_risk_reduction,
            "bootstrap": boot,
            "gates": gates,
            "available_feature_go": bool(all(v is True for v in gates.values() if v is not None)),
            "full_c3_go": False,
            "full_c3_go_reason": (
                "grounding, lesion sensitivity, site, and rare-claim strata are absent on this cohort; "
                "the available source score is global and externally trained, not claim-conditional"
            ),
        },
        "fold_selection": {
            "baseline": base_folds,
            "with_domain": domain_folds,
            **({"with_domain_and_source": source_folds} if source_folds is not None else {}),
        },
        "rows": [
            {
                "question_id": row["question_id"],
                "patient_id": row["patient_id"],
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "prediction": row["scores"]["original"]["prediction"],
                "error": int(y[i]),
                "baseline_oof_risk": float(base_oof[i]),
                "domain_oof_risk": float(domain_oof[i]),
                **(
                    {"domain_source_oof_risk": float(final_oof[i])}
                    if source_record is not None
                    else {}
                ),
            }
            for i, row in enumerate(rows)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "audit": audit, "comparison": result["comparison"], "models": result["models"]}, indent=2))


if __name__ == "__main__":
    main()
