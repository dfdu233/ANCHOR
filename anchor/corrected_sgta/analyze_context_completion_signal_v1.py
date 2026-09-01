#!/usr/bin/env python3
"""Post-hoc audit of context-completion signal in the observation-policy probe.

The frozen candidate is

    delta_ctx = margin(zoom_true_context_panel) - margin(zoom_sham_panel)

under the neutral prompt.  The two panels share the same main ROI and layout;
only the small context panel contains the true radiograph versus a
phase-scrambled version.  This script deliberately does *not* call the result a
confirmation: the Huatuo confirmation set had already been inspected before
this candidate was proposed.

No external ML dependency is used.  A small L2-logistic model is fitted by
Newton/IRLS within repeated, finding-and-label-stratified cross-fitting folds.
All reported intervals use an image bootstrap stratified by finding and label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VERSION = "context-completion-signal-v1"
SEED = 20260812
PROMPTS = ("neutral", "random_provenance", "suspicious_provenance")
RENDERS = (
    "full",
    "native_context_removed",
    "native_sham_panel",
    "zoom_sham_panel",
    "zoom_true_context_panel",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    """Binary AUROC with average ranks for tied scores."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positive = int(y.sum())
    negative = len(y) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    prediction = probability > 0.5
    positive = y == 1
    negative = ~positive
    return {
        "auroc": float(auc_score(y, probability)),
        "nll": float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))),
        "brier": float(np.mean((probability - y) ** 2)),
        "accuracy": float(np.mean(prediction == y)),
        "sensitivity": float(np.mean(prediction[positive])) if positive.any() else float("nan"),
        "specificity": float(np.mean(~prediction[negative])) if negative.any() else float("nan"),
        "positive_rate": float(np.mean(prediction)),
    }


def fit_logistic_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    penalty: np.ndarray,
    ridge: float = 1.0,
    max_iter: int = 100,
) -> np.ndarray:
    """Fit a small logistic model with damped Newton updates."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)
    penalty = np.asarray(penalty, dtype=float)

    def objective(candidate: np.ndarray) -> float:
        linear = x @ candidate
        likelihood = float(np.logaddexp(0.0, linear).sum() - y @ linear)
        return likelihood + 0.5 * ridge * float(np.sum(penalty * candidate**2))

    previous = objective(beta)
    for _ in range(max_iter):
        probability = sigmoid(x @ beta)
        weight = np.clip(probability * (1 - probability), 1e-7, None)
        gradient = x.T @ (probability - y) + ridge * penalty * beta
        hessian = x.T @ (weight[:, None] * x) + ridge * np.diag(penalty)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        scale = 1.0
        accepted = False
        while scale >= 1e-6:
            candidate = beta - scale * step
            value = objective(candidate)
            if value <= previous + 1e-12:
                beta, previous, accepted = candidate, value, True
                break
            scale *= 0.5
        if not accepted or np.max(np.abs(scale * step)) < 1e-8:
            break
    return beta


def stratified_folds(
    labels: np.ndarray,
    findings: np.ndarray,
    folds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    assignments = np.full(len(labels), -1, dtype=int)
    for finding in sorted(set(findings.tolist())):
        for label in (0, 1):
            indices = np.flatnonzero((findings == finding) & (labels == label))
            rng.shuffle(indices)
            assignments[indices] = np.arange(len(indices)) % folds
    if np.any(assignments < 0):
        raise AssertionError("fold assignment incomplete")
    return assignments


def design_matrix(
    continuous: np.ndarray,
    findings: np.ndarray,
    train_indices: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_values = continuous[train_indices]
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (continuous[target_indices] - mean) / scale
    levels = sorted(set(findings.tolist()))
    # Drop one finding level; all fold-training sets contain every level.
    dummies = np.column_stack([
        (findings[target_indices] == level).astype(float) for level in levels[1:]
    ])
    matrix = np.column_stack([np.ones(len(target_indices)), standardized, dummies])
    penalty = np.ones(matrix.shape[1], dtype=float)
    penalty[0] = 0.0
    return matrix, penalty


def crossfit_probability(
    continuous: np.ndarray,
    labels: np.ndarray,
    findings: np.ndarray,
    *,
    folds: int,
    repeats: int,
    ridge: float,
    seed: int,
) -> np.ndarray:
    predictions = np.zeros(len(labels), dtype=float)
    counts = np.zeros(len(labels), dtype=int)
    all_indices = np.arange(len(labels))
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + 104729 * repeat)
        assignments = stratified_folds(labels, findings, folds, rng)
        for fold in range(folds):
            test = all_indices[assignments == fold]
            train = all_indices[assignments != fold]
            x_train, penalty = design_matrix(continuous, findings, train, train)
            x_test, _ = design_matrix(continuous, findings, train, test)
            beta = fit_logistic_ridge(x_train, labels[train], penalty=penalty, ridge=ridge)
            predictions[test] += sigmoid(x_test @ beta)
            counts[test] += 1
    if np.any(counts != repeats):
        raise AssertionError("cross-fitting coverage contract failed")
    return predictions / counts


def bootstrap_strata(labels: np.ndarray, findings: np.ndarray) -> list[np.ndarray]:
    output = []
    for finding in sorted(set(findings.tolist())):
        for label in (0, 1):
            indices = np.flatnonzero((findings == finding) & (labels == label))
            if len(indices):
                output.append(indices)
    return output


def bootstrap_indices(
    strata: Iterable[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    return np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in strata])


def interval(values: list[float]) -> list[float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    return np.quantile(finite, [0.025, 0.975]).tolist()


def summarize_raw_score(labels: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    result = binary_metrics(labels, sigmoid(score))
    result.update({
        "negative_mean": float(score[labels == 0].mean()),
        "positive_mean": float(score[labels == 1].mean()),
        "positive_minus_negative_mean": float(score[labels == 1].mean() - score[labels == 0].mean()),
    })
    return result


def analyze(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    selections = {int(row["qid"]): row for row in read_jsonl(args.selections)}
    raw = {}
    for row in read_jsonl(args.raw):
        if row.get("status") != "ok":
            continue
        qid = int(row.get("question_id", row.get("qid")))
        raw[qid] = row
    if set(selections) != set(raw):
        raise ValueError(f"coverage mismatch selections={len(selections)} raw={len(raw)}")

    score_table: dict[tuple[str, str, str], float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for qid, selection in selections.items():
        if raw[qid].get("image_sha256") != selection.get("render_sha256"):
            raise ValueError(f"image hash mismatch at qid={qid}")
        key = (selection["sample_id"], selection["render"], selection["prompt"])
        score_table[key] = float(raw[qid]["scores"]["original_margin"])
        metadata.setdefault(selection["sample_id"], selection)
    samples = sorted(metadata)
    if any((sample, render, prompt) not in score_table for sample in samples for render in RENDERS for prompt in PROMPTS):
        raise ValueError("factorial score table is incomplete")

    labels = np.asarray([int(metadata[sample]["label"]) for sample in samples], dtype=int)
    findings = np.asarray([metadata[sample]["finding"] for sample in samples], dtype=object)
    if labels.sum() != 62 or len(labels) != 124:
        raise ValueError("expected frozen balanced 124-image Huatuo panel")

    def score(render: str, prompt: str = "neutral") -> np.ndarray:
        return np.asarray([score_table[(sample, render, prompt)] for sample in samples], dtype=float)

    full = score("full")
    crop = score("zoom_sham_panel")
    true_panel = score("zoom_true_context_panel")
    delta = true_panel - crop

    base_features = np.column_stack([full, crop])
    enhanced_features = np.column_stack([full, crop, delta])
    base_probability = crossfit_probability(
        base_features, labels, findings, folds=args.folds, repeats=args.repeats,
        ridge=args.ridge, seed=args.seed,
    )
    enhanced_probability = crossfit_probability(
        enhanced_features, labels, findings, folds=args.folds, repeats=args.repeats,
        ridge=args.ridge, seed=args.seed,
    )
    single_probabilities = {
        name: crossfit_probability(
            values[:, None], labels, findings, folds=args.folds, repeats=args.repeats,
            ridge=args.ridge, seed=args.seed,
        )
        for name, values in {
            "full": full,
            "crop_phase_scrambled_context": crop,
            "true_context_panel": true_panel,
            "delta_context": delta,
        }.items()
    }

    metric_table = {
        "base_full_plus_crop_crossfit": binary_metrics(labels, base_probability),
        "enhanced_base_plus_delta_crossfit": binary_metrics(labels, enhanced_probability),
    }
    for name, probability in single_probabilities.items():
        metric_table[f"{name}_crossfit"] = binary_metrics(labels, probability)
    raw_metrics = {
        "full": summarize_raw_score(labels, full),
        "crop_phase_scrambled_context": summarize_raw_score(labels, crop),
        "true_context_panel": summarize_raw_score(labels, true_panel),
        "delta_context": summarize_raw_score(labels, delta),
    }

    rng = np.random.default_rng(args.seed)
    strata = bootstrap_strata(labels, findings)
    boot_delta_auc = []
    boot_delta_mean_gap = []
    boot_auc_gain = []
    boot_nll_gain = []
    boot_brier_gain = []
    boot_true_vs_crop_auc = []
    boot_true_vs_crop_nll = []
    for _ in range(args.bootstrap_draws):
        indices = bootstrap_indices(strata, rng)
        y = labels[indices]
        boot_delta_auc.append(auc_score(y, delta[indices]))
        boot_delta_mean_gap.append(float(delta[indices][y == 1].mean() - delta[indices][y == 0].mean()))
        base_metric = binary_metrics(y, base_probability[indices])
        enhanced_metric = binary_metrics(y, enhanced_probability[indices])
        boot_auc_gain.append(enhanced_metric["auroc"] - base_metric["auroc"])
        boot_nll_gain.append(base_metric["nll"] - enhanced_metric["nll"])
        boot_brier_gain.append(base_metric["brier"] - enhanced_metric["brier"])
        boot_true_vs_crop_auc.append(auc_score(y, true_panel[indices]) - auc_score(y, crop[indices]))
        true_metric = binary_metrics(y, sigmoid(true_panel[indices]))
        crop_metric = binary_metrics(y, sigmoid(crop[indices]))
        boot_true_vs_crop_nll.append(crop_metric["nll"] - true_metric["nll"])

    by_finding = {}
    for finding in sorted(set(findings.tolist())):
        indices = np.flatnonzero(findings == finding)
        y = labels[indices]
        by_finding[finding] = {
            "n": len(indices),
            "delta_context": summarize_raw_score(y, delta[indices]),
            "base_crossfit": binary_metrics(y, base_probability[indices]),
            "enhanced_crossfit": binary_metrics(y, enhanced_probability[indices]),
            "enhanced_minus_base_auroc": float(
                auc_score(y, enhanced_probability[indices]) - auc_score(y, base_probability[indices])
            ),
            "base_minus_enhanced_nll": float(
                binary_metrics(y, base_probability[indices])["nll"]
                - binary_metrics(y, enhanced_probability[indices])["nll"]
            ),
        }

    prompt_robustness = {}
    for prompt in PROMPTS:
        prompt_delta = score("zoom_true_context_panel", prompt) - score("zoom_sham_panel", prompt)
        prompt_robustness[prompt] = summarize_raw_score(labels, prompt_delta)

    base_point = metric_table["base_full_plus_crop_crossfit"]
    enhanced_point = metric_table["enhanced_base_plus_delta_crossfit"]
    delta_point = raw_metrics["delta_context"]
    majority_findings_auc = sum(
        row["delta_context"]["auroc"] > 0.5 for row in by_finding.values()
    ) >= math.ceil(len(by_finding) / 2)
    # This is only a routing gate for a fresh confirmation. It can never turn
    # the already-inspected panel into confirmatory evidence.
    fresh_holdout_gate = bool(
        delta_point["auroc"] >= 0.60
        and interval(boot_delta_auc)[0] > 0.50
        and enhanced_point["auroc"] - base_point["auroc"] >= 0.01
        and interval(boot_auc_gain)[0] > 0
        and interval(boot_nll_gain)[0] > 0
        and majority_findings_auc
    )

    result = {
        "version": VERSION,
        "status": "complete_posthoc_hypothesis_generation_only",
        "candidate": {
            "definition": "delta_ctx = margin(zoom_true_context_panel) - margin(zoom_sham_panel)",
            "prompt": "neutral",
            "interpretation": (
                "paired response to real full-radiograph context versus a phase-scrambled context panel, "
                "holding the main ROI and panel layout fixed"
            ),
        },
        "sample": {
            "n": len(samples),
            "negative": int((labels == 0).sum()),
            "positive": int((labels == 1).sum()),
            "findings": {finding: int(np.sum(findings == finding)) for finding in sorted(set(findings.tolist()))},
        },
        "raw_score_metrics": raw_metrics,
        "crossfit_metrics": metric_table,
        "incremental": {
            "enhanced_minus_base_auroc": float(enhanced_point["auroc"] - base_point["auroc"]),
            "enhanced_minus_base_auroc_ci95": interval(boot_auc_gain),
            "base_minus_enhanced_nll": float(base_point["nll"] - enhanced_point["nll"]),
            "base_minus_enhanced_nll_ci95": interval(boot_nll_gain),
            "base_minus_enhanced_brier": float(base_point["brier"] - enhanced_point["brier"]),
            "base_minus_enhanced_brier_ci95": interval(boot_brier_gain),
            "delta_context_auroc_ci95": interval(boot_delta_auc),
            "delta_context_positive_minus_negative_mean_ci95": interval(boot_delta_mean_gap),
            "true_panel_minus_crop_auroc": float(raw_metrics["true_context_panel"]["auroc"] - raw_metrics["crop_phase_scrambled_context"]["auroc"]),
            "true_panel_minus_crop_auroc_ci95": interval(boot_true_vs_crop_auc),
            "crop_minus_true_panel_raw_nll": float(raw_metrics["crop_phase_scrambled_context"]["nll"] - raw_metrics["true_context_panel"]["nll"]),
            "crop_minus_true_panel_raw_nll_ci95": interval(boot_true_vs_crop_nll),
        },
        "by_finding": by_finding,
        "finding_consistency": {
            "delta_auroc_above_half": int(sum(row["delta_context"]["auroc"] > 0.5 for row in by_finding.values())),
            "enhanced_auroc_better_than_base": int(sum(row["enhanced_minus_base_auroc"] > 0 for row in by_finding.values())),
            "enhanced_nll_better_than_base": int(sum(row["base_minus_enhanced_nll"] > 0 for row in by_finding.values())),
            "total_findings": len(by_finding),
        },
        "prompt_robustness": prompt_robustness,
        "routing": {
            "fresh_holdout_gate": fresh_holdout_gate,
            "gate_rule": (
                "posthoc routing only: delta AUROC>=0.60 with bootstrap lower>0.50; adding delta to "
                "cross-fit full+crop improves AUROC>=0.01 with lower>0 and NLL with lower>0; "
                "delta AUROC>0.5 in a majority of findings"
            ),
            "confirmation_status": (
                "The Huatuo panel and its render summaries were inspected before this candidate was frozen. "
                "Any positive result is hypothesis-generating and requires a fresh holdout or model."
            ),
        },
        "configuration": {
            "folds": args.folds,
            "repeats": args.repeats,
            "ridge": args.ridge,
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
            "selections_sha256": sha256_file(args.selections),
            "raw_sha256": sha256_file(args.raw),
            "source_sha256": sha256_file(Path(__file__)),
            "command": " ".join(sys.argv),
        },
        "boundary": (
            "This audit asks whether true context carries label-specific response beyond a crop and a "
            "phase-scrambled control. It does not establish causal clinical evidence, mitigation, or novelty."
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps({"incremental": result["incremental"], "routing": result["routing"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
