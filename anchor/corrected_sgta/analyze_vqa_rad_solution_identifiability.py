#!/usr/bin/env python3
"""Retrospective discovery analysis for visual solution identifiability.

Conditional on a label-selected answer-changing natural-image pair, the
candidate score itself is label-free: it measures how far apart the two
prompt-averaged answer distributions are. Ground-truth labels are not used by
the score, but they were used upstream to construct the opposite-answer pair
and are used downstream to evaluate whether low separation identifies error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file
from corrected_sgta.run_vqa_rad_underidentification_pilot import PROMPT_TEMPLATES, auc


VERSION = "vqa-rad-solution-identifiability-discovery-v2"
STATES = ("supported", "refuted", "undetermined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--permutation-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=260814)
    return parser.parse_args()


def entropy(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(-np.sum(array * np.log(np.maximum(array, 1e-12))))


def js_divergence(vectors: Sequence[Sequence[float]]) -> float:
    array = np.asarray(vectors, dtype=np.float64)
    return float(entropy(array.mean(axis=0)) - np.mean([entropy(row) for row in array]))


def probability_vector(score: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(score["probabilities"][state]) for state in STATES])


def margin(score: Mapping[str, Any]) -> float:
    return float(score["logits"]["supported"] - score["logits"]["refuted"])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stratified_bootstrap(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    candidate_values, deltas = [], []
    for _ in range(draws):
        indices = np.concatenate((
            rng.choice(positives, len(positives), replace=True),
            rng.choice(negatives, len(negatives), replace=True),
        ))
        sampled_labels = labels[indices]
        candidate_auc = auc(sampled_labels, candidate[indices])
        baseline_auc = auc(sampled_labels, baseline[indices])
        if candidate_auc is not None:
            candidate_values.append(candidate_auc)
        if candidate_auc is not None and baseline_auc is not None:
            deltas.append(candidate_auc - baseline_auc)
    observed_candidate = auc(labels, candidate)
    observed_baseline = auc(labels, baseline)
    return {
        "candidate": {
            "estimate": observed_candidate,
            "ci_low": float(np.quantile(candidate_values, 0.025)),
            "ci_high": float(np.quantile(candidate_values, 0.975)),
            "valid_draws": len(candidate_values),
        },
        "candidate_minus_baseline": {
            "estimate": float(observed_candidate - observed_baseline),
            "ci_low": float(np.quantile(deltas, 0.025)),
            "ci_high": float(np.quantile(deltas, 0.975)),
            "valid_draws": len(deltas),
        },
    }


def permutation_pvalue(
    labels: np.ndarray, scores: np.ndarray, draws: int, seed: int
) -> dict[str, Any]:
    observed = float(auc(labels, scores))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        shuffled = rng.permutation(labels)
        exceed += int(float(auc(shuffled, scores)) >= observed)
    return {
        "alternative": "AUROC greater than chance under label exchangeability",
        "observed": observed,
        "draws": draws,
        "p_value": float((exceed + 1) / (draws + 1)),
    }


def derive(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    derived = []
    prompt_names = tuple(PROMPT_TEMPLATES)
    for row in records:
        if row.get("status") != "ok":
            continue
        probabilities: dict[str, np.ndarray] = {}
        margins: dict[str, np.ndarray] = {}
        for role in ("positive", "negative"):
            probabilities[role] = np.stack([
                probability_vector(row["scores"][role][name]) for name in prompt_names
            ])
            margins[role] = np.asarray([
                margin(row["scores"][role][name]) for name in prompt_names
            ])
        positive_mean = probabilities["positive"].mean(axis=0)
        negative_mean = probabilities["negative"].mean(axis=0)
        between_js = js_divergence((positive_mean, negative_mean))
        within_js = 0.5 * (
            js_divergence(probabilities["positive"])
            + js_divergence(probabilities["negative"])
        )
        contrast = margins["positive"] - margins["negative"]
        positive_base = row["scores"]["positive"]["canonical"]
        negative_base = row["scores"]["negative"]["canonical"]
        positive_error = int(positive_base["state"] != "supported")
        negative_error = int(negative_base["state"] != "refuted")
        promptwise_js = {
            name: js_divergence((probabilities["positive"][index], probabilities["negative"][index]))
            for index, name in enumerate(prompt_names)
        }
        leave_one_out_js = {}
        for index, name in enumerate(prompt_names):
            keep = [other for other in range(len(prompt_names)) if other != index]
            leave_one_out_js[name] = js_divergence((
                probabilities["positive"][keep].mean(axis=0),
                probabilities["negative"][keep].mean(axis=0),
            ))
        derived.append({
            "pair_id": row["pair_id"],
            "any_error": int(positive_error or negative_error),
            "positive_error": positive_error,
            "negative_error": negative_error,
            "baseline_mean_entropy": 0.5 * (
                entropy(probability_vector(positive_base))
                + entropy(probability_vector(negative_base))
            ),
            "between_image_js": between_js,
            "within_image_prompt_js": within_js,
            "image_explained_separation": between_js * between_js / (between_js + within_js + 1e-12),
            "image_fraction": between_js / (between_js + within_js + 1e-12),
            "absolute_margin_separation": float(abs(contrast.mean())),
            "worst_prompt_margin_separation": float(np.min(np.abs(contrast))),
            "robust_margin_separation": float(max(0.0, abs(contrast.mean()) - contrast.std())),
            "label_aware_directional_oracle": float(contrast.mean()),
            "promptwise_between_js": promptwise_js,
            "leave_one_prompt_out_between_js": leave_one_out_js,
        })
    return derived


def analyze(derived: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    labels = np.asarray([int(row["any_error"]) for row in derived])
    entropy_scores = np.asarray([float(row["baseline_mean_entropy"]) for row in derived])
    metric_names = (
        "between_image_js",
        "image_explained_separation",
        "image_fraction",
        "absolute_margin_separation",
        "worst_prompt_margin_separation",
        "robust_margin_separation",
    )
    aurocs = {
        name: auc(labels, -np.asarray([float(row[name]) for row in derived]))
        for name in metric_names
    }
    aurocs["baseline_mean_entropy"] = auc(labels, entropy_scores)
    aurocs["label_aware_directional_oracle"] = auc(
        labels,
        -np.asarray([float(row["label_aware_directional_oracle"]) for row in derived]),
    )
    promptwise = {}
    leave_one_out = {}
    for name in PROMPT_TEMPLATES:
        promptwise[name] = auc(
            labels,
            -np.asarray([float(row["promptwise_between_js"][name]) for row in derived]),
        )
        leave_one_out[name] = auc(
            labels,
            -np.asarray([float(row["leave_one_prompt_out_between_js"][name]) for row in derived]),
        )
    candidate = -np.asarray([float(row["between_image_js"]) for row in derived])
    bootstrap = stratified_bootstrap(
        labels, candidate, entropy_scores, args.bootstrap_draws, args.seed
    )
    permutation = permutation_pvalue(
        labels, candidate, args.permutation_draws, args.seed + 1
    )
    order = np.argsort(candidate)
    quartile = max(1, len(order) // 4)
    gate = {
        "pre_registered_for_future_confirmation": False,
        "status": "retrospective_discovery_only",
        "candidate_auroc_gt_0p75": bool(aurocs["between_image_js"] > 0.75),
        "candidate_ci_low_gt_chance": bool(bootstrap["candidate"]["ci_low"] > 0.5),
        "delta_vs_entropy_ci_low_gt_zero": bool(
            bootstrap["candidate_minus_baseline"]["ci_low"] > 0.0
        ),
        "all_leave_one_prompt_out_auroc_gt_0p75": bool(min(leave_one_out.values()) > 0.75),
    }
    gate["all_discovery_criteria_met"] = all(
        value for key, value in gate.items() if key not in {"pre_registered_for_future_confirmation", "status"}
    )
    return {
        "scientific_role": (
            "retrospective candidate discovery on the same natural-counterfactual panel; "
            "a held-out panel is required for confirmation"
        ),
        "candidate_definition": (
            "between-image Jensen-Shannon divergence of prompt-averaged answer distributions; "
            "the score is label-free and swap-invariant, but upstream opposite-answer pair "
            "selection is label-aware"
        ),
        "n_pairs": len(derived),
        "n_errors": int(labels.sum()),
        "pair_error_auroc": aurocs,
        "promptwise_between_image_js_error_auroc": promptwise,
        "leave_one_prompt_out_between_image_js_error_auroc": leave_one_out,
        "bootstrap": bootstrap,
        "permutation": permutation,
        "candidate_quartiles": {
            "quartile_n": quartile,
            "highest_identifiability_error_rate": float(labels[order[:quartile]].mean()),
            "lowest_identifiability_error_rate": float(labels[order[-quartile:]].mean()),
        },
        "discovery_gate": gate,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    records = read_jsonl(args.raw_jsonl)
    derived = derive(records)
    metrics = analyze(derived, args)
    command = " ".join(sys.argv)
    code_sha = sha256_file(Path(__file__))
    fingerprint_payload = {
        "version": VERSION,
        "raw_sha256": sha256_file(args.raw_jsonl),
        "source_config_sha256": sha256_file(args.source_config),
        "seed": args.seed,
        "code_sha256": code_sha,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    args.output_dir.mkdir(parents=True)
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "MedHEval VQA-RAD exact-question natural Yes/No image pairs",
        "model": "HuatuoGPT-Vision-7B (inherited frozen raw logits)",
        "method": "label-free visual solution identifiability discovery analysis",
        "raw_jsonl": str(args.raw_jsonl.resolve()),
        "source_config": str(args.source_config.resolve()),
        "seed": args.seed,
        "command": command,
        "code_sha256": code_sha,
        "fingerprint": fingerprint,
    }
    atomic_json(args.output_dir / "config.json", config)
    atomic_json(args.output_dir / "summary.json", {"metrics": metrics, "derived_pairs": derived})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
