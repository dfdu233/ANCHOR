#!/usr/bin/env python3
"""Calibration-safe optimization for domain-center SGTA point prediction.

The test split is never used to learn style reliability or select a graph
configuration.  Hyperparameters are selected by cross-validation inside the
outer calibration split, after which the style prior is refit on the complete
outer calibration split and evaluated once on the held-out test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from corrected_sgta.cache import decode_array, iter_successes
from corrected_sgta.methods import entropy_weighted_fusion, softmax_np
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION, deterministic_split

METHOD_VERSION = "domain-center-sgta-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument(
        "--gamma", type=float, nargs="*", default=(0.0, 0.1, 0.35, 0.75, 1.5)
    )
    parser.add_argument("--iterations", type=int, nargs="*", default=(0, 2, 4, 8))
    parser.add_argument(
        "--style-temperature", type=float, nargs="*", default=(0.25, 0.5, 1.0)
    )
    parser.add_argument(
        "--prior-strength", type=float, nargs="*", default=(0.0, 0.5, 1.0, 2.0)
    )
    parser.add_argument("--prior-pseudocount", type=float, default=8.0)
    parser.add_argument(
        "--min-cv-accuracy-gain",
        type=float,
        default=0.015,
        help="Minimum CV accuracy gain over original required before selecting a non-original policy.",
    )
    parser.add_argument(
        "--cv-nll-tolerance",
        type=float,
        default=0.0,
        help="Allowed CV NLL increase over original for selecting a non-original policy.",
    )
    return parser.parse_args()


def style_kernel(features: np.ndarray) -> np.ndarray:
    """Full RBF graph over normalized style-view features."""

    normalized = features.astype(np.float64)
    normalized /= np.clip(
        np.linalg.norm(normalized, axis=-1, keepdims=True), 1e-12, None
    )
    squared = np.clip(2.0 - 2.0 * normalized @ normalized.T, 0.0, None)
    np.fill_diagonal(squared, 0.0)
    distance = np.sqrt(squared)
    nonzero = distance[distance > 0]
    sigma = max(float(nonzero.mean()), 1e-6) if len(nonzero) else 1.0
    return np.exp(-squared / (2.0 * sigma**2))


def sgta_probability(
    logits: np.ndarray,
    kernel: np.ndarray,
    gamma: float,
    iterations: int,
    style_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Reliability-weighted SGTA; uniform weights reproduce legacy SGTA."""

    values = np.asarray(logits, dtype=np.float64)
    n_styles = values.shape[0]
    if style_weights is None:
        weights = np.full(n_styles, 1.0 / n_styles, dtype=np.float64)
    else:
        weights = np.asarray(style_weights, dtype=np.float64)
        if weights.shape != (n_styles,) or not np.isfinite(weights).all():
            raise ValueError("style_weights must be a finite vector over styles")
        weights = np.clip(weights, 0.0, None)
        weights /= np.clip(weights.sum(), 1e-12, None)
    z = softmax_np(values)
    # Multiplication by n_styles keeps the uniform-prior update identical to
    # the original implementation while allowing unreliable nodes to send a
    # smaller graph message.
    message_weights = weights * n_styles
    for _ in range(iterations):
        z = softmax_np(values + gamma * kernel @ (z * message_weights[:, None]))
    probability = np.sum(weights[:, None] * z, axis=0)
    return probability / np.clip(probability.sum(), 1e-12, None)


def _stat_key(kind: str, style: str) -> str:
    return f"{kind}::{style}"


def learn_style_reliability(
    records: list[dict], pseudocount: float = 8.0
) -> dict[str, float]:
    """Learn shrunken per-task, per-domain style log-likelihood priors."""

    exact: dict[str, list[float]] = defaultdict(list)
    by_style: dict[str, list[float]] = defaultdict(list)
    all_values: list[float] = []
    for row in records:
        gt = int(row["gt_index"])
        kind = str(row.get("question_type", "unknown"))
        for style, logits in zip(row["style_names"], row["logits"]):
            value = math.log(max(float(softmax_np(logits)[gt]), 1e-12))
            exact[_stat_key(kind, style)].append(value)
            by_style[style].append(value)
            all_values.append(value)
    global_mean = float(np.mean(all_values)) if all_values else -math.log(2.0)
    scores: dict[str, float] = {"__global__": global_mean}
    for style, values in by_style.items():
        scores[f"*::{style}"] = float(
            (sum(values) + pseudocount * global_mean) / (len(values) + pseudocount)
        )
    for key, values in exact.items():
        style = key.split("::", 1)[1]
        parent = scores.get(f"*::{style}", global_mean)
        scores[key] = float(
            (sum(values) + pseudocount * parent) / (len(values) + pseudocount)
        )
    return scores


def reliability_weights(
    style_names: list[str],
    question_type: str,
    scores: dict[str, float],
    temperature: float,
    strength: float,
) -> np.ndarray:
    """Convert calibration likelihoods into a normalized style prior."""

    if temperature <= 0:
        raise ValueError("style temperature must be positive")
    if strength < 0:
        raise ValueError("prior strength must be non-negative")
    values = np.asarray(
        [
            scores.get(
                _stat_key(question_type, style),
                scores.get(f"*::{style}", scores.get("__global__", 0.0)),
            )
            for style in style_names
        ],
        dtype=np.float64,
    )
    values = strength * (values - values.mean()) / temperature
    return softmax_np(values, axis=0)


def _predict(row: dict, config: dict, scores: dict[str, float]) -> np.ndarray:
    mode = config["mode"]
    if mode == "original":
        return softmax_np(row["logits"][0])
    if mode == "entropy":
        return softmax_np(entropy_weighted_fusion(row["logits"]))
    if mode != "sgta":
        raise ValueError(f"unknown mode: {mode}")
    weights = reliability_weights(
        row["style_names"],
        str(row.get("question_type", "unknown")),
        scores,
        float(config["style_temperature"]),
        float(config["prior_strength"]),
    )
    return sgta_probability(
        row["logits"],
        row["kernel"],
        float(config["gamma"]),
        int(config["iterations"]),
        weights,
    )


def evaluate(records: list[dict], config: dict, scores: dict[str, float]) -> dict:
    correct, nll = [], []
    for row in records:
        probability = _predict(row, config, scores)
        gt = int(row["gt_index"])
        correct.append(int(np.argmax(probability)) == gt)
        nll.append(-math.log(max(float(probability[gt]), 1e-12)))
    return {
        "n": len(records),
        "accuracy": float(np.mean(correct)) if correct else None,
        "nll": float(np.mean(nll)) if nll else None,
    }


def _fold(qid: object, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{qid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def candidate_configs(args: argparse.Namespace) -> list[dict]:
    candidates = [
        {"mode": "original"},
        {"mode": "entropy"},
    ]
    seen = set()
    for gamma in args.gamma:
        for iterations in args.iterations:
            for temperature in args.style_temperature:
                for strength in args.prior_strength:
                    # Temperature has no effect under a uniform prior.
                    effective_temperature = 1.0 if strength == 0 else temperature
                    key = (gamma, iterations, effective_temperature, strength)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "mode": "sgta",
                            "gamma": float(gamma),
                            "iterations": int(iterations),
                            "style_temperature": float(effective_temperature),
                            "prior_strength": float(strength),
                        }
                    )
    return candidates


def cross_validated_score(
    records: list[dict], config: dict, folds: int, seed: int, pseudocount: float
) -> dict:
    predictions: list[tuple[bool, float]] = []
    for held_fold in range(folds):
        train = [row for row in records if _fold(row["qid"], folds, seed) != held_fold]
        held = [row for row in records if _fold(row["qid"], folds, seed) == held_fold]
        if not train or not held:
            continue
        scores = learn_style_reliability(train, pseudocount)
        for row in held:
            probability = _predict(row, config, scores)
            gt = int(row["gt_index"])
            predictions.append(
                (
                    int(np.argmax(probability)) == gt,
                    -math.log(max(float(probability[gt]), 1e-12)),
                )
            )
    return {
        "n": len(predictions),
        "cv_accuracy": float(np.mean([value[0] for value in predictions]))
        if predictions
        else None,
        "cv_nll": float(np.mean([value[1] for value in predictions]))
        if predictions
        else None,
    }


def _complexity(config: dict) -> tuple:
    if config["mode"] == "original":
        return (0, 0.0, 0, 0.0)
    if config["mode"] == "entropy":
        return (1, 0.0, 0, 0.0)
    return (
        2,
        float(config["prior_strength"]),
        int(config["iterations"]),
        float(config["gamma"]),
    )


def main() -> None:
    args = parse_args()
    if not 2 <= args.cv_folds:
        raise ValueError("cv-folds must be at least 2")
    metadata = json.loads(
        args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text()
    )
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported cache protocol")
    fingerprint = metadata["fingerprint"]
    records = list(iter_successes(args.cache, fingerprint))
    for row in records:
        row["qid"] = str(row["qid"])
        row["logits"] = np.asarray(row["style_logits"], dtype=np.float64)
        row["style_names"] = list(row["style_names"])
        row["kernel"] = style_kernel(decode_array(row["style_features"]))
    qids = [row["qid"] for row in records]
    calibration_qids, test_qids = deterministic_split(
        qids, args.calibration_fraction, args.seed
    )
    by_qid = {row["qid"]: row for row in records}
    calibration = [by_qid[qid] for qid in calibration_qids]
    test = [by_qid[qid] for qid in test_qids]
    search = []
    for config in candidate_configs(args):
        score = cross_validated_score(
            calibration, config, args.cv_folds, args.seed + 101, args.prior_pseudocount
        )
        search.append({**config, **score})
    valid = [row for row in search if row["cv_accuracy"] is not None]
    if not valid:
        raise RuntimeError("not enough calibration rows for cross-validation")
    baseline_cv = next(row for row in valid if row["mode"] == "original")
    two_vote_margin = 2.0 / max(1, int(baseline_cv.get("n") or 1))
    accuracy_margin = max(float(args.min_cv_accuracy_gain), two_vote_margin)

    def passes_safeguard(row: dict) -> bool:
        if row["mode"] == "original":
            return True
        return (
            row["cv_accuracy"] >= baseline_cv["cv_accuracy"] + accuracy_margin
            and row["cv_nll"] <= baseline_cv["cv_nll"] + float(args.cv_nll_tolerance)
        )

    selectable = [row for row in valid if passes_safeguard(row)]
    best = sorted(
        selectable,
        key=lambda row: (
            -row["cv_accuracy"],
            row["cv_nll"],
            _complexity(row),
        ),
    )[0]
    best_config = {
        key: value
        for key, value in best.items()
        if key not in {"n", "cv_accuracy", "cv_nll"}
    }
    fitted_scores = learn_style_reliability(calibration, args.prior_pseudocount)
    baseline_config = {"mode": "original"}
    entropy_config = {"mode": "entropy"}
    fixed_config = {
        "mode": "sgta",
        "gamma": 0.35,
        "iterations": 8,
        "style_temperature": 1.0,
        "prior_strength": 0.0,
    }
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "method_version": METHOD_VERSION,
        "source_cache": str(args.cache),
        "fingerprint": fingerprint,
        "split": {
            "seed": args.seed,
            "calibration_fraction": args.calibration_fraction,
            "cv_folds_inside_calibration": args.cv_folds,
            "n_calibration": len(calibration),
            "n_test": len(test),
            "calibration_qids": calibration_qids,
            "test_qids": test_qids,
        },
        "test_results": {
            "baseline_original": evaluate(test, baseline_config, fitted_scores),
            "tta_entropy": evaluate(test, entropy_config, fitted_scores),
            "fixed_uniform_sgta": evaluate(test, fixed_config, fitted_scores),
            "domain_calibrated_sgta": evaluate(test, best_config, fitted_scores),
        },
        "selected_by_calibration_cv": {**best_config, **{
            "cv_accuracy": best["cv_accuracy"],
            "cv_nll": best["cv_nll"],
        }},
        "selection_safeguard": {
            "baseline_cv_accuracy": baseline_cv["cv_accuracy"],
            "baseline_cv_nll": baseline_cv["cv_nll"],
            "required_accuracy_margin": accuracy_margin,
            "cv_nll_tolerance": float(args.cv_nll_tolerance),
            "selectable_candidates": len(selectable),
            "valid_candidates": len(valid),
        },
        "style_reliability_log_likelihood": fitted_scores,
        "search": search,
        "scope": (
            "Outer test labels are evaluation-only. Style priors and graph settings are learned "
            "only from the outer calibration split using internal cross-validation. The original "
            "view is a selectable safeguard; no universal improvement guarantee is claimed."
        ),
    }
    baseline = report["test_results"]["baseline_original"]["accuracy"]
    optimized = report["test_results"]["domain_calibrated_sgta"]["accuracy"]
    report["test_delta_vs_baseline"] = optimized - baseline
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "search"}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
