#!/usr/bin/env python3
"""Paired SCA-T analysis for original and Source-ERM CE cache arms.

There is no selector or per-example routing.  Both arms use the same qids,
calibration marginal, transductive pool, prototypes, and optimization
hyperparameters.  The positive prototype scale is fitted independently for
each arm without labels because the adapter can change hidden-state scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from corrected_sgta.cache import decode_array, iter_successes
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION, deterministic_split, file_sha256
from corrected_sgta.scat_methods import fit_logit_scale, tim_probabilities


VERSION = "source-adapter-paired-scat-v1"
METHODS = ("tim", "tim_kl")


class PairedScatError(RuntimeError):
    """Raised for cache, prototype, or paired-cohort incompatibility."""


def exact_mcnemar_p(rescues: int, harms: int) -> float:
    discordant = int(rescues) + int(harms)
    if rescues < 0 or harms < 0:
        raise ValueError("paired counts must be non-negative")
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(rescues, harms) + 1))
    return float(min(1.0, 2.0 * tail / (2**discordant)))


def paired_metrics(
    ground_truth: np.ndarray,
    original_prediction: np.ndarray,
    adapted_prediction: np.ndarray,
) -> dict[str, Any]:
    gt = np.asarray(ground_truth, dtype=np.int64)
    left = np.asarray(original_prediction, dtype=np.int64)
    right = np.asarray(adapted_prediction, dtype=np.int64)
    if gt.shape != left.shape or gt.shape != right.shape or gt.ndim != 1:
        raise ValueError("ground truth and paired predictions must be aligned vectors")
    original_correct = left == gt
    adapted_correct = right == gt
    rescues = int((~original_correct & adapted_correct).sum())
    harms = int((original_correct & ~adapted_correct).sum())
    original_accuracy = float(original_correct.mean()) if gt.size else float("nan")
    adapted_accuracy = float(adapted_correct.mean()) if gt.size else float("nan")
    return {
        "n": int(gt.size),
        "original_accuracy": original_accuracy,
        "source_erm_accuracy": adapted_accuracy,
        "delta_percentage_points": 100.0 * (adapted_accuracy - original_accuracy),
        "rescues": rescues,
        "harms": harms,
        "net_rescues": rescues - harms,
        "discordant": rescues + harms,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(rescues, harms),
    }


def run_arm_scat(
    features: np.ndarray,
    reference_logits: np.ndarray,
    prototypes: np.ndarray,
    calibration_counts: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    device: str,
) -> tuple[dict[str, np.ndarray], float]:
    """Fit one source-free scale and run TIM/TIM-KL for one frozen arm."""

    scale = fit_logit_scale(features, prototypes, reference_logits)
    probabilities = {
        "tim": tim_probabilities(
            features,
            prototypes,
            scale,
            iterations=iterations,
            learning_rate=learning_rate,
            observed_marginal=None,
            entropy_weight=1.0,
            device=device,
        ),
        "tim_kl": tim_probabilities(
            features,
            prototypes,
            scale,
            iterations=iterations,
            learning_rate=learning_rate,
            observed_marginal=calibration_counts,
            entropy_weight=1.0,
            device=device,
        ),
    }
    return probabilities, scale


def load_inputs(
    cache: Path, prototypes_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], np.ndarray]:
    cache_meta_path = cache.with_suffix(cache.suffix + ".meta.json")
    prototype_meta_path = prototypes_path.with_suffix(
        prototypes_path.suffix + ".meta.json"
    )
    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
    prototype_meta = json.loads(prototype_meta_path.read_text(encoding="utf-8"))
    if (
        cache_meta.get("protocol_version") != PROTOCOL_VERSION
        or prototype_meta.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise PairedScatError("unsupported cache/prototype protocol")
    if cache_meta.get("config", {}).get("model") != "llava":
        raise PairedScatError("paired source-adapter cache must be LLaVA")
    if prototype_meta.get("model") != "llava":
        raise PairedScatError("prototype model must be LLaVA")
    fingerprint = str(cache_meta.get("fingerprint"))
    records = [
        row
        for row in iter_successes(cache, fingerprint)
        if row.get("question_type") == "binary"
        and row.get("labels") == ["Yes", "No"]
    ]
    if len(records) < 2:
        raise PairedScatError("paired SCA-T requires at least two Yes/No rows")
    qids = [str(row["qid"]) for row in records]
    if len(qids) != len(set(qids)):
        raise PairedScatError("duplicate qids in paired cache")
    style_names = tuple(cache_meta.get("config", {}).get("style_names", ()))
    if len(style_names) != 2 or style_names[0] != "original":
        raise PairedScatError(f"unexpected cache style arms: {style_names}")
    if style_names[1] not in {"source_erm", "anchor"}:
        raise PairedScatError(f"unsupported adapted style: {style_names[1]}")
    for row in records:
        if tuple(row.get("style_names", ())) != style_names:
            raise PairedScatError(f"unexpected style arms for qid={row.get('qid')}")
        features = decode_array(row["style_features"]).astype(np.float32)
        logits = np.asarray(row["style_logits"], dtype=np.float32)
        if features.shape != (2, 4096) or logits.shape != (2, 2):
            raise PairedScatError(f"invalid paired evidence shape for qid={row.get('qid')}")
        row["_features"] = features
        row["_logits"] = logits
    prototypes = np.load(prototypes_path, allow_pickle=False)["prototypes"].astype(
        np.float32
    )
    if prototypes.shape != (2, 4096):
        raise PairedScatError(f"invalid prototype shape: {prototypes.shape}")
    return records, cache_meta, prototype_meta, prototypes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--prototypes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise PairedScatError("calibration-fraction must lie in (0,1)")
    records, cache_meta, prototype_meta, prototypes = load_inputs(
        args.cache, args.prototypes
    )
    by_qid = {str(row["qid"]): row for row in records}
    calibration_qids, test_qids = deterministic_split(
        list(by_qid), args.calibration_fraction, args.seed
    )
    ordered_qids = calibration_qids + test_qids
    ground_truth = np.asarray(
        [int(by_qid[qid]["gt_index"]) for qid in ordered_qids], dtype=np.int64
    )
    calibration_counts = np.bincount(
        ground_truth[: len(calibration_qids)], minlength=2
    )
    style_names = tuple(cache_meta["config"]["style_names"])
    adapted_style = style_names[1]
    arm_probabilities: dict[str, dict[str, np.ndarray]] = {}
    scales: dict[str, float] = {}
    for arm_index, arm in enumerate(style_names):
        features = np.stack(
            [by_qid[qid]["_features"][arm_index] for qid in ordered_qids]
        )
        logits = np.stack(
            [by_qid[qid]["_logits"][arm_index] for qid in ordered_qids]
        )
        arm_probabilities[arm], scales[arm] = run_arm_scat(
            features,
            logits,
            prototypes,
            calibration_counts,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            device=args.device,
        )

    test_slice = slice(len(calibration_qids), len(ordered_qids))
    test_ground_truth = ground_truth[test_slice]
    comparisons = {}
    arm_accuracy: dict[str, dict[str, float]] = {arm: {} for arm in style_names}
    for method in METHODS:
        original_prediction = arm_probabilities["original"][method].argmax(1)[
            test_slice
        ]
        adapted_prediction = arm_probabilities[adapted_style][method].argmax(1)[
            test_slice
        ]
        comparisons[method] = paired_metrics(
            test_ground_truth, original_prediction, adapted_prediction
        )
        arm_accuracy["original"][method] = comparisons[method]["original_accuracy"]
        arm_accuracy[adapted_style][method] = comparisons[method][
            "source_erm_accuracy"
        ]

    fingerprint_payload = {
        "version": VERSION,
        "cache_sha256": file_sha256(args.cache),
        "cache_meta_sha256": file_sha256(
            args.cache.with_suffix(args.cache.suffix + ".meta.json")
        ),
        "cache_fingerprint": cache_meta["fingerprint"],
        "prototypes_sha256": file_sha256(args.prototypes),
        "prototype_meta_sha256": file_sha256(
            args.prototypes.with_suffix(args.prototypes.suffix + ".meta.json")
        ),
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "device": args.device,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "source_cache": str(args.cache.resolve()),
        "source_prototypes": str(args.prototypes.resolve()),
        "prototype_model": prototype_meta.get("model"),
        "n_yes_no": len(records),
        "split": {
            "seed": args.seed,
            "calibration_fraction": args.calibration_fraction,
            "n_calibration": len(calibration_qids),
            "n_test": len(test_qids),
            "calibration_qids": calibration_qids,
            "test_qids": test_qids,
            "calibration_class_counts": calibration_counts.tolist(),
            "shared_across_arms": True,
        },
        "adaptation": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "scale_fit": (
                "one positive scale fitted independently per arm on the shared "
                "full transductive pool using only feature/prototype geometry "
                "and that arm's surface logits; no labels"
            ),
            "fitted_positive_logit_scale": scales,
            "tim_kl_observed_marginal": (
                "one shared calibration-label count vector for both arms"
            ),
        },
        "test_accuracy": arm_accuracy,
        f"paired_{adapted_style}_vs_original": comparisons,
        "method_scope": {
            "classes": "fixed Yes/No only",
            "routing": "none; every test qid is evaluated in both arms",
            "tim": "uniform-marginal full-pool transduction",
            "tim_kl": "shared calibration-label marginal, full-pool transduction",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
