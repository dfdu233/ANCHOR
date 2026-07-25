#!/usr/bin/env python3
"""SCA-T TIM/TIM(KL) on cached fixed-class Yes/No MedHEval rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from corrected_sgta.analyze_ce import conformal_report, point_summary
from corrected_sgta.cache import decode_array, iter_successes
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION, deterministic_split
from corrected_sgta.scat_methods import fit_logit_scale, tim_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--prototypes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, nargs="*", default=(0.1, 0.05))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_metadata = json.loads(
        args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text()
    )
    prototype_metadata = json.loads(
        args.prototypes.with_suffix(args.prototypes.suffix + ".meta.json").read_text()
    )
    if (
        cache_metadata.get("protocol_version") != PROTOCOL_VERSION
        or prototype_metadata.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise RuntimeError("unsupported protocol")
    cache_model = cache_metadata["config"]["model"]
    if prototype_metadata.get("model") != cache_model:
        raise RuntimeError("prototype/cache model mismatch")
    fingerprint = cache_metadata["fingerprint"]
    records = [
        row
        for row in iter_successes(args.cache, fingerprint)
        if row.get("question_type") == "binary" and row.get("labels") == ["Yes", "No"]
    ]
    if len(records) < 2:
        raise RuntimeError("SCA-T requires at least two fixed Yes/No rows")
    for row in records:
        row["features"] = decode_array(row["style_features"])[0].astype(np.float32)
        row["base_logits"] = np.asarray(row["style_logits"][0], dtype=np.float32)
    qids = [str(row["qid"]) for row in records]
    calibration_qids, test_qids = deterministic_split(
        qids, args.calibration_fraction, args.seed
    )
    by_qid = {str(row["qid"]): row for row in records}
    ordered_qids = calibration_qids + test_qids
    features = np.stack([by_qid[qid]["features"] for qid in ordered_qids])
    base_logits = np.stack([by_qid[qid]["base_logits"] for qid in ordered_qids])
    prototypes = np.load(args.prototypes, allow_pickle=False)["prototypes"].astype(
        np.float32
    )
    if prototypes.shape != (2, features.shape[1]):
        raise RuntimeError(
            f"prototype shape {prototypes.shape} incompatible with features {features.shape}"
        )
    scale = fit_logit_scale(features, prototypes, base_logits)
    normalized_features = features / np.clip(
        np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None
    )
    normalized_prototypes = prototypes / np.clip(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12, None
    )
    initial_logits = scale * (normalized_features @ normalized_prototypes.T)
    initial_exp = np.exp(initial_logits - initial_logits.max(axis=1, keepdims=True))
    initial_probabilities = initial_exp / initial_exp.sum(axis=1, keepdims=True)
    counts = np.bincount(
        [by_qid[qid]["gt_index"] for qid in calibration_qids], minlength=2
    )
    method_arrays = {
        "scat_tim": tim_probabilities(
            features,
            prototypes,
            scale,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            observed_marginal=None,
            entropy_weight=1.0,
            device=args.device,
        ),
        "scat_tim_kl": tim_probabilities(
            features,
            prototypes,
            scale,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            observed_marginal=counts,
            entropy_weight=1.0,
            device=args.device,
        ),
    }
    probabilities = {
        "baseline_surface_logits": {
            qid: torch.softmax(torch.tensor(by_qid[qid]["base_logits"]), -1).numpy()
            for qid in ordered_qids
        },
        "scat_initial_prototype": {
            qid: initial_probabilities[index] for index, qid in enumerate(ordered_qids)
        },
    }
    for method, array in method_arrays.items():
        probabilities[method] = {
            qid: array[index] for index, qid in enumerate(ordered_qids)
        }
    predictions = {
        method: {
            qid: int(np.argmax(probability)) for qid, probability in values.items()
        }
        for method, values in probabilities.items()
    }
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "source_cache": str(args.cache),
        "source_prototypes": str(args.prototypes),
        "fingerprint": fingerprint,
        "n_yes_no": len(records),
        "split": {
            "seed": args.seed,
            "calibration_fraction": args.calibration_fraction,
            "n_calibration": len(calibration_qids),
            "n_test": len(test_qids),
            "calibration_qids": calibration_qids,
            "test_qids": test_qids,
            "calibration_class_counts": counts.tolist(),
        },
        "adaptation": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "conditional_entropy_weight": 1.0,
            "fitted_positive_logit_scale": scale,
            "device": args.device,
        },
        "point_accuracy": {
            method: point_summary({qid: values[qid] for qid in test_qids}, records)
            for method, values in predictions.items()
        },
        "point_accuracy_transductive_pool": {
            method: point_summary(values, records)
            for method, values in predictions.items()
        },
        "conformal": {
            method: conformal_report(
                method,
                values,
                by_qid,
                calibration_qids,
                test_qids,
                list(args.alpha),
                args.seed,
            )
            for method, values in probabilities.items()
        },
        "method_scope": {
            "classes": "Yes/No only; MC letters excluded because semantics vary by question",
            "surface": (
                "last multimodal prompt hidden state plus averaged normalized semantic "
                "LM-head surface-token prototypes"
            ),
            "scat_tim": "upstream uniform-marginal TIM objective, joint calibration+test transduction",
            "scat_tim_kl": (
                "upstream observed calibration-label marginal TIM(KL), joint calibration+test transduction"
            ),
            "guarantee": "transductive empirical coverage, matching SCA-T evaluation scope",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["point_accuracy"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
