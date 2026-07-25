#!/usr/bin/env python3
"""ConfGen over the finite-label predictions from SGTA image styles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from conformal_generation import ConformalGeneration

from corrected_sgta.cache import iter_successes
from corrected_sgta.methods import softmax_np
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION, deterministic_split


SUPPORTED_PROTOCOL_VERSIONS = {PROTOCOL_VERSION, "medheval-sgta-v5.2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, nargs="*", default=(0.8, 0.9, 0.95))
    return parser.parse_args()


def style_candidates(row: dict) -> list[dict]:
    outputs = []
    for style, logits in zip(row["style_names"], row["style_logits"]):
        probabilities = softmax_np(np.asarray(logits, dtype=np.float32))
        outputs.append(
            {
                "style": style,
                "prediction": int(np.argmax(probabilities)),
                "confidence": float(np.max(probabilities)),
                "class_count": len(probabilities),
            }
        )
    return outputs


def analyze(
    by_qid: dict[str, dict],
    calibration_qids: list[str],
    test_qids: list[str],
    gammas: list[float],
) -> dict:
    calibration_outputs = [style_candidates(by_qid[qid]) for qid in calibration_qids]
    calibration_admissibility = [
        [candidate["prediction"] == by_qid[qid]["gt_index"] for candidate in candidates]
        for qid, candidates in zip(calibration_qids, calibration_outputs)
    ]
    generator = ConformalGeneration.from_score_function(
        input_dataset=[{"qid": qid} for qid in calibration_qids],
        raw_generated_dataset=calibration_outputs,
        score_fn=lambda _instance, output: output["confidence"],
        score_method="running_max",
        admissibility_dataset=calibration_admissibility,
        admissibility_aggregation=max,
        admissibility_function_lower_bound=0.0,
        use_cache=True,
    )
    output = {
        "selector": "conf-gen RunningMaxSequenceSelector",
        "score": "maximum constrained semantic-class probability",
        "n_calibration": len(calibration_qids),
        "n_test": len(test_qids),
        "raw_style_oracle_coverage_evaluation_only": mean(
            any(
                candidate["prediction"] == by_qid[qid]["gt_index"]
                for candidate in style_candidates(by_qid[qid])
            )
            for qid in test_qids
        ),
        "gamma": {},
    }
    for gamma in gammas:
        generator.calibrate(gamma=gamma, recalibrate=True)
        covered, sizes, unique_sizes, reduced_correct = [], [], [], []
        for qid in test_qids:
            row = by_qid[qid]
            selected = generator.select({"qid": qid}, style_candidates(row))
            covered.append(
                any(
                    candidate["prediction"] == row["gt_index"] for candidate in selected
                )
            )
            sizes.append(len(selected))
            unique_sizes.append(
                len({candidate["prediction"] for candidate in selected})
            )
            # Reference-free single-answer reduction.
            chosen = max(selected, key=lambda candidate: candidate["confidence"])
            reduced_correct.append(chosen["prediction"] == row["gt_index"])
        threshold = generator.conformal_threshold
        output["gamma"][str(gamma)] = {
            "lambda": None if not math.isfinite(threshold) else float(threshold),
            "lambda_is_infinite": not math.isfinite(threshold),
            "empirical_coverage": mean(covered),
            "coverage_gap": mean(covered) - gamma,
            "average_style_set_size": mean(sizes),
            "average_unique_label_set_size": mean(unique_sizes),
            "confidence_reduced_accuracy": mean(reduced_correct),
        }
    return output


def main() -> None:
    args = parse_args()
    metadata = json.loads(
        args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text()
    )
    source_protocol_version = metadata.get("protocol_version")
    if source_protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise RuntimeError(f"unsupported cache protocol: {source_protocol_version}")
    fingerprint = metadata["fingerprint"]
    records = list(iter_successes(args.cache, fingerprint))
    if len(records) < 2:
        raise RuntimeError("ConfGen analysis needs at least two successful rows")
    qids = [str(row["qid"]) for row in records]
    calibration_qids, test_qids = deterministic_split(
        qids, args.calibration_fraction, args.seed
    )
    by_qid = {str(row["qid"]): row for row in records}
    baseline_accuracy = mean(
        style_candidates(by_qid[qid])[0]["prediction"] == by_qid[qid]["gt_index"]
        for qid in test_qids
    )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "source_protocol_version": source_protocol_version,
        "source_cache": str(args.cache),
        "fingerprint": fingerprint,
        "n": len(records),
        "split": {
            "seed": args.seed,
            "calibration_fraction": args.calibration_fraction,
            "n_calibration": len(calibration_qids),
            "n_test": len(test_qids),
            "calibration_qids": calibration_qids,
            "test_qids": test_qids,
        },
        "original_style_baseline_test_accuracy": baseline_accuracy,
        "sgta_confgen_ce_styles": analyze(
            by_qid, calibration_qids, test_qids, list(args.gamma)
        ),
        "method_scope": {
            "candidate_sequence": "original, feddg_center, gamma_0.8, gamma_1.2",
            "admissibility": "at least one selected style predicts the finite-label ground truth",
            "test_selection": "RunningMax confidence only; no test label use",
            "relationship_to_sgta": (
                "ConfGen over SGTA's style views; separate from the per-image feature-graph point estimator"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["sgta_confgen_ce_styles"]["gamma"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
