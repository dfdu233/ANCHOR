#!/usr/bin/env python3
"""Inference-only invariant solution-set decoding on cached prompt orbits.

This is a general algorithm screen, not a disease-specific classifier.  A
candidate semantic solution is admitted only when its average log-probability
margin across equivalent prompts exceeds the prompt-induced variation of that
margin.  No labels are used by prediction, confidence, or gating.
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
from corrected_sgta.run_vqa_rad_underidentification_pilot import PROMPT_TEMPLATES, auc, entropy


VERSION = "vqa-rad-invariant-solution-decoding-v1"
STATES = ("supported", "refuted", "undetermined")
FACTUAL_STATES = ("supported", "refuted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-raw", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variation-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260814)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def probabilities(score: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(score["probabilities"][state]) for state in STATES], dtype=np.float64)


def normalize_log_scores(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    output = np.exp(shifted)
    return output / output.sum()


def predict_orbit(scores: Sequence[Mapping[str, Any]], penalty: float) -> dict[str, Any]:
    matrix = np.stack([probabilities(score) for score in scores])
    logs = np.log(np.maximum(matrix, 1e-12))
    arithmetic = matrix.mean(axis=0)
    geometric = normalize_log_scores(logs.mean(axis=0))
    worst_case = normalize_log_scores(logs.min(axis=0))
    factual_logs = logs[:, :2]
    factual_mean = factual_logs.mean(axis=0)
    chosen_index = int(np.argmax(factual_mean))
    competitor_index = 1 - chosen_index
    prompt_margins = factual_logs[:, chosen_index] - factual_logs[:, competitor_index]
    robust_margin = float(prompt_margins.mean() - penalty * prompt_margins.std())
    chosen_state = FACTUAL_STATES[chosen_index]
    unanimous_states = [STATES[int(np.argmax(row))] for row in matrix]
    return {
        "canonical": STATES[int(np.argmax(matrix[0]))],
        "arithmetic_mean": STATES[int(np.argmax(arithmetic))],
        "geometric_intersection": STATES[int(np.argmax(geometric))],
        "worst_case_intersection": STATES[int(np.argmax(worst_case))],
        "unanimity_gate": unanimous_states[0] if len(set(unanimous_states)) == 1 else "undetermined",
        "isd_gate": chosen_state if robust_margin > 0 else "undetermined",
        "isd_candidate": chosen_state,
        "robust_margin": robust_margin,
        "mean_factual_margin": float(prompt_margins.mean()),
        "factual_margin_std": float(prompt_margins.std()),
        "prompt_entropy": float(np.mean([entropy(row) for row in matrix])),
        "prompt_agreement": float(max(unanimous_states.count(state) for state in STATES) / len(unanimous_states)),
    }


def accuracy(rows: Sequence[Mapping[str, Any]], method: str) -> float:
    return float(np.mean([row[method] == row["expected_state"] for row in rows]))


def selective_metrics(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    committed = [row for row in rows if row[method] != "undetermined"]
    errors = sum(row[method] != row["expected_state"] for row in committed)
    return {
        "coverage": len(committed) / len(rows),
        "committed_error_rate": errors / len(committed) if committed else None,
        "committed_accuracy": 1 - errors / len(committed) if committed else None,
        "overall_accuracy_counting_abstention_as_error": accuracy(rows, method),
    }


def coverage_risk(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (-float(row["robust_margin"]), row["pair_id"], row["role"]))
    output = {}
    for fraction in (0.25, 0.5, 0.75, 1.0):
        count = max(1, round(len(ordered) * fraction))
        retained = ordered[:count]
        errors = [row["isd_candidate"] != row["expected_state"] for row in retained]
        output[str(fraction)] = {"n": count, "error_rate": float(np.mean(errors))}
    return output


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    source_records = [row for row in read_jsonl(args.source_raw) if row.get("status") == "ok"]
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "VQA-RAD exact-question natural-counterfactual panel (68 unique images)",
        "source_raw": str(args.source_raw.resolve()),
        "source_raw_sha256": sha256_file(args.source_raw),
        "source_config": str(args.source_config.resolve()),
        "source_config_sha256": sha256_file(args.source_config),
        "model": "HuatuoGPT-Vision-7B (scores inherited from source config)",
        "method": "inference-only label-free invariant solution-set decoding over semantic prompt orbit",
        "prompt_templates": PROMPT_TEMPLATES,
        "variation_penalty": args.variation_penalty,
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    atomic_json(args.output_dir / "config.json", config)
    derived = []
    for pair in source_records:
        for role, expected in (("positive", "supported"), ("negative", "refuted")):
            orbit = [pair["scores"][role][name] for name in PROMPT_TEMPLATES]
            row = predict_orbit(orbit, args.variation_penalty)
            row.update({
                "pair_id": pair["pair_id"],
                "role": role,
                "image": pair[f"{role}_image"],
                "expected_state": expected,
            })
            derived.append(row)
    methods = (
        "canonical",
        "arithmetic_mean",
        "geometric_intersection",
        "worst_case_intersection",
        "unanimity_gate",
        "isd_gate",
    )
    canonical_errors = [int(row["canonical"] != row["expected_state"]) for row in derived]
    metrics = {
        "n_images": len(derived),
        "accuracy": {method: accuracy(derived, method) for method in methods},
        "selective": {method: selective_metrics(derived, method) for method in ("unanimity_gate", "isd_gate")},
        "canonical_error_detection_auroc": {
            "negative_robust_margin": auc(canonical_errors, [-row["robust_margin"] for row in derived]),
            "negative_mean_factual_margin": auc(canonical_errors, [-row["mean_factual_margin"] for row in derived]),
            "factual_margin_std": auc(canonical_errors, [row["factual_margin_std"] for row in derived]),
            "prompt_entropy": auc(canonical_errors, [row["prompt_entropy"] for row in derived]),
            "negative_prompt_agreement": auc(canonical_errors, [-row["prompt_agreement"] for row in derived]),
        },
        "isd_candidate_coverage_risk": coverage_risk(derived),
        "interpretation": (
            "Screen of the language-invariance half of ISD. It is promising only if robust "
            "intersection improves forced-choice accuracy or reduces committed error at useful "
            "coverage. Shared prompt-invariant language priors require a visual-view axis next."
        ),
    }
    atomic_json(args.output_dir / "summary.json", {"version": VERSION, "config": config, "metrics": metrics, "derived_images": derived})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
