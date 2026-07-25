#!/usr/bin/env python3
"""Closed-form TIM posterior projection onto original-to-source-center paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from corrected_sgta.analyze_feature_sgta import normalize_rows, softmax
from corrected_sgta.analyze_conditional_source_fisher import question_group
from corrected_sgta.cache import decode_array, iter_successes
from corrected_sgta.protocol import deterministic_split, file_sha256
from corrected_sgta.scat_methods import fit_logit_scale, tim_probabilities


VERSION = "center-path-information-projection-v1"
TARGET_DATASET = Path(
    "/root/autodl-tmp/MedHEval/benchmark_data/"
    "Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--semantic-prototypes", required=True, type=Path)
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-dataset", type=Path, default=TARGET_DATASET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank-offset", type=int, default=64)
    parser.add_argument("--max-calibration", type=int, default=32)
    parser.add_argument("--max-test", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def ranked_slice(qids: list[str], limit: int, offset: int, seed: int, role: str) -> list[str]:
    ranked = sorted(
        qids,
        key=lambda qid: hashlib.sha256(f"{seed}:{role}:{qid}".encode()).hexdigest(),
    )
    return ranked[offset : offset + limit]


def exact_mcnemar(left: np.ndarray, right: np.ndarray) -> dict:
    left_only = int(np.sum(left & ~right))
    right_only = int(np.sum(~left & right))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def source_centers(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> dict[tuple[str, int], np.ndarray]:
    centers = {}
    for group in sorted(set(groups.tolist())):
        for label in (0, 1):
            mask = (groups == group) & (labels == label)
            if int(mask.sum()) >= 2:
                centers[(group, label)] = normalize_rows(features[mask].mean(0)[None])[0]
    return centers


def project(
    original_margin: np.ndarray,
    tim_yes: np.ndarray,
    groups: list[str],
    tim_prediction: np.ndarray,
    centers: dict[tuple[str, int], np.ndarray],
    semantic_direction: np.ndarray,
    scale: float,
    swap_labels: bool,
) -> tuple[np.ndarray, list[dict]]:
    probabilities = np.empty(len(original_margin), dtype=np.float64)
    diagnostics = []
    clipped_q = np.clip(tim_yes.astype(np.float64), 1e-8, 1.0 - 1e-8)
    desired_margin = np.log(clipped_q / (1.0 - clipped_q))
    for index, group in enumerate(groups):
        label = int(tim_prediction[index])
        source_label = 1 - label if swap_labels else label
        key = (group, source_label)
        if key not in centers:
            key = ("presence", source_label)
        center_margin = float(scale * np.dot(centers[key], semantic_direction))
        delta = center_margin - float(original_margin[index])
        p0 = float(1.0 / (1.0 + np.exp(-original_margin[index])))
        derivative = delta * (p0 - float(tim_yes[index]))
        if abs(delta) <= 1e-12:
            alpha = 0.0
        else:
            alpha = float(
                np.clip(
                    (desired_margin[index] - original_margin[index]) / delta,
                    0.0,
                    1.0,
                )
            )
        final_margin = float(original_margin[index] + alpha * delta)
        probabilities[index] = 1.0 / (1.0 + np.exp(-final_margin))
        diagnostics.append(
            {
                "source_group": key[0],
                "source_label": source_label,
                "original_margin": float(original_margin[index]),
                "center_margin": center_margin,
                "delta": delta,
                "risk_derivative": derivative,
                "alpha_star": alpha,
            }
        )
    return probabilities, diagnostics


def main() -> None:
    args = parse_args()
    cache_meta = json.loads(
        args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text()
    )
    questions = {
        str(row["qid"]): str(row["question"])
        for row in json.loads(args.target_dataset.read_text())
    }
    rows = []
    for row in iter_successes(args.cache, cache_meta["fingerprint"]):
        qid = str(row["qid"])
        if (
            row.get("question_type") == "binary"
            and row.get("labels") == ["Yes", "No"]
            and qid in questions
        ):
            rows.append(
                {
                    "qid": qid,
                    "gt": int(row["gt_index"]),
                    "feature": decode_array(row["style_features"])[0].astype(np.float32),
                    "logits": np.asarray(row["style_logits"][0], dtype=np.float32),
                    "group": question_group(questions[qid]),
                }
            )
    calibration_qids, test_qids = deterministic_split(
        [row["qid"] for row in rows], 0.5, args.seed
    )
    calibration_qids = ranked_slice(
        calibration_qids, args.max_calibration, args.rank_offset, args.seed, "calibration"
    )
    test_qids = ranked_slice(
        test_qids, args.max_test, args.rank_offset, args.seed, "test"
    )
    selected = set(calibration_qids + test_qids)
    ordered = sorted(
        (row for row in rows if row["qid"] in selected),
        key=lambda row: (row["qid"] not in calibration_qids, row["qid"]),
    )
    qids = [row["qid"] for row in ordered]
    features = normalize_rows(np.stack([row["feature"] for row in ordered]))
    logits = np.stack([row["logits"] for row in ordered])
    labels = np.asarray([row["gt"] for row in ordered], dtype=np.int64)
    groups = [row["group"] for row in ordered]
    test_mask = np.asarray([qid in test_qids for qid in qids])
    counts = np.bincount(
        [row["gt"] for row in ordered if row["qid"] in calibration_qids],
        minlength=2,
    )
    semantic = normalize_rows(
        np.load(args.semantic_prototypes)["prototypes"].astype(np.float32)
    )
    scale = fit_logit_scale(features, semantic, logits)
    tim = tim_probabilities(
        features,
        semantic,
        scale,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        observed_marginal=counts,
        device=args.device,
    )
    semantic_direction = semantic[0] - semantic[1]
    original_margin = scale * (features @ semantic_direction)
    source = np.load(args.source_bank)
    centers = source_centers(
        normalize_rows(source["features"].astype(np.float32)),
        source["labels"].astype(np.int64),
        source["question_groups"].astype(str),
    )
    matched_yes, matched_diag = project(
        original_margin,
        tim[:, 0],
        groups,
        tim.argmax(1),
        centers,
        semantic_direction,
        scale,
        False,
    )
    swapped_yes, swapped_diag = project(
        original_margin,
        tim[:, 0],
        groups,
        tim.argmax(1),
        centers,
        semantic_direction,
        scale,
        True,
    )
    predictions = {
        "geometric_surface": (original_margin < 0).astype(np.int64),
        "alpha0_tim_kl": tim.argmax(1),
        "matched_projection": (matched_yes < 0.5).astype(np.int64),
        "label_swapped_projection": (swapped_yes < 0.5).astype(np.int64),
    }
    metrics = {}
    for name, prediction in predictions.items():
        correct = prediction[test_mask] == labels[test_mask]
        metrics[name] = {
            "accuracy": float(correct.mean()),
            "correct": int(correct.sum()),
            "n": int(test_mask.sum()),
        }
    matched_correct = predictions["matched_projection"][test_mask] == labels[test_mask]
    comparisons = {}
    for baseline in ("alpha0_tim_kl", "label_swapped_projection"):
        baseline_correct = predictions[baseline][test_mask] == labels[test_mask]
        comparisons[f"matched_vs_{baseline}"] = exact_mcnemar(
            matched_correct, baseline_correct
        ) | {
            "delta": metrics["matched_projection"]["accuracy"]
            - metrics[baseline]["accuracy"]
        }
    gate = {
        "status": (
            "green"
            if all(value["delta"] >= 0.0625 for value in comparisons.values())
            else "red"
        ),
        "pilot_rule": "green only if matched gains >=2/32 over TIM-KL and label-swapped; otherwise stop",
    }
    audit = []
    for index, qid in enumerate(qids):
        if not test_mask[index]:
            continue
        audit.append(
            {
                "qid": qid,
                "question": questions[qid],
                "question_group": groups[index],
                "gt_index": int(labels[index]),
                "tim_yes_probability": float(tim[index, 0]),
                "matched": matched_diag[index],
                "label_swapped": swapped_diag[index],
                "predictions": {
                    name: int(prediction[index])
                    for name, prediction in predictions.items()
                },
            }
        )
    report = {
        "version": VERSION,
        "inputs": {
            "cache": str(args.cache),
            "cache_sha256": file_sha256(args.cache),
            "semantic_prototypes": str(args.semantic_prototypes),
            "semantic_prototypes_sha256": file_sha256(args.semantic_prototypes),
            "source_bank": str(args.source_bank),
            "source_bank_sha256": file_sha256(args.source_bank),
            "target_dataset": str(args.target_dataset),
            "target_dataset_sha256": file_sha256(args.target_dataset),
        },
        "split": {
            "seed": args.seed,
            "rank_offset": args.rank_offset,
            "calibration_qids": calibration_qids,
            "test_qids": test_qids,
            "calibration_counts": counts.tolist(),
        },
        "method": {
            "risk_derivative": "delta * (p_geometric_yes - p_tim_yes)",
            "alpha_star": "clip((logit(p_tim_yes) - original_margin) / delta, 0, 1)",
            "hyperparameter_sweep": False,
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "gate": gate,
        "prediction_audit": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"metrics": metrics, "comparisons": comparisons, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
