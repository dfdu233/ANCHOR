#!/usr/bin/env python3
"""Closed-form conditional empirical-Bayes source centers after TIM-KL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from corrected_sgta.analyze_feature_sgta import normalize_rows, softmax
from corrected_sgta.cache import decode_array, iter_successes
from corrected_sgta.protocol import deterministic_split, file_sha256
from corrected_sgta.scat_methods import fit_logit_scale, tim_probabilities


VERSION = "conditional-source-eb-v1"
TARGET_DATASET = Path(
    "/root/autodl-tmp/MedHEval/benchmark_data/"
    "Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json"
)
GROUPS = ("presence", "location", "count", "attribute", "modality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--semantic-prototypes", required=True, type=Path)
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-dataset", type=Path, default=TARGET_DATASET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--max-calibration", type=int, default=32)
    parser.add_argument("--max-test", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def question_group(question: str) -> str:
    text = " ".join(str(question).lower().split())
    if any(token in text for token in ("modality", "x-ray", "xray", " mri", " ct ", "ultrasound")):
        return "modality"
    if any(token in text for token in ("where", "location", "located", "which side", "left or right")):
        return "location"
    if any(token in text for token in ("how many", "number of", "count")):
        return "count"
    if any(token in text for token in ("size", "shape", "color", "appearance", "plane", "view")):
        return "attribute"
    return "presence"


def ranked_limit(qids: list[str], limit: int, seed: int, role: str) -> list[str]:
    ranked = sorted(
        qids,
        key=lambda qid: hashlib.sha256(f"{seed}:{role}:{qid}".encode()).hexdigest(),
    )
    return ranked if limit <= 0 else ranked[:limit]


def weighted_stats(features: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float, float]:
    total = float(weights.sum())
    if total <= 1e-8:
        raise ValueError("empty weighted class")
    mean = np.sum(features * weights[:, None], axis=0) / total
    residual = features - mean
    variance = float(np.sum(weights * np.sum(residual * residual, axis=1)) / total)
    n_eff = float(total * total / np.clip(np.sum(weights * weights), 1e-12, None))
    return mean, variance, n_eff


def eb_centers(
    target_features: np.ndarray,
    target_probabilities: np.ndarray,
    target_groups: list[str],
    source_features: np.ndarray,
    source_labels: np.ndarray,
    source_groups: np.ndarray,
    control: str,
) -> tuple[np.ndarray, dict]:
    centers = np.zeros((len(target_features), 2, target_features.shape[1]), dtype=np.float32)
    diagnostics: dict[str, dict] = {}
    group_cycle = {GROUPS[i]: GROUPS[(i + 1) % len(GROUPS)] for i in range(len(GROUPS))}
    global_target = np.ones(len(target_features), dtype=bool)
    for group in sorted(set(target_groups)):
        target_mask = np.asarray([value == group for value in target_groups])
        if int(target_mask.sum()) < 2:
            target_mask = global_target
        source_group = group_cycle[group] if control == "wrong_qtype" else group
        source_mask_group = source_groups == source_group
        if not np.any(source_mask_group):
            source_mask_group = np.ones(len(source_features), dtype=bool)
        diagnostics[group] = {}
        for target_class in range(2):
            source_class = 1 - target_class if control == "shuffled_label" else target_class
            class_source = source_features[source_mask_group & (source_labels == source_class)]
            if len(class_source) < 2:
                class_source = source_features[source_labels == source_class]
            source_mean = class_source.mean(axis=0)
            source_variance = float(np.mean(np.sum((class_source - source_mean) ** 2, axis=1)))
            target_mean, target_variance, n_eff = weighted_stats(
                target_features[target_mask],
                target_probabilities[target_mask, target_class],
            )
            v_source = source_variance / max(len(class_source), 1)
            v_target = target_variance / max(n_eff, 1.0)
            disagreement = float(np.sum((source_mean - target_mean) ** 2))
            tau2 = max(disagreement - v_source - v_target, 0.0)
            weight = v_target / max(v_target + v_source + tau2, 1e-12)
            posterior = (1.0 - weight) * target_mean + weight * source_mean
            centers[target_mask, target_class] = posterior
            diagnostics[group][str(target_class)] = {
                "source_group": source_group,
                "source_class": source_class,
                "n_source": int(len(class_source)),
                "target_effective_n": n_eff,
                "v_source": v_source,
                "v_target": v_target,
                "disagreement": disagreement,
                "tau2": tau2,
                "source_weight": weight,
            }
    return normalize_rows(centers), diagnostics


def exact_mcnemar(left_correct: np.ndarray, right_correct: np.ndarray) -> dict:
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {"left_only": left_only, "right_only": right_only, "discordant": discordant, "p_value": p_value}


def main() -> None:
    args = parse_args()
    cache_meta = json.loads(args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text())
    questions = {
        str(row["qid"]): str(row["question"])
        for row in json.loads(args.target_dataset.read_text())
    }
    rows = []
    for row in iter_successes(args.cache, cache_meta["fingerprint"]):
        if row.get("question_type") != "binary" or row.get("labels") != ["Yes", "No"]:
            continue
        qid = str(row["qid"])
        if qid not in questions:
            continue
        rows.append(
            {
                "qid": qid,
                "gt": int(row["gt_index"]),
                "feature": decode_array(row["style_features"])[0].astype(np.float32),
                "logits": np.asarray(row["style_logits"][0], dtype=np.float32),
                "group": question_group(questions[qid]),
            }
        )
    all_qids = [row["qid"] for row in rows]
    calibration_qids, test_qids = deterministic_split(
        all_qids, args.calibration_fraction, args.seed
    )
    calibration_qids = ranked_limit(calibration_qids, args.max_calibration, args.seed, "calibration")
    test_qids = ranked_limit(test_qids, args.max_test, args.seed, "test")
    selected = set(calibration_qids + test_qids)
    ordered = [row for row in rows if row["qid"] in selected]
    ordered.sort(key=lambda row: (row["qid"] not in calibration_qids, row["qid"]))
    qids = [row["qid"] for row in ordered]
    features = normalize_rows(np.stack([row["feature"] for row in ordered]))
    labels = np.asarray([row["gt"] for row in ordered], dtype=np.int64)
    groups = [row["group"] for row in ordered]
    semantic = np.load(args.semantic_prototypes)["prototypes"].astype(np.float32)
    base_logits = np.stack([row["logits"] for row in ordered])
    scale = fit_logit_scale(features, semantic, base_logits)
    counts = np.bincount(
        [row["gt"] for row in ordered if row["qid"] in calibration_qids],
        minlength=2,
    )
    alpha0 = tim_probabilities(
        features,
        semantic,
        scale,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        observed_marginal=counts,
        device=args.device,
    )
    source = np.load(args.source_bank)
    source_features = normalize_rows(source["features"].astype(np.float32))
    source_labels = source["labels"].astype(np.int64)
    source_groups = source["question_groups"].astype(str)
    probabilities = {
        "surface": softmax(base_logits),
        "alpha0_tim_kl": alpha0,
    }
    diagnostics = {}
    for control in ("matched", "shuffled_label", "wrong_qtype"):
        centers, diagnostics[control] = eb_centers(
            features,
            alpha0,
            groups,
            source_features,
            source_labels,
            source_groups,
            control,
        )
        logits = scale * np.einsum("nd,ncd->nc", features, centers)
        probabilities[control] = softmax(logits)
    target_only_centers, diagnostics["target_only"] = eb_centers(
        features,
        alpha0,
        groups,
        source_features,
        source_labels,
        source_groups,
        "matched",
    )
    # Setting the empirical-Bayes source weight to zero isolates the target-centroid update.
    for group in diagnostics["target_only"].values():
        for cell in group.values():
            cell["source_weight"] = 0.0
    target_only = np.zeros_like(target_only_centers)
    for group in sorted(set(groups)):
        mask = np.asarray([value == group for value in groups])
        if int(mask.sum()) < 2:
            mask = np.ones(len(groups), dtype=bool)
        for cls in range(2):
            mean, _, _ = weighted_stats(features[mask], alpha0[mask, cls])
            target_only[mask, cls] = mean
    probabilities["target_only"] = softmax(
        scale * np.einsum("nd,ncd->nc", features, normalize_rows(target_only))
    )
    test_indices = np.asarray([qid in test_qids for qid in qids])
    test_labels = labels[test_indices]
    metrics = {}
    predictions = {}
    for name, values in probabilities.items():
        prediction = values.argmax(axis=1)
        predictions[name] = prediction
        metrics[name] = {
            "accuracy": float(np.mean(prediction[test_indices] == test_labels)),
            "correct": int(np.sum(prediction[test_indices] == test_labels)),
            "n": int(test_indices.sum()),
        }
    comparisons = {}
    matched_correct = predictions["matched"][test_indices] == test_labels
    for baseline in ("alpha0_tim_kl", "target_only", "shuffled_label", "wrong_qtype"):
        baseline_correct = predictions[baseline][test_indices] == test_labels
        comparisons[f"matched_vs_{baseline}"] = exact_mcnemar(
            matched_correct, baseline_correct
        ) | {
            "delta": metrics["matched"]["accuracy"] - metrics[baseline]["accuracy"]
        }
    audit = []
    for index, qid in enumerate(qids):
        if not test_indices[index]:
            continue
        audit.append(
            {
                "qid": qid,
                "question": questions[qid],
                "question_group": groups[index],
                "gt_index": int(labels[index]),
                "probabilities": {
                    name: values[index].tolist() for name, values in probabilities.items()
                },
                "predictions": {
                    name: int(values[index]) for name, values in predictions.items()
                },
            }
        )
    matched_delta = comparisons["matched_vs_alpha0_tim_kl"]["delta"]
    gate = {
        "status": (
            "green"
            if matched_delta >= 0.0625
            and comparisons["matched_vs_shuffled_label"]["delta"] > 0
            and comparisons["matched_vs_wrong_qtype"]["delta"] > 0
            else "yellow"
            if matched_delta > 0
            and comparisons["matched_vs_shuffled_label"]["delta"] > 0
            and comparisons["matched_vs_wrong_qtype"]["delta"] > 0
            else "red"
        ),
        "pilot_rule": "green >=2/32 gains vs alpha0 and beats both controls; yellow +1/32 and beats both; otherwise red",
    }
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
            "calibration_qids": calibration_qids,
            "test_qids": test_qids,
            "calibration_counts": counts.tolist(),
        },
        "method": {
            "scale": scale,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "closed_form_weight": "v_target / (v_target + v_source + tau2)",
            "tau2": "max(||mu_source-mu_target||^2-v_source-v_target, 0)",
            "hyperparameter_sweep": False,
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "gate": gate,
        "diagnostics": diagnostics,
        "prediction_audit": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"metrics": metrics, "comparisons": comparisons, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
