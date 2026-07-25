#!/usr/bin/env python3
"""Conditional source Fisher directions with calibration-prior thresholding."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from corrected_sgta.analyze_feature_sgta import normalize_rows, softmax
from corrected_sgta.cache import decode_array, iter_successes
from corrected_sgta.protocol import deterministic_split, file_sha256
from corrected_sgta.scat_methods import fit_logit_scale, tim_probabilities


VERSION = "conditional-source-fisher-v1"
TARGET_DATASET = Path(
    "/root/autodl-tmp/MedHEval/benchmark_data/"
    "Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json"
)
WRONG_GROUP = {
    "presence": "attribute",
    "attribute": "modality",
    "modality": "presence",
    "location": "presence",
    "count": "presence",
}


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
    parser.add_argument("--rank-offset", type=int, default=0)
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


def ranked_slice(
    qids: list[str], limit: int, offset: int, seed: int, role: str
) -> list[str]:
    ranked = sorted(
        qids,
        key=lambda qid: hashlib.sha256(f"{seed}:{role}:{qid}".encode()).hexdigest(),
    )
    if limit <= 0:
        return ranked[offset:]
    return ranked[offset : offset + limit]


def exact_mcnemar(left_correct: np.ndarray, right_correct: np.ndarray) -> dict:
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
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


def source_scores(
    target_features: np.ndarray,
    target_groups: list[str],
    source_features: np.ndarray,
    source_labels: np.ndarray,
    source_groups: np.ndarray,
    control: str,
) -> tuple[np.ndarray, dict]:
    scores = np.zeros(len(target_features), dtype=np.float64)
    diagnostics = {}
    for group in sorted(set(target_groups)):
        target_mask = np.asarray([value == group for value in target_groups])
        source_group = WRONG_GROUP[group] if control == "wrong_qtype" else group
        source_mask = source_groups == source_group
        labels = source_labels[source_mask].copy()
        features = source_features[source_mask]
        if control == "shuffled_label":
            labels = 1 - labels
        counts = np.bincount(labels, minlength=2)
        if len(features) < 4 or int(counts.min()) < 2:
            source_group = "global"
            features = source_features
            labels = source_labels.copy()
            if control == "shuffled_label":
                labels = 1 - labels
            counts = np.bincount(labels, minlength=2)
        fisher = LinearDiscriminantAnalysis(solver="svd")
        fisher.fit(features, labels)
        scores[target_mask] = fisher.decision_function(target_features[target_mask])
        diagnostics[group] = {
            "source_group": source_group,
            "source_counts": counts.tolist(),
            "source_training_accuracy": float(np.mean(fisher.predict(features) == labels)),
            "score_mean": float(scores[target_mask].mean()),
            "score_std": float(scores[target_mask].std()),
        }
    return scores, diagnostics


def prior_matched_predictions(
    scores: np.ndarray,
    groups: list[str],
    qids: list[str],
    class_one_prior: float,
) -> tuple[np.ndarray, dict]:
    predictions = np.zeros(len(scores), dtype=np.int64)
    diagnostics = {}
    for group in sorted(set(groups)):
        indices = np.flatnonzero(np.asarray([value == group for value in groups]))
        count_one = int(round(class_one_prior * len(indices)))
        ranked = sorted(
            indices,
            key=lambda index: (float(scores[index]), qids[index]),
            reverse=True,
        )
        if count_one > 0:
            predictions[np.asarray(ranked[:count_one])] = 1
        diagnostics[group] = {
            "n": int(len(indices)),
            "predicted_class_one": count_one,
            "class_one_prior": class_one_prior,
        }
    return predictions, diagnostics


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
    all_qids = [row["qid"] for row in rows]
    calibration_qids, test_qids = deterministic_split(
        all_qids, args.calibration_fraction, args.seed
    )
    calibration_qids = ranked_slice(
        calibration_qids,
        args.max_calibration,
        args.rank_offset,
        args.seed,
        "calibration",
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
    labels = np.asarray([row["gt"] for row in ordered], dtype=np.int64)
    groups = [row["group"] for row in ordered]
    base_logits = np.stack([row["logits"] for row in ordered])
    semantic = np.load(args.semantic_prototypes)["prototypes"].astype(np.float32)
    scale = fit_logit_scale(features, semantic, base_logits)
    calibration_mask = np.asarray([qid in calibration_qids for qid in qids])
    test_mask = np.asarray([qid in test_qids for qid in qids])
    counts = np.bincount(labels[calibration_mask], minlength=2)
    class_one_prior = float(counts[1] / counts.sum())
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
    predictions = {
        "surface": base_logits.argmax(axis=1),
        "alpha0_tim_kl": alpha0.argmax(axis=1),
    }
    score_diagnostics = {}
    prior_diagnostics = {}
    scores_by_method = {}
    for control in ("matched", "shuffled_label", "wrong_qtype"):
        scores, score_diagnostics[control] = source_scores(
            features,
            groups,
            source_features,
            source_labels,
            source_groups,
            control,
        )
        prediction, prior_diagnostics[control] = prior_matched_predictions(
            scores, groups, qids, class_one_prior
        )
        scores_by_method[control] = scores
        predictions[control] = prediction
    metrics = {}
    test_labels = labels[test_mask]
    for name, prediction in predictions.items():
        correct = prediction[test_mask] == test_labels
        metrics[name] = {
            "accuracy": float(correct.mean()),
            "correct": int(correct.sum()),
            "n": int(test_mask.sum()),
        }
    comparisons = {}
    matched_correct = predictions["matched"][test_mask] == test_labels
    for baseline in ("alpha0_tim_kl", "shuffled_label", "wrong_qtype"):
        baseline_correct = predictions[baseline][test_mask] == test_labels
        comparisons[f"matched_vs_{baseline}"] = exact_mcnemar(
            matched_correct, baseline_correct
        ) | {
            "delta": metrics["matched"]["accuracy"] - metrics[baseline]["accuracy"]
        }
    delta = comparisons["matched_vs_alpha0_tim_kl"]["delta"]
    beats_controls = all(
        comparisons[f"matched_vs_{name}"]["delta"] > 0
        for name in ("shuffled_label", "wrong_qtype")
    )
    gate = {
        "status": (
            "green"
            if delta >= 0.0625 and beats_controls
            else "yellow"
            if delta > 0 and beats_controls
            else "red"
        ),
        "pilot_rule": "green >=2/32 gains vs alpha0 and beats both controls; yellow +1/32 and beats both; otherwise red",
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
                "predictions": {
                    name: int(prediction[index])
                    for name, prediction in predictions.items()
                },
                "source_scores": {
                    name: float(scores[index])
                    for name, scores in scores_by_method.items()
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
            "source_direction": "per-question-group Fisher LDA with SVD pseudoinverse",
            "target_threshold": "rank threshold fixed by calibration class marginal",
            "class_one_prior": class_one_prior,
            "hyperparameter_sweep": False,
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "gate": gate,
        "score_diagnostics": score_diagnostics,
        "prior_diagnostics": prior_diagnostics,
        "prediction_audit": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"metrics": metrics, "comparisons": comparisons, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
