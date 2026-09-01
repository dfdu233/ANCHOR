"""Fast, outcome-honest competition routing pilot on cached CE generations.

This is deliberately a fixed analysis, not a reusable evaluation framework.  It
uses only already-generated Knowledge-MIMIC outputs, splits by patient cluster,
selects a candidate on validation, and opens the test split once.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("corrected_runs/paper_baselines_v1/full_matrix_v1")
OUT = Path("corrected_runs/competition_routing_pilot_v1/result.json")

EXPERTS = {
    "hulu_no_context": ROOT / "shared_rag_generation/hulu/knowledge_mimic_ce/no_context",
    "huatuo_native": ROOT / "native_ce/huatuo/knowledge_mimic_ce/greedy",
    "huatuo_common_prompt": ROOT / "shared_rag_generation/huatuo/knowledge_mimic_ce/no_context",
    "huatuo_rag": ROOT / "shared_rag_generation/huatuo/knowledge_mimic_ce/rag",
    "qwen_vcd": ROOT / "derived_scores/qwen/VCD/knowledge_mimic_ce",
    "qwen_dola": ROOT / "derived_scores/qwen/DoLa/knowledge_mimic_ce",
}


def _binary(value) -> int | None:
    if isinstance(value, list):
        value = value[0] if len(value) == 1 else None
    if value is None:
        return None
    value = str(value).strip().lower()
    if value == "yes":
        return 1
    if value == "no":
        return 0
    return None


def _load_expert(directory: Path) -> dict[str, dict]:
    evaluation = json.loads((directory / "evaluation_ce_v7.json").read_text())
    details = {str(row["question_id"]): row for row in evaluation["details"]}
    answers = {}
    with (directory / "answers.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            answers[str(row["question_id"])] = row
    joined = {}
    for qid in details.keys() & answers.keys():
        detail, answer = details[qid], answers[qid]
        pred = _binary(detail.get("prediction"))
        target = _binary(detail.get("ground_truth"))
        if pred is None or target is None:
            continue
        metadata = answer.get("metadata", {})
        joined[qid] = {
            "pred": pred,
            "target": target,
            "cluster": str(detail.get("cluster_id", qid)),
            "nll": float(metadata.get("mean_token_nll", 50.0)),
            "tokens": float(metadata.get("generated_token_count", 0.0)),
        }
    return joined


def _split(cluster: str) -> str:
    bucket = int(hashlib.sha256(cluster.encode()).hexdigest()[:8], 16) % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "validation"
    return "test"


def _metrics(target: np.ndarray, pred: np.ndarray) -> dict:
    target, pred = target.astype(int), pred.astype(int)
    tp = int(np.sum((target == 1) & (pred == 1)))
    tn = int(np.sum((target == 0) & (pred == 0)))
    fp = int(np.sum((target == 0) & (pred == 1)))
    fn = int(np.sum((target == 1) & (pred == 0)))
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": int(len(target)),
        "accuracy": float(np.mean(target == pred)),
        "balanced_accuracy": float((tpr + tnr) / 2),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _bootstrap_delta(
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    clusters: np.ndarray,
) -> dict:
    rng = np.random.default_rng(42)
    unique = np.unique(clusters)
    deltas = []
    for _ in range(2000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == cluster) for cluster in sampled])
        cand = _metrics(target[indices], candidate[indices])["balanced_accuracy"]
        base = _metrics(target[indices], baseline[indices])["balanced_accuracy"]
        deltas.append(cand - base)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {"delta": float(np.mean(deltas)), "ci95": [float(lo), float(hi)]}


def main() -> None:
    loaded = {name: _load_expert(path) for name, path in EXPERTS.items()}
    qids = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    names = list(EXPERTS)
    records = []
    for qid in qids:
        rows = [loaded[name][qid] for name in names]
        if len({row["target"] for row in rows}) != 1:
            raise RuntimeError(f"ground-truth mismatch for {qid}")
        if len({row["cluster"] for row in rows}) != 1:
            raise RuntimeError(f"cluster mismatch for {qid}")
        records.append(rows)

    predictions = np.asarray([[row["pred"] for row in rows] for rows in records])
    nll = np.asarray([[row["nll"] for row in rows] for rows in records])
    tokens = np.asarray([[row["tokens"] for row in rows] for rows in records])
    target = np.asarray([rows[0]["target"] for rows in records])
    clusters = np.asarray([rows[0]["cluster"] for rows in records])
    split = np.asarray([_split(cluster) for cluster in clusters])

    vote_fraction = predictions.mean(axis=1, keepdims=True)
    agreement = np.abs(vote_fraction - 0.5) * 2
    features = np.concatenate([predictions, nll, tokens, vote_fraction, agreement], axis=1)
    indices = {name: np.flatnonzero(split == name) for name in ("train", "validation", "test")}

    single_validation = {
        name: _metrics(target[indices["validation"]], predictions[indices["validation"], i])
        for i, name in enumerate(names)
    }
    best_single = max(single_validation, key=lambda name: single_validation[name]["balanced_accuracy"])
    best_index = names.index(best_single)

    majority = (predictions.sum(axis=1) > len(names) / 2).astype(int)
    ties = predictions.sum(axis=1) == len(names) / 2
    majority[ties] = predictions[ties, best_index]

    models = {
        "logistic_stack": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000, random_state=42)
        ),
        "hist_gradient_stack": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=100, max_leaf_nodes=9, l2_regularization=1.0, random_state=42
        ),
        "random_forest_stack": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=12, class_weight="balanced", n_jobs=-1, random_state=42
        ),
    }
    candidate_predictions = {"majority": majority}
    selected_thresholds = {"majority": None}
    train = indices["train"]
    for name, model in models.items():
        model.fit(features[train], target[train])
        score = model.predict_proba(features)[:, 1]
        validation = indices["validation"]
        threshold_grid = np.linspace(0.1, 0.9, 81)
        threshold = max(
            threshold_grid,
            key=lambda value: _metrics(target[validation], (score[validation] >= value).astype(int))[
                "balanced_accuracy"
            ],
        )
        selected_thresholds[name] = float(threshold)
        candidate_predictions[name] = (score >= threshold).astype(int)

    validation_metrics = {
        name: _metrics(target[indices["validation"]], pred[indices["validation"]])
        for name, pred in candidate_predictions.items()
    }
    selected = max(validation_metrics, key=lambda name: validation_metrics[name]["balanced_accuracy"])
    test = indices["test"]
    baseline_pred = predictions[:, best_index]
    selected_pred = candidate_predictions[selected]
    oracle = np.any(predictions == target[:, None], axis=1).astype(int)
    # Oracle predictions equal the target whenever any expert is correct.  When
    # all experts are wrong in a binary task, every expert shares 1-target.
    oracle_pred = np.where(oracle == 1, target, 1 - target)

    def run_fixed_hist(feature_matrix: np.ndarray) -> dict:
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=100, max_leaf_nodes=9,
            l2_regularization=1.0, random_state=42,
        )
        model.fit(feature_matrix[train], target[train])
        score = model.predict_proba(feature_matrix)[:, 1]
        validation = indices["validation"]
        threshold_grid = np.linspace(0.1, 0.9, 81)
        threshold = max(
            threshold_grid,
            key=lambda value: _metrics(
                target[validation], (score[validation] >= value).astype(int)
            )["balanced_accuracy"],
        )
        pred = (score >= threshold).astype(int)
        return {
            "threshold": float(threshold),
            "validation": _metrics(target[validation], pred[validation]),
            "test": _metrics(target[test], pred[test]),
        }

    rng = np.random.default_rng(42)
    permuted_nll = nll.copy()
    for part in indices.values():
        for column in range(permuted_nll.shape[1]):
            permuted_nll[part, column] = rng.permutation(permuted_nll[part, column])
    ablations = {
        "prediction_signature_only": run_fixed_hist(
            np.concatenate([predictions, vote_fraction, agreement], axis=1)
        ),
        "confidence_and_length_only": run_fixed_hist(
            np.concatenate([nll, tokens, vote_fraction, agreement], axis=1)
        ),
        "permuted_confidence_control": run_fixed_hist(
            np.concatenate([predictions, permuted_nll, tokens, vote_fraction, agreement], axis=1)
        ),
        "medical_outputs_only": run_fixed_hist(
            np.concatenate([
                predictions[:, :4], nll[:, :4], tokens[:, :4],
                predictions[:, :4].mean(axis=1, keepdims=True),
            ], axis=1)
        ),
        "hulu_plus_huatuo_native": run_fixed_hist(
            np.concatenate([
                predictions[:, :2], nll[:, :2], tokens[:, :2],
                predictions[:, :2].mean(axis=1, keepdims=True),
            ], axis=1)
        ),
    }

    result = {
        "status": "completed_cached_cpu_pilot",
        "dataset": "MedHEval Knowledge-MIMIC CE",
        "experts": names,
        "intersection_n": len(qids),
        "split": {name: int(len(idx)) for name, idx in indices.items()},
        "split_rule": "sha256(patient_cluster) mod 10: 0-5 train, 6-7 validation, 8-9 test",
        "best_single_selected_on_validation": best_single,
        "single_validation": single_validation,
        "candidate_validation": validation_metrics,
        "validation_selected_thresholds": selected_thresholds,
        "selected_candidate": selected,
        "mechanism_ablations": ablations,
        "test": {
            "best_single": _metrics(target[test], baseline_pred[test]),
            "selected": _metrics(target[test], selected_pred[test]),
            "majority": _metrics(target[test], majority[test]),
            "oracle": _metrics(target[test], oracle_pred[test]),
            "selected_minus_best_single_cluster_bootstrap": _bootstrap_delta(
                target[test], selected_pred[test], baseline_pred[test], clusters[test]
            ),
        },
        "claim_boundary": (
            "Competition complementarity diagnostic only. The learned stacker is not a training-free "
            "hallucination method and the benchmark labels do not establish clinician-verified OE efficacy."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
