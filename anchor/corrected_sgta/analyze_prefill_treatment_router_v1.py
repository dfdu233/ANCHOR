#!/usr/bin/env python3
"""Source-trained one-probe treatment-effect router for plain versus RAG.

Unlike response stacking, the routing decision sees only metadata from the
plain generation.  It predicts the individual change in 0/1 loss caused by
requesting RAG, chooses a threshold on source validation, and is then applied
unchanged to a different target dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from anchor.corrected_sgta.analyze_competition_routing_pilot_v1 import (
    _bootstrap_delta, _load_expert, _metrics, _split,
)


ROOT = Path("corrected_runs/paper_baselines_v1/full_matrix_v1/shared_rag_generation/huatuo")
OUT = Path("corrected_runs/prefill_treatment_router_v1/result.json")


def load(dataset: str) -> dict[str, np.ndarray]:
    plain = _load_expert(ROOT / dataset / "no_context")
    rag = _load_expert(ROOT / dataset / "rag")
    qids = sorted(set(plain) & set(rag))
    return {
        "qid": np.asarray(qids),
        "target": np.asarray([plain[qid]["target"] for qid in qids]),
        "cluster": np.asarray([plain[qid]["cluster"] for qid in qids]),
        "plain_pred": np.asarray([plain[qid]["pred"] for qid in qids]),
        "rag_pred": np.asarray([rag[qid]["pred"] for qid in qids]),
        "plain_nll": np.asarray([plain[qid]["nll"] for qid in qids]),
        "plain_tokens": np.asarray([plain[qid]["tokens"] for qid in qids]),
    }


def features(data: dict[str, np.ndarray]) -> np.ndarray:
    nll = np.clip(data["plain_nll"], 0.0, 20.0)
    confidence = np.exp(-nll)
    pred = data["plain_pred"].astype(float)
    tokens = np.log1p(data["plain_tokens"])
    return np.column_stack([pred, nll, confidence, tokens, pred * nll, pred * tokens])


def run(source: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict:
    split = np.asarray([_split(cluster) for cluster in source["cluster"]])
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    # Positive treatment effect means RAG reduces error relative to plain.
    effect = (
        (source["plain_pred"] != source["target"]).astype(float)
        - (source["rag_pred"] != source["target"]).astype(float)
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.04, max_iter=120, max_leaf_nodes=7,
        l2_regularization=2.0, min_samples_leaf=25, random_state=42,
    )
    model.fit(features(source)[train], effect[train])
    source_effect = model.predict(features(source))
    target_effect = model.predict(features(target))

    # Tune separate treatment thresholds after a positive versus negative plain
    # answer.  The policy must be non-inferior to fixed RAG in both FP and FN on
    # source validation; this prevents overall accuracy from rewarding deletion.
    rag_validation = _metrics(
        source["target"][validation], source["rag_pred"][validation]
    )
    choices = []
    grid = np.linspace(-0.2, 0.4, 31)
    for negative_threshold in grid:
        for positive_threshold in grid:
            thresholds = np.where(source["plain_pred"] == 1, positive_threshold, negative_threshold)
            request = source_effect > thresholds
            prediction = np.where(request, source["rag_pred"], source["plain_pred"])
            metric = _metrics(source["target"][validation], prediction[validation])
            feasible = metric["fp"] <= rag_validation["fp"] and metric["fn"] <= rag_validation["fn"]
            cost = float(request[validation].mean())
            choices.append((feasible, metric["balanced_accuracy"] - 0.01 * cost,
                            metric["balanced_accuracy"], -cost, negative_threshold, positive_threshold))
    feasible, _, _, _, negative_threshold, positive_threshold = max(choices)
    if not feasible:
        raise RuntimeError("no source-validation policy is FP/FN non-inferior to fixed RAG")
    source_thresholds = np.where(source["plain_pred"] == 1, positive_threshold, negative_threshold)
    target_thresholds = np.where(target["plain_pred"] == 1, positive_threshold, negative_threshold)
    source_request = source_effect > source_thresholds
    target_request = target_effect > target_thresholds
    source_output = np.where(source_request, source["rag_pred"], source["plain_pred"])
    target_output = np.where(target_request, target["rag_pred"], target["plain_pred"])
    singles = {
        "plain": _metrics(target["target"], target["plain_pred"]),
        "rag": _metrics(target["target"], target["rag_pred"]),
    }
    best_name = max(singles, key=lambda name: singles[name]["balanced_accuracy"])
    best = target[f"{best_name}_pred"]
    plain_wrong = target["plain_pred"] != target["target"]
    rag_wrong = target["rag_pred"] != target["target"]
    oracle = np.where(rag_wrong < plain_wrong, target["rag_pred"], target["plain_pred"])
    return {
        "source_train_n": int(len(train)),
        "source_validation_n": int(len(validation)),
        "thresholds_source_validation": {
            "after_plain_negative": float(negative_threshold),
            "after_plain_positive": float(positive_threshold),
        },
        "source_validation": {
            "policy": _metrics(source["target"][validation], source_output[validation]),
            "rag_request_rate": float(source_request[validation].mean()),
        },
        "target": {
            "best_fixed_name": best_name,
            "best_fixed": singles[best_name],
            "policy": _metrics(target["target"], target_output),
            "rag_request_rate": float(target_request.mean()),
            "average_full_generation_count": float(1.0 + target_request.mean()),
            "delta_vs_best_fixed": _bootstrap_delta(
                target["target"], target_output, best, target["cluster"]
            ),
            "oracle_two_treatment": _metrics(target["target"], oracle),
        },
    }


def main() -> None:
    source = load("knowledge_mimic_ce")
    target = load("cxr_vishal")
    result = {
        "status": "source_trained_cross_dataset_complete",
        "features_available_before_rag": [
            "plain_prediction", "plain_mean_token_nll", "plain_token_count",
        ],
        "source": "knowledge_mimic_ce",
        "target": "cxr_vishal",
        "result": run(source, target),
        "decision_rule": (
            "advance prefill treatment routing only if it beats the best fixed treatment "
            "with CI excluding zero and does not increase both FP and FN"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
