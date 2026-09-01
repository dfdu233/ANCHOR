"""Compress the competition stacker into simple intervention-code decoders."""

from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from anchor.corrected_sgta.analyze_competition_routing_pilot_v1 import (
    _bootstrap_delta, _load_expert, _metrics, _split,
)


ROOT = Path("corrected_runs/paper_baselines_v1/full_matrix_v1")
OUT = Path("corrected_runs/simple_interventional_decoder_v1/result.json")
SOURCES = {
    "knowledge_mimic_huatuo": {
        "native": ROOT / "native_ce/huatuo/knowledge_mimic_ce/greedy",
        "common_prompt": ROOT / "shared_rag_generation/huatuo/knowledge_mimic_ce/no_context",
        "rag": ROOT / "shared_rag_generation/huatuo/knowledge_mimic_ce/rag",
    },
    "cxr_vishal": {
        "huatuo_no_context": ROOT / "shared_rag_generation/huatuo/cxr_vishal/no_context",
        "huatuo_rag": ROOT / "shared_rag_generation/huatuo/cxr_vishal/rag",
        "qwen_vcd": ROOT / "cross_model_methods/qwen/vcd/cxr_vishal",
        "qwen_dola": ROOT / "derived_scores/qwen/DoLa/cxr_vishal",
    },
    "slake_binary": {
        "huatuo_no_context": ROOT / "shared_rag_generation/huatuo/slake_fine_grained/no_context",
        "huatuo_rag": ROOT / "shared_rag_generation/huatuo/slake_fine_grained/rag",
        "qwen_vcd": ROOT / "cross_model_methods/qwen/vcd/slake_fine_grained",
        "qwen_dola": ROOT / "derived_scores/qwen/DoLa/slake_fine_grained",
    },
}


def evaluate(source: dict[str, Path]) -> dict:
    names = list(source)
    loaded = {name: _load_expert(path) for name, path in source.items()}
    qids = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    rows = [[loaded[name][qid] for name in names] for qid in qids]
    pred = np.asarray([[item["pred"] for item in row] for row in rows])
    nll = np.asarray([[item["nll"] for item in row] for row in rows])
    tokens = np.asarray([[item["tokens"] for item in row] for row in rows])
    target = np.asarray([row[0]["target"] for row in rows])
    clusters = np.asarray([row[0]["cluster"] for row in rows])
    split = np.asarray([_split(cluster) for cluster in clusters])
    idx = {part: np.flatnonzero(split == part) for part in ("train", "validation", "test")}
    train, validation, test = idx["train"], idx["validation"], idx["test"]
    vote = pred.mean(axis=1, keepdims=True)
    full = np.concatenate([pred, nll, tokens, vote, np.abs(vote - 0.5) * 2], axis=1)
    pattern = np.concatenate([pred, vote], axis=1)
    signed_evidence = (2 * pred - 1) * np.exp(-np.clip(nll, 0.0, 20.0))

    models = {
        "linear_code": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000, random_state=42),
        ),
        "signed_evidence_linear": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000, random_state=42),
        ),
        "shallow_tree_code": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=20, class_weight="balanced", random_state=42
        ),
        "boosted_code_reference": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=100, max_leaf_nodes=9,
            l2_regularization=1.0, random_state=42,
        ),
    }
    features = {
        "linear_code": full,
        "signed_evidence_linear": signed_evidence,
        "shallow_tree_code": full,
        "boosted_code_reference": full,
    }
    results, predictions = {}, {}
    for name, model in models.items():
        matrix = features[name]
        model.fit(matrix[train], target[train])
        probability = model.predict_proba(matrix)[:, 1]
        grid = np.linspace(0.1, 0.9, 81)
        threshold = max(
            grid,
            key=lambda value: _metrics(
                target[validation], (probability[validation] >= value).astype(int)
            )["balanced_accuracy"],
        )
        output = (probability >= threshold).astype(int)
        predictions[name] = output
        results[name] = {
            "threshold": float(threshold),
            "validation": _metrics(target[validation], output[validation]),
            "test": _metrics(target[test], output[test]),
        }

    # Parameter-free confidence vote apart from a validation-only operating threshold.
    equal_probability = 1.0 / (1.0 + np.exp(-signed_evidence.sum(axis=1)))
    grid = np.linspace(0.1, 0.9, 81)
    equal_threshold = max(
        grid,
        key=lambda value: _metrics(
            target[validation], (equal_probability[validation] >= value).astype(int)
        )["balanced_accuracy"],
    )
    equal_output = (equal_probability >= equal_threshold).astype(int)
    predictions["equal_signed_evidence"] = equal_output
    results["equal_signed_evidence"] = {
        "threshold": float(equal_threshold),
        "validation": _metrics(target[validation], equal_output[validation]),
        "test": _metrics(target[test], equal_output[test]),
    }

    subset_candidates = {}
    for size in range(2, len(names) + 1):
        for columns in itertools.combinations(range(len(names)), size):
            probability = 1.0 / (1.0 + np.exp(-signed_evidence[:, columns].sum(axis=1)))
            threshold = max(
                grid,
                key=lambda value: _metrics(
                    target[validation], (probability[validation] >= value).astype(int)
                )["balanced_accuracy"],
            )
            output = (probability >= threshold).astype(int)
            key = "+".join(names[column] for column in columns)
            subset_candidates[key] = {
                "threshold": float(threshold),
                "validation": _metrics(target[validation], output[validation]),
                "test": _metrics(target[test], output[test]),
            }
    best_subset = max(
        subset_candidates,
        key=lambda name: subset_candidates[name]["validation"]["balanced_accuracy"],
    )

    # Discrete response-code likelihood with Laplace smoothing; no continuous confidence.
    counts = {}
    for row, label in zip(pred[train], target[train]):
        key = tuple(map(int, row))
        positives, total = counts.get(key, (0, 0))
        counts[key] = (positives + int(label), total + 1)
    prior = float(target[train].mean())
    probability = np.asarray([
        (counts.get(tuple(map(int, row)), (prior * 2, 0))[0] + prior * 2)
        / (counts.get(tuple(map(int, row)), (0, 0))[1] + 2)
        for row in pred
    ])
    grid = np.linspace(0.1, 0.9, 81)
    threshold = max(
        grid,
        key=lambda value: _metrics(
            target[validation], (probability[validation] >= value).astype(int)
        )["balanced_accuracy"],
    )
    output = (probability >= threshold).astype(int)
    predictions["discrete_codebook"] = output
    results["discrete_codebook"] = {
        "threshold": float(threshold),
        "validation": _metrics(target[validation], output[validation]),
        "test": _metrics(target[test], output[test]),
    }

    single_validation = {
        name: _metrics(target[validation], pred[validation, column])
        for column, name in enumerate(names)
    }
    best = max(single_validation, key=lambda name: single_validation[name]["balanced_accuracy"])
    best_column = names.index(best)
    for name, output in predictions.items():
        results[name]["test_delta_vs_best_single"] = _bootstrap_delta(
            target[test], output[test], pred[test, best_column], clusters[test]
        )
    selected = max(results, key=lambda name: results[name]["validation"]["balanced_accuracy"])
    return {
        "experts": names,
        "intersection_n": len(qids),
        "best_single": best,
        "best_single_test": _metrics(target[test], pred[test, best_column]),
        "decoders": results,
        "equal_signed_subset_selected_on_validation": best_subset,
        "equal_signed_subset_result": subset_candidates[best_subset],
        "selected_on_validation": selected,
        "selected_delta": _bootstrap_delta(
            target[test], predictions[selected][test], pred[test, best_column], clusters[test]
        ),
    }


def main() -> None:
    result = {
        "status": "development_only_test_already_opened",
        "datasets": {name: evaluate(source) for name, source in SOURCES.items()},
        "formal_confirmation": (
            "Freeze the selected decoder before applying it to unfinished model/method cells; "
            "these test partitions have already been inspected during algorithm development."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
