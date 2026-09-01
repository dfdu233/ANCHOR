"""Test whether decoder-intervention response codes predict clinical polarity.

The pilot uses cached outputs only.  It deliberately compares a same-Huatuo
intervention code (native prompt, common prompt, RAG) with a cross-model code
that additionally includes Hulu.  Patient clusters define train/validation/test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from anchor.corrected_sgta.analyze_competition_routing_pilot_v1 import (
    EXPERTS,
    _bootstrap_delta,
    _load_expert,
    _metrics,
    _split,
)


OUT = Path("corrected_runs/interventional_code_pilot_v1/result.json")
COHORTS = {
    "huatuo_interventions": ["huatuo_native", "huatuo_common_prompt", "huatuo_rag"],
    "medical_cross_model": [
        "hulu_no_context", "huatuo_native", "huatuo_common_prompt", "huatuo_rag"
    ],
}


def run_cohort(names: list[str], loaded: dict[str, dict[str, dict]]) -> dict:
    qids = sorted(set.intersection(*(set(loaded[name]) for name in names)))
    rows = [[loaded[name][qid] for name in names] for qid in qids]
    predictions = np.asarray([[item["pred"] for item in row] for row in rows])
    nll = np.asarray([[item["nll"] for item in row] for row in rows])
    tokens = np.asarray([[item["tokens"] for item in row] for row in rows])
    target = np.asarray([row[0]["target"] for row in rows])
    clusters = np.asarray([row[0]["cluster"] for row in rows])
    if any(len({item["target"] for item in row}) != 1 for row in rows):
        raise RuntimeError("ground-truth mismatch")
    if any(len({item["cluster"] for item in row}) != 1 for row in rows):
        raise RuntimeError("cluster mismatch")
    split = np.asarray([_split(cluster) for cluster in clusters])
    idx = {part: np.flatnonzero(split == part) for part in ("train", "validation", "test")}
    train, validation, test = idx["train"], idx["validation"], idx["test"]

    vote = predictions.mean(axis=1, keepdims=True)
    full_features = np.concatenate([predictions, nll, tokens, vote, np.abs(vote - 0.5) * 2], axis=1)
    variants = {
        "full_interventional_code": full_features,
        "answer_pattern_only": np.concatenate([predictions, vote], axis=1),
        "confidence_only": np.concatenate([nll, tokens], axis=1),
    }
    rng = np.random.default_rng(42)
    permuted = nll.copy()
    for part in idx.values():
        for column in range(permuted.shape[1]):
            permuted[part, column] = rng.permutation(permuted[part, column])
    variants["permuted_confidence_control"] = np.concatenate(
        [predictions, permuted, tokens, vote], axis=1
    )

    outputs = {}
    all_predictions = {}
    for variant, features in variants.items():
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=100, max_leaf_nodes=9,
            l2_regularization=1.0, random_state=42,
        )
        model.fit(features[train], target[train])
        probability = model.predict_proba(features)[:, 1]
        grid = np.linspace(0.1, 0.9, 81)
        threshold = max(
            grid,
            key=lambda value: _metrics(
                target[validation], (probability[validation] >= value).astype(int)
            )["balanced_accuracy"],
        )
        pred = (probability >= threshold).astype(int)
        all_predictions[variant] = pred
        outputs[variant] = {
            "threshold": float(threshold),
            "validation": _metrics(target[validation], pred[validation]),
            "test": _metrics(target[test], pred[test]),
        }

    validation_single = {
        name: _metrics(target[validation], predictions[validation, column])
        for column, name in enumerate(names)
    }
    best_single = max(validation_single, key=lambda name: validation_single[name]["balanced_accuracy"])
    best_column = names.index(best_single)
    selected = max(outputs, key=lambda name: outputs[name]["validation"]["balanced_accuracy"])

    signature_table = []
    for signature in sorted({tuple(row) for row in predictions[test]}):
        mask = np.all(predictions[test] == np.asarray(signature), axis=1)
        local = test[mask]
        signature_table.append({
            "signature": "".join(map(str, signature)),
            "n": int(len(local)),
            "positive_rate": float(target[local].mean()),
            "hulu_or_first_expert_accuracy": float(
                np.mean(predictions[local, 0] == target[local])
            ),
            "selected_accuracy": float(
                np.mean(all_predictions[selected][local] == target[local])
            ),
        })

    return {
        "experts": names,
        "intersection_n": len(qids),
        "split_n": {part: int(len(value)) for part, value in idx.items()},
        "best_single_selected_on_validation": best_single,
        "single_validation": validation_single,
        "variants": outputs,
        "selected_variant": selected,
        "test_best_single": _metrics(target[test], predictions[test, best_column]),
        "selected_minus_best_single_cluster_bootstrap": _bootstrap_delta(
            target[test], all_predictions[selected][test], predictions[test, best_column], clusters[test]
        ),
        "test_signatures": signature_table,
    }


def main() -> None:
    loaded = {name: _load_expert(path) for name, path in EXPERTS.items()}
    result = {
        "status": "completed_cached_cpu_pilot",
        "split_rule": "sha256(patient_cluster) mod 10: 0-5 train, 6-7 validation, 8-9 test",
        "cohorts": {name: run_cohort(experts, loaded) for name, experts in COHORTS.items()},
        "claim_boundary": (
            "This tests predictiveness of cached intervention codes. It does not yet show a causal "
            "internal mechanism or open-ended clinical hallucination reduction."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
