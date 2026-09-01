"""Blind-split audit of patient-aligned intervention response codes on CXR-VisHal.

This cached-output experiment asks whether gains come from the *paired response
pattern* for one case, rather than from the marginal accuracy of a generic
ensemble.  The decisive placebo independently permutes every non-anchor arm
within each split, preserving each arm's predictions/confidence/length while
destroying patient-level intervention alignment.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from anchor.corrected_sgta.analyze_competition_routing_pilot_v1 import (
    _bootstrap_delta,
    _load_expert,
    _metrics,
    _split,
)


ROOT = Path("corrected_runs/paper_baselines_v1/full_matrix_v1")
OUT = Path("corrected_runs/intervention_code_cxr_v2/result.json")
ARMS = {
    "huatuo_plain": ROOT / "shared_rag_generation/huatuo/cxr_vishal/no_context",
    "huatuo_rag": ROOT / "shared_rag_generation/huatuo/cxr_vishal/rag",
    "hulu_plain": ROOT / "shared_rag_generation/hulu/cxr_vishal/no_context",
    "hulu_rag": ROOT / "shared_rag_generation/hulu/cxr_vishal/rag",
    "qwen_dola": ROOT / "derived_scores/qwen/DoLa/cxr_vishal",
}


def fit_decode(x: np.ndarray, y: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, float]:
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=9,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(x[train], y[train])
    score = model.predict_proba(x)[:, 1]
    grid = np.linspace(0.1, 0.9, 81)
    threshold = max(
        grid,
        key=lambda value: _metrics(
            y[validation], (score[validation] >= value).astype(int)
        )["balanced_accuracy"],
    )
    return (score >= threshold).astype(int), float(threshold)


def fit_decode_indices(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, float]:
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=9,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(x[train], y[train])
    score = model.predict_proba(x)[:, 1]
    grid = np.linspace(0.1, 0.9, 81)
    threshold = max(
        grid,
        key=lambda value: _metrics(
            y[validation], (score[validation] >= value).astype(int)
        )["balanced_accuracy"],
    )
    return (score >= threshold).astype(int), float(threshold)


def make_features(pred: np.ndarray, nll: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    vote = pred.mean(axis=1, keepdims=True)
    return np.concatenate([pred, nll, np.log1p(tokens), vote, np.abs(vote - 0.5) * 2], axis=1)


def paired_placebo(
    names: list[str],
    pred: np.ndarray,
    nll: np.ndarray,
    tokens: np.ndarray,
    split: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260810)
    outputs = [pred.copy(), nll.copy(), tokens.copy()]
    for part in ("train", "validation", "test"):
        local = np.flatnonzero(split == part)
        # Preserve one untreated output from every model.  Only destroy the
        # within-patient alignment of added treatments (RAG), retaining the
        # marginal strength of every base model.
        treatment_columns = [
            column for column, name in enumerate(names) if name.endswith("_rag")
        ]
        for column in treatment_columns:
            order = rng.permutation(local)
            for source, target in zip((pred, nll, tokens), outputs):
                target[local, column] = source[order, column]
    return tuple(outputs)


def five_fold_crossfit(
    names: list[str],
    pred: np.ndarray,
    nll: np.ndarray,
    tokens: np.ndarray,
    target: np.ndarray,
    clusters: np.ndarray,
) -> dict:
    fold = np.asarray([
        int(hashlib.sha256(str(cluster).encode()).hexdigest()[:8], 16) % 5
        for cluster in clusters
    ])
    paired_out = np.zeros(len(target), dtype=int)
    placebo_out = np.zeros(len(target), dtype=int)
    baseline_out = np.zeros(len(target), dtype=int)
    selected = []
    for heldout in range(5):
        test = np.flatnonzero(fold == heldout)
        validation = np.flatnonzero(fold == ((heldout + 1) % 5))
        train = np.flatnonzero((fold != heldout) & (fold != ((heldout + 1) % 5)))
        decoded, _ = fit_decode_indices(
            make_features(pred, nll, tokens), target, train, validation
        )
        paired_out[test] = decoded[test]

        split = np.full(len(target), "train", dtype=object)
        split[validation] = "validation"
        split[test] = "test"
        ppred, pnll, ptokens = paired_placebo(names, pred, nll, tokens, split)
        placebo, _ = fit_decode_indices(
            make_features(ppred, pnll, ptokens), target, train, validation
        )
        placebo_out[test] = placebo[test]

        validation_scores = [
            _metrics(target[validation], pred[validation, column])["balanced_accuracy"]
            for column in range(pred.shape[1])
        ]
        chosen = int(np.argmax(validation_scores))
        selected.append(names[chosen])
        baseline_out[test] = pred[test, chosen]
    return {
        "folds": 5,
        "selected_single_by_fold": selected,
        "baseline": _metrics(target, baseline_out),
        "paired_code": _metrics(target, paired_out),
        "paired_delta_vs_baseline": _bootstrap_delta(
            target, paired_out, baseline_out, clusters
        ),
        "patient_misaligned_placebo": _metrics(target, placebo_out),
        "placebo_delta_vs_paired": _bootstrap_delta(
            target, placebo_out, paired_out, clusters
        ),
    }


def crossfit_features(
    x: np.ndarray, target: np.ndarray, clusters: np.ndarray
) -> np.ndarray:
    fold = np.asarray([
        int(hashlib.sha256(str(cluster).encode()).hexdigest()[:8], 16) % 5
        for cluster in clusters
    ])
    output = np.zeros(len(target), dtype=int)
    for heldout in range(5):
        test = np.flatnonzero(fold == heldout)
        validation = np.flatnonzero(fold == ((heldout + 1) % 5))
        train = np.flatnonzero((fold != heldout) & (fold != ((heldout + 1) % 5)))
        prediction, _ = fit_decode_indices(x, target, train, validation)
        output[test] = prediction[test]
    return output


def treatment_feature_ablation(
    names: list[str],
    pred: np.ndarray,
    nll: np.ndarray,
    tokens: np.ndarray,
    target: np.ndarray,
    clusters: np.ndarray,
) -> dict | None:
    required = ["huatuo_plain", "huatuo_rag", "hulu_plain", "hulu_rag"]
    if names != required:
        return None
    plain = [0, 2]
    rag = [1, 3]
    plain_features = make_features(pred[:, plain], nll[:, plain], tokens[:, plain])
    variants = {
        "plain_code": plain_features,
        "plain_plus_rag_answers": np.concatenate([plain_features, pred[:, rag]], axis=1),
        "plain_plus_rag_confidence_response": np.concatenate(
            [
                plain_features,
                nll[:, rag],
                np.log1p(tokens[:, rag]),
                nll[:, rag] - nll[:, plain],
                np.log1p(tokens[:, rag]) - np.log1p(tokens[:, plain]),
            ],
            axis=1,
        ),
        "full_code": make_features(pred, nll, tokens),
    }
    predictions = {
        name: crossfit_features(x, target, clusters) for name, x in variants.items()
    }
    base = predictions["plain_code"]
    return {
        name: {
            "metrics": _metrics(target, output),
            "delta_vs_plain_code": _bootstrap_delta(
                target, output, base, clusters
            ),
        }
        for name, output in predictions.items()
    }


def evaluate_cohort(names: list[str], loaded: dict[str, dict[str, dict]]) -> dict:
    qids = sorted(set.intersection(*(set(loaded[name]) for name in names)))
    pred = np.asarray([[loaded[name][qid]["pred"] for name in names] for qid in qids])
    nll = np.asarray([[loaded[name][qid]["nll"] for name in names] for qid in qids])
    tokens = np.asarray([[loaded[name][qid]["tokens"] for name in names] for qid in qids])
    target = np.asarray([loaded[names[0]][qid]["target"] for qid in qids])
    clusters = np.asarray([loaded[names[0]][qid]["cluster"] for qid in qids])
    split = np.asarray([_split(cluster) for cluster in clusters])
    test = np.flatnonzero(split == "test")

    singles = {
        name: _metrics(target[test], pred[test, column])
        for column, name in enumerate(names)
    }
    validation = np.flatnonzero(split == "validation")
    validation_singles = {
        name: _metrics(target[validation], pred[validation, column])
        for column, name in enumerate(names)
    }
    best_name = max(
        validation_singles,
        key=lambda name: validation_singles[name]["balanced_accuracy"],
    )
    best = pred[:, names.index(best_name)]

    paired_prediction, paired_threshold = fit_decode(
        make_features(pred, nll, tokens), target, split
    )
    pattern_prediction, pattern_threshold = fit_decode(
        make_features(pred, np.zeros_like(nll), np.zeros_like(tokens)), target, split
    )
    ppred, pnll, ptokens = paired_placebo(names, pred, nll, tokens, split)
    placebo_prediction, placebo_threshold = fit_decode(
        make_features(ppred, pnll, ptokens), target, split
    )

    return {
        "arms": names,
        "n": len(qids),
        "split_n": {part: int(np.sum(split == part)) for part in ("train", "validation", "test")},
        "best_single_selected_on_validation": best_name,
        "test_singles": singles,
        "test_best_single": _metrics(target[test], best[test]),
        "paired_code": {
            "threshold": paired_threshold,
            "test": _metrics(target[test], paired_prediction[test]),
            "delta_vs_best_single": _bootstrap_delta(
                target[test], paired_prediction[test], best[test], clusters[test]
            ),
        },
        "answer_pattern_only": {
            "threshold": pattern_threshold,
            "test": _metrics(target[test], pattern_prediction[test]),
            "delta_vs_best_single": _bootstrap_delta(
                target[test], pattern_prediction[test], best[test], clusters[test]
            ),
        },
        "patient_misaligned_placebo": {
            "threshold": placebo_threshold,
            "test": _metrics(target[test], placebo_prediction[test]),
            "delta_vs_paired_code": _bootstrap_delta(
                target[test], placebo_prediction[test], paired_prediction[test], clusters[test]
            ),
        },
        "five_fold_crossfit": five_fold_crossfit(
            names, pred, nll, tokens, target, clusters
        ),
        "treatment_feature_ablation": treatment_feature_ablation(
            names, pred, nll, tokens, target, clusters
        ),
    }


def main() -> None:
    loaded = {name: _load_expert(path) for name, path in ARMS.items()}
    cohorts = {
        "same_model_two_treatments": ["huatuo_plain", "huatuo_rag"],
        "two_models_plain_only": ["huatuo_plain", "hulu_plain"],
        "two_models_four_treatments": [
            "huatuo_plain", "huatuo_rag", "hulu_plain", "hulu_rag"
        ],
        "full_five_arm_code": list(ARMS),
    }
    result = {
        "status": "cached_cpu_blind_split_audit",
        "dataset": "MedHEval CXR-VisHal",
        "placebo_semantics": (
            "independent within-split patient permutation preserves every arm's marginal "
            "distribution but destroys the same-case intervention response code"
        ),
        "cohorts": {
            name: evaluate_cohort(arms, loaded) for name, arms in cohorts.items()
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
