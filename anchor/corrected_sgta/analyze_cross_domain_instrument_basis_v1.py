"""Test whether treatment-response coordinates transfer across medical domains.

The decoder is fitted only on Knowledge-MIMIC patient splits and applied with
frozen parameters to CXR-VisHal.  It compares a two-model plain ensemble with
an explicit intervention basis: each model's plain signed confidence plus the
RAG-induced change while the image/question identity is held fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from anchor.corrected_sgta.analyze_competition_routing_pilot_v1 import (
    _bootstrap_delta,
    _load_expert,
    _metrics,
    _split,
)


ROOT = Path("corrected_runs/paper_baselines_v1/full_matrix_v1/shared_rag_generation")
OUT = Path("corrected_runs/cross_domain_instrument_basis_v1/result.json")


def load(dataset: str) -> dict[str, np.ndarray]:
    paths = [
        ROOT / "huatuo" / dataset / "no_context",
        ROOT / "huatuo" / dataset / "rag",
        ROOT / "hulu" / dataset / "no_context",
        ROOT / "hulu" / dataset / "rag",
    ]
    arms = [_load_expert(path) for path in paths]
    qids = sorted(set.intersection(*(set(arm) for arm in arms)))
    pred = np.asarray([[arm[qid]["pred"] for arm in arms] for qid in qids])
    nll = np.asarray([[arm[qid]["nll"] for arm in arms] for qid in qids])
    signed = (2 * pred - 1) * np.exp(-np.clip(nll, 0.0, 20.0))
    return {
        "qid": np.asarray(qids),
        "pred": pred,
        "signed": signed,
        "target": np.asarray([arms[0][qid]["target"] for qid in qids]),
        "cluster": np.asarray([arms[0][qid]["cluster"] for qid in qids]),
    }


def feature_sets(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    s = data["signed"]
    plain = s[:, [0, 2]]
    delta = s[:, [1, 3]] - plain
    return {
        "plain_two_model": plain,
        "intervention_basis_linear": np.concatenate([plain, delta], axis=1),
        "intervention_basis_augmented": np.concatenate(
            [plain, delta, np.abs(delta), plain * delta], axis=1
        ),
    }


def run(source: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict:
    source_split = np.asarray([_split(cluster) for cluster in source["cluster"]])
    train = np.flatnonzero(source_split == "train")
    validation = np.flatnonzero(source_split == "validation")
    source_features = feature_sets(source)
    target_features = feature_sets(target)
    target_predictions = {}
    rows = {}
    for name in source_features:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000, random_state=42),
        )
        model.fit(source_features[name][train], source["target"][train])
        source_score = model.predict_proba(source_features[name])[:, 1]
        target_score = model.predict_proba(target_features[name])[:, 1]
        grid = np.linspace(0.1, 0.9, 81)
        threshold = max(
            grid,
            key=lambda value: _metrics(
                source["target"][validation],
                (source_score[validation] >= value).astype(int),
            )["balanced_accuracy"],
        )
        target_predictions[name] = (target_score >= threshold).astype(int)
        rows[name] = {
            "threshold_source_validation": float(threshold),
            "source_validation": _metrics(
                source["target"][validation],
                (source_score[validation] >= threshold).astype(int),
            ),
            "target": _metrics(target["target"], target_predictions[name]),
        }

    source_single_scores = {
        column: _metrics(source["target"][validation], source["pred"][validation, column])
        for column in (0, 1, 2, 3)
    }
    selected_column = max(
        source_single_scores,
        key=lambda column: source_single_scores[column]["balanced_accuracy"],
    )
    baseline = target["pred"][:, selected_column]
    for name in rows:
        rows[name]["delta_vs_source_selected_single"] = _bootstrap_delta(
            target["target"], target_predictions[name], baseline, target["cluster"]
        )
    rows["intervention_basis_linear"]["delta_vs_plain_two_model"] = _bootstrap_delta(
        target["target"],
        target_predictions["intervention_basis_linear"],
        target_predictions["plain_two_model"],
        target["cluster"],
    )
    rows["intervention_basis_augmented"]["delta_vs_plain_two_model"] = _bootstrap_delta(
        target["target"],
        target_predictions["intervention_basis_augmented"],
        target_predictions["plain_two_model"],
        target["cluster"],
    )
    return {
        "source_train_n": int(len(train)),
        "source_validation_n": int(len(validation)),
        "target_n": int(len(target["target"])),
        "source_selected_single_column": int(selected_column),
        "source_selected_single_target": _metrics(target["target"], baseline),
        "variants": rows,
    }


def main() -> None:
    source = load("knowledge_mimic_ce")
    target = load("cxr_vishal")
    result = {
        "status": "source_trained_cross_domain_cached_test",
        "source": "Knowledge-MIMIC CE",
        "target": "CXR-VisHal",
        "arm_order": ["huatuo_plain", "huatuo_rag", "hulu_plain", "hulu_rag"],
        "result": run(source, target),
        "claim_boundary": (
            "Target labels are used only for evaluation, but this target was inspected in prior "
            "research; an unopened paired benchmark is still required."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
