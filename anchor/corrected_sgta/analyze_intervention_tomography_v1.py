"""Test whether controlled inference interventions encode model reliability.

This is a development-only, CPU analysis over cached generations.  It does
not claim a paper result: the target datasets have already been inspected.
The purpose is to decide whether intervention responses contain a transferable
error signal beyond the confidence of a single answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from anchor.corrected_sgta.analyze_competition_routing_pilot_v1 import (
    _bootstrap_delta,
    _load_expert,
    _metrics,
    _split,
)


ROOT = Path("corrected_runs/paper_baselines_v1/full_matrix_v1")
OUT = Path("corrected_runs/intervention_tomography_v1/result.json")

PAIRS = {
    "knowledge_mimic_huatuo": {
        "plain": ROOT / "shared_rag_generation/huatuo/knowledge_mimic_ce/no_context",
        "rag": ROOT / "shared_rag_generation/huatuo/knowledge_mimic_ce/rag",
    },
    "cxr_vishal_huatuo": {
        "plain": ROOT / "shared_rag_generation/huatuo/cxr_vishal/no_context",
        "rag": ROOT / "shared_rag_generation/huatuo/cxr_vishal/rag",
    },
    "knowledge_mimic_hulu": {
        "plain": ROOT / "shared_rag_generation/hulu/knowledge_mimic_ce/no_context",
        "rag": ROOT / "shared_rag_generation/hulu/knowledge_mimic_ce/rag",
    },
    "cxr_vishal_hulu": {
        "plain": ROOT / "shared_rag_generation/hulu/cxr_vishal/no_context",
        "rag": ROOT / "shared_rag_generation/hulu/cxr_vishal/rag",
    },
}


def _signed(row: dict) -> float:
    return float((2 * row["pred"] - 1) * np.exp(-np.clip(row["nll"], 0.0, 20.0)))


def load_pair(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    loaded = {name: _load_expert(path) for name, path in paths.items()}
    qids = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    first = next(iter(loaded))
    return {
        "qid": np.asarray(qids),
        "target": np.asarray([loaded[first][qid]["target"] for qid in qids]),
        "cluster": np.asarray([loaded[first][qid]["cluster"] for qid in qids]),
        "plain_pred": np.asarray([loaded["plain"][qid]["pred"] for qid in qids]),
        "rag_pred": np.asarray([loaded["rag"][qid]["pred"] for qid in qids]),
        "plain_e": np.asarray([_signed(loaded["plain"][qid]) for qid in qids]),
        "rag_e": np.asarray([_signed(loaded["rag"][qid]) for qid in qids]),
    }


def features(data: dict[str, np.ndarray], geometry: bool) -> np.ndarray:
    rag_abs = np.abs(data["rag_e"])[:, None]
    if not geometry:
        return rag_abs
    plain, rag = data["plain_e"], data["rag_e"]
    return np.column_stack(
        [
            rag_abs[:, 0],
            np.abs(plain),
            plain,
            rag,
            np.abs(plain - rag),
            np.minimum(np.abs(plain), np.abs(rag)),
            (np.sign(plain) != np.sign(rag)).astype(float),
        ]
    )


def pair_summary(data: dict[str, np.ndarray]) -> dict:
    target = data["target"]
    plain, rag = data["plain_pred"], data["rag_pred"]
    pattern = 2 * plain + rag
    rows = {}
    for code, name in enumerate(("00", "01", "10", "11")):
        mask = pattern == code
        rows[name] = {
            "n": int(mask.sum()),
            "positive_rate": float(target[mask].mean()) if mask.any() else None,
            "plain_accuracy": float((plain[mask] == target[mask]).mean()) if mask.any() else None,
            "rag_accuracy": float((rag[mask] == target[mask]).mean()) if mask.any() else None,
        }
    rag_wrong = rag != target
    plain_wrong = plain != target
    disagreement = plain != rag
    matrix = np.column_stack([data["plain_e"], data["rag_e"]])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    return {
        "n": int(len(target)),
        "plain": _metrics(target, plain),
        "rag": _metrics(target, rag),
        "disagreement_rate": float(disagreement.mean()),
        "error_rate_when_agree": float(rag[~disagreement].__ne__(target[~disagreement]).mean()),
        "error_rate_when_disagree": float(rag[disagreement].__ne__(target[disagreement]).mean()) if disagreement.any() else None,
        "rag_errors_rescuable_by_plain": float((rag_wrong & ~plain_wrong).sum() / max(1, rag_wrong.sum())),
        "plain_errors_rescuable_by_rag": float((plain_wrong & ~rag_wrong).sum() / max(1, plain_wrong.sum())),
        "response_patterns": rows,
        "two_probe_pc1_variance_ratio": float(variance[0] / variance.sum()),
    }


def transferable_error_detection(source: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict:
    source_split = np.asarray([_split(value) for value in source["cluster"]])
    train = np.flatnonzero(source_split != "test")
    source_error = (source["rag_pred"] != source["target"]).astype(int)
    target_error = (target["rag_pred"] != target["target"]).astype(int)
    output = {}
    for name, use_geometry in (("confidence_only", False), ("intervention_geometry", True)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000, random_state=42),
        )
        model.fit(features(source, use_geometry)[train], source_error[train])
        source_score = model.predict_proba(features(source, use_geometry))[:, 1]
        target_score = model.predict_proba(features(target, use_geometry))[:, 1]
        output[name] = {
            "source_test_auroc": float(
                roc_auc_score(source_error[source_split == "test"], source_score[source_split == "test"])
            ),
            "target_all_auroc": float(roc_auc_score(target_error, target_score)),
        }
    output["target_auroc_gain"] = (
        output["intervention_geometry"]["target_all_auroc"]
        - output["confidence_only"]["target_all_auroc"]
    )
    return output


def transferable_routing(source: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict:
    """Learn on the source which of two intervention responses to trust."""

    source_split = np.asarray([_split(value) for value in source["cluster"]])
    train = np.flatnonzero(source_split == "train")
    validation = np.flatnonzero(source_split == "validation")
    x_source = features(source, True)
    x_target = features(target, True)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000, random_state=42),
    )
    model.fit(x_source[train], source["target"][train])
    source_probability = model.predict_proba(x_source)[:, 1]
    target_probability = model.predict_proba(x_target)[:, 1]
    grids = np.linspace(0.2, 0.8, 61)
    decision_threshold = max(
        grids,
        key=lambda value: _metrics(
            source["target"][validation],
            (source_probability[validation] >= value).astype(int),
        )["balanced_accuracy"],
    )
    full_source = (source_probability >= decision_threshold).astype(int)
    full_target = (target_probability >= decision_threshold).astype(int)

    # Cost-aware sequential version: only request the second probe when the
    # first response is insufficiently confident.  The 1pp probe-cost penalty
    # is frozen here rather than selected against target performance.
    confidence_grid = np.linspace(0.1, 0.95, 86)
    candidates = []
    for trigger in confidence_grid:
        source_probe = np.abs(source["plain_e"]) < trigger
        source_output = np.where(source_probe, full_source, source["plain_pred"])
        score = _metrics(source["target"][validation], source_output[validation])["balanced_accuracy"]
        cost = float(source_probe[validation].mean())
        candidates.append((score - 0.01 * cost, score, -cost, trigger))
    _, _, _, trigger = max(candidates)
    source_probe = np.abs(source["plain_e"]) < trigger
    target_probe = np.abs(target["plain_e"]) < trigger
    adaptive_source = np.where(source_probe, full_source, source["plain_pred"])
    adaptive_target = np.where(target_probe, full_target, target["plain_pred"])

    target_singles = {
        "plain": _metrics(target["target"], target["plain_pred"]),
        "rag": _metrics(target["target"], target["rag_pred"]),
    }
    best_name = max(target_singles, key=lambda key: target_singles[key]["balanced_accuracy"])
    best_pred = target[f"{best_name}_pred"]
    return {
        "source_train_n": int(len(train)),
        "source_validation_n": int(len(validation)),
        "decision_threshold_source_validation": float(decision_threshold),
        "adaptive_trigger_source_validation": float(trigger),
        "source_validation": {
            "full_two_probe": _metrics(source["target"][validation], full_source[validation]),
            "adaptive": _metrics(source["target"][validation], adaptive_source[validation]),
            "adaptive_second_probe_rate": float(source_probe[validation].mean()),
        },
        "target_all": {
            "best_single_name": best_name,
            "best_single": target_singles[best_name],
            "full_two_probe": _metrics(target["target"], full_target),
            "full_delta_vs_best_single": _bootstrap_delta(
                target["target"], full_target, best_pred, target["cluster"]
            ),
            "adaptive": _metrics(target["target"], adaptive_target),
            "adaptive_second_probe_rate": float(target_probe.mean()),
            "adaptive_average_probe_count": float(1.0 + target_probe.mean()),
            "adaptive_delta_vs_best_single": _bootstrap_delta(
                target["target"], adaptive_target, best_pred, target["cluster"]
            ),
        },
    }


def main() -> None:
    available = {
        name: load_pair(paths)
        for name, paths in PAIRS.items()
        if all(
            (path / "answers.jsonl").exists() and (path / "evaluation_ce_v7.json").exists()
            for path in paths.values()
        )
    }
    result = {
        "status": "development_only_target_labels_previously_seen",
        "purpose": (
            "Test whether paired intervention responses transfer an error-detection signal "
            "beyond single-answer confidence; not a baseline or paper claim."
        ),
        "pairs": {name: pair_summary(data) for name, data in available.items()},
        "transfers": {},
        "routing_transfers": {},
        "decision_rule": (
            "Continue intervention tomography only if geometry improves target error AUROC "
            "for at least two model/source-target pairs and later reproduces on an unopened cell."
        ),
    }
    for model in ("huatuo", "hulu"):
        source_name = f"knowledge_mimic_{model}"
        target_name = f"cxr_vishal_{model}"
        if source_name in available and target_name in available:
            result["transfers"][f"{source_name}_to_{target_name}"] = transferable_error_detection(
                available[source_name], available[target_name]
            )
            result["routing_transfers"][f"{source_name}_to_{target_name}"] = transferable_routing(
                available[source_name], available[target_name]
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
