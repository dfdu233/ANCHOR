#!/usr/bin/env python3
"""Audit whether a frozen CXR specialist provides FP-specific counter-evidence.

This is a diagnostic gate, not a proposed mitigation method.  A one-bit veto is
fit on development predictions under a strict true-positive harm budget, then
evaluated unchanged on image-disjoint confirmation data.  If low expert scores
do not preferentially identify VLM false positives, expert veto/fusion is only
a criterion shift and must be closed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from anchor.corrected_sgta.screen_external_visual_increment_v1 import FINDINGS, load_claims
from anchor.corrected_sgta.screen_xrv_visual_increment_v1 import FINDING_TARGETS, XRV_LABELS


def load_logits(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    labels = payload["labels"].astype(str).tolist()
    if labels != list(XRV_LABELS):
        raise ValueError("XRV label order drift")
    return {
        str(image_id): np.asarray(row, dtype=np.float64)
        for image_id, row in zip(payload["image_ids"], payload["logits"])
    }


def expert_score(row: dict[str, Any], logits: dict[str, np.ndarray]) -> float:
    index = {name: i for i, name in enumerate(XRV_LABELS)}
    return float(max(logits[row["image_id"]][index[name]] for name in FINDING_TARGETS[row["finding"]]))


def design(rows: list[dict[str, Any]], logits: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        [
            [float(row["finding"] == finding) for finding in FINDINGS[:-1]]
            + [expert_score(row, logits)]
            for row in rows
        ],
        dtype=np.float64,
    )


def fit_expert_probability(
    development: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    logits: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([row["label"] for row in development], dtype=np.int64)
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    model.fit(design(development, logits), y)
    return (
        model.predict_proba(design(development, logits))[:, 1],
        model.predict_proba(design(confirmation, logits))[:, 1],
    )


def choose_veto_threshold(rows: list[dict[str, Any]], probability: np.ndarray, harm_budget: float) -> float:
    predicted_positive = np.asarray([row["margin"] > 0 for row in rows])
    label = np.asarray([row["label"] for row in rows], dtype=np.int64)
    candidates = np.r_[-np.inf, np.unique(probability[predicted_positive]), np.inf]
    best = -np.inf
    best_fp = -1
    tp_total = max(1, int(np.sum(predicted_positive & (label == 1))))
    for threshold in candidates:
        veto = predicted_positive & (probability < threshold)
        tp_harm = int(np.sum(veto & (label == 1))) / tp_total
        fp_removed = int(np.sum(veto & (label == 0)))
        if tp_harm <= harm_budget + 1e-12 and fp_removed > best_fp:
            best, best_fp = float(threshold), fp_removed
    return best


def evaluate(rows: list[dict[str, Any]], probability: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted_positive = np.asarray([row["margin"] > 0 for row in rows])
    label = np.asarray([row["label"] for row in rows], dtype=np.int64)
    veto = predicted_positive & (probability < threshold)
    fp = predicted_positive & (label == 0)
    tp = predicted_positive & (label == 1)
    fp_removed = int(np.sum(veto & fp))
    tp_harmed = int(np.sum(veto & tp))
    selected = predicted_positive
    correctness = label[selected]
    veto_score = -probability[selected]
    return {
        "threshold": threshold,
        "predicted_positive": int(np.sum(selected)),
        "fp": int(np.sum(fp)),
        "tp": int(np.sum(tp)),
        "fp_removed": fp_removed,
        "tp_harmed": tp_harmed,
        "fp_removal_rate": fp_removed / max(1, int(np.sum(fp))),
        "tp_harm_rate": tp_harmed / max(1, int(np.sum(tp))),
        "counterevidence_fp_vs_tp_auroc": float(roc_auc_score(1 - correctness, veto_score)),
    }


def bootstrap(
    rows: list[dict[str, Any]], probability: np.ndarray, threshold: float, draws: int, seed: int
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["image_id"]].append(i)
    image_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    fp_rates, tp_rates, gaps = [], [], []
    for _ in range(draws):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        indices = np.asarray([i for image_id in sampled for i in groups[image_id]])
        result = evaluate([rows[i] for i in indices], probability[indices], threshold)
        fp_rates.append(result["fp_removal_rate"])
        tp_rates.append(result["tp_harm_rate"])
        gaps.append(result["fp_removal_rate"] - result["tp_harm_rate"])

    def summarize(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values)
        return {
            "mean": float(array.mean()),
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        }

    return {
        "draws": draws,
        "fp_removal_rate": summarize(fp_rates),
        "tp_harm_rate": summarize(tp_rates),
        "specificity_gap": summarize(gaps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-dev", type=Path, required=True)
    parser.add_argument("--huatuo-confirmation", type=Path, required=True)
    parser.add_argument("--hulu-dev", type=Path, required=True)
    parser.add_argument("--hulu-confirmation", type=Path, required=True)
    parser.add_argument("--xrv-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--harm-budget", type=float, default=0.01)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logits = load_logits(args.xrv_logits)
    sources = {
        "huatuo": (args.huatuo_dev, args.huatuo_confirmation),
        "hulu": (args.hulu_dev, args.hulu_confirmation),
    }
    analyses = {}
    for model_name, (dev_path, confirmation_path) in sources.items():
        development = load_claims(dev_path, "development", model_name)
        confirmation = load_claims(confirmation_path, "confirmation", model_name)
        p_dev, p_confirmation = fit_expert_probability(development, confirmation, logits, args.seed)
        threshold = choose_veto_threshold(development, p_dev, args.harm_budget)
        analyses[model_name] = {
            "development": evaluate(development, p_dev, threshold),
            "confirmation": evaluate(confirmation, p_confirmation, threshold),
            "confirmation_image_bootstrap": bootstrap(
                confirmation, p_confirmation, threshold, args.bootstrap_draws, args.seed
            ),
        }

    passes = []
    for analysis in analyses.values():
        point = analysis["confirmation"]
        boot = analysis["confirmation_image_bootstrap"]
        passes.append(
            point["fp_removal_rate"] >= 0.20
            and point["tp_harm_rate"] <= 0.01
            and boot["specificity_gap"]["ci95"][0] > 0
        )
    result = {
        "status": "complete_diagnostic_only",
        "decision": "PASS_COUNTEREVIDENCE_GATE" if all(passes) else "NO_GO_COUNTEREVIDENCE_GATE",
        "harm_budget": args.harm_budget,
        "analyses": analyses,
        "boundary": "Passing would not establish novelty; expert veto is adjacent to selective prediction and expert gating.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
