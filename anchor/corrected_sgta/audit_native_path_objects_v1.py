#!/usr/bin/env python3
"""CPU-only fatal audit for VLM-native layer-path objects.

This is deliberately an information screen, not a mitigation method.  It asks
whether simple trajectory summaries that could feed a conformal martingale,
persistence functional, or one-dimensional transport cost add held-out label
information beyond the final claim margin.  All fitting is on development;
confirmation is image-disjoint.  No model is loaded.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "native-layer-path-object-audit-v1"
FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["finding"] not in FINDINGS or int(row["positive_votes"]) not in (0, 3):
                continue
            lens = row["diagnostic_plain_logit_lens"]
            layers = sorted(lens, key=lambda value: int(value))
            margins = np.asarray(
                [float(lens[layer]["supported"] - lens[layer]["refuted"]) for layer in layers],
                dtype=np.float64,
            )
            if len(margins) != 4:
                raise ValueError(f"Expected four frozen layers, got {len(margins)}")
            rows.append(
                {
                    "image_id": str(row["image_id"]),
                    "finding": str(row["finding"]),
                    "label": int(row["positive_votes"] == 3),
                    "margins": margins,
                }
            )
    return rows


def features(row: dict[str, Any], mode: str) -> list[float]:
    margins = row["margins"]
    one_hot = [float(row["finding"] == finding) for finding in FINDINGS[:-1]]
    if mode == "final":
        values = [margins[-1]]
    elif mode == "full_path":
        values = list(margins)
    elif mode == "path_shape":
        increments = np.diff(margins)
        values = [
            margins[-1],
            float(np.max(margins[:-1])),
            float(np.min(margins[:-1])),
            float(np.abs(increments).sum()),
            float(np.max(margins) - np.min(margins)),
            float(np.max(margins[:-1]) - margins[-1]),
            float(increments[-1]),
            float(np.mean(margins)),
            float(np.sum(np.sign(margins[1:]) != np.sign(margins[:-1]))),
        ]
    else:
        raise ValueError(mode)
    return one_hot + values


def design(rows: list[dict[str, Any]], mode: str) -> np.ndarray:
    return np.asarray([features(row, mode) for row in rows], dtype=np.float64)


def macro_auc(rows: list[dict[str, Any]], score: np.ndarray) -> float:
    values = []
    for finding in FINDINGS:
        indices = [index for index, row in enumerate(rows) if row["finding"] == finding]
        labels = np.asarray([rows[index]["label"] for index in indices])
        values.append(roc_auc_score(labels, score[indices]))
    return float(np.mean(values))


def fit_predict(
    development: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    mode: str,
    estimator: str,
    seed: int,
) -> np.ndarray:
    labels = np.asarray([row["label"] for row in development])
    if estimator == "linear":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, random_state=seed),
        )
    elif estimator == "nonlinear":
        model = make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=3,
                learning_rate=0.05,
                l2_regularization=1.0,
                random_state=seed,
            ),
        )
    else:
        raise ValueError(estimator)
    model.fit(design(development, mode), labels)
    return model.predict_proba(design(confirmation, mode))[:, 1]


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    baseline: np.ndarray,
    candidate: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["image_id"]].append(index)
    image_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        sampled = rng.choice(image_ids, len(image_ids), replace=True)
        indices = np.asarray([index for image_id in sampled for index in groups[image_id]])
        sampled_rows = [rows[index] for index in indices]
        try:
            deltas.append(
                macro_auc(sampled_rows, candidate[indices])
                - macro_auc(sampled_rows, baseline[indices])
            )
        except ValueError:
            continue
    values = np.asarray(deltas)
    return {
        "draws": len(values),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def analyze(
    development: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    predictions = {
        "final_linear": fit_predict(development, confirmation, "final", "linear", seed),
        "full_path_linear": fit_predict(
            development, confirmation, "full_path", "linear", seed
        ),
        "path_shape_linear": fit_predict(
            development, confirmation, "path_shape", "linear", seed
        ),
        "full_path_nonlinear": fit_predict(
            development, confirmation, "full_path", "nonlinear", seed
        ),
        "path_shape_nonlinear": fit_predict(
            development, confirmation, "path_shape", "nonlinear", seed
        ),
    }
    auc = {name: macro_auc(confirmation, score) for name, score in predictions.items()}
    baseline = predictions["final_linear"]
    comparisons = {
        name: cluster_bootstrap(confirmation, baseline, score, draws, seed + index)
        for index, (name, score) in enumerate(predictions.items())
        if name != "final_linear"
    }
    return {
        "development_n": len(development),
        "confirmation_n": len(confirmation),
        "confirmation_images": len({row["image_id"] for row in confirmation}),
        "macro_auroc": auc,
        "delta_over_final": {
            name: auc[name] - auc["final_linear"] for name in comparisons
        },
        "image_cluster_bootstrap_delta_over_final": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-dev", type=Path, required=True)
    parser.add_argument("--huatuo-test", type=Path, required=True)
    parser.add_argument("--hulu-dev", type=Path, required=True)
    parser.add_argument("--hulu-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise RuntimeError("CPU-only audit; set CUDA_VISIBLE_DEVICES='' or '-1'")
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {
        "huatuo": (args.huatuo_dev, args.huatuo_test),
        "hulu": (args.hulu_dev, args.hulu_test),
    }
    analyses = {}
    for model, (development_path, confirmation_path) in sources.items():
        analyses[model] = analyze(
            load_rows(development_path),
            load_rows(confirmation_path),
            args.draws,
            args.seed,
        )
    gates = []
    for result in analyses.values():
        best_name = max(
            result["delta_over_final"], key=result["delta_over_final"].get
        )
        delta = result["delta_over_final"][best_name]
        ci = result["image_cluster_bootstrap_delta_over_final"][best_name]["ci95"]
        gates.append(delta >= 0.02 and ci[0] > 0)
        result["best_path_candidate"] = best_name
    output = {
        "version": VERSION,
        "status": "complete_cpu_secondary_fatal_audit",
        "decision": "PASS" if all(gates) else "NO_GO",
        "decision_rule": (
            "On both models, the best development-frozen native path object must add "
            ">=0.02 confirmation macro AUROC over final margin with image-bootstrap "
            "95% CI lower bound >0."
        ),
        "scope": (
            "Information screen for simple VLM-native path objects. A failure does not "
            "prove that every nonlinear functional is useless; action-channel degeneration "
            "is established separately. Labels were previously opened, so this is secondary."
        ),
        "analyses": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
