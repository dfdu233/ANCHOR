#!/usr/bin/env python3
"""Fatal screen for cross-source *spatial* evidence beyond score ensembling.

The test is deliberately stronger than showing that two CXR specialists are
useful.  It asks whether agreement in where their exact class evidence lies
adds held-out information after the VLM margin and both specialist logits are
already known.  Failure closes this candidate rather than motivating a new
threshold.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from anchor.corrected_sgta.screen_external_visual_increment_v1 import load_claims, sha256_file


FINDING_TO_LABEL = {"cardiomegaly": "Cardiomegaly", "pleural_effusion": "Effusion"}


def load_cams(path: Path):
    payload = np.load(path)
    ids = payload["image_ids"].astype(str).tolist()
    domains = payload["domains"].astype(str).tolist()
    labels = payload["labels"].astype(str).tolist()
    logits = payload["logits"].astype(np.float64)
    cams = payload["cams"].astype(np.float64)
    if len(domains) != 2:
        raise ValueError("The preregistered pilot requires exactly two source experts")
    return domains, labels, {image_id: (logits[:, i], cams[:, i]) for i, image_id in enumerate(ids)}


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    output = np.exp(shifted)
    return output / np.sum(output)


def spatial_features(cams: np.ndarray) -> np.ndarray:
    """Three scale-free measures of source agreement in patch support."""
    flat = cams.reshape(2, -1)
    centered = flat - flat.mean(axis=1, keepdims=True)
    scaled = centered / np.maximum(centered.std(axis=1, keepdims=True), 1e-6)
    cosine = float(np.dot(scaled[0], scaled[1]) / max(np.linalg.norm(scaled[0]) * np.linalg.norm(scaled[1]), 1e-6))
    distributions = np.stack([softmax(row) for row in scaled])
    overlap = float(np.minimum(distributions[0], distributions[1]).sum())
    positive_intersection = float(np.minimum(np.maximum(scaled[0], 0), np.maximum(scaled[1], 0)).mean())
    return np.asarray([cosine, overlap, positive_intersection], dtype=np.float64)


def matrix(rows, labels, encoded, with_spatial: bool) -> tuple[np.ndarray, np.ndarray]:
    features, targets = [], []
    for row in rows:
        label_index = labels.index(FINDING_TO_LABEL[row["finding"]])
        logits, cams = encoded[row["image_id"]]
        values = [float(row["margin"]), *logits[:, label_index].tolist()]
        if with_spatial:
            values.extend(spatial_features(cams[:, label_index]).tolist())
        features.append(values)
        targets.append(row["label"])
    return np.asarray(features), np.asarray(targets, dtype=np.int64)


def fit_predict(dev_rows, test_rows, labels, encoded, with_spatial: bool):
    x_dev, y_dev = matrix(dev_rows, labels, encoded, with_spatial)
    x_test, y_test = matrix(test_rows, labels, encoded, with_spatial)
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42),
    )
    estimator.fit(x_dev, y_dev)
    probability = estimator.predict_proba(x_test)[:, 1]
    return y_test, probability


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def paired_bootstrap(rows, y, scalar_p, spatial_p, draws: int, seed: int) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["image_id"]].append(i)
    image_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    values = {"auroc_delta": [], "nll_reduction": [], "brier_reduction": []}
    for _ in range(draws):
        sampled = rng.choice(image_ids, len(image_ids), replace=True)
        idx = np.asarray([i for image_id in sampled for i in groups[image_id]])
        if len(np.unique(y[idx])) < 2:
            continue
        left, right = metrics(y[idx], scalar_p[idx]), metrics(y[idx], spatial_p[idx])
        values["auroc_delta"].append(right["auroc"] - left["auroc"])
        values["nll_reduction"].append(left["nll"] - right["nll"])
        values["brier_reduction"].append(left["brier"] - right["brier"])
    result = {}
    for name, samples in values.items():
        array = np.asarray(samples)
        result[name] = {
            "mean": float(array.mean()),
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-dev", type=Path, required=True)
    parser.add_argument("--hulu-dev", type=Path, required=True)
    parser.add_argument("--huatuo-confirmation", type=Path, required=True)
    parser.add_argument("--hulu-confirmation", type=Path, required=True)
    parser.add_argument("--expert-cams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    domains, labels, encoded = load_cams(args.expert_cams)
    findings = set(FINDING_TO_LABEL)
    paths = {
        "huatuo": (args.huatuo_dev, args.huatuo_confirmation),
        "hulu": (args.hulu_dev, args.hulu_confirmation),
    }
    analyses = {}
    passes = []
    for model, (dev_path, test_path) in paths.items():
        dev = [row for row in load_claims(dev_path, "dev", model) if row["finding"] in findings]
        test = [row for row in load_claims(test_path, "confirmation", model) if row["finding"] in findings]
        y_scalar, scalar = fit_predict(dev, test, labels, encoded, with_spatial=False)
        y_spatial, spatial = fit_predict(dev, test, labels, encoded, with_spatial=True)
        if not np.array_equal(y_scalar, y_spatial):
            raise RuntimeError("Target drift between arms")
        bootstrap = paired_bootstrap(test, y_scalar, scalar, spatial, args.bootstrap_draws, args.seed)
        analyses[model] = {
            "n_dev": len(dev),
            "n_confirmation": len(test),
            "margin_plus_two_source_logits": metrics(y_scalar, scalar),
            "plus_three_spatial_consensus_features": metrics(y_spatial, spatial),
            "paired_bootstrap": bootstrap,
        }
        passes.append(
            bootstrap["auroc_delta"]["mean"] >= 0.02
            and bootstrap["auroc_delta"]["ci95"][0] > 0
            and bootstrap["nll_reduction"]["ci95"][0] > 0
        )

    result = {
        "status": "complete",
        "decision": "PASS_L0" if all(passes) else "NO_GO_L0",
        "command": shlex.join(sys.argv),
        "domains": domains,
        "findings": sorted(findings),
        "input_sha256": {
            "expert_cams": sha256_file(args.expert_cams),
            **{f"{model}_{split}": sha256_file(path) for model, pair in paths.items() for split, path in zip(("dev", "confirmation"), pair)},
        },
        "preregistered_gate": {
            "comparison": "spatial consensus must improve over VLM margin + both specialist logits",
            "models": "Huatuo and Hulu must both pass",
            "auroc_delta": ">=0.02 and image-bootstrap CI lower > 0",
            "nll_reduction": "image-bootstrap CI lower > 0",
            "failure_action": "close cross-source spatial-consensus candidate; do not tune a new threshold",
        },
        "analyses": analyses,
        "boundary": "This is only a fatal information screen. PASS would justify a localization and causal token-gating experiment; it would not by itself establish a mitigation method.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
