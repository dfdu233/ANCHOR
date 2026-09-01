#!/usr/bin/env python3
"""CPU gate for local counterexample geometry in a frozen CXR specialist.

This is an information upper-bound, not a mitigation method.  It asks whether
the position of a case relative to held-out positive/negative neighbours in the
full XRV logit space adds information beyond both the VLM claim margin and the
XRV score for that claim.  A within-finding label permutation is the placebo.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from anchor.corrected_sgta.screen_external_visual_increment_v1 import FINDINGS, load_claims
from anchor.corrected_sgta.screen_xrv_visual_increment_v1 import (
    FINDING_TARGETS,
    XRV_LABELS,
)


def load_logits(path: Path) -> dict[str, np.ndarray]:
    pack = np.load(path, allow_pickle=False)
    labels = [str(x) for x in pack["labels"]]
    if labels != list(XRV_LABELS):
        raise ValueError("XRV label order drift")
    return {
        str(image_id): np.asarray(vector, dtype=np.float64)
        for image_id, vector in zip(pack["image_ids"], pack["logits"])
    }


def attach_scalar(rows: list[dict[str, Any]], logits: dict[str, np.ndarray]) -> None:
    index = {name: i for i, name in enumerate(XRV_LABELS)}
    for row in rows:
        row["xrv_scalar"] = float(
            max(logits[row["image_id"]][index[name]] for name in FINDING_TARGETS[row["finding"]])
        )


def fit_standardizer(rows: list[dict[str, Any]], logits: dict[str, np.ndarray]):
    image_ids = sorted({row["image_id"] for row in rows})
    matrix = np.stack([logits[image_id] for image_id in image_ids])
    mean, scale = matrix.mean(0), matrix.std(0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def neighbor_score(
    query: np.ndarray,
    banks: dict[int, tuple[np.ndarray, np.ndarray]],
    k: int,
    query_id: str,
) -> float:
    distances = {}
    for label in (0, 1):
        matrix, image_ids = banks[label]
        eligible = image_ids != query_id
        matrix = matrix[eligible]
        if not len(matrix):
            raise ValueError("Leave-one-image-out removed an entire neighbour class")
        d = np.linalg.norm(matrix - query[None, :], axis=1)
        distances[label] = float(np.partition(d, min(k, len(d)) - 1)[:k].mean())
    return distances[0] - distances[1]


def attach_geometry(
    development: list[dict[str, Any]],
    target: list[dict[str, Any]],
    logits: dict[str, np.ndarray],
    k: int,
    seed: int,
) -> None:
    mean, scale = fit_standardizer(development, logits)
    standardized = {key: (value - mean) / scale for key, value in logits.items()}
    rng = np.random.default_rng(seed)
    for finding in FINDINGS:
        dev = [row for row in development if row["finding"] == finding]
        banks = {
            label: (
                np.stack([standardized[row["image_id"]] for row in dev if row["label"] == label]),
                np.asarray([row["image_id"] for row in dev if row["label"] == label]),
            )
            for label in (0, 1)
        }
        shuffled = np.asarray([row["label"] for row in dev], dtype=np.int64)
        rng.shuffle(shuffled)
        placebo_banks = {
            label: (
                np.stack([standardized[row["image_id"]] for row, assigned in zip(dev, shuffled) if assigned == label]),
                np.asarray([row["image_id"] for row, assigned in zip(dev, shuffled) if assigned == label]),
            )
            for label in (0, 1)
        }
        for row in target:
            if row["finding"] != finding:
                continue
            z = standardized[row["image_id"]]
            row["neighbor_boundary"] = neighbor_score(z, banks, k, row["image_id"])
            row["placebo_boundary"] = neighbor_score(z, placebo_banks, k, row["image_id"])


def stats(rows: list[dict[str, Any]], fields: tuple[str, ...]):
    out = {}
    for finding in FINDINGS:
        selected = [row for row in rows if row["finding"] == finding]
        out[finding] = {}
        for field in fields:
            values = np.asarray([row[field] for row in selected])
            out[finding][field] = (float(values.mean()), float(max(values.std(), 1e-8)))
    return out


def design(rows, normalization, fields):
    matrix = []
    for row in rows:
        onehot = [float(row["finding"] == finding) for finding in FINDINGS[:-1]]
        normalized = [
            (row[field] - normalization[row["finding"]][field][0])
            / normalization[row["finding"]][field][1]
            for field in fields
        ]
        matrix.append(onehot + normalized)
    return np.asarray(matrix, dtype=np.float64)


def macro_auc(rows, score):
    values = []
    for finding in FINDINGS:
        idx = [i for i, row in enumerate(rows) if row["finding"] == finding]
        values.append(roc_auc_score([rows[i]["label"] for i in idx], score[idx]))
    return float(np.mean(values))


def fit_predict(dev, test, fields, seed):
    normal = stats(dev, fields)
    model = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
    model.fit(design(dev, normal, fields), [row["label"] for row in dev])
    return model.predict_proba(design(test, normal, fields))[:, 1]


def metrics(rows, probability):
    y = np.asarray([row["label"] for row in rows])
    return {
        "macro_auroc": macro_auc(rows, probability),
        "nll": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
    }


def bootstrap(rows, predictions, draws, seed):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["image_id"]].append(i)
    ids = sorted(groups)
    rng = np.random.default_rng(seed)
    comparisons = {name: [] for name in predictions if name != "vlm_plus_xrv"}
    for _ in range(draws):
        sampled = rng.choice(ids, len(ids), replace=True)
        idx = np.asarray([i for image_id in sampled for i in groups[image_id]])
        sampled_rows = [rows[i] for i in idx]
        base = macro_auc(sampled_rows, predictions["vlm_plus_xrv"][idx])
        for name in comparisons:
            comparisons[name].append(macro_auc(sampled_rows, predictions[name][idx]) - base)
    return {
        name: {
            "mean": float(np.mean(values)),
            "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
        }
        for name, values in comparisons.items()
    }


def analyze(dev, test, seed, draws):
    specifications = {
        "vlm_only": ("margin",),
        "vlm_plus_xrv": ("margin", "xrv_scalar"),
        "true_boundary": ("margin", "xrv_scalar", "neighbor_boundary"),
        "placebo_boundary": ("margin", "xrv_scalar", "placebo_boundary"),
    }
    predictions = {name: fit_predict(dev, test, fields, seed) for name, fields in specifications.items()}
    points = {name: metrics(test, probability) for name, probability in predictions.items()}
    return {
        "points": points,
        "deltas_over_vlm_plus_xrv": {
            name: {key: points[name][key] - points["vlm_plus_xrv"][key] for key in points[name]}
            for name in ("true_boundary", "placebo_boundary")
        },
        "image_bootstrap_auroc_delta_over_vlm_plus_xrv": bootstrap(test, predictions, draws, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-dev", type=Path, required=True)
    parser.add_argument("--huatuo-test", type=Path, required=True)
    parser.add_argument("--hulu-dev", type=Path, required=True)
    parser.add_argument("--hulu-test", type=Path, required=True)
    parser.add_argument("--xrv-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise RuntimeError("CPU-only gate; set CUDA_VISIBLE_DEVICES=''")
    if args.output.exists():
        raise FileExistsError(args.output)
    logits = load_logits(args.xrv_logits)
    sources = {
        "huatuo": (args.huatuo_dev, args.huatuo_test),
        "hulu": (args.hulu_dev, args.hulu_test),
    }
    analyses = {}
    for model, (dev_path, test_path) in sources.items():
        dev = load_claims(dev_path, "development", model)
        test = load_claims(test_path, "confirmation", model)
        attach_scalar(dev + test, logits)
        attach_geometry(dev, dev + test, logits, args.k, args.seed)
        analyses[model] = analyze(dev, test, args.seed, args.draws)
    gates = []
    for analysis in analyses.values():
        delta = analysis["deltas_over_vlm_plus_xrv"]["true_boundary"]["macro_auroc"]
        ci = analysis["image_bootstrap_auroc_delta_over_vlm_plus_xrv"]["true_boundary"]["ci95"]
        placebo = analysis["deltas_over_vlm_plus_xrv"]["placebo_boundary"]["macro_auroc"]
        gates.append(delta >= .02 and ci[0] > 0 and delta > placebo)
    result = {
        "status": "complete_cpu_upper_bound",
        "decision": "PASS" if all(gates) else "NO_GO",
        "boundary": "Counterexample geometry is standard supervised kNN and is not a mitigation contribution.",
        "gate": "Both VLMs: true boundary adds >=.02 macro AUROC over VLM+XRV scalar, CI lower>0, and exceeds shuffled-label placebo.",
        "k": args.k,
        "analyses": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
