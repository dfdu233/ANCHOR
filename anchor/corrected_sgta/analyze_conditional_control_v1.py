#!/usr/bin/env python3
"""Blind screen of conditional normal controls on cached VinDr states.

The method asks whether a claim score should be interpreted relative to a
visually matched, same-finding negative control rather than relative to zero
or a global per-finding offset.  Candidate selection is dev-only and the
confirmation split is touched once after the specification is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


def load(directory: Path) -> dict:
    rows = [json.loads(line) for line in (directory / "metadata.jsonl").read_text().splitlines() if line]
    arrays = np.load(directory / "hidden_states.npz", allow_pickle=False)
    layers = np.asarray(arrays["layers"], dtype=int)
    margins = np.asarray([
        [float(row["diagnostic_plain_logit_lens"][str(layer)]["supported"])
         - float(row["diagnostic_plain_logit_lens"][str(layer)]["refuted"])
         for layer in layers]
        for row in rows
    ])
    return {
        "rows": rows,
        "layers": layers,
        "margin": margins,
        "style": np.asarray(arrays["visual_mean"], dtype=np.float32),
        "finding": np.asarray([row["finding"] for row in rows]),
        "votes": np.asarray([int(row["positive_votes"]) for row in rows]),
        "image": np.asarray([row["image_id"] for row in rows]),
    }


def threshold_for_bacc(y: np.ndarray, score: np.ndarray) -> float:
    values = np.unique(score)
    candidates = np.r_[values[0] - 1e-6, (values[:-1] + values[1:]) / 2, values[-1] + 1e-6]
    metrics = np.asarray([balanced_accuracy_score(y, score >= threshold) for threshold in candidates])
    best = np.flatnonzero(metrics == metrics.max())
    return float(candidates[best[np.argmin(np.abs(candidates[best]))]])


def fit_style(dev: dict, layer_index: int, components: int) -> tuple[StandardScaler, PCA, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dev["style"][:, layer_index])
    pca = PCA(n_components=min(components, len(scaled) - 1), random_state=42)
    return scaler, pca, pca.fit_transform(scaled)


def transform_style(data: dict, layer_index: int, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    return pca.transform(scaler.transform(data["style"][:, layer_index]))


def conditional_offset(
    query: dict,
    reference: dict,
    query_style: np.ndarray,
    reference_style: np.ndarray,
    margin_layer: int,
    k: int,
    leave_self_out: bool,
) -> np.ndarray:
    output = np.empty(len(query["rows"]), dtype=float)
    for index, (finding, image) in enumerate(zip(query["finding"], query["image"])):
        mask = (reference["finding"] == finding) & (reference["votes"] == 0)
        if leave_self_out:
            mask &= reference["image"] != image
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            raise ValueError(f"no negative control for {finding}")
        delta = reference_style[candidates] - query_style[index]
        distance = np.einsum("ij,ij->i", delta, delta)
        nearest = candidates[np.argsort(distance)[: min(k, len(candidates))]]
        weights = 1.0 / (np.sqrt(distance[np.argsort(distance)[: len(nearest)]]) + 1e-4)
        weights /= weights.sum()
        output[index] = float(np.dot(weights, reference["margin"][nearest, margin_layer]))
    return output


def global_offset(query: dict, dev: dict, margin_layer: int) -> np.ndarray:
    means = {
        finding: float(dev["margin"][(dev["finding"] == finding) & (dev["votes"] == 0), margin_layer].mean())
        for finding in np.unique(dev["finding"])
    }
    return np.asarray([means[finding] for finding in query["finding"]])


def clear(data: dict) -> np.ndarray:
    return (data["votes"] == 0) | (data["votes"] == 3)


def metrics(data: dict, score: np.ndarray, threshold: float) -> dict:
    mask = clear(data)
    y = (data["votes"][mask] == 3).astype(int)
    s = score[mask]
    prediction = s >= threshold
    probability = 1.0 / (1.0 + np.exp(-np.clip(s - threshold, -30, 30)))
    return {
        "n": int(mask.sum()),
        "auroc": float(roc_auc_score(y, s)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "brier_uncalibrated": float(brier_score_loss(y, probability)),
        "false_positive": int(((prediction == 1) & (y == 0)).sum()),
        "false_negative": int(((prediction == 0) & (y == 1)).sum()),
    }


def cluster_bootstrap(data: dict, candidate: np.ndarray, baseline: np.ndarray, ct: float, bt: float) -> dict:
    mask = clear(data)
    y = (data["votes"][mask] == 3).astype(int)
    images = data["image"][mask]
    a, b = candidate[mask], baseline[mask]
    unique = np.unique(images)
    rng = np.random.default_rng(42)
    auc_delta, bacc_delta = [], []
    for _ in range(5000):
        sampled = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([np.flatnonzero(images == item) for item in sampled])
        if len(np.unique(y[index])) < 2:
            continue
        auc_delta.append(roc_auc_score(y[index], a[index]) - roc_auc_score(y[index], b[index]))
        bacc_delta.append(
            balanced_accuracy_score(y[index], a[index] >= ct)
            - balanced_accuracy_score(y[index], b[index] >= bt)
        )
    return {
        "auroc_delta": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        "auroc_ci95": np.quantile(auc_delta, [0.025, 0.975]).tolist(),
        "bacc_delta": float(balanced_accuracy_score(y, a >= ct) - balanced_accuracy_score(y, b >= bt)),
        "bacc_ci95": np.quantile(bacc_delta, [0.025, 0.975]).tolist(),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dev, test = load(args.dev), load(args.confirmation)
    if not np.array_equal(dev["layers"], test["layers"]):
        raise ValueError("layer mismatch")
    dev_mask = clear(dev)
    dev_y = (dev["votes"][dev_mask] == 3).astype(int)
    final = len(dev["layers"]) - 1
    raw_dev, raw_test = dev["margin"][:, final], test["margin"][:, final]
    raw_threshold = threshold_for_bacc(dev_y, raw_dev[dev_mask])
    global_dev = raw_dev - global_offset(dev, dev, final)
    global_test = raw_test - global_offset(test, dev, final)
    global_threshold = threshold_for_bacc(dev_y, global_dev[dev_mask])
    candidates = []
    for style_layer in range(len(dev["layers"])):
        for components in (8, 16, 32):
            scaler, pca, dev_style = fit_style(dev, style_layer, components)
            test_style = transform_style(test, style_layer, scaler, pca)
            for k in (1, 3, 5, 8):
                dev_offset = conditional_offset(dev, dev, dev_style, dev_style, final, k, True)
                dev_score = raw_dev - dev_offset
                threshold = threshold_for_bacc(dev_y, dev_score[dev_mask])
                score = metrics(dev, dev_score, threshold)
                candidates.append((score["balanced_accuracy"], score["auroc"], -components, -k,
                                   style_layer, components, k, threshold, scaler, pca, test_style))
    selected = max(candidates, key=lambda item: item[:4])
    _, _, _, _, style_layer, components, k, conditional_threshold, scaler, pca, test_style = selected
    dev_style = transform_style(dev, style_layer, scaler, pca)
    conditional_dev = raw_dev - conditional_offset(dev, dev, dev_style, dev_style, final, k, True)
    conditional_test = raw_test - conditional_offset(test, dev, test_style, dev_style, final, k, False)
    result = {
        "status": "blind_confirmation_complete",
        "hypothesis": "a visually matched same-finding negative control removes patient-specific domain/style bias",
        "provenance": {
            "dev_hidden_sha256": sha256(args.dev / "hidden_states.npz"),
            "confirmation_hidden_sha256": sha256(args.confirmation / "hidden_states.npz"),
        },
        "frozen_selection": {
            "margin_layer": int(dev["layers"][final]),
            "style_layer": int(dev["layers"][style_layer]),
            "pca_components": components,
            "neighbors": k,
            "threshold": conditional_threshold,
        },
        "dev": {
            "raw": metrics(dev, raw_dev, raw_threshold),
            "global_control": metrics(dev, global_dev, global_threshold),
            "conditional_control": metrics(dev, conditional_dev, conditional_threshold),
        },
        "confirmation": {
            "raw": metrics(test, raw_test, raw_threshold),
            "global_control": metrics(test, global_test, global_threshold),
            "conditional_control": metrics(test, conditional_test, conditional_threshold),
            "conditional_vs_global": cluster_bootstrap(
                test, conditional_test, global_test, conditional_threshold, global_threshold
            ),
            "conditional_vs_raw": cluster_bootstrap(
                test, conditional_test, raw_test, conditional_threshold, raw_threshold
            ),
        },
        "decision_rule": (
            "continue only if conditional control beats global calibration by >=0.02 AUROC or BAcc "
            "with CI excluding zero on both Huatuo and Hulu"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
