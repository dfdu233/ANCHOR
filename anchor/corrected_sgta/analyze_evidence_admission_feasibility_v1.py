#!/usr/bin/env python3
"""Blind feasibility screen for intermediate clinical evidence admission.

Fits only a nearest-centroid direction per finding on the development split.
The best non-final layer/family is frozen on development, then compared with
the same-family final layer on the independent confirmation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def load(directory: Path) -> dict:
    rows = [json.loads(line) for line in (directory / "metadata.jsonl").read_text().splitlines() if line]
    arrays = np.load(directory / "hidden_states.npz", allow_pickle=False)
    return {
        "rows": rows,
        "layers": np.asarray(arrays["layers"], dtype=int),
        "claim": np.asarray(arrays["claim"], dtype=np.float32),
        "visual_mean": np.asarray(arrays["visual_mean"], dtype=np.float32),
        "finding": np.asarray([row["finding"] for row in rows]),
        "votes": np.asarray([int(row["positive_votes"]) for row in rows]),
        "image": np.asarray([row["image_id"] for row in rows]),
    }


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fit_directions(data: dict, features: np.ndarray) -> dict[str, np.ndarray]:
    x = unit(features.astype(np.float64))
    output = {}
    for finding in np.unique(data["finding"]):
        neg = x[(data["finding"] == finding) & (data["votes"] == 0)]
        pos = x[(data["finding"] == finding) & (data["votes"] == 3)]
        direction = pos.mean(0) - neg.mean(0)
        output[finding] = direction / max(np.linalg.norm(direction), 1e-12)
    return output


def score(data: dict, features: np.ndarray, directions: dict[str, np.ndarray]) -> np.ndarray:
    x = unit(features.astype(np.float64))
    return np.asarray([x[i] @ directions[f] for i, f in enumerate(data["finding"])])


def threshold(y: np.ndarray, score: np.ndarray) -> float:
    values = np.unique(score)
    grid = np.r_[values[0] - 1e-8, (values[:-1] + values[1:]) / 2, values[-1] + 1e-8]
    return float(max(grid, key=lambda value: balanced_accuracy_score(y, score >= value)))


def evaluate(data: dict, score: np.ndarray, rules: dict[str, float]) -> dict:
    clear = np.isin(data["votes"], (0, 3))
    labels = (data["votes"] == 3).astype(int)
    by_finding = {}
    predictions = np.zeros(len(score), dtype=int)
    for finding in np.unique(data["finding"]):
        mask = clear & (data["finding"] == finding)
        predictions[mask] = score[mask] >= rules[finding]
        by_finding[finding] = {
            "auroc": float(roc_auc_score(labels[mask], score[mask])),
            "balanced_accuracy": float(balanced_accuracy_score(labels[mask], predictions[mask])),
        }
    return {
        "macro_auroc": float(np.mean([row["auroc"] for row in by_finding.values()])),
        "macro_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in by_finding.values()])),
        "by_finding": by_finding,
        "predictions": predictions,
    }


def fit_rules(data: dict, score_values: np.ndarray) -> dict[str, float]:
    labels = (data["votes"] == 3).astype(int)
    return {
        finding: threshold(
            labels[(data["finding"] == finding) & np.isin(data["votes"], (0, 3))],
            score_values[(data["finding"] == finding) & np.isin(data["votes"], (0, 3))],
        )
        for finding in np.unique(data["finding"])
    }


def bootstrap(test: dict, candidate: np.ndarray, final: np.ndarray) -> dict:
    clear = np.isin(test["votes"], (0, 3))
    labels = (test["votes"] == 3).astype(int)
    images = np.unique(test["image"][clear])
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(5000):
        sampled = rng.choice(images, len(images), replace=True)
        index = np.concatenate([np.flatnonzero((test["image"] == image) & clear) for image in sampled])
        if len(np.unique(labels[index])) < 2:
            continue
        deltas.append(roc_auc_score(labels[index], candidate[index]) - roc_auc_score(labels[index], final[index]))
    return {"pooled_auroc_delta_ci95": np.quantile(deltas, [0.025, 0.975]).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dev, test = load(args.dev), load(args.confirmation)
    candidates = {}
    caches = {}
    for family in ("claim", "visual_mean"):
        for layer_index, layer in enumerate(dev["layers"]):
            directions = fit_directions(dev, dev[family][:, layer_index])
            dev_score = score(dev, dev[family][:, layer_index], directions)
            test_score = score(test, test[family][:, layer_index], directions)
            rules = fit_rules(dev, dev_score)
            candidates[f"{family}:{layer}"] = {
                "development": {k: v for k, v in evaluate(dev, dev_score, rules).items() if k != "predictions"},
                "confirmation": {k: v for k, v in evaluate(test, test_score, rules).items() if k != "predictions"},
            }
            caches[(family, int(layer))] = (test_score, rules)
    nonfinal = [key for key in candidates if int(key.split(":")[1]) != int(dev["layers"][-1])]
    selected = max(nonfinal, key=lambda key: candidates[key]["development"]["macro_auroc"])
    family, layer_text = selected.split(":")
    final_key = f"{family}:{int(dev['layers'][-1])}"
    selected_score, _ = caches[(family, int(layer_text))]
    final_score, _ = caches[(family, int(dev["layers"][-1]))]
    result = {
        "status": "blind_confirmation_complete",
        "selected_from_development": selected,
        "matched_final_comparator": final_key,
        "selected_confirmation": candidates[selected]["confirmation"],
        "final_confirmation": candidates[final_key]["confirmation"],
        "macro_auroc_delta": candidates[selected]["confirmation"]["macro_auroc"] - candidates[final_key]["confirmation"]["macro_auroc"],
        "bootstrap": bootstrap(test, selected_score, final_score),
        "all_candidates": candidates,
        "decision_rule": "advance evidence-admission training only if non-final macro AUROC exceeds matched final by >=0.02 on both models",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "all_candidates"}, indent=2))


if __name__ == "__main__":
    main()
