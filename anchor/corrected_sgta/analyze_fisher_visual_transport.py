#!/usr/bin/env python3
"""Analyze a fixed-radius Fisher transport from image/neutral-view logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VERSION = "fisher-visual-transport-analysis-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def fisher_transport(
    image_logits: np.ndarray,
    neutral_logits: np.ndarray,
    rho: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Move a fixed local Fisher distance along visual log-evidence."""
    probability = softmax(np.asarray(image_logits, dtype=np.float64))
    residual = np.asarray(image_logits, dtype=np.float64) - np.asarray(
        neutral_logits, dtype=np.float64
    )
    centered = residual - np.sum(
        probability * residual, axis=-1, keepdims=True
    )
    fisher_norm = np.sqrt(
        np.sum(probability * np.square(centered), axis=-1, keepdims=True)
        + epsilon
    )
    return np.asarray(image_logits, dtype=np.float64) + rho * centered / fisher_norm


def cluster_bootstrap(
    differences: np.ndarray,
    clusters: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    unique = np.unique(clusters)
    by_cluster = {
        cluster: differences[clusters == cluster] for cluster in unique
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        values = np.concatenate([by_cluster[cluster] for cluster in sampled])
        estimates[index] = values.mean()
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def analyze(path: Path, rho: float, draws: int, seed: int) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("status") == "ok"]
    if not rows:
        raise ValueError(f"no successful rows: {path}")
    for row in rows:
        if row.get("style_names") != ["image", "blank"]:
            raise ValueError(f"expected image/blank cache: {path}")
    ground_truth = np.asarray([row["gt_index"] for row in rows], dtype=np.int64)
    baseline_prediction = []
    transported_prediction = []
    for row in rows:
        logits = np.asarray(row["style_logits"], dtype=np.float64)
        baseline_prediction.append(int(logits[0].argmax()))
        transported_prediction.append(
            int(
                fisher_transport(
                    logits[0][None, :], logits[1][None, :], rho
                )[0].argmax()
            )
        )
    baseline_prediction = np.asarray(baseline_prediction, dtype=np.int64)
    transported_prediction = np.asarray(
        transported_prediction, dtype=np.int64
    )
    baseline_correct = baseline_prediction == ground_truth
    transported_correct = transported_prediction == ground_truth
    difference = (
        transported_correct.astype(np.float64)
        - baseline_correct.astype(np.float64)
    )
    clusters = np.asarray(
        [str(row.get("img_name") or row["qid"]) for row in rows]
    )
    lower, upper = cluster_bootstrap(difference, clusters, draws, seed)
    return {
        "input": str(path.resolve()),
        "n": len(rows),
        "n_clusters": int(len(np.unique(clusters))),
        "rho": rho,
        "baseline_accuracy": float(baseline_correct.mean()),
        "transported_accuracy": float(transported_correct.mean()),
        "delta_pp": float(100.0 * difference.mean()),
        "cluster_bootstrap_95ci_pp": [100.0 * lower, 100.0 * upper],
        "rescues": int(np.sum(~baseline_correct & transported_correct)),
        "harms": int(np.sum(baseline_correct & ~transported_correct)),
        "prediction_change_rate": float(
            np.mean(baseline_prediction != transported_prediction)
        ),
    }


def main() -> None:
    args = parse_args()
    payload = {
        "version": VERSION,
        "formula": (
            "l'=l_image+rho*d/sqrt(E_p[d^2]+1e-8), "
            "d=(l_image-l_blank)-E_p[l_image-l_blank]"
        ),
        "predeclared_rho": args.rho,
        "bootstrap_draws": args.bootstrap,
        "seed": args.seed,
        "results": [
            analyze(path, args.rho, args.bootstrap, args.seed + index)
            for index, path in enumerate(args.input)
        ],
        "scope": (
            "full-coverage constrained-label diagnostic; blank is a visual "
            "reference, not proof of a recovered VLM source domain"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
