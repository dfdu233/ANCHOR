#!/usr/bin/env python3
"""L0 screen for incremental spatial dependence in claim-conditioned patch scores.

This deliberately reuses an already-opened confirmation panel.  It is a
candidate-killing screen, not a fresh confirmatory experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from anchor.corrected_sgta.analyze_sparse_patch_scan_v1 import (
    FINDINGS,
    SEED,
    arrays,
    build_features,
    hidden_rows,
    macro_auc,
    patch_artifact,
)


VERSION = "patch-dependence-l0-screen-v1"
BOOTSTRAP_DRAWS = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def edge_vectors(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.concatenate((grid[:, :-1].ravel(), grid[:-1, :].ravel()))
    right = np.concatenate((grid[:, 1:].ravel(), grid[1:, :].ravel()))
    return left, right


def graph_features(z: np.ndarray, groups: int, side: int) -> dict[str, float]:
    grids = z.reshape(groups, side, side)
    signed_products = []
    positive_products = []
    squared_differences = []
    moran_numerators = []
    moran_denominators = []
    for grid in grids:
        left, right = edge_vectors(grid)
        signed_products.append(np.mean(left * right))
        positive_products.append(np.mean(np.maximum(left, 0) * np.maximum(right, 0)))
        squared_differences.append(np.mean((left - right) ** 2))
        centered = grid - grid.mean()
        cleft, cright = edge_vectors(centered)
        moran_numerators.append(np.sum(cleft * cright))
        moran_denominators.append(np.sum(centered**2))
    denominator = max(float(np.sum(moran_denominators)), 1e-12)
    edge_count = groups * 2 * side * (side - 1)
    node_count = groups * side * side
    return {
        "neighbor_product": float(np.mean(signed_products)),
        "positive_neighbor_product": float(np.mean(positive_products)),
        "total_variation": float(np.mean(squared_differences)),
        "morans_i": float((node_count / edge_count) * np.sum(moran_numerators) / denominator),
    }


def attach_graph_features(
    dev_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    scores: np.ndarray,
    image_index: dict[str, int],
    groups: int,
    side: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    finding_index = {name: index for index, name in enumerate(FINDINGS)}
    null: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for finding in FINDINGS:
        column = finding_index[finding]
        negatives = [row for row in dev_rows if row["finding"] == finding and row["vote"] == 0]
        values = np.stack([scores[image_index[row["image_id"]], :, column] for row in negatives])
        mean, std = values.mean(0), values.std(0)
        positive_std = std[std > 0]
        floor = float(np.quantile(positive_std, 0.10)) if len(positive_std) else 1.0
        null[finding] = mean, np.maximum(std, floor)

    def transform(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            column = finding_index[row["finding"]]
            raw = scores[image_index[row["image_id"]], :, column]
            mean, std = null[row["finding"]]
            output.append({**row, **graph_features((raw - mean) / std, groups, side)})
        return output

    return transform(dev_rows), transform(test_rows)


def design(rows: list[dict[str, Any]], graph: bool) -> np.ndarray:
    fixed = np.column_stack(
        [np.asarray([row["finding"] == name for row in rows], dtype=float) for name in FINDINGS[:-1]]
    )
    base_names = (
        "final_margin",
        "patch_mean",
        "patch_max_z",
        "patch_top5_z",
        "multiscale_scan",
    )
    graph_names = (
        "neighbor_product",
        "positive_neighbor_product",
        "total_variation",
        "morans_i",
    )
    names = base_names + graph_names if graph else base_names
    numeric = np.column_stack([[float(row[name]) for row in rows] for name in names])
    return np.column_stack((fixed, numeric))


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fixed = len(FINDINGS) - 1
    mean = train[:, fixed:].mean(0)
    std = train[:, fixed:].std(0)
    std[std == 0] = 1.0
    left, right = train.copy(), test.copy()
    left[:, fixed:] = (left[:, fixed:] - mean) / std
    right[:, fixed:] = (right[:, fixed:] - mean) / std
    return left, right


def fit(dev_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, Any]:
    base_dev, base_test = standardize(design(dev_rows, False), design(test_rows, False))
    full_dev, full_test = standardize(design(dev_rows, True), design(test_rows, True))
    y_dev = np.asarray([row["label"] for row in dev_rows], dtype=int)
    base_model = LogisticRegression(C=0.1, max_iter=10000, random_state=SEED).fit(base_dev, y_dev)
    full_model = LogisticRegression(C=0.1, max_iter=10000, random_state=SEED).fit(full_dev, y_dev)
    return base_model.predict_proba(base_test)[:, 1], full_model.predict_proba(full_test)[:, 1], full_model


def bootstrap(rows: list[dict[str, Any]], base: np.ndarray, full: np.ndarray) -> dict[str, Any]:
    finding = np.asarray([row["finding"] for row in rows])
    label = np.asarray([row["label"] for row in rows], dtype=int)
    cells = [
        np.flatnonzero((finding == name) & (label == value))
        for name in FINDINGS
        for value in (0, 1)
    ]
    rng = np.random.default_rng(SEED + 14)
    auc_delta, nll_gain, brier_gain = [], [], []
    for _ in range(BOOTSTRAP_DRAWS):
        index = np.concatenate([rng.choice(cell, len(cell), replace=True) for cell in cells])
        y = label[index]
        auc_delta.append(macro_auc(finding[index], y, full[index]) - macro_auc(finding[index], y, base[index]))
        nll_gain.append(log_loss(y, base[index], labels=[0, 1]) - log_loss(y, full[index], labels=[0, 1]))
        brier_gain.append(np.mean((y - base[index]) ** 2) - np.mean((y - full[index]) ** 2))
    return {
        "draws": BOOTSTRAP_DRAWS,
        "macro_auroc_delta_ci95": np.quantile(auc_delta, [0.025, 0.975]).tolist(),
        "nll_improvement_ci95": np.quantile(nll_gain, [0.025, 0.975]).tolist(),
        "brier_improvement_ci95": np.quantile(brier_gain, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-hidden", type=Path, required=True)
    parser.add_argument("--confirmation-hidden", type=Path, required=True)
    parser.add_argument("--patch-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    dev_rows = hidden_rows(args.development_hidden)
    test_rows = hidden_rows(args.confirmation_hidden)
    scores, image_index, geometry = patch_artifact(args.patch_scores)
    dev_rows, test_rows, _ = build_features(
        dev_rows, test_rows, scores, image_index, geometry["groups"], geometry["side"]
    )
    dev_rows, test_rows = attach_graph_features(
        dev_rows, test_rows, scores, image_index, geometry["groups"], geometry["side"]
    )
    base, full, full_model = fit(dev_rows, test_rows)
    data = arrays(test_rows)
    y = data["label"]
    base_auc = macro_auc(data["finding"], y, base)
    full_auc = macro_auc(data["finding"], y, full)
    boot = bootstrap(test_rows, base, full)
    delta = full_auc - base_auc
    gate = bool(
        delta >= 0.02
        and boot["macro_auroc_delta_ci95"][0] > 0
        and boot["nll_improvement_ci95"][0] > 0
    )
    result = {
        "version": VERSION,
        "status": "complete",
        "scope": "exploratory L0 only; confirmation labels were previously opened",
        "inputs": {
            "development_hidden": str(args.development_hidden.resolve()),
            "confirmation_hidden": str(args.confirmation_hidden.resolve()),
            "patch_scores": str(args.patch_scores.resolve()),
            "patch_scores_sha256": sha256(args.patch_scores / "patch_scores.npz"),
            "source_sha256": sha256(Path(__file__)),
        },
        "features": {
            "base": "finding fixed effects + final margin + patch mean/max/top5 + multiscale scan",
            "added": "adjacent signed/positive products + total variation + Moran's I",
            "graph_coefficients": full_model.coef_[0, -4:].tolist(),
        },
        "result": {
            "n": len(test_rows),
            "base_macro_auroc": base_auc,
            "enhanced_macro_auroc": full_auc,
            "macro_auroc_delta": delta,
            "base_nll": float(log_loss(y, base, labels=[0, 1])),
            "enhanced_nll": float(log_loss(y, full, labels=[0, 1])),
            "base_brier": float(np.mean((y - base) ** 2)),
            "enhanced_brier": float(np.mean((y - full) ** 2)),
            "bootstrap": boot,
        },
        "gate": {
            "rule": "delta AUROC >= .02 and AUROC/NLL CI lower bounds > 0",
            "pass": gate,
            "decision": "PASS_TO_FRESH_CONFIRMATION" if gate else "NO_GO_PATCH_DEPENDENCE",
        },
        "claim_boundary": "A pass would show incremental spatial dependence in one cached Huatuo interface; it would not establish localization, causal control, mitigation, or cross-model generality.",
    }
    atomic_json(args.output, result)
    print(json.dumps({"result": result["result"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
