#!/usr/bin/env python3
"""Preregistered CPU gate for partition-coherent local evidence pooling.

This is deliberately a calibration-only mechanism test.  Patch scores are not
treated as likelihood ratios.  Development vote-0 images turn each fixed
location score into a rank p-value, a frozen p-to-e calibrator turns that into
local evidence, and fixed uniform prior mass is averaged over a true partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from anchor.corrected_sgta.analyze_sparse_patch_scan_v1 import (
    BOOTSTRAP_DRAWS,
    FINDINGS,
    SEED,
    hidden_rows,
    patch_artifact,
)


VERSION = "evidence-conserving-partition-pool-v1"
PARTITIONS = (16, 64, 576)
GAMMA = 0.5


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


def block_means(values: np.ndarray, side: int, count: int) -> np.ndarray:
    root = int(round(count ** 0.5))
    if root * root != count or side % root:
        raise ValueError(f"partition {count} is incompatible with {side}x{side}")
    block = side // root
    grid = values.reshape(side, side)
    return grid.reshape(root, block, root, block).mean((1, 3)).reshape(-1)


def build_rows(
    dev_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]],
    scores: np.ndarray, image_index: dict[str, int], side: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    finding_index = {name: index for index, name in enumerate(FINDINGS)}
    dev_null: dict[tuple[str, int], np.ndarray] = {}
    for finding in FINDINGS:
        column = finding_index[finding]
        indices = [image_index[r["image_id"]] for r in dev_rows if r["finding"] == finding and r["vote"] == 0]
        for count in PARTITIONS:
            matrix = np.stack([block_means(scores[index, :, column], side, count) for index in indices])
            dev_null[(finding, count)] = matrix

    def transform(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            finding = row["finding"]
            column = finding_index[finding]
            raw_patch = scores[image_index[row["image_id"]], :, column]
            values: dict[str, float] = {
                "patch_mean": float(raw_patch.mean()),
                "patch_max": float(raw_patch.max()),
                "patch_top5": float(np.mean(np.partition(raw_patch, -max(1, int(np.ceil(.05 * len(raw_patch)))))[-max(1, int(np.ceil(.05 * len(raw_patch)))):])),
            }
            for count in PARTITIONS:
                region = block_means(raw_patch, side, count)
                null = dev_null[(finding, count)]
                # Position-specific upper-tail rank p-values; no test labels used.
                p = (1.0 + (null >= region[None, :]).sum(0)) / (null.shape[0] + 1.0)
                local_e = GAMMA * np.power(p, GAMMA - 1.0)
                values[f"raw_max_{count}"] = float(region.max())
                values[f"e_mix_{count}"] = float(local_e.mean())
            output.append({**row, **values})
        return output

    return transform(dev_rows), transform(test_rows)


def macro_auc(finding: np.ndarray, label: np.ndarray, score: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(label[finding == name], score[finding == name]) for name in FINDINGS]))


def design(rows: list[dict[str, Any]], enhanced: bool) -> np.ndarray:
    fixed = np.column_stack([[r["finding"] == name for r in rows] for name in FINDINGS[:-1]]).astype(float)
    names = ["final_margin", "patch_mean", "patch_max", "patch_top5"]
    if enhanced:
        names += [f"e_mix_{count}" for count in PARTITIONS]
    numeric = np.asarray([[r[name] for name in names] for r in rows], dtype=float)
    return np.column_stack([fixed, numeric])


def standardize(dev: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right = dev.copy(), test.copy()
    start = len(FINDINGS) - 1
    mean, std = left[:, start:].mean(0), left[:, start:].std(0)
    std[std == 0] = 1.0
    left[:, start:] = (left[:, start:] - mean) / std
    right[:, start:] = (right[:, start:] - mean) / std
    return left, right


def fit(dev_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], enhanced: bool) -> tuple[np.ndarray, float]:
    x_dev, x_test = standardize(design(dev_rows, enhanced), design(test_rows, enhanced))
    y_dev = np.asarray([r["label"] for r in dev_rows], dtype=int)
    model = LogisticRegression(C=.1, max_iter=10000, random_state=SEED).fit(x_dev, y_dev)
    return model.predict_proba(x_test)[:, 1], float(model.coef_[0, -1])


def bootstrap(rows: list[dict[str, Any]], base: np.ndarray, enhanced: np.ndarray) -> dict[str, Any]:
    finding = np.asarray([r["finding"] for r in rows])
    label = np.asarray([r["label"] for r in rows], dtype=int)
    cells = [np.flatnonzero((finding == name) & (label == y)) for name in FINDINGS for y in (0, 1)]
    rng = np.random.default_rng(SEED)
    auc_delta, nll_delta = [], []
    for _ in range(BOOTSTRAP_DRAWS):
        index = np.concatenate([rng.choice(cell, len(cell), replace=True) for cell in cells])
        auc_delta.append(macro_auc(finding[index], label[index], enhanced[index]) - macro_auc(finding[index], label[index], base[index]))
        nll_delta.append(log_loss(label[index], base[index], labels=[0, 1]) - log_loss(label[index], enhanced[index], labels=[0, 1]))
    return {
        "draws": BOOTSTRAP_DRAWS,
        "macro_auroc_delta_ci95": np.quantile(auc_delta, [.025, .975]).tolist(),
        "nll_improvement_ci95": np.quantile(nll_delta, [.025, .975]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
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
    if geometry["groups"] != 1 or geometry["side"] != 24:
        raise ValueError(f"first protocol is frozen to Huatuo 1x24x24, got {geometry}")
    dev_rows, test_rows = build_rows(dev_rows, test_rows, scores, image_index, geometry["side"])
    base, _ = fit(dev_rows, test_rows, False)
    enhanced, last_coef = fit(dev_rows, test_rows, True)
    finding = np.asarray([r["finding"] for r in test_rows])
    label = np.asarray([r["label"] for r in test_rows], dtype=int)
    base_auc, enhanced_auc = macro_auc(finding, label, base), macro_auc(finding, label, enhanced)
    by_finding = {}
    for name in FINDINGS:
        mask = finding == name
        by_finding[name] = float(roc_auc_score(label[mask], enhanced[mask]) - roc_auc_score(label[mask], base[mask]))
    boot = bootstrap(test_rows, base, enhanced)
    positive = sum(value > 0 for value in by_finding.values())
    gate = bool(enhanced_auc - base_auc >= .02 and boot["macro_auroc_delta_ci95"][0] > 0 and boot["nll_improvement_ci95"][0] > 0 and positive >= 5)

    negatives = [r for r in test_rows if r["label"] == 0]
    result = {
        "version": VERSION,
        "status": "complete",
        "model": args.model,
        "development_n": len(dev_rows),
        "confirmation_n": len(test_rows),
        "configuration": {"partitions": list(PARTITIONS), "p_to_e_gamma": GAMMA, "seed": SEED, "source_sha256": sha256(Path(__file__))},
        "null_partition_drift": {
            str(count): {
                "raw_max_mean": float(np.mean([r[f"raw_max_{count}"] for r in negatives])),
                "e_mix_mean": float(np.mean([r[f"e_mix_{count}"] for r in negatives])),
            } for count in PARTITIONS
        },
        "confirmation": {
            "base_macro_auroc": base_auc,
            "enhanced_macro_auroc": enhanced_auc,
            "macro_auroc_delta": enhanced_auc - base_auc,
            "base_nll": float(log_loss(label, base, labels=[0, 1])),
            "enhanced_nll": float(log_loss(label, enhanced, labels=[0, 1])),
            "by_finding_delta": by_finding,
            "positive_finding_count": positive,
            "last_e_mix_coefficient": last_coef,
            "bootstrap": boot,
        },
        "gate": {"rule": "AUROC +0.02; AUROC/NLL CI lower>0; >=5/7 finding deltas positive", "pass": gate},
        "boundary": "A pass is incremental calibrated pooling evidence, not localization, causal control, or hallucination mitigation.",
        "command": " ".join(sys.argv),
    }
    atomic_json(args.output, result)
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
