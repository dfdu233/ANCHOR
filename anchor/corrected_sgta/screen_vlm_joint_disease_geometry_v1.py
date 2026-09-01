#!/usr/bin/env python3
"""Cache-only screen for joint disease geometry in VLM visual tokens.

This does not define a mitigation method.  It asks a narrower question needed
before considering a graphical-model generation operator: do the six off-claim
coordinates in the VLM's own projected visual-token field add held-out label
information beyond the final answer margin and the target-claim coordinate?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


VERSION = "vlm-joint-disease-geometry-screen-v1"
FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hidden(directory: Path) -> list[dict[str, Any]]:
    output = []
    for line in (directory / "metadata.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("finding") not in FINDINGS or int(row.get("positive_votes", -1)) not in (0, 3):
            continue
        layers = sorted(int(value) for value in row["diagnostic_plain_logit_lens"])
        logits = row["diagnostic_plain_logit_lens"][str(layers[-1])]
        output.append(
            {
                "image_id": str(row["image_id"]),
                "finding": str(row["finding"]),
                "label": int(int(row["positive_votes"]) == 3),
                "final_margin": float(logits["supported"] - logits["refuted"]),
            }
        )
    return output


def load_patch_means(directory: Path) -> tuple[np.ndarray, dict[str, int]]:
    scores = np.asarray(np.load(directory / "patch_scores.npz")["patch_scores"], dtype=float)
    metadata = [json.loads(line) for line in (directory / "metadata.jsonl").read_text().splitlines()]
    if scores.shape[0] != len(metadata) or scores.shape[2] != len(FINDINGS):
        raise ValueError("patch score shape or metadata mismatch")
    index = {str(row["image_id"]): position for position, row in enumerate(metadata)}
    if len(index) != len(metadata):
        raise ValueError("patch metadata repeats image IDs")
    return scores.mean(axis=1), index


def design(rows: list[dict[str, Any]], means: np.ndarray, index: dict[str, int], joint: bool) -> np.ndarray:
    fixed = np.column_stack([(np.asarray([row["finding"] for row in rows]) == name).astype(float) for name in FINDINGS[:-1]])
    final = np.asarray([row["final_margin"] for row in rows], dtype=float)[:, None]
    target = np.asarray(
        [means[index[row["image_id"]], FINDINGS.index(row["finding"])] for row in rows],
        dtype=float,
    )[:, None]
    columns = [fixed, final, target]
    if joint:
        columns.append(np.stack([means[index[row["image_id"]]] for row in rows]))
    return np.concatenate(columns, axis=1)


def standardize(dev: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fixed = len(FINDINGS) - 1
    left, right = dev.copy(), test.copy()
    mean = left[:, fixed:].mean(axis=0)
    std = left[:, fixed:].std(axis=0)
    std[std == 0] = 1.0
    left[:, fixed:] = (left[:, fixed:] - mean) / std
    right[:, fixed:] = (right[:, fixed:] - mean) / std
    return left, right


def macro_auc(rows: list[dict[str, Any]], probability: np.ndarray) -> float:
    finding = np.asarray([row["finding"] for row in rows])
    label = np.asarray([row["label"] for row in rows])
    return float(np.mean([roc_auc_score(label[finding == name], probability[finding == name]) for name in FINDINGS]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-hidden", type=Path, required=True)
    parser.add_argument("--confirmation-hidden", type=Path, required=True)
    parser.add_argument("--patch-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    dev_rows = load_hidden(args.development_hidden)
    test_rows = load_hidden(args.confirmation_hidden)
    means, index = load_patch_means(args.patch_scores)
    for row in dev_rows + test_rows:
        if row["image_id"] not in index:
            raise ValueError(f"missing patch scores for {row['image_id']}")

    y_dev = np.asarray([row["label"] for row in dev_rows])
    y_test = np.asarray([row["label"] for row in test_rows])
    base_dev, base_test = standardize(
        design(dev_rows, means, index, False), design(test_rows, means, index, False)
    )
    joint_dev, joint_test = standardize(
        design(dev_rows, means, index, True), design(test_rows, means, index, True)
    )
    base_model = LogisticRegression(C=0.1, max_iter=10000, random_state=args.seed).fit(base_dev, y_dev)
    joint_model = LogisticRegression(C=0.1, max_iter=10000, random_state=args.seed).fit(joint_dev, y_dev)
    base = base_model.predict_proba(base_test)[:, 1]
    joint = joint_model.predict_proba(joint_test)[:, 1]
    base_auc, joint_auc = macro_auc(test_rows, base), macro_auc(test_rows, joint)

    cells = {
        (finding, label): np.asarray(
            [i for i, row in enumerate(test_rows) if row["finding"] == finding and row["label"] == label]
        )
        for finding in FINDINGS
        for label in (0, 1)
    }
    rng = np.random.default_rng(args.seed)
    auc_delta, nll_improvement, brier_improvement = [], [], []
    for _ in range(args.bootstrap_draws):
        chosen = np.concatenate([rng.choice(cell, len(cell), replace=True) for cell in cells.values()])
        sampled = [test_rows[i] for i in chosen]
        left, right, y = base[chosen], joint[chosen], y_test[chosen]
        auc_delta.append(macro_auc(sampled, right) - macro_auc(sampled, left))
        nll_improvement.append(log_loss(y, left, labels=[0, 1]) - log_loss(y, right, labels=[0, 1]))
        brier_improvement.append(np.mean((y - left) ** 2) - np.mean((y - right) ** 2))

    result = {
        "version": VERSION,
        "status": "complete_cache_only",
        "scope": "one-model secondary screen on previously opened confirmation labels",
        "inputs": {
            "development_hidden": str(args.development_hidden.resolve()),
            "confirmation_hidden": str(args.confirmation_hidden.resolve()),
            "patch_scores": str(args.patch_scores.resolve()),
            "patch_scores_sha256": sha256(args.patch_scores / "patch_scores.npz"),
        },
        "features": {
            "base": "finding fixed effects + final answer margin + target finding projected-token mean",
            "joint": "base + all seven projected-token finding means",
            "fit": "development-only logistic regression, C=0.1",
        },
        "result": {
            "development_n": len(dev_rows),
            "confirmation_n": len(test_rows),
            "base_macro_auroc": base_auc,
            "joint_macro_auroc": joint_auc,
            "macro_auroc_delta": joint_auc - base_auc,
            "nll_improvement": float(log_loss(y_test, base, labels=[0, 1]) - log_loss(y_test, joint, labels=[0, 1])),
            "brier_improvement": float(np.mean((y_test - base) ** 2) - np.mean((y_test - joint) ** 2)),
            "bootstrap": {
                "draws": args.bootstrap_draws,
                "macro_auroc_delta_ci95": np.quantile(auc_delta, [0.025, 0.975]).tolist(),
                "nll_improvement_ci95": np.quantile(nll_improvement, [0.025, 0.975]).tolist(),
                "brier_improvement_ci95": np.quantile(brier_improvement, [0.025, 0.975]).tolist(),
            },
        },
        "gate": {
            "rule": "delta macro AUROC >= .02 and AUROC/NLL/Brier CI lower bounds > 0",
            "pass": bool(
                joint_auc - base_auc >= 0.02
                and np.quantile(auc_delta, 0.025) > 0
                and np.quantile(nll_improvement, 0.025) > 0
                and np.quantile(brier_improvement, 0.025) > 0
            ),
        },
        "boundary": (
            "A pass would establish joint visual decodability only. DPP, graph energy, or "
            "hypergraph decoding would still be selection or logit reweighting, not a new operator."
        ),
        "source_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
