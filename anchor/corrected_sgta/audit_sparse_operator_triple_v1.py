#!/usr/bin/env python3
"""Cached fatal audit for local-CFAR and H0-persistence patch statistics.

This script intentionally does not modify or query a VLM.  It reuses the
frozen Huatuo patch-score artifact and asks whether either statistic adds
held-out information above the already strong final/mean/max/top5/scan
baseline.  The confirmation labels have been opened by earlier experiments,
so this is a secondary audit rather than a new blind confirmation.
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
    hidden_rows,
    multiscale_scan,
    patch_artifact,
)


VERSION = "sparse-operator-triple-audit-v1"
SEED = 20260813
BOOTSTRAP_DRAWS = 5000


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


def robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


def local_cfar(z: np.ndarray, groups: int, side: int) -> float:
    """Max guard-ring robust local contrast.

    The outer Chebyshev radius is 3 and the guard radius is 1.  Adding one
    tenth of the image-global robust scale regularizes small rings while
    preserving exact invariance to z -> a*z+b for every a>0 (away from the
    degenerate all-constant field).
    """

    grids = z.reshape(groups, side, side)
    global_scale = robust_scale(z)
    if global_scale == 0:
        return 0.0
    values: list[float] = []
    for grid in grids:
        for row in range(side):
            for col in range(side):
                ring = []
                for rr in range(max(0, row - 3), min(side, row + 4)):
                    for cc in range(max(0, col - 3), min(side, col + 4)):
                        distance = max(abs(rr - row), abs(cc - col))
                        if 1 < distance <= 3:
                            ring.append(float(grid[rr, cc]))
                ring_values = np.asarray(ring, dtype=float)
                background = float(np.median(ring_values))
                denominator = robust_scale(ring_values) + 0.1 * global_scale
                values.append((float(grid[row, col]) - background) / denominator)
    return float(max(values))


def h0_persistences(z: np.ndarray, groups: int, side: int) -> np.ndarray:
    """Finite H0 lifetimes of the superlevel cubical filtration."""

    all_lifetimes: list[float] = []
    for grid in z.reshape(groups, side, side):
        flat = grid.reshape(-1)
        order = np.argsort(-flat, kind="stable")
        parent = np.arange(flat.size)
        birth = flat.copy()
        active = np.zeros(flat.size, dtype=bool)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        for index_value in order:
            index = int(index_value)
            active[index] = True
            row, col = divmod(index, side)
            neighbors = []
            if row > 0:
                neighbors.append(index - side)
            if row + 1 < side:
                neighbors.append(index + side)
            if col > 0:
                neighbors.append(index - 1)
            if col + 1 < side:
                neighbors.append(index + 1)
            for neighbor in neighbors:
                if not active[neighbor]:
                    continue
                left, right = find(index), find(neighbor)
                if left == right:
                    continue
                # Elder rule: the component with the larger birth survives.
                if birth[left] >= birth[right]:
                    older, younger = left, right
                else:
                    older, younger = right, left
                lifetime = float(birth[younger] - flat[index])
                if lifetime > 0:
                    all_lifetimes.append(lifetime)
                parent[younger] = older
                parent[index] = older
    return np.asarray(all_lifetimes, dtype=float)


def persistent_prominence(z: np.ndarray, groups: int, side: int) -> tuple[float, float]:
    lifetimes = h0_persistences(z, groups, side)
    if not len(lifetimes):
        return 0.0, 0.0
    ordered = np.sort(lifetimes)
    return float(ordered[-1]), float(ordered[-min(3, len(ordered)) :].sum())


def build_rows(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    image_index: dict[str, int],
    null: dict[str, tuple[np.ndarray, np.ndarray]],
    groups: int,
    side: int,
) -> list[dict[str, Any]]:
    finding_index = {name: index for index, name in enumerate(FINDINGS)}
    output = []
    for row in rows:
        column = finding_index[row["finding"]]
        raw = scores[image_index[row["image_id"]], :, column]
        mean, std = null[row["finding"]]
        z = (raw - mean) / std
        top_n = max(1, int(np.ceil(0.05 * len(z))))
        ph_max, ph_top3 = persistent_prominence(z, groups, side)
        output.append(
            {
                **row,
                "patch_mean": float(raw.mean()),
                "patch_max_z": float(z.max()),
                "patch_top5_z": float(np.partition(z, -top_n)[-top_n:].mean()),
                "multiscale_scan": multiscale_scan(z, groups, side),
                "local_cfar": local_cfar(z, groups, side),
                "ph_max": ph_max,
                "ph_top3": ph_top3,
            }
        )
    return output


def arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    names = (
        "final_margin",
        "patch_mean",
        "patch_max_z",
        "patch_top5_z",
        "multiscale_scan",
        "local_cfar",
        "ph_max",
        "ph_top3",
    )
    return {
        "finding": np.asarray([row["finding"] for row in rows]),
        "label": np.asarray([row["label"] for row in rows], dtype=int),
        **{name: np.asarray([row[name] for row in rows], dtype=float) for name in names},
    }


def macro_auc(finding: np.ndarray, label: np.ndarray, score: np.ndarray) -> float:
    return float(
        np.mean(
            [roc_auc_score(label[finding == name], score[finding == name]) for name in FINDINGS]
        )
    )


def design(data: dict[str, np.ndarray], extra: tuple[str, ...]) -> np.ndarray:
    fixed = np.column_stack(
        [(data["finding"] == name).astype(float) for name in FINDINGS[:-1]]
    )
    base_names = (
        "final_margin",
        "patch_mean",
        "patch_max_z",
        "patch_top5_z",
        "multiscale_scan",
    )
    return np.column_stack([fixed, *[data[name] for name in base_names + extra]])


def fit_predict(
    dev: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    extra: tuple[str, ...],
) -> tuple[np.ndarray, list[float]]:
    left, right = design(dev, extra), design(test, extra)
    fixed_columns = len(FINDINGS) - 1
    mean = left[:, fixed_columns:].mean(0)
    std = left[:, fixed_columns:].std(0)
    std[std == 0] = 1.0
    left[:, fixed_columns:] = (left[:, fixed_columns:] - mean) / std
    right[:, fixed_columns:] = (right[:, fixed_columns:] - mean) / std
    model = LogisticRegression(C=0.1, max_iter=10000, random_state=SEED).fit(
        left, dev["label"]
    )
    return model.predict_proba(right)[:, 1], model.coef_[0, -len(extra) :].tolist()


def summarize(
    test: dict[str, np.ndarray], base: np.ndarray, enhanced: np.ndarray
) -> dict[str, Any]:
    base_auc = macro_auc(test["finding"], test["label"], base)
    enhanced_auc = macro_auc(test["finding"], test["label"], enhanced)
    by_delta = {}
    for name in FINDINGS:
        mask = test["finding"] == name
        by_delta[name] = float(
            roc_auc_score(test["label"][mask], enhanced[mask])
            - roc_auc_score(test["label"][mask], base[mask])
        )
    cells = [
        np.flatnonzero((test["finding"] == name) & (test["label"] == label))
        for name in FINDINGS
        for label in (0, 1)
    ]
    rng = np.random.default_rng(SEED)
    auc_delta, nll_delta, brier_delta = [], [], []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = np.concatenate([rng.choice(cell, len(cell), replace=True) for cell in cells])
        finding, label = test["finding"][indices], test["label"][indices]
        lhs, rhs = base[indices], enhanced[indices]
        auc_delta.append(macro_auc(finding, label, rhs) - macro_auc(finding, label, lhs))
        nll_delta.append(log_loss(label, lhs, labels=[0, 1]) - log_loss(label, rhs, labels=[0, 1]))
        brier_delta.append(np.mean((label - lhs) ** 2) - np.mean((label - rhs) ** 2))
    return {
        "base_macro_auroc": base_auc,
        "enhanced_macro_auroc": enhanced_auc,
        "macro_auroc_delta": enhanced_auc - base_auc,
        "macro_auroc_delta_ci95": np.quantile(auc_delta, [0.025, 0.975]).tolist(),
        "nll_improvement": float(
            log_loss(test["label"], base, labels=[0, 1])
            - log_loss(test["label"], enhanced, labels=[0, 1])
        ),
        "nll_improvement_ci95": np.quantile(nll_delta, [0.025, 0.975]).tolist(),
        "brier_improvement": float(
            np.mean((test["label"] - base) ** 2)
            - np.mean((test["label"] - enhanced) ** 2)
        ),
        "brier_improvement_ci95": np.quantile(brier_delta, [0.025, 0.975]).tolist(),
        "by_finding_auroc_delta": by_delta,
        "positive_finding_count": sum(value > 0 for value in by_delta.values()),
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
    finding_index = {name: index for index, name in enumerate(FINDINGS)}
    null = {}
    for finding in FINDINGS:
        column = finding_index[finding]
        subset = [row for row in dev_rows if row["finding"] == finding and row["vote"] == 0]
        values = np.stack([scores[image_index[row["image_id"]], :, column] for row in subset])
        mean, std = values.mean(0), values.std(0)
        floor = float(np.quantile(std[std > 0], 0.10))
        null[finding] = mean, np.maximum(std, floor)

    dev = arrays(
        build_rows(
            dev_rows, scores, image_index, null, geometry["groups"], geometry["side"]
        )
    )
    test = arrays(
        build_rows(
            test_rows, scores, image_index, null, geometry["groups"], geometry["side"]
        )
    )
    base, _ = fit_predict(dev, test, ())
    result: dict[str, Any] = {}
    for name, extra in {
        "local_cfar": ("local_cfar",),
        "h0_persistence": ("ph_max", "ph_top3"),
    }.items():
        enhanced, coefficients = fit_predict(dev, test, extra)
        result[name] = {
            **summarize(test, base, enhanced),
            "extra_features": list(extra),
            "extra_coefficients": coefficients,
        }
        result[name]["gate_pass"] = bool(
            result[name]["macro_auroc_delta"] >= 0.02
            and result[name]["macro_auroc_delta_ci95"][0] > 0
            and result[name]["nll_improvement_ci95"][0] > 0
            and result[name]["positive_finding_count"] >= 5
        )

    artifact = {
        "version": VERSION,
        "status": "complete",
        "scope": "zero-GPU cached secondary fatal audit; not blind confirmation",
        "configuration": {
            "seed": SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "findings": list(FINDINGS),
            "base": "finding fixed effects + final margin + patch mean/max/top5 + multiscale scan",
            "gate": "delta AUROC >= .02, AUROC/NLL lower CI > 0, >=5/7 findings positive",
            "local_cfar": "guard radius 1, outer Chebyshev radius 3, median/MAD ring",
            "h0_persistence": "finite H0 superlevel lifetimes, max and top-3 sum",
        },
        "inputs": {
            "development_hidden": str(args.development_hidden.resolve()),
            "development_metadata_sha256": sha256(args.development_hidden / "metadata.jsonl"),
            "confirmation_hidden": str(args.confirmation_hidden.resolve()),
            "confirmation_metadata_sha256": sha256(args.confirmation_hidden / "metadata.jsonl"),
            "patch_scores": str(args.patch_scores.resolve()),
            "patch_scores_sha256": sha256(args.patch_scores / "patch_scores.npz"),
        },
        "result": result,
        "boundary": (
            "A pass would establish only incremental held-out information in a cached scalar "
            "patch field. It would not establish causal lesion localization or a mitigation method."
        ),
    }
    atomic_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
