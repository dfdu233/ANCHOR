#!/usr/bin/env python3
"""Test whether multiscale spatial scan recovers diluted sparse visual evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


VERSION = "sparse-patch-multiscale-scan-analysis-v2"
FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
SEED = 20260812
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


def hidden_rows(directory: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (directory / "metadata.jsonl").read_text().splitlines()]
    output = []
    for row in rows:
        if row["finding"] not in FINDINGS or int(row["positive_votes"]) not in (0, 3):
            continue
        layers = sorted(int(value) for value in row["diagnostic_plain_logit_lens"])
        logits = row["diagnostic_plain_logit_lens"][str(layers[-1])]
        output.append(
            {
                "record_key": row["record_key"],
                "image_id": row["image_id"],
                "finding": row["finding"],
                "vote": int(row["positive_votes"]),
                "label": int(row["positive_votes"] == 3),
                "final_margin": float(logits["supported"] - logits["refuted"]),
            }
        )
    return output


def patch_artifact(directory: Path) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    scores = np.asarray(np.load(directory / "patch_scores.npz")["patch_scores"], dtype=np.float64)
    metadata = [json.loads(line) for line in (directory / "metadata.jsonl").read_text().splitlines()]
    if scores.shape[0] != len(metadata):
        raise ValueError("patch score/metadata length mismatch")
    index = {row["image_id"]: i for i, row in enumerate(metadata)}
    if len(index) != len(metadata):
        raise ValueError("patch artifact repeats image ids")
    geometry = {(int(row["grid_groups"]), int(row["grid_side"])) for row in metadata}
    if len(geometry) != 1:
        raise ValueError(f"patch geometry drift: {geometry}")
    groups, side = next(iter(geometry))
    return scores, index, {"groups": groups, "side": side, "metadata": metadata}


def multiscale_scan(z: np.ndarray, groups: int, side: int) -> float:
    grids = z.reshape(groups, side, side)
    sizes = sorted({1, 2, 4, max(1, side // 2)})
    candidates = []
    for size in sizes:
        if size > side:
            continue
        n_windows = groups * (side - size + 1) ** 2
        penalty = math.sqrt(2.0 * math.log(max(2, n_windows)))
        for grid in grids:
            integral = np.pad(grid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            sums = (
                integral[size:, size:]
                - integral[:-size, size:]
                - integral[size:, :-size]
                + integral[:-size, :-size]
            )
            candidates.append(float(np.max(sums / math.sqrt(size * size) - penalty)))
    return max(candidates)


def higher_criticism(z: np.ndarray) -> float:
    p = np.sort(np.clip(norm.sf(z), 1e-8, 1 - 1e-8))
    n = len(p)
    indices = np.arange(1, n + 1)
    # HC scans the lower tail only.  Restricting the order index as well as the
    # p-value prevents a tied p=0.5 null vector from attaining its spurious
    # maximum at i=n.
    mask = (indices <= max(1, int(0.5 * n))) & (p <= 0.5)
    if not mask.any():
        return 0.0
    values = np.sqrt(n) * (indices / n - p) / np.sqrt(p * (1 - p))
    return float(np.max(values[mask]))


def build_features(
    dev_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    scores: np.ndarray,
    image_index: dict[str, int],
    groups: int,
    side: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    finding_index = {name: index for index, name in enumerate(FINDINGS)}
    null = {}
    for finding in FINDINGS:
        column = finding_index[finding]
        rows = [row for row in dev_rows if row["finding"] == finding and row["vote"] == 0]
        values = np.stack([scores[image_index[row["image_id"]], :, column] for row in rows])
        mean, std = values.mean(0), values.std(0)
        floor = float(np.quantile(std[std > 0], 0.10))
        null[finding] = (mean, np.maximum(std, floor))

    def transform(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            column = finding_index[row["finding"]]
            raw = scores[image_index[row["image_id"]], :, column]
            mean, std = null[row["finding"]]
            z = (raw - mean) / std
            top_n = max(1, int(math.ceil(0.05 * len(z))))
            output.append(
                {
                    **row,
                    "patch_mean": float(raw.mean()),
                    "patch_max_z": float(z.max()),
                    "patch_top5_z": float(np.partition(z, -top_n)[-top_n:].mean()),
                    "higher_criticism": higher_criticism(z),
                    "multiscale_scan": multiscale_scan(z, groups, side),
                }
            )
        return output

    return transform(dev_rows), transform(test_rows), {
        "null": "development 0/3 images; positionwise mean/std per finding",
        "std_floor": "10th percentile of positive positionwise std per finding",
        "grid_groups": groups,
        "grid_side": side,
        "scan_window_sides": sorted({1, 2, 4, max(1, side // 2)}),
    }


def arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "finding": np.asarray([row["finding"] for row in rows]),
        "label": np.asarray([row["label"] for row in rows], dtype=int),
        **{
            name: np.asarray([row[name] for row in rows], dtype=float)
            for name in (
                "final_margin",
                "patch_mean",
                "patch_max_z",
                "patch_top5_z",
                "higher_criticism",
                "multiscale_scan",
            )
        },
    }


def macro_auc(finding: np.ndarray, label: np.ndarray, score: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(label[finding == name], score[finding == name]) for name in FINDINGS]))


def finding_auc(finding: np.ndarray, label: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {name: float(roc_auc_score(label[finding == name], score[finding == name])) for name in FINDINGS}


def design(data: dict[str, np.ndarray], include_scan: bool) -> np.ndarray:
    fixed = np.column_stack([(data["finding"] == name).astype(float) for name in FINDINGS[:-1]])
    # A spatial scan only counts as new evidence if it improves on both the
    # final answer and the standard non-spatial medical-MIL patch poolers.
    columns = [
        fixed,
        data["final_margin"][:, None],
        data["patch_mean"][:, None],
        data["patch_max_z"][:, None],
        data["patch_top5_z"][:, None],
    ]
    if include_scan:
        columns.append(data["multiscale_scan"][:, None])
    return np.concatenate(columns, axis=1)


def standardize(train: np.ndarray, test: np.ndarray, fixed_columns: int) -> tuple[np.ndarray, np.ndarray]:
    left, right = train.copy(), test.copy()
    mean = left[:, fixed_columns:].mean(0)
    std = left[:, fixed_columns:].std(0)
    std[std == 0] = 1.0
    left[:, fixed_columns:] = (left[:, fixed_columns:] - mean) / std
    right[:, fixed_columns:] = (right[:, fixed_columns:] - mean) / std
    return left, right


def fit_nested(dev: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    base_dev, base_test = standardize(design(dev, False), design(test, False), len(FINDINGS) - 1)
    enhanced_dev, enhanced_test = standardize(design(dev, True), design(test, True), len(FINDINGS) - 1)
    base_model = LogisticRegression(C=0.1, max_iter=10000, random_state=SEED).fit(base_dev, dev["label"])
    enhanced_model = LogisticRegression(C=0.1, max_iter=10000, random_state=SEED).fit(enhanced_dev, dev["label"])
    return (
        base_model.predict_proba(base_test)[:, 1],
        enhanced_model.predict_proba(enhanced_test)[:, 1],
        {
            "regularization": "LogisticRegression C=0.1 fixed before confirmation",
            "base_features": (
                "finding fixed effects + final margin + patch mean + patch max "
                "+ patch top-5%; all standard aggregators included together"
            ),
            "enhanced_features": "base + multiscale scan",
            "enhanced_scan_coefficient": float(enhanced_model.coef_[0, -1]),
        },
    )


def bootstrap(
    data: dict[str, np.ndarray], base: np.ndarray, enhanced: np.ndarray
) -> dict[str, Any]:
    cells = [np.flatnonzero((data["finding"] == name) & (data["label"] == label)) for name in FINDINGS for label in (0, 1)]
    rng = np.random.default_rng(SEED)
    auc_delta, nll_delta, brier_delta = [], [], []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = np.concatenate([rng.choice(cell, len(cell), replace=True) for cell in cells])
        f, y = data["finding"][indices], data["label"][indices]
        left, right = base[indices], enhanced[indices]
        auc_delta.append(macro_auc(f, y, right) - macro_auc(f, y, left))
        nll_delta.append(log_loss(y, left, labels=[0, 1]) - log_loss(y, right, labels=[0, 1]))
        brier_delta.append(np.mean((y - left) ** 2) - np.mean((y - right) ** 2))
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "unit": "paired resampling within finding x clear label cells",
        "macro_auroc_delta_ci95": np.quantile(auc_delta, [0.025, 0.975]).tolist(),
        "nll_improvement_ci95": np.quantile(nll_delta, [0.025, 0.975]).tolist(),
        "brier_improvement_ci95": np.quantile(brier_delta, [0.025, 0.975]).tolist(),
    }


def analyze_model(
    model: str,
    dev_dir: Path,
    test_dir: Path,
    patch_dir: Path,
) -> dict[str, Any]:
    dev_rows, test_rows = hidden_rows(dev_dir), hidden_rows(test_dir)
    patch_scores, image_index, geometry = patch_artifact(patch_dir)
    dev_rows, test_rows, feature_contract = build_features(
        dev_rows,
        test_rows,
        patch_scores,
        image_index,
        geometry["groups"],
        geometry["side"],
    )
    dev, test = arrays(dev_rows), arrays(test_rows)
    base_probability, enhanced_probability, fit_contract = fit_nested(dev, test)
    base_auc = macro_auc(test["finding"], test["label"], base_probability)
    enhanced_auc = macro_auc(test["finding"], test["label"], enhanced_probability)
    base_by = finding_auc(test["finding"], test["label"], base_probability)
    enhanced_by = finding_auc(test["finding"], test["label"], enhanced_probability)
    deltas = {name: enhanced_by[name] - base_by[name] for name in FINDINGS}
    boot = bootstrap(test, base_probability, enhanced_probability)
    positive_findings = sum(value > 0 for value in deltas.values())
    gate = bool(
        enhanced_auc - base_auc >= 0.02
        and boot["macro_auroc_delta_ci95"][0] > 0
        and boot["nll_improvement_ci95"][0] > 0
        and positive_findings >= 5
        and fit_contract["enhanced_scan_coefficient"] > 0
    )
    return {
        "model": model,
        "development_n": len(dev_rows),
        "confirmation_n": len(test_rows),
        "feature_contract": feature_contract,
        "fit_contract": fit_contract,
        "confirmation": {
            "individual_macro_auroc": {
                name: macro_auc(test["finding"], test["label"], test[name])
                for name in (
                    "final_margin",
                    "patch_mean",
                    "patch_max_z",
                    "patch_top5_z",
                    "higher_criticism",
                    "multiscale_scan",
                )
            },
            "base_macro_auroc": base_auc,
            "enhanced_macro_auroc": enhanced_auc,
            "macro_auroc_delta": enhanced_auc - base_auc,
            "base_nll": float(log_loss(test["label"], base_probability, labels=[0, 1])),
            "enhanced_nll": float(log_loss(test["label"], enhanced_probability, labels=[0, 1])),
            "base_brier": float(np.mean((test["label"] - base_probability) ** 2)),
            "enhanced_brier": float(np.mean((test["label"] - enhanced_probability) ** 2)),
            "by_finding_base_auroc": base_by,
            "by_finding_enhanced_auroc": enhanced_by,
            "by_finding_delta": deltas,
            "positive_finding_count": positive_findings,
            "bootstrap": boot,
        },
        "gate": {
            "rule": "AUROC delta >=0.02; AUROC and NLL bootstrap lower bounds >0; >=5/7 finding deltas positive; scan coefficient positive",
            "pass": gate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu"), required=True)
    parser.add_argument("--development-hidden", type=Path, required=True)
    parser.add_argument("--confirmation-hidden", type=Path, required=True)
    parser.add_argument("--patch-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {
        "version": VERSION,
        "status": "complete",
        "scope": "development-fitted sparse patch mechanism gate; fresh confirmation labels already opened for earlier endpoints",
        "configuration": {
            "findings": list(FINDINGS),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
            "command": " ".join(sys.argv),
            "source_sha256": sha256(Path(__file__)),
        },
        "inputs": {
            "development_hidden": str(args.development_hidden.resolve()),
            "development_metadata_sha256": sha256(args.development_hidden / "metadata.jsonl"),
            "confirmation_hidden": str(args.confirmation_hidden.resolve()),
            "confirmation_metadata_sha256": sha256(args.confirmation_hidden / "metadata.jsonl"),
            "patch_scores": str(args.patch_scores.resolve()),
            "patch_score_sha256": sha256(args.patch_scores / "patch_scores.npz"),
        },
        "result": analyze_model(
            args.model,
            args.development_hidden,
            args.confirmation_hidden,
            args.patch_scores,
        ),
        "boundary": "A pass establishes incremental sparse-patch decodability for a supervised development direction. It is not localization, causal control, or open-ended mitigation.",
    }
    atomic_json(args.output, result)
    print(json.dumps(result["result"]["gate"], indent=2))


if __name__ == "__main__":
    main()
