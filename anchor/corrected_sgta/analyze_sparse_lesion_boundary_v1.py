#!/usr/bin/env python3
"""Test whether lesion sparsity predicts VLM misses on reader-unanimous VinDr cases.

The endpoint is deliberately natural and pre-method: among 3/3 reader-positive
claims, does a smaller radiologist-box union predict a lower supported-minus-
refuted final margin?  Finding fixed effects and an area-null randomization stop
global disease identity from being mistaken for a case-level sparse boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from corrected_sgta.clinical_claims import normalize_term


VERSION = "sparse-lesion-boundary-audit-v1"
SEED = 20260812
BOOTSTRAP_DRAWS = 5000
PERMUTATIONS = 5000


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


def area(box: dict[str, float]) -> float:
    return max(0.0, box["x_max"] - box["x_min"]) * max(
        0.0, box["y_max"] - box["y_min"]
    )


def intersection(left: dict[str, float], right: dict[str, float]) -> float:
    return max(0.0, min(left["x_max"], right["x_max"]) - max(left["x_min"], right["x_min"])) * max(
        0.0, min(left["y_max"], right["y_max"]) - max(left["y_min"], right["y_min"]))


def union_area(boxes: list[dict[str, float]]) -> float:
    """Exact rectangle-union area by x-slab sweep."""

    xs = sorted({box["x_min"] for box in boxes} | {box["x_max"] for box in boxes})
    total = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (box["y_min"], box["y_max"])
            for box in boxes
            if box["x_min"] < right and box["x_max"] > left
        )
        if not intervals:
            continue
        length = 0.0
        start, end = intervals[0]
        for new_start, new_end in intervals[1:]:
            if new_start > end:
                length += end - start
                start, end = new_start, new_end
            else:
                end = max(end, new_end)
        length += end - start
        total += (right - left) * length
    return total


def load_dimensions(root: Path, image_ids: set[str]) -> dict[str, tuple[int, int]]:
    import pydicom

    result = {}
    for image_id in sorted(image_ids):
        path = root / f"{image_id}.dicom"
        header = pydicom.dcmread(
            str(path), stop_before_pixels=True, specific_tags=["Rows", "Columns"]
        )
        result[image_id] = (int(header.Rows), int(header.Columns))
    return result


def load_box_summary(
    path: Path, dimensions: dict[str, tuple[int, int]], findings: set[str]
) -> dict[tuple[str, str], dict[str, float]]:
    panel = {"R8", "R9", "R10"}
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image_id = str(row["image_id"])
            finding = normalize_term(row["class_name"])
            if (
                image_id not in dimensions
                or finding not in findings
                or str(row["rad_id"]) not in panel
                or not row["x_min"]
            ):
                continue
            grouped[(image_id, finding)].append(
                {key: float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max")}
            )
    result = {}
    for key, boxes in grouped.items():
        height, width = dimensions[key[0]]
        normalized = union_area(boxes) / float(height * width)
        readers = len(boxes)
        pairwise_iou = []
        for i, left in enumerate(boxes):
            for right in boxes[i + 1 :]:
                value = intersection(left, right)
                pairwise_iou.append(value / (area(left) + area(right) - value))
        result[key] = {
            "union_area_fraction": normalized,
            "log_union_area_fraction": float(np.log10(max(normalized, 1e-8))),
            "bbox_rows": readers,
            "mean_pairwise_iou": float(np.mean(pairwise_iou)) if pairwise_iou else 0.0,
        }
    return result


def load_rows(directory: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (directory / "metadata.jsonl").read_text().splitlines()]
    output = []
    for row in rows:
        if int(row["positive_votes"]) != 3:
            continue
        layers = sorted(int(value) for value in row["diagnostic_plain_logit_lens"])
        logits = row["diagnostic_plain_logit_lens"][str(layers[-1])]
        output.append(
            {
                "image_id": row["image_id"],
                "finding": row["finding"],
                "margin": float(logits["supported"] - logits["refuted"]),
            }
        )
    return output


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        mask = values == value
        ranks[mask] = ranks[mask].mean()
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def partial_spearman(area_value: np.ndarray, margin: np.ndarray, finding: np.ndarray) -> float:
    names = sorted(set(finding.tolist()))
    design = np.column_stack([np.ones(len(finding))] + [(finding == name).astype(float) for name in names[1:]])
    area_rank, margin_rank = rank(area_value), rank(margin)
    area_residual = area_rank - Ridge(alpha=1e-10, fit_intercept=False).fit(design, area_rank).predict(design)
    margin_residual = margin_rank - Ridge(alpha=1e-10, fit_intercept=False).fit(design, margin_rank).predict(design)
    return correlation(area_residual, margin_residual)


def within_finding_spearman(
    area_value: np.ndarray, margin: np.ndarray, finding: np.ndarray
) -> tuple[float, dict[str, float]]:
    values = {
        name: correlation(rank(area_value[finding == name]), rank(margin[finding == name]))
        for name in sorted(set(finding.tolist()))
    }
    return float(np.mean(list(values.values()))), values


def analyze(
    model: str,
    hidden: Path,
    box_summary: dict[tuple[str, str], dict[str, float]],
) -> dict[str, Any]:
    rows = []
    for row in load_rows(hidden):
        summary = box_summary.get((row["image_id"], row["finding"]))
        if summary is not None:
            rows.append({**row, **summary})
    area_values = np.asarray([row["log_union_area_fraction"] for row in rows])
    margins = np.asarray([row["margin"] for row in rows])
    finding = np.asarray([row["finding"] for row in rows])
    pooled_partial = partial_spearman(area_values, margins, finding)
    observed, finding_correlations = within_finding_spearman(
        area_values, margins, finding
    )
    rng = np.random.default_rng(SEED)
    permuted = []
    for _ in range(PERMUTATIONS):
        shuffled = area_values.copy()
        for name in sorted(set(finding.tolist())):
            indices = np.flatnonzero(finding == name)
            shuffled[indices] = shuffled[rng.permutation(indices)]
        permuted.append(within_finding_spearman(shuffled, margins, finding)[0])
    permutation_p = float((1 + np.sum(np.asarray(permuted) >= observed)) / (PERMUTATIONS + 1))
    bootstrap = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(finding == name),
                    int(np.sum(finding == name)),
                    replace=True,
                )
                for name in sorted(set(finding.tolist()))
            ]
        )
        bootstrap.append(
            within_finding_spearman(
                area_values[indices], margins[indices], finding[indices]
            )[0]
        )
    ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    by_finding = {}
    for name in sorted(set(finding.tolist())):
        mask = finding == name
        by_finding[name] = {
            "n": int(mask.sum()),
            "margin_nonpositive_rate": float(np.mean(margins[mask] <= 0)),
            "spearman_area_margin": finding_correlations[name],
            "median_union_area_fraction": float(np.median(10 ** area_values[mask])),
        }
    positive_findings = int(sum(value > 0 for value in finding_correlations.values()))
    required_positive = int(math.ceil(2 * len(finding_correlations) / 3))
    passes = bool(
        observed >= 0.20
        and ci[0] > 0
        and permutation_p <= 0.05
        and positive_findings >= required_positive
    )
    return {
        "model": model,
        "n": len(rows),
        "unique_images": len(set(row["image_id"] for row in rows)),
        "miss_rate_margin_nonpositive": float(np.mean(margins <= 0)),
        "macro_within_finding_spearman_log_area_margin": observed,
        "pooled_partial_spearman_supplementary": pooled_partial,
        "stratified_image_bootstrap_ci95": ci,
        "within_finding_permutation_p_one_sided": permutation_p,
        "positive_finding_count": positive_findings,
        "by_finding": by_finding,
        "gate": {
            "rule": "macro within-finding Spearman >=0.20, stratified bootstrap lower CI >0, permutation p<=0.05, >=ceil(2K/3) findings positive",
            "required_positive_findings": required_positive,
            "pass": passes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-hidden", type=Path, required=True)
    parser.add_argument("--hulu-hidden", type=Path, required=True)
    parser.add_argument("--bbox-csv", type=Path, required=True)
    parser.add_argument("--dicom-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    huatuo_rows, hulu_rows = load_rows(args.huatuo_hidden), load_rows(args.hulu_hidden)
    if {(row["image_id"], row["finding"]) for row in huatuo_rows} != {
        (row["image_id"], row["finding"]) for row in hulu_rows
    }:
        raise ValueError("model artifacts do not share the same 3/3 claim set")
    image_ids = {row["image_id"] for row in huatuo_rows}
    findings = {row["finding"] for row in huatuo_rows}
    dimensions = load_dimensions(args.dicom_root, image_ids)
    boxes = load_box_summary(args.bbox_csv, dimensions, findings)
    models = {
        "huatuo": analyze("huatuo", args.huatuo_hidden, boxes),
        "hulu": analyze("hulu", args.hulu_hidden, boxes),
    }
    result = {
        "version": VERSION,
        "status": "complete",
        "scope": "post-hoc natural-phenomenon audit; 3/3 reader-positive claims with released boxes",
        "inputs": {
            "huatuo_hidden": {"path": str(args.huatuo_hidden.resolve()), "metadata_sha256": sha256(args.huatuo_hidden / "metadata.jsonl")},
            "hulu_hidden": {"path": str(args.hulu_hidden.resolve()), "metadata_sha256": sha256(args.hulu_hidden / "metadata.jsonl")},
            "bbox_csv": {"path": str(args.bbox_csv.resolve()), "sha256": sha256(args.bbox_csv)},
            "dicom_root": str(args.dicom_root.resolve()),
        },
        "configuration": {
            "seed": SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "permutations": PERMUTATIONS,
            "command": " ".join(sys.argv),
            "source_sha256": sha256(Path(__file__)),
        },
        "models": models,
        "joint_gate": {
            "both_models_pass": all(value["gate"]["pass"] for value in models.values()),
            "decision": "collect_patch_statistics" if all(value["gate"]["pass"] for value in models.values()) else "do_not_promote_area_sparsity_to_patch_method",
        },
        "boundary": "Released reader boxes are an imperfect extent proxy; missing boxes exclude a claim. The audit does not test patch aggregation itself.",
    }
    atomic_json(args.output, result)
    print(json.dumps(result["joint_gate"], indent=2))


if __name__ == "__main__":
    main()
