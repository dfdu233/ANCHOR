#!/usr/bin/env python3
"""Audit whether VinDr lesion extent survives simple visibility confounds.

The primary quantity is the released-box *extent*, not lesion conspicuity.  The
audit asks whether log box-union area still adds information after location,
local intensity/texture proxies, box multiplicity, and reader agreement.  All
models are fitted on the old development cohort and evaluated once on the
held-out 133-claim confirmation cohort.
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
import pydicom
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from corrected_sgta.analyze_sparse_lesion_boundary_v1 import (
    intersection,
    load_rows,
    normalize_term,
    union_area,
)


VERSION = "lesion-area-confound-audit-v1"
SEED = 20260812
BOOTSTRAP_DRAWS = 5000
PANEL = {"R8", "R9", "R10"}

NAME_TO_FINDING = {
    "Aortic enlargement": "aortic_enlargement",
    "Cardiomegaly": "cardiomegaly",
    "Lung Opacity": "lung_opacity",
    "Nodule/Mass": "nodule_mass",
    "Pleural effusion": "pleural_effusion",
    "Pleural thickening": "pleural_thickening",
    "Pulmonary fibrosis": "pulmonary_fibrosis",
}

CONFOUNDS = [
    "radial_position",
    "centroid_x",
    "centroid_y",
    "local_abs_contrast",
    "local_texture_ratio",
    "log_box_count",
    "fragmentation",
    "reader_mask_iou",
    "reader_centroid_dispersion",
]


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


def raster_mask(boxes: list[dict[str, float]], height: int, width: int, side: int = 128) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    for box in boxes:
        x0 = int(np.floor(box["x_min"] / width * side))
        x1 = int(np.ceil(box["x_max"] / width * side))
        y0 = int(np.floor(box["y_min"] / height * side))
        y1 = int(np.ceil(box["y_max"] / height * side))
        mask[max(y0, 0) : min(y1, side), max(x0, 0) : min(x1, side)] = True
    return mask


def load_annotations(path: Path, keys: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, list[dict[str, float]]]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            finding = NAME_TO_FINDING.get(row["class_name"], normalize_term(row["class_name"]))
            key = (row["image_id"], finding)
            if key not in keys or row["rad_id"] not in PANEL or not row["x_min"]:
                continue
            grouped[key][row["rad_id"]].append(
                {name: float(row[name]) for name in ("x_min", "y_min", "x_max", "y_max")}
            )
    return grouped


def geometry_features(
    by_reader: dict[str, list[dict[str, float]]],
    pixels: np.ndarray,
    robust_scale: float,
) -> dict[str, float]:
    height, width = pixels.shape
    all_boxes = [box for boxes in by_reader.values() for box in boxes]
    extent = union_area(all_boxes) / float(height * width)
    reader_values: list[dict[str, float]] = []
    masks: list[np.ndarray] = []
    centroids: list[tuple[float, float]] = []
    contrasts: list[float] = []
    textures: list[float] = []
    for boxes in by_reader.values():
        areas = np.asarray(
            [(box["x_max"] - box["x_min"]) * (box["y_max"] - box["y_min"]) for box in boxes]
        )
        total = max(float(areas.sum()), 1e-9)
        weights = areas / total
        cx = np.asarray([(box["x_min"] + box["x_max"]) / (2 * width) for box in boxes])
        cy = np.asarray([(box["y_min"] + box["y_max"]) / (2 * height) for box in boxes])
        centroid_x, centroid_y = float(np.sum(cx * weights)), float(np.sum(cy * weights))
        centroids.append((centroid_x, centroid_y))
        radial = np.sqrt(((cx - 0.5) * 2) ** 2 + ((cy - 0.5) * 2) ** 2)
        reader_values.append(
            {
                "radial_position": float(np.sum(radial * weights)),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "box_count": float(len(boxes)),
                "fragmentation": 1.0 - float(areas.max()) / total,
            }
        )
        masks.append(raster_mask(boxes, height, width))

        inner = np.zeros((height, width), dtype=bool)
        outer = np.zeros((height, width), dtype=bool)
        for box in boxes:
            x0, x1 = max(0, int(box["x_min"])), min(width, int(math.ceil(box["x_max"])))
            y0, y1 = max(0, int(box["y_min"])), min(height, int(math.ceil(box["y_max"])))
            if x1 <= x0 or y1 <= y0:
                continue
            inner[y0:y1, x0:x1] = True
            pad_x = max(8, int(0.25 * (x1 - x0)))
            pad_y = max(8, int(0.25 * (y1 - y0)))
            outer[max(0, y0 - pad_y) : min(height, y1 + pad_y), max(0, x0 - pad_x) : min(width, x1 + pad_x)] = True
        ring = outer & ~inner
        if inner.any() and ring.any():
            contrasts.append(abs(float(pixels[inner].mean() - pixels[ring].mean())) / robust_scale)
            textures.append(float(pixels[inner].std()) / max(float(pixels[ring].std()), 1e-6))

    pair_iou = []
    for i, left in enumerate(masks):
        for right in masks[i + 1 :]:
            union = np.logical_or(left, right).sum()
            pair_iou.append(float(np.logical_and(left, right).sum() / max(union, 1)))
    c = np.asarray(centroids)
    dispersion = float(np.sqrt(((c - c.mean(axis=0)) ** 2).sum(axis=1)).mean())
    result = {
        name: float(np.mean([values[name] for values in reader_values]))
        for name in ("radial_position", "centroid_x", "centroid_y", "box_count", "fragmentation")
    }
    result.update(
        {
            "log_union_area_fraction": float(np.log10(max(extent, 1e-8))),
            "union_area_fraction": float(extent),
            "log_box_count": float(np.log1p(result["box_count"])),
            "reader_mask_iou": float(np.mean(pair_iou)) if pair_iou else 0.0,
            "reader_centroid_dispersion": dispersion,
            "local_abs_contrast": float(np.mean(contrasts)) if contrasts else np.nan,
            "local_texture_ratio": float(np.mean(textures)) if textures else np.nan,
            "reader_count": len(by_reader),
            "annotation_rows": len(all_boxes),
        }
    )
    return result


def extract_features(
    annotations: dict[tuple[str, str], dict[str, list[dict[str, float]]]],
    dicom_root: Path,
) -> dict[tuple[str, str], dict[str, float]]:
    by_image: dict[str, list[tuple[str, dict[str, list[dict[str, float]]]]]] = defaultdict(list)
    for (image_id, finding), readers in annotations.items():
        by_image[image_id].append((finding, readers))
    output: dict[tuple[str, str], dict[str, float]] = {}
    for image_id in sorted(by_image):
        ds = pydicom.dcmread(dicom_root / f"{image_id}.dicom")
        pixels = ds.pixel_array.astype(np.float32)
        pixels = pixels * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
        sample = pixels[:: max(pixels.shape[0] // 256, 1), :: max(pixels.shape[1] // 256, 1)]
        q25, q75 = np.percentile(sample, [25, 75])
        robust_scale = max(float((q75 - q25) / 1.349), 1e-6)
        for finding, readers in by_image[image_id]:
            output[(image_id, finding)] = geometry_features(readers, pixels, robust_scale)
    return output


def join_rows(path: Path, features: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, Any]]:
    result = []
    for row in load_rows(path):
        values = features.get((row["image_id"], row["finding"]))
        if values is not None:
            result.append({**row, **values, "miss": int(row["margin"] <= 0)})
    return result


def rank_within_finding(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    result = np.empty(len(rows), dtype=float)
    finding = np.asarray([row["finding"] for row in rows])
    values = np.asarray([row[field] for row in rows], dtype=float)
    for name in np.unique(finding):
        index = np.flatnonzero(finding == name)
        result[index] = rankdata(values[index], method="average") / len(index)
    return result


def prepare_design(
    rows: list[dict[str, Any]], findings: list[str], feature_names: list[str],
    medians: dict[str, float] | None = None, scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, dict[str, float], StandardScaler]:
    one_hot = np.column_stack(
        [np.asarray([row["finding"] == name for row in rows], dtype=float) for name in findings[:-1]]
    )
    if not feature_names:
        # Only used for the unadjusted rank diagnostic.  Keep the return shape
        # compatible without asking StandardScaler to fit a zero-column array.
        return one_hot, {}, scaler  # type: ignore[return-value]
    if medians is None:
        medians = {
            name: float(np.nanmedian([row[name] for row in rows])) for name in feature_names
        }
    numeric = np.asarray(
        [[row[name] if np.isfinite(row[name]) else medians[name] for name in feature_names] for row in rows],
        dtype=float,
    )
    if scaler is None:
        scaler = StandardScaler().fit(numeric)
    numeric = scaler.transform(numeric)
    return np.column_stack([one_hot, numeric]), medians, scaler


def grouped_cv_indices(rows: list[dict[str, Any]], y: np.ndarray, classification: bool):
    groups = np.asarray([row["image_id"] for row in rows])
    if classification:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        return list(splitter.split(np.zeros(len(rows)), y, groups))
    return list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups))


def choose_logistic_c(x: np.ndarray, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
    choices = [0.01, 0.1, 1.0, 10.0]
    losses = []
    for value in choices:
        fold = []
        for train, valid in splits:
            model = LogisticRegression(C=value, max_iter=10000, random_state=SEED).fit(x[train], y[train])
            fold.append(log_loss(y[valid], model.predict_proba(x[valid])[:, 1]))
        losses.append(float(np.mean(fold)))
    return choices[int(np.argmin(losses))]


def choose_ridge_alpha(x: np.ndarray, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
    choices = [0.01, 0.1, 1.0, 10.0, 100.0]
    losses = []
    for value in choices:
        losses.append(float(np.mean([
            mean_squared_error(y[valid], Ridge(alpha=value).fit(x[train], y[train]).predict(x[valid]))
            for train, valid in splits
        ])))
    return choices[int(np.argmin(losses))]


def cluster_bootstrap_metrics(
    rows: list[dict[str, Any]], y_binary: np.ndarray, y_margin: np.ndarray,
    p_base: np.ndarray, p_full: np.ndarray, m_base: np.ndarray, m_full: np.ndarray,
) -> dict[str, list[float]]:
    image_ids = sorted({row["image_id"] for row in rows})
    cells = {image: np.flatnonzero(np.asarray([row["image_id"] for row in rows]) == image) for image in image_ids}
    rng = np.random.default_rng(SEED)
    auc, nll, mse = [], [], []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(image_ids, len(image_ids), replace=True)
        index = np.concatenate([cells[image] for image in sampled])
        if len(np.unique(y_binary[index])) == 2:
            auc.append(roc_auc_score(y_binary[index], p_full[index]) - roc_auc_score(y_binary[index], p_base[index]))
        nll.append(log_loss(y_binary[index], p_base[index]) - log_loss(y_binary[index], p_full[index]))
        mse.append(mean_squared_error(y_margin[index], m_base[index]) - mean_squared_error(y_margin[index], m_full[index]))
    return {
        "auroc_delta_ci95": np.quantile(auc, [0.025, 0.975]).tolist(),
        "nll_improvement_ci95": np.quantile(nll, [0.025, 0.975]).tolist(),
        "margin_mse_improvement_ci95": np.quantile(mse, [0.025, 0.975]).tolist(),
    }


def partial_rank(rows: list[dict[str, Any]], confounds: list[str]) -> float:
    findings = sorted({row["finding"] for row in rows})
    design, _, _ = prepare_design(rows, findings, confounds)
    design = np.column_stack([np.ones(len(rows)), design])
    area_rank = rank_within_finding(rows, "log_union_area_fraction")
    margin_rank = rank_within_finding(rows, "margin")
    area_res = area_rank - Ridge(alpha=1.0, fit_intercept=False).fit(design, area_rank).predict(design)
    margin_res = margin_rank - Ridge(alpha=1.0, fit_intercept=False).fit(design, margin_rank).predict(design)
    return float(np.corrcoef(area_res, margin_res)[0, 1])


def partial_rank_bootstrap(rows: list[dict[str, Any]], confounds: list[str]) -> list[float]:
    image_ids = sorted({row["image_id"] for row in rows})
    cells = {image: [row for row in rows if row["image_id"] == image] for image in image_ids}
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = []
        for occurrence, image in enumerate(rng.choice(image_ids, len(image_ids), replace=True)):
            sample.extend([{**row, "image_id": f"{image}:{occurrence}"} for row in cells[image]])
        values.append(partial_rank(sample, confounds))
    return np.quantile(values, [0.025, 0.975]).tolist()


def audit_model(model_name: str, dev: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    findings = sorted({row["finding"] for row in dev} & {row["finding"] for row in test})
    dev = [row for row in dev if row["finding"] in findings]
    test = [row for row in test if row["finding"] in findings]
    base_features = list(CONFOUNDS)
    full_features = base_features + ["log_union_area_fraction"]
    x0_dev, med0, scale0 = prepare_design(dev, findings, base_features)
    x0_test, _, _ = prepare_design(test, findings, base_features, med0, scale0)
    x1_dev, med1, scale1 = prepare_design(dev, findings, full_features)
    x1_test, _, _ = prepare_design(test, findings, full_features, med1, scale1)
    y_dev = np.asarray([row["miss"] for row in dev])
    y_test = np.asarray([row["miss"] for row in test])
    margin_dev = np.asarray([row["margin"] for row in dev])
    margin_test = np.asarray([row["margin"] for row in test])

    cls_splits = grouped_cv_indices(dev, y_dev, True)
    c0, c1 = choose_logistic_c(x0_dev, y_dev, cls_splits), choose_logistic_c(x1_dev, y_dev, cls_splits)
    log0 = LogisticRegression(C=c0, max_iter=10000, random_state=SEED).fit(x0_dev, y_dev)
    log1 = LogisticRegression(C=c1, max_iter=10000, random_state=SEED).fit(x1_dev, y_dev)
    p0, p1 = log0.predict_proba(x0_test)[:, 1], log1.predict_proba(x1_test)[:, 1]

    reg_splits = grouped_cv_indices(dev, margin_dev, False)
    a0, a1 = choose_ridge_alpha(x0_dev, margin_dev, reg_splits), choose_ridge_alpha(x1_dev, margin_dev, reg_splits)
    ridge0, ridge1 = Ridge(alpha=a0).fit(x0_dev, margin_dev), Ridge(alpha=a1).fit(x1_dev, margin_dev)
    m0, m1 = ridge0.predict(x0_test), ridge1.predict(x1_test)

    auc0, auc1 = roc_auc_score(y_test, p0), roc_auc_score(y_test, p1)
    nll0, nll1 = log_loss(y_test, p0), log_loss(y_test, p1)
    mse0, mse1 = mean_squared_error(margin_test, m0), mean_squared_error(margin_test, m1)
    cis = cluster_bootstrap_metrics(test, y_test, margin_test, p0, p1, m0, m1)
    unadjusted = partial_rank(test, [])
    adjusted = partial_rank(test, CONFOUNDS)
    adjusted_ci = partial_rank_bootstrap(test, CONFOUNDS)
    survives = bool(
        adjusted > 0 and adjusted_ci[0] > 0
        and (cis["nll_improvement_ci95"][0] > 0 or cis["margin_mse_improvement_ci95"][0] > 0)
        and log1.coef_[0, -1] < 0
        and ridge1.coef_[-1] > 0
    )
    return {
        "model": model_name,
        "development_n": len(dev),
        "confirmation_n": len(test),
        "development_unique_images": len({row["image_id"] for row in dev}),
        "confirmation_unique_images": len({row["image_id"] for row in test}),
        "development_confirmation_image_overlap": len(
            {row["image_id"] for row in dev} & {row["image_id"] for row in test}
        ),
        "confirmation_miss_rate": float(y_test.mean()),
        "rank_association": {
            "unadjusted_within_finding_rank_correlation": unadjusted,
            "confound_adjusted_partial_rank_correlation": adjusted,
            "confound_adjusted_image_bootstrap_ci95": adjusted_ci,
        },
        "nested_development_to_confirmation": {
            "miss_auroc_confound_only": float(auc0),
            "miss_auroc_plus_area": float(auc1),
            "miss_auroc_delta": float(auc1 - auc0),
            "miss_nll_confound_only": float(nll0),
            "miss_nll_plus_area": float(nll1),
            "miss_nll_improvement": float(nll0 - nll1),
            "margin_mse_confound_only": float(mse0),
            "margin_mse_plus_area": float(mse1),
            "margin_mse_improvement": float(mse0 - mse1),
            **cis,
            "selected_logistic_c": {"confound_only": c0, "plus_area": c1},
            "selected_ridge_alpha": {"confound_only": a0, "plus_area": a1},
            "standardized_area_coefficient": {
                "miss_logistic": float(log1.coef_[0, -1]),
                "continuous_margin_ridge": float(ridge1.coef_[-1]),
            },
        },
        "gate": {
            "rule": "fresh adjusted rank CI lower>0; NLL or margin-MSE improvement CI lower>0; area coefficients have expected signs",
            "area_survives_measured_confounds": survives,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-development", type=Path, required=True)
    parser.add_argument("--huatuo-confirmation", type=Path, required=True)
    parser.add_argument("--hulu-development", type=Path, required=True)
    parser.add_argument("--hulu-confirmation", type=Path, required=True)
    parser.add_argument("--bbox-csv", type=Path, required=True)
    parser.add_argument("--dicom-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hidden = [args.huatuo_development, args.huatuo_confirmation, args.hulu_development, args.hulu_confirmation]
    rows = [row for directory in hidden for row in load_rows(directory)]
    keys = {(row["image_id"], row["finding"]) for row in rows}
    annotations = load_annotations(args.bbox_csv, keys)
    features = extract_features(annotations, args.dicom_root)
    models = {
        "huatuo": audit_model(
            "huatuo", join_rows(args.huatuo_development, features), join_rows(args.huatuo_confirmation, features)
        ),
        "hulu": audit_model(
            "hulu", join_rows(args.hulu_development, features), join_rows(args.hulu_confirmation, features)
        ),
    }
    result = {
        "version": VERSION,
        "status": "complete",
        "scope": "3/3 reader-positive VinDr claims with panel R8/R9/R10 boxes; development-fitted, confirmation-tested",
        "dataset": "VinDr-CXR 1.0.0 train independent-reader panel",
        "models": models,
        "method": "finding-fixed regularized multivariable confound audit",
        "joint_conclusion": {
            "both_models_survive": all(value["gate"]["area_survives_measured_confounds"] for value in models.values()),
            "language": "Box extent remains an association after measured proxies; it is not a conspicuity ground truth."
            if all(value["gate"]["area_survives_measured_confounds"] for value in models.values())
            else "The box-area association is not robust across both models after measured confounds and must be weakened or withdrawn.",
        },
        "configuration": {
            "seed": SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "confounds": CONFOUNDS,
            "command": " ".join(sys.argv),
            "source_sha256": sha256(Path(__file__)),
            "inputs": {
                "bbox_csv": {"path": str(args.bbox_csv.resolve()), "sha256": sha256(args.bbox_csv)},
                **{
                    name: {"path": str(path.resolve()), "metadata_sha256": sha256(path / "metadata.jsonl")}
                    for name, path in zip(
                        ["huatuo_development", "huatuo_confirmation", "hulu_development", "hulu_confirmation"], hidden
                    )
                },
                "dicom_root": str(args.dicom_root.resolve()),
            },
        },
        "boundary": "Area is released annotation extent, not lesion visibility. Local contrast/texture are label-free pixel proxies, not clinical conspicuity truth; residual confounding remains possible.",
    }
    atomic_json(args.output, result)
    print(json.dumps(result["joint_conclusion"], indent=2))


if __name__ == "__main__":
    main()
