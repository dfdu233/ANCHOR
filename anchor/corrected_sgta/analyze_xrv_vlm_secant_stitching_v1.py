#!/usr/bin/env python3
"""CPU representation gate for XRV-to-VLM point and secant stitching.

The existing XRV cache overlaps only the *development* portion of the raw VLM
feature cache.  Consequently this script performs an explicitly exploratory,
deterministic image-disjoint split within that overlap; it must not be reported
as the frozen Evidence Addressability confirmation gate.

For each VLM feature space it compares:

* pointwise ridge: XRV(x) -> pooled VLM(x), then subtract predictions;
* secant ridge: XRV(x+) - XRV(x-) -> VLM(x+) - VLM(x-).

Positive/negative pairs are matched within finding using the non-target XRV
logits.  A leave-one-finding-out panel checks whether a learned translation is
generic rather than merely a disease lookup table.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge


FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
XRV_LABELS = (
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
)
TARGETS = {
    "aortic_enlargement": ("Enlarged Cardiomediastinum",),
    "cardiomegaly": ("Cardiomegaly",),
    "lung_opacity": ("Lung Opacity",),
    "nodule_mass": ("Nodule", "Mass", "Lung Lesion"),
    "pleural_effusion": ("Effusion",),
    "pleural_thickening": ("Pleural_Thickening",),
    "pulmonary_fibrosis": ("Fibrosis",),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_xrv(path: Path) -> dict[str, np.ndarray]:
    pack = np.load(path, allow_pickle=False)
    if tuple(str(value) for value in pack["labels"]) != XRV_LABELS:
        raise ValueError("XRV label-order drift")
    return {
        str(image_id): np.asarray(vector, dtype=np.float64)
        for image_id, vector in zip(pack["image_ids"], pack["logits"])
    }


def load_raw(directory: Path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    metadata = read_jsonl(directory / "metadata.jsonl")
    arrays = np.load(directory / "features.npz", allow_pickle=False)
    if len(metadata) != len(next(iter(arrays.values()))):
        raise ValueError("Raw feature metadata/array length mismatch")
    indices = {row["image_id"]: int(row["ordered_index"]) for row in metadata}
    if sorted(indices.values()) != list(range(len(metadata))):
        raise ValueError("ordered_index is not a permutation")
    return indices, {key: np.asarray(arrays[key], dtype=np.float64) for key in arrays.files}


def load_labels(path: Path, eligible: set[str]) -> dict[str, dict[str, int]]:
    output = {finding: {} for finding in FINDINGS}
    for row in read_jsonl(path):
        finding, image_id = row["finding"], row["image_id"]
        if finding in output and image_id in eligible and int(row["positive_votes"]) in (0, 3):
            output[finding][image_id] = int(row["positive_votes"] == 3)
    return output


def standardized_xrv(xrv: dict[str, np.ndarray], image_ids: list[str]) -> dict[str, np.ndarray]:
    matrix = np.stack([xrv[image_id] for image_id in image_ids])
    mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {image_id: (xrv[image_id] - mean) / scale for image_id in image_ids}


def matched_pairs(
    labels: dict[str, dict[str, int]],
    xrv: dict[str, np.ndarray],
    allowed: set[str],
) -> list[dict[str, str]]:
    index = {name: i for i, name in enumerate(XRV_LABELS)}
    pairs = []
    for finding in FINDINGS:
        positives = sorted(
            image_id for image_id, label in labels[finding].items() if label == 1 and image_id in allowed
        )
        negatives = sorted(
            image_id for image_id, label in labels[finding].items() if label == 0 and image_id in allowed
        )
        if not positives or not negatives:
            continue
        nuisance = [i for i in range(len(XRV_LABELS)) if XRV_LABELS[i] not in TARGETS[finding]]
        pos = np.stack([xrv[image_id][nuisance] for image_id in positives])
        neg = np.stack([xrv[image_id][nuisance] for image_id in negatives])
        distances = np.linalg.norm(pos[:, None, :] - neg[None, :, :], axis=-1)
        pos_indices, neg_indices = linear_sum_assignment(distances)
        pairs.extend(
            {
                "finding": finding,
                "positive": positives[i],
                "negative": negatives[j],
            }
            for i, j in zip(pos_indices, neg_indices)
        )
    return pairs


def pair_arrays(
    pairs: list[dict[str, str]],
    xrv: dict[str, np.ndarray],
    raw: np.ndarray,
    raw_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([xrv[pair["positive"]] - xrv[pair["negative"]] for pair in pairs])
    y = np.stack(
        [raw[raw_index[pair["positive"]]] - raw[raw_index[pair["negative"]]] for pair in pairs]
    )
    return x, y


def point_arrays(
    image_ids: list[str],
    xrv: dict[str, np.ndarray],
    raw: np.ndarray,
    raw_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([xrv[image_id] for image_id in image_ids]),
        np.stack([raw[raw_index[image_id]] for image_id in image_ids]),
    )


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.sum((actual - predicted) ** 2, axis=1)
    zero_error = np.sum(actual**2, axis=1)
    denominator = np.linalg.norm(actual, axis=1) * np.linalg.norm(predicted, axis=1)
    cosine = np.divide(
        np.sum(actual * predicted, axis=1),
        denominator,
        out=np.zeros(len(actual), dtype=np.float64),
        where=denominator > 1e-12,
    )
    return {
        "n_pairs": int(len(actual)),
        "relative_mse_vs_zero": float(error.sum() / max(zero_error.sum(), 1e-12)),
        "variance_explained_vs_zero": float(1.0 - error.sum() / max(zero_error.sum(), 1e-12)),
        "mean_cosine": float(cosine.mean()),
        "median_cosine": float(np.median(cosine)),
    }


def fit_compare(
    train_images: list[str],
    train_pairs: list[dict[str, str]],
    test_pairs: list[dict[str, str]],
    xrv: dict[str, np.ndarray],
    raw: np.ndarray,
    raw_index: dict[str, int],
) -> dict[str, Any]:
    point_x, point_y = point_arrays(train_images, xrv, raw, raw_index)
    train_dx, train_dy = pair_arrays(train_pairs, xrv, raw, raw_index)
    test_dx, test_dy = pair_arrays(test_pairs, xrv, raw, raw_index)
    point_model = Ridge(alpha=1.0).fit(point_x, point_y)
    secant_model = Ridge(alpha=1.0, fit_intercept=False).fit(train_dx, train_dy)
    # For a linear point model, subtracting two predictions exactly cancels
    # the intercept; coef_.T maps the held-out XRV secant to a VLM secant.
    point_prediction = test_dx @ point_model.coef_.T
    secant_prediction = secant_model.predict(test_dx)
    return {
        "train_images": len(train_images),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "pointwise_ridge_on_pair_deltas": score(test_dy, point_prediction),
        "secant_ridge": score(test_dy, secant_prediction),
    }


def analyze_model(
    raw_directory: Path,
    labels_path: Path,
    xrv: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    raw_index, arrays = load_raw(raw_directory)
    overlap = sorted(set(raw_index) & set(xrv))
    split_counts = {}
    for row in read_jsonl(raw_directory / "metadata.jsonl"):
        if row["image_id"] in xrv:
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    labels = load_labels(labels_path, set(overlap))
    standardized = standardized_xrv(xrv, overlap)

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(overlap)
    rng.shuffle(shuffled)
    cut = int(round(0.7 * len(shuffled)))
    train_set, test_set = set(shuffled[:cut]), set(shuffled[cut:])
    train_pairs = matched_pairs(labels, standardized, train_set)
    test_pairs = matched_pairs(labels, standardized, test_set)
    used_train_images = sorted(
        {pair[key] for pair in train_pairs for key in ("positive", "negative")}
    )

    spaces = {}
    for feature_name in ("pre_mean", "post_mean"):
        spaces[feature_name] = {
            "internal_image_disjoint_70_30": fit_compare(
                used_train_images,
                train_pairs,
                test_pairs,
                standardized,
                arrays[feature_name],
                raw_index,
            ),
            "leave_one_finding_out": {},
        }
        all_pairs = matched_pairs(labels, standardized, set(overlap))
        for held_out in FINDINGS:
            test = [pair for pair in all_pairs if pair["finding"] == held_out]
            test_images = {pair[key] for pair in test for key in ("positive", "negative")}
            train = [
                pair
                for pair in all_pairs
                if pair["finding"] != held_out
                and pair["positive"] not in test_images
                and pair["negative"] not in test_images
            ]
            images = sorted({pair[key] for pair in train for key in ("positive", "negative")})
            if len(train) < 10 or len(test) < 5:
                spaces[feature_name]["leave_one_finding_out"][held_out] = {
                    "status": "insufficient_pairs",
                    "train_pairs": len(train),
                    "test_pairs": len(test),
                }
            else:
                spaces[feature_name]["leave_one_finding_out"][held_out] = fit_compare(
                    images, train, test, standardized, arrays[feature_name], raw_index
                )
    return {
        "raw_xrv_overlap_images": len(overlap),
        "overlap_by_raw_split": split_counts,
        "frozen_confirmation_joinable": bool(split_counts.get("confirmation", 0)),
        "unanimous_claim_counts": {
            finding: {
                "negative": sum(label == 0 for label in labels[finding].values()),
                "positive": sum(label == 1 for label in labels[finding].values()),
            }
            for finding in FINDINGS
        },
        "feature_spaces": spaces,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-raw", type=Path, required=True)
    parser.add_argument("--hulu-raw", type=Path, required=True)
    parser.add_argument("--huatuo-labels", type=Path, required=True)
    parser.add_argument("--hulu-labels", type=Path, required=True)
    parser.add_argument("--xrv-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise RuntimeError("CPU-only analysis; set CUDA_VISIBLE_DEVICES=''")
    if args.output.exists():
        raise FileExistsError(args.output)
    xrv = load_xrv(args.xrv_logits)
    result = {
        "status": "complete_exploratory_representation_gate",
        "scope_warning": (
            "The cached XRV images overlap only raw-cache development images. Results use a new "
            "deterministic internal split and are not a frozen confirmation result."
        ),
        "models": {
            "huatuo": analyze_model(args.huatuo_raw, args.huatuo_labels, xrv, args.seed),
            "hulu": analyze_model(args.hulu_raw, args.hulu_labels, xrv, args.seed),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
