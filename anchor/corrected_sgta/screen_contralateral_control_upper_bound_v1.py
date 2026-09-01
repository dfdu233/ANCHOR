#!/usr/bin/env python3
"""CPU fatal gate for patient-specific contralateral controls.

This is deliberately an oracle upper bound.  VinDr boxes identify a unilateral
finding only for case selection; a deployable method would have to replace the
box by a frozen, label-blind anatomy/localization model.  The gate asks a more
basic question first: does replacing the target with its contralateral partner
remove pathology-specific evidence rather than merely perturbing the image?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from anchor.corrected_sgta.screen_xrv_visual_increment_v1 import (
    FINDING_TARGETS,
    XRV_LABELS,
    load_xrv,
)


VERSION = "contralateral-control-oracle-upper-bound-v1"
PANEL = {"R8", "R9", "R10"}
FINDINGS = {
    "Nodule/Mass": "nodule_mass",
    "Lung Opacity": "lung_opacity",
    "Pleural effusion": "pleural_effusion",
}


def stable(seed: int, *parts: str) -> str:
    return hashlib.sha256((str(seed) + ":" + ":".join(parts)).encode()).hexdigest()


def load_image(path: Path) -> tuple[np.ndarray, dict]:
    ds = pydicom.dcmread(str(path), force=True)
    array = ds.pixel_array.astype(np.float32)
    max_value = float(2 ** int(ds.BitsStored) - 1)
    if str(ds.PhotometricInterpretation).upper() == "MONOCHROME1":
        array = max_value - array
    array = np.clip(array / max(max_value, 1.0), 0, 1)
    return array, {"height": int(array.shape[0]), "width": int(array.shape[1])}


def normalized_union(rows: list[dict], width: int, height: int) -> list[float]:
    return [
        min(float(row["x_min"]) for row in rows) / width,
        min(float(row["y_min"]) for row in rows) / height,
        max(float(row["x_max"]) for row in rows) / width,
        max(float(row["y_max"]) for row in rows) / height,
    ]


def pixel_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    pad_x, pad_y = 0.10 * (x1 - x0), 0.10 * (y1 - y0)
    xa = max(1, int(np.floor((x0 - pad_x) * width)))
    xb = min(width - 1, int(np.ceil((x1 + pad_x) * width)))
    ya = max(1, int(np.floor((y0 - pad_y) * height)))
    yb = min(height - 1, int(np.ceil((y1 + pad_y) * height)))
    if xb - xa < 8 or yb - ya < 8:
        raise ValueError("degenerate box")
    return xa, ya, xb, yb


def mirrored_box(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    xa, ya, xb, yb = box
    return width - xb, ya, width - xa, yb


def feather(height: int, width: int) -> np.ndarray:
    distance_y = np.minimum(np.arange(height), np.arange(height)[::-1])[:, None]
    distance_x = np.minimum(np.arange(width), np.arange(width)[::-1])[None, :]
    distance = np.minimum(distance_y, distance_x).astype(np.float32)
    radius = max(2.0, min(height, width) / 6.0)
    return np.sin(0.5 * np.pi * np.clip(distance / radius, 0, 1)) ** 2


def paste_partner(
    image: np.ndarray,
    destination: tuple[int, int, int, int],
    source: tuple[int, int, int, int],
) -> np.ndarray:
    xa, ya, xb, yb = destination
    sx0, sy0, sx1, sy1 = source
    patch = image[sy0:sy1, sx0:sx1][:, ::-1].copy()
    target_shape = (yb - ya, xb - xa)
    patch = np.asarray(
        torch.nn.functional.interpolate(
            torch.from_numpy(patch)[None, None],
            size=target_shape,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0, 0]
    )
    # Match only low-order acquisition intensity; do not histogram-match away
    # focal structure.
    rim = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    target = image[ya:yb, xa:xb]
    target_rim = np.concatenate([target[0], target[-1], target[:, 0], target[:, -1]])
    patch = (patch - rim.mean()) / (rim.std() + 1e-5)
    patch = np.clip(patch * (target_rim.std() + 1e-5) + target_rim.mean(), 0, 1)
    alpha = feather(*target_shape)
    result = image.copy()
    result[ya:yb, xa:xb] = alpha * patch + (1 - alpha) * target
    return result


def to_xrv(image: np.ndarray) -> torch.Tensor:
    height, width = image.shape
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    image = image[top : top + side, left : left + side]
    tensor = torch.from_numpy((image * 2.0 - 1.0) * 1024.0)[None, None]
    return F.interpolate(tensor, (224, 224), mode="bilinear", antialias=True)[0]


def target_logit(logits: np.ndarray, finding_key: str) -> float:
    index = {name: idx for idx, name in enumerate(XRV_LABELS)}
    return float(max(logits[index[name]] for name in FINDING_TARGETS[finding_key]))


def boot_auc(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray, draws: int, seed: int):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    values = []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        if len(np.unique(labels[idx])) == 2:
            values.append(float(roc_auc_score(labels[idx], scores[idx])))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def boot_mean(values: np.ndarray, groups: np.ndarray, draws: int, seed: int):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    estimates = []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        estimates.append(float(values[idx].mean()))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("/workspace/vinbigdata/train.csv"))
    parser.add_argument("--dicom-root", type=Path, default=Path("/workspace/vinbigdata/train"))
    parser.add_argument("--models-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CPU-only gate: set CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(args.threads)

    annotations: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    all_images: set[str] = set()
    with args.csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["rad_id"] not in PANEL:
                continue
            all_images.add(row["image_id"])
            if row["class_name"] in FINDINGS and row["x_min"]:
                annotations[row["image_id"]][row["class_name"]].append(row)

    model = load_xrv(args.models_source, args.checkpoint)
    rows_out = []
    for finding_name, finding_key in FINDINGS.items():
        positives = [
            image_id
            for image_id, by_finding in annotations.items()
            if len({row["rad_id"] for row in by_finding.get(finding_name, [])}) >= 2
            and (args.dicom_root / f"{image_id}.dicom").is_file()
        ]
        positives = sorted(positives, key=lambda x: stable(args.seed, finding_name, "p", x))[: args.per_class]
        excluded = {image_id for image_id, by_finding in annotations.items() if by_finding.get(finding_name)}
        negatives = [
            image_id
            for image_id in all_images
            if image_id not in excluded and (args.dicom_root / f"{image_id}.dicom").is_file()
        ]
        negatives = sorted(negatives, key=lambda x: stable(args.seed, finding_name, "n", x))[: args.per_class]
        if len(positives) < args.per_class or len(negatives) < args.per_class:
            raise RuntimeError(f"insufficient cases for {finding_name}: {len(positives)}/{len(negatives)}")

        templates = []
        for image_id in positives:
            _, shape = load_image(args.dicom_root / f"{image_id}.dicom")
            templates.append(
                normalized_union(annotations[image_id][finding_name], shape["width"], shape["height"])
            )

        cases = [(image_id, 1, templates[i]) for i, image_id in enumerate(positives)]
        cases += [(image_id, 0, templates[i]) for i, image_id in enumerate(negatives)]
        for image_id, label, norm_box in cases:
            image, shape = load_image(args.dicom_root / f"{image_id}.dicom")
            target = pixel_box(norm_box, shape["width"], shape["height"])
            partner = mirrored_box(target, shape["width"])
            # Reject midline boxes: contralateral replacement is not defined there.
            if max(target[0], partner[0]) < min(target[2], partner[2]):
                continue
            target_cf = paste_partner(image, target, partner)
            sham_cf = paste_partner(image, partner, target)
            batch = torch.stack([to_xrv(image), to_xrv(target_cf), to_xrv(sham_cf)])
            with torch.inference_mode():
                if hasattr(model, "features2") and hasattr(model, "classifier"):
                    logits = model.classifier(model.features2(batch)).cpu().numpy()
                else:
                    probs = torch.clamp(model(batch), 1e-6, 1 - 1e-6)
                    logits = torch.logit(probs).cpu().numpy()
            native, target_score, sham_score = [target_logit(row, finding_key) for row in logits]
            rows_out.append(
                {
                    "image_id": image_id,
                    "finding": finding_key,
                    "label": label,
                    "box": norm_box,
                    "native": native,
                    "target_replaced": target_score,
                    "sham_replaced": sham_score,
                    "target_delta": native - target_score,
                    "sham_delta": native - sham_score,
                    "specific_delta": sham_score - target_score,
                }
            )

    labels = np.asarray([row["label"] for row in rows_out])
    groups = np.asarray([row["image_id"] for row in rows_out])
    target_delta = np.asarray([row["target_delta"] for row in rows_out])
    specific_delta = np.asarray([row["specific_delta"] for row in rows_out])
    native = np.asarray([row["native"] for row in rows_out])
    positive = labels == 1
    negative = labels == 0
    statistics = {
        "n": len(rows_out),
        "native_auroc": float(roc_auc_score(labels, native)),
        "target_delta_auroc": float(roc_auc_score(labels, target_delta)),
        "target_delta_auroc_ci95": boot_auc(labels, target_delta, groups, args.bootstrap_draws, args.seed),
        "specific_delta_auroc": float(roc_auc_score(labels, specific_delta)),
        "specific_delta_auroc_ci95": boot_auc(labels, specific_delta, groups, args.bootstrap_draws, args.seed + 1),
        "positive_target_delta_mean": float(target_delta[positive].mean()),
        "positive_target_delta_ci95": boot_mean(target_delta[positive], groups[positive], args.bootstrap_draws, args.seed + 2),
        "negative_target_delta_mean": float(target_delta[negative].mean()),
        "negative_target_delta_ci95": boot_mean(target_delta[negative], groups[negative], args.bootstrap_draws, args.seed + 3),
        "positive_specific_delta_mean": float(specific_delta[positive].mean()),
        "positive_specific_delta_ci95": boot_mean(specific_delta[positive], groups[positive], args.bootstrap_draws, args.seed + 4),
    }
    passes = (
        statistics["target_delta_auroc"] >= 0.65
        and statistics["target_delta_auroc_ci95"][0] > 0.5
        and statistics["positive_target_delta_ci95"][0] > 0
        and statistics["positive_specific_delta_ci95"][0] > 0
    )
    result = {
        "version": VERSION,
        "decision": "PASS_ORACLE_UPPER_BOUND" if passes else "NO_GO",
        "decision_rule": "delta AUROC >=.65 with CI>0.5; positive target and target-vs-sham deltas have CI>0",
        "boundary": "Oracle VinDr boxes; establishes neither deployability nor VLM mitigation.",
        "configuration": {
            "per_class": args.per_class,
            "findings": FINDINGS,
            "panel": sorted(PANEL),
            "seed": args.seed,
            "bootstrap_draws": args.bootstrap_draws,
        },
        "statistics": statistics,
        "rows": rows_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
