#!/usr/bin/env python3
"""Audit public CXR dataset-domain and FedDG frequency hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def stable_sample(paths: list[Path], maximum: int, seed: int) -> list[Path]:
    scored = [
        (hashlib.sha256(f"{seed}:{path}".encode()).hexdigest(), path)
        for path in paths
    ]
    return [path for _, path in sorted(scored)[:maximum]]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def public_domain_paths(root: Path, maximum: int, seed: int) -> dict[str, list[Path]]:
    mimic = [
        root / row["path_in_repo"]
        for row in load_jsonl(root / "data/mimic_cxr_rule/image_manifest.jsonl")
    ]

    iu_rows = json.loads(
        (root / "data/mmedrag/test/report/iuxray_test.json").read_text()
    )
    iu = [
        root / "data/medheval/images/IU-Xray" / row["image_path"][0]
        for row in iu_rows
        if row.get("image_path")
    ]

    chexpert = [
        root / "data/chexpert_subset_report" / row["relative_path"]
        for row in load_jsonl(
            root / "data/chexpert_subset_report/image_manifest.jsonl"
        )
    ]
    sources = {"mimic": mimic, "iuxray": iu, "chexpert_proxy": chexpert}
    return {
        name: stable_sample([path for path in paths if path.is_file()], maximum, seed)
        for name, paths in sources.items()
    }


def visible_array(path: Path, size: int, crop_fraction: float) -> np.ndarray:
    with Image.open(path) as raw:
        image = raw.convert("RGB")
        if crop_fraction < 1.0:
            width, height = image.size
            dx = round(width * (1.0 - crop_fraction) / 2.0)
            dy = round(height * (1.0 - crop_fraction) / 2.0)
            image = image.crop((dx, dy, width - dx, height - dy))
        image = image.resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.float32) / 255.0


def radial_profile(values: np.ndarray, bins: int) -> np.ndarray:
    height, width = values.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2)
    index = np.minimum((radius / radius.max() * bins).astype(int), bins - 1)
    return np.asarray(
        [np.median(values[index == band]) for band in range(bins)],
        dtype=np.float32,
    )


def descriptors(image: np.ndarray, bins: int) -> dict[str, np.ndarray]:
    gray = image.mean(axis=2)
    amplitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    radial = radial_profile(amplitude, bins)
    low_count = max(2, bins // 8)
    high_start = bins // 2
    stats = np.asarray(
        [
            *image.mean(axis=(0, 1)),
            *image.std(axis=(0, 1)),
            np.quantile(gray, 0.05),
            np.quantile(gray, 0.50),
            np.quantile(gray, 0.95),
        ],
        dtype=np.float32,
    )
    return {
        "intensity_stats": stats,
        "radial_all": radial,
        "radial_low": radial[:low_count],
        "radial_mid_equal": radial[bins // 2 : bins // 2 + low_count],
        "radial_high_equal": radial[-low_count:],
        "radial_high": radial[high_start:],
    }


def classifier() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )


def cross_validated_scores(
    features: np.ndarray, labels: np.ndarray, folds: int
) -> dict[str, float]:
    predicted = np.empty_like(labels)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for train, test in splitter.split(features, labels):
        model = classifier().fit(features[train], labels[train])
        predicted[test] = model.predict(features[test])
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "chance_accuracy": float(1.0 / len(np.unique(labels))),
    }


def feddg_swap(source: np.ndarray, target: np.ndarray, ratio: float) -> np.ndarray:
    src_fft = np.fft.fft2(source, axes=(0, 1))
    trg_fft = np.fft.fft2(target, axes=(0, 1))
    src_amp = np.fft.fftshift(np.abs(src_fft), axes=(0, 1))
    trg_amp = np.fft.fftshift(np.abs(trg_fft), axes=(0, 1))
    phase = np.angle(src_fft)
    border = max(1, int(math.floor(min(source.shape[:2]) * ratio)))
    cy, cx = source.shape[0] // 2, source.shape[1] // 2
    src_amp[
        cy - border : cy + border + 1, cx - border : cx + border + 1
    ] = trg_amp[cy - border : cy + border + 1, cx - border : cx + border + 1]
    restored_amp = np.fft.ifftshift(src_amp, axes=(0, 1))
    restored = np.fft.ifft2(restored_amp * np.exp(1j * phase), axes=(0, 1)).real
    return np.clip(restored, 0.0, 1.0).astype(np.float32)


def structure_metrics(source: np.ndarray, changed: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((source - changed) ** 2))
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    source_edge = ndimage.sobel(source.mean(axis=2), axis=0) ** 2
    source_edge += ndimage.sobel(source.mean(axis=2), axis=1) ** 2
    changed_edge = ndimage.sobel(changed.mean(axis=2), axis=0) ** 2
    changed_edge += ndimage.sobel(changed.mean(axis=2), axis=1) ** 2
    source_flat = np.sqrt(source_edge).ravel()
    changed_flat = np.sqrt(changed_edge).ravel()
    correlation = float(np.corrcoef(source_flat, changed_flat)[0, 1])
    return {"psnr": psnr, "edge_correlation": correlation}


def mean_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def transfer_audit(
    images: list[np.ndarray],
    labels: np.ndarray,
    names: list[str],
    bins: int,
    ratios: list[float],
    seed: int,
) -> dict[str, Any]:
    indices = np.arange(len(images))
    train, test = train_test_split(
        indices, test_size=0.25, random_state=seed, stratify=labels
    )
    feature_names = ("radial_all", "radial_low")
    models = {}
    for feature_name in feature_names:
        train_features = np.stack(
            [descriptors(images[index], bins)[feature_name] for index in train]
        )
        models[feature_name] = classifier().fit(train_features, labels[train])

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    train_by_label = {
        label: train[labels[train] == label] for label in np.unique(labels)
    }
    for index in test:
        source_label = int(labels[index])
        target_label = int(rng.choice([x for x in np.unique(labels) if x != source_label]))
        target_index = int(rng.choice(train_by_label[target_label]))
        for ratio in ratios:
            changed = feddg_swap(images[index], images[target_index], ratio)
            row: dict[str, Any] = {
                "source": names[source_label],
                "target": names[target_label],
                "ratio": ratio,
                **structure_metrics(images[index], changed),
            }
            for feature_name, model in models.items():
                original_feature = descriptors(images[index], bins)[feature_name][None]
                changed_feature = descriptors(changed, bins)[feature_name][None]
                original_probability = model.predict_proba(original_feature)[0]
                changed_probability = model.predict_proba(changed_feature)[0]
                row[feature_name] = {
                    "source_probability_delta": float(
                        changed_probability[source_label]
                        - original_probability[source_label]
                    ),
                    "target_probability_delta": float(
                        changed_probability[target_label]
                        - original_probability[target_label]
                    ),
                    "predicts_target": bool(
                        int(model.predict(changed_feature)[0]) == target_label
                    ),
                }
            rows.append(row)

    summary: dict[str, Any] = {}
    for ratio in ratios:
        selected = [row for row in rows if row["ratio"] == ratio]
        summary[str(ratio)] = {
            "n": len(selected),
            "psnr": mean_summary([row["psnr"] for row in selected]),
            "edge_correlation": mean_summary(
                [row["edge_correlation"] for row in selected]
            ),
        }
        for feature_name in feature_names:
            summary[str(ratio)][feature_name] = {
                "source_probability_delta": mean_summary(
                    [
                        row[feature_name]["source_probability_delta"]
                        for row in selected
                    ]
                ),
                "target_probability_delta": mean_summary(
                    [
                        row[feature_name]["target_probability_delta"]
                        for row in selected
                    ]
                ),
                "target_prediction_rate": float(
                    np.mean([row[feature_name]["predicts_target"] for row in selected])
                ),
            }
    return {"summary": summary, "records": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-domain", type=int, default=500)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feddg-ratios", type=float, nargs="+", default=(0.01, 0.03))
    args = parser.parse_args()
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    paths_by_domain = public_domain_paths(
        args.repo_root.resolve(), args.samples_per_domain, args.seed
    )
    names = list(paths_by_domain)
    label_by_name = {name: index for index, name in enumerate(names)}

    audit: dict[str, Any] = {}
    full_images: list[np.ndarray] = []
    full_labels: list[int] = []
    read_errors: list[dict[str, str]] = []
    for crop_name, crop_fraction in (("full", 1.0), ("center80", 0.8)):
        feature_rows: dict[str, list[np.ndarray]] = {}
        labels: list[int] = []
        for name, paths in paths_by_domain.items():
            for path in paths:
                try:
                    image = visible_array(path, args.size, crop_fraction)
                    for feature_name, feature in descriptors(image, args.bins).items():
                        feature_rows.setdefault(feature_name, []).append(feature)
                    labels.append(label_by_name[name])
                    if crop_name == "full":
                        full_images.append(image)
                        full_labels.append(label_by_name[name])
                except Exception as exc:
                    read_errors.append({"path": str(path), "error": repr(exc)})
        label_array = np.asarray(labels, dtype=np.int64)
        audit[crop_name] = {
            feature_name: cross_validated_scores(
                np.stack(features), label_array, args.folds
            )
            for feature_name, features in feature_rows.items()
        }

    transfer = transfer_audit(
        full_images,
        np.asarray(full_labels, dtype=np.int64),
        names,
        args.bins,
        list(args.feddg_ratios),
        args.seed,
    )
    payload = {
        "version": "public-cxr-domain-hypothesis-audit-v1",
        "limitations": [
            "Public dataset identity is a proxy for institution and export pipeline, not pure hospital acquisition style.",
            "center80 removes borders but is not a learned lung-field segmentation.",
            "PSNR and edge correlation cannot establish clinical-content preservation.",
        ],
        "config": {
            "samples_per_domain": args.samples_per_domain,
            "size": args.size,
            "bins": args.bins,
            "folds": args.folds,
            "seed": args.seed,
            "feddg_ratios": args.feddg_ratios,
        },
        "domains": {
            name: {"requested": args.samples_per_domain, "available": len(paths)}
            for name, paths in paths_by_domain.items()
        },
        "source_classification": audit,
        "feddg_transfer": transfer,
        "read_errors": read_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "domains": payload["domains"],
                "source_classification": audit,
                "feddg_transfer": transfer["summary"],
                "read_error_count": len(read_errors),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
