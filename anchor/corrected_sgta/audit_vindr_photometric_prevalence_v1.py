#!/usr/bin/env python3
"""Audit whether a real VinDr acquisition tag is visually recoverable and label-coupled.

This CPU-only admission test does not score a VLM.  It asks whether the
MONOCHROME1/2 DICOM source proxy (a) carries a large reader-prevalence vector
and (b) remains predictable from standardized render style among unanimously
normal images, where pathology content cannot explain the classifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .audit_diagnostic_completion_substrate_v1 import sha256_file
from .prepare_pragmatic_commitment_confirmatory_v1 import canonical_hash
from .run_huatuo_dicom_render_pilot_v1 import percentile_render, read_dicom_pixels


VERSION = "vindr-photometric-prevalence-admission-v1"
GROUPS = ("MONOCHROME1", "MONOCHROME2")
TRAIN_PER_GROUP = 400
TEST_PER_GROUP = 400
SEED = 67241


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def load_reader_panel(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        labels = [name for name in (reader.fieldnames or []) if name not in {"image_id", "rad_id"}]
        for row in reader:
            by_image[str(row["image_id"])].append(row)
    panel = {}
    for image_id, rows in by_image.items():
        if len(rows) != 3:
            continue
        votes = {
            label: sum(int(row[label]) for row in rows)
            for label in labels
        }
        disease_labels = [label for label in labels if label != "No finding"]
        unanimously_normal = bool(
            votes["No finding"] == 3
            and all(votes[label] == 0 for label in disease_labels)
        )
        panel[image_id] = {
            "votes": votes,
            "unanimously_normal": unanimously_normal,
        }
    return labels, panel


def dicom_header(path: Path) -> tuple[str, str]:
    import pydicom

    dataset = pydicom.dcmread(
        str(path), stop_before_pixels=True, force=True,
        specific_tags=["PhotometricInterpretation"],
    )
    return path.stem, str(getattr(dataset, "PhotometricInterpretation", "")).upper()


def render_features(path: Path) -> tuple[str, list[float]]:
    from PIL import Image

    pixels = read_dicom_pixels(path)
    rendered = percentile_render(pixels)
    resized = np.asarray(
        Image.fromarray(np.uint8(np.round(rendered * 255.0))).resize((128, 128)),
        dtype=np.float32,
    ) / 255.0
    histogram = np.histogram(resized, bins=32, range=(0.0, 1.0), density=True)[0]
    dy = np.diff(resized, axis=0, append=resized[-1:])
    dx = np.diff(resized, axis=1, append=resized[:, -1:])
    gradient = np.sqrt(dx * dx + dy * dy)
    gradient_histogram = np.histogram(
        gradient, bins=16, range=(0.0, 0.5), density=True
    )[0]
    spectrum = np.log1p(np.abs(np.fft.rfft2(resized - resized.mean())))
    radial_summary = []
    height, width = spectrum.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy / max(height - 1, 1)) ** 2 + (xx / max(width - 1, 1)) ** 2)
    for left, right in zip(np.linspace(0.0, 1.0, 17)[:-1], np.linspace(0.0, 1.0, 17)[1:]):
        values = spectrum[(radius >= left) & (radius < right)]
        radial_summary.append(float(values.mean()) if values.size else 0.0)
    border = np.concatenate(
        (resized[:8].ravel(), resized[-8:].ravel(), resized[:, :8].ravel(), resized[:, -8:].ravel())
    )
    scalars = [
        float(resized.mean()),
        float(resized.std()),
        *[float(value) for value in np.quantile(resized, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])],
        float(gradient.mean()),
        float(gradient.std()),
        float(border.mean()),
        float(border.std()),
        math.log(float(pixels.metadata["rows"]) / float(pixels.metadata["columns"])),
        math.log(float(pixels.metadata["rows"]) * float(pixels.metadata["columns"])),
    ]
    feature = np.concatenate(
        (np.asarray(scalars), histogram, gradient_histogram, np.asarray(radial_summary))
    )
    if not np.isfinite(feature).all():
        raise ValueError(f"non-finite render feature: {path}")
    return path.stem, feature.astype(float).tolist()


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("photometric prevalence audit is write-once")

    from multiprocessing import Pool
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels, panel = load_reader_panel(args.labels)
    paths = sorted(args.image_root.glob("*.dicom"))
    with Pool(args.workers) as pool:
        headers = dict(pool.map(dicom_header, paths, chunksize=100))
    eligible_ids = sorted(set(panel) & set(headers))
    group_counts = Counter(headers[image_id] for image_id in eligible_ids)
    prevalence = {}
    for group in GROUPS:
        group_ids = [image_id for image_id in eligible_ids if headers[image_id] == group]
        prevalence[group] = {
            label: float(np.mean([panel[image_id]["votes"][label] / 3.0 for image_id in group_ids]))
            for label in labels
        }
    prevalence_delta = {}
    for label in labels:
        left, right = prevalence[GROUPS[0]][label], prevalence[GROUPS[1]][label]
        logit = lambda value: math.log((value + 1e-3) / (1.0 - value + 1e-3))
        prevalence_delta[label] = {
            GROUPS[0]: left,
            GROUPS[1]: right,
            "log_odds_delta_mono1_minus_mono2": logit(left) - logit(right),
        }

    selected: dict[str, dict[str, list[str]]] = {}
    for group in GROUPS:
        normals = [
            image_id
            for image_id in eligible_ids
            if headers[image_id] == group and panel[image_id]["unanimously_normal"]
        ]
        ordered = sorted(
            normals,
            key=lambda image_id: hashlib.sha256(
                f"{VERSION}:{SEED}:{group}:{image_id}".encode()
            ).hexdigest(),
        )
        required = TRAIN_PER_GROUP + TEST_PER_GROUP
        if len(ordered) < required:
            raise ValueError(f"not enough unanimous-normal images for {group}: {len(ordered)}")
        selected[group] = {
            "train": ordered[:TRAIN_PER_GROUP],
            "test": ordered[TRAIN_PER_GROUP:required],
        }
    selected_ids = [
        image_id
        for group in GROUPS
        for split in ("train", "test")
        for image_id in selected[group][split]
    ]
    with Pool(args.workers) as pool:
        features = dict(
            pool.map(
                render_features,
                [args.image_root / f"{image_id}.dicom" for image_id in selected_ids],
                chunksize=8,
            )
        )

    def matrix(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        ids = [image_id for group in GROUPS for image_id in selected[group][split]]
        x = np.asarray([features[image_id] for image_id in ids], dtype=np.float64)
        y = np.asarray([int(headers[image_id] == "MONOCHROME1") for image_id in ids])
        return x, y, ids

    x_train, y_train, train_ids = matrix("train")
    x_test, y_test, test_ids = matrix("test")
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
    )
    classifier.fit(x_train, y_train)
    probability = classifier.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, probability))
    accuracy = float(accuracy_score(y_test, prediction))
    rng = np.random.default_rng(SEED)
    auc_draws, accuracy_draws = [], []
    for _ in range(5000):
        indices = np.concatenate(
            [
                rng.choice(np.flatnonzero(y_test == value), size=int((y_test == value).sum()), replace=True)
                for value in (0, 1)
            ]
        )
        auc_draws.append(roc_auc_score(y_test[indices], probability[indices]))
        accuracy_draws.append(accuracy_score(y_test[indices], prediction[indices]))

    payload = {
        "version": VERSION,
        "status": "cpu_admission_only_no_vlm_scored",
        "labels_path": str(args.labels.resolve()),
        "labels_sha256": sha256_file(args.labels),
        "image_root": str(args.image_root.resolve()),
        "registered_dicom_count": len(paths),
        "exact_three_reader_images": len(panel),
        "photometric_counts": dict(sorted(group_counts.items())),
        "prevalence": prevalence,
        "prevalence_delta": prevalence_delta,
        "normal_only_source_classifier": {
            "contract": (
                "0.5/99.5 percentile render with MONOCHROME-aware canonical polarity; "
                "histogram, gradient, spectrum, border, and shape features; unanimous-normal only"
            ),
            "train_per_group": TRAIN_PER_GROUP,
            "test_per_group": TEST_PER_GROUP,
            "feature_dimension": int(x_train.shape[1]),
            "auc": auc,
            "auc_ci95": percentile_interval(np.asarray(auc_draws)),
            "accuracy": accuracy,
            "accuracy_ci95": percentile_interval(np.asarray(accuracy_draws)),
        },
        "selection": selected,
        "train_ids_sha256": canonical_hash(train_ids),
        "test_ids_sha256": canonical_hash(test_ids),
        "seed": SEED,
        "admission_rule": (
            "retain only if normal-only held-out source AUROC lower CI exceeds 0.80 and "
            "the reader-prevalence vector is nontrivial; this does not establish VLM shortcut use"
        ),
        "promotion_prohibited_without": [
            "same-image label-preserving source-style transport",
            "claim-logit shift alignment with the frozen prevalence vector",
            "content/evidence preservation controls",
            "held-out findings and second model",
        ],
    }
    payload["fingerprint"] = canonical_hash(payload)
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "auc": auc,
                "auc_ci95": payload["normal_only_source_classifier"]["auc_ci95"],
                "accuracy": accuracy,
                "admission_passed": payload["normal_only_source_classifier"]["auc_ci95"][0] > 0.80,
                "fingerprint": payload["fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
