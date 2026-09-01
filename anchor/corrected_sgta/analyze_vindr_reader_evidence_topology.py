#!/usr/bin/env python3
"""Audit reader-box topology before claiming a visual-clarity mechanism."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file


VERSION = "vindr-reader-evidence-topology-audit-v1"


def area(box: dict[str, float]) -> float:
    return (float(box["x_max"]) - float(box["x_min"])) * (
        float(box["y_max"]) - float(box["y_min"])
    )


def intersection_area(boxes: tuple[dict[str, float], ...]) -> float:
    x0 = max(float(box["x_min"]) for box in boxes)
    y0 = max(float(box["y_min"]) for box in boxes)
    x1 = min(float(box["x_max"]) for box in boxes)
    y1 = min(float(box["y_max"]) for box in boxes)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def union_area(boxes: list[dict[str, float]]) -> float:
    total = 0.0
    for count in range(1, len(boxes) + 1):
        sign = 1.0 if count % 2 else -1.0
        total += sign * sum(intersection_area(group) for group in combinations(boxes, count))
    return total


def pairwise_iou(boxes: list[dict[str, float]]) -> float:
    values = []
    for left, right in combinations(boxes, 2):
        intersection = intersection_area((left, right))
        denominator = area(left) + area(right) - intersection
        values.append(intersection / denominator if denominator > 0 else 0.0)
    return float(np.mean(values)) if values else float("nan")


def centroid_spread(boxes: list[dict[str, float]], width: float, height: float) -> float:
    centers = [
        (
            0.5 * (float(box["x_min"]) + float(box["x_max"])),
            0.5 * (float(box["y_min"]) + float(box["y_max"])),
        )
        for box in boxes
    ]
    values = [np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in combinations(centers, 2)]
    return float(np.mean(values) / np.hypot(width, height)) if values else float("nan")


def residualize(target: np.ndarray, design: np.ndarray) -> np.ndarray:
    augmented = np.column_stack([np.ones(len(design)), design])
    coefficients, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    return target - augmented @ coefficients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bboxes", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import pydicom

    metadata = [json.loads(line) for line in args.metadata.read_text().splitlines() if line.strip()]
    manifest = {
        f"{row['finding']}:{row['image_id']}": row
        for row in (json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip())
    }
    bbox = {
        f"{row['finding']}:{row['image_id']}": row["boxes"]
        for row in (json.loads(line) for line in args.bboxes.read_text().splitlines() if line.strip())
    }
    dimensions = {}
    records = []
    for item in metadata:
        key = str(item["record_key"])
        source = manifest[key]
        boxes = bbox.get(key, [])
        positive_readers = {
            str(row["rad_id"])
            for row in source["reader_votes"]
            if int(row["vote"]) == 1
        }
        reader_boxes = [box for box in boxes if str(box["rad_id"]) in positive_readers]
        if len({str(box["rad_id"]) for box in reader_boxes}) != len(positive_readers):
            continue
        image_id = str(item["image_id"])
        if image_id not in dimensions:
            header = pydicom.dcmread(
                str(args.image_root / f"{image_id}.dicom"), stop_before_pixels=True
            )
            dimensions[image_id] = (float(header.Columns), float(header.Rows))
        width, height = dimensions[image_id]
        if any(
            not (
                0 <= float(box["x_min"]) < float(box["x_max"]) <= width
                and 0 <= float(box["y_min"]) < float(box["y_max"]) <= height
            )
            for box in reader_boxes
        ):
            raise ValueError(f"bbox outside DICOM dimensions for {key}")
        final_layer = max(map(int, item["diagnostic_plain_logit_lens"]))
        logits = item["diagnostic_plain_logit_lens"][str(final_layer)]
        records.append(
            {
                "record_key": key,
                "image_id": image_id,
                "finding": str(item["finding"]),
                "positive_votes": int(item["positive_votes"]),
                "positive_reader_count": len(positive_readers),
                "pairwise_iou": pairwise_iou(reader_boxes),
                "mean_box_area_fraction": float(np.mean([area(box) for box in reader_boxes]) / (width * height)),
                "union_box_area_fraction": union_area(reader_boxes) / (width * height),
                "centroid_spread": centroid_spread(reader_boxes, width, height),
                "final_commitment": max(float(logits["supported"]), float(logits["refuted"])) - float(logits["undetermined"]),
                "final_polarity": float(logits["supported"]) - float(logits["refuted"]),
            }
        )
    if len(records) != len(metadata):
        raise RuntimeError(f"bbox-complete rows {len(records)} != metadata rows {len(metadata)}")

    labels = np.asarray([int(row["positive_votes"] == 3) for row in records])
    geometry_auc = {}
    for field in ("pairwise_iou", "mean_box_area_fraction", "union_box_area_fraction"):
        values = np.asarray([row[field] for row in records], dtype=float)
        geometry_auc[field] = float(roc_auc_score(labels, values))
    findings = sorted({row["finding"] for row in records})
    finding_columns = np.column_stack(
        [[float(row["finding"] == finding) for row in records] for finding in findings[1:]]
    )
    vote = np.asarray([row["positive_votes"] for row in records], dtype=float)
    agreement = np.asarray([row["pairwise_iou"] for row in records], dtype=float)
    commitment = np.asarray([row["final_commitment"] for row in records], dtype=float)
    nuisance = np.column_stack(
        [
            vote,
            [row["mean_box_area_fraction"] for row in records],
            [row["union_box_area_fraction"] for row in records],
            [row["centroid_spread"] for row in records],
            finding_columns,
        ]
    )
    adjusted_agreement = residualize(agreement, nuisance)
    adjusted_commitment = residualize(commitment, nuisance)
    partial = spearmanr(adjusted_agreement, adjusted_commitment)
    within_bin = {}
    for finding in findings:
        within_bin[finding] = {}
        for value in (2, 3):
            subset = [row for row in records if row["finding"] == finding and row["positive_votes"] == value]
            within_bin[finding][str(value)] = {
                "n": len(subset),
                "iou_mean": float(np.mean([row["pairwise_iou"] for row in subset])),
                "iou_std": float(np.std([row["pairwise_iou"] for row in subset])),
                "commitment_iou_spearman": float(
                    spearmanr(
                        [row["pairwise_iou"] for row in subset],
                        [row["final_commitment"] for row in subset],
                    ).statistic
                ),
            }
    output = {
        "version": VERSION,
        "n": len(records),
        "construct_name": "reader_grounded_evidence_topology_not_pure_visual_clarity",
        "geometry_vote_3_vs_2_auroc": geometry_auc,
        "partial_spearman_spatial_agreement_vs_commitment": {
            "estimate": float(partial.statistic),
            "pvalue": float(partial.pvalue),
            "nuisance_controls": [
                "vote_bin", "mean_box_area", "union_box_area", "centroid_spread", "finding"
            ],
        },
        "within_finding_vote_bin": within_bin,
        "records": records,
        "provenance": {
            "metadata_sha256": sha256_file(args.metadata),
            "manifest_sha256": sha256_file(args.manifest),
            "bboxes_sha256": sha256_file(args.bboxes),
        },
    }
    atomic_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
