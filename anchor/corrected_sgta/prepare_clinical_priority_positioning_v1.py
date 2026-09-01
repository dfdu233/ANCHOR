#!/usr/bin/env python3
"""Prepare a label-blind canary for clinical-priority visual positioning.

The output freezes claim rows and exact TorchXRayVision class-activation maps
before either medical VLM is loaded.  Labels are used only to stratify the
evaluation panel; they never enter the priority map or the VLM intervention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from anchor.corrected_sgta.screen_xrv_visual_increment_v1 import (
    FINDING_TARGETS,
    XRV_LABELS,
    dicom_tensor,
    load_xrv,
)


CSV_FINDINGS = {
    "aortic_enlargement": "Aortic enlargement",
    "cardiomegaly": "Cardiomegaly",
    "lung_opacity": "Lung Opacity",
    "nodule_mass": "Nodule/Mass",
    "pleural_effusion": "Pleural effusion",
    "pleural_thickening": "Pleural thickening",
    "pulmonary_fibrosis": "Pulmonary fibrosis",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bbox_index(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    boxes: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["x_min"]:
                continue
            key = (row["image_id"], row["class_name"])
            boxes[key].append(tuple(float(row[name]) for name in ("x_min", "y_min", "x_max", "y_max")))
    output = {}
    for key, values in boxes.items():
        x0 = min(value[0] for value in values)
        y0 = min(value[1] for value in values)
        x1 = max(value[2] for value in values)
        y1 = max(value[3] for value in values)
        output[key] = {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}
    return output


def select_panel(
    manifest: Path,
    boxes: dict[tuple[str, str], dict[str, float]],
    image_root: Path,
    small: int,
    large: int,
    negative: int,
) -> list[dict[str, Any]]:
    source = read_jsonl(manifest)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        finding = str(row["finding"])
        if finding in CSV_FINDINGS:
            grouped[finding].append(row)
    panel: list[dict[str, Any]] = []
    for finding in sorted(CSV_FINDINGS):
        positives = []
        negatives = []
        for row in grouped[finding]:
            image_id = str(row["image_id"])
            votes = int(row["positive_votes"])
            box = boxes.get((image_id, CSV_FINDINGS[finding]))
            if votes == 3 and box is not None:
                import pydicom

                ds = pydicom.dcmread(str(image_root / f"{image_id}.dicom"), stop_before_pixels=True)
                area = (box["x_max"] - box["x_min"]) * (box["y_max"] - box["y_min"])
                area /= float(ds.Rows * ds.Columns)
                positives.append((area, row, box))
            elif votes == 0:
                negatives.append(row)
        positives.sort(key=lambda item: (item[0], item[1]["image_id"]))
        negatives.sort(key=lambda item: item["image_id"])
        chosen_pos = positives[:small]
        if large:
            chosen_pos += positives[-large:]
        if len(chosen_pos) != small + large or len(negatives) < negative:
            raise RuntimeError(f"insufficient panel for {finding}")
        for rank, (area, row, box) in enumerate(chosen_pos):
            panel.append(
                {
                    "image_id": str(row["image_id"]),
                    "finding": finding,
                    "label": 1,
                    "positive_votes": 3,
                    "size_stratum": "small" if rank < small else "large",
                    "bbox_area_ratio": area,
                    "bbox": box,
                }
            )
        for row in negatives[:negative]:
            panel.append(
                {
                    "image_id": str(row["image_id"]),
                    "finding": finding,
                    "label": 0,
                    "positive_votes": 0,
                    "size_stratum": "negative",
                    "bbox_area_ratio": None,
                    "bbox": None,
                }
            )
    return panel


def encode_cams(
    panel: list[dict[str, Any]],
    image_root: Path,
    models_source: Path,
    checkpoint: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model = load_xrv(models_source, checkpoint)
    label_index = {label: index for index, label in enumerate(XRV_LABELS)}
    cams, logits, chosen_labels = [], [], []
    with torch.inference_mode():
        for row in panel:
            image = dicom_tensor(image_root / f"{row['image_id']}.dicom")[None]
            feature_map = F.relu(model.features(image), inplace=False)
            targets = FINDING_TARGETS[row["finding"]]
            indices = torch.tensor([label_index[target] for target in targets])
            weights = model.classifier.weight.index_select(0, indices)
            biases = model.classifier.bias.index_select(0, indices)
            candidate_cams = torch.einsum("oc,bchw->bohw", weights, feature_map)[0]
            candidate_logits = candidate_cams.mean(dim=(-2, -1)) + biases
            chosen = int(candidate_logits.argmax())
            direct = model.classifier(F.adaptive_avg_pool2d(feature_map, 1).flatten(1))[0, indices]
            if not torch.allclose(candidate_logits, direct, atol=1e-5, rtol=1e-5):
                raise RuntimeError("CAM identity failed")
            cams.append(candidate_cams[chosen].cpu().numpy().astype(np.float32))
            logits.append(candidate_logits.cpu().numpy().astype(np.float32))
            chosen_labels.append(targets[chosen])
    width = max(len(value) for value in logits)
    padded = np.full((len(logits), width), np.nan, dtype=np.float32)
    for index, value in enumerate(logits):
        padded[index, : len(value)] = value
    return np.stack(cams), padded, chosen_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bbox-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--models-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-per-finding", type=int, default=4)
    parser.add_argument("--large-per-finding", type=int, default=2)
    parser.add_argument("--negative-per-finding", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    boxes = bbox_index(args.bbox_csv)
    panel = select_panel(
        args.manifest,
        boxes,
        args.image_root,
        args.small_per_finding,
        args.large_per_finding,
        args.negative_per_finding,
    )
    cams, logits, chosen = encode_cams(
        panel, args.image_root, args.models_source, args.checkpoint
    )
    provenance = {
        "protocol": "clinical-priority-positioning-preparation-v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "bbox_csv": str(args.bbox_csv.resolve()),
        "bbox_csv_sha256": sha256(args.bbox_csv),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "models_source": str(args.models_source.resolve()),
        "models_source_sha256": sha256(args.models_source),
        "selection": {
            "small_per_finding": args.small_per_finding,
            "large_per_finding": args.large_per_finding,
            "negative_per_finding": args.negative_per_finding,
        },
        "label_blind_intervention": "labels select the frozen panel only; XRV CAM determines token priority",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        panel=np.asarray([json.dumps(row, sort_keys=True) for row in panel]),
        cams=cams,
        target_logits=logits,
        chosen_xrv_labels=np.asarray(chosen),
        provenance=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    print(json.dumps({"status": "complete", "rows": len(panel), "cam_shape": list(cams.shape)}))


if __name__ == "__main__":
    main()
