#!/usr/bin/env python3
"""Build the frozen, outcome-blind metric-calibration probe for VinDr/VinBigData.

The probe never treats DICOM PixelSpacing as patient-size truth.  It uses
reader-consensus boxes only to stabilize object localization and introduces
explicitly hypothetical, certified patient-plane calibrations for the exact
transformation-law conditions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pydicom
from PIL import Image


VERSION = "metric-calibration-probe-manifest-v1"
SEED = 20260803
FINDING = "Nodule/Mass"
MIN_READERS = 3
MIN_MEDIAN_IOU = 0.5
DEFAULT_DATA = Path("/workspace/vinbigdata")
DEFAULT_OUTPUT = Path("corrected_runs/metric_calibration_probe_v1")
CONDITIONS = (
    "certified_x0p5",
    "certified_x1",
    "certified_x2",
    "certified_cm",
    "missing",
    "detector_only",
    "header_unknown",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(*parts: object) -> str:
    return hashlib.sha256(":".join((str(SEED), *(str(part) for part in parts))).encode()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_identical(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def consensus_box(boxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    return tuple(float(statistics.median(box[index] for box in boxes)) for index in range(4))  # type: ignore[return-value]


def box_endpoints(box: Sequence[float], rows: int, columns: int) -> tuple[tuple[float, float], tuple[float, float]]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if (x2 - x1) >= (y2 - y1):
        points = ((x1 / columns, cy / rows), (x2 / columns, cy / rows))
    else:
        points = ((cx / columns, y1 / rows), (cx / columns, y2 / rows))
    return tuple(tuple(round(max(0.0, min(1.0, value)), 8) for value in point) for point in points)  # type: ignore[return-value]


def nominal_distance_mm(
    endpoints: Sequence[Sequence[float]], rows: int, columns: int, spacing: Sequence[float]
) -> float:
    dx = (endpoints[1][0] - endpoints[0][0]) * columns * spacing[1]
    dy = (endpoints[1][1] - endpoints[0][1]) * rows * spacing[0]
    return math.hypot(dx, dy)


def render_dicom(dicom_path: Path, output_path: Path, maximum_side: int = 1024) -> None:
    dataset = pydicom.dcmread(dicom_path)
    pixels = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept
    finite = pixels[np.isfinite(pixels)]
    if not finite.size:
        raise ValueError(f"DICOM has no finite pixels: {dicom_path}")
    low, high = np.percentile(finite, [0.5, 99.5])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    normalized = np.clip((pixels - low) / max(high - low, 1e-6), 0, 1)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        normalized = 1.0 - normalized
    image = Image.fromarray(np.uint8(np.round(normalized * 255)), mode="L").convert("RGB")
    scale = min(1.0, maximum_side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.png")
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, output_path)


def metadata_line(condition: str, base_spacing: Sequence[float]) -> tuple[str, str, float | None, str | None]:
    row, column = base_spacing
    if condition == "certified_x0p5":
        factor, unit = 0.5, "mm"
    elif condition == "certified_x1":
        factor, unit = 1.0, "mm"
    elif condition == "certified_x2":
        factor, unit = 2.0, "mm"
    elif condition == "certified_cm":
        factor, unit = 0.1, "cm"
    elif condition == "missing":
        return "No pixel calibration or scale bar is available.", "pixel-only/unknown", None, None
    elif condition == "detector_only":
        return (
            f"Detector-plane pixel pitch is {row:.6f} mm/pixel vertically and {column:.6f} mm/pixel horizontally; patient magnification is unknown.",
            "detector-mm",
            None,
            None,
        )
    elif condition == "header_unknown":
        return (
            f"The header contains nominal spacing {row:.6f} by {column:.6f} mm/pixel, but its calibration plane and magnification correction are undocumented.",
            "pixel-only/unknown",
            None,
            None,
        )
    else:
        raise ValueError(condition)
    shown_row, shown_column = row * factor, column * factor
    return (
        f"For this controlled task, patient-plane calibration is certified at {shown_row:.6f} {unit}/pixel vertically and {shown_column:.6f} {unit}/pixel horizontally.",
        "patient-mm",
        factor,
        unit,
    )


def structured_prompt(metadata: str, endpoints: Sequence[Sequence[float]] | None) -> str:
    endpoint_instruction = (
        "Use these certified normalized endpoints (x,y): " + canonical(endpoints) + "."
        if endpoints is not None
        else "Locate the single pulmonary nodule/mass and estimate normalized endpoints (x,y) of its longest apparent box extent."
    )
    return f"""Analyze the supplied chest radiograph for one pulmonary nodule/mass.
{metadata}
{endpoint_instruction}
Return one JSON object only with exactly these fields:
{{"visible":"yes|no|uncertain","endpoints_normalized":[[x1,y1],[x2,y2]] or null,"measurement_type":"patient-mm|detector-mm|pixel-only/unknown","physical_value":number or null,"unit":"mm|cm" or null}}
Emit a physical value only when patient-plane calibration is certified. Detector pitch is not patient size."""


def direct_prompt(metadata: str) -> str:
    return f"""Review the supplied chest radiograph for the pulmonary nodule/mass.
{metadata}
What is its maximum apparent diameter in millimeters? If patient-space size is not identifiable, state that directly and name the missing evidence. Do not guess."""


def build(data_root: Path, output: Path) -> dict[str, Any]:
    csv_path = data_root / "train.csv"
    grouped: dict[str, dict[str, list[tuple[float, float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["class_name"] != FINDING:
                continue
            grouped[row["image_id"]][row["rad_id"]].append(
                tuple(float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
            )

    eligible = []
    for image_id, by_reader in grouped.items():
        if len(by_reader) < MIN_READERS or any(len(boxes) != 1 for boxes in by_reader.values()):
            continue
        boxes = [boxes[0] for boxes in by_reader.values()]
        median_iou = statistics.median(box_iou(a, b) for a, b in itertools.combinations(boxes, 2))
        if median_iou >= MIN_MEDIAN_IOU:
            eligible.append((image_id, by_reader, boxes, float(median_iou)))
    eligible.sort(key=lambda value: stable_key(value[0]))

    image_rows, prompt_rows = [], []
    for image_id, by_reader, boxes, median_iou in eligible:
        dicom_path = data_root / "train" / f"{image_id}.dicom"
        header = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        rows, columns = int(header.Rows), int(header.Columns)
        spacing_raw = getattr(header, "PixelSpacing", None)
        if spacing_raw is None:
            base_spacing = (0.15, 0.15)
            spacing_seed_source = "frozen_hypothetical_default_not_dicom_truth"
        else:
            base_spacing = (float(spacing_raw[0]), float(spacing_raw[1]))
            spacing_seed_source = "dicom_nominal_seed_not_patient_truth"
        consensus = consensus_box(boxes)
        endpoints = box_endpoints(consensus, rows, columns)
        image_path = output / "images" / f"{image_id}.png"
        render_dicom(dicom_path, image_path)
        image_row = {
            "version": VERSION,
            "image_id": image_id,
            "dicom_path": str(dicom_path.resolve()),
            "dicom_sha256": sha256_file(dicom_path),
            "rendered_image": str(image_path.resolve()),
            "rendered_sha256": sha256_file(image_path),
            "rows": rows,
            "columns": columns,
            "readers": sorted(by_reader),
            "reader_count": len(by_reader),
            "median_pairwise_iou": median_iou,
            "reader_boxes": {reader: list(values[0]) for reader, values in sorted(by_reader.items())},
            "consensus_box": list(consensus),
            "oracle_endpoints_normalized": endpoints,
            "base_spacing": base_spacing,
            "spacing_seed_source": spacing_seed_source,
            "dicom_calibration_provenance": {
                "PixelSpacing_present": spacing_raw is not None,
                "ImagerPixelSpacing_present": hasattr(header, "ImagerPixelSpacing"),
                "NominalScannedPixelSpacing_present": hasattr(header, "NominalScannedPixelSpacing"),
                "PixelSpacingCalibrationType_present": hasattr(header, "PixelSpacingCalibrationType"),
            },
        }
        image_rows.append(image_row)
        for arm, arm_endpoints in (("oracle_coordinate", endpoints), ("vision_coordinate", None)):
            for condition in CONDITIONS:
                metadata, expected_type, factor, unit = metadata_line(condition, base_spacing)
                expected_value = None
                if factor is not None:
                    value_mm = nominal_distance_mm(endpoints, rows, columns, base_spacing) * factor
                    expected_value = value_mm if unit == "mm" else value_mm / 10.0
                prompt_rows.append(
                    {
                        "version": VERSION,
                        "item_id": f"{image_id}:{arm}:{condition}:structured",
                        "image_id": image_id,
                        "image_path": str(image_path.resolve()),
                        "arm": arm,
                        "condition": condition,
                        "question_contract": "structured_neutral",
                        "prompt": structured_prompt(metadata, arm_endpoints),
                        "expected_measurement_type": expected_type,
                        "expected_physical_value": expected_value,
                        "expected_unit": unit,
                        "patient_value_identifiable": factor is not None,
                        "model_outcome_opened": False,
                    }
                )
        for condition in ("missing", "detector_only", "header_unknown", "certified_x1"):
            metadata, expected_type, factor, unit = metadata_line(condition, base_spacing)
            prompt_rows.append(
                {
                    "version": VERSION,
                    "item_id": f"{image_id}:vision_coordinate:{condition}:direct",
                    "image_id": image_id,
                    "image_path": str(image_path.resolve()),
                    "arm": "vision_coordinate",
                    "condition": condition,
                    "question_contract": "clinical_direct",
                    "prompt": direct_prompt(metadata),
                    "expected_measurement_type": expected_type,
                    "expected_physical_value": None,
                    "expected_unit": unit,
                    "patient_value_identifiable": factor is not None,
                    "model_outcome_opened": False,
                }
            )

    images_text = "".join(canonical(row) + "\n" for row in image_rows)
    prompts_text = "".join(canonical(row) + "\n" for row in prompt_rows)
    audit = {
        "version": VERSION,
        "decision": "MANIFEST_READY_GPU_NOT_AUTHORIZED_BY_THIS_ARTIFACT",
        "finding": FINDING,
        "selection": {
            "seed": SEED,
            "minimum_readers": MIN_READERS,
            "minimum_median_pairwise_iou": MIN_MEDIAN_IOU,
            "eligible_images": len(image_rows),
        },
        "prompt_counts": {
            "total": len(prompt_rows),
            "by_question_contract": {
                contract: sum(row["question_contract"] == contract for row in prompt_rows)
                for contract in ("structured_neutral", "clinical_direct")
            },
        },
        "provenance": {
            "train_csv": str(csv_path.resolve()),
            "train_csv_sha256": sha256_file(csv_path),
            "images_sha256": hashlib.sha256(images_text.encode()).hexdigest(),
            "prompts_sha256": hashlib.sha256(prompts_text.encode()).hexdigest(),
            "code_sha256": sha256_file(Path(__file__)),
        },
        "construct_locks": {
            "dicom_spacing_is_patient_truth": False,
            "reader_bbox_is_clinical_caliper_truth": False,
            "certified_conditions_are_explicit_hypothetical_interventions": True,
            "absolute_patient_mm_accuracy_claim_authorized": False,
            "model_outputs_opened": False,
        },
    }
    atomic_identical(output / "image_manifest.jsonl", images_text.encode())
    atomic_identical(output / "prompt_manifest.jsonl", prompts_text.encode())
    atomic_identical(output / "manifest_audit.json", (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode())
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.data_root, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
