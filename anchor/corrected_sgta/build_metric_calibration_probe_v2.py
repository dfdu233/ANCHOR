#!/usr/bin/env python3
"""Construct-corrected v2 prompts over the immutable v1 image substrate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from anchor.corrected_sgta.build_metric_calibration_probe_v1 import (
    CONDITIONS,
    atomic_identical,
    canonical,
    direct_prompt,
    metadata_line,
    nominal_distance_mm,
    sha256_file,
)


VERSION = "metric-calibration-probe-manifest-v2"
DEFAULT_V1 = Path("corrected_runs/metric_calibration_probe_v1")
DEFAULT_OUTPUT = Path("corrected_runs/metric_calibration_probe_v2")


def structured_prompt_v2(
    metadata: str,
    endpoints: Sequence[Sequence[float]] | None,
    *,
    rows: int,
    columns: int,
) -> str:
    if endpoints is not None:
        endpoint_instruction = (
            "An independent reader panel has already verified that the nodule/mass is visible. "
            f"Use its frozen normalized box-extent endpoints (x,y): {canonical(endpoints)}. "
            "Do not relocalize or override them."
        )
    else:
        endpoint_instruction = (
            "Locate the single pulmonary nodule/mass and estimate normalized endpoints (x,y) "
            "of its longest apparent reader-box extent."
        )
    return f"""Analyze the supplied chest radiograph for one pulmonary nodule/mass.
The source raster has exactly {columns} columns and {rows} rows before model preprocessing.
{metadata}
{endpoint_instruction}
Return one JSON object only with exactly these fields:
{{"visible":"yes|no|uncertain","endpoints_normalized":[[x1,y1],[x2,y2]] or null,"measurement_type":"patient-mm|detector-mm|pixel-only/unknown","physical_value":number or null,"unit":"mm|cm" or null}}
Emit physical_value and unit only when patient-plane calibration is certified. A detector-plane pitch may be typed as detector-mm but is not patient size and must have null physical_value/unit."""


def expected_value(
    condition: str,
    endpoints: Sequence[Sequence[float]],
    rows: int,
    columns: int,
    spacing: Sequence[float],
) -> tuple[float | None, str | None]:
    base_mm = nominal_distance_mm(endpoints, rows, columns, spacing)
    if condition == "certified_x0p5":
        return base_mm * 0.5, "mm"
    if condition == "certified_x1":
        return base_mm, "mm"
    if condition == "certified_x2":
        return base_mm * 2.0, "mm"
    if condition == "certified_cm":
        return base_mm / 10.0, "cm"
    return None, None


def build(v1_dir: Path, output: Path) -> dict[str, Any]:
    v1_images_path = v1_dir / "image_manifest.jsonl"
    v1_audit_path = v1_dir / "manifest_audit.json"
    images = [json.loads(line) for line in v1_images_path.read_text().splitlines() if line.strip()]
    prompt_rows = []
    for image in images:
        rows, columns = int(image["rows"]), int(image["columns"])
        endpoints = image["oracle_endpoints_normalized"]
        spacing = image["base_spacing"]
        for arm, arm_endpoints in (("oracle_coordinate", endpoints), ("vision_coordinate", None)):
            for condition in CONDITIONS:
                metadata, expected_type, _, _ = metadata_line(condition, spacing)
                value, unit = expected_value(condition, endpoints, rows, columns, spacing)
                prompt_rows.append(
                    {
                        "version": VERSION,
                        "item_id": f"{image['image_id']}:{arm}:{condition}:structured-v2",
                        "image_id": image["image_id"],
                        "image_path": image["rendered_image"],
                        "arm": arm,
                        "condition": condition,
                        "question_contract": "structured_neutral_v2",
                        "prompt": structured_prompt_v2(
                            metadata, arm_endpoints, rows=rows, columns=columns
                        ),
                        "expected_measurement_type": expected_type,
                        "expected_physical_value": value,
                        "expected_unit": unit,
                        "patient_value_identifiable": value is not None,
                        "oracle_visible": True if arm == "oracle_coordinate" else None,
                        "model_outcome_opened": False,
                    }
                )
        for condition in ("missing", "detector_only", "header_unknown", "certified_x1"):
            metadata, expected_type, _, _ = metadata_line(condition, spacing)
            value, unit = expected_value(condition, endpoints, rows, columns, spacing)
            prompt_rows.append(
                {
                    "version": VERSION,
                    "item_id": f"{image['image_id']}:vision_coordinate:{condition}:direct-v2",
                    "image_id": image["image_id"],
                    "image_path": image["rendered_image"],
                    "arm": "vision_coordinate",
                    "condition": condition,
                    "question_contract": "clinical_direct_v2",
                    "prompt": direct_prompt(metadata),
                    "expected_measurement_type": expected_type,
                    "expected_physical_value": value,
                    "expected_unit": unit,
                    "patient_value_identifiable": value is not None,
                    "oracle_visible": None,
                    "model_outcome_opened": False,
                }
            )
    images_text = "".join(canonical({**row, "prompt_manifest_version": VERSION}) + "\n" for row in images)
    prompts_text = "".join(canonical(row) + "\n" for row in prompt_rows)
    audit = {
        "version": VERSION,
        "decision": "CONSTRUCT_CORRECTED_MANIFEST_READY",
        "images": len(images),
        "prompts": len(prompt_rows),
        "supersedes_prompt_manifest": str((v1_dir / "prompt_manifest.jsonl").resolve()),
        "v1_smoke_status": "construct_invalid_missing_raster_dimensions_and_conflated_detector_value_metric",
        "provenance": {
            "v1_image_manifest": str(v1_images_path.resolve()),
            "v1_image_manifest_sha256": sha256_file(v1_images_path),
            "v1_audit_sha256": sha256_file(v1_audit_path),
            "images_sha256": hashlib.sha256(images_text.encode()).hexdigest(),
            "prompts_sha256": hashlib.sha256(prompts_text.encode()).hexdigest(),
            "code_sha256": sha256_file(Path(__file__)),
        },
        "construct_corrections": {
            "source_raster_rows_columns_exposed": True,
            "oracle_visibility_and_endpoints_frozen": True,
            "centimetre_expected_value_reexpression_fixed": True,
            "detector_typed_value_separated_from_patient_overcommitment": True,
        },
        "absolute_patient_mm_accuracy_claim_authorized": False,
        "gpu_authorized": False,
    }
    atomic_identical(output / "image_manifest.jsonl", images_text.encode())
    atomic_identical(output / "prompt_manifest.jsonl", prompts_text.encode())
    atomic_identical(output / "manifest_audit.json", (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode())
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.v1_dir, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
