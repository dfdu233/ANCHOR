#!/usr/bin/env python3
"""Build a fail-closed VinDr independent-reader spatial-extent screen.

This script deliberately derives only image-coordinate attributes from R8/R9/R10
bounding boxes.  It does not infer patient left/right, lobes, diagnoses, or a
clinical ontology.  Mapping ``single_image_hemifield`` to "unilateral" (or an
image-height label to a lung-zone label) requires a separately frozen clinical
admission audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from corrected_sgta.clinical_claims import normalize_term


VERSION = "vindr-spatial-reader-extent-screen-v1"
PANEL = ("R8", "R9", "R10")
PRIMARY_FINDINGS = (
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pulmonary_fibrosis",
)
LOCAL_FINDINGS = (
    "aortic_enlargement",
    "atelectasis",
    "calcification",
    "cardiomegaly",
    "consolidation",
    "ild",
    "infiltration",
    "lung_opacity",
    "nodule_mass",
    "other_lesion",
    "pleural_effusion",
    "pleural_thickening",
    "pneumothorax",
    "pulmonary_fibrosis",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def split_image(image_id: str, seed: int = 42) -> str:
    """Assign an image globally to pilot/dev/test using 20/20/60 intervals."""

    key = f"{seed}|spatial-reader-split|{image_id}".encode()
    value = int(hashlib.sha256(key).hexdigest()[:16], 16) / float(16**16)
    if value < 0.2:
        return "pilot"
    if value < 0.4:
        return "dev"
    return "test"


def _normalized_boxes(
    boxes: Sequence[Mapping[str, object]], width: float, height: float
) -> list[dict[str, float]]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    normalized = []
    for box in boxes:
        values = {name: float(box[name]) for name in ("x_min", "y_min", "x_max", "y_max")}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("non-finite bbox coordinate")
        if not (
            0 <= values["x_min"] < values["x_max"] <= width
            and 0 <= values["y_min"] < values["y_max"] <= height
        ):
            raise ValueError("bbox lies outside DICOM dimensions")
        normalized.append(
            {
                "x_min": values["x_min"] / width,
                "y_min": values["y_min"] / height,
                "x_max": values["x_max"] / width,
                "y_max": values["y_max"] / height,
            }
        )
    if not normalized:
        raise ValueError("at least one bbox is required")
    return normalized


def classify_horizontal_extent(
    boxes: Sequence[Mapping[str, object]], width: float, height: float
) -> str:
    """Classify one reader's boxes without assigning anatomical left/right.

    A five-percent central guard makes near-midline boxes explicitly ambiguous.
    A single wide box or separated lateral boxes can establish both-hemifield
    coverage; otherwise all boxes must remain on one guarded side.
    """

    values = _normalized_boxes(boxes, width, height)
    centers = [0.5 * (box["x_min"] + box["x_max"]) for box in values]
    has_left = any(center <= 0.45 for center in centers)
    has_right = any(center >= 0.55 for center in centers)
    wide = any(box["x_min"] <= 0.35 and box["x_max"] >= 0.65 for box in values)
    if wide or (has_left and has_right):
        return "both_image_hemifields"
    if all(box["x_max"] <= 0.55 for box in values) and has_left:
        return "single_image_hemifield"
    if all(box["x_min"] >= 0.45 for box in values) and has_right:
        return "single_image_hemifield"
    return "ambiguous_central_or_crossing"


def classify_vertical_extent(
    boxes: Sequence[Mapping[str, object]], width: float, height: float
) -> str:
    """Classify image-height extent; never call this a lung-zone or lobe label."""

    values = _normalized_boxes(boxes, width, height)
    centers = [0.5 * (box["y_min"] + box["y_max"]) for box in values]
    has_upper = any(center <= 0.42 for center in centers)
    has_lower = any(center >= 0.58 for center in centers)
    tall = any(box["y_min"] <= 0.30 and box["y_max"] >= 0.70 for box in values)
    if tall or (has_upper and has_lower):
        return "multiple_image_height_zones"
    if all(box["y_max"] <= 0.60 for box in values) and has_upper:
        return "upper_image_region"
    if all(box["y_min"] >= 0.40 for box in values) and has_lower:
        return "lower_image_region"
    return "ambiguous_mid_or_crossing"


def read_label_votes(path: Path) -> dict[tuple[str, str], set[str]]:
    votes: dict[tuple[str, str], set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("empty label CSV")
        reader_column = "rad_id" if "rad_id" in reader.fieldnames else "rad_ID"
        if not {"image_id", reader_column} <= set(reader.fieldnames):
            raise ValueError("label CSV lacks image_id/rad_id")
        finding_columns = [
            name for name in reader.fieldnames if name not in {"image_id", reader_column}
        ]
        for row in reader:
            rad_id = str(row[reader_column])
            if rad_id not in PANEL:
                continue
            for source_name in finding_columns:
                raw = str(row.get(source_name, "")).strip()
                if raw and float(raw) > 0:
                    votes[(str(row["image_id"]), normalize_term(source_name))].add(rad_id)
    return votes


def read_bboxes(
    path: Path,
) -> dict[tuple[str, str], dict[str, list[dict[str, float]]]]:
    output: dict[tuple[str, str], dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "class_name", "rad_id", "x_min", "y_min", "x_max", "y_max"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"bbox CSV missing {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            rad_id = str(row["rad_id"])
            raw = [str(row[name]).strip() for name in ("x_min", "y_min", "x_max", "y_max")]
            if rad_id not in PANEL or not all(raw):
                continue
            finding = normalize_term(str(row["class_name"]))
            if finding not in LOCAL_FINDINGS:
                continue
            output[(str(row["image_id"]), finding)][rad_id].append(
                dict(zip(("x_min", "y_min", "x_max", "y_max"), map(float, raw)))
            )
    return output


def read_dicom_header(path: Path) -> dict[str, object]:
    try:
        import pydicom
    except ImportError as error:
        raise RuntimeError(
            "pydicom is required for the real audit; use the established Huatuo/Hulu runtime"
        ) from error
    tags = [
        "Rows",
        "Columns",
        "PatientID",
        "StudyInstanceUID",
        "ViewPosition",
        "PatientOrientation",
        "ImageLaterality",
        "Laterality",
        "ImageOrientationPatient",
    ]
    dataset = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=tags)

    def value(name: str) -> str | None:
        raw = getattr(dataset, name, None)
        text = "" if raw is None else str(raw).strip()
        return text or None

    return {
        "rows": int(dataset.Rows),
        "columns": int(dataset.Columns),
        "patient_id": value("PatientID"),
        "study_uid": value("StudyInstanceUID"),
        "view_position": value("ViewPosition"),
        "patient_orientation": value("PatientOrientation"),
        "image_laterality": value("ImageLaterality"),
        "laterality": value("Laterality"),
        "image_orientation_patient": value("ImageOrientationPatient"),
    }


def _consensus(states: Sequence[str], ambiguous_prefix: str) -> dict[str, object]:
    definite = [state for state in states if not state.startswith(ambiguous_prefix)]
    counts = Counter(definite)
    winner, winner_count = (counts.most_common(1)[0] if counts else (None, 0))
    return {
        "reader_states": list(states),
        "definite_reader_count": len(definite),
        "support_counts": dict(sorted(counts.items())),
        "majority_value": winner if winner_count >= 2 else None,
        "unanimous_value": winner if winner_count == 3 else None,
    }


def build_records(
    label_votes: Mapping[tuple[str, str], set[str]],
    bboxes: Mapping[tuple[str, str], Mapping[str, Sequence[Mapping[str, object]]]],
    headers: Mapping[str, Mapping[str, object]],
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    invalid_coordinates: list[str] = []
    bbox_label_mismatches: list[str] = []
    eligible_keys = sorted(
        key for key, readers in label_votes.items()
        if key[1] in LOCAL_FINDINGS and readers == set(PANEL)
    )
    for image_id, finding in eligible_keys:
        by_reader = bboxes.get((image_id, finding), {})
        if set(by_reader) != set(PANEL):
            bbox_label_mismatches.append(f"{image_id}:{finding}")
            continue
        header = headers[image_id]
        width, height = float(header["columns"]), float(header["rows"])
        horizontal: list[str] = []
        vertical: list[str] = []
        try:
            for reader in PANEL:
                horizontal.append(classify_horizontal_extent(by_reader[reader], width, height))
                vertical.append(classify_vertical_extent(by_reader[reader], width, height))
        except ValueError:
            invalid_coordinates.append(f"{image_id}:{finding}")
            continue
        rows.append(
            {
                "dataset": "vindr-cxr-1.0.0",
                "record_id": hashlib.sha256(f"{image_id}|{finding}".encode()).hexdigest()[:20],
                "image_id": image_id,
                "dicom_relpath": f"train/{image_id}.dicom",
                "finding": finding,
                "reader_panel": list(PANEL),
                "parent_finding_support": {"positive_votes": 3, "reader_count": 3},
                "experiment_split": split_image(image_id, seed),
                "split_unit": "image_id; patient grouping unavailable in released identifiers",
                "horizontal_extent": _consensus(horizontal, "ambiguous_"),
                "vertical_extent": _consensus(vertical, "ambiguous_"),
                "reader_box_counts": {reader: len(by_reader[reader]) for reader in PANEL},
                "coordinate_semantics": {
                    "horizontal": "image hemifield extent; no anatomical side",
                    "vertical": "normalized image-height region; no lung zone/lobe",
                },
            }
        )
    audit = {
        "eligible_three_reader_positive_claims": len(eligible_keys),
        "manifest_rows": len(rows),
        "bbox_label_mismatches": len(bbox_label_mismatches),
        "bbox_label_mismatch_examples": bbox_label_mismatches[:10],
        "invalid_coordinate_claims": len(invalid_coordinates),
        "invalid_coordinate_examples": invalid_coordinates[:10],
    }
    return rows, audit


def summarize(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    records = list(rows)
    result: dict[str, object] = {}
    for finding in sorted({str(row["finding"]) for row in records}):
        subset = [row for row in records if row["finding"] == finding]
        item: dict[str, object] = {"three_reader_parent_positive": len(subset)}
        for family in ("horizontal_extent", "vertical_extent"):
            fully_definite = [row for row in subset if row[family]["definite_reader_count"] == 3]
            unanimous = [row for row in fully_definite if row[family]["unanimous_value"]]
            labels = Counter(str(row[family]["unanimous_value"]) for row in unanimous)
            by_split = {
                split: dict(
                    sorted(
                        Counter(
                            str(row[family]["unanimous_value"])
                            for row in unanimous
                            if row["experiment_split"] == split
                        ).items()
                    )
                )
                for split in ("pilot", "dev", "test")
            }
            reader_prevalence = {}
            representation_crosstab: dict[str, Counter[str]] = defaultdict(Counter)
            for index, reader in enumerate(PANEL):
                states = [str(row[family]["reader_states"][index]) for row in subset]
                reader_prevalence[reader] = dict(sorted(Counter(states).items()))
                for row, state in zip(subset, states):
                    count = int(row["reader_box_counts"][reader])
                    representation_crosstab[state]["one_box" if count == 1 else "multiple_boxes"] += 1
            item[family] = {
                "fully_definite": len(fully_definite),
                "fully_definite_fraction": len(fully_definite) / len(subset) if subset else 0.0,
                "unanimous": len(unanimous),
                "unanimity_given_fully_definite": len(unanimous) / len(fully_definite) if fully_definite else 0.0,
                "unanimous_values": dict(sorted(labels.items())),
                "unanimous_values_by_split": by_split,
                "per_reader_state_counts": reader_prevalence,
                "reader_box_count_by_state": {
                    state: dict(sorted(counts.items()))
                    for state, counts in sorted(representation_crosstab.items())
                },
            }
        result[finding] = item
    return result


def data_progression_gate(per_finding: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Power/consistency gate only; never authorizes clinical semantics."""

    labels = ("single_image_hemifield", "both_image_hemifields")
    decisions = {}
    for finding in PRIMARY_FINDINGS:
        item = per_finding.get(finding)
        reasons = []
        if item is None:
            reasons.append("finding_absent")
        else:
            horizontal = item["horizontal_extent"]
            if horizontal["fully_definite_fraction"] < 0.75:
                reasons.append("fully_definite_fraction_below_0.75")
            if horizontal["unanimity_given_fully_definite"] < 0.65:
                reasons.append("unanimity_below_0.65")
            minima = {"pilot": 10, "dev": 10, "test": 20}
            for split, minimum in minima.items():
                counts = horizontal["unanimous_values_by_split"][split]
                for label in labels:
                    if int(counts.get(label, 0)) < minimum:
                        reasons.append(f"{split}:{label}_below_{minimum}")
        decisions[finding] = {"pass": not reasons, "reasons": reasons}
    passing = [finding for finding, value in decisions.items() if value["pass"]]
    return {
        "definition": (
            "horizontal coordinate-screen only: definite>=0.75, unanimity>=0.65, "
            "each class >=10 pilot, >=10 dev, >=20 test"
        ),
        "per_finding": decisions,
        "passing_primary_findings": passing,
        "screen_pass": len(passing) >= 2,
        "clinical_semantics_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--bbox-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = read_label_votes(args.labels_csv)
    boxes = read_bboxes(args.bbox_csv)
    image_ids = sorted(
        image_id for image_id, finding in labels
        if finding in LOCAL_FINDINGS and labels[(image_id, finding)] == set(PANEL)
    )
    unique_image_ids = sorted(set(image_ids))
    headers = {}
    missing = []
    for image_id in unique_image_ids:
        path = args.image_root / f"{image_id}.dicom"
        if not path.is_file():
            missing.append(image_id)
            continue
        headers[image_id] = read_dicom_header(path)
    if missing:
        raise FileNotFoundError(f"missing DICOMs: {missing[:10]}")

    records, coordinate_audit = build_records(labels, boxes, headers, args.seed)
    per_finding = summarize(records)
    orientation_fields = (
        "view_position",
        "patient_orientation",
        "image_laterality",
        "laterality",
        "image_orientation_patient",
    )
    metadata_presence = {
        field: sum(headers[image_id][field] is not None for image_id in headers)
        for field in orientation_fields
    }
    patient_ids = {str(header["patient_id"]) for header in headers.values() if header["patient_id"]}
    study_uids = {str(header["study_uid"]) for header in headers.values() if header["study_uid"]}
    progression = data_progression_gate(per_finding)
    formal_blockers = [
        "released DICOM/CSV has no auditable patient grouping; split is image-disjoint only",
        "bbox hemifield coverage has not been clinically admitted as unilateral/bilateral",
        "image-height regions have not been admitted as lung zones or lobes",
        "reader box style and multiple-lesion representation remain possible attribute confounds",
    ]
    summary = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "labels_csv": str(args.labels_csv.resolve()),
            "labels_sha256": sha256_file(args.labels_csv),
            "bbox_csv": str(args.bbox_csv.resolve()),
            "bbox_sha256": sha256_file(args.bbox_csv),
            "image_root": str(args.image_root.resolve()),
        },
        "reader_panel": list(PANEL),
        "primary_findings_frozen_before_audit": list(PRIMARY_FINDINGS),
        "split_contract": {
            "assignment": "global image SHA256 20/20/60",
            "counts": dict(sorted(Counter(str(row["experiment_split"]) for row in records).items())),
            "image_disjoint": True,
            "patient_disjoint": False,
            "patient_disjoint_reason": "patient/study identifiers absent after release de-identification",
        },
        "dicom_metadata_audit": {
            "unique_images": len(headers),
            "orientation_field_non_null_counts": metadata_presence,
            "non_null_patient_ids": len(patient_ids),
            "non_null_study_uids": len(study_uids),
            "source_view_contract": "official VinDr-CXR documentation states PA-only; local DICOM tags are not used to infer view",
        },
        "coordinate_audit": coordinate_audit,
        "per_finding": per_finding,
        "data_progression_gate": progression,
        "formal_authorized": False,
        "formal_blockers": formal_blockers,
        "claim_ceiling": (
            "independent-reader agreement on bounding-box image-coordinate extent; "
            "not clinical laterality, lung-zone truth, patient-generalization, or VLM hallucination"
        ),
        "command": " ".join(sys.argv),
    }
    manifest_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in sorted(records, key=lambda row: (str(row["experiment_split"]), str(row["finding"]), str(row["image_id"])))
    )
    atomic_text(args.output_dir / "spatial_reader_manifest_v1.jsonl", manifest_payload)
    summary["manifest_sha256"] = hashlib.sha256(manifest_payload.encode()).hexdigest()
    fingerprint_source = {key: value for key, value in summary.items() if key not in {"created_at", "command"}}
    summary["fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_source, sort_keys=True).encode()
    ).hexdigest()
    atomic_text(args.output_dir / "summary_v1.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest_rows": len(records),
        "data_screen_pass": progression["screen_pass"],
        "passing_findings": progression["passing_primary_findings"],
        "formal_authorized": False,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
