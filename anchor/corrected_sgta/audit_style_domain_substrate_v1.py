#!/usr/bin/env python3
"""Outcome-blind audit of natural acquisition/style variables in VinDr-CXR.

The audit reads DICOM headers only.  It deliberately does not infer a scanner,
hospital, or source domain from raster dimensions or intensity statistics.
Those are recorded as image/export properties, not provenance labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pydicom


PROTOCOL = "style-domain-substrate-audit-v1"
TAGS = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "AcquisitionNumber",
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "InstitutionName",
    "StationName",
    "DetectorID",
    "SoftwareVersions",
    "ViewPosition",
    "PatientPosition",
    "PhotometricInterpretation",
    "WindowCenter",
    "WindowWidth",
    "VOILUTFunction",
    "PresentationLUTShape",
    "PixelIntensityRelationship",
    "PixelIntensityRelationshipSign",
    "ImageType",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "PixelSpacing",
    "ImagerPixelSpacing",
    "RescaleSlope",
    "RescaleIntercept",
    "PixelPaddingValue",
    "PixelPaddingRangeLimit",
)


def normalized(value: Any) -> str:
    if value is None:
        return "<MISSING>"
    if isinstance(value, (list, tuple)):
        return "\\".join(str(part) for part in value)
    # pydicom MultiValue is iterable but deliberately avoid expanding strings.
    if not isinstance(value, (str, bytes)) and hasattr(value, "__iter__"):
        try:
            return "\\".join(str(part) for part in value)
        except TypeError:
            pass
    text = str(value).strip()
    return text if text else "<EMPTY>"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def top(counter: Counter[str], n: int = 20) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-group-size", type=int, default=100)
    args = parser.parse_args()

    paths = sorted(args.dicom_root.glob("*.dicom"))
    if not paths:
        raise FileNotFoundError(f"no DICOMs under {args.dicom_root}")

    counts = {tag: Counter() for tag in TAGS}
    exact_signatures: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            row = {tag: normalized(getattr(dataset, tag, None)) for tag in TAGS}
            for tag, value in row.items():
                counts[tag][value] += 1
            signature = json.dumps(row, sort_keys=True, separators=(",", ":"))
            exact_signatures[hashlib.sha256(signature.encode()).hexdigest()] += 1
        except Exception as exc:  # pragma: no cover - outcome is audited
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    annotation_images: set[str] = set()
    rad_ids: Counter[str] = Counter()
    class_ids: Counter[str] = Counter()
    with args.annotations.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            annotation_images.add(str(row.get("image_id", "")))
            rad_ids[str(row.get("rad_id", "<MISSING>"))] += 1
            class_ids[str(row.get("class_id", "<MISSING>"))] += 1

    source_tags = (
        "Manufacturer",
        "ManufacturerModelName",
        "InstitutionName",
        "StationName",
        "DetectorID",
        "SoftwareVersions",
    )
    protocol_tags = (
        "ViewPosition",
        "PatientPosition",
        "VOILUTFunction",
        "PresentationLUTShape",
        "PixelIntensityRelationship",
        "PixelIntensityRelationshipSign",
    )

    def eligible_nonmissing(tag: str) -> list[dict[str, Any]]:
        return [
            {"value": value, "count": count}
            for value, count in counts[tag].most_common()
            if value not in {"<MISSING>", "<EMPTY>"} and count >= args.minimum_group_size
        ]

    payload = {
        "protocol_id": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dicom_root": str(args.dicom_root),
            "annotations": str(args.annotations),
            "annotations_sha256": sha256(args.annotations),
        },
        "counts": {
            "dicom_paths": len(paths),
            "read_successes": len(paths) - len(failures),
            "read_failures": len(failures),
            "annotation_unique_images": len(annotation_images),
            "annotation_rad_ids": top(rad_ids),
            "annotation_class_ids": top(class_ids),
            "exact_header_signatures": len(exact_signatures),
            "largest_exact_header_signature_groups": top(exact_signatures),
        },
        "tag_counts": {
            tag: {
                "n_distinct_including_missing": len(counts[tag]),
                "n_missing": counts[tag]["<MISSING>"],
                "top": top(counts[tag]),
            }
            for tag in TAGS
        },
        "natural_source_groups_ge_minimum": {
            tag: eligible_nonmissing(tag) for tag in source_tags
        },
        "natural_protocol_groups_ge_minimum": {
            tag: eligible_nonmissing(tag) for tag in protocol_tags
        },
        "construct_rules": {
            "source_provenance_requires": "at least two nonmissing source-tag groups; raster/export signatures do not qualify",
            "paired_counterfactual_requires": "same acquisition/pathology rendered under two independently admitted source/protocol states",
            "pixel_shape_is_not_source": True,
            "window_or_photometric_difference_is_not_clinical_equivalence": True,
        },
        "failures": failures[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
