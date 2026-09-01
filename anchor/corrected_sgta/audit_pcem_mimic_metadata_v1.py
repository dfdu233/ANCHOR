#!/usr/bin/env python3
"""CPU-only admission audit for projection-conditioned evidence misbinding.

This script counts MIMIC-CXR AP/PA substrate without images.  Echo-qualified
counts are emitted only when a real MIMIC-IV-ECHO structured measurement file
is supplied; absent/restricted files yield null, never a fabricated zero.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROTOCOL = "pcem-mimic-metadata-substrate-v1"
WINDOW_HOURS = (6, 24, 72, 168, 720)


def open_text(path: Path):
    return gzip.open(path, "rt", newline="", encoding="utf-8") if path.suffix == ".gz" else path.open("r", newline="", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_cxr_datetime(date: str, time: str) -> datetime | None:
    digits = "".join(ch for ch in str(date) if ch.isdigit())
    if len(digits) != 8:
        return None
    time_text = str(time).strip()
    if not time_text:
        time_text = "000000"
    main, dot, fractional = time_text.partition(".")
    # DICOM TM values often lose their leading zero when exported to CSV
    # (e.g. 08:05:56 becomes ``80556.875``), so pad on the left.
    main = "".join(ch for ch in main if ch.isdigit()).zfill(6)[-6:]
    frac = ("".join(ch for ch in fractional if ch.isdigit()) + "000000")[:6] if dot else "000000"
    try:
        return datetime.strptime(digits + main + frac, "%Y%m%d%H%M%S%f")
    except ValueError:
        return None


def nearest_pairs(ap: list[tuple[datetime, str, str]], pa: list[tuple[datetime, str, str]]) -> list[tuple[float, str, str, str, str]]:
    pa_sorted = sorted(pa)
    pa_times = [row[0] for row in pa_sorted]
    pairs: set[tuple[str, str]] = set()
    out: list[tuple[float, str, str, str, str]] = []
    for ap_time, ap_dicom, ap_study in sorted(ap):
        index = bisect.bisect_left(pa_times, ap_time)
        candidates = []
        if index < len(pa_sorted):
            candidates.append(pa_sorted[index])
        if index:
            candidates.append(pa_sorted[index - 1])
        if not candidates:
            continue
        pa_time, pa_dicom, pa_study = min(candidates, key=lambda row: abs((row[0] - ap_time).total_seconds()))
        key = (ap_dicom, pa_dicom)
        if key in pairs:
            continue
        pairs.add(key)
        out.append((abs((pa_time - ap_time).total_seconds()) / 3600.0, ap_dicom, pa_dicom, ap_study, pa_study))
    return out


def echo_schema(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "access_status": "ACCESS_BLOCKED_OR_NOT_SUPPLIED",
            "rows": None,
            "subjects": None,
            "measurements": None,
            "lvidd_candidates": None,
            "echo_qualified_ap_pa_pairs": None,
            "borderline_echo_truth_pairs": None,
        }
    rows = 0
    subjects: set[str] = set()
    measurements: Counter[str] = Counter()
    descriptions: Counter[str] = Counter()
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "measurement_datetime", "measurement", "measurement_description", "result", "unit"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"echo schema missing fields: {missing}")
        for row in reader:
            rows += 1
            subjects.add(str(row["subject_id"]))
            measurements[str(row["measurement"]).strip().lower()] += 1
            descriptions[str(row["measurement_description"]).strip().lower()] += 1
    lvidd = [
        {"name": name, "count": count}
        for name, count in (measurements + descriptions).most_common()
        if "lvid" in name or "left ventricular internal" in name or "left ventricle diameter" in name
    ]
    return {
        "access_status": "AVAILABLE_SCHEMA_ONLY",
        "sha256": sha256(path),
        "rows": rows,
        "subjects": len(subjects),
        "measurements": len(measurements),
        "lvidd_candidates": lvidd,
        "echo_qualified_ap_pa_pairs": "NOT_COMPUTED_BY_SCHEMA_AUDIT",
        "borderline_echo_truth_pairs": "NOT_COMPUTED_BY_SCHEMA_AUDIT",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxr-metadata", type=Path, required=True)
    parser.add_argument("--cxr-split", type=Path, required=True)
    parser.add_argument("--echo-structured", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    splits: dict[str, str] = {}
    split_counts = Counter()
    with open_text(args.cxr_split) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dicom = str(row["dicom_id"])
            split = str(row["split"])
            splits[dicom] = split
            split_counts[split] += 1

    view_counts = Counter()
    view_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    subjects: set[str] = set()
    subjects_by_view: dict[str, set[str]] = defaultdict(set)
    studies_by_view: dict[str, set[str]] = defaultdict(set)
    by_subject: dict[str, dict[str, list[tuple[datetime, str, str]]]] = defaultdict(lambda: defaultdict(list))
    rows = 0
    bad_datetime = 0
    with open_text(args.cxr_metadata) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            subject = str(row["subject_id"])
            study = str(row["study_id"])
            dicom = str(row["dicom_id"])
            view = str(row.get("ViewPosition", "")).strip().upper() or "<MISSING>"
            split = splits.get(dicom, "<UNMATCHED>")
            view_counts[view] += 1
            view_by_split[split][view] += 1
            subjects.add(subject)
            subjects_by_view[view].add(subject)
            studies_by_view[view].add(study)
            if view in {"AP", "PA"}:
                timestamp = parse_cxr_datetime(row.get("StudyDate", ""), row.get("StudyTime", ""))
                if timestamp is None:
                    bad_datetime += 1
                else:
                    by_subject[subject][view].append((timestamp, dicom, study))

    both_subjects = sorted(subjects_by_view["AP"].intersection(subjects_by_view["PA"]))
    pairs: list[tuple[str, float, str, str, str, str]] = []
    for subject in both_subjects:
        for distance, ap_dicom, pa_dicom, ap_study, pa_study in nearest_pairs(by_subject[subject]["AP"], by_subject[subject]["PA"]):
            pairs.append((subject, distance, ap_dicom, pa_dicom, ap_study, pa_study))

    same_study = [pair for pair in pairs if pair[4] == pair[5]]
    window_counts = {
        str(hours): {
            "nearest_pairs": sum(pair[1] <= hours for pair in pairs),
            "unique_subjects": len({pair[0] for pair in pairs if pair[1] <= hours}),
            "same_study_pairs": sum(pair[1] <= hours and pair[4] == pair[5] for pair in pairs),
        }
        for hours in WINDOW_HOURS
    }

    payload = {
        "protocol_id": PROTOCOL,
        "official_construct_audit": {
            "chexchonet_v1": {
                "institution": "Columbia University Irving Medical Center",
                "cxr_images": 71589,
                "patients": 24689,
                "projection_selection": "PA_ONLY",
                "portable_ap_excluded_by_design": True,
                "echo_link_window_days": 365,
                "continuous_measurements": ["IVSd", "LVIDd", "LVPWd"],
                "binary_labels": ["SLVH", "DLV", "SLVH_OR_DLV"],
                "defines_radiographic_cardiomegaly": False,
                "mimic_subject_id_linkage": False,
                "pcem_projection_contrast_eligible": False,
            },
            "mimic_iv_echo_v1": {
                "institution": "Beth Israel Deaconess Medical Center",
                "structured_studies_official": 206488,
                "patients_official": 91372,
                "tte_studies_official": 179928,
                "stress_studies_official": 16389,
                "tee_studies_official": 10171,
                "subject_id_consistent_with_mimic_iv": True,
                "measurement_datetime_available": True,
                "clinician_verified_structured_measurements": True,
                "local_access_probe": "HTTP_403_AFTER_BASIC_AUTH",
                "local_bytes_downloaded": 0,
            },
        },
        "inputs": {
            "cxr_metadata": str(args.cxr_metadata),
            "cxr_metadata_sha256": sha256(args.cxr_metadata),
            "cxr_split": str(args.cxr_split),
            "cxr_split_sha256": sha256(args.cxr_split),
            "echo_structured": str(args.echo_structured) if args.echo_structured else None,
        },
        "cxr": {
            "rows": rows,
            "subjects": len(subjects),
            "split_counts": dict(split_counts),
            "view_counts": dict(view_counts),
            "view_by_split": {split: dict(counts) for split, counts in view_by_split.items()},
            "subjects_by_view": {view: len(values) for view, values in subjects_by_view.items()},
            "studies_by_view": {view: len(values) for view, values in studies_by_view.items()},
            "ap_pa_subject_intersection": len(both_subjects),
            "ap_pa_nearest_pairs": len(pairs),
            "ap_pa_same_study_nearest_pairs": len(same_study),
            "ap_pa_window_counts_hours": window_counts,
            "ap_pa_bad_datetime_rows": bad_datetime,
        },
        "echo": echo_schema(args.echo_structured),
        "admission": {
            "metadata_projection_substrate_identified": bool(both_subjects),
            "echo_join_identified": args.echo_structured is not None and args.echo_structured.exists(),
            "independent_heart_size_truth_identified": False,
            "borderline_truth_identified": False,
            "gpu_authorized": False,
            "decision": "ACCESS_BLOCKED_UNIDENTIFIED" if args.echo_structured is None or not args.echo_structured.exists() else "REQUIRES_EXPLICIT_ECHO_CONSTRUCT_AUDIT",
        },
        "construct_warning": "LVIDd measures LV cavity diameter; SLVH/DLV/composite are structural-heart-disease labels, not interchangeable with radiographic cardiomegaly or total cardiac silhouette size.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
