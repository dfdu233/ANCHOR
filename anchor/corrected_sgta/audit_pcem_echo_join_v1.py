#!/usr/bin/env python3
"""Outcome-blind CPU admission audit for the PCEM substrate.

The audit joins MIMIC-CXR AP/PA acquisition episodes to MIMIC-IV-ECHO TTE
studies.  It deliberately does *not* turn echo measurements into cardiomegaly
truth, choose clinical thresholds, download images, or authorize a GPU run.
Only schema, provenance, temporal support, and construct-review feasibility are
reported.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from anchor.corrected_sgta.audit_pcem_mimic_metadata_v1 import (
    open_text,
    parse_cxr_datetime,
    sha256,
)


PROTOCOL_ID = "pcem-echo-temporal-join-audit-v1"
PAIR_WINDOWS_HOURS = (6, 24, 72)
ECHO_WINDOWS_HOURS = (24, 72)
REQUIRED_ECHO_FIELDS = {
    "subject_id",
    "measurement_id",
    "measurement_datetime",
    "test_type",
    "measurement",
    "measurement_description",
    "result",
    "unit",
}
REQUIRED_CXR_FIELDS = {
    "dicom_id",
    "subject_id",
    "study_id",
    "ViewPosition",
    "StudyDate",
    "StudyTime",
}
REQUIRED_SPLIT_FIELDS = {"dicom_id", "subject_id", "study_id", "split"}

# This list creates a review inventory only.  It is not a clinical composite,
# and a keyword hit never defines intrinsic enlargement or cardiomegaly truth.
CHAMBER_SIZE_LEXEMES = (
    "lvid",
    "left ventricular internal",
    "left ventricular diameter",
    "left ventricle diameter",
    "left ventricular end diastolic",
    "left ventricular end-diastolic",
    "lv end diastolic",
    "lv end-diastolic",
    "left atrial volume",
    "left atrium volume",
    "left atrial diameter",
    "la volume",
    "right ventricular diameter",
    "right ventricular basal",
    "right ventricle diameter",
    "rv basal",
    "pericardial effusion",
)


@dataclass(frozen=True)
class EchoStudy:
    subject_id: str
    measurement_id: str
    timestamp: datetime
    test_type: str
    has_size_candidate: bool


@dataclass(frozen=True)
class CxrImage:
    timestamp: datetime
    dicom_id: str
    study_id: str
    split: str


@dataclass(frozen=True)
class ProjectionEpisode:
    subject_id: str
    ap: CxrImage
    pa: CxrImage
    distance_hours: float
    center: datetime

    @property
    def same_study(self) -> bool:
        return self.ap.study_id == self.pa.study_id


def _field_gate(reader: csv.DictReader, required: set[str], label: str) -> None:
    missing = sorted(required.difference(reader.fieldnames or []))
    if missing:
        raise ValueError(f"{label} schema missing fields: {missing}")


def parse_echo_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
    except ValueError:
        pass
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d %H%M%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def normalize_test_type(value: object) -> str:
    text = " ".join(str(value or "").strip().upper().split())
    if text == "TTE" or "TRANSTHORACIC" in text:
        return "TTE"
    if text == "TEE" or "TRANSESOPHAGEAL" in text:
        return "TEE"
    if "STRESS" in text:
        return "STRESS"
    return text or "<MISSING>"


def is_size_inventory_candidate(measurement: object, description: object) -> bool:
    text = " ".join(
        f"{str(measurement or '').strip()} {str(description or '').strip()}"
        .lower()
        .split()
    )
    return any(lexeme in text for lexeme in CHAMBER_SIZE_LEXEMES)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_echo_studies(path: Path) -> tuple[dict[str, list[EchoStudy]], dict[str, Any]]:
    studies: dict[str, tuple[str, datetime, str]] = {}
    size_candidate_ids: set[str] = set()
    measurement_counts: Counter[str] = Counter()
    description_counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    raw_test_counts: Counter[str] = Counter()
    rows = 0
    invalid_datetimes = 0
    empty_results = 0

    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        _field_gate(reader, REQUIRED_ECHO_FIELDS, "echo")
        for row in reader:
            rows += 1
            subject = str(row["subject_id"]).strip()
            measurement_id = str(row["measurement_id"]).strip()
            if not subject or not measurement_id:
                raise ValueError("echo row has empty subject_id or measurement_id")
            timestamp = parse_echo_datetime(row["measurement_datetime"])
            if timestamp is None:
                invalid_datetimes += 1
                continue
            test_type = normalize_test_type(row["test_type"])
            raw_test_counts[str(row["test_type"]).strip() or "<MISSING>"] += 1
            identity = (subject, timestamp, test_type)
            previous = studies.setdefault(measurement_id, identity)
            if previous != identity:
                raise ValueError(
                    "echo measurement_id maps to inconsistent subject/time/test_type: "
                    f"{measurement_id}"
                )
            measurement = str(row["measurement"]).strip()
            description = str(row["measurement_description"]).strip()
            unit = str(row["unit"]).strip() or "<MISSING>"
            measurement_counts[measurement] += 1
            description_counts[description] += 1
            unit_counts[unit] += 1
            if not str(row["result"]).strip():
                empty_results += 1
            if is_size_inventory_candidate(measurement, description):
                size_candidate_ids.add(measurement_id)

    by_subject: dict[str, list[EchoStudy]] = defaultdict(list)
    test_study_counts: Counter[str] = Counter()
    for measurement_id, (subject, timestamp, test_type) in studies.items():
        test_study_counts[test_type] += 1
        by_subject[subject].append(
            EchoStudy(
                subject_id=subject,
                measurement_id=measurement_id,
                timestamp=timestamp,
                test_type=test_type,
                has_size_candidate=measurement_id in size_candidate_ids,
            )
        )
    for subject in by_subject:
        by_subject[subject].sort(key=lambda item: (item.timestamp, item.measurement_id))

    inventory = []
    combined = measurement_counts + description_counts
    for name, count in sorted(combined.items(), key=lambda item: (-item[1], item[0])):
        if is_size_inventory_candidate(name, ""):
            inventory.append({"name": name, "row_count": count})

    parse_rate = (rows - invalid_datetimes) / rows if rows else 0.0
    audit = {
        "rows": rows,
        "sha256": sha256(path),
        "unique_subjects": len(by_subject),
        "unique_studies_with_valid_datetime": len(studies),
        "invalid_datetime_rows": invalid_datetimes,
        "datetime_parse_rate": parse_rate,
        "test_type_study_counts": dict(sorted(test_study_counts.items())),
        "raw_test_type_row_counts": dict(sorted(raw_test_counts.items())),
        "unique_measurement_names": len(measurement_counts),
        "unique_measurement_descriptions": len(description_counts),
        "unit_row_counts": dict(sorted(unit_counts.items())),
        "empty_result_rows": empty_results,
        "size_inventory_candidate_studies": len(size_candidate_ids),
        "size_inventory_lexical_candidates": inventory,
        "size_inventory_is_clinical_truth": False,
    }
    return dict(by_subject), audit


def load_projection_episodes(
    metadata_path: Path, split_path: Path
) -> tuple[list[ProjectionEpisode], dict[str, Any]]:
    split_rows: dict[str, tuple[str, str, str]] = {}
    split_counts: Counter[str] = Counter()
    with open_text(split_path) as handle:
        reader = csv.DictReader(handle)
        _field_gate(reader, REQUIRED_SPLIT_FIELDS, "CXR split")
        for row in reader:
            dicom_id = str(row["dicom_id"]).strip()
            identity = (
                str(row["subject_id"]).strip(),
                str(row["study_id"]).strip(),
                str(row["split"]).strip(),
            )
            if dicom_id in split_rows and split_rows[dicom_id] != identity:
                raise ValueError(f"CXR split has conflicting dicom_id: {dicom_id}")
            split_rows[dicom_id] = identity
            split_counts[identity[2]] += 1

    by_subject: dict[str, dict[str, list[CxrImage]]] = defaultdict(
        lambda: defaultdict(list)
    )
    metadata_rows = 0
    bad_datetime = 0
    split_mismatches = 0
    ap_pa_rows = 0
    with open_text(metadata_path) as handle:
        reader = csv.DictReader(handle)
        _field_gate(reader, REQUIRED_CXR_FIELDS, "CXR metadata")
        for row in reader:
            metadata_rows += 1
            view = str(row["ViewPosition"]).strip().upper()
            if view not in {"AP", "PA"}:
                continue
            ap_pa_rows += 1
            dicom_id = str(row["dicom_id"]).strip()
            subject = str(row["subject_id"]).strip()
            study = str(row["study_id"]).strip()
            split_identity = split_rows.get(dicom_id)
            if split_identity is None or split_identity[:2] != (subject, study):
                split_mismatches += 1
                continue
            timestamp = parse_cxr_datetime(row["StudyDate"], row["StudyTime"])
            if timestamp is None:
                bad_datetime += 1
                continue
            by_subject[subject][view].append(
                CxrImage(timestamp, dicom_id, study, split_identity[2])
            )
    if split_mismatches:
        raise ValueError(f"CXR metadata/split identity mismatches: {split_mismatches}")

    episodes: list[ProjectionEpisode] = []
    for subject in sorted(by_subject):
        ap_rows = sorted(
            by_subject[subject].get("AP", []),
            key=lambda item: (item.timestamp, item.dicom_id),
        )
        pa_rows = sorted(
            by_subject[subject].get("PA", []),
            key=lambda item: (item.timestamp, item.dicom_id),
        )
        if not ap_rows or not pa_rows:
            continue
        pa_times = [item.timestamp for item in pa_rows]
        candidates: list[ProjectionEpisode] = []
        for ap in ap_rows:
            index = bisect.bisect_left(pa_times, ap.timestamp)
            nearby = pa_rows[max(0, index - 1) : min(len(pa_rows), index + 1)]
            for pa in nearby:
                distance = abs((pa.timestamp - ap.timestamp).total_seconds()) / 3600.0
                center = min(ap.timestamp, pa.timestamp) + abs(pa.timestamp - ap.timestamp) / 2
                candidates.append(ProjectionEpisode(subject, ap, pa, distance, center))
        chosen = min(
            candidates,
            key=lambda item: (
                not item.same_study,
                item.distance_hours,
                item.center,
                item.ap.dicom_id,
                item.pa.dicom_id,
            ),
        )
        episodes.append(chosen)

    audit = {
        "metadata_rows": metadata_rows,
        "metadata_sha256": sha256(metadata_path),
        "split_sha256": sha256(split_path),
        "split_counts": dict(sorted(split_counts.items())),
        "ap_pa_rows": ap_pa_rows,
        "bad_ap_pa_datetime_rows": bad_datetime,
        "metadata_split_identity_mismatches": split_mismatches,
        "outcome_blind_one_episode_per_subject": True,
        "episode_selection": "same-study first, then minimum AP/PA time distance",
        "subjects_with_selected_episode": len(episodes),
        "selected_same_study_episodes": sum(item.same_study for item in episodes),
        "selected_episode_counts_by_pair_window_hours": {
            str(hours): sum(item.distance_hours <= hours for item in episodes)
            for hours in PAIR_WINDOWS_HOURS
        },
    }
    return episodes, audit


def nearest_tte(
    timestamp: datetime, studies: Iterable[EchoStudy]
) -> tuple[EchoStudy, float] | None:
    tte = [study for study in studies if study.test_type == "TTE"]
    if not tte:
        return None
    chosen = min(
        tte,
        key=lambda study: (
            abs((study.timestamp - timestamp).total_seconds()),
            study.timestamp,
            study.measurement_id,
        ),
    )
    return chosen, abs((chosen.timestamp - timestamp).total_seconds()) / 3600.0


def temporal_join_counts(
    episodes: Iterable[ProjectionEpisode], echo_by_subject: dict[str, list[EchoStudy]]
) -> dict[str, Any]:
    rows = []
    for episode in episodes:
        nearest = nearest_tte(episode.center, echo_by_subject.get(episode.subject_id, []))
        if nearest is None:
            continue
        study, echo_distance = nearest
        rows.append((episode, study, echo_distance))

    cells: dict[str, Any] = {}
    for pair_hours in PAIR_WINDOWS_HOURS:
        for echo_hours in ECHO_WINDOWS_HOURS:
            eligible = [
                row
                for row in rows
                if row[0].distance_hours <= pair_hours and row[2] <= echo_hours
            ]
            cells[f"pair_le_{pair_hours}h__echo_le_{echo_hours}h"] = {
                "unique_patients": len(eligible),
                "same_study_projection_episodes": sum(row[0].same_study for row in eligible),
                "episodes_with_lexical_size_inventory_candidate": sum(
                    row[1].has_size_candidate for row in eligible
                ),
            }
    return {
        "subjects_with_any_tte_link": len(rows),
        "cells": cells,
        "patient_identifiers_written": False,
    }


def build_audit(
    *, metadata_path: Path, split_path: Path, echo_path: Path
) -> dict[str, Any]:
    echo_by_subject, echo_audit = load_echo_studies(echo_path)
    episodes, cxr_audit = load_projection_episodes(metadata_path, split_path)
    join = temporal_join_counts(episodes, echo_by_subject)
    primary = join["cells"]["pair_le_6h__echo_le_24h"]
    schema_passed = (
        echo_audit["rows"] > 0
        and echo_audit["datetime_parse_rate"] >= 0.999
        and echo_audit["test_type_study_counts"].get("TTE", 0) > 0
    )
    count_floor_passed = primary["unique_patients"] >= 300
    construct_inventory_available = (
        primary["episodes_with_lexical_size_inventory_candidate"] >= 300
    )
    if not schema_passed:
        decision = "SCHEMA_GATE_FAILED"
    elif not count_floor_passed:
        decision = "TEMPORAL_JOIN_COUNT_GATE_FAILED"
    elif not construct_inventory_available:
        decision = "CONSTRUCT_INVENTORY_COVERAGE_GATE_FAILED"
    else:
        decision = "DATA_GATE_COUNTS_AVAILABLE_CONSTRUCT_REVIEW_REQUIRED"

    code_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "inputs": {
            "cxr_metadata": str(metadata_path.resolve()),
            "cxr_split": str(split_path.resolve()),
            "echo_structured": str(echo_path.resolve()),
        },
        "code_sha256": sha256(code_path),
        "echo_schema": echo_audit,
        "cxr_projection_episodes": cxr_audit,
        "temporal_join": join,
        "admission": {
            "schema_passed": schema_passed,
            "primary_unique_patient_floor": 300,
            "primary_unique_patient_floor_passed": count_floor_passed,
            "construct_inventory_coverage_floor": 300,
            "construct_inventory_coverage_floor_passed": construct_inventory_available,
            "independent_heart_size_truth_identified": False,
            "positive_negative_borderline_bins_identified": False,
            "construct_review_required": decision
            == "DATA_GATE_COUNTS_AVAILABLE_CONSTRUCT_REVIEW_REQUIRED",
            "image_download_authorized": False,
            "gpu_authorized": False,
            "decision": decision,
        },
        "claim_firewall": [
            "A lexical echo-field inventory is not a clinical heart-size composite.",
            "LVIDd, chamber dimensions, and pericardial effusion are not interchangeable with radiographic cardiomegaly.",
            "No result in this artifact defines supported/refuted/borderline intrinsic enlargement.",
            "No image download or GPU experiment is authorized by this data audit.",
        ],
    }
    payload["fingerprint"] = _canonical_hash(payload)
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxr-metadata", type=Path, required=True)
    parser.add_argument("--cxr-split", type=Path, required=True)
    parser.add_argument("--echo-structured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.cxr_metadata, args.cxr_split, args.echo_structured):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = build_audit(
        metadata_path=args.cxr_metadata,
        split_path=args.cxr_split,
        echo_path=args.echo_structured,
    )
    atomic_json(args.output, payload)
    print(json.dumps({
        "protocol_id": payload["protocol_id"],
        "fingerprint": payload["fingerprint"],
        "decision": payload["admission"]["decision"],
        "gpu_authorized": payload["admission"]["gpu_authorized"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
