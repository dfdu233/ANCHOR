from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from anchor.corrected_sgta.audit_pcem_echo_join_v1 import (
    build_audit,
    is_size_inventory_candidate,
    load_echo_studies,
    normalize_test_type,
    parse_echo_datetime,
)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    metadata = tmp_path / "metadata.csv"
    split = tmp_path / "split.csv"
    echo = tmp_path / "structured_measurement.csv"
    metadata_fields = [
        "dicom_id",
        "subject_id",
        "study_id",
        "ViewPosition",
        "StudyDate",
        "StudyTime",
    ]
    split_fields = ["dicom_id", "subject_id", "study_id", "split"]
    metadata_rows = [
        {"dicom_id": "a1", "subject_id": "1", "study_id": "10", "ViewPosition": "AP", "StudyDate": "20200101", "StudyTime": "100000"},
        {"dicom_id": "p1", "subject_id": "1", "study_id": "10", "ViewPosition": "PA", "StudyDate": "20200101", "StudyTime": "110000"},
        {"dicom_id": "a2", "subject_id": "2", "study_id": "20", "ViewPosition": "AP", "StudyDate": "20200102", "StudyTime": "100000"},
        {"dicom_id": "p2", "subject_id": "2", "study_id": "21", "ViewPosition": "PA", "StudyDate": "20200102", "StudyTime": "120000"},
        {"dicom_id": "a3", "subject_id": "3", "study_id": "30", "ViewPosition": "AP", "StudyDate": "20200103", "StudyTime": "100000"},
    ]
    _write_csv(metadata, metadata_fields, metadata_rows)
    _write_csv(
        split,
        split_fields,
        [
            {"dicom_id": row["dicom_id"], "subject_id": row["subject_id"], "study_id": row["study_id"], "split": "train"}
            for row in metadata_rows
        ],
    )
    echo_fields = [
        "subject_id",
        "measurement_id",
        "measurement_datetime",
        "test_type",
        "measurement",
        "measurement_description",
        "result",
        "unit",
    ]
    _write_csv(
        echo,
        echo_fields,
        [
            {"subject_id": "1", "measurement_id": "e1", "measurement_datetime": "2020-01-01 10:30:00", "test_type": "TTE", "measurement": "lvidd", "measurement_description": "Left ventricular internal diameter", "result": "45", "unit": "mm"},
            {"subject_id": "1", "measurement_id": "e1", "measurement_datetime": "2020-01-01 10:30:00", "test_type": "TTE", "measurement": "lvef", "measurement_description": "Ejection fraction", "result": "60", "unit": "%"},
            {"subject_id": "2", "measurement_id": "e2", "measurement_datetime": "2020-01-02 11:00:00", "test_type": "Transthoracic echocardiogram", "measurement": "lvef", "measurement_description": "Ejection fraction", "result": "55", "unit": "%"},
            {"subject_id": "3", "measurement_id": "e3", "measurement_datetime": "2020-01-03 10:00:00", "test_type": "TEE", "measurement": "lvidd", "measurement_description": "LVIDd", "result": "40", "unit": "mm"},
        ],
    )
    return metadata, split, echo


def test_parse_and_inventory_helpers() -> None:
    assert parse_echo_datetime("2026-08-03 12:34:56").isoformat() == "2026-08-03T12:34:56"
    assert parse_echo_datetime("") is None
    assert normalize_test_type("transthoracic echocardiogram") == "TTE"
    assert normalize_test_type("stress echo") == "STRESS"
    assert is_size_inventory_candidate("LVIDd", "")
    assert not is_size_inventory_candidate("LVEF", "Ejection fraction")


def test_build_audit_counts_temporal_links_without_claiming_truth(tmp_path: Path) -> None:
    metadata, split, echo = _fixture(tmp_path)
    audit = build_audit(metadata_path=metadata, split_path=split, echo_path=echo)
    primary = audit["temporal_join"]["cells"]["pair_le_6h__echo_le_24h"]
    assert primary == {
        "unique_patients": 2,
        "same_study_projection_episodes": 1,
        "episodes_with_lexical_size_inventory_candidate": 1,
    }
    assert audit["cxr_projection_episodes"]["subjects_with_selected_episode"] == 2
    assert audit["echo_schema"]["test_type_study_counts"] == {"TEE": 1, "TTE": 2}
    assert audit["admission"]["decision"] == "TEMPORAL_JOIN_COUNT_GATE_FAILED"
    assert audit["admission"]["independent_heart_size_truth_identified"] is False
    assert audit["admission"]["gpu_authorized"] is False
    assert audit["temporal_join"]["patient_identifiers_written"] is False


def test_fingerprint_is_deterministic(tmp_path: Path) -> None:
    metadata, split, echo = _fixture(tmp_path)
    first = build_audit(metadata_path=metadata, split_path=split, echo_path=echo)
    second = build_audit(metadata_path=metadata, split_path=split, echo_path=echo)
    assert first == second
    assert len(first["fingerprint"]) == 64


def test_echo_schema_is_fail_closed(tmp_path: Path) -> None:
    echo = tmp_path / "bad.csv"
    _write_csv(echo, ["subject_id"], [{"subject_id": "1"}])
    with pytest.raises(ValueError, match="echo schema missing fields"):
        load_echo_studies(echo)


def test_echo_measurement_identity_must_be_consistent(tmp_path: Path) -> None:
    echo = tmp_path / "inconsistent.csv"
    fields = [
        "subject_id",
        "measurement_id",
        "measurement_datetime",
        "test_type",
        "measurement",
        "measurement_description",
        "result",
        "unit",
    ]
    _write_csv(
        echo,
        fields,
        [
            {"subject_id": "1", "measurement_id": "e1", "measurement_datetime": "2020-01-01 00:00:00", "test_type": "TTE", "measurement": "lvidd", "measurement_description": "LVIDd", "result": "45", "unit": "mm"},
            {"subject_id": "2", "measurement_id": "e1", "measurement_datetime": "2020-01-01 00:00:00", "test_type": "TTE", "measurement": "lvidd", "measurement_description": "LVIDd", "result": "45", "unit": "mm"},
        ],
    )
    with pytest.raises(ValueError, match="inconsistent subject/time/test_type"):
        load_echo_studies(echo)


def test_truncated_gzip_never_becomes_an_available_echo_schema(tmp_path: Path) -> None:
    echo = tmp_path / "structured_measurement.csv.gz"
    echo.write_bytes(b"\x1f\x8b\x08\x00truncated")
    with pytest.raises((EOFError, gzip.BadGzipFile)):
        load_echo_studies(echo)


def test_count_qualified_substrate_still_requires_construct_review(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    split = tmp_path / "split.csv"
    echo = tmp_path / "structured_measurement.csv"
    metadata_fields = [
        "dicom_id",
        "subject_id",
        "study_id",
        "ViewPosition",
        "StudyDate",
        "StudyTime",
    ]
    split_fields = ["dicom_id", "subject_id", "study_id", "split"]
    echo_fields = [
        "subject_id",
        "measurement_id",
        "measurement_datetime",
        "test_type",
        "measurement",
        "measurement_description",
        "result",
        "unit",
    ]
    metadata_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    echo_rows: list[dict[str, object]] = []
    for index in range(300):
        subject = str(100000 + index)
        study = str(200000 + index)
        for view, suffix, time in (("AP", "a", "100000"), ("PA", "p", "110000")):
            dicom = f"{subject}-{suffix}"
            metadata_rows.append(
                {
                    "dicom_id": dicom,
                    "subject_id": subject,
                    "study_id": study,
                    "ViewPosition": view,
                    "StudyDate": "20200101",
                    "StudyTime": time,
                }
            )
            split_rows.append(
                {
                    "dicom_id": dicom,
                    "subject_id": subject,
                    "study_id": study,
                    "split": "train",
                }
            )
        echo_rows.append(
            {
                "subject_id": subject,
                "measurement_id": f"e-{subject}",
                "measurement_datetime": "2020-01-01 10:30:00",
                "test_type": "TTE",
                "measurement": "lvidd",
                "measurement_description": "Left ventricular internal diameter",
                "result": "45",
                "unit": "mm",
            }
        )
    _write_csv(metadata, metadata_fields, metadata_rows)
    _write_csv(split, split_fields, split_rows)
    _write_csv(echo, echo_fields, echo_rows)

    audit = build_audit(metadata_path=metadata, split_path=split, echo_path=echo)
    assert audit["admission"]["decision"] == "DATA_GATE_COUNTS_AVAILABLE_CONSTRUCT_REVIEW_REQUIRED"
    assert audit["admission"]["construct_review_required"] is True
    assert audit["admission"]["independent_heart_size_truth_identified"] is False
    assert audit["admission"]["positive_negative_borderline_bins_identified"] is False
    assert audit["admission"]["image_download_authorized"] is False
    assert audit["admission"]["gpu_authorized"] is False
