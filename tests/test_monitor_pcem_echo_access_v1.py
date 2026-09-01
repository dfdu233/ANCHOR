from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.monitor_pcem_echo_access_v1 import (
    MONITOR_VERSION,
    _sha256,
    heartbeat_payload,
    resolve_echo_candidate,
    valid_existing_audit,
)


def test_resolve_echo_candidate_waits_and_requires_unique_file(tmp_path: Path) -> None:
    first = tmp_path / "structured_measurement.csv"
    second = tmp_path / "structured_measurement.csv.gz"
    assert resolve_echo_candidate([first], root=tmp_path) is None
    first.write_text("a\n", encoding="utf-8")
    assert resolve_echo_candidate([first], root=tmp_path) == first.resolve()
    second.write_text("b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="multiple MIMIC-IV-ECHO"):
        resolve_echo_candidate([first, second], root=tmp_path)


def test_resolve_echo_candidate_rejects_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes protected data root"):
        resolve_echo_candidate([outside], root=root)


def test_existing_audit_requires_exact_input_and_safety_firewall(tmp_path: Path) -> None:
    echo = tmp_path / "structured_measurement.csv"
    echo.write_text("protected bytes\n", encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    row = {
        "protocol_id": "pcem-echo-temporal-join-audit-v1",
        "inputs": {"echo_structured": str(echo)},
        "echo_schema": {"sha256": _sha256(echo)},
        "admission": {
            "gpu_authorized": False,
            "image_download_authorized": False,
            "independent_heart_size_truth_identified": False,
        },
        "fingerprint": "a" * 64,
    }
    audit_path.write_text(json.dumps(row), encoding="utf-8")
    assert valid_existing_audit(audit_path, echo)

    row["admission"]["gpu_authorized"] = True
    audit_path.write_text(json.dumps(row), encoding="utf-8")
    assert not valid_existing_audit(audit_path, echo)


def test_existing_audit_detects_raw_file_drift(tmp_path: Path) -> None:
    echo = tmp_path / "structured_measurement.csv"
    echo.write_text("v1\n", encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "protocol_id": "pcem-echo-temporal-join-audit-v1",
                "inputs": {"echo_structured": str(echo)},
                "echo_schema": {"sha256": _sha256(echo)},
                "admission": {
                    "gpu_authorized": False,
                    "image_download_authorized": False,
                    "independent_heart_size_truth_identified": False,
                },
                "fingerprint": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    echo.write_text("v2\n", encoding="utf-8")
    assert not valid_existing_audit(audit_path, echo)


def test_heartbeat_never_implies_authentication_or_compute_authority() -> None:
    row = heartbeat_payload("waiting_for_authorized_echo_mount")
    assert row["version"] == MONITOR_VERSION
    assert row["authentication_attempted"] is False
    assert row["protected_data_downloaded_by_monitor"] is False
    assert row["image_download_authorized"] is False
    assert row["gpu_authorized"] is False
