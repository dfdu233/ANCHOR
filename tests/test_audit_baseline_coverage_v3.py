import hashlib
import json
from pathlib import Path

import pytest

from anchor.medeval.audit_baseline_coverage_v3 import EXPECTED_CONTROLS, _validate_internal
from anchor.medeval.hashing import sha256_file


def _internal(tmp_path: Path, summary: dict) -> tuple[Path, Path, Path]:
    contract = tmp_path / "contract.json"
    evidence = tmp_path / "evidence.json"
    registry = tmp_path / "registry.jsonl"
    contract.write_text("{}")
    evidence.write_text("{}")
    registry.write_text("")
    payload = {
        "version": "internal-baseline-control-qualification-audit-v1",
        "status": "partial_fail_closed",
        "paper_control_claim_authorized": False,
        "contract": {"path": str(contract.resolve()), "sha256": sha256_file(contract)},
        "method_evidence": {"sha256": sha256_file(evidence)},
        "artifact_registry": {"sha256": sha256_file(registry)},
        "methods": [{"name": name} for name in EXPECTED_CONTROLS],
        "summary": summary,
    }
    fingerprint_payload = {
        "version": payload["version"],
        "contract_sha256": payload["contract"]["sha256"],
        "method_evidence_sha256": payload["method_evidence"]["sha256"],
        "artifact_registry_sha256": payload["artifact_registry"]["sha256"],
        "summary": summary,
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "internal.json"
    path.write_text(json.dumps(payload))
    return path, evidence, registry


def test_v3_accepts_partial_t2_progress_without_promoting_full(tmp_path: Path) -> None:
    summary = {
        "controls": 3,
        "t1_pass": EXPECTED_CONTROLS,
        "t2_pass": EXPECTED_CONTROLS[:2],
        "t2_missing": EXPECTED_CONTROLS[2:],
        "t2_failed": [],
        "t3_pass": [],
        "full_pass": [],
        "stale_registry_events": 0,
    }
    path, evidence, registry = _internal(tmp_path, summary)
    internal, _ = _validate_internal(
        path=path,
        evidence_path=evidence,
        registry_path=registry,
        configured_names=[*EXPECTED_CONTROLS, "greedy"],
    )
    assert internal["summary"]["t2_pass"] == EXPECTED_CONTROLS[:2]
    assert internal["paper_control_claim_authorized"] is False


def test_v3_rejects_non_partitioned_control_status(tmp_path: Path) -> None:
    summary = {
        "controls": 3,
        "t1_pass": EXPECTED_CONTROLS,
        "t2_pass": EXPECTED_CONTROLS[:2],
        "t2_missing": [EXPECTED_CONTROLS[1]],
        "t2_failed": [],
        "t3_pass": [],
        "full_pass": [],
        "stale_registry_events": 0,
    }
    path, evidence, registry = _internal(tmp_path, summary)
    with pytest.raises(ValueError):
        _validate_internal(
            path=path,
            evidence_path=evidence,
            registry_path=registry,
            configured_names=EXPECTED_CONTROLS,
        )
