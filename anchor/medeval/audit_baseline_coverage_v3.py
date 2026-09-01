#!/usr/bin/env python3
"""Baseline coverage audit that admits T2 controls without implying efficacy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .audit_baseline_coverage_v1 import _named_path, audit as audit_v1
from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "baseline-coverage-audit-v3"
EXPECTED_CONTROLS = [
    "temperature_length_controls",
    "self_consistency",
    "calibrated_abstention",
]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_internal(
    *, path: Path, evidence_path: Path, registry_path: Path, configured_names: list[str]
) -> tuple[dict[str, Any], Path]:
    internal = _load(path)
    summary = internal.get("summary", {})
    names = [row.get("name") for row in internal.get("methods", [])]
    fingerprint_payload = {
        "version": internal.get("version"),
        "contract_sha256": internal.get("contract", {}).get("sha256"),
        "method_evidence_sha256": internal.get("method_evidence", {}).get("sha256"),
        "artifact_registry_sha256": internal.get("artifact_registry", {}).get("sha256"),
        "summary": summary,
    }
    expected_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract_record = internal.get("contract", {})
    contract_path = Path(str(contract_record.get("path", "")))
    t2_pass = list(summary.get("t2_pass", []))
    t2_missing = list(summary.get("t2_missing", []))
    t2_failed = list(summary.get("t2_failed", []))
    partition = t2_pass + t2_missing + t2_failed
    if (
        internal.get("version") != "internal-baseline-control-qualification-audit-v1"
        or internal.get("status") != "partial_fail_closed"
        or internal.get("paper_control_claim_authorized") is not False
        or internal.get("fingerprint") != expected_fingerprint
        or internal.get("method_evidence", {}).get("sha256") != sha256_file(evidence_path)
        or internal.get("artifact_registry", {}).get("sha256") != sha256_file(registry_path)
        or not contract_path.is_file()
        or contract_record.get("sha256") != sha256_file(contract_path)
        or names != EXPECTED_CONTROLS
        or not set(EXPECTED_CONTROLS).issubset(configured_names)
        or summary.get("controls") != len(EXPECTED_CONTROLS)
        or summary.get("t1_pass") != EXPECTED_CONTROLS
        or sorted(partition) != sorted(EXPECTED_CONTROLS)
        or len(partition) != len(set(partition))
        or summary.get("t3_pass") != []
        or summary.get("full_pass") != []
        or summary.get("stale_registry_events") != 0
    ):
        raise ValueError("internal baseline-control qualification is stale or permissive")
    return internal, contract_path


def audit(
    *,
    config_path: Path,
    t0_path: Path,
    evidence_path: Path,
    registry_path: Path,
    native_acceptance_path: Path,
    rag_causal_path: Path,
    internal_control_path: Path,
    report_audits: list[tuple[str, Path]],
    physician_analysis_path: Path | None = None,
) -> dict[str, Any]:
    result = audit_v1(
        config_path=config_path,
        t0_path=t0_path,
        evidence_path=evidence_path,
        registry_path=registry_path,
        native_acceptance_path=native_acceptance_path,
        rag_causal_path=rag_causal_path,
        report_audits=report_audits,
        physician_analysis_path=physician_analysis_path,
    )
    configured = [row.get("name") for row in result.get("methods", [])]
    internal, contract_path = _validate_internal(
        path=internal_control_path,
        evidence_path=evidence_path,
        registry_path=registry_path,
        configured_names=configured,
    )
    summary = internal["summary"]
    result["version"] = VERSION
    result["gates"]["internal_control_contract_enforced"] = True
    result["gates"]["internal_controls_t2_complete"] = not summary["t2_missing"] and not summary["t2_failed"]
    result["gates"]["internal_controls_clinical_efficacy_complete"] = bool(summary["full_pass"])
    result["gates"]["paper_main_table_authorized"] = False
    result["paper_baseline_claim_authorized"] = False
    result["summary"]["internal_control_qualification"] = {
        "status": internal["status"],
        "t2_pass": summary["t2_pass"],
        "t2_missing": summary["t2_missing"],
        "t2_failed": summary["t2_failed"],
        "full_pass": summary["full_pass"],
        "fingerprint": internal["fingerprint"],
    }
    # V1 predates explicit negative execution evidence and labels every
    # non-pass as "missing". Preserve its field for compatibility while
    # correcting the partition in this stricter version.
    prior_missing = list(result["summary"].get("t2_missing_after_t0_pass", []))
    result["summary"]["t2_missing_after_t0_pass"] = [
        name for name in prior_missing if name not in summary["t2_failed"]
    ]
    result["summary"]["t2_failed_after_t0_pass"] = list(summary["t2_failed"])
    result["provenance"]["internal_control_qualification"] = {
        "path": str(internal_control_path.resolve()),
        "sha256": sha256_file(internal_control_path),
    }
    result["provenance"]["internal_control_contract"] = {
        "path": str(contract_path.resolve()),
        "sha256": sha256_file(contract_path),
    }
    fingerprint_payload = {
        "version": VERSION,
        "inputs": {name: row["sha256"] for name, row in result["provenance"].items()},
        "method_names": configured,
        "gates": result["gates"],
    }
    result["fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--t0", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--native-acceptance", type=Path, required=True)
    parser.add_argument("--rag-causal", type=Path, required=True)
    parser.add_argument("--internal-control-qualification", type=Path, required=True)
    parser.add_argument("--report-audit", action="append", type=_named_path, default=[])
    parser.add_argument("--physician-analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        config_path=args.config,
        t0_path=args.t0,
        evidence_path=args.evidence,
        registry_path=args.registry,
        native_acceptance_path=args.native_acceptance,
        rag_causal_path=args.rag_causal,
        internal_control_path=args.internal_control_qualification,
        report_audits=args.report_audit,
        physician_analysis_path=args.physician_analysis,
    )
    atomic_write_json(args.output, result)
    print(json.dumps({"status": result["status"], **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
