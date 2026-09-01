#!/usr/bin/env python3
"""Fail-closed qualification audit for common-protocol internal controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .artifact_registry import latest_by_artifact
from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "internal-baseline-control-qualification-audit-v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _lookup(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def _validate_stage_artifact(
    *,
    payload: dict[str, Any],
    method: str,
    stage: str,
    spec: dict[str, Any],
    contract_sha256: str,
) -> tuple[bool, list[str], bool]:
    failures: list[str] = []
    if payload.get("protocol_version") != spec.get("artifact_protocol"):
        failures.append("artifact_protocol_mismatch")
    if payload.get("contract_sha256") != contract_sha256:
        failures.append("contract_hash_mismatch")
    if payload.get("method") != method:
        failures.append("method_mismatch")
    if payload.get("stage") != stage:
        failures.append("stage_mismatch")
    for dotted, expected in spec.get("required_values", {}).items():
        try:
            observed = _lookup(payload, dotted)
        except KeyError:
            failures.append(f"missing:{dotted}")
        else:
            if observed != expected:
                failures.append(f"value:{dotted}")
    for dotted in spec.get("required_nonempty", []):
        try:
            observed = _lookup(payload, dotted)
        except KeyError:
            failures.append(f"missing:{dotted}")
        else:
            if observed is None or observed == "" or observed == [] or observed == {}:
                failures.append(f"empty:{dotted}")
    for dotted in spec.get("required_positive", []):
        try:
            observed = _lookup(payload, dotted)
        except KeyError:
            failures.append(f"missing:{dotted}")
        else:
            if isinstance(observed, bool) or not isinstance(observed, (int, float)) or observed <= 0:
                failures.append(f"not_positive:{dotted}")
    for dotted, bounds in spec.get("required_ranges", {}).items():
        try:
            observed = _lookup(payload, dotted)
        except KeyError:
            failures.append(f"missing:{dotted}")
            continue
        low, high, include_low, include_high = bounds
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            failures.append(f"not_numeric:{dotted}")
            continue
        low_ok = observed >= low if include_low else observed > low
        high_ok = observed <= high if include_high else observed < high
        if not (low_ok and high_ok):
            failures.append(f"range:{dotted}")
    for left, right in spec.get("required_distinct", []):
        try:
            equal = _lookup(payload, left) == _lookup(payload, right)
        except KeyError:
            failures.append(f"missing_distinct_pair:{left}:{right}")
        else:
            if equal:
                failures.append(f"not_distinct:{left}:{right}")
    full_failures: list[str] = []
    for dotted, expected in spec.get("full_required_values", {}).items():
        try:
            observed = _lookup(payload, dotted)
        except KeyError:
            full_failures.append(f"missing:{dotted}")
        else:
            if observed != expected:
                full_failures.append(f"value:{dotted}")
    return not failures, failures, not full_failures and bool(spec.get("full_required_values"))


def _valid_scoped_events(registry: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for artifact, event in latest_by_artifact(registry).items():
        path = Path(artifact)
        qualification = event.get("qualification_path")
        artifact_ok = path.is_file() and sha256_file(path) == event.get("artifact_sha256")
        qualification_ok = qualification is None or (
            Path(str(qualification)).is_file()
            and sha256_file(Path(str(qualification))) == event.get("qualification_sha256")
        )
        if artifact_ok and qualification_ok:
            valid.append(event)
        else:
            stale.append(
                {
                    "event_id": event.get("event_id"),
                    "artifact_path": artifact,
                    "artifact_ok": artifact_ok,
                    "qualification_ok": qualification_ok,
                }
            )
    return valid, stale


def audit(*, contract_path: Path, evidence_path: Path, registry_path: Path) -> dict[str, Any]:
    contract = _load(contract_path)
    evidence = _load(evidence_path)
    if contract.get("protocol_version") != "internal-baseline-control-contract-v1":
        raise ValueError("unsupported internal-control contract")
    contract_sha = sha256_file(contract_path)
    evidence_by_name = {row.get("name"): row for row in evidence.get("methods", [])}
    valid, stale = _valid_scoped_events(registry_path)
    template = str(contract.get("evidence_scope_template", ""))
    if template != "internal control qualification; {method}; {stage}":
        raise ValueError("unexpected internal-control evidence scope template")

    rows: list[dict[str, Any]] = []
    for control in contract.get("controls", []):
        name = str(control.get("name", ""))
        method_evidence = evidence_by_name.get(name)
        if not name or method_evidence is None:
            raise ValueError(f"control absent from method evidence ladder: {name!r}")
        t1_status = method_evidence.get("stages", {}).get("T1", {}).get("status")
        stages: dict[str, Any] = {
            "T1": {
                "status": "pass" if t1_status == "pass" else "failed",
                "reason": "canonical backend identity passed" if t1_status == "pass" else "canonical backend identity missing or failed",
            }
        }
        t2_passed = False
        for stage in ("T2", "T3"):
            spec = control[stage.lower()]
            scope = template.format(method=name, stage=stage)
            events = [
                event
                for event in valid
                if event.get("status") in {"admissible", "failed_cutoff"}
                and event.get("evidence_scope") == scope
            ]
            checked: list[dict[str, Any]] = []
            any_pass = False
            any_full = False
            for event in events:
                path = Path(str(event["artifact_path"]))
                try:
                    payload = _load(path)
                    passed, failures, full_passed = _validate_stage_artifact(
                        payload=payload,
                        method=name,
                        stage=stage,
                        spec=spec,
                        contract_sha256=contract_sha,
                    )
                    if event.get("status") == "failed_cutoff":
                        passed = False
                        full_passed = False
                        failures = [*failures, "registry_failed_cutoff"]
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    passed, failures, full_passed = False, [f"unreadable:{type(error).__name__}"], False
                if stage == "T3" and not t2_passed:
                    passed = False
                    failures = [*failures, "t2_prerequisite_not_passed"]
                    full_passed = False
                any_pass = any_pass or passed
                any_full = any_full or (passed and full_passed)
                checked.append(
                    {
                        "event_id": event.get("event_id"),
                        "artifact_path": str(path.resolve()),
                        "artifact_sha256": event.get("artifact_sha256"),
                        "registry_status": event.get("status"),
                        "passed": passed,
                        "failures": failures,
                    }
                )
            if not events:
                status = "missing"
                reason = "; ".join(spec.get("blockers_without_artifact", [])) or "no registered qualification artifact"
            elif any_pass:
                status = "pass"
                reason = "at least one current registered artifact satisfies the frozen contract"
            else:
                status = "failed"
                reason = "registered artifacts fail the frozen contract"
            stages[stage] = {"status": status, "reason": reason, "evidence": checked}
            if stage == "T2":
                t2_passed = status == "pass"
            else:
                stages["full"] = {
                    "status": "pass" if any_full else "not_authorized",
                    "reason": "T3 efficacy and no-exchange requirements passed" if any_full else "no T3 artifact passes all efficacy and no-exchange requirements",
                }
        rows.append({"name": name, "stages": stages})

    summary = {
        "controls": len(rows),
        "t1_pass": [row["name"] for row in rows if row["stages"]["T1"]["status"] == "pass"],
        "t2_pass": [row["name"] for row in rows if row["stages"]["T2"]["status"] == "pass"],
        "t2_missing": [row["name"] for row in rows if row["stages"]["T2"]["status"] == "missing"],
        "t2_failed": [row["name"] for row in rows if row["stages"]["T2"]["status"] == "failed"],
        "t3_pass": [row["name"] for row in rows if row["stages"]["T3"]["status"] == "pass"],
        "full_pass": [row["name"] for row in rows if row["stages"]["full"]["status"] == "pass"],
        "stale_registry_events": len(stale),
    }
    fingerprint_payload = {
        "version": VERSION,
        "contract_sha256": contract_sha,
        "method_evidence_sha256": sha256_file(evidence_path),
        "artifact_registry_sha256": sha256_file(registry_path),
        "summary": summary,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "version": VERSION,
        "status": "qualified" if len(summary["full_pass"]) == len(rows) else "partial_fail_closed",
        "paper_control_claim_authorized": len(summary["full_pass"]) == len(rows),
        "contract": {"path": str(contract_path.resolve()), "sha256": contract_sha},
        "method_evidence": {"path": str(evidence_path.resolve()), "sha256": sha256_file(evidence_path)},
        "artifact_registry": {"path": str(registry_path.resolve()), "sha256": sha256_file(registry_path)},
        "methods": rows,
        "summary": summary,
        "stale_registry_events": stale,
        "fingerprint": fingerprint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--method-evidence", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        contract_path=args.contract,
        evidence_path=args.method_evidence,
        registry_path=args.registry,
    )
    atomic_write_json(args.output, result)
    print(json.dumps({"status": result["status"], **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
