#!/usr/bin/env python3
"""Build a fail-closed T0--full evidence ladder from immutable artifacts.

This module intentionally recognizes only evidence scopes emitted by the
maintained unified-evaluation runners.  Historical files, filenames that merely
contain a method name, and unregistered outputs can never promote a method.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .artifact_registry import latest_by_artifact
from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "method-evidence-ladder-audit-v1"
RAG_SCOPE = re.compile(
    r"^common_protocol visual CE-G; (?P<dataset>[^;]+); "
    r"(?P<model>[^;]+); (?P<arm>no_context|rag); (?P<stage>T2_n32|T3_n200)$"
)
OE_CONTROL_SCOPE = re.compile(
    r"^canonical OE-VQA functional smoke; (?P<dataset>[^;]+); "
    r"(?P<model>[^;]+); (?P<arm>greedy256|beam\d+_\d+); (?P<stage>T2_n32)$"
)
MITIGATION_T2_SCOPE = re.compile(
    r"^canonical OE-VQA mitigation smoke; vqa-rad; llava; "
    r"(?P<method>VCD|OPERA|PAI|AvisC|VISTA); T2_n32$"
)
INTERNAL_CONTROL_SCOPE = re.compile(
    r"^internal control qualification; "
    r"(?P<method>temperature_length_controls|self_consistency|calibrated_abstention); "
    r"(?P<stage>T2|T3)$"
)


def _valid_events(registry: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for artifact, event in latest_by_artifact(registry).items():
        artifact_path = Path(artifact)
        qualification = event.get("qualification_path")
        artifact_ok = artifact_path.is_file() and sha256_file(artifact_path) == event.get(
            "artifact_sha256"
        )
        qualification_ok = qualification is None or (
            Path(qualification).is_file()
            and sha256_file(Path(qualification)) == event.get("qualification_sha256")
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


def _stage(status: str, reason: str, evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"status": status, "reason": reason, "evidence": evidence or []}


def _compact(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "artifact_path": event["artifact_path"],
        "artifact_sha256": event["artifact_sha256"],
        "evidence_scope": event["evidence_scope"],
        "status": event["status"],
    }


def _identity_evidence(identity_gate: Path | None) -> tuple[str, list[dict[str, Any]], str]:
    if identity_gate is None or not identity_gate.is_file():
        return "missing", [], "no frozen cross-backend identity gate was supplied"
    payload = json.loads(identity_gate.read_text())
    evidence = [
        {
            "artifact_path": str(identity_gate.resolve()),
            "artifact_sha256": sha256_file(identity_gate),
            "protocol": payload.get("protocol"),
            "backends": sorted(payload.get("backends", {})),
        }
    ]
    if payload.get("passed") is True:
        return "pass", evidence, "frozen runtime identity gate passed"
    return "failed", evidence, "frozen runtime identity gate failed"


def _vista_identity_evidence(
    events: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], str]:
    """Certify VISTA's method-off identity from its registered ablation audit.

    VISTA has a method-specific off arm, so the generic mitigation-backend
    identity fixture is necessary but not sufficient.  Only the frozen audit
    schema with exact generated-token and byte identity can satisfy T1.
    """

    evidence: list[dict[str, Any]] = []
    valid = False
    for event in events:
        path = Path(str(event.get("artifact_path", "")))
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        method_off = payload.get("method_off", {})
        t1 = payload.get("t1", {})
        t2 = payload.get("t2", {})
        passed = (
            payload.get("version") == "vista-llava-med-t2-256-ablation-audit-v1"
            and method_off.get("passed") is True
            and method_off.get("generated_token_exact_rate") == 1.0
            and method_off.get("answer_files_byte_identical") is True
            and t1.get("status") == "passed"
            and t1.get("generated_token_exact_rate") == 1.0
            and t2.get("status") == "passed_functional_activation_only"
            and int(t2.get("changed_generated_sequences", 0)) > 0
            and payload.get("clinical_efficacy_claim") is False
            and payload.get("t3_authorized") is False
        )
        evidence.append(
            {
                "artifact_path": str(path.resolve()),
                "artifact_sha256": event.get("artifact_sha256"),
                "event_id": event.get("event_id"),
                "protocol": payload.get("version"),
                "generated_token_exact_rate": t1.get(
                    "generated_token_exact_rate"
                ),
                "changed_generated_sequences": t2.get(
                    "changed_generated_sequences"
                ),
            }
        )
        valid = valid or passed
    if valid:
        return (
            "pass",
            evidence,
            "registered VISTA-off generated-token and answer-byte identity passed",
        )
    return (
        "missing" if not evidence else "failed",
        evidence,
        "no registered exact VISTA method-off identity audit passed",
    )


def audit(
    *,
    t0_audit: Path,
    registry: Path,
    identity_gate: Path | None = None,
    mitigation_identity_gate: Path | None = None,
    rag_causal_summary: Path | None = None,
) -> dict[str, Any]:
    t0 = json.loads(t0_audit.read_text())
    valid, stale = _valid_events(registry)
    admissible = [event for event in valid if event.get("status") == "admissible"]
    identity_status, identity_evidence, identity_reason = _identity_evidence(identity_gate)
    mitigation_identity_status, mitigation_identity_evidence, mitigation_identity_reason = (
        _identity_evidence(mitigation_identity_gate)
    )

    rag: dict[tuple[str, str], list[dict[str, Any]]] = {}
    oe_controls: dict[str, list[dict[str, Any]]] = {}
    mitigation_t2: dict[str, list[dict[str, Any]]] = {}
    internal_controls: dict[tuple[str, str], list[dict[str, Any]]] = {}
    internal_control_failures: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in valid:
        internal_match = INTERNAL_CONTROL_SCOPE.fullmatch(
            str(event.get("evidence_scope", ""))
        )
        if internal_match and event.get("status") in {"admissible", "failed_cutoff"}:
            target = (
                internal_controls
                if event.get("status") == "admissible"
                else internal_control_failures
            )
            target.setdefault(
                (internal_match.group("method"), internal_match.group("stage")), []
            ).append(event)
    for event in admissible:
        match = RAG_SCOPE.fullmatch(str(event.get("evidence_scope", "")))
        if match:
            key = (match.group("arm"), match.group("stage"))
            rag.setdefault(key, []).append(event)
        control_match = OE_CONTROL_SCOPE.fullmatch(str(event.get("evidence_scope", "")))
        if control_match:
            arm = control_match.group("arm")
            method = "greedy" if arm == "greedy256" else "beam"
            oe_controls.setdefault(method, []).append(event)
        mitigation_match = MITIGATION_T2_SCOPE.fullmatch(
            str(event.get("evidence_scope", ""))
        )
        if mitigation_match:
            mitigation_t2.setdefault(mitigation_match.group("method"), []).append(event)

    oe_smoke = [
        event
        for event in admissible
        if event.get("evidence_scope") == "OE generation qualification smoke only"
    ]
    oe_full = [
        event
        for event in admissible
        if str(event.get("evidence_scope", "")).startswith("qualified raw OE generation;")
    ]

    causal_payload: dict[str, Any] | None = None
    if rag_causal_summary is not None and rag_causal_summary.is_file():
        causal_payload = json.loads(rag_causal_summary.read_text())
    causal_evidence = []
    if causal_payload is not None:
        causal_evidence = [
            {
                "artifact_path": str(rag_causal_summary.resolve()),
                "artifact_sha256": sha256_file(rag_causal_summary),
                "protocol": causal_payload.get("protocol_version"),
                "supported": causal_payload.get("supported", []),
            }
        ]

    methods = []
    for method in t0["methods"]:
        name = method["name"]
        t0_status = method["t0_status"]
        row = {
            "name": name,
            "family": method["family"],
            "tracks": method["tracks"],
            "tasks": method["tasks"],
            "cutoff": method["cutoff"],
            "stages": {
                "T0": _stage(t0_status, ", ".join(method.get("t0_reasons", [])) or "source gate passed"),
                "T1": _stage("missing", "no registered method-off identity evidence"),
                "T2": _stage("missing", "no registered functional smoke evidence"),
                "T3": _stage("missing", "no registered paired pilot evidence"),
                "full": _stage("not_authorized", "T3 promotion gate has not passed"),
            },
        }
        if t0_status != "pass":
            for stage in ("T1", "T2", "T3", "full"):
                row["stages"][stage] = _stage(
                    "not_admissible", "blocked by the fail-closed T0 source/license/checkpoint gate"
                )
        elif name == "greedy":
            row["stages"]["T1"] = _stage(identity_status, identity_reason, identity_evidence)
            t2_evidence = (
                rag.get(("no_context", "T2_n32"), [])
                + oe_smoke
                + oe_controls.get("greedy", [])
            )
            t3_evidence = rag.get(("no_context", "T3_n200"), []) + oe_full
            row["stages"]["T2"] = _stage(
                "pass" if t2_evidence else "missing",
                "qualified canonical generation exists" if t2_evidence else "no qualified smoke",
                [_compact(event) for event in t2_evidence],
            )
            row["stages"]["T3"] = _stage(
                "pass" if t3_evidence else "missing",
                "qualified canonical generation exists; OE clinical claim scoring remains separate"
                if t3_evidence
                else "no qualified pilot",
                [_compact(event) for event in t3_evidence],
            )
            row["stages"]["full"] = _stage(
                "reference_only",
                "greedy is the retained reference arm, not a mitigation efficacy claim",
            )
        elif name == "beam":
            row["stages"]["T1"] = _stage(identity_status, identity_reason, identity_evidence)
            evidence = oe_controls.get("beam", [])
            row["stages"]["T2"] = _stage(
                "pass" if evidence else "missing",
                "qualified canonical beam functional smoke exists"
                if evidence
                else "no qualified canonical beam smoke",
                [_compact(event) for event in evidence],
            )
            row["stages"]["T3"] = _stage(
                "missing", "no paired clinical claim pilot at matched length/coverage"
            )
            row["stages"]["full"] = _stage(
                "not_authorized", "beam is not promoted by format or lexical qualification alone"
            )
        elif name in {
            "temperature_length_controls",
            "self_consistency",
            "calibrated_abstention",
        }:
            # These controls execute through the canonical backend rather than
            # a separate method fork.  T1 therefore certifies that backend;
            # it does not imply that the control itself has passed T2.
            row["stages"]["T1"] = _stage(
                identity_status,
                identity_reason,
                identity_evidence,
            )
            t2_events = internal_controls.get((name, "T2"), [])
            t3_events = internal_controls.get((name, "T3"), [])
            t2_failures = internal_control_failures.get((name, "T2"), [])
            t3_failures = internal_control_failures.get((name, "T3"), [])
            row["stages"]["T2"] = _stage(
                "pass" if t2_events else ("failed_cutoff" if t2_failures else "missing"),
                "registered frozen-contract functional qualification passed; no efficacy claim"
                if t2_events
                else (
                    "executed qualification failed its frozen non-degeneracy or action cutoff"
                    if t2_failures
                    else "no registered functional smoke for this canonical-backend control"
                ),
                [_compact(event) for event in (t2_events or t2_failures)],
            )
            row["stages"]["T3"] = _stage(
                "pass" if t3_events else ("failed_cutoff" if t3_failures else "missing"),
                "registered paired clinical-claim pilot passed"
                if t3_events
                else (
                    "executed clinical pilot failed its frozen cutoff"
                    if t3_failures
                    else "no paired clinical-claim pilot at matched length and coverage"
                ),
                [_compact(event) for event in (t3_events or t3_failures)],
            )
            row["stages"]["full"] = _stage(
                "not_authorized",
                "T2/T3 promotion gates have not passed",
            )
        elif name == "shared_medical_rag":
            row["stages"]["T1"] = _stage(identity_status, identity_reason, identity_evidence)
            for stage, scope_stage in (("T2", "T2_n32"), ("T3", "T3_n200")):
                events = rag.get(("rag", scope_stage), [])
                row["stages"][stage] = _stage(
                    "pass" if events else "missing",
                    "generation/evaluation qualification passed; this does not establish retrieval grounding"
                    if events
                    else "no qualified common-protocol RAG artifact",
                    [_compact(event) for event in events],
                )
            supported = [] if causal_payload is None else causal_payload.get("supported", [])
            row["stages"]["full"] = _stage(
                "pass" if supported else ("failed_cutoff" if causal_payload is not None else "not_authorized"),
                "causal retrieval and image-identity grounding gates passed"
                if supported
                else (
                    "no dataset passed the preregistered relevance plus image-identity grounding gate"
                    if causal_payload is not None
                    else "causal grounding summary missing"
                ),
                causal_evidence,
            )
        elif name in {"VCD", "OPERA", "PAI", "AvisC", "VISTA"}:
            evidence = mitigation_t2.get(name, [])
            if name == "VISTA":
                vista_identity_status, vista_identity_evidence, vista_identity_reason = (
                    _vista_identity_evidence(evidence)
                )
                row["stages"]["T1"] = _stage(
                    vista_identity_status,
                    vista_identity_reason,
                    vista_identity_evidence,
                )
            else:
                row["stages"]["T1"] = _stage(
                    mitigation_identity_status,
                    mitigation_identity_reason,
                    mitigation_identity_evidence,
                )
            row["stages"]["T2"] = _stage(
                "pass" if evidence else "missing",
                (
                    "trace-certified method-off identity and nondegenerate method activation passed; "
                    "clinical efficacy remains untested"
                    if evidence
                    else "no qualified current-backend mitigation smoke"
                ),
                [_compact(event) for event in evidence],
            )
            row["stages"]["T3"] = _stage(
                "missing",
                "no paired clinical-claim pilot at matched length and coverage",
            )
            row["stages"]["full"] = _stage(
                "not_authorized",
                "T2 execution evidence cannot establish hallucination mitigation",
            )
        methods.append(row)

    return {
        "protocol_version": VERSION,
        "t0_audit": str(t0_audit.resolve()),
        "t0_audit_sha256": sha256_file(t0_audit),
        "artifact_registry": str(registry.resolve()),
        "artifact_registry_sha256": sha256_file(registry),
        "identity_gate": None if identity_gate is None else str(identity_gate.resolve()),
        "mitigation_identity_gate": (
            None
            if mitigation_identity_gate is None
            else str(mitigation_identity_gate.resolve())
        ),
        "rag_causal_summary": (
            None if rag_causal_summary is None else str(rag_causal_summary.resolve())
        ),
        "methods": methods,
        "summary": {
            "methods": len(methods),
            "t0_pass": sum(row["stages"]["T0"]["status"] == "pass" for row in methods),
            "t0_not_admissible": sum(
                row["stages"]["T0"]["status"] != "pass" for row in methods
            ),
            "t3_pass": [row["name"] for row in methods if row["stages"]["T3"]["status"] == "pass"],
            "full_pass": [
                row["name"] for row in methods if row["stages"]["full"]["status"] == "pass"
            ],
            "stale_registry_events": len(stale),
        },
        "stale_registry_events": stale,
        "interpretation": (
            "A stage pass certifies execution/evaluation evidence only. Full is the sole mitigation "
            "efficacy gate; missing and failed_cutoff results must not be reported as positive baselines."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-audit", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--identity-gate", type=Path)
    parser.add_argument("--mitigation-identity-gate", type=Path)
    parser.add_argument("--rag-causal-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        t0_audit=args.t0_audit,
        registry=args.registry,
        identity_gate=args.identity_gate,
        mitigation_identity_gate=args.mitigation_identity_gate,
        rag_causal_summary=args.rag_causal_summary,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
