"""Consolidate common-protocol and paper-native RAG evidence without track mixing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json


PAPER_NATIVE = ("RULE", "MMed-RAG", "FactMM-RAG", "MR-RAG")


def _index_methods(payload: dict) -> dict[str, dict]:
    return {row["name"]: row for row in payload.get("methods", [])}


def audit_dual_track(evidence_path: Path, factmm_path: Path) -> dict:
    evidence = json.loads(evidence_path.read_text())
    factmm = json.loads(factmm_path.read_text())
    methods = _index_methods(evidence)
    shared = methods["shared_medical_rag"]
    causal_artifacts = []
    causal_records = []
    for event in shared["stages"]["full"].get("evidence", []):
        artifact_path = Path(event["artifact_path"])
        actual_hash = sha256_file(artifact_path)
        if actual_hash != event.get("artifact_sha256"):
            raise ValueError("shared RAG causal-control evidence hash mismatch")
        payload = json.loads(artifact_path.read_text())
        causal_artifacts.append(
            {
                "path": str(artifact_path.resolve()),
                "sha256": actual_hash,
                "protocol": payload.get("protocol_version"),
            }
        )
        for row in payload.get("records", []):
            relevance = row.get("relevance_control") or {}
            causal_records.append(
                {
                    "dataset": row.get("dataset"),
                    "model": row.get("model"),
                    "relevant_minus_shuffled_accuracy": relevance.get("accuracy_delta"),
                    "relevant_minus_shuffled_ci_low": relevance.get("accuracy_delta_ci_low"),
                    "relevant_minus_shuffled_ci_high": relevance.get("accuracy_delta_ci_high"),
                    "relevance_passed": bool(row.get("relevance_passed")),
                    "image_identity_control_present": row.get("image_identity_control") is not None,
                    "image_identity_passed": bool(row.get("image_identity_passed")),
                    "rag_grounding_supported": bool(row.get("rag_grounding_supported")),
                }
            )
    common = {
        "method": "shared_medical_rag",
        "track": "common_protocol",
        "T0": shared["stages"]["T0"]["status"],
        "T1": shared["stages"]["T1"]["status"],
        "T2": shared["stages"]["T2"]["status"],
        "T3_generation_qualification": shared["stages"]["T3"]["status"],
        "full_efficacy": shared["stages"]["full"]["status"],
        "full_reason": shared["stages"]["full"]["reason"],
        "causal_control_artifacts": causal_artifacts,
        "causal_control_records": causal_records,
        "clinical_grounding_or_utility_authorized": False,
    }
    native_rows = []
    for name in PAPER_NATIVE:
        row = methods[name]
        t0_status = row["stages"]["T0"]["status"]
        t0_reason = row["stages"]["T0"]["reason"]
        if name == "FactMM-RAG":
            t0_status = factmm["paper_native_t0_status"]
            t0_reason = ", ".join(factmm["missing_requirements"])
        native_rows.append(
            {
                "method": name,
                "track": "paper_native",
                "T0": t0_status,
                "T0_reason": t0_reason,
                "T1_authorized": t0_status == "pass",
                "efficacy_authorized": False,
            }
        )
    result = {
        "protocol_version": "rag-dual-track-qualification-v1",
        "method_evidence": str(evidence_path.resolve()),
        "method_evidence_sha256": sha256_file(evidence_path),
        "factmm_qualification": str(factmm_path.resolve()),
        "factmm_qualification_sha256": sha256_file(factmm_path),
        "common_protocol": common,
        "paper_native": native_rows,
        "tracks_kept_separate": True,
        "any_rag_efficacy_authorized": False,
        "paper_table_decision": "retain as audited negative/boundary evidence; no efficacy row",
        "claim_boundary": (
            "Common-protocol T3 generation qualification is not retrieval efficacy; "
            "paper-native methods cannot borrow the shared retriever or generator."
        ),
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-evidence", type=Path, required=True)
    parser.add_argument("--factmm-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_dual_track(args.method_evidence, args.factmm_qualification)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
