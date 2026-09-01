"""Build a fail-closed paper-progress supplement for repaired OE and RAG baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def build(
    *,
    base_audit: Path,
    rag_audit: Path,
    v1_failure: Path,
    v2_execution: Path,
    v2_provenance: Path,
    v1_job_state: Path,
    v2_job_state: Path,
) -> dict[str, Any]:
    base = _load(base_audit)
    rag = _load(rag_audit)
    failure = _load(v1_failure)
    execution = _load(v2_execution)
    provenance = _load(v2_provenance)
    v1_state = _load(v1_job_state)
    v2_state = _load(v2_job_state)
    if base.get("paper_ready") is not False:
        raise ValueError("base ICLR audit is unexpectedly paper-ready")
    if failure.get("all_eligible") is not False:
        raise ValueError("T3-v1 repair basis no longer records a qualification failure")
    trigger = execution["repair_trigger"]
    if trigger.get("artifact_sha256") != sha256_file(v1_failure):
        raise ValueError("T3-v2 execution is not bound to the frozen v1 failure")
    if provenance.get("execution_contract_sha256") != sha256_file(v2_execution):
        raise ValueError("T3-v2 provenance/contract mismatch")
    if rag.get("tracks_kept_separate") is not True or rag.get("any_rag_efficacy_authorized") is not False:
        raise ValueError("RAG dual-track audit is stale or permissive")

    v2_status = str(v2_state.get("status", "missing"))
    result = {
        "version": "iclr-evaluation-progress-supplement-v2",
        "paper_ready": False,
        "submission_claim_authorized": False,
        "human_labels_synthesized": False,
        "inputs": {
            "base_completion_audit": _record(base_audit),
            "rag_dual_track_audit": _record(rag_audit),
            "t3_v1_failure_prereg": _record(v1_failure),
            "t3_v2_execution_contract": _record(v2_execution),
            "t3_v2_freeze_provenance": _record(v2_provenance),
            "t3_v1_job_state": _record(v1_job_state),
            "t3_v2_job_state": _record(v2_job_state),
        },
        "evaluation_state": {
            "t3_v1_job_status": v1_state.get("status"),
            "t3_v1_clinical_disposition": "identity_and_length_stress_only",
            "t3_v1_physician_pack_authorized": False,
            "t3_v2_job_status": v2_status,
            "t3_v2_generation_qualified": False,
            "t3_v2_clinical_efficacy_authorized": False,
            "common_rag_t3_generation_qualification": rag["common_protocol"]["T3_generation_qualification"],
            "common_rag_full_efficacy": rag["common_protocol"]["full_efficacy"],
            "paper_native_rag_t0": {
                row["method"]: row["T0"] for row in rag["paper_native"]
            },
            "any_rag_efficacy_authorized": False,
        },
        "requirements": [
            {
                "id": "R4",
                "requirement": "OE mitigation improves physician claim fidelity without omission, length, coverage, or refusal exchange",
                "status": "t3_v2_generation_repair_pending" if v2_status in {"running", "starting"} else "missing",
                "evidence": "T3-v1 was stopped before physician packaging because three Huatuo sampling arms exceeded the frozen cap-hit ceiling; a common 512-token two-model repair is frozen without labels.",
            },
            {
                "id": "R5",
                "requirement": "Unified identity- and response-form-qualified CE/OE evaluation substrate",
                "status": "engineering_repair_in_progress",
                "evidence": "The held-out manifest is byte-identical across v1/v2; v2 must pass exact traces and operational OE gates before clinical packaging.",
            },
            {
                "id": "R6",
                "requirement": "Recent decoding and RAG baselines are honestly source- and efficacy-qualified",
                "status": "rag_boundary_complete_decoding_t3_pending",
                "evidence": "Common-protocol RAG passed generation qualification but failed the frozen causal grounding cutoff; RULE, MMed-RAG, FactMM-RAG and MR-RAG remain separate paper-native T0 decisions.",
            },
            {
                "id": "R7",
                "requirement": "Independent physician evaluation",
                "status": "not_authorized_until_t3_v2_generation_passes",
                "evidence": "No T3-v1 physician pack is admissible and no T3-v2 physician labels exist.",
            },
        ],
        "interpretation": "Evaluation engineering advanced, but neither generation qualification nor source availability is clinical efficacy. The paper remains fail-closed.",
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--rag-audit", type=Path, required=True)
    parser.add_argument("--v1-failure", type=Path, required=True)
    parser.add_argument("--v2-execution", type=Path, required=True)
    parser.add_argument("--v2-provenance", type=Path, required=True)
    parser.add_argument("--v1-job-state", type=Path, required=True)
    parser.add_argument("--v2-job-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        base_audit=args.base_audit,
        rag_audit=args.rag_audit,
        v1_failure=args.v1_failure,
        v2_execution=args.v2_execution,
        v2_provenance=args.v2_provenance,
        v1_job_state=args.v1_job_state,
        v2_job_state=args.v2_job_state,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
