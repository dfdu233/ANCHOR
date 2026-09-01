#!/usr/bin/env python3
"""Outcome-blind assembler for an explicit human listing admission decision.

The assembler does not infer equivalence from rows.  It validates completed
human adjudication, copies the attested top-level ``admit``/``reject`` decision,
and hash-binds every human/upstream/source artifact.  Reject creates a terminal
non-authorizing receipt; only explicit admit can create an authorizing receipt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from corrected_sgta.build_vindr_cecd_listing_admission_pack_v1 import ROLES
from corrected_sgta.prepare_vindr_reader_manifest import sha256_file
from corrected_sgta.validate_vindr_cecd_listing_scientific_admission_v1 import (
    SOURCE as VALIDATOR_SOURCE,
    VERSION,
    file_record,
    validate_admit_eligibility,
    validate_human_evidence,
    validate_upstream_binary_ce,
)


SOURCE = Path(__file__).resolve()


class AdmissionAssemblerError(RuntimeError):
    pass


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    expected = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise AdmissionAssemblerError(f"write-once admission collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(expected)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def evidence_records(handoff_path: Path) -> dict[str, Any]:
    root = handoff_path.parent
    return {
        "frozen_returns": {
            role: {
                "completed": file_record(root / "frozen_returns" / f"{role}.completed.csv"),
                "attestation": file_record(root / "frozen_returns" / f"{role}.attestation.json"),
            }
            for role in ROLES
        },
        "clinical_adjudication_completed": file_record(
            root / "clinical_adjudication.completed.csv"
        ),
        "prompt_adjudication_completed": file_record(
            root / "prompt_adjudication.completed.csv"
        ),
        "adjudicator_attestation": file_record(
            root / "adjudicator.attestation.completed.json"
        ),
    }


def assemble_receipt(
    *, handoff_path: Path, expected_handoff_sha256: str,
    upstream_gate_path: Path, expected_upstream_gate_sha256: str,
    pack_manifest_path: Path, experiment_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    if sha256_file(handoff_path) != expected_handoff_sha256:
        raise AdmissionAssemblerError("adjudication handoff does not match pinned hash")
    evidence = evidence_records(handoff_path)
    validate_human_evidence(handoff_path=handoff_path, evidence=evidence)
    upstream = validate_upstream_binary_ce(
        input_gate_path=upstream_gate_path,
        expected_input_gate_sha256=expected_upstream_gate_sha256,
    )
    attestation = json.loads(
        Path(evidence["adjudicator_attestation"]["path"]).read_text(encoding="utf-8")
    )
    decision = attestation["human_admission_decision"]
    admitted = decision == "admit"
    if admitted:
        validate_admit_eligibility(
            clinical_completed=Path(evidence["clinical_adjudication_completed"]["path"]),
            prompt_completed=Path(evidence["prompt_adjudication_completed"]["path"]),
            pack_manifest_path=pack_manifest_path,
        )
    experiment = json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
    pack = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    receipt = {
        "schema_version": VERSION,
        "status": (
            "independently_admitted_for_model_scoring"
            if admitted
            else "human_adjudication_rejected_terminal"
        ),
        "four_independent_human_returns_validated": True,
        "listing_render_equivalence_admitted": admitted,
        "listing_prompt_equivalence_admitted": admitted,
        "adjudication_complete": True,
        "human_admission_decision": decision,
        "upstream_binary_ce_gate_authorized": True,
        "upstream_binary_ce_authorization_sha256": expected_upstream_gate_sha256,
        "model_scoring_authorized": admitted,
        "gpu_authorized": admitted,
        "model_outputs_read_for_admission": False,
        "authorized_model_ids": ["huatuo", "hulu"] if admitted else [],
        "pack_manifest_sha256": sha256_file(pack_manifest_path),
        "experiment_manifest_sha256": sha256_file(experiment_manifest_path),
        "reference_file_sha256": experiment.get("reference_contract", {}).get(
            "reference_file_sha256"
        ),
        "computational_guard_failure_pair_ids_sha256": pack.get(
            "clinical_review", {}
        ).get("computational_guard_failure_pair_ids_sha256"),
        "adjudication_handoff": file_record(handoff_path),
        "human_evidence": evidence,
        "admission_validator_source": file_record(VALIDATOR_SOURCE),
        "admission_assembler_source": file_record(SOURCE),
        "upstream_binary_ce": {
            "input_gate": file_record(upstream_gate_path),
            "confirmation_locked": file_record(upstream["confirmation_path"]),
        },
    }
    _write_once(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--expected-handoff-sha256", required=True)
    parser.add_argument("--upstream-gate", type=Path, required=True)
    parser.add_argument("--expected-upstream-gate-sha256", required=True)
    parser.add_argument("--pack-manifest", type=Path, required=True)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble_receipt(
        handoff_path=args.handoff,
        expected_handoff_sha256=args.expected_handoff_sha256,
        upstream_gate_path=args.upstream_gate,
        expected_upstream_gate_sha256=args.expected_upstream_gate_sha256,
        pack_manifest_path=args.pack_manifest,
        experiment_manifest_path=args.experiment_manifest,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
