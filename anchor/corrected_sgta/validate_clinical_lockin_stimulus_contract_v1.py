#!/usr/bin/env python3
"""Exact-string construct audit for Clinical Autoregressive Lock-in stimuli.

This validator deliberately does *not* infer grammaticality.  It exposes every
serialized stimulus exactly, verifies pre-registration and provenance, and
requires a completed independent construct review before a future v5 manifest
can be considered by a GPU runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


AUDIT_PROTOCOL_ID = "clinical-lockin-exact-string-construct-audit-v1"
REJECTED_LEGACY_PROTOCOL = "clinical-autoregressive-lockin-manifest-v4-claim-specific-prompt"
FUTURE_PROTOCOL = "clinical-autoregressive-lockin-manifest-v5-natural-tokenwise"
FUTURE_MODE = "single_natural_full_sequence_tokenwise_visual_residual"


class ConstructContractError(RuntimeError):
    """The stimulus contract is absent, stale, or not independently approved."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def audit_legacy_v4(manifest: Path, metadata: Path) -> dict[str, Any]:
    """Expose the exact invalid concatenations and reject the entire orbit."""

    meta = json.loads(metadata.read_text())
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if meta.get("manifest_protocol_id") != REJECTED_LEGACY_PROTOCOL:
        raise ConstructContractError("legacy audit was given a non-v4 manifest")
    if _sha_file(manifest) != meta.get("manifest_sha256"):
        raise ConstructContractError("legacy manifest/metadata hash mismatch")
    stimuli: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        for step in row["prefix_ladder"]:
            stimuli[(str(row["finding"]), int(step["step"]))].add(
                str(step["prefix"]) + str(row["embedded_claim"])
            )
    exact = []
    for (finding, step), values in sorted(stimuli.items()):
        if len(values) != 1:
            raise ConstructContractError("one legacy finding/step has multiple exact stimuli")
        stimulus = values.pop()
        exact.append(
            {
                "finding": finding,
                "step": step,
                "exact_concatenation": stimulus,
                "exact_concatenation_sha256": _sha(stimulus.encode()),
                "independently_preregistered_complete_stimulus": False,
                "human_naturalness_review": "missing",
                "human_proposition_control_review": "missing",
            }
        )
    return {
        "audit_protocol_id": AUDIT_PROTOCOL_ID,
        "status": "rejected_f6_construct_invalid",
        "gpu_authorized": False,
        "manifest_protocol_id": meta["manifest_protocol_id"],
        "manifest_sha256": meta["manifest_sha256"],
        "metadata_sha256": _sha_file(metadata),
        "validator_source_sha256": _sha_file(Path(__file__)),
        "decision_basis": (
            "Every prefix+continuation was assembled post hoc rather than admitted as a complete "
            "natural stimulus. Exact strings reveal grammatical and discourse-boundary changes. "
            "No automatic grammar judgment is used; missing independent construct admission alone "
            "is sufficient for fail-closed rejection."
        ),
        "exact_serialized_stimuli": exact,
        "forbidden_interpretation": (
            "early-versus-late likelihood differences cannot be called autoregressive lock-in"
        ),
        "required_successor": FUTURE_PROTOCOL,
    }


def validate_v5_contract(
    contract: Mapping[str, Any],
    *,
    pilot_exact_surfaces: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate exact v5 strings and independent attestation, never grammar."""

    if contract.get("manifest_protocol_id") != FUTURE_PROTOCOL:
        raise ConstructContractError("future stimulus protocol is not frozen v5")
    if contract.get("measurement_mode") != FUTURE_MODE:
        raise ConstructContractError("v5 must trace one natural full sequence tokenwise")
    if contract.get("fixed_continuation_across_manually_assembled_prefixes") is not False:
        raise ConstructContractError("manual prefix+fixed-continuation comparisons are forbidden")
    if contract.get("context_changes_across_token_positions_acknowledged") is not True:
        raise ConstructContractError("v5 must acknowledge that tokenwise contexts differ")
    if contract.get("prefix_policy") != "tokenizer_boundaries_of_one_frozen_full_sequence_only":
        raise ConstructContractError("v5 prefixes must be derived only from one full sequence")
    stimuli = contract.get("stimuli")
    if not isinstance(stimuli, list) or not stimuli:
        raise ConstructContractError("v5 has no full-sequence stimuli")
    seen = set()
    normalized = []
    for row in stimuli:
        finding = str(row.get("finding", ""))
        full_sequence = str(row.get("full_sequence", ""))
        if not finding or finding in seen or not full_sequence:
            raise ConstructContractError("v5 stimulus finding is empty or duplicated")
        seen.add(finding)
        if row.get("full_sequence_sha256") != _sha(full_sequence.encode()):
            raise ConstructContractError(f"v5 exact full-sequence hash mismatch: {finding}")
        pilot = pilot_exact_surfaces.get(finding)
        if not pilot or full_sequence != pilot.get("text"):
            raise ConstructContractError(
                f"v5 full sequence is not the frozen exact pilot surface: {finding}"
            )
        if row.get("pilot_full_sequence_sha256") != pilot.get("text_sha256"):
            raise ConstructContractError(f"v5 pilot provenance hash mismatch: {finding}")
        if row.get("manual_prefixes") not in (None, []):
            raise ConstructContractError("v5 cannot contain manually authored prefix stimuli")
        normalized.append(
            {
                "finding": finding,
                "full_sequence_sha256": row["full_sequence_sha256"],
                "prefix_derivation": "runtime tokenizer boundaries only",
            }
        )
    controls = contract.get("natural_control_sequences")
    if not isinstance(controls, list) or not controls:
        raise ConstructContractError("v5 natural token-position control sequences are missing")
    control_findings = set()
    normalized_controls = []
    for row in controls:
        finding = str(row.get("finding", ""))
        full_sequence = str(row.get("full_sequence", ""))
        if not finding or finding in control_findings or finding not in seen or not full_sequence:
            raise ConstructContractError("v5 natural control finding is missing, duplicated, or unmatched")
        control_findings.add(finding)
        if row.get("role") != "token_position_matched_non_target_natural_control":
            raise ConstructContractError("v5 natural control role is not frozen")
        if row.get("full_sequence_sha256") != _sha(full_sequence.encode()):
            raise ConstructContractError(f"v5 natural control hash mismatch: {finding}")
        if row.get("manual_prefixes") not in (None, []):
            raise ConstructContractError("v5 natural controls cannot contain manual prefixes")
        normalized_controls.append(
            {
                "finding": finding,
                "full_sequence_sha256": row["full_sequence_sha256"],
                "role": row["role"],
            }
        )
    if control_findings != seen:
        raise ConstructContractError("v5 needs one natural control sequence per finding")
    review = contract.get("independent_construct_review")
    if not isinstance(review, dict):
        raise ConstructContractError("independent construct review is missing")
    required_review = {
        "reviewer_id",
        "reviewer_role",
        "reviewed_contract_sha256",
        "all_full_sequences_natural",
        "proposition_leakage_assessed",
        "token_position_context_change_accepted",
        "approved_for_mechanistic_dev_only",
        "attestation",
    }
    if required_review - review.keys():
        raise ConstructContractError("independent construct review is incomplete")
    review_free = {key: value for key, value in contract.items() if key != "independent_construct_review"}
    if review["reviewed_contract_sha256"] != _sha(_canonical(review_free)):
        raise ConstructContractError("construct reviewer signed a different v5 contract")
    if not str(review["reviewer_id"]).strip() or not str(review["attestation"]).strip():
        raise ConstructContractError("construct reviewer identity/attestation is blank")
    if review["reviewer_role"] not in {"clinical_language_reviewer", "radiologist"}:
        raise ConstructContractError("construct reviewer role is not admissible")
    boolean_fields = (
        "all_full_sequences_natural",
        "proposition_leakage_assessed",
        "token_position_context_change_accepted",
        "approved_for_mechanistic_dev_only",
    )
    if any(review[field] is not True for field in boolean_fields):
        raise ConstructContractError("independent construct review did not approve every gate")
    return {
        "audit_protocol_id": AUDIT_PROTOCOL_ID,
        "status": "construct_admitted_cpu_only",
        "gpu_authorized": False,
        "reason_gpu_still_false": "tokenwise v5 model runtime has not been implemented or conformance-tested",
        "measurement_mode": FUTURE_MODE,
        "stimuli": normalized,
        "natural_control_sequences": normalized_controls,
        "reviewer_id": review["reviewer_id"],
        "reviewed_contract_sha256": review["reviewed_contract_sha256"],
        "validator_source_sha256": _sha_file(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--legacy-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_legacy_v4(args.legacy_manifest, args.legacy_metadata)
    _atomic_write(
        args.output,
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
