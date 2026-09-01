#!/usr/bin/env python3
"""Validate a human-produced listing admission and canonical binary-CE GO.

This validator never derives an equivalence decision.  It only verifies that
the externally supplied decision is bound to the four frozen returns, completed
human adjudication, its attestation, this validator source, and the exact
canonical three-stage GO plus locked-confirmation artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from corrected_sgta.build_vindr_cecd_listing_admission_pack_v1 import (
    ALLOWED,
    CLINICAL_DECISION_FIELDS,
    PROMPT_DECISION_FIELDS,
    ROLES,
)
from corrected_sgta.analyze_clinical_equivalence_composition_defect_v1 import (
    CONFIRMATION_VERSION,
)
from corrected_sgta.cecd_admission_gate import (
    EXPECTED_VERSION as BINARY_CE_ADMISSION_VERSION,
)
from corrected_sgta.prepare_vindr_cecd_listing_adjudication_handoff_v1 import (
    CLINICAL_FINAL_FIELDS,
    PROMPT_FINAL_FIELDS,
    VERSION as HANDOFF_VERSION,
    canonical_sha256,
)
from corrected_sgta.prepare_vindr_reader_manifest import sha256_file
from corrected_sgta.verify_cecd_three_stage_v3 import (
    VERSION as THREE_STAGE_VERSION,
)


VERSION = "vindr-cecd-listing-scientific-admission-v1"
SOURCE = Path(__file__).resolve()
ROOT = Path("/home/dbw/ANCHOR")
UPSTREAM_GATE_RELATIVE = Path("corrected_runs/vindr_v2/cecd_three_stage_v3/input_gate.json")
UPSTREAM_CONFIRMATION_RELATIVE = Path(
    "corrected_runs/vindr_v2/cecd_three_stage_v3/confirmation_locked.json"
)
UPSTREAM_DEV_FIT_RELATIVE = Path("corrected_runs/vindr_v2/cecd_three_stage_v3/dev_fit.json")
UPSTREAM_ADMISSION_RELATIVE = Path(
    "corrected_runs/vindr_v2/cecd_human_admission_v2/analysis.json"
)
EXPECTED_SELECTION_HASHES = {
    "pilot_screen": "276bac3ffe3f06e47e6377f3dcc2b5877959a9ad372cd1c5801629719051a24a",
    "dev_fit": "2e9b0b0c427068e017a5ce1fbc098dbe7028bfacfa3604d395aa782807e57420",
    "confirmation_locked": "39195d0f606da9acfa1b2b2de413176496efa6e4e235c09913c417a95c6bd1e9",
}
HEX64 = re.compile(r"[0-9a-f]{64}")


class ScientificAdmissionError(RuntimeError):
    pass


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ScientificAdmissionError(f"{label} must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScientificAdmissionError(f"cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ScientificAdmissionError(f"{label} must be a JSON object")
    return payload


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ScientificAdmissionError(f"required regular file is absent: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def validate_file_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
        raise ScientificAdmissionError(f"{label} file-record schema drift")
    path = Path(str(record["path"]))
    if not path.is_absolute() or file_record(path) != dict(record):
        raise ScientificAdmissionError(f"{label} file record/hash mismatch")
    return path.resolve()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ScientificAdmissionError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _validate_completed_sheet(
    *, template: Path, completed: Path, final_fields: Sequence[str], clinical: bool
) -> None:
    template_header, template_rows = _read_csv(template)
    header, rows = _read_csv(completed)
    if header != template_header or len(rows) != len(template_rows):
        raise ScientificAdmissionError("completed adjudication header/row count drift")
    immutable = [field for field in header if field not in final_fields]
    for line, (source, row) in enumerate(zip(template_rows, rows), 2):
        if any(row[field] != source[field] for field in immutable):
            raise ScientificAdmissionError(f"adjudication immutable field changed at row {line}")
        if not row["adjudication_rationale"].strip():
            raise ScientificAdmissionError(f"adjudication rationale missing at row {line}")
        for final in final_fields:
            if final == "adjudication_rationale":
                continue
            source_field = final.removeprefix("adjudicated_")
            if source_field == "changed_finding_ids":
                continue
            if row[final] not in ALLOWED[source_field]:
                raise ScientificAdmissionError(f"invalid final adjudication at row {line}: {final}")
        if clinical:
            support = row["adjudicated_same_support_state_for_all_14"]
            changed = [value.strip() for value in row["adjudicated_changed_finding_ids"].split(";") if value.strip()]
            if (support == "no") != bool(changed):
                raise ScientificAdmissionError(f"changed-finding adjudication inconsistent at row {line}")
        unable_fields = (
            (
                "adjudicated_same_support_state_for_all_14",
                "adjudicated_visibility_change",
                "adjudicated_listing_interchangeable",
            )
            if clinical
            else tuple(
                f"adjudicated_{field}"
                for field in PROMPT_DECISION_FIELDS
                if field not in {"unable_to_judge", "comments"}
            )
        )
        expected_unable = "yes" if any(row[field] == "unable" for field in unable_fields) else "no"
        if row["adjudicated_unable_to_judge"] != expected_unable:
            raise ScientificAdmissionError(f"unable adjudication inconsistent at row {line}")


def validate_human_evidence(
    *, handoff_path: Path, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    handoff = _load_object(handoff_path, "adjudication handoff")
    fingerprint = handoff.get("fingerprint")
    if (
        handoff.get("schema_version") != HANDOFF_VERSION
        or handoff.get("status") != "ready_for_human_adjudication"
        or fingerprint
        != canonical_sha256({key: value for key, value in handoff.items() if key != "fingerprint"})
        or handoff.get("admission_receipt_created") is not False
    ):
        raise ScientificAdmissionError("adjudication handoff contract mismatch")
    expected_keys = {
        "frozen_returns",
        "clinical_adjudication_completed",
        "prompt_adjudication_completed",
        "adjudicator_attestation",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_keys:
        raise ScientificAdmissionError("human evidence schema drift")

    frozen = evidence["frozen_returns"]
    if not isinstance(frozen, Mapping) or set(frozen) != set(ROLES):
        raise ScientificAdmissionError("frozen-return role closure drift")
    inventory = {
        str(row["path"]): row for row in handoff.get("handoff_inventory", [])
        if isinstance(row, Mapping)
    }
    for role in ROLES:
        row = frozen[role]
        if not isinstance(row, Mapping) or set(row) != {"completed", "attestation"}:
            raise ScientificAdmissionError(f"{role}: evidence record schema drift")
        for kind, suffix in (("completed", ".completed.csv"), ("attestation", ".attestation.json")):
            path = validate_file_record(row[kind], label=f"{role}:{kind}")
            relative = f"frozen_returns/{role}{suffix}"
            expected = inventory.get(relative)
            if not isinstance(expected, Mapping) or (
                expected.get("sha256") != sha256_file(path)
                or expected.get("bytes") != path.stat().st_size
                or path != (handoff_path.parent / relative).resolve()
            ):
                raise ScientificAdmissionError(f"{role}:{kind} is not the handoff-frozen file")

    clinical = validate_file_record(
        evidence["clinical_adjudication_completed"], label="clinical adjudication"
    )
    prompt = validate_file_record(
        evidence["prompt_adjudication_completed"], label="prompt adjudication"
    )
    attestation_path = validate_file_record(
        evidence["adjudicator_attestation"], label="adjudicator attestation"
    )
    _validate_completed_sheet(
        template=handoff_path.parent / "clinical_adjudication.template.csv",
        completed=clinical,
        final_fields=CLINICAL_FINAL_FIELDS,
        clinical=True,
    )
    _validate_completed_sheet(
        template=handoff_path.parent / "prompt_adjudication.template.csv",
        completed=prompt,
        final_fields=PROMPT_FINAL_FIELDS,
        clinical=False,
    )
    attestation = _load_object(attestation_path, "adjudicator attestation")
    if set(attestation) != {
        "schema_version", "handoff_fingerprint", "human_admission_decision", "adjudicators"
    }:
        raise ScientificAdmissionError("adjudicator attestation schema drift")
    if attestation["schema_version"] != VERSION or attestation["handoff_fingerprint"] != fingerprint:
        raise ScientificAdmissionError("adjudicator attestation binding mismatch")
    if attestation["human_admission_decision"] not in {"admit", "reject"}:
        raise ScientificAdmissionError("human admission decision must be explicit admit or reject")
    adjudicators = attestation["adjudicators"]
    if not isinstance(adjudicators, list) or len(adjudicators) != 2:
        raise ScientificAdmissionError("clinical and prompt adjudicators are both required")
    reviewer_ids = {
        str(row["reviewer_id"])
        for row in handoff.get("validated_return_summary", {}).get("roles", [])
    }
    scopes: set[str] = set()
    ids: list[str] = []
    for row in adjudicators:
        expected = {
            "scope",
            "adjudicator_id",
            "professional_role",
            "independent_adjudication",
            "blinded_to_model_outputs",
            "completed_at_utc",
        }
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ScientificAdmissionError("adjudicator record schema drift")
        scope = str(row["scope"])
        adjudicator_id = str(row["adjudicator_id"]).strip()
        if scope not in {"clinical", "prompt"} or not adjudicator_id:
            raise ScientificAdmissionError("adjudicator scope/identity invalid")
        if row["independent_adjudication"] is not True or row["blinded_to_model_outputs"] is not True:
            raise ScientificAdmissionError("adjudicator independence/blinding invalid")
        if scope == "clinical" and row["professional_role"] != "physician":
            raise ScientificAdmissionError("clinical adjudicator must be a physician")
        if scope == "prompt" and row["professional_role"] not in {"physician", "language_expert"}:
            raise ScientificAdmissionError("prompt adjudicator role invalid")
        completed = datetime.fromisoformat(str(row["completed_at_utc"]).replace("Z", "+00:00"))
        if completed.tzinfo is None:
            raise ScientificAdmissionError("adjudicator completion needs timezone")
        scopes.add(scope)
        ids.append(adjudicator_id)
    if scopes != {"clinical", "prompt"} or len(ids) != len(set(ids)) or set(ids) & reviewer_ids:
        raise ScientificAdmissionError("adjudicators must be distinct from each other and all reviewers")
    return handoff


def validate_admit_eligibility(
    *, clinical_completed: Path, prompt_completed: Path, pack_manifest_path: Path
) -> None:
    """Check that an explicit human ``admit`` is row-wise self-consistent."""

    pack = _load_object(pack_manifest_path, "listing pack manifest")
    failure_hash = pack.get("clinical_review", {}).get(
        "computational_guard_failure_pair_ids_sha256"
    )
    exempt_pairs: set[str] = set()
    if failure_hash is not None:
        mapping_path = pack_manifest_path.parent / "sealed_mapping.json"
        expected_mapping = pack.get("artifact_sha256", {}).get("sealed_mapping.json")
        if (
            not mapping_path.is_file()
            or sha256_file(mapping_path) != expected_mapping
        ):
            raise ScientificAdmissionError("sealed computational-guard mapping hash drift")
        mapping = _load_object(mapping_path, "sealed computational-guard mapping")
        exempt_pairs = {
            str(row["pair_id"])
            for row in mapping.get("clinical_pairs", [])
            if row.get("transform_guard", {}).get("clinical_guard_pass") is False
        }
        if canonical_sha256(sorted(exempt_pairs)) != failure_hash:
            raise ScientificAdmissionError("computational-guard exemption identity drift")

    _, clinical_rows = _read_csv(clinical_completed)
    observed_pairs = {str(row.get("pair_id", "")) for row in clinical_rows}
    if not exempt_pairs <= observed_pairs:
        raise ScientificAdmissionError("guard-exempt clinical pair is absent from adjudication")
    for line, row in enumerate(clinical_rows, 2):
        if str(row.get("pair_id", "")) in exempt_pairs:
            continue
        required = (
            row.get("adjudicated_same_support_state_for_all_14") == "yes"
            and row.get("adjudicated_visibility_change") == "unchanged"
            and row.get("adjudicated_listing_interchangeable") == "yes"
            and row.get("adjudicated_changed_finding_ids") == ""
            and row.get("adjudicated_unable_to_judge") == "no"
        )
        if not required:
            raise ScientificAdmissionError(
                f"human admit conflicts with clinical adjudication at row {line}"
            )
    _, prompt_rows = _read_csv(prompt_completed)
    required_prompt_fields = [
        f"adjudicated_{field}"
        for field in PROMPT_DECISION_FIELDS
        if field not in {"unable_to_judge", "comments"}
    ]
    for line, row in enumerate(prompt_rows, 2):
        if (
            any(row.get(field) != "yes" for field in required_prompt_fields)
            or row.get("adjudicated_unable_to_judge") != "no"
        ):
            raise ScientificAdmissionError(
                f"human admit conflicts with prompt adjudication at row {line}"
            )


def validate_upstream_binary_ce(
    *, input_gate_path: Path, expected_input_gate_sha256: str, root: Path = ROOT
) -> dict[str, Any]:
    if not HEX64.fullmatch(str(expected_input_gate_sha256)):
        raise ScientificAdmissionError("expected upstream gate hash is not SHA-256")
    canonical_gate = (root / UPSTREAM_GATE_RELATIVE).resolve()
    canonical_confirmation = (root / UPSTREAM_CONFIRMATION_RELATIVE).resolve()
    canonical_dev = (root / UPSTREAM_DEV_FIT_RELATIVE).resolve()
    canonical_admission = (root / UPSTREAM_ADMISSION_RELATIVE).resolve()
    if input_gate_path.resolve() != canonical_gate:
        raise ScientificAdmissionError("upstream input gate is not the frozen canonical path")
    if sha256_file(input_gate_path) != expected_input_gate_sha256:
        raise ScientificAdmissionError("canonical upstream input-gate hash mismatch")
    gate = _load_object(input_gate_path, "canonical three-stage input gate")
    confirmation_record = gate.get("confirmation_locked")
    if (
        gate.get("version") != THREE_STAGE_VERSION
        or gate.get("status") != "passed"
        or gate.get("passed") is not True
        or gate.get("authorized_for_method_level_treble_adapter_run") is not True
        or not isinstance(confirmation_record, Mapping)
        or confirmation_record.get("behavioral_gate_passed") is not True
    ):
        raise ScientificAdmissionError("canonical three-stage input gate is not a binary CE GO")
    confirmation_path = validate_file_record(
        {key: confirmation_record[key] for key in ("path", "sha256")}
        | {"bytes": Path(str(confirmation_record["path"])).stat().st_size},
        label="locked confirmation",
    )
    if confirmation_path != canonical_confirmation:
        raise ScientificAdmissionError("locked confirmation is not the canonical path")
    dev_record = gate.get("dev_fit")
    admission_record = gate.get("admission")
    if (
        not isinstance(dev_record, Mapping)
        or set(dev_record) != {"path", "sha256"}
        or Path(str(dev_record["path"])).resolve() != canonical_dev
        or not canonical_dev.is_file()
        or sha256_file(canonical_dev) != dev_record["sha256"]
        or not isinstance(admission_record, Mapping)
        or set(admission_record) != {"path", "sha256", "version"}
        or Path(str(admission_record["path"])).resolve() != canonical_admission
        or not canonical_admission.is_file()
        or sha256_file(canonical_admission) != admission_record["sha256"]
        or admission_record["version"] != BINARY_CE_ADMISSION_VERSION
    ):
        raise ScientificAdmissionError("upstream dev-fit/human-admission build receipt drift")
    runs = gate.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != set(EXPECTED_SELECTION_HASHES):
        raise ScientificAdmissionError("upstream three-stage run closure drift")
    for stage, selection_hash in EXPECTED_SELECTION_HASHES.items():
        rows = runs[stage]
        if (
            not isinstance(rows, list)
            or len(rows) != 2
            or {row.get("family") for row in rows if isinstance(row, Mapping)}
            != {"huatuo", "hulu"}
            or any(
                not isinstance(row, Mapping)
                or row.get("stage") != stage
                or row.get("selection_keys_sha256") != selection_hash
                or row.get("admission_sha256") != admission_record["sha256"]
                for row in rows
            )
        ):
            raise ScientificAdmissionError(f"upstream {stage} selection/build binding drift")
    confirmation = _load_object(confirmation_path, "locked confirmation")
    method_gate = confirmation.get("gate", {})
    if (
        confirmation.get("version") != CONFIRMATION_VERSION
        or confirmation.get("status") != "complete"
        or confirmation.get("stage_label") != "confirmation_locked"
        or method_gate.get("both_models_pass") is not True
        or method_gate.get("authorized_for_method_level_treble_adapter_run") is not True
    ):
        raise ScientificAdmissionError("locked confirmation is not the canonical two-model GO")
    return {"input_gate": gate, "confirmation": confirmation, "confirmation_path": confirmation_path}


def validate_scientific_admission(
    *,
    receipt_path: Path,
    expected_receipt_sha256: str,
    handoff_path: Path,
    expected_handoff_sha256: str,
    upstream_gate_path: Path,
    expected_upstream_gate_sha256: str,
    pack_manifest_path: Path,
    experiment_manifest_path: Path,
) -> dict[str, Any]:
    for value, label in (
        (expected_receipt_sha256, "receipt"),
        (expected_handoff_sha256, "handoff"),
    ):
        if not HEX64.fullmatch(str(value)):
            raise ScientificAdmissionError(f"expected {label} hash is not SHA-256")
    if sha256_file(receipt_path) != expected_receipt_sha256:
        raise ScientificAdmissionError("listing admission receipt hash mismatch")
    if sha256_file(handoff_path) != expected_handoff_sha256:
        raise ScientificAdmissionError("adjudication handoff hash mismatch")
    receipt = _load_object(receipt_path, "listing admission receipt")
    required = {
        "schema_version", "status", "four_independent_human_returns_validated",
        "listing_render_equivalence_admitted", "listing_prompt_equivalence_admitted",
        "adjudication_complete", "human_admission_decision", "upstream_binary_ce_gate_authorized",
        "upstream_binary_ce_authorization_sha256", "model_scoring_authorized", "gpu_authorized",
        "model_outputs_read_for_admission", "authorized_model_ids", "pack_manifest_sha256",
        "experiment_manifest_sha256", "reference_file_sha256",
        "computational_guard_failure_pair_ids_sha256", "adjudication_handoff",
        "human_evidence", "admission_validator_source", "upstream_binary_ce",
        "admission_assembler_source",
    }
    if set(receipt) != required:
        raise ScientificAdmissionError("listing admission receipt schema drift")
    required_true = (
        "four_independent_human_returns_validated", "listing_render_equivalence_admitted",
        "listing_prompt_equivalence_admitted", "adjudication_complete",
        "upstream_binary_ce_gate_authorized", "model_scoring_authorized", "gpu_authorized",
    )
    if (
        receipt["schema_version"] != VERSION
        or receipt["status"] != "independently_admitted_for_model_scoring"
        or any(receipt[field] is not True for field in required_true)
        or receipt["model_outputs_read_for_admission"] is not False
        or receipt["authorized_model_ids"] != ["huatuo", "hulu"]
        or receipt["human_admission_decision"] != "admit"
    ):
        raise ScientificAdmissionError("listing admission is not a genuine admitted state")
    if validate_file_record(receipt["adjudication_handoff"], label="receipt handoff") != handoff_path.resolve():
        raise ScientificAdmissionError("receipt binds a different adjudication handoff")
    handoff = validate_human_evidence(handoff_path=handoff_path, evidence=receipt["human_evidence"])
    if receipt["human_admission_decision"] == "admit":
        validate_admit_eligibility(
            clinical_completed=Path(
                receipt["human_evidence"]["clinical_adjudication_completed"]["path"]
            ),
            prompt_completed=Path(
                receipt["human_evidence"]["prompt_adjudication_completed"]["path"]
            ),
            pack_manifest_path=pack_manifest_path,
        )
    if validate_file_record(receipt["admission_validator_source"], label="admission validator") != SOURCE:
        raise ScientificAdmissionError("receipt is not bound to the canonical validator source")
    assembler_source = SOURCE.with_name("analyze_vindr_cecd_listing_admission_v1.py")
    if (
        validate_file_record(receipt["admission_assembler_source"], label="admission assembler")
        != assembler_source.resolve()
    ):
        raise ScientificAdmissionError("receipt is not bound to the canonical assembler source")
    upstream = validate_upstream_binary_ce(
        input_gate_path=upstream_gate_path,
        expected_input_gate_sha256=expected_upstream_gate_sha256,
    )
    upstream_record = receipt["upstream_binary_ce"]
    if not isinstance(upstream_record, Mapping) or set(upstream_record) != {"input_gate", "confirmation_locked"}:
        raise ScientificAdmissionError("receipt upstream binding schema drift")
    if (
        validate_file_record(upstream_record["input_gate"], label="receipt upstream gate") != upstream_gate_path.resolve()
        or validate_file_record(upstream_record["confirmation_locked"], label="receipt confirmation")
        != upstream["confirmation_path"]
        or receipt["upstream_binary_ce_authorization_sha256"] != expected_upstream_gate_sha256
    ):
        raise ScientificAdmissionError("receipt does not bind the exact canonical upstream GO")
    if receipt["pack_manifest_sha256"] != sha256_file(pack_manifest_path):
        raise ScientificAdmissionError("receipt pack-manifest hash drift")
    experiment = _load_object(experiment_manifest_path, "listing experiment manifest")
    if (
        receipt["experiment_manifest_sha256"] != sha256_file(experiment_manifest_path)
        or receipt["reference_file_sha256"]
        != experiment.get("reference_contract", {}).get("reference_file_sha256")
    ):
        raise ScientificAdmissionError("receipt experiment/reference hash drift")
    return {"receipt": receipt, "handoff": handoff, "upstream": upstream}
