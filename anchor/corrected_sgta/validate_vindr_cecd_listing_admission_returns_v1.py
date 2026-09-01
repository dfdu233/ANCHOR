#!/usr/bin/env python3
"""Validate future independent VinDr listing-admission returns and attestations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from corrected_sgta.build_vindr_cecd_listing_admission_pack_v1 import (
    ALLOWED,
    CLINICAL_DECISION_FIELDS,
    PROFESSIONAL_ROLE,
    PROMPT_DECISION_FIELDS,
    ROLES,
    TARGET_FINDINGS,
    VERSION as PACK_VERSION,
)
from corrected_sgta.verify_vindr_cecd_listing_admission_pack_v1 import verify


VERSION = "vindr-cecd-listing-admission-return-validation-v1"
FORMULA_PREFIXES = ("=", "+", "-", "@")


@dataclass(frozen=True)
class ValidatedReturn:
    role: str
    reviewer_id: str
    rows: int
    completed_at_utc: str


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return list(reader.fieldnames), list(reader)


def validate_return(
    *, pack_dir: Path, role: str, completed_path: Path, attestation_path: Path
) -> ValidatedReturn:
    if role not in ROLES:
        raise ValueError(f"unknown role {role}")
    template_path = pack_dir / f"{role}.csv"
    template_header, template_rows = _read_csv(template_path)
    header, rows = _read_csv(completed_path)
    if header != template_header or len(rows) != len(template_rows):
        raise ValueError(f"{role}: header or row count changed")
    clinical = role.startswith("clinical_reviewer_")
    decisions = CLINICAL_DECISION_FIELDS if clinical else PROMPT_DECISION_FIELDS
    immutable = [field for field in header if field not in decisions]
    id_field = "pair_id" if clinical else "item_id"
    if [row[id_field] for row in rows] != [row[id_field] for row in template_rows]:
        raise ValueError(f"{role}: row order or IDs changed")
    allowed_findings = {finding for finding, _ in TARGET_FINDINGS}
    for index, (template, row) in enumerate(zip(template_rows, rows), 2):
        for field in immutable:
            if row.get(field, "") != template.get(field, ""):
                raise ValueError(f"{role}: immutable field changed at row {index}: {field}")
        for field in decisions:
            value = row.get(field, "")
            if field == "comments":
                if value.lstrip().startswith(FORMULA_PREFIXES):
                    raise ValueError(f"{role}: formula-like comment at row {index}")
            elif field == "changed_finding_ids":
                values = [item.strip() for item in value.split(";") if item.strip()]
                if len(values) != len(set(values)) or not set(values) <= allowed_findings:
                    raise ValueError(f"{role}: invalid changed_finding_ids at row {index}")
                support = row["same_support_state_for_all_14"]
                if support == "no" and not values:
                    raise ValueError(f"{role}: changed finding required at row {index}")
                if support != "no" and values:
                    raise ValueError(f"{role}: changed finding must be empty at row {index}")
            elif value not in ALLOWED[field]:
                raise ValueError(f"{role}: invalid {field} at row {index}")
        primary = (
            (
                "same_support_state_for_all_14",
                "visibility_change",
                "listing_interchangeable",
            )
            if clinical
            else (
                "same_target_ontology",
                "same_inclusion_obligation",
                "same_speech_act",
                "same_certainty_demand",
                "same_answer_space",
                "same_output_grammar",
            )
        )
        unable = any(row[field] == "unable" for field in primary)
        if unable != (row["unable_to_judge"] == "yes"):
            raise ValueError(f"{role}: unable_to_judge inconsistent at row {index}")

    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    if set(payload) != {"protocol_id", "review_role", "reviewer"}:
        raise ValueError(f"{role}: attestation top-level keys changed")
    if payload["protocol_id"] != PACK_VERSION or payload["review_role"] != role:
        raise ValueError(f"{role}: attestation protocol/role mismatch")
    record = payload["reviewer"]
    keys = {
        "reviewer_id",
        "professional_role",
        "independent_review",
        "blinded_to_sealed_mapping",
        "completed_at_utc",
    }
    if not isinstance(record, dict) or set(record) != keys:
        raise ValueError(f"{role}: attestation reviewer keys changed")
    reviewer_id = record["reviewer_id"]
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError(f"{role}: reviewer_id must be nonempty")
    if record["professional_role"] != PROFESSIONAL_ROLE[role]:
        raise ValueError(f"{role}: professional role mismatch")
    if record["independent_review"] is not True:
        raise ValueError(f"{role}: independent_review must be true")
    if record["blinded_to_sealed_mapping"] is not True:
        raise ValueError(f"{role}: blinded_to_sealed_mapping must be true")
    completed_at = record["completed_at_utc"]
    if not isinstance(completed_at, str) or not completed_at.strip():
        raise ValueError(f"{role}: completed_at_utc must be nonempty")
    parsed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{role}: completed_at_utc needs timezone")
    return ValidatedReturn(role, reviewer_id.strip(), len(rows), completed_at)


def validate_all(
    *, pack_dir: Path, completed: dict[str, Path], attestations: dict[str, Path]
) -> dict[str, Any]:
    verify(pack_dir)
    if set(completed) != set(ROLES) or set(attestations) != set(ROLES):
        raise ValueError("completed and attestation maps must contain exactly four roles")
    results = [
        validate_return(
            pack_dir=pack_dir,
            role=role,
            completed_path=completed[role],
            attestation_path=attestations[role],
        )
        for role in ROLES
    ]
    ids = [row.reviewer_id for row in results]
    if len(ids) != len(set(ids)):
        raise ValueError("four roles require four distinct reviewer IDs")
    return {
        "version": VERSION,
        "protocol_id": PACK_VERSION,
        "status": "four_independent_returns_structurally_valid",
        "roles": [
            {
                "role": row.role,
                "reviewer_id": row.reviewer_id,
                "rows": row.rows,
                "completed_at_utc": row.completed_at_utc,
            }
            for row in results
        ],
        "clinical_or_prompt_labels_synthesized": False,
        "attestations_synthesized": False,
        "admission_decision_computed": False,
        "model_or_gpu_authorized": False,
    }


def _role_paths(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or role in output:
            raise ValueError(f"expected unique role=/path, got {value!r}")
        output[role] = Path(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--completed", action="append", default=[])
    parser.add_argument("--attestation", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = validate_all(
        pack_dir=args.pack_dir,
        completed=_role_paths(args.completed),
        attestations=_role_paths(args.attestation),
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
