#!/usr/bin/env python3
"""Validate four CECD v3 human returns and explicit role attestations."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from anchor.corrected_sgta.build_cecd_reviewer_deliveries_v1 import (
    CLINICAL_FIELDS,
    LANGUAGE_FIELDS,
    ROLES,
    SOURCE_VERSION,
)
from anchor.medeval.package_cecd_deliveries_v3 import ALLOWED, PROFESSIONAL_ROLE


FORMULA_PREFIXES = ("=", "+", "-", "@")
VERSION = "cecd-blinded-human-return-validation-v3"


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
    *,
    pack_dir: Path,
    role: str,
    completed_path: Path,
    attestation_path: Path,
) -> ValidatedReturn:
    if role not in ROLES:
        raise ValueError(f"unknown CECD reviewer role: {role}")
    spec = ROLES[role]
    template_header, template_rows = _read_csv(pack_dir / spec["sheet"])
    header, rows = _read_csv(completed_path)
    if header != template_header:
        raise ValueError(f"{role}: CSV header changed")
    if len(rows) != len(template_rows):
        raise ValueError(f"{role}: row count changed")
    decision_fields = CLINICAL_FIELDS if spec["kind"] == "clinical" else LANGUAGE_FIELDS
    immutable = [field for field in header if field not in decision_fields]
    id_field = "pair_id" if spec["kind"] == "clinical" else "item_id"
    if [row[id_field] for row in rows] != [row[id_field] for row in template_rows]:
        raise ValueError(f"{role}: row order or IDs changed")
    for index, (template, row) in enumerate(zip(template_rows, rows), start=2):
        for field in immutable:
            if row.get(field, "") != template.get(field, ""):
                raise ValueError(f"{role}: immutable field changed at row {index}: {field}")
        for field in decision_fields:
            if field == "comments":
                if row.get(field, "").lstrip().startswith(FORMULA_PREFIXES):
                    raise ValueError(f"{role}: formula-like comments at row {index}")
            elif row.get(field, "") not in ALLOWED[field]:
                raise ValueError(f"{role}: invalid {field} at row {index}")
        if spec["kind"] == "clinical":
            any_unable = any(
                row[field] == "unable"
                for field in (
                    "support_state_same_supported_refuted_undetermined",
                    "lesion_visibility",
                    "clinically_interchangeable",
                )
            )
            if any_unable != (row["unable_to_judge"] == "yes"):
                raise ValueError(f"{role}: inconsistent unable_to_judge at row {index}")

    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    if set(payload) != {"protocol_id", "review_role", "reviewer"}:
        raise ValueError(f"{role}: attestation top-level keys differ")
    if payload["protocol_id"] != SOURCE_VERSION or payload["review_role"] != role:
        raise ValueError(f"{role}: attestation protocol or role mismatch")
    record = payload["reviewer"]
    expected_keys = {
        "reviewer_id",
        "professional_role",
        "independent_review",
        "blinded_to_sealed_mapping",
        "completed_at_utc",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError(f"{role}: reviewer attestation keys differ")
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
    try:
        parsed_time = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{role}: completed_at_utc must be ISO-8601") from error
    if parsed_time.tzinfo is None:
        raise ValueError(f"{role}: completed_at_utc must include a timezone")
    return ValidatedReturn(
        role=role,
        reviewer_id=reviewer_id.strip(),
        rows=len(rows),
        completed_at_utc=completed_at,
    )


def validate_all(
    *,
    pack_dir: Path,
    completed: dict[str, Path],
    attestations: dict[str, Path],
) -> dict[str, Any]:
    if set(completed) != set(ROLES) or set(attestations) != set(ROLES):
        raise ValueError("CECD return maps must contain exactly the four frozen roles")
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
        raise ValueError("the four CECD review roles require distinct reviewer IDs")
    return {
        "version": VERSION,
        "protocol_id": SOURCE_VERSION,
        "status": "four_independent_returns_validated",
        "roles": [
            {
                "role": row.role,
                "reviewer_id": row.reviewer_id,
                "rows": row.rows,
                "completed_at_utc": row.completed_at_utc,
            }
            for row in results
        ],
        "clinical_or_language_labels_synthesized": False,
        "attestations_synthesized": False,
    }
