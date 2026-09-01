#!/usr/bin/env python3
"""Fail-closed validation for the Specificity Ratchet physician adjudication.

This validator never derives clinical truth.  It only verifies that two
independent physician sheets and the blinded adjudication are complete,
schema-valid, mutually joined, and internally coherent before a mechanism
manifest is allowed to exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL_ID = "specificity-ratchet-physician-pack-v2"
REVIEW_FIELDS = (
    "edge_entailment_admitted",
    "parent_visual_support",
    "child_visual_support",
    "increment_observability",
    "logical_scope_preserved",
    "reviewer_confidence",
    "clinical_usefulness_if_backed_off",
    "clinically_harmful_if_wrong",
)
FINAL_FIELDS = (
    "edge_entailment_admitted",
    "parent_visual_support",
    "child_visual_support",
    "increment_observability",
    "logical_scope_preserved",
    "clinical_usefulness_if_backed_off",
    "clinically_harmful_if_wrong",
)
SOURCE_REQUIRED_STATES = {
    "requires_other_view_or_sequence",
    "requires_history_lab_pathology_or_prior",
    "fundamentally_nonvisual_knowledge",
}


class AdjudicationValidationError(ValueError):
    """Raised with all detected integrity failures, without partial admission."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


@dataclass(frozen=True)
class ValidatedAdjudication:
    candidates: tuple[dict[str, Any], ...]
    reviewer_rows: tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]
    final_rows: dict[str, dict[str, str]]
    reviewer_ids: tuple[str, str]
    adjudicator_id: str
    input_sha256: dict[str, str]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AdjudicationValidationError([f"missing CSV header: {path}"])
        return list(reader.fieldnames), list(reader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_value(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return ""
    return str(value)


def _index_rows(
    rows: list[dict[str, str]], expected_ids: list[str], label: str, issues: list[str]
) -> dict[str, dict[str, str]]:
    ids = [row.get("edge_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        issues.append(f"{label}: duplicate edge_id")
    if ids != expected_ids:
        issues.append(f"{label}: row order or edge-id set differs from blinded candidates")
    return {row.get("edge_id", ""): row for row in rows}


def _check_attestations(
    path: Path,
    reviewer_ids: tuple[str, str],
    adjudicator_id: str,
    issues: list[str],
) -> None:
    if not path.is_file():
        issues.append(f"missing physician attestation file: {path}")
        return
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        issues.append(f"invalid physician attestation JSON: {exc}")
        return
    if payload.get("protocol_id") != PROTOCOL_ID:
        issues.append("physician attestation protocol_id mismatch")
    reviewers = payload.get("reviewers")
    if not isinstance(reviewers, list) or len(reviewers) != 2:
        issues.append("physician attestations must contain exactly two reviewers")
        reviewers = []
    attested_ids: list[str] = []
    for record in reviewers:
        if not isinstance(record, dict):
            issues.append("malformed reviewer attestation")
            continue
        attested_ids.append(str(record.get("reviewer_id", "")))
        if record.get("role") != "physician":
            issues.append("every reviewer attestation must state role=physician")
        if record.get("independent_review") is not True:
            issues.append("every reviewer must attest independent_review=true")
        if record.get("blinded_to_private_provenance") is not True:
            issues.append("every reviewer must attest blinded_to_private_provenance=true")
        if not str(record.get("completed_at_utc", "")).strip():
            issues.append("every reviewer attestation needs completed_at_utc")
    if sorted(attested_ids) != sorted(reviewer_ids):
        issues.append("attested reviewer IDs do not match the two completed sheets")
    adjudicator = payload.get("adjudicator")
    if not isinstance(adjudicator, dict):
        issues.append("missing adjudicator physician attestation")
        return
    if str(adjudicator.get("adjudicator_id", "")) != adjudicator_id:
        issues.append("attested adjudicator ID does not match adjudication.csv")
    if adjudicator.get("role") != "physician":
        issues.append("adjudicator attestation must state role=physician")
    if adjudicator.get("blinded_to_private_provenance") is not True:
        issues.append("adjudicator must attest blinded_to_private_provenance=true")
    if not str(adjudicator.get("completed_at_utc", "")).strip():
        issues.append("adjudicator attestation needs completed_at_utc")


def validate_adjudication(
    pack: Path, attestations: Path | None = None
) -> ValidatedAdjudication:
    pack = pack.resolve()
    attestations = (attestations or pack / "physician_attestations.json").resolve()
    issues: list[str] = []
    required = [
        pack / "candidates.blinded.jsonl",
        pack / "annotation_schema.json",
        pack / "annotations.reviewer_1.csv",
        pack / "annotations.reviewer_2.csv",
        pack / "adjudication.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AdjudicationValidationError([f"missing required input: {path}" for path in missing])

    candidates = _read_jsonl(required[0])
    schema = json.loads(required[1].read_text())
    expected_ids = [str(row.get("edge_id", "")) for row in candidates]
    if not candidates or len(expected_ids) != len(set(expected_ids)) or "" in expected_ids:
        issues.append("blinded candidates are empty or have invalid/duplicate edge IDs")
    allowed = schema.get("fields", {})

    reviewer_indexes: list[dict[str, dict[str, str]]] = []
    reviewer_ids: list[str] = []
    immutable_fields = tuple(candidates[0]) if candidates else ()
    for reviewer_number in (1, 2):
        path = pack / f"annotations.reviewer_{reviewer_number}.csv"
        header, rows = _read_csv(path)
        label = f"reviewer_{reviewer_number}"
        missing_columns = [
            field
            for field in (*immutable_fields, "reviewer_id", *REVIEW_FIELDS, "rationale")
            if field not in header
        ]
        if missing_columns:
            issues.append(f"{label}: missing columns {missing_columns}")
        index = _index_rows(rows, expected_ids, label, issues)
        reviewer_indexes.append(index)
        ids = {row.get("reviewer_id", "").strip() for row in rows}
        if len(ids) != 1 or "" in ids:
            issues.append(f"{label}: reviewer_id must be one non-empty ID across all rows")
            reviewer_ids.append("")
        else:
            reviewer_ids.append(next(iter(ids)))
        for candidate, row in zip(candidates, rows):
            edge_id = candidate["edge_id"]
            for field in immutable_fields:
                if row.get(field, "") != _csv_value(candidate[field]):
                    issues.append(f"{label}/{edge_id}: immutable field changed: {field}")
            for field in REVIEW_FIELDS:
                value = row.get(field, "").strip()
                if not value:
                    issues.append(f"{label}/{edge_id}: blank {field}")
                elif value not in allowed.get(field, []):
                    issues.append(f"{label}/{edge_id}: invalid {field}={value!r}")
            if not row.get("rationale", "").strip():
                issues.append(f"{label}/{edge_id}: blank rationale")

    reviewer_id_pair = (reviewer_ids + ["", ""])[:2]
    if reviewer_id_pair[0] == reviewer_id_pair[1]:
        issues.append("reviewer sheets must have distinct reviewer IDs")

    _, adjudication_rows = _read_csv(pack / "adjudication.csv")
    adjudication_index = _index_rows(adjudication_rows, expected_ids, "adjudication", issues)
    adjudicator_ids = {row.get("adjudicator_id", "").strip() for row in adjudication_rows}
    adjudicator_id = next(iter(adjudicator_ids)) if len(adjudicator_ids) == 1 else ""
    if len(adjudicator_ids) != 1 or not adjudicator_id:
        issues.append("adjudication: adjudicator_id must be one non-empty ID across all rows")
    elif adjudicator_id in set(reviewer_id_pair):
        issues.append("adjudicator ID must differ from both independent reviewer IDs")

    for edge_id in expected_ids:
        final = adjudication_index.get(edge_id, {})
        if final.get("case_id") != next(
            (row["case_id"] for row in candidates if row["edge_id"] == edge_id), None
        ):
            issues.append(f"adjudication/{edge_id}: case_id mismatch")
        categorical_disagreement = False
        for reviewer_number, source_index in enumerate(reviewer_indexes, start=1):
            source = source_index.get(edge_id, {})
            for field in (*REVIEW_FIELDS, "rationale"):
                copied = final.get(f"r{reviewer_number}_{field}", "")
                if copied != source.get(field, ""):
                    issues.append(
                        f"adjudication/{edge_id}: r{reviewer_number}_{field} does not "
                        "exactly copy the frozen reviewer sheet"
                    )
            if reviewer_number == 2:
                other = reviewer_indexes[0].get(edge_id, {})
                categorical_disagreement = any(
                    source.get(field, "") != other.get(field, "") for field in REVIEW_FIELDS
                )
        for field in FINAL_FIELDS:
            value = final.get(f"final_{field}", "").strip()
            if not value:
                issues.append(f"adjudication/{edge_id}: blank final_{field}")
            elif value not in allowed.get(field, []):
                issues.append(f"adjudication/{edge_id}: invalid final_{field}={value!r}")
        if not final.get("adjudication_rationale", "").strip():
            issues.append(f"adjudication/{edge_id}: blank adjudication_rationale")
        if categorical_disagreement and not final.get("disagreement_reason", "").strip():
            issues.append(f"adjudication/{edge_id}: disagreement_reason required")

        admitted = final.get("final_edge_entailment_admitted")
        parent_state = final.get("final_parent_visual_support")
        child_state = final.get("final_child_visual_support")
        source_state = final.get("final_increment_observability")
        scope = final.get("final_logical_scope_preserved")
        if admitted == "yes" and scope not in {"yes", "not_applicable"}:
            issues.append(f"adjudication/{edge_id}: admitted edge does not preserve logical scope")
        if admitted == "yes" and child_state == "supported" and parent_state != "supported":
            issues.append(f"adjudication/{edge_id}: supported child cannot entail unsupported parent")
        if admitted == "yes" and source_state in SOURCE_REQUIRED_STATES and child_state != "unobservable":
            issues.append(
                f"adjudication/{edge_id}: unavailable evidence source requires child=unobservable"
            )
        if admitted == "yes" and source_state == "observable_on_supplied_image" and child_state == "unobservable":
            issues.append(
                f"adjudication/{edge_id}: observable increment cannot have child=unobservable"
            )

    _check_attestations(
        attestations,
        (reviewer_id_pair[0], reviewer_id_pair[1]),
        adjudicator_id,
        issues,
    )
    if issues:
        raise AdjudicationValidationError(issues)

    input_paths = {
        "candidates": pack / "candidates.blinded.jsonl",
        "schema": pack / "annotation_schema.json",
        "reviewer_1": pack / "annotations.reviewer_1.csv",
        "reviewer_2": pack / "annotations.reviewer_2.csv",
        "adjudication": pack / "adjudication.csv",
        "physician_attestations": attestations,
    }
    return ValidatedAdjudication(
        candidates=tuple(candidates),
        reviewer_rows=(reviewer_indexes[0], reviewer_indexes[1]),
        final_rows=adjudication_index,
        reviewer_ids=(reviewer_id_pair[0], reviewer_id_pair[1]),
        adjudicator_id=adjudicator_id,
        input_sha256={name: _sha256(path) for name, path in input_paths.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2"),
    )
    parser.add_argument("--attestations", type=Path)
    args = parser.parse_args()
    try:
        result = validate_adjudication(args.pack, args.attestations)
    except AdjudicationValidationError as exc:
        print(
            json.dumps(
                {"status": "refused", "n_issues": len(exc.issues), "issues": exc.issues},
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(
        json.dumps(
            {
                "status": "admitted",
                "edges": len(result.candidates),
                "reviewer_ids": list(result.reviewer_ids),
                "adjudicator_id": result.adjudicator_id,
                "input_sha256": result.input_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
