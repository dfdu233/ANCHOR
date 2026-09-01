#!/usr/bin/env python3
"""Merge two completed blinded reviewer sheets into an adjudication template.

This tool never creates or adjudicates clinical labels. It validates the two
independent returns against the frozen candidates/schema, then copies reviewer
fields byte-for-byte into the corresponding ``r1_*`` and ``r2_*`` columns. All
final and adjudicator fields remain blank for subsequent blinded adjudication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL_ID = "specificity-ratchet-review-merge-v1"
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
COPY_FIELDS = (*REVIEW_FIELDS, "rationale")
FORMULA_PREFIXES = ("=", "+", "-", "@")


class ReviewMergeError(ValueError):
    """Fail-closed review-return integrity error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ReviewMergeError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _csv_value(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return ""
    return str(value)


def _validated_reviewer(
    path: Path,
    *,
    reviewer_number: int,
    candidates: list[dict[str, Any]],
    allowed: dict[str, list[str]],
) -> tuple[str, list[dict[str, str]]]:
    header, rows = _read_csv(path)
    immutable_fields = tuple(candidates[0]) if candidates else ()
    required = (*immutable_fields, "reviewer_id", *COPY_FIELDS)
    missing = [field for field in required if field not in header]
    if missing:
        raise ReviewMergeError(f"reviewer_{reviewer_number}: missing columns {missing}")
    expected_ids = [str(row["edge_id"]) for row in candidates]
    actual_ids = [row.get("edge_id", "") for row in rows]
    if actual_ids != expected_ids:
        raise ReviewMergeError(
            f"reviewer_{reviewer_number}: row order or edge-id set differs from candidates"
        )
    ids = {row.get("reviewer_id", "").strip() for row in rows}
    if len(ids) != 1 or "" in ids:
        raise ReviewMergeError(
            f"reviewer_{reviewer_number}: reviewer_id must be one non-empty ID"
        )
    for candidate, row in zip(candidates, rows):
        edge_id = str(candidate["edge_id"])
        for field in immutable_fields:
            if row.get(field, "") != _csv_value(candidate[field]):
                raise ReviewMergeError(
                    f"reviewer_{reviewer_number}/{edge_id}: immutable field changed: {field}"
                )
        for field in REVIEW_FIELDS:
            value = row.get(field, "").strip()
            if value not in allowed.get(field, []):
                raise ReviewMergeError(
                    f"reviewer_{reviewer_number}/{edge_id}: invalid {field}={value!r}"
                )
        rationale = row.get("rationale", "")
        if not rationale.strip():
            raise ReviewMergeError(
                f"reviewer_{reviewer_number}/{edge_id}: blank rationale"
            )
        if rationale.lstrip().startswith(FORMULA_PREFIXES):
            raise ReviewMergeError(
                f"reviewer_{reviewer_number}/{edge_id}: rationale has spreadsheet-formula prefix"
            )
    return next(iter(ids)), rows


def merge_reviews(
    *,
    candidates_path: Path,
    schema_path: Path,
    template_path: Path,
    reviewer_1_path: Path,
    reviewer_2_path: Path,
) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    candidates = [
        json.loads(line)
        for line in candidates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not candidates:
        raise ReviewMergeError("candidate manifest is empty")
    edge_ids = [str(row.get("edge_id", "")) for row in candidates]
    if "" in edge_ids or len(edge_ids) != len(set(edge_ids)):
        raise ReviewMergeError("candidate edge IDs are empty or duplicated")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("protocol_id") != "specificity-ratchet-physician-pack-v2":
        raise ReviewMergeError("annotation schema protocol mismatch")
    allowed = schema.get("fields")
    if not isinstance(allowed, dict):
        raise ReviewMergeError("annotation schema lacks allowed fields")

    reviewer_1_id, reviewer_1 = _validated_reviewer(
        reviewer_1_path,
        reviewer_number=1,
        candidates=candidates,
        allowed=allowed,
    )
    reviewer_2_id, reviewer_2 = _validated_reviewer(
        reviewer_2_path,
        reviewer_number=2,
        candidates=candidates,
        allowed=allowed,
    )
    if reviewer_1_id == reviewer_2_id:
        raise ReviewMergeError("reviewers must use distinct reviewer IDs")

    header, template = _read_csv(template_path)
    template_ids = [row.get("edge_id", "") for row in template]
    if template_ids != edge_ids:
        raise ReviewMergeError("adjudication template row order or edge IDs differ")
    required_template = {"case_id", "edge_id"}
    required_template.update(
        f"r{reviewer}_{field}"
        for reviewer in (1, 2)
        for field in COPY_FIELDS
    )
    missing_template = sorted(required_template - set(header))
    if missing_template:
        raise ReviewMergeError(
            f"adjudication template missing columns {missing_template}"
        )
    for row in template:
        copied_columns = [
            f"r{reviewer}_{field}"
            for reviewer in (1, 2)
            for field in COPY_FIELDS
        ]
        if any(row.get(field, "") for field in copied_columns):
            raise ReviewMergeError("adjudication template already contains reviewer values")
        if any(
            value
            for key, value in row.items()
            if key.startswith("final_")
            or key in {"adjudicator_id", "disagreement_reason", "adjudication_rationale"}
        ):
            raise ReviewMergeError("adjudication template already contains final values")

    merged: list[dict[str, str]] = []
    for template_row, first, second, candidate in zip(
        template, reviewer_1, reviewer_2, candidates
    ):
        if template_row.get("case_id") != str(candidate.get("case_id", "")):
            raise ReviewMergeError(
                f"adjudication/{candidate['edge_id']}: case_id mismatch"
            )
        row = dict(template_row)
        for reviewer_number, source in ((1, first), (2, second)):
            for field in COPY_FIELDS:
                row[f"r{reviewer_number}_{field}"] = source[field]
        merged.append(row)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "status": "reviewer_fields_copied_final_adjudication_blank",
        "edges": len(merged),
        "reviewer_ids": [reviewer_1_id, reviewer_2_id],
        "input_sha256": {
            "candidates": _sha256(candidates_path),
            "schema": _sha256(schema_path),
            "template": _sha256(template_path),
            "reviewer_1": _sha256(reviewer_1_path),
            "reviewer_2": _sha256(reviewer_2_path),
        },
        "clinical_truth_created": False,
        "final_fields_blank": True,
    }
    return header, merged, metadata


def _write_csv_once(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    if path.exists():
        raise ReviewMergeError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--reviewer-1", type=Path, required=True)
    parser.add_argument("--reviewer-2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        header, rows, metadata = merge_reviews(
            candidates_path=args.pack / "candidates.blinded.jsonl",
            schema_path=args.pack / "annotation_schema.json",
            template_path=args.pack / "adjudication.csv",
            reviewer_1_path=args.reviewer_1,
            reviewer_2_path=args.reviewer_2,
        )
        _write_csv_once(args.output, header, rows)
    except (OSError, json.JSONDecodeError, ReviewMergeError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(2)
    metadata["output"] = str(args.output.resolve())
    metadata["output_sha256"] = _sha256(args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
