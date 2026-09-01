"""Merge two validated blinded OE reviews into a still-blinded adjudication sheet.

This module copies independent clinical annotations but never creates consensus
labels.  Final annotation fields remain exactly as blank as in the source
template until a third blinded clinician adjudicates them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hashing import sha256_file
from .store import atomic_write_json
from .validate_physician_oe_review import load_jsonl, validate_completed


VERSION = "anchor-physician-oe-adjudication-preparation-v1"


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite adjudication artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validation_matches(
    validation: Mapping[str, Any], template_path: Path, completed_path: Path, slot: str
) -> None:
    if validation.get("passed") is not True:
        raise ValueError(f"reviewer {slot} validation did not pass")
    if validation.get("reviewer_slot") != slot:
        raise ValueError(f"reviewer {slot} validation slot mismatch")
    if validation.get("template_sha256") != sha256_file(template_path):
        raise ValueError(f"reviewer {slot} template hash mismatch")
    if validation.get("completed_sha256") != sha256_file(completed_path):
        raise ValueError(f"reviewer {slot} completed hash mismatch")


def prepare_adjudication(
    master: Sequence[Mapping[str, Any]],
    reviewer_a: Sequence[Mapping[str, Any]],
    reviewer_b: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not master or len(master) != len(reviewer_a) or len(master) != len(reviewer_b):
        raise ValueError("master and reviewer group counts differ")
    by_a = {str(row["group_id"]): row for row in reviewer_a}
    by_b = {str(row["group_id"]): row for row in reviewer_b}
    if len(by_a) != len(reviewer_a) or len(by_b) != len(reviewer_b):
        raise ValueError("duplicate reviewer group")
    output = []
    for source in master:
        group_id = str(source["group_id"])
        if group_id not in by_a or group_id not in by_b:
            raise ValueError(f"reviewer return lacks group {group_id}")
        a = by_a[group_id]
        b = by_b[group_id]
        answer_a = {str(item["answer_id"]): item for item in a["candidate_answers"]}
        answer_b = {str(item["answer_id"]): item for item in b["candidate_answers"]}
        expected = [str(item["answer_id"]) for item in source["candidate_answers"]]
        if set(answer_a) != set(expected) or set(answer_b) != set(expected):
            raise ValueError(f"reviewer answer set differs for {group_id}")
        row = json.loads(json.dumps(source))
        row["independent_reviews"] = {
            "A": a["reference_annotation"],
            "B": b["reference_annotation"],
        }
        for candidate in row["candidate_answers"]:
            answer_id = str(candidate["answer_id"])
            candidate["independent_reviews"] = {
                "A": answer_a[answer_id]["annotation"],
                "B": answer_b[answer_id]["annotation"],
            }
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-template", type=Path, required=True)
    parser.add_argument("--reviewer-a-template", type=Path, required=True)
    parser.add_argument("--reviewer-a-completed", type=Path, required=True)
    parser.add_argument("--reviewer-a-validation", type=Path, required=True)
    parser.add_argument("--reviewer-b-template", type=Path, required=True)
    parser.add_argument("--reviewer-b-completed", type=Path, required=True)
    parser.add_argument("--reviewer-b-validation", type=Path, required=True)
    parser.add_argument("--clarification-log", type=Path, required=True)
    parser.add_argument("--output-template", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    clarification = args.clarification_log.read_text(encoding="utf-8")
    if "- Pending." in clarification or not clarification.strip():
        raise ValueError("calibration clarification log is not frozen")
    reviewer_templates = {
        "A": load_jsonl(args.reviewer_a_template),
        "B": load_jsonl(args.reviewer_b_template),
    }
    completed_paths = {
        "A": args.reviewer_a_completed,
        "B": args.reviewer_b_completed,
    }
    completed = {
        slot: load_jsonl(path) for slot, path in completed_paths.items()
    }
    validations = {
        "A": json.loads(args.reviewer_a_validation.read_text()),
        "B": json.loads(args.reviewer_b_validation.read_text()),
    }
    for slot, template_path, validation_path in (
        ("A", args.reviewer_a_template, args.reviewer_a_validation),
        ("B", args.reviewer_b_template, args.reviewer_b_validation),
    ):
        validate_completed(reviewer_templates[slot], completed[slot])
        _validation_matches(
            validations[slot], template_path, completed_paths[slot], slot
        )
        if validations[slot].get("protocol_version") != "anchor-physician-oe-review-validation-v1":
            raise ValueError(f"reviewer {slot} validation protocol mismatch")
        if not validation_path.is_file():
            raise FileNotFoundError(validation_path)
    master = load_jsonl(args.master_template)
    adjudication = prepare_adjudication(master, completed["A"], completed["B"])
    _write_jsonl_once(args.output_template, adjudication)
    manifest = {
        "protocol_version": VERSION,
        "bundle_id": master[0]["bundle_id"],
        "groups": len(adjudication),
        "answer_units": sum(len(row["candidate_answers"]) for row in adjudication),
        "clinical_consensus_created": False,
        "model_identity_visible": False,
        "private_mapping_joined": False,
        "final_fields_blank": True,
        "master_template_sha256": sha256_file(args.master_template),
        "reviewer_a_completed_sha256": sha256_file(args.reviewer_a_completed),
        "reviewer_b_completed_sha256": sha256_file(args.reviewer_b_completed),
        "reviewer_a_validation_sha256": sha256_file(args.reviewer_a_validation),
        "reviewer_b_validation_sha256": sha256_file(args.reviewer_b_validation),
        "clarification_log_sha256": sha256_file(args.clarification_log),
        "adjudication_template": str(args.output_template.resolve()),
        "adjudication_template_sha256": sha256_file(args.output_template),
        "required_next_step": (
            "A third blinded clinician fills only the final reference_annotation and "
            "candidate annotation fields; independent_reviews and immutable content must not change."
        ),
    }
    if args.output_manifest.exists():
        raise FileExistsError("adjudication manifest is write-once")
    atomic_write_json(args.output_manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
