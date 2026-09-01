"""Validate blinded adjudication and freeze a clean physician OE consensus.

Finalization strips the copied independent-review fields before any private
method mapping is joined, while binding every reviewer, clarification, and
adjudication input by hash.
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


VERSION = "anchor-physician-oe-consensus-v1"


def _independent_view(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "group_id": row.get("group_id"),
            "reference": row.get("independent_reviews"),
            "answers": [
                {
                    "answer_id": item.get("answer_id"),
                    "reviews": item.get("independent_reviews"),
                }
                for item in row.get("candidate_answers", [])
            ],
        }
        for row in rows
    ]


def clean_consensus(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = json.loads(json.dumps(rows))
    for row in output:
        row.pop("independent_reviews", None)
        for candidate in row["candidate_answers"]:
            candidate.pop("independent_reviews", None)
    return output


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite consensus: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-template", type=Path, required=True)
    parser.add_argument("--adjudication-template", type=Path, required=True)
    parser.add_argument("--completed-adjudication", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--adjudicator-id", required=True)
    parser.add_argument("--attest-model-blinded", action="store_true")
    parser.add_argument("--attest-no-private-mapping", action="store_true")
    parser.add_argument("--output-consensus", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    args = parser.parse_args()
    if not args.attest_model_blinded or not args.attest_no_private_mapping:
        raise ValueError("both blinded-adjudication attestations are required")
    if not args.adjudicator_id.strip():
        raise ValueError("adjudicator ID is empty")
    master = load_jsonl(args.master_template)
    template = load_jsonl(args.adjudication_template)
    completed = load_jsonl(args.completed_adjudication)
    preparation = json.loads(args.preparation_manifest.read_text())
    if preparation.get("protocol_version") != "anchor-physician-oe-adjudication-preparation-v1":
        raise ValueError("adjudication preparation protocol mismatch")
    if preparation.get("adjudication_template_sha256") != sha256_file(
        args.adjudication_template
    ):
        raise ValueError("adjudication template hash mismatch")
    if preparation.get("master_template_sha256") != sha256_file(args.master_template):
        raise ValueError("master template hash mismatch")
    if _independent_view(template) != _independent_view(completed):
        raise ValueError("independent physician reviews changed during adjudication")
    # Immutable model-blinded content and all final clinical fields must pass the
    # same contract used for the independent reviews.
    validate_completed(template, completed)
    consensus = clean_consensus(completed)
    validate_completed(master, consensus)
    _write_jsonl_once(args.output_consensus, consensus)
    provenance = {
        "protocol_version": VERSION,
        "bundle_id": master[0]["bundle_id"],
        "reviewers": ["A", "B"],
        "adjudicator_id": args.adjudicator_id,
        "unresolved_disagreements": 0,
        "model_identity_visible_during_adjudication": False,
        "private_mapping_joined_before_consensus": False,
        "consensus": str(args.output_consensus.resolve()),
        "consensus_sha256": sha256_file(args.output_consensus),
        "master_template_sha256": sha256_file(args.master_template),
        "adjudication_template_sha256": sha256_file(args.adjudication_template),
        "completed_adjudication_sha256": sha256_file(args.completed_adjudication),
        "preparation_manifest_sha256": sha256_file(args.preparation_manifest),
        "reviewer_a_completed_sha256": preparation["reviewer_a_completed_sha256"],
        "reviewer_b_completed_sha256": preparation["reviewer_b_completed_sha256"],
        "reviewer_a_validation_sha256": preparation["reviewer_a_validation_sha256"],
        "reviewer_b_validation_sha256": preparation["reviewer_b_validation_sha256"],
        "clarification_log_sha256": preparation["clarification_log_sha256"],
        "clinical_consensus_created": True,
        "unblinding_authorized": True,
    }
    if args.output_provenance.exists():
        raise FileExistsError("consensus provenance is write-once")
    atomic_write_json(args.output_provenance, provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
