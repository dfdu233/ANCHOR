#!/usr/bin/env python3
"""Build hash-bound A/B deliveries from a blinded physician OE template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "anchor-physician-oe-review-deliveries-v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_deliveries(
    template: list[dict[str, Any]],
    *,
    calibration_groups: int,
    double_review_groups: int,
) -> dict[str, list[dict[str, Any]]]:
    if not template:
        raise ValueError("review template is empty")
    if not 0 <= calibration_groups <= double_review_groups <= len(template):
        raise ValueError(
            "require 0 <= calibration_groups <= double_review_groups <= groups"
        )
    group_ids = [str(row["group_id"]) for row in template]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("review template has duplicate group IDs")
    answer_ids = [
        str(candidate["answer_id"])
        for row in template
        for candidate in row["candidate_answers"]
    ]
    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError("reviewer-visible template has duplicate answer IDs")

    deliveries: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for reviewer in deliveries:
        for index, source in enumerate(template):
            if index >= double_review_groups:
                continue
            row = json.loads(json.dumps(source))
            row["reviewer_slot"] = reviewer
            row["review_phase"] = (
                "calibration" if index < calibration_groups else "double_review"
            )
            deliveries[reviewer].append(row)
    return deliveries


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-groups", type=int, default=10)
    parser.add_argument("--double-review-groups", type=int, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix reviewer delivery: {args.output_dir}")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("bundle_sha256") != sha256_file(args.template):
        raise ValueError("review template hash differs from source metadata")
    template = load_jsonl(args.template)
    deliveries = build_deliveries(
        template,
        calibration_groups=args.calibration_groups,
        double_review_groups=args.double_review_groups,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for reviewer, rows in deliveries.items():
        path = args.output_dir / f"reviewer_{reviewer}.blinded.jsonl"
        write_jsonl(path, rows)
        paths[reviewer] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "groups": len(rows),
            "answer_units": sum(len(row["candidate_answers"]) for row in rows),
        }
    clarification = args.output_dir / "clarification_log.template.md"
    clarification.write_text(
        "# Blinded calibration clarification log\n\n"
        "Bundle ID: `" + str(metadata["bundle_id"]) + "`\n\n"
        "Do not record model guesses, method names, scores, or private mapping.\n\n"
        "## Clarifications after the first calibration pass\n\n"
        "- Pending.\n",
        encoding="utf-8",
    )
    delivery = {
        "version": VERSION,
        "bundle_id": metadata["bundle_id"],
        "source_template": str(args.template.resolve()),
        "source_template_sha256": sha256_file(args.template),
        "source_metadata": str(args.metadata.resolve()),
        "source_metadata_sha256": sha256_file(args.metadata),
        "calibration_groups": args.calibration_groups,
        "double_review_groups": args.double_review_groups,
        "reviewers": paths,
        "clarification_log_template": str(clarification.resolve()),
        "clarification_log_template_sha256": sha256_file(clarification),
        "private_mapping_in_delivery": False,
        "unblinding_authorized": False,
        "required_next_step": "Reviewers independently complete the calibration groups, freeze and hash the clarification log, then revise calibration and finish the remaining double-review groups.",
    }
    atomic_write_json(args.output_dir / "delivery_manifest.json", delivery)
    print(json.dumps(delivery, indent=2))


if __name__ == "__main__":
    main()
