#!/usr/bin/env python3
"""Audit every historical high-efficiency answer file before reuse."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .legacy import audit_legacy_answers


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def state_index(root: Path) -> dict[str, dict[str, Any]]:
    index = {}
    for state_file in root.rglob("queue_state.jsonl"):
        for row in load_jsonl(state_file):
            answers = row.get("answers")
            if answers:
                index[str(Path(answers).resolve())] = {**row, "state_file": str(state_file)}
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    states = state_index(args.root)
    audits = []
    hashes: dict[str, list[str]] = defaultdict(list)
    for answers in sorted(args.root.rglob("*.answers.jsonl")):
        questions = answers.with_name(answers.name.replace(".answers.jsonl", ".questions.json"))
        if not questions.exists():
            audits.append({
                "answers_path": str(answers.resolve()),
                "grade": "C", "action": "rerun",
                "degenerate_reasons": ["missing_question_manifest"],
            })
            continue
        question_rows = json.loads(questions.read_text())
        expected_ids = [str(row.get("qid", row.get("id", index)))
                        for index, row in enumerate(question_rows)]
        audit = audit_legacy_answers(answers, expected_ids)
        state = states.get(str(answers.resolve()), {})
        audit.update({
            "questions_path": str(questions.resolve()),
            "questions_sha256": sha256_file(questions),
            "legacy_state": state,
        })
        audits.append(audit)
        hashes[audit["answers_sha256"]].append(str(answers.resolve()))
    identical_groups = [paths for paths in hashes.values() if len(paths) > 1]
    for audit in audits:
        answer_hash = audit.get("answers_sha256")
        group = hashes.get(answer_hash, [])
        audit["byte_identical_answer_files"] = group if len(group) > 1 else []
        if len(group) > 1 and audit["grade"] != "C":
            audit["grade"] = "C"
            audit["action"] = "rerun"
            audit.setdefault("degenerate_reasons", []).append(
                "byte_identical_across_nominally_distinct_runs"
            )
    summary = {
        "root": str(args.root.resolve()),
        "files": len(audits),
        "grade_counts": {
            grade: sum(row.get("grade") == grade for row in audits)
            for grade in ("A", "B", "C")
        },
        "action_counts": {
            action: sum(row.get("action") == action for row in audits)
            for action in ("reuse", "rescore_only", "rerun")
        },
        "identical_groups": identical_groups,
        "audits": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "files", "grade_counts", "action_counts"
    )}, indent=2))


if __name__ == "__main__":
    main()
