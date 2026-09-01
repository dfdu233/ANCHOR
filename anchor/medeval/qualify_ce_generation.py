#!/usr/bin/env python3
"""Task-aware qualification for decoded CE benchmark artifacts.

The gate validates alignment, authoritative references, and actual CE parsing
for binary, ternary, and multiple-choice rows.  Generation length is retained
as a diagnostic and never decides admissibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anchor.corrected_sgta.evaluate_medheval_answers import (
    PROTOCOL_VERSION as EVALUATOR_VERSION,
    align_answers_with_questions,
    evaluate_rows,
)
from anchor.medeval.hashing import sha256_file

from .audit_retrieval_split import read_rows
from .store import atomic_write_json


VERSION = "ce-generation-qualification-v2-task-aware-structural"


def qid(row: dict[str, Any], index: int) -> str:
    return str(row.get("question_id", row.get("qid", row.get("id", index))))


def qualify(
    manifest: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    max_new_tokens: int,
    *,
    minimum_nonempty_rate: float = 0.95,
    minimum_parse_rate: float = 0.90,
) -> dict[str, Any]:
    expected = [qid(row, index) for index, row in enumerate(manifest)]
    observed = [qid(row, index) for index, row in enumerate(answers)]
    texts = [str(row.get("text", row.get("prediction", row.get("output", ""))) or "") for row in answers]
    counts = []
    for row in answers:
        raw = (row.get("metadata") or {}).get("generated_token_count")
        try:
            counts.append(int(raw))
        except (TypeError, ValueError):
            counts.append(-1)
    n = len(expected)
    source: dict[str, dict[str, Any]] = {}
    duplicate_manifest_ids = len(set(expected)) != len(expected)
    duplicate_answer_ids = len(set(observed)) != len(observed)
    for index, row in enumerate(manifest):
        source[qid(row, index)] = row

    alignment_error = None
    report = None
    if not duplicate_manifest_ids and not duplicate_answer_ids:
        try:
            merged = align_answers_with_questions(answers, source)
            report = evaluate_rows(merged)
        except ValueError as error:
            alignment_error = str(error)

    strict = (report or {}).get("primary_multiclass", {})
    # evaluate_rows does not attach this convenience view until CLI finalization.
    if report is not None:
        from anchor.corrected_sgta.evaluate_medheval_answers import _multiclass_metrics

        strict = _multiclass_metrics(report["details"])
    parse_rate = float(strict.get("parse_rate", 0.0) or 0.0)
    accuracy = float(strict.get("accuracy_invalid_as_error", 0.0) or 0.0)
    invalid_ground_truth = int((report or {}).get("invalid_ground_truth", n))
    nonempty_rate = sum(bool(text.strip()) for text in texts) / n if n else 0.0
    count_coverage = sum(value >= 0 for value in counts) / n if n else 0.0
    cap_hit_rate = (
        sum(value >= max_new_tokens for value in counts if value >= 0)
        / max(sum(value >= 0 for value in counts), 1)
    )
    result: dict[str, Any] = {
        "protocol_version": VERSION,
        "evaluator_protocol_version": EVALUATOR_VERSION,
        "expected_count": n,
        "received_count": len(answers),
        "exact_qid_alignment": expected == observed,
        "same_qid_set": set(expected) == set(observed),
        "duplicate_manifest_ids": duplicate_manifest_ids,
        "duplicate_answer_ids": duplicate_answer_ids,
        "alignment_or_reference_error": alignment_error,
        "nonempty_rate": nonempty_rate,
        "strict_parse_rate": parse_rate,
        "strict_accuracy_diagnostic": accuracy,
        "invalid_ground_truth_count": invalid_ground_truth,
        "by_answer_type": strict.get("by_answer_type", {}),
        "parse_failures": (report or {}).get("parse_failures", {}),
        "generated_token_count_coverage": count_coverage,
        "cap_hit_rate": cap_hit_rate,
        "max_new_tokens_for_diagnostic": max_new_tokens,
        "cap_hit_is_diagnostic_only": True,
        "minimum_nonempty_rate": minimum_nonempty_rate,
        "minimum_strict_parse_rate": minimum_parse_rate,
        "metric_floor_policy": (
            "accuracy is diagnostic and never gates a scientific result; the gate "
            "only verifies that outputs and references can be judged correctly"
        ),
    }
    result["passed"] = bool(
        n > 0
        and result["exact_qid_alignment"]
        and result["same_qid_set"]
        and not duplicate_manifest_ids
        and not duplicate_answer_ids
        and alignment_error is None
        and invalid_ground_truth == 0
        and nonempty_rate >= minimum_nonempty_rate
        and parse_rate >= minimum_parse_rate
    )
    result["artifact_status"] = "admissible" if result["passed"] else "regenerate_or_repair_parser"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-nonempty-rate", type=float, default=0.95)
    parser.add_argument("--min-parse-rate", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_rows(args.manifest)
    if args.limit:
        manifest = manifest[: args.limit]
    result = qualify(
        manifest,
        read_rows(args.answers),
        args.max_new_tokens,
        minimum_nonempty_rate=args.min_nonempty_rate,
        minimum_parse_rate=args.min_parse_rate,
    )
    result.update(
        manifest=str(args.manifest.resolve()),
        manifest_sha256=sha256_file(args.manifest),
        answers=[str(args.answers.resolve())],
        answer_sha256=[sha256_file(args.answers)],
        evaluator_source=str(Path(__file__).resolve()),
        evaluator_source_sha256=sha256_file(Path(__file__).resolve()),
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
