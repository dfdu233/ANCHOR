#!/usr/bin/env python3
"""Freeze an auditable, image-grounded CE query set for common-protocol RAG."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from corrected_sgta.evaluate_medheval_answers import normalize_binary_reference

from .audit_retrieval_split import read_rows
from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "common-rag-query-contract-v1"

MANAGEMENT_RE = re.compile(
    r"\b(?:treat(?:ed|ment)?|therap(?:y|ies)|manage(?:ment)?|medication|"
    r"recommend(?:ed|ation)?|require treatment|need of immediate treatment|"
    r"should (?:the )?patient|should .* withdrawn|monitor(?:ed)? for changes)\b",
    re.I,
)
TEMPORAL_RE = re.compile(
    r"\b(?:compared (?:with|to)|since (?:the )?(?:last|previous|prior)|"
    r"interval change|has there been .*change|remain(?:s|ed)? stable|"
    r"stable appearance|unchanged|new since|resolved since|worsen(?:ed|ing)?)\b",
    re.I,
)
HISTORY_CAUSE_RE = re.compile(
    r"\b(?:history of|has .* undergone|likely (?:had|undergone|suffered)|"
    r"due to|caused by|cause of|associated with (?:previous|a history)|"
    r"indicative of a history|previous (?:treatment|surgery|operation))\b",
    re.I,
)


def sample_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("question_id", row.get("qid", row.get("id", index))))


def route(row: dict[str, Any]) -> tuple[str, str]:
    """Route by information requirement; do not infer the clinical answer."""

    answer = row.get("answer", row.get("gt_ans", row.get("ground_truth", "")))
    if normalize_binary_reference(answer) is None:
        return "invalid_reference", "reference_has_no_leading_explicit_yes_no"
    question = str(row.get("question", row.get("text", ""))).replace("<image>", " ")
    if MANAGEMENT_RE.search(question):
        return "knowledge_claim", "management_or_treatment"
    if TEMPORAL_RE.search(question):
        return "unobservable", "requires_temporal_comparison"
    if HISTORY_CAUSE_RE.search(question):
        return "knowledge_claim", "history_or_etiology_inference"
    return "image_grounded", "single_image_finding_or_attribute"


def prepare(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    admitted, excluded = [], []
    seen: set[str] = set()
    for index, source in enumerate(rows):
        qid = sample_id(source, index)
        if qid in seen:
            raise ValueError(f"duplicate query id: {qid}")
        seen.add(qid)
        observability, reason = route(source)
        row = dict(source)
        row.update({
            "qid": qid,
            "observability": observability,
            "observability_rule": reason,
            "reference_contract": "leading_explicit_yes_no",
        })
        (admitted if observability == "image_grounded" else excluded).append(row)
    return admitted, excluded


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = read_rows(args.input)
    admitted, excluded = prepare(source)
    admitted_path = args.output_dir / "image_grounded.jsonl"
    excluded_path = args.output_dir / "excluded.jsonl"
    write_jsonl(admitted_path, admitted)
    write_jsonl(excluded_path, excluded)
    summary = {
        "protocol_version": VERSION,
        "source": str(args.input.resolve()),
        "source_sha256": sha256_file(args.input),
        "source_n": len(source),
        "image_grounded_n": len(admitted),
        "excluded_n": len(excluded),
        "routing_counts": dict(sorted(Counter(row["observability"] for row in admitted + excluded).items())),
        "reason_counts": dict(sorted(Counter(row["observability_rule"] for row in admitted + excluded).items())),
        "image_grounded": str(admitted_path.resolve()),
        "image_grounded_sha256": sha256_file(admitted_path),
        "excluded": str(excluded_path.resolve()),
        "excluded_sha256": sha256_file(excluded_path),
        "claim_ceiling": (
            "routing is a frozen information-requirement audit, not clinical truth; "
            "only image_grounded rows enter the visual-claim common protocol"
        ),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
