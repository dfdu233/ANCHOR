#!/usr/bin/env python3
"""Fail-closed identity gate for a method-specific inference backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anchor.medeval.evaluate_oe_vqa import answer_tokens, normalize_answer, token_f1
from anchor.medeval.hashing import sha256_file
from anchor.medeval.legacy import FUNCTION_WORD_ONLY


ID_KEYS = ("qid", "question_id", "id", "sample_id")
TEXT_KEYS = ("text", "answer", "prediction", "output")


def first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def load_answers(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"answer file contains a non-object row: {path}")
    return rows


def surface(text: str) -> str:
    return " ".join(text.lower().split()).strip(".,:;!?()[]{}")


def evaluate_conformance(
    canonical_path: Path,
    candidate_path: Path,
    *,
    limit: int = 0,
    min_normalized_exact: float = 0.95,
    min_token_f1: float = 0.98,
    max_function_word_only_rate: float = 0.49,
    require_token_exact: bool = False,
) -> dict[str, Any]:
    canonical = load_answers(canonical_path)
    candidate = load_answers(candidate_path)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit:
        canonical = canonical[:limit]
        candidate = candidate[:limit]
    canonical_ids = [str(first(row, ID_KEYS)) for row in canonical]
    candidate_ids = [str(first(row, ID_KEYS)) for row in candidate]
    canonical_text = [str(first(row, TEXT_KEYS) or "").strip() for row in canonical]
    candidate_text = [str(first(row, TEXT_KEYS) or "").strip() for row in candidate]
    aligned = bool(
        canonical_ids == candidate_ids
        and len(canonical_ids) == len(set(canonical_ids))
        and len(canonical_ids) > 0
    )
    canonical_token_ids = [row.get("metadata", {}).get("generated_token_ids") for row in canonical]
    candidate_token_ids = [row.get("metadata", {}).get("generated_token_ids") for row in candidate]
    token_ids_available = bool(canonical) and all(
        isinstance(value, list) for value in canonical_token_ids + candidate_token_ids
    )
    token_exact_rate = (
        sum(left == right for left, right in zip(canonical_token_ids, candidate_token_ids))
        / len(canonical)
        if token_ids_available and aligned
        else None
    )
    if aligned:
        normalized_exact = sum(
            normalize_answer(left) == normalize_answer(right)
            for left, right in zip(canonical_text, candidate_text)
        ) / len(canonical_text)
        mean_token_f1 = sum(
            token_f1(left, right)
            for left, right in zip(canonical_text, candidate_text)
        ) / len(canonical_text)
    else:
        normalized_exact = 0.0
        mean_token_f1 = 0.0
    candidate_nonempty_rate = sum(
        bool(answer_tokens(text)) for text in candidate_text
    ) / max(len(candidate_text), 1)
    function_word_only_rate = sum(
        surface(text) in FUNCTION_WORD_ONLY for text in candidate_text
    ) / max(len(candidate_text), 1)
    reasons = []
    if not aligned:
        reasons.append("qid_order_or_cardinality_mismatch")
    if candidate_nonempty_rate < 0.95:
        reasons.append("candidate_semantic_nonempty_rate_below_95_percent")
    if function_word_only_rate > max_function_word_only_rate:
        reasons.append("candidate_function_word_only_rate_too_high")
    if normalized_exact < min_normalized_exact:
        reasons.append("normalized_exact_below_threshold")
    if mean_token_f1 < min_token_f1:
        reasons.append("token_f1_below_threshold")
    if require_token_exact and token_exact_rate != 1.0:
        reasons.append("generated_token_ids_not_exact")
    return {
        "protocol": "backend-identity-conformance-v1",
        "canonical_answers": str(canonical_path.resolve()),
        "canonical_sha256": sha256_file(canonical_path),
        "candidate_answers": str(candidate_path.resolve()),
        "candidate_sha256": sha256_file(candidate_path),
        "prefix_limit": limit,
        "n": len(canonical),
        "aligned": aligned,
        "normalized_exact": normalized_exact,
        "minimum_normalized_exact": min_normalized_exact,
        "mean_token_f1": mean_token_f1,
        "minimum_token_f1": min_token_f1,
        "candidate_semantic_nonempty_rate": candidate_nonempty_rate,
        "candidate_function_word_only_rate": function_word_only_rate,
        "maximum_function_word_only_rate": max_function_word_only_rate,
        "generated_token_ids_available": token_ids_available,
        "generated_token_exact_rate": token_exact_rate,
        "generated_token_exact_required": require_token_exact,
        "passed": not reasons,
        "failure_reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-normalized-exact", type=float, default=0.95)
    parser.add_argument("--min-token-f1", type=float, default=0.98)
    parser.add_argument("--max-function-word-only-rate", type=float, default=0.49)
    parser.add_argument("--require-token-exact", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Compare the same ordered prefix without rewriting either artifact.",
    )
    args = parser.parse_args()
    result = evaluate_conformance(
        args.canonical,
        args.candidate,
        limit=args.limit,
        min_normalized_exact=args.min_normalized_exact,
        min_token_f1=args.min_token_f1,
        max_function_word_only_rate=args.max_function_word_only_rate,
        require_token_exact=args.require_token_exact,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
