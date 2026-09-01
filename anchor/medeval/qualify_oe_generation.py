#!/usr/bin/env python3
"""Fail-closed structural qualification for open-ended generation outputs.

Length and token-budget exhaustion are diagnostics, not validity failures.  A
long medical answer is admissible unless it is empty, collapsed, misaligned,
or contains an obvious autoregressive repetition loop.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from anchor.medeval.evaluate_oe_vqa import _load_json, _load_jsonl, answer_tokens, normalize_answer
from anchor.medeval.hashing import sha256_file
from anchor.medeval.legacy import FUNCTION_WORD_ONLY


TERMINAL_QUESTION_POLICIES = {"all", "explicit_sentence_instruction"}


def has_repetition_loop(text: str) -> bool:
    """Detect only conspicuous repeated spans, not legitimate medical reuse."""

    tokens = answer_tokens(text)
    if len(tokens) < 12:
        return False
    # Three adjacent copies of a 4--24 token span are a generation loop.  The
    # adjacency requirement avoids rejecting ordinary repeated terminology.
    for width in range(4, min(24, len(tokens) // 3) + 1):
        for start in range(0, len(tokens) - 3 * width + 1):
            span = tokens[start : start + width]
            if (
                span == tokens[start + width : start + 2 * width]
                and span == tokens[start + 2 * width : start + 3 * width]
            ):
                return True
    return False


def generated_token_count(row: dict[str, Any]) -> int | None:
    """Read the token count from certified native and mitigation-port schemas."""

    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    for key in ("generated_token_count", "decoded_sequence_token_count"):
        value = metadata.get(key)
        if value is not None:
            return int(value)
    token_ids = metadata.get("generated_token_ids")
    return len(token_ids) if isinstance(token_ids, list) else None


def terminal_required(question: str, policy: str) -> bool:
    """Return whether the prompt explicitly calls for sentence-form prose.

    Open VQA references are commonly noun phrases, so punctuation cannot be a
    universal completion criterion.  This frozen policy uses prompt wording
    only and never inspects the prediction or reference answer.
    """

    if policy not in TERMINAL_QUESTION_POLICIES:
        raise ValueError(f"unsupported terminal question policy: {policy}")
    if policy == "all":
        return True
    normalized = " ".join(question.lower().split()).lstrip()
    return bool(re.match(r"^(?:please\s+)?(?:describe|explain)\b|^why\b", normalized))


def qualify(
    manifest: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    *,
    limit: int,
    min_nonempty_rate: float = 0.95,
    min_unique_rate: float = 0.10,
    max_function_word_only_rate: float = 0.01,
    max_new_tokens: int | None = None,
    max_cap_hit_rate: float = 0.05,
    max_repetition_loop_rate: float = 0.01,
    require_terminal_completeness: bool = False,
    min_terminal_completeness_rate: float = 0.95,
    terminal_question_policy: str = "all",
) -> dict[str, Any]:
    expected = [str(row.get("qid") or row.get("id")) for row in manifest[:limit]]
    received = [str(row.get("qid") or row.get("question_id") or row.get("id")) for row in answers]
    predictions = [str(row.get("text") or row.get("answer") or "") for row in answers]
    normalized = [normalize_answer(text) for text in predictions]
    surface_normalized = [
        " ".join(text.lower().split()).strip(".,:;!?()[]{}")
        for text in predictions
    ]
    nonempty_rate = sum(bool(text) for text in normalized) / max(len(normalized), 1)
    unique_rate = len(set(normalized)) / max(len(normalized), 1)
    function_word_only_rate = (
        sum(text in FUNCTION_WORD_ONLY for text in surface_normalized)
        / max(len(normalized), 1)
    )
    sentinel_count = sum(text in {"skipped", "error", "failed"} for text in normalized)
    exact_qid_alignment = received == expected
    generated_counts = [generated_token_count(row) for row in answers]
    count_coverage = sum(value is not None for value in generated_counts) / max(
        len(answers), 1
    )
    cap_hit_rate = None
    if max_new_tokens is not None and count_coverage == 1.0:
        cap_hit_rate = sum(int(value) >= max_new_tokens for value in generated_counts) / max(
            len(generated_counts), 1
        )
    repetition_loop_count = sum(has_repetition_loop(text) for text in predictions)
    repetition_loop_rate = repetition_loop_count / max(len(predictions), 1)
    terminal_mask = [
        terminal_required(str(row.get("question", "")), terminal_question_policy)
        for row in manifest[:limit]
    ]
    terminal_outcomes = [
        bool(re.search(r"[.!?][\]\)}'\"]*\s*$", text.strip()))
        for text, required in zip(predictions, terminal_mask)
        if required
    ]
    terminal_rate = (
        sum(terminal_outcomes) / len(terminal_outcomes) if terminal_outcomes else 1.0
    )
    terminal_gate = (
        not require_terminal_completeness
        or terminal_rate >= min_terminal_completeness_rate
    )
    passed = bool(
        len(answers) == limit
        and exact_qid_alignment
        and nonempty_rate >= min_nonempty_rate
        and unique_rate >= min_unique_rate
        and function_word_only_rate <= max_function_word_only_rate
        and sentinel_count == 0
        and repetition_loop_rate <= max_repetition_loop_rate
        and terminal_gate
    )
    lengths = [len(answer_tokens(text)) for text in predictions]
    if passed:
        artifact_status = "admissible"
    elif len(answers) == limit and exact_qid_alignment and nonempty_rate >= min_nonempty_rate:
        artifact_status = "regenerate"
    else:
        artifact_status = "regenerate"
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "expected_count": limit,
        "received_count": len(answers),
        "exact_qid_alignment": exact_qid_alignment,
        "nonempty_rate": nonempty_rate,
        "minimum_nonempty_rate": min_nonempty_rate,
        "unique_prediction_rate": unique_rate,
        "minimum_unique_prediction_rate": min_unique_rate,
        "function_word_only_rate": function_word_only_rate,
        "maximum_function_word_only_rate": max_function_word_only_rate,
        "sentinel_count": sentinel_count,
        "median_prediction_tokens": statistics.median(lengths) if lengths else 0,
        "generated_token_count_coverage": count_coverage,
        "max_new_tokens": max_new_tokens,
        "cap_hit_rate": cap_hit_rate,
        "maximum_cap_hit_rate": max_cap_hit_rate,
        "cap_hit_is_diagnostic_only": True,
        "repetition_loop_count": repetition_loop_count,
        "repetition_loop_rate": repetition_loop_rate,
        "maximum_repetition_loop_rate": max_repetition_loop_rate,
        "terminal_completeness_rate": terminal_rate,
        "terminal_required_count": len(terminal_outcomes),
        "terminal_question_policy": terminal_question_policy,
        "terminal_completeness_required": require_terminal_completeness,
        "minimum_terminal_completeness_rate": min_terminal_completeness_rate,
        "artifact_status": artifact_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--answers", type=Path, nargs="+", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-nonempty-rate", type=float, default=0.95)
    parser.add_argument("--min-unique-rate", type=float, default=0.10)
    parser.add_argument("--max-function-word-only-rate", type=float, default=0.01)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-cap-hit-rate", type=float, default=0.05)
    parser.add_argument("--max-repetition-loop-rate", type=float, default=0.01)
    parser.add_argument("--require-terminal-completeness", action="store_true")
    parser.add_argument("--min-terminal-completeness-rate", type=float, default=0.95)
    parser.add_argument(
        "--terminal-question-policy",
        choices=sorted(TERMINAL_QUESTION_POLICIES),
        default="all",
    )
    args = parser.parse_args()
    result = qualify(
        _load_json(args.manifest),
        _load_jsonl(args.answers),
        limit=args.limit,
        min_nonempty_rate=args.min_nonempty_rate,
        min_unique_rate=args.min_unique_rate,
        max_function_word_only_rate=args.max_function_word_only_rate,
        max_new_tokens=args.max_new_tokens,
        max_cap_hit_rate=args.max_cap_hit_rate,
        max_repetition_loop_rate=args.max_repetition_loop_rate,
        require_terminal_completeness=args.require_terminal_completeness,
        min_terminal_completeness_rate=args.min_terminal_completeness_rate,
        terminal_question_policy=args.terminal_question_policy,
    )
    result.update(
        protocol_version="oe-generation-qualification-v3-structural",
        manifest=str(args.manifest.resolve()),
        manifest_sha256=sha256_file(args.manifest),
        answers=[str(path.resolve()) for path in args.answers],
        answer_sha256=[sha256_file(path) for path in args.answers],
        evaluator_source=str(Path(__file__).resolve()),
        evaluator_source_sha256=sha256_file(Path(__file__).resolve()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
