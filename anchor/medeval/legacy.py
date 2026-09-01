"""Conservative evidence grading for historical answer files."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .hashing import sha256_file


ID_KEYS = ("sample_id", "question_id", "id", "qid")
TEXT_KEYS = ("prediction", "text", "answer", "output")
FUNCTION_WORD_ONLY = {
    "a", "an", "the", "this", "that", "these", "those", "it", "there",
    "in", "on", "at", "to", "of", "for", "from", "with", "and", "or",
}


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def audit_legacy_answers(
    answers_path: Path,
    expected_ids: Iterable[str],
    exact_run_fingerprint: str | None = None,
    companion_manifest: Path | None = None,
    allow_short_answers: bool = False,
    enforce_behavioral_quality: bool = True,
) -> dict[str, Any]:
    expected = [str(item) for item in expected_ids]
    parse_errors: list[int] = []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(answers_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError
            rows.append(row)
        except (json.JSONDecodeError, TypeError):
            parse_errors.append(number)
    ids = [str(_first(row, ID_KEYS)) for row in rows]
    texts = [str(_first(row, TEXT_KEYS) or "").strip() for row in rows]
    counts = Counter(ids)
    duplicates = sorted(item for item, count in counts.items() if count > 1)
    missing = sorted(set(expected) - set(ids))
    extra = sorted(set(ids) - set(expected))
    empty = sum(not text for text in texts)
    word_lengths = [len(text.split()) for text in texts]
    unique_predictions = len(set(texts))
    dominant_fraction = max(Counter(texts).values(), default=0) / max(len(texts), 1)
    function_word_only = sum(
        " ".join(text.lower().split()).strip(".,:;!?()[]{}") in FUNCTION_WORD_ONLY
        for text in texts
    )
    function_word_only_fraction = function_word_only / max(len(texts), 1)
    behavioral_warnings: list[str] = []
    if texts and statistics.median(word_lengths) <= 1 and not allow_short_answers:
        behavioral_warnings.append("median_prediction_at_most_one_word")
    if len(texts) >= 20 and dominant_fraction >= 0.90:
        behavioral_warnings.append("one_prediction_dominates_at_least_90_percent")
    if len(texts) >= 20 and function_word_only_fraction >= 0.50:
        behavioral_warnings.append("function_word_only_predictions_at_least_50_percent")
    degenerate_reasons = list(behavioral_warnings) if enforce_behavioral_quality else []
    if empty:
        degenerate_reasons.append("empty_predictions")
    aligned = not (parse_errors or duplicates or missing or extra) and len(rows) == len(expected)
    manifest_matches = False
    if exact_run_fingerprint and companion_manifest and companion_manifest.exists():
        payload = json.loads(companion_manifest.read_text())
        manifest_matches = payload.get("run_fingerprint") == exact_run_fingerprint
    if aligned and not degenerate_reasons and manifest_matches:
        grade, action = "A", "reuse"
    elif aligned and not degenerate_reasons:
        grade, action = "B", "rescore_only"
    else:
        grade, action = "C", "rerun"
    return {
        "answers_path": str(answers_path.resolve()),
        "answers_sha256": sha256_file(answers_path),
        "grade": grade,
        "action": action,
        "aligned": aligned,
        "expected": len(expected),
        "observed": len(rows),
        "parse_error_lines": parse_errors,
        "duplicates": duplicates,
        "missing": missing,
        "extra": extra,
        "empty_predictions": empty,
        "median_words": statistics.median(word_lengths) if word_lengths else 0,
        "unique_predictions": unique_predictions,
        "dominant_prediction_fraction": dominant_fraction,
        "function_word_only_predictions": function_word_only,
        "function_word_only_fraction": function_word_only_fraction,
        "allow_short_answers": allow_short_answers,
        "enforce_behavioral_quality": enforce_behavioral_quality,
        "behavioral_warnings": behavioral_warnings,
        "degenerate_reasons": degenerate_reasons,
        "manifest_matches": manifest_matches,
    }
