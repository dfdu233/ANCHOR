"""Strict MedHEval protocol with deterministic recovery for malformed options.

This supersedes :mod:`corrected_sgta.protocol`.  It is separate only because
the current host kernel cannot update existing files through the patch helper.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .protocol import (
    IMAGE_ROOT,
    ProtocolError,
    deterministic_split,
    file_sha256,
    resolve_image,
)

PROTOCOL_VERSION = "medheval-sgta-v5.3"
CACHE_SCHEMA_VERSION = "sgta-evidence-cache-v5.5"

_MARKER = re.compile(r"(?:^|(?<=[,;\n])|(?<=\s))\s*([A-Fa-f])\s*[\.\)\]:,]\s*")
_QUESTION_MARKER = re.compile(r"\(([A-Fa-f])\)\s*")
_LETTER = re.compile(r"^\s*(?:answer\s*(?:is|:)?\s*)?([A-F])(?:\b|[\.\):])", re.I)
_BINARY = re.compile(r"^\s*(?:answer\s*(?:is|:)?\s*)?(yes|no)(?:\b|[\.\),:])", re.I)


def protocol_fingerprint(config: dict[str, Any]) -> str:
    payload = {"protocol_version": PROTOCOL_VERSION, **config}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


@dataclass(frozen=True)
class ChoiceSet:
    labels: tuple[str, ...]
    texts: tuple[str, ...]

    def index_for_answer(self, answer: str) -> int:
        value = str(answer).strip()
        letter = _LETTER.match(value)
        if letter and letter.group(1).upper() in self.labels:
            return self.labels.index(letter.group(1).upper())
        normalized = normalize_text(value)
        exact = [
            i for i, text in enumerate(self.texts) if normalize_text(text) == normalized
        ]
        if len(exact) == 1:
            return exact[0]
        # Full-sentence GT is accepted only if exactly one complete normalized
        # option occurs in it. No edit-distance or nearest-option guessing.
        contained = []
        for i, text in enumerate(self.texts):
            option = normalize_text(text)
            if option and re.search(
                rf"(?:^|\s){re.escape(option)}(?:$|\s)", normalized
            ):
                contained.append(i)
        if len(contained) == 1:
            return contained[0]
        raise ProtocolError(f"answer does not identify exactly one option: {answer!r}")


def _validated(labels: Sequence[str], texts: Sequence[str], raw: Any) -> ChoiceSet:
    if len(labels) < 2 or len(labels) != len(texts):
        raise ProtocolError(f"could not parse at least two choices: {raw!r}")
    if len(set(labels)) != len(labels) or any(not text for text in texts):
        raise ProtocolError(f"duplicate/empty choices: {raw!r}")
    expected = [chr(ord("A") + i) for i in range(len(labels))]
    if list(labels) != expected:
        raise ProtocolError(f"choices must be contiguous from A: {raw!r}")
    return ChoiceSet(tuple(labels), tuple(texts))


def _parse_marked(raw: str, pattern: re.Pattern[str]) -> ChoiceSet:
    candidates = list(pattern.finditer(raw.strip()))
    matches = []
    expected = "A"
    for candidate in candidates:
        if candidate.group(1).upper() == expected:
            matches.append(candidate)
            expected = chr(ord(expected) + 1)
    labels: list[str] = []
    texts: list[str] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        labels.append(match.group(1).upper())
        texts.append(raw[start:end].strip().rstrip(" ,;?."))
    return _validated(labels, texts, raw)


def parse_choices(raw: Any) -> ChoiceSet:
    if raw is None or raw == "":
        return ChoiceSet((), ())
    if isinstance(raw, (list, tuple)):
        labels: list[str] = []
        texts: list[str] = []
        for i, item in enumerate(raw):
            expected_label = chr(ord("A") + i)
            value = str(item).strip()
            # List order is authoritative.  A choice's clinical/scientific
            # text may itself begin with e.g. ``C. difficile``; interpreting
            # arbitrary leading A-F punctuation as a label corrupts such
            # benchmark rows.  Strip only the prefix matching this position.
            match = re.match(
                rf"^\s*({expected_label})\s*[\.\)\]:,]\s*(.+?)\s*$",
                value,
                re.I,
            )
            if match:
                labels.append(expected_label)
                texts.append(match.group(2).rstrip(" ,;"))
            else:
                labels.append(expected_label)
                texts.append(value)
        return _validated(labels, texts, raw)
    if not isinstance(raw, str):
        raise ProtocolError(f"unsupported choices type: {type(raw).__name__}")
    return _parse_marked(raw, _MARKER)


def choices_for_sample(sample: dict[str, Any]) -> ChoiceSet:
    try:
        return parse_choices(sample.get("choices"))
    except ProtocolError as explicit_error:
        try:
            return _parse_marked(str(sample.get("question", "")), _QUESTION_MARKER)
        except ProtocolError:
            raise explicit_error


def task_kind(sample: dict[str, Any]) -> str:
    qtype = str(
        sample.get("question_type") or sample.get("ground_truth_type") or ""
    ).lower()
    choices = sample.get("choices")
    if qtype in {"multi-choice", "multi_choice", "multiple_choice"}:
        return "multichoice"
    if qtype in {"binary", "binary_ce", "binary_question"}:
        # A few corrupted rows are tagged binary while carrying explicit MC
        # options. Prefer the observable option set over the bad tag.
        if isinstance(choices, str) and choices.strip():
            return "multichoice"
        return "binary"
    if (isinstance(choices, str) and choices.strip()) or (
        isinstance(choices, (list, tuple)) and len(choices) > 0
    ):
        return "multichoice"
    return "open"


def labels_for_sample(sample: dict[str, Any]) -> tuple[str, ...]:
    kind = task_kind(sample)
    if kind == "binary":
        return ("Yes", "No")
    if kind == "multichoice":
        return choices_for_sample(sample).labels
    raise ProtocolError("open-ended rows do not have a finite answer label set")


def ground_truth_index(sample: dict[str, Any]) -> int:
    answer = str(
        sample.get("answer", sample.get("gt", sample.get("gt_ans", "")))
    ).strip()
    kind = task_kind(sample)
    if kind == "binary":
        match = _BINARY.match(answer)
        if not match:
            # Candidate labels cannot be derived from a single GT at test time;
            # exclude non-Yes/No rows rather than leaking the answer vocabulary.
            raise ProtocolError(f"invalid Yes/No ground truth: {answer!r}")
        return 0 if match.group(1).lower() == "yes" else 1
    if kind == "multichoice":
        return choices_for_sample(sample).index_for_answer(answer)
    raise ProtocolError("open-ended rows do not have a class index")


def prediction_index(text: str, sample: dict[str, Any]) -> int | None:
    kind = task_kind(sample)
    if kind == "binary":
        match = _BINARY.match(str(text))
        return None if not match else (0 if match.group(1).lower() == "yes" else 1)
    if kind == "multichoice":
        choices = choices_for_sample(sample)
        match = _LETTER.match(str(text))
        if match and match.group(1).upper() in choices.labels:
            return choices.labels.index(match.group(1).upper())
        normalized = normalize_text(text)
        exact = [
            i
            for i, option in enumerate(choices.texts)
            if normalize_text(option) == normalized
        ]
        return exact[0] if len(exact) == 1 else None
    return None


def build_prompt(sample: dict[str, Any]) -> str:
    question = str(sample["question"]).strip()
    kind = task_kind(sample)
    if kind == "binary":
        return f"{question} Please answer Yes or No."
    if kind == "multichoice":
        choices = choices_for_sample(sample)
        # Remove inline question choices before adding one normalized list only
        # when recovery came from the question itself.
        options = "\n".join(
            f"{label}. {text}" for label, text in zip(choices.labels, choices.texts)
        )
        return f"{question}\n{options}\nAnswer with the option letter only."
    return question


def validate_dataset(
    rows: Iterable[dict[str, Any]], require_images: bool = True
) -> dict[str, Any]:
    counts = {
        "binary": 0,
        "multichoice": 0,
        "open": 0,
        "missing_image": 0,
        "invalid": 0,
    }
    errors: list[str] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows):
        try:
            qid = str(row["qid"])
            if qid in seen:
                raise ProtocolError(f"duplicate qid {qid}")
            seen.add(qid)
            kind = task_kind(row)
            counts[kind] += 1
            if kind != "open":
                labels_for_sample(row)
                ground_truth_index(row)
            if require_images and resolve_image(row.get("img_name", "")) is None:
                counts["missing_image"] += 1
        except (KeyError, ProtocolError) as exc:
            counts["invalid"] += 1
            if len(errors) < 20:
                errors.append(f"row {row_number}: {exc}")
    return {**counts, "n": len(seen), "errors": errors}
