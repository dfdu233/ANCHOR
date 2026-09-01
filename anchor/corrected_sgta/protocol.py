"""Shared MedHEval parsing, prompting, judging, and cache provenance."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROTOCOL_VERSION = "medheval-sgta-v5.0"


def _first_existing(*values: str) -> Path:
    candidates = [Path(value) for value in values]
    return next((path for path in candidates if path.exists()), candidates[0])


IMAGE_ROOT = _first_existing(
    "/home/dbw/ANCHOR/data/medheval/images",
    "/root/autodl-tmp/MedHEval/images",
)
IMAGE_SUBDIRS = ("IU-Xray", "Slake", "VQA-RAD")

# A marker is accepted at the beginning or after an option separator.  This
# supports both forms present in MedHEval:
#   A. foo, B. bar       and       A, foo; B, bar
_CHOICE_MARKER = re.compile(r"(?:^|(?<=[,;\n]))\s*([A-Fa-f])\s*[\.\)\]:,-]\s*")
_PRED_LETTER = re.compile(r"^\s*(?:answer\s*(?:is|:)?\s*)?([A-F])(?:\b|[\.\):])", re.I)
_PRED_BINARY = re.compile(
    r"^\s*(?:answer\s*(?:is|:)?\s*)?(yes|no)(?:\b|[\.\),:])", re.I
)


class ProtocolError(ValueError):
    """Raised for malformed benchmark rows instead of silently guessing."""


@dataclass(frozen=True)
class ChoiceSet:
    labels: tuple[str, ...]
    texts: tuple[str, ...]

    def index_for_answer(self, answer: str) -> int:
        value = str(answer).strip()
        letter = _PRED_LETTER.match(value)
        if letter and letter.group(1).upper() in self.labels:
            return self.labels.index(letter.group(1).upper())

        normalized = normalize_text(value)
        exact = [
            i for i, text in enumerate(self.texts) if normalize_text(text) == normalized
        ]
        if len(exact) == 1:
            return exact[0]
        raise ProtocolError(f"answer does not identify exactly one option: {answer!r}")


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def parse_choices(raw: Any) -> ChoiceSet:
    """Parse MedHEval options without splitting option text on every comma.

    A list/tuple is also accepted for compatibility with other MedUniEval
    datasets.  Malformed choices fail closed; evaluation never falls back to
    fuzzy matching against a guessed option.
    """

    if raw is None or raw == "":
        return ChoiceSet((), ())
    if isinstance(raw, (list, tuple)):
        labels: list[str] = []
        texts: list[str] = []
        for i, item in enumerate(raw):
            value = str(item).strip()
            match = re.match(r"^\s*([A-Fa-f])\s*[\.\)\]:,-]\s*(.+?)\s*$", value)
            if match:
                label, text = match.group(1).upper(), match.group(2)
            else:
                label, text = chr(ord("A") + i), value
            labels.append(label)
            texts.append(text.rstrip(" ,;"))
        return _validated_choices(labels, texts, raw)
    if not isinstance(raw, str):
        raise ProtocolError(f"unsupported choices type: {type(raw).__name__}")

    matches = list(_CHOICE_MARKER.finditer(raw.strip()))
    labels: list[str] = []
    texts: list[str] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        labels.append(match.group(1).upper())
        texts.append(raw[start:end].strip().rstrip(" ,;"))
    return _validated_choices(labels, texts, raw)


def _validated_choices(
    labels: Sequence[str], texts: Sequence[str], raw: Any
) -> ChoiceSet:
    if len(labels) < 2 or len(labels) != len(texts):
        raise ProtocolError(f"could not parse at least two choices: {raw!r}")
    if len(set(labels)) != len(labels) or any(not text for text in texts):
        raise ProtocolError(f"duplicate/empty choices: {raw!r}")
    expected = [chr(ord("A") + i) for i in range(len(labels))]
    if list(labels) != expected:
        raise ProtocolError(f"choices must be contiguous from A: {raw!r}")
    return ChoiceSet(tuple(labels), tuple(texts))


def task_kind(sample: dict[str, Any]) -> str:
    qtype = str(
        sample.get("question_type") or sample.get("ground_truth_type") or ""
    ).lower()
    choices = sample.get("choices")
    if (
        qtype in {"multi-choice", "multi_choice", "multiple_choice"}
        or (isinstance(choices, str) and choices.strip())
        or isinstance(choices, (list, tuple))
    ):
        return "multichoice"
    if qtype in {"binary", "binary_ce", "binary_question"}:
        return "binary"
    return "open"


def labels_for_sample(sample: dict[str, Any]) -> tuple[str, ...]:
    kind = task_kind(sample)
    if kind == "binary":
        return ("Yes", "No")
    if kind == "multichoice":
        return parse_choices(sample.get("choices")).labels
    raise ProtocolError("open-ended rows do not have a finite answer label set")


def ground_truth_index(sample: dict[str, Any]) -> int:
    answer = str(
        sample.get("answer", sample.get("gt", sample.get("gt_ans", "")))
    ).strip()
    kind = task_kind(sample)
    if kind == "binary":
        match = _PRED_BINARY.match(answer)
        if not match:
            raise ProtocolError(f"invalid binary ground truth: {answer!r}")
        return 0 if match.group(1).lower() == "yes" else 1
    if kind == "multichoice":
        return parse_choices(sample.get("choices")).index_for_answer(answer)
    raise ProtocolError("open-ended rows do not have a class index")


def prediction_index(text: str, sample: dict[str, Any]) -> int | None:
    """Strictly parse a generated answer; return None instead of guessing."""

    kind = task_kind(sample)
    if kind == "binary":
        match = _PRED_BINARY.match(str(text))
        return None if not match else (0 if match.group(1).lower() == "yes" else 1)
    if kind == "multichoice":
        choices = parse_choices(sample.get("choices"))
        match = _PRED_LETTER.match(str(text))
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
        return f"{question}\nAnswer with exactly one word: Yes or No."
    if kind == "multichoice":
        choices = parse_choices(sample.get("choices"))
        options = "\n".join(
            f"{label}. {text}" for label, text in zip(choices.labels, choices.texts)
        )
        return f"{question}\n{options}\nAnswer with the option letter only."
    return question


def resolve_image(name: str, image_root: Path = IMAGE_ROOT) -> Path | None:
    value = str(name or "")
    direct = Path(value)
    candidates = [
        direct,
        image_root / value,
        *(image_root / sub / value for sub in IMAGE_SUBDIRS),
    ]
    return next((path for path in candidates if path.is_file()), None)


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


def deterministic_split(
    qids: Sequence[Any], calibration_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Stable split independent of input row order and Python hash randomization."""

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    keyed = []
    for qid in map(str, qids):
        digest = hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()
        keyed.append((digest, qid))
    ordered = [qid for _, qid in sorted(keyed)]
    n_cal = max(1, min(len(ordered) - 1, round(len(ordered) * calibration_fraction)))
    return ordered[:n_cal], ordered[n_cal:]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_fingerprint(config: dict[str, Any]) -> str:
    payload = {"protocol_version": PROTOCOL_VERSION, **config}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
