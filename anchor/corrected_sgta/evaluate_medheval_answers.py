"""Auditable evaluation for decoded MedHEval close-ended answers.

The primary metric counts invalid/ambiguous generations as errors.  RULE's
POPE/ScienceQA conventions and parseable-only accuracy are diagnostics.
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from anchor.medeval.hashing import sha256_file

PROTOCOL_VERSION = "medheval-decoded-eval-v9-strict-primary-source-audited"
OFFICIAL_EVAL_ROOT = Path(__file__).resolve().parents[2] / "data/medheval/code/evaluation/close_ended_evaluation"
OFFICIAL_EVAL_SOURCES = (
    OFFICIAL_EVAL_ROOT / "utils/type1_utils.py",
    OFFICIAL_EVAL_ROOT / "utils/eval_yesno.py",
    OFFICIAL_EVAL_ROOT / "utils/eval_multichoice.py",
)
ANSWER_IS_RE = re.compile(r"\b(?:the\s+)?(?:correct\s+)?answer\s+(?:is|would\s+be)\s*[:\-]?\s*\(?([A-Z])\)?\b", re.I)
LEADING_LABEL_RE = re.compile(r"^\s*(?:option\s*)?\(?([A-Z])\)?(?:\s*[.),:-]|\s*$)", re.I)
STANDALONE_LABEL_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", re.I)
OPTION_RE = re.compile(r"(?:^|[,;]\s*|\s+)([A-Z])\s*[.:),]\s*(.*?)(?=(?:[,;]\s*|\s+)[A-Z]\s*[.:),]\s*|$)", re.I)
YES_RE = re.compile(r"\byes\b|\btrue\b", re.I)
NO_RE = re.compile(r"\bno\b|\bfalse\b", re.I)
LEADING_BINARY_RE = re.compile(
    r"^\s*[-*'\"`(\[]*\s*(?:answer\s*:\s*)?(yes|no)\b",
    re.I,
)
PRESENT_RE = re.compile(r"\bpresent\b|\bvisible\b|\bidentified\b|\bseen\b", re.I)
ABSENT_RE = re.compile(r"\babsent\b|\bnot\s+(?:present|visible|identified|seen)\b|\bwithout\b", re.I)
UNFINISHED_RE = re.compile(r"^\s*(?:the|the answer|answer|it is|this is|there is|there are)\s*[.:,-]?\s*$", re.I)
TERNARY_RE = re.compile(
    r"^\s*(?:answer\s*(?:is|:)?\s*)?(yes|no|maybe|uncertain|undetermined|unclear|equivocal)\b",
    re.I,
)


@dataclass(frozen=True)
class ParsedAnswer:
    labels: tuple[str, ...] | None
    status: str
    parser: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--questions", type=Path, help="Optional source questions for task/category metadata")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def _first_sentence(text: object) -> str:
    value = "" if text is None else str(text).replace("\n", " ").strip()
    return re.split(r"(?<=[.!?])\s+", value, maxsplit=1)[0].strip()


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_options(prompt: object) -> dict[str, str]:
    if isinstance(prompt, (list, tuple)):
        options: dict[str, str] = {}
        for index, item in enumerate(prompt):
            label = chr(ord("A") + index)
            value = str(item).strip()
            value = re.sub(
                rf"^\s*{label}\s*[.:),-]\s*", "", value, flags=re.I
            ).strip()
            if value:
                options[label] = value
        return options if len(options) >= 2 else {}
    text = str(prompt or "")
    marker = re.search(r"(?:options|choices|choose)\s*:\s*", text, re.I)
    option_text = text[marker.end():] if marker else text
    option_text = re.split(r"\n\s*Answer\s+with\b", option_text, maxsplit=1, flags=re.I)[0].strip()
    # Some released SLAKE rows contain label-only choices (``A, B, C, D``).
    # Treat these as four valid labels instead of the misleading A->B, C->D
    # pairs produced by the general option regex.
    label_only = re.fullmatch(
        r"\s*([A-Z])(?:\s*[,;/]\s*([A-Z]))+(?:\s*[,;/]\s*)?",
        option_text,
        re.I,
    )
    if label_only:
        labels = re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", option_text, re.I)
        return {label.upper(): label.upper() for label in labels}
    return {label.upper(): value.strip() for label, value in OPTION_RE.findall(option_text)}


def source_answer_type(row: dict[str, Any], ground_truth: object) -> str:
    """Trust MedHEval's source task type; only split explicit uncertainty."""

    source = str(row.get("source_question_type") or row.get("question_type") or "").strip().lower()
    if source in {"multi-choice", "multiple-choice", "multichoice", "mcq", "choice"}:
        return "choice"
    if source == "binary":
        return "ternary" if normalize_ternary_reference(ground_truth) == "maybe" else "binary"
    if source == "ternary":
        return "ternary"
    if source == "short_answer":
        return "short_answer"
    return infer_answer_type(ground_truth)


def official_binary_label(text: object) -> str:
    """Released MedHEval eval_yesno: first sentence, No/no/not else Yes."""

    sentence = "" if text is None else str(text)
    if "." in sentence:
        sentence = sentence.split(".", 1)[0]
    words = sentence.replace(",", "").split(" ")
    return "no" if any(word in {"No", "no", "not"} for word in words) else "yes"


def official_choice_options(choices: object) -> list[tuple[str, str]]:
    if isinstance(choices, (list, tuple)):
        return [
            (chr(ord("A") + index), f"{chr(ord('A') + index)}: {value}")
            for index, value in enumerate(choices)
        ]
    # Exact segmentation used by released type1_utils.split_choice.
    text = str(choices or "")
    matches = list(re.finditer(r"(?<!\w)([A-Z][:.])|([A-Z]\))", text))
    segments: list[str] = []
    previous = 0
    for match in matches:
        if previous < match.start():
            segments.append(text[previous:match.start()].strip(", "))
        previous = match.start()
    segments.append(text[previous:].strip(", "))
    return [
        (segment[0], segment)
        for segment in segments
        if segment and segment[0] in "ABCDEF"
    ]


def official_choice_label(text: object, choices: object) -> str | None:
    """Released eval_multichoice nearest-option SequenceMatcher mapping."""

    options = official_choice_options(choices)
    if len(options) < 2:
        return None
    target = "" if text is None else str(text)
    best_label = None
    best_similarity = 0.0
    for label, option in options:
        similarity = difflib.SequenceMatcher(None, option, target).ratio()
        if similarity > best_similarity:
            best_label = label
            best_similarity = similarity
    return best_label


def _legacy_semantic_binary(text: object) -> ParsedAnswer:
    """Historical whole-sentence parser retained only for discrepancy audits."""

    sentence = _first_sentence(text)
    if not sentence:
        return ParsedAnswer(None, "empty", "binary_semantic_legacy")
    if UNFINISHED_RE.fullmatch(sentence):
        return ParsedAnswer(None, "unfinished", "binary_semantic_legacy")
    absent = bool(ABSENT_RE.search(sentence))
    positive = bool(YES_RE.search(sentence)) or (bool(PRESENT_RE.search(sentence)) and not absent)
    negative = bool(NO_RE.search(sentence)) or absent
    if positive and negative:
        return ParsedAnswer(None, "ambiguous", "binary_semantic_legacy")
    if positive:
        return ParsedAnswer(("yes",), "parsed", "binary_semantic_legacy")
    if negative:
        return ParsedAnswer(("no",), "parsed", "binary_semantic_legacy")
    return ParsedAnswer(None, "unparsed", "binary_semantic_legacy")


def _binary(text: object) -> ParsedAnswer:
    """CE-G parser: the output must begin with an explicit Yes or No."""

    sentence = _first_sentence(text)
    if not sentence:
        return ParsedAnswer(None, "empty", "binary_leading_explicit")
    match = LEADING_BINARY_RE.search(sentence)
    if match is None:
        return ParsedAnswer(None, "invalid_no_leading_decision", "binary_leading_explicit")
    return ParsedAnswer((match.group(1).lower(),), "parsed", "binary_leading_explicit")


def binary_inconsistency(text: object, leading_label: str | None = None) -> bool:
    """Flag explanatory text that conflicts with a valid leading decision."""

    if leading_label is None:
        parsed = _binary(text)
        leading_label = parsed.labels[0] if parsed.labels else None
    if leading_label is None:
        return False
    legacy = _legacy_semantic_binary(text)
    legacy_label = legacy.labels[0] if legacy.labels else None
    return legacy.status == "ambiguous" or (
        legacy_label is not None and legacy_label != leading_label
    )


def _ternary(text: object) -> ParsedAnswer:
    sentence = _first_sentence(text)
    if not sentence:
        return ParsedAnswer(None, "empty", "ternary_leading_explicit")
    match = TERNARY_RE.search(sentence)
    if not match:
        return ParsedAnswer(None, "unparsed", "ternary_leading_explicit")
    label = match.group(1).lower()
    if label in {"uncertain", "undetermined", "unclear", "equivocal"}:
        label = "maybe"
    return ParsedAnswer((label,), "parsed", "ternary_leading_explicit")


def _choice(text: object, options: dict[str, str], ground_truth: object) -> ParsedAnswer:
    sentence = _first_sentence(text)
    if not sentence:
        return ParsedAnswer(None, "empty", "choice_explicit")
    if UNFINISHED_RE.fullmatch(sentence):
        return ParsedAnswer(None, "unfinished", "choice_explicit")
    valid = set(options) or set("ABCD")
    labels: list[str] = []
    explicit = ANSWER_IS_RE.search(sentence)
    if explicit and explicit.group(1).upper() in valid:
        labels.append(explicit.group(1).upper())
        tail = sentence[explicit.end():]
        if re.match(r"^\s*(?:,|and|&|/)", tail, re.I):
            labels.extend(x.upper() for x in STANDALONE_LABEL_RE.findall(tail))
    else:
        leading = LEADING_LABEL_RE.search(sentence)
        if leading and leading.group(1).upper() in valid:
            labels.append(leading.group(1).upper())
            labels.extend(x.upper() for x in STANDALONE_LABEL_RE.findall(sentence[leading.end():]))
    labels = list(dict.fromkeys(x for x in labels if x in valid))
    if labels:
        return ParsedAnswer(tuple(sorted(labels)), "parsed", "choice_label")
    normalized = _normal(sentence)
    exact_matches = [
        label for label, option in options.items()
        if _normal(option) and normalized == _normal(option)
    ]
    if len(exact_matches) == 1:
        return ParsedAnswer((exact_matches[0],), "parsed", "choice_text_exact")
    phrase_matches = [
        (label, len(_normal(option).split()))
        for label, option in options.items()
        if _normal(option) and f" {_normal(option)} " in f" {normalized} "
    ]
    if phrase_matches:
        longest = max(length for _, length in phrase_matches)
        winners = sorted(label for label, length in phrase_matches if length == longest)
        if len(winners) == 1:
            return ParsedAnswer((winners[0],), "parsed", "choice_text_longest_phrase")
        return ParsedAnswer(None, "ambiguous", "choice_text_longest_phrase")
    return ParsedAnswer(None, "unparsed", "choice_text")


def parse_choice_reference(text: object, choices: object) -> ParsedAnswer:
    """Resolve a benchmark choice reference while preserving official fallback.

    Explicit labels and exact/unique option text are deterministic.  Verbose
    source references are mapped with the released MedHEval nearest-option
    rule, and the parser name records that fallback for auditability.
    """

    options = parse_options(choices)
    parsed = _choice(text, options, text)
    if parsed.labels is not None:
        return ParsedAnswer(parsed.labels, "parsed", f"reference_{parsed.parser}")
    official = official_choice_label(text, choices)
    if official is not None and official in options:
        return ParsedAnswer(
            (official,), "parsed", "official_nearest_option_reference_fallback"
        )
    return ParsedAnswer(None, "invalid", "unresolved_choice_reference")


def _short_answer(text: object) -> ParsedAnswer:
    sentence = _first_sentence(text)
    value = _normal(sentence)
    if not value:
        return ParsedAnswer(None, "empty", "short_answer_normalized_exact")
    return ParsedAnswer((value,), "parsed", "short_answer_normalized_exact")


def infer_answer_type(ground_truth: object) -> str:
    value = _first_sentence(ground_truth)
    return (
        "binary"
        if _normal(value) in {"yes", "no", "true", "false", "present", "absent"}
        else "short_answer"
    )


def normalize_binary_reference(text: object) -> str | None:
    """Normalize dataset truth without relaxing the generated-answer contract."""

    sentence = _first_sentence(text)
    value = _normal(sentence)
    if value in {"yes", "true", "present"}:
        return "yes"
    if value in {"no", "false", "absent"}:
        return "no"
    # Several source datasets retain a short explanation in the reference
    # (for example, ``No (the silhouette is normal)``).  The truth contract is
    # still explicit-label-only: accept the same leading Yes/No syntax used by
    # CE-G, never infer truth from later clinical words.
    match = LEADING_BINARY_RE.search(sentence)
    if match is not None:
        return match.group(1).lower()
    return None


def normalize_ternary_reference(text: object) -> str | None:
    value = _normal(_first_sentence(text))
    if value in {"yes", "true", "present"}:
        return "yes"
    if value in {"no", "false", "absent"}:
        return "no"
    if value in {"maybe", "uncertain", "undetermined", "unclear", "equivocal"}:
        return "maybe"
    return None


def parse_answer(text: object, *, answer_type: str, prompt: object = "", ground_truth: object = "") -> ParsedAnswer:
    if answer_type == "binary":
        return _binary(text)
    if answer_type == "ternary":
        return _ternary(text)
    if answer_type == "choice":
        return _choice(text, parse_options(prompt), ground_truth)
    if answer_type == "short_answer":
        return _short_answer(text)
    raise ValueError(f"unsupported answer type: {answer_type}")


def rule_pope_prediction(text: object) -> str | None:
    """RULE/LLaVA POPE's first-sentence negative-word convention."""
    sentence = "" if text is None else str(text).replace("\n", " ").strip()
    if "." in sentence:
        sentence = sentence.split(".", 1)[0]
    words = sentence.replace(",", "").split(" ")
    return "no" if any(word in {"No", "no", "not"} for word in words) else "yes"


def _summary(n: int, correct: int, parseable: int) -> dict[str, Any]:
    return {
        "n": n,
        "correct": correct,
        "parseable": parseable,
        "parse_rate": parseable / n if n else 0.0,
        "accuracy_invalid_as_error": correct / n if n else 0.0,
        "accuracy_parseable_only": correct / parseable if parseable else 0.0,
    }


def _binary_metrics(confusion: Counter[str], invalid_positive: int = 0) -> dict[str, Any]:
    tp = confusion["yes->yes"]
    tn = confusion["no->no"]
    fp = confusion["no->yes"]
    fn = confusion["yes->no"] + invalid_positive
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _label(value: object) -> str | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    return "+".join(str(item) for item in value)


def _classification_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in details if _label(row["ground_truth"]) is not None]
    labels = sorted({_label(row["ground_truth"]) for row in valid})
    confusion: Counter[str] = Counter()
    for row in valid:
        truth = _label(row["ground_truth"])
        prediction = _label(row["prediction"]) or "__invalid__"
        confusion[f"{truth}->{prediction}"] += 1
    per_class = {}
    for label in labels:
        tp = confusion[f"{label}->{label}"]
        fn = sum(value for key, value in confusion.items() if key.startswith(f"{label}->") and key != f"{label}->{label}")
        fp = sum(value for key, value in confusion.items() if key.endswith(f"->{label}") and key != f"{label}->{label}")
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": tp + fn, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    return {
        "n": len(details),
        "valid_ground_truth_n": len(valid),
        "classes": labels,
        "accuracy_invalid_as_error": sum(bool(row["correct"]) for row in details) / len(details) if details else 0.0,
        "balanced_accuracy": sum(row["recall"] for row in per_class.values()) / len(per_class) if per_class else 0.0,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(per_class) if per_class else 0.0,
        "parse_rate": sum(_label(row["prediction"]) is not None and _label(row["ground_truth"]) is not None for row in details) / len(details) if details else 0.0,
        "per_class": per_class,
        "confusion": dict(sorted(confusion.items())),
    }


def _multiclass_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall accuracy, with class-balanced metrics only within task type."""

    by_type = {
        answer_type: _classification_metrics(
            [row for row in details if row["answer_type"] == answer_type]
        )
        for answer_type in ("binary", "ternary", "choice")
        if any(row["answer_type"] == answer_type for row in details)
    }
    return {
        "n": len(details),
        "valid_ground_truth_n": sum(_label(row["ground_truth"]) is not None for row in details),
        "classes": None,
        "accuracy_invalid_as_error": sum(bool(row["correct"]) for row in details) / len(details) if details else 0.0,
        "balanced_accuracy": None,
        "macro_f1": None,
        "parse_rate": sum(_label(row["prediction"]) is not None and _label(row["ground_truth"]) is not None for row in details) / len(details) if details else 0.0,
        "per_class": None,
        "confusion": None,
        "by_answer_type": by_type,
        "metric_scope": "balanced_accuracy and macro_f1 are undefined across heterogeneous answer spaces; use by_answer_type",
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _cluster_bootstrap(details: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        clusters.setdefault(str(row["cluster_id"]), []).append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    names = ("accuracy_invalid_as_error", "parse_rate")
    draws = {name: [] for name in names}
    for _ in range(replicates):
        sample = [row for _ in keys for row in clusters[rng.choice(keys)]]
        metrics = _multiclass_metrics(sample)
        for name in names:
            draws[name].append(float(metrics[name]))
    point = _multiclass_metrics(details)
    return {
        name: {"estimate": float(point[name]), "ci95_lower": _quantile(draws[name], 0.025), "ci95_upper": _quantile(draws[name], 0.975), "clusters": len(keys), "replicates": replicates, "seed": seed}
        for name in names
    }


def _official_accuracy_bootstrap(details: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any] | None:
    eligible = [row for row in details if row.get("official_benchmark_ground_truth") is not None]
    if not eligible:
        return None
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        clusters.setdefault(str(row["cluster_id"]), []).append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sample = [row for _ in keys for row in clusters[rng.choice(keys)]]
        draws.append(sum(bool(row["official_benchmark_correct"]) for row in sample) / len(sample))
    estimate = sum(bool(row["official_benchmark_correct"]) for row in eligible) / len(eligible)
    return {
        "estimate": estimate,
        "ci95_lower": _quantile(draws, 0.025),
        "ci95_upper": _quantile(draws, 0.975),
        "clusters": len(keys),
        "replicates": replicates,
        "seed": seed,
    }


def _clinical_cluster_id(row: dict[str, Any]) -> str:
    image_identity = str(row.get("img_name") or row.get("image") or "")
    patient_parts = [
        part[1:]
        for part in Path(image_identity).parts
        if part.startswith("p") and part[1:].isdigit()
    ]
    patient_from_path = max(patient_parts, key=len, default=None)
    return str(
        patient_from_path
        or row.get("patient_id")
        or row.get("image_sha256")
        or image_identity
        or row.get("question_id")
        or row.get("qid")
    )


def evaluate_rows(rows: Iterable[dict[str, Any]], answers_path: str | None = None) -> dict[str, Any]:
    rows = list(rows)
    details: list[dict[str, Any]] = []
    counts = Counter()
    by_type = {"binary": Counter(), "ternary": Counter(), "choice": Counter(), "short_answer": Counter()}
    confusion = Counter()
    by_hallucination: dict[str, Counter[str]] = {}
    pope_correct = pope_n = 0
    pope_confusion = Counter()
    official_counts = Counter()
    official_confusion = Counter()
    legacy_semantic_correct = legacy_semantic_parseable = 0
    legacy_semantic_confusion = Counter()
    inconsistency_count = 0
    for row in rows:
        gt_raw = row.get("gt_ans", row.get("gt_answer", row.get("ground_truth", row.get("answer"))))
        pred_raw = row.get("text", row.get("prediction", row.get("output")))
        if pred_raw is None and "gt_answer" in row:
            pred_raw = row.get("answer")
        answer_type = source_answer_type(row, gt_raw)
        if answer_type not in by_type:
            answer_type = infer_answer_type(gt_raw)
        prompt = row.get("prompt", row.get("question", ""))
        source_type = str(row.get("source_question_type") or row.get("question_type") or "").strip().lower()
        source_options = parse_options(row.get("choices", "")) or parse_options(prompt)
        if answer_type == "binary":
            normalized_gt = normalize_binary_reference(gt_raw)
            if normalized_gt is None and source_type == "binary":
                normalized_gt = official_binary_label(gt_raw)
            gt = (
                ParsedAnswer((normalized_gt,), "parsed", "normalized_binary_reference")
                if normalized_gt is not None
                else ParsedAnswer(None, "invalid", "normalized_binary_reference")
            )
        elif answer_type == "ternary":
            normalized_gt = normalize_ternary_reference(gt_raw)
            gt = (
                ParsedAnswer((normalized_gt,), "parsed", "normalized_ternary_reference")
                if normalized_gt is not None
                else ParsedAnswer(None, "invalid", "normalized_ternary_reference")
            )
        elif answer_type == "choice":
            gt = parse_choice_reference(gt_raw, row.get("choices", ""))
        else:
            gt = _short_answer(gt_raw)
        pred = (
            _choice(pred_raw, source_options, gt_raw)
            if answer_type == "choice"
            else parse_answer(pred_raw, answer_type=answer_type, prompt=prompt, ground_truth=gt_raw)
        )
        parsed = pred.labels is not None and gt.labels is not None
        correct = parsed and pred.labels == gt.labels
        counts["correct"] += int(correct)
        counts["parseable"] += int(parsed)
        counts[f"prediction_{pred.status}"] += 1
        counts["invalid_ground_truth"] += int(gt.labels is None)
        by_type[answer_type]["n"] += 1
        by_type[answer_type]["correct"] += int(correct)
        by_type[answer_type]["parseable"] += int(parsed)
        category = str(row.get("hallucination_type") or "unknown")
        category_counts = by_hallucination.setdefault(category, Counter())
        category_counts["n"] += 1
        category_counts["correct"] += int(correct)
        category_counts["parseable"] += int(parsed)
        if answer_type == "binary":
            legacy = _legacy_semantic_binary(pred_raw)
            legacy_label = legacy.labels[0] if legacy.labels else None
            leading_label = pred.labels[0] if pred.labels else None
            inconsistent = binary_inconsistency(pred_raw, leading_label)
            inconsistency_count += int(inconsistent)
            pope = rule_pope_prediction(pred_raw)
            pope_gt = rule_pope_prediction(gt_raw)
            pope_n += int(pope is not None)
            pope_correct += int(pope is not None and pope_gt is not None and pope == pope_gt)
            if pope is not None and pope_gt is not None:
                pope_confusion[f"{pope_gt}->{pope}"] += 1
            if gt.labels is not None:
                confusion[f"{gt.labels[0]}->{pred.labels[0] if pred.labels else 'invalid'}"] += 1
                legacy_semantic_parseable += int(legacy_label is not None)
                legacy_semantic_correct += int(legacy_label == gt.labels[0])
                legacy_semantic_confusion[
                    f"{gt.labels[0]}->{legacy_label if legacy_label is not None else 'invalid'}"
                ] += 1
        else:
            legacy_label = None
            inconsistent = False
        if source_type == "binary":
            official_gt = official_binary_label(gt_raw)
            official_prediction = official_binary_label(pred_raw)
        elif source_type in {"multi-choice", "multiple-choice", "multichoice", "mcq", "choice"}:
            official_gt = official_choice_label(gt_raw, row.get("choices", ""))
            official_prediction = official_choice_label(pred_raw, row.get("choices", ""))
        else:
            official_gt = official_prediction = None
        official_candidate = source_type == "binary" or source_type in {"multi-choice", "multiple-choice", "multichoice", "mcq", "choice"}
        official_eligible = official_gt is not None
        official_correct = official_eligible and official_prediction == official_gt
        official_counts["candidate_n"] += int(official_candidate)
        official_counts["invalid_source_choice_format"] += int(official_candidate and not official_eligible)
        official_counts["n"] += int(official_eligible)
        official_counts["correct"] += int(official_correct)
        official_counts["empty_predictions"] += int(official_eligible and not str(pred_raw or "").strip())
        if official_eligible:
            official_confusion[f"{source_type}:{official_gt}->{official_prediction or '__invalid__'}"] += 1
        details.append({
            "question_id": row.get("question_id", row.get("qid")),
            "cluster_id": _clinical_cluster_id(row),
            "modality": row.get("modality"),
            "source_dataset": row.get("source_dataset"),
            "answer_type": answer_type,
            "source_question_type": source_type,
            "prediction": None if pred.labels is None else list(pred.labels),
            "ground_truth": None if gt.labels is None else list(gt.labels),
            "parse_status": pred.status,
            "parser": pred.parser,
            "answer_inconsistency": inconsistent,
            "legacy_semantic_prediction": legacy_label,
            "correct": bool(correct),
            "official_benchmark_prediction": official_prediction,
            "official_benchmark_ground_truth": official_gt,
            "official_benchmark_correct": bool(official_correct),
            "text": pred_raw,
            "gt_ans": gt_raw,
        })
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_type": "real_gt_decoded_output",
        "primary_metric": "primary_multiclass.accuracy_invalid_as_error",
        "answers": answers_path,
        "decoded_strict": _summary(len(rows), counts["correct"], counts["parseable"]),
        "by_answer_type": {key: _summary(value["n"], value["correct"], value["parseable"]) for key, value in by_type.items() if value["n"]},
        "by_hallucination_type": {key: _summary(value["n"], value["correct"], value["parseable"]) for key, value in sorted(by_hallucination.items())},
        "by_modality": {
            modality: _summary(
                len(group),
                sum(bool(row["correct"]) for row in group),
                sum(row["prediction"] is not None and row["ground_truth"] is not None for row in group),
            )
            for modality in sorted(
                {str(row["modality"]) for row in details if row.get("modality")}
            )
            for group in [[row for row in details if str(row.get("modality")) == modality]]
        },
        "parse_failures": {key.removeprefix("prediction_"): value for key, value in sorted(counts.items()) if key.startswith("prediction_") and key != "prediction_parsed"},
        "invalid_ground_truth": counts["invalid_ground_truth"],
        "official_benchmark_proxy": {
            "n": official_counts["n"],
            "candidate_n": official_counts["candidate_n"],
            "invalid_source_choice_format_n": official_counts["invalid_source_choice_format"],
            "correct": official_counts["correct"],
            "accuracy": official_counts["correct"] / official_counts["n"] if official_counts["n"] else None,
            "empty_predictions_mapped_by_official_else_yes_rule": official_counts["empty_predictions"],
            "confusion": dict(sorted(official_confusion.items())),
            "semantics": "Released MedHEval compatibility diagnostic: binary first-sentence No/no/not else Yes; multi-choice nearest option by SequenceMatcher for both GT and prediction. This proxy is not the primary metric because it maps empty/invalid generations to a class.",
            "diagnostic_only": True,
        },
        "rule_compatible_binary_diagnostic": {
            "n": pope_n,
            "correct": pope_correct,
            "accuracy": pope_correct / pope_n if pope_n else None,
            **_binary_metrics(pope_confusion),
            "note": "RULE/LLaVA POPE first-sentence convention; diagnostic only.",
        },
        "strict_binary_metrics": _binary_metrics(
            confusion, invalid_positive=confusion["yes->invalid"]
        ),
        "binary_confusion": dict(sorted(confusion.items())),
        "details": details,
    }
    binary_n = by_type["binary"]["n"]
    report["ce_g_leading"] = {
        **_summary(binary_n, by_type["binary"]["correct"], by_type["binary"]["parseable"]),
        "parser": "leading explicit yes/no; invalid generations count as errors",
        "answer_inconsistency_count": inconsistency_count,
        "answer_inconsistency_rate": inconsistency_count / binary_n if binary_n else 0.0,
    }
    report["legacy_semantic_diagnostic"] = {
        **_summary(binary_n, legacy_semantic_correct, legacy_semantic_parseable),
        "confusion": dict(sorted(legacy_semantic_confusion.items())),
        "diagnostic_only": True,
    }
    report.update(_summary(len(rows), counts["correct"], counts["parseable"]))
    return report


def _load_questions(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text()
    payload = json.loads(text) if path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
    output: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(payload):
        qid = str(row.get("qid", row.get("id", row.get("question_id", index))))
        if qid in output:
            raise ValueError(f"duplicate question id {qid!r}")
        output[qid] = row
    return output


def align_answers_with_questions(
    rows: list[dict[str, Any]], source: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bind decoded outputs to authoritative manifest labels, fail closed."""

    observed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        qid = str(row.get("question_id", row.get("qid", row.get("id", index))))
        if qid in observed:
            raise ValueError(f"duplicate answer question id {qid!r}")
        observed[qid] = row
    missing = sorted(set(source) - set(observed))
    extra = sorted(set(observed) - set(source))
    if missing or extra:
        raise ValueError(
            f"answer alignment failure: missing={missing[:10]} extra={extra[:10]}"
        )
    merged: list[dict[str, Any]] = []
    for qid, question in source.items():
        row = dict(observed[qid])
        manifest_gt = question.get(
            "answer", question.get("gt_ans", question.get("ground_truth"))
        )
        embedded_gt = row.get(
            "gt_ans", row.get("gt_answer", row.get("ground_truth"))
        )
        if manifest_gt is None:
            raise ValueError(f"manifest question {qid!r} has no ground truth")
        if embedded_gt is not None and _normal(str(embedded_gt)) != _normal(str(manifest_gt)):
            raise ValueError(f"embedded reference mismatch for {qid!r}")
        row["gt_ans"] = manifest_gt
        row["question_id"] = qid
        for key in (
            "question", "question_type", "source_question_type",
            "hallucination_type", "choices", "source", "patient_id",
            "image_sha256", "img_name", "image", "modality", "source_dataset",
        ):
            if key in question:
                row[key] = question[key]
        merged.append(row)
    return merged


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    with args.answers.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.questions:
        source = _load_questions(args.questions)
        rows = align_answers_with_questions(rows, source)
    report = evaluate_rows(rows, str(args.answers))
    report["answers_sha256"] = sha256_file(args.answers)
    report["questions"] = str(args.questions.resolve()) if args.questions else None
    report["questions_sha256"] = sha256_file(args.questions) if args.questions else None
    report["evaluator_source"] = str(Path(__file__).resolve())
    report["evaluator_source_sha256"] = sha256_file(Path(__file__).resolve())
    report["official_medheval_source_provenance"] = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in OFFICIAL_EVAL_SOURCES
    ]
    report["primary_multiclass"] = _multiclass_metrics(report["details"])
    report["cluster_bootstrap_ci95"] = _cluster_bootstrap(report["details"], args.bootstrap_replicates, args.bootstrap_seed)
    report["official_benchmark_proxy"]["cluster_bootstrap_ci95"] = _official_accuracy_bootstrap(
        report["details"], args.bootstrap_replicates, args.bootstrap_seed
    )
    report["bootstrap_contract"] = {"unit": "longest MIMIC patient path component, else patient_id, image hash/path, or qid", "replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
