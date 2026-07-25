"""Auditable evaluation for decoded MedHEval close-ended answers.

The primary metric counts invalid/ambiguous generations as errors.  RULE's
POPE/ScienceQA conventions and parseable-only accuracy are diagnostics.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROTOCOL_VERSION = "medheval-decoded-eval-v4"
ANSWER_IS_RE = re.compile(r"\b(?:the\s+)?(?:correct\s+)?answer\s+(?:is|would\s+be)\s*[:\-]?\s*\(?([A-Z])\)?\b", re.I)
LEADING_LABEL_RE = re.compile(r"^\s*(?:option\s*)?\(?([A-Z])\)?(?:\s*[.),:-]|\s*$)", re.I)
STANDALONE_LABEL_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", re.I)
OPTION_RE = re.compile(r"(?:^|,\s+)([A-Z])[.:)]\s*(.*?)(?=,\s+[A-Z][.:)]\s|$)", re.I)
YES_RE = re.compile(r"\byes\b|\btrue\b", re.I)
NO_RE = re.compile(r"\bno\b|\bfalse\b", re.I)
PRESENT_RE = re.compile(r"\bpresent\b|\bvisible\b|\bidentified\b|\bseen\b", re.I)
ABSENT_RE = re.compile(r"\babsent\b|\bnot\s+(?:present|visible|identified|seen)\b|\bwithout\b", re.I)
UNFINISHED_RE = re.compile(r"^\s*(?:the|the answer|answer|it is|this is|there is|there are)\s*[.:,-]?\s*$", re.I)


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
    return parser.parse_args()


def _first_sentence(text: object) -> str:
    value = "" if text is None else str(text).replace("\n", " ").strip()
    return re.split(r"(?<=[.!?])\s+", value, maxsplit=1)[0].strip()


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_options(prompt: object) -> dict[str, str]:
    text = str(prompt or "")
    marker = re.search(r"(?:options|choices|choose)\s*:\s*", text, re.I)
    option_text = text[marker.end():] if marker else text
    return {label.upper(): value.strip() for label, value in OPTION_RE.findall(option_text)}


def _binary(text: object) -> ParsedAnswer:
    sentence = _first_sentence(text)
    if not sentence:
        return ParsedAnswer(None, "empty", "binary_explicit")
    if UNFINISHED_RE.fullmatch(sentence):
        return ParsedAnswer(None, "unfinished", "binary_explicit")
    absent = bool(ABSENT_RE.search(sentence))
    positive = bool(YES_RE.search(sentence)) or (bool(PRESENT_RE.search(sentence)) and not absent)
    negative = bool(NO_RE.search(sentence)) or absent
    if positive and negative:
        return ParsedAnswer(None, "ambiguous", "binary_explicit")
    if positive:
        return ParsedAnswer(("yes",), "parsed", "binary_explicit")
    if negative:
        return ParsedAnswer(("no",), "parsed", "binary_explicit")
    return ParsedAnswer(None, "unparsed", "binary_explicit")


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
    matches = [label for label, option in options.items() if _normal(option) and (normalized == _normal(option) or _normal(option) in normalized)]
    if matches:
        return ParsedAnswer(tuple(sorted(matches)), "parsed", "choice_text")
    return ParsedAnswer(None, "unparsed", "choice_text")


def infer_answer_type(ground_truth: object) -> str:
    value = _first_sentence(ground_truth)
    parsed = _binary(value)
    return "binary" if parsed.labels is not None and _normal(value) in {"yes", "no", "true", "false", "present", "absent"} else "choice"


def parse_answer(text: object, *, answer_type: str, prompt: object = "", ground_truth: object = "") -> ParsedAnswer:
    if answer_type == "binary":
        return _binary(text)
    if answer_type == "choice":
        return _choice(text, parse_options(prompt), ground_truth)
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


def evaluate_rows(rows: Iterable[dict[str, Any]], answers_path: str | None = None) -> dict[str, Any]:
    rows = list(rows)
    details: list[dict[str, Any]] = []
    counts = Counter()
    by_type = {"binary": Counter(), "choice": Counter()}
    confusion = Counter()
    by_hallucination: dict[str, Counter[str]] = {}
    pope_correct = pope_n = 0
    pope_confusion = Counter()
    for row in rows:
        gt_raw = row.get("gt_ans", row.get("gt_answer", row.get("ground_truth", row.get("answer"))))
        pred_raw = row.get("text", row.get("prediction", row.get("output")))
        if pred_raw is None and "gt_answer" in row:
            pred_raw = row.get("answer")
        answer_type = str(row.get("question_type") or infer_answer_type(gt_raw)).lower()
        answer_type = "choice" if answer_type in {"multi-choice", "multiple-choice", "mcq"} else answer_type
        if answer_type not in by_type:
            answer_type = infer_answer_type(gt_raw)
        prompt = row.get("prompt", row.get("question", ""))
        normalized_prompt = _normal(str(prompt))
        if "answer yes or no" in normalized_prompt or "two options yes no" in normalized_prompt:
            answer_type = "binary"
        elif parse_options(prompt):
            answer_type = "choice"
        gt = parse_answer(gt_raw, answer_type=answer_type, prompt=prompt, ground_truth=gt_raw)
        if answer_type == "binary" and gt.labels is None:
            official_gt = rule_pope_prediction(gt_raw)
            gt = ParsedAnswer((official_gt,), "parsed", "official_binary_ground_truth") if official_gt else gt
        pred = parse_answer(pred_raw, answer_type=answer_type, prompt=prompt, ground_truth=gt_raw)
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
            pope = rule_pope_prediction(pred_raw)
            pope_gt = rule_pope_prediction(gt_raw)
            pope_n += int(pope is not None)
            pope_correct += int(pope is not None and pope_gt is not None and pope == pope_gt)
            if pope is not None and pope_gt is not None:
                pope_confusion[f"{pope_gt}->{pope}"] += 1
            if gt.labels is not None:
                confusion[f"{gt.labels[0]}->{pred.labels[0] if pred.labels else 'invalid'}"] += 1
        details.append({
            "question_id": row.get("question_id", row.get("qid")),
            "answer_type": answer_type,
            "prediction": None if pred.labels is None else list(pred.labels),
            "ground_truth": None if gt.labels is None else list(gt.labels),
            "parse_status": pred.status,
            "parser": pred.parser,
            "correct": bool(correct),
            "text": pred_raw,
            "gt_ans": gt_raw,
        })
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_type": "real_gt_decoded_output",
        "primary_metric": "decoded_strict.accuracy_invalid_as_error",
        "answers": answers_path,
        "decoded_strict": _summary(len(rows), counts["correct"], counts["parseable"]),
        "by_answer_type": {key: _summary(value["n"], value["correct"], value["parseable"]) for key, value in by_type.items() if value["n"]},
        "by_hallucination_type": {key: _summary(value["n"], value["correct"], value["parseable"]) for key, value in sorted(by_hallucination.items())},
        "parse_failures": {key.removeprefix("prediction_"): value for key, value in sorted(counts.items()) if key.startswith("prediction_") and key != "prediction_parsed"},
        "invalid_ground_truth": counts["invalid_ground_truth"],
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
    report.update(_summary(len(rows), counts["correct"], counts["parseable"]))
    return report


def _load_questions(path: Path) -> dict[object, dict[str, Any]]:
    text = path.read_text()
    payload = json.loads(text) if path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
    return {row.get("qid", row.get("id", row.get("question_id"))): row for row in payload}


def main() -> None:
    args = parse_args()
    with args.answers.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.questions:
        source = _load_questions(args.questions)
        for row in rows:
            question = source.get(row.get("question_id", row.get("qid")), {})
            for key in ("question_type", "hallucination_type", "choices", "source"):
                if key in question:
                    row[key] = question[key]
    report = evaluate_rows(rows, str(args.answers))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
