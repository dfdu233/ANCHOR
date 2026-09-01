#!/usr/bin/env python3
"""Strict, auditable evaluation of RULE VQA generations.

The common CE-G primary metric requires a leading explicit Yes/No decision.
RULE/POPE negative-word conventions and semantic fallbacks remain diagnostics
for paper-native reproduction and historical discrepancy analysis only.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from corrected_sgta.evaluate_medheval_answers import (
    parse_answer,
    rule_pope_prediction,
)

PROTOCOL_VERSION = "rule-vqa-leading-ceg-v6"


class RuleEvaluationError(ValueError):
    """Raised when answer and question artifacts cannot be joined exactly."""


def _qid(row: dict[str, Any]) -> str:
    value = row.get("question_id", row.get("qid", row.get("id")))
    if value is None:
        raise RuleEvaluationError("row has no question_id/qid/id")
    return str(value)


def _prediction_text(row: dict[str, Any]) -> str:
    for key in ("answer", "text", "prediction", "output"):
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def parse_rule_ground_truth(text: object) -> tuple[str | None, str]:
    """Parse only an explicit leading yes/no label from RULE ground truth."""
    value = "" if text is None else str(text)
    match = re.match(r"^\s*[-*\'\"`(\[]*\s*(yes|no)\b", value, re.IGNORECASE)
    if match is None:
        return None, "ambiguous"
    return match.group(1).lower(), "explicit"


def rule_normalized_prediction(text: object) -> str:
    """Apply RULE's first-sentence convention after punctuation normalization.

    RULE's vendored POPE evaluator tokenizes only on spaces after deleting
    commas. Consequently, semantically identical outputs such as ``No.`` and
    ``[no].`` receive different labels. This parser preserves the published
    convention (a first-sentence ``no`` or ``not`` token means negative) while
    using word boundaries so surrounding punctuation cannot change the score.
    """
    first_sentence = ("" if text is None else str(text)).split(".", 1)[0]
    has_negative_word = re.search(
        r"(?<![A-Za-z])(no|not)(?![A-Za-z])",
        first_sentence,
        flags=re.IGNORECASE,
    )
    return "no" if has_negative_word else "yes"


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _binary_metrics(confusion: Counter[str]) -> dict[str, Any]:
    tp = confusion["yes->yes"]
    tn = confusion["no->no"]
    fp = confusion["no->yes"]
    fn = confusion["yes->no"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
    }


def evaluate_rule_rows(
    questions: Iterable[dict[str, Any]],
    answers: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate exact one-to-one RULE answers and return metrics plus records."""
    question_rows = list(questions)
    answer_rows = list(answers)
    question_ids = [_qid(row) for row in question_rows]
    answer_ids = [_qid(row) for row in answer_rows]
    if len(question_ids) != len(set(question_ids)):
        raise RuleEvaluationError("duplicate question ids in questions")
    if len(answer_ids) != len(set(answer_ids)):
        raise RuleEvaluationError("duplicate question ids in answers")
    if answer_ids != question_ids:
        missing = sorted(set(question_ids) - set(answer_ids))
        extra = sorted(set(answer_ids) - set(question_ids))
        raise RuleEvaluationError(
            "answer qids/order do not exactly match questions: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    pope_correct = 0
    normalized_correct = 0
    explicit_gt_count = 0
    explicit_gt_correct = 0
    ambiguous_gt_count = 0
    strict_correct = 0
    strict_parseable = 0
    leading_correct = 0
    leading_parseable = 0
    decision_first_correct = 0
    decision_first_parseable = 0
    pope_confusion: Counter[str] = Counter()
    normalized_confusion: Counter[str] = Counter()
    explicit_gt_confusion: Counter[str] = Counter()
    strict_confusion: Counter[str] = Counter()
    leading_confusion: Counter[str] = Counter()
    decision_first_confusion: Counter[str] = Counter()
    parse_status: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for question, answer in zip(question_rows, answer_rows):
        raw_gt = question.get("answer", question.get("gt_answer"))
        raw_text = _prediction_text(answer)
        gt_explicit, gt_status = parse_rule_ground_truth(raw_gt)
        gt_pope = rule_pope_prediction(raw_gt)
        pred_pope = rule_pope_prediction(raw_text)
        pred_normalized = rule_normalized_prediction(raw_text)
        pred_leading, pred_leading_status = parse_rule_ground_truth(raw_text)
        strict = parse_answer(raw_text, answer_type="binary")
        strict_label = strict.labels[0] if strict.labels else None
        decision_first_label = (
            pred_leading if pred_leading is not None else strict_label
        )
        decision_first_source = (
            "leading_explicit" if pred_leading is not None else "strict_fallback"
        )

        pope_is_correct = pred_pope == gt_pope
        pope_correct += int(pope_is_correct)
        pope_confusion[f"{gt_pope}->{pred_pope}"] += 1

        explicit_is_correct = None
        strict_is_correct = None
        decision_first_is_correct = None
        if gt_explicit is None:
            ambiguous_gt_count += 1
        else:
            explicit_gt_count += 1
            explicit_is_correct = pred_pope == gt_explicit
            strict_is_correct = (
                strict_label is not None and strict_label == gt_explicit
            )
            explicit_gt_correct += int(explicit_is_correct)
            normalized_is_correct = pred_normalized == gt_explicit
            normalized_correct += int(normalized_is_correct)
            strict_correct += int(strict_is_correct)
            strict_parseable += int(strict_label is not None)
            leading_is_correct = (
                pred_leading is not None and pred_leading == gt_explicit
            )
            leading_correct += int(leading_is_correct)
            leading_parseable += int(pred_leading is not None)
            decision_first_is_correct = (
                decision_first_label is not None
                and decision_first_label == gt_explicit
            )
            decision_first_correct += int(decision_first_is_correct)
            decision_first_parseable += int(decision_first_label is not None)
            parse_status[strict.status] += 1
            explicit_gt_confusion[f"{gt_explicit}->{pred_pope}"] += 1
            normalized_confusion[f"{gt_explicit}->{pred_normalized}"] += 1
            strict_value = strict_label if strict_label is not None else "invalid"
            strict_confusion[f"{gt_explicit}->{strict_value}"] += 1
            leading_value = pred_leading if pred_leading is not None else "invalid"
            leading_confusion[f"{gt_explicit}->{leading_value}"] += 1
            decision_value = (
                decision_first_label
                if decision_first_label is not None
                else "invalid"
            )
            decision_first_confusion[f"{gt_explicit}->{decision_value}"] += 1

        records.append(
            {
                "question_id": _qid(question),
                "image": question.get("image"),
                "question": question.get("question"),
                "ground_truth_raw": raw_gt,
                "ground_truth_explicit": gt_explicit,
                "ground_truth_status": gt_status,
                "ground_truth_pope_diagnostic": gt_pope,
                "raw_text": raw_text,
                "pope_prediction": pred_pope,
                "rule_normalized_prediction": pred_normalized,
                "leading_explicit_prediction": pred_leading,
                "leading_explicit_status": pred_leading_status,
                "decision_first_prediction": decision_first_label,
                "decision_first_source": decision_first_source,
                "decision_first_correct": decision_first_is_correct,
                "rule_normalized_correct": (
                    None
                    if gt_explicit is None
                    else pred_normalized == gt_explicit
                ),
                "explicit_ground_truth_correct": explicit_is_correct,
                "pope_diagnostic_correct": pope_is_correct,
                "strict_prediction": strict_label,
                "strict_status": strict.status,
                "strict_correct": strict_is_correct,
            }
        )

    n = len(records)
    metrics = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_metric": "leading_explicit.accuracy_invalid_as_error",
        "n": n,
        "qid_order_exact": True,
        "rule_normalized": {
            "n": explicit_gt_count,
            "ambiguous_excluded": ambiguous_gt_count,
            "coverage": _safe_div(explicit_gt_count, n),
            "correct": normalized_correct,
            "accuracy": _safe_div(normalized_correct, explicit_gt_count),
            **_binary_metrics(normalized_confusion),
            "confusion": dict(sorted(normalized_confusion.items())),
            "prediction_parser": (
                "RULE first-sentence no/not convention with punctuation "
                "normalization"
            ),
            "diagnostic_only": True,
            "note": (
                "Historical generated-sentence diagnostic. It preserves RULE's "
                "first-sentence negative-word semantics but prevents brackets "
                "or punctuation from changing the predicted label."
            ),
        },
        "explicit_ground_truth": {
            "n": explicit_gt_count,
            "ambiguous_excluded": ambiguous_gt_count,
            "coverage": _safe_div(explicit_gt_count, n),
            "correct": explicit_gt_correct,
            "accuracy": _safe_div(explicit_gt_correct, explicit_gt_count),
            **_binary_metrics(explicit_gt_confusion),
            "confusion": dict(sorted(explicit_gt_confusion.items())),
            "prediction_parser": "RULE/LLaVA POPE response convention",
            "note": (
                "Legacy POPE-parser diagnostic. Only ground truths beginning with an "
                "explicit yes/no label are included; ambiguous ground truths "
                "are excluded and counted."
            ),
        },
        "pope_compatible": {
            "correct": pope_correct,
            "accuracy": _safe_div(pope_correct, n),
            **_binary_metrics(pope_confusion),
            "confusion": dict(sorted(pope_confusion.items())),
            "diagnostic_only": True,
            "rule_paper_main_table_reproduction": False,
            "note": (
                "Diagnostic reconstruction of RULE/LLaVA eval_pope applied to "
                "both predictions and free-form RULE ground truths. It is not a "
                "validated reproduction of the RULE paper main-table metric."
            ),
        },
        "strict_explicit": {
            "ground_truth_n": explicit_gt_count,
            "ambiguous_ground_truth_excluded": ambiguous_gt_count,
            "correct": strict_correct,
            "parseable": strict_parseable,
            "parse_rate": _safe_div(strict_parseable, explicit_gt_count),
            "accuracy_invalid_as_error": _safe_div(
                strict_correct, explicit_gt_count
            ),
            "accuracy_parseable_only": _safe_div(
                strict_correct, strict_parseable
            ),
            "parse_status": dict(sorted(parse_status.items())),
            "confusion": dict(sorted(strict_confusion.items())),
            "note": (
                "Prediction parser diagnostic restricted to explicit ground "
                "truth rows. Empty, ambiguous, unfinished, and unparseable "
                "predictions are errors."
            ),
        },
        "leading_explicit": {
            "ground_truth_n": explicit_gt_count,
            "ambiguous_ground_truth_excluded": ambiguous_gt_count,
            "correct": leading_correct,
            "parseable": leading_parseable,
            "parse_rate": _safe_div(leading_parseable, explicit_gt_count),
            "accuracy_invalid_as_error": _safe_div(
                leading_correct, explicit_gt_count
            ),
            "accuracy_parseable_only": _safe_div(
                leading_correct, leading_parseable
            ),
            "confusion": dict(sorted(leading_confusion.items())),
            "prediction_parser": "explicit leading yes/no token",
            "note": (
                "Common-protocol CE-G primary metric. Invalid outputs are "
                "errors, and later negations never override the leading answer."
            ),
        },
        "decision_first": {
            "ground_truth_n": explicit_gt_count,
            "ambiguous_ground_truth_excluded": ambiguous_gt_count,
            "correct": decision_first_correct,
            "parseable": decision_first_parseable,
            "parse_rate": _safe_div(
                decision_first_parseable, explicit_gt_count
            ),
            "accuracy_invalid_as_error": _safe_div(
                decision_first_correct, explicit_gt_count
            ),
            "accuracy_parseable_only": _safe_div(
                decision_first_correct, decision_first_parseable
            ),
            "confusion": dict(sorted(decision_first_confusion.items())),
            "prediction_parser": (
                "explicit leading yes/no token, otherwise strict semantic parser"
            ),
            "note": (
                "Preferred Huatuo full-sentence metric. A leading decision has "
                "priority over negations in explanatory clauses; naturally "
                "phrased answers without a leading token use the strict parser."
            ),
        },
    }
    return metrics, records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, records = evaluate_rule_rows(
        load_jsonl(args.questions), load_jsonl(args.answers)
    )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.metrics.with_suffix(args.metrics.suffix + ".tmp")
    temporary.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    temporary.replace(args.metrics)
    write_jsonl(args.records, records)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
