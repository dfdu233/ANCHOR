#!/usr/bin/env python3
"""Fail-closed paired evaluation for RULE mitigation outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from corrected_sgta.evaluate_medheval_answers import rule_pope_prediction
from corrected_sgta.evaluate_rule_vqa import parse_rule_ground_truth
from corrected_sgta.rule_mitigation_backend import prompt_manifest

PROTOCOL_VERSION = "rule-vqa-paired-explicit-gt-v1"
_DECODER_INTERVENTION_FIELDS = {
    "attention_alpha",
    "attention_intervention",
    "cfg_gamma",
    "cfg_intervention",
    "cfg_prompt",
    "end_layer_exclusive",
    "image_range",
    "runtime_control_for",
    "start_layer",
}
_ACTIVATION_INTERVENTION_FIELDS = {
    "method",
    "generation",
    "counters",
    "pai_audits",
    "key_positions",
    "m3id_audits",
    "avisc_audits",
}


class PairedEvaluationError(ValueError):
    """Raised when paired protocol identity cannot be established."""


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise PairedEvaluationError(f"{path}: expected JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise PairedEvaluationError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return rows


def qid(row: dict[str, Any]) -> str:
    value = row.get("question_id", row.get("qid", row.get("id")))
    if value is None:
        raise PairedEvaluationError("row has no question id")
    return str(value)


def answer_text(row: dict[str, Any]) -> str:
    for key in ("answer", "text", "prediction", "output"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def exact_mcnemar(rescues: int, harms: int) -> dict[str, Any]:
    discordant = rescues + harms
    if discordant == 0:
        p_value = 1.0
    else:
        numerator = sum(
            math.comb(discordant, index)
            for index in range(min(rescues, harms) + 1)
        )
        p_value = min(1.0, float(2 * Fraction(numerator, 2**discordant)))
    return {
        "control_wrong_method_right": rescues,
        "control_right_method_wrong": harms,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def _artifact_hash(meta: dict[str, Any], name: str, actual: Path) -> None:
    item = meta.get("artifacts", {}).get(name)
    if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
        raise PairedEvaluationError(f"meta has no hashed {name} artifact")
    if sha256_file(actual) != item["sha256"]:
        raise PairedEvaluationError(f"{name} hash differs from meta")


def _validate_meta(
    meta: dict[str, Any],
    *,
    questions: Path,
    answers: Path,
    activation: Path,
) -> dict[str, Any]:
    if meta.get("status") != "complete":
        raise PairedEvaluationError("meta status is not complete")
    if not isinstance(meta.get("protocol_version"), str):
        raise PairedEvaluationError("meta protocol version is missing")
    if not isinstance(meta.get("fingerprint"), str):
        raise PairedEvaluationError("meta fingerprint is missing")
    payload = meta.get("payload")
    if not isinstance(payload, dict):
        raise PairedEvaluationError("meta payload is missing")
    fingerprint_envelope = {"protocol_version": meta["protocol_version"], **payload}
    expected_fingerprint = sha256_bytes(
        stable_json(fingerprint_envelope).encode()
    )
    if meta["fingerprint"] != expected_fingerprint:
        raise PairedEvaluationError("meta fingerprint does not match payload")
    for key in (
        "dataset",
        "method",
        "decoding",
        "expected_qids",
        "questions_sha256",
        "ordered_prompt_sha256",
        "model_tree_sha256",
        "runtime_versions",
        "runner_sha256",
        "evaluator_sha256",
    ):
        if payload.get(key) is None:
            raise PairedEvaluationError(f"meta payload is missing {key}")
    if sha256_file(questions) != payload["questions_sha256"]:
        raise PairedEvaluationError("questions hash differs from meta")
    _artifact_hash(meta, "answers", answers)
    _artifact_hash(meta, "activation", activation)
    payload = dict(payload)
    payload["_runner_protocol_version"] = meta["protocol_version"]
    return payload


def _strip_keys(value: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def _validate_pair_identity(
    control_meta: dict[str, Any],
    method_meta: dict[str, Any],
    control_activation: dict[str, Any],
    method_activation: dict[str, Any],
) -> dict[str, Any]:
    control_payload = dict(control_meta)
    method_payload = dict(method_meta)
    control_decoder = control_payload.pop("decoding")
    method_decoder = method_payload.pop("decoding")
    control_provenance = control_payload.pop("method_provenance", {})
    method_provenance = method_payload.pop("method_provenance", {})
    if _strip_keys(control_provenance, {"upstream", "note"}) != _strip_keys(
        method_provenance, {"upstream", "note"}
    ):
        raise PairedEvaluationError("non-intervention method provenance differs")
    for payload in (control_payload, method_payload):
        payload.pop("method", None)
    if control_payload != method_payload:
        raise PairedEvaluationError("model/runtime/prompt/evaluator payload identity differs")
    if _strip_keys(control_decoder, _DECODER_INTERVENTION_FIELDS) != _strip_keys(
        method_decoder, _DECODER_INTERVENTION_FIELDS
    ):
        raise PairedEvaluationError("non-intervention decoder fields differ")
    control_extra = set(control_decoder) - set(method_decoder)
    method_extra = set(method_decoder) - set(control_decoder)
    if not (control_extra | method_extra) <= _DECODER_INTERVENTION_FIELDS:
        raise PairedEvaluationError("unexpected method-specific decoder fields")
    if _strip_keys(control_activation, _ACTIVATION_INTERVENTION_FIELDS) != _strip_keys(
        method_activation, _ACTIVATION_INTERVENTION_FIELDS
    ):
        raise PairedEvaluationError("activation protocol identity differs")
    for payload, activation in (
        (control_meta, control_activation),
        (method_meta, method_activation),
    ):
        expected_generation = dict(payload["decoding"])
        expected_generation.pop("conv_mode", None)
        expected_generation.pop("configuration_source", None)
        if activation.get("generation") != expected_generation:
            raise PairedEvaluationError("activation generation differs from meta decoder")
        if activation.get("method") != payload["method"]:
            raise PairedEvaluationError("activation method differs from meta")
    return control_payload


def evaluate_paired(
    questions: list[dict[str, Any]],
    control_answers: list[dict[str, Any]],
    method_answers: list[dict[str, Any]],
    *,
    control_meta: dict[str, Any],
    method_meta: dict[str, Any],
    control_activation: dict[str, Any],
    method_activation: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity = _validate_pair_identity(
        control_meta, method_meta, control_activation, method_activation
    )
    expected = [qid(row) for row in questions]
    if len(expected) != len(set(expected)):
        raise PairedEvaluationError("duplicate question ids")
    if [qid(row) for row in control_answers] != expected:
        raise PairedEvaluationError("control answer qids/order differ")
    if [qid(row) for row in method_answers] != expected:
        raise PairedEvaluationError("method answer qids/order differ")
    if control_meta["expected_qids"] != expected:
        raise PairedEvaluationError("meta expected qids/order differ")
    dataset = control_meta["dataset"]
    prompt_audit = prompt_manifest(dataset, questions)
    if prompt_audit["ordered_prompt_sha256"] != control_meta["ordered_prompt_sha256"]:
        raise PairedEvaluationError("ordered prompt hash differs from questions")
    prompt_hashes = {item["question_id"]: item["sha256"] for item in prompt_audit["prompts"]}

    explicit = 0
    ambiguous = 0
    control_correct = 0
    method_correct = 0
    rescues = 0
    harms = 0
    changed = 0
    outcomes: Counter[str] = Counter()
    conflicts: list[dict[str, Any]] = []
    for question, control, method in zip(questions, control_answers, method_answers):
        current_qid = qid(question)
        for name, answer in (("control", control), ("method", method)):
            observed = answer.get("metadata", {}).get("prompt_sha256")
            if observed != prompt_hashes[current_qid]:
                raise PairedEvaluationError(f"{name} answer prompt hash differs")
        gt, gt_status = parse_rule_ground_truth(
            question.get("answer", question.get("gt_answer"))
        )
        control_text = answer_text(control)
        method_text = answer_text(method)
        control_pred = rule_pope_prediction(control_text)
        method_pred = rule_pope_prediction(method_text)
        if gt is None:
            ambiguous += 1
            continue
        explicit += 1
        control_ok = control_pred == gt
        method_ok = method_pred == gt
        control_correct += int(control_ok)
        method_correct += int(method_ok)
        outcome = f"{int(control_ok)}->{int(method_ok)}"
        outcomes[outcome] += 1
        if control_pred != method_pred:
            changed += 1
            if not control_ok and method_ok:
                classification = "rescue"
                rescues += 1
            elif control_ok and not method_ok:
                classification = "harm"
                harms += 1
            else:
                classification = "changed_same_correctness"
            conflicts.append(
                {
                    "question_id": current_qid,
                    "image": question.get("image"),
                    "question": question.get("question"),
                    "ground_truth_raw": question.get("answer", question.get("gt_answer")),
                    "ground_truth_explicit": gt,
                    "ground_truth_status": gt_status,
                    "control_text": control_text,
                    "control_prediction": control_pred,
                    "control_correct": control_ok,
                    "method_text": method_text,
                    "method_prediction": method_pred,
                    "method_correct": method_ok,
                    "classification": classification,
                }
            )
    control_accuracy = control_correct / explicit if explicit else 0.0
    method_accuracy = method_correct / explicit if explicit else 0.0
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_metric": "explicit_ground_truth.paired_accuracy_delta",
        "n_questions": len(questions),
        "explicit_ground_truth_n": explicit,
        "ambiguous_ground_truth_excluded": ambiguous,
        "control": {"correct": control_correct, "accuracy": control_accuracy},
        "method": {"correct": method_correct, "accuracy": method_accuracy},
        "delta": {
            "accuracy": method_accuracy - control_accuracy,
            "percentage_points": 100.0 * (method_accuracy - control_accuracy),
        },
        "changed_predictions": changed,
        "rescues": rescues,
        "harms": harms,
        "net_rescues": rescues - harms,
        "paired_outcomes": dict(sorted(outcomes.items())),
        "mcnemar_exact": exact_mcnemar(rescues, harms),
        "conflict_cases": len(conflicts),
        "identity": identity,
    }
    return report, conflicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--control-answers", type=Path, required=True)
    parser.add_argument("--method-answers", type=Path, required=True)
    parser.add_argument("--control-meta", type=Path, required=True)
    parser.add_argument("--method-meta", type=Path, required=True)
    parser.add_argument("--control-activation", type=Path, required=True)
    parser.add_argument("--method-activation", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--conflicts", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control_meta_document = load_json(args.control_meta)
    method_meta_document = load_json(args.method_meta)
    control_payload = _validate_meta(
        control_meta_document,
        questions=args.questions,
        answers=args.control_answers,
        activation=args.control_activation,
    )
    method_payload = _validate_meta(
        method_meta_document,
        questions=args.questions,
        answers=args.method_answers,
        activation=args.method_activation,
    )
    report, conflicts = evaluate_paired(
        load_jsonl(args.questions),
        load_jsonl(args.control_answers),
        load_jsonl(args.method_answers),
        control_meta=control_payload,
        method_meta=method_payload,
        control_activation=load_json(args.control_activation),
        method_activation=load_json(args.method_activation),
    )
    report["inputs"] = {
        "questions_sha256": sha256_file(args.questions),
        "control_answers_sha256": sha256_file(args.control_answers),
        "method_answers_sha256": sha256_file(args.method_answers),
        "control_meta_sha256": sha256_file(args.control_meta),
        "method_meta_sha256": sha256_file(args.method_meta),
        "control_activation_sha256": sha256_file(args.control_activation),
        "method_activation_sha256": sha256_file(args.method_activation),
        "evaluator_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(args.summary, report)
    atomic_jsonl(args.conflicts, conflicts)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
