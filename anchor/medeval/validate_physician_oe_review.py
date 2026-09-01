"""Fail-closed validation for completed blinded physician OE reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "anchor-physician-oe-review-validation-v1"
OBSERVABILITY = {"observable", "partially_observable", "unobservable", "indeterminate"}
CORRECTNESS = {"correct", "partially_correct", "incorrect", "indeterminate"}
ANSWER_STATE = {"supported", "refuted", "undetermined", "unobservable"}
CLAIM_TYPE = {"visual", "knowledge", "unobservable"}
VISUAL_SUPPORT = {"supported", "refuted", "undetermined", "not_applicable"}
COMMITMENT = {"definite", "uncertain", "unknown"}
RELEVANCE = {"required", "optional", "out_of_scope"}
ERROR_TYPE = {
    "none", "fabricated", "false_negation", "location", "attribute",
    "inappropriate_certainty", "indeterminate",
}
HARM = {"no", "possibly", "yes", "indeterminate"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalized_claim(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{context}: normalized_claim must be an object")
    _require(isinstance(value.get("finding"), str) and value["finding"].strip(), f"{context}: finding is empty")
    _require(value.get("polarity") in {"present", "absent"}, f"{context}: invalid polarity")
    _require(value.get("uncertainty") in COMMITMENT, f"{context}: invalid uncertainty")
    _require(value.get("anatomy") is None or isinstance(value.get("anatomy"), str), f"{context}: invalid anatomy")
    attributes = value.get("attributes")
    _require(isinstance(attributes, list) and all(isinstance(x, str) for x in attributes), f"{context}: invalid attributes")
    return value


def _immutable_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": row.get("bundle_id"),
        "group_id": row.get("group_id"),
        "review_order": row.get("review_order"),
        "review_phase": row.get("review_phase"),
        "reviewer_slot": row.get("reviewer_slot"),
        "image": row.get("image"),
        "question": row.get("question"),
        "benchmark_reference": row.get("benchmark_reference"),
        "candidates": [
            {"answer_id": item.get("answer_id"), "answer_text": item.get("answer_text")}
            for item in row.get("candidate_answers", [])
        ],
    }


def validate_completed(
    template: list[dict[str, Any]], completed: list[dict[str, Any]]
) -> dict[str, Any]:
    _require(len(completed) == len(template) > 0, "review group count changed")
    _require(
        [_immutable_view(row) for row in completed] == [_immutable_view(row) for row in template],
        "reviewer-visible immutable content changed",
    )
    answer_count = 0
    atomic_count = 0
    for row in completed:
        group = str(row["group_id"])
        reference = row.get("reference_annotation")
        _require(isinstance(reference, dict), f"{group}: missing reference annotation")
        observability = reference.get("visual_observability")
        _require(observability in OBSERVABILITY, f"{group}: invalid visual observability")
        _require(reference.get("benchmark_reference_correctness") in CORRECTNESS, f"{group}: invalid benchmark correctness")
        required = reference.get("required_answer_claims")
        _require(isinstance(required, list), f"{group}: required claims must be a list")
        required_ids = []
        for index, claim in enumerate(required):
            context = f"{group}/required/{index}"
            _require(isinstance(claim, dict), f"{context}: claim must be an object")
            claim_id = claim.get("claim_id")
            _require(isinstance(claim_id, str) and claim_id.strip(), f"{context}: claim_id is empty")
            required_ids.append(claim_id)
            _normalized_claim(claim.get("normalized_claim"), context)
        _require(len(required_ids) == len(set(required_ids)), f"{group}: duplicate required claim IDs")
        if observability in {"observable", "partially_observable"}:
            _require(required_ids, f"{group}: observable question has no required answer claim")
        _require(isinstance(reference.get("notes", ""), str), f"{group}: notes must be text")

        for candidate in row.get("candidate_answers", []):
            answer_count += 1
            answer_id = str(candidate.get("answer_id"))
            text = str(candidate.get("answer_text", ""))
            annotation = candidate.get("annotation")
            context = f"{group}/{answer_id}"
            _require(isinstance(annotation, dict), f"{context}: missing annotation")
            _require(annotation.get("direct_answer_correctness") in CORRECTNESS, f"{context}: invalid direct correctness")
            _require(annotation.get("direct_answer_state") in ANSWER_STATE, f"{context}: invalid direct state")
            claims = annotation.get("atomic_claims")
            no_claims = annotation.get("no_clinical_claims")
            _require(isinstance(claims, list), f"{context}: atomic_claims must be a list")
            _require(isinstance(no_claims, bool), f"{context}: no_clinical_claims must be boolean")
            _require(no_claims == (len(claims) == 0), f"{context}: claim/no-claim XOR failed")
            claim_ids = []
            for index, claim in enumerate(claims):
                claim_context = f"{context}/claim/{index}"
                _require(isinstance(claim, dict), f"{claim_context}: claim must be an object")
                claim_id = claim.get("claim_id")
                _require(isinstance(claim_id, str) and claim_id.strip(), f"{claim_context}: empty claim_id")
                claim_ids.append(claim_id)
                span = claim.get("text_span")
                _require(isinstance(span, str) and span and span in text, f"{claim_context}: text_span is not verbatim")
                normalized = _normalized_claim(claim.get("normalized_claim"), claim_context)
                claim_type = claim.get("claim_type")
                support = claim.get("visual_support")
                commitment = claim.get("commitment")
                error = claim.get("error_type")
                _require(claim_type in CLAIM_TYPE, f"{claim_context}: invalid claim type")
                _require(support in VISUAL_SUPPORT, f"{claim_context}: invalid visual support")
                _require(commitment in COMMITMENT, f"{claim_context}: invalid commitment")
                _require(commitment == normalized["uncertainty"], f"{claim_context}: commitment/uncertainty mismatch")
                _require(claim.get("relevance") in RELEVANCE, f"{claim_context}: invalid relevance")
                _require(error in ERROR_TYPE, f"{claim_context}: invalid error type")
                if claim_type == "visual":
                    _require(support != "not_applicable", f"{claim_context}: visual claim lacks support state")
                    if support == "undetermined" and commitment == "definite":
                        _require(error == "inappropriate_certainty", f"{claim_context}: definite undetermined claim is not marked")
                    if support == "refuted":
                        _require(error != "none", f"{claim_context}: refuted visual claim marked error-free")
                else:
                    _require(support == "not_applicable", f"{claim_context}: nonvisual claim assigned visual support")
                    _require(error in {"none", "indeterminate"}, f"{claim_context}: nonvisual claim mislabeled as visual error")
                atomic_count += 1
            _require(len(claim_ids) == len(set(claim_ids)), f"{context}: duplicate atomic claim IDs")
            omissions = annotation.get("omitted_required_claim_ids")
            _require(isinstance(omissions, list) and set(omissions) <= set(required_ids), f"{context}: invalid omission IDs")
            harm = annotation.get("overall_clinically_harmful")
            confidence = annotation.get("reviewer_confidence")
            rationale = annotation.get("rationale")
            _require(harm in HARM, f"{context}: invalid harmfulness")
            _require(isinstance(confidence, int) and 1 <= confidence <= 5, f"{context}: invalid confidence")
            _require(isinstance(rationale, str), f"{context}: rationale must be text")
            if harm != "no" or annotation["direct_answer_correctness"] == "indeterminate":
                _require(rationale.strip(), f"{context}: harmful/indeterminate annotation needs rationale")
    return {
        "protocol_version": VERSION,
        "passed": True,
        "groups": len(completed),
        "answer_units": answer_count,
        "atomic_claims": atomic_count,
        "reviewer_slot": completed[0].get("reviewer_slot"),
        "interpretation": "schema and immutable-content validation only; no agreement or efficacy claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_completed(load_jsonl(args.template), load_jsonl(args.completed))
    result.update({
        "template": str(args.template.resolve()),
        "template_sha256": sha256_file(args.template),
        "completed": str(args.completed.resolve()),
        "completed_sha256": sha256_file(args.completed),
    })
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
