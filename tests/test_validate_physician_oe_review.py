import copy

import pytest

from anchor.medeval.validate_physician_oe_review import validate_completed


def _row():
    return {
        "bundle_id": "b",
        "group_id": "g",
        "review_order": 0,
        "review_phase": "independent_double_review",
        "reviewer_slot": "A",
        "image": {"relative_path": "x.png", "sha256": "a" * 64},
        "question": "What is present?",
        "benchmark_reference": "effusion",
        "reference_annotation": {
            "visual_observability": None,
            "benchmark_reference_correctness": None,
            "required_answer_claims": [],
            "notes": "",
        },
        "candidate_answers": [{
            "answer_id": "a1",
            "answer_text": "Small left pleural effusion.",
            "annotation": {
                "direct_answer_correctness": None,
                "direct_answer_state": None,
                "atomic_claims": [],
                "no_clinical_claims": None,
                "omitted_required_claim_ids": [],
                "overall_clinically_harmful": None,
                "reviewer_confidence": None,
                "rationale": "",
            },
        }],
    }


def _completed():
    row = _row()
    row["reference_annotation"].update({
        "visual_observability": "observable",
        "benchmark_reference_correctness": "correct",
        "required_answer_claims": [{
            "claim_id": "r1",
            "normalized_claim": {
                "finding": "pleural effusion", "polarity": "present",
                "uncertainty": "definite", "anatomy": None, "attributes": [],
            },
        }],
    })
    row["candidate_answers"][0]["annotation"].update({
        "direct_answer_correctness": "correct",
        "direct_answer_state": "supported",
        "no_clinical_claims": False,
        "atomic_claims": [{
            "claim_id": "c1", "text_span": "Small left pleural effusion",
            "normalized_claim": {
                "finding": "pleural effusion", "polarity": "present",
                "uncertainty": "definite", "anatomy": "left",
                "attributes": ["small"],
            },
            "claim_type": "visual", "visual_support": "supported",
            "commitment": "definite", "relevance": "required", "error_type": "none",
        }],
        "overall_clinically_harmful": "no",
        "reviewer_confidence": 5,
    })
    return row


def test_completed_physician_review_passes_frozen_contract():
    result = validate_completed([_row()], [_completed()])
    assert result["passed"] and result["atomic_claims"] == 1


def test_review_rejects_immutable_text_change_and_blank_claim_xor():
    changed = _completed()
    changed["candidate_answers"][0]["answer_text"] = "changed"
    with pytest.raises(ValueError, match="immutable"):
        validate_completed([_row()], [changed])
    incomplete = _completed()
    incomplete["candidate_answers"][0]["annotation"]["no_clinical_claims"] = True
    with pytest.raises(ValueError, match="XOR"):
        validate_completed([_row()], [incomplete])


def test_nonvisual_claim_cannot_enter_visual_hallucination_labels():
    completed = _completed()
    claim = completed["candidate_answers"][0]["annotation"]["atomic_claims"][0]
    claim["claim_type"] = "knowledge"
    claim["visual_support"] = "refuted"
    claim["error_type"] = "fabricated"
    with pytest.raises(ValueError, match="nonvisual"):
        validate_completed([_row()], [completed])
