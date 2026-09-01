from __future__ import annotations

import copy

from anchor.medeval.finalize_physician_oe_consensus import (
    _independent_view,
    clean_consensus,
)
from anchor.medeval.prepare_physician_oe_adjudication import prepare_adjudication
from anchor.medeval.validate_physician_oe_review import validate_completed


def _master() -> list[dict]:
    return [
        {
            "bundle_id": "bundle",
            "group_id": "group",
            "review_order": 0,
            "image": {"relative_path": "x.jpg", "sha256": "a" * 64},
            "question": "What finding is present?",
            "benchmark_reference": "finding",
            "reference_annotation": {
                "visual_observability": None,
                "benchmark_reference_correctness": None,
                "required_answer_claims": [],
                "notes": "",
            },
            "candidate_answers": [
                {
                    "answer_id": "answer",
                    "answer_text": "finding",
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
                }
            ],
        }
    ]


def _completed(slot: str) -> list[dict]:
    rows = copy.deepcopy(_master())
    rows[0]["reviewer_slot"] = slot
    rows[0]["review_phase"] = "double_review"
    rows[0]["reference_annotation"] = {
        "visual_observability": "observable",
        "benchmark_reference_correctness": "correct",
        "required_answer_claims": [
            {
                "claim_id": "r1",
                "normalized_claim": {
                    "finding": "finding",
                    "polarity": "present",
                    "uncertainty": "definite",
                    "anatomy": None,
                    "attributes": [],
                },
            }
        ],
        "notes": "",
    }
    rows[0]["candidate_answers"][0]["annotation"] = {
        "direct_answer_correctness": "correct",
        "direct_answer_state": "supported",
        "atomic_claims": [
            {
                "claim_id": "c1",
                "text_span": "finding",
                "normalized_claim": {
                    "finding": "finding",
                    "polarity": "present",
                    "uncertainty": "definite",
                    "anatomy": None,
                    "attributes": [],
                },
                "claim_type": "visual",
                "visual_support": "supported",
                "commitment": "definite",
                "relevance": "required",
                "error_type": "none",
            }
        ],
        "no_clinical_claims": False,
        "omitted_required_claim_ids": [],
        "overall_clinically_harmful": "no",
        "reviewer_confidence": 5,
        "rationale": "",
    }
    return rows


def test_adjudication_preserves_independent_reviews_and_cleans_before_unblinding() -> None:
    master = _master()
    adjudication = prepare_adjudication(master, _completed("A"), _completed("B"))
    assert adjudication[0]["reference_annotation"]["visual_observability"] is None
    assert set(adjudication[0]["independent_reviews"]) == {"A", "B"}
    assert set(adjudication[0]["candidate_answers"][0]["independent_reviews"]) == {
        "A",
        "B",
    }

    final = copy.deepcopy(adjudication)
    final[0]["reference_annotation"] = copy.deepcopy(
        adjudication[0]["independent_reviews"]["A"]
    )
    final[0]["candidate_answers"][0]["annotation"] = copy.deepcopy(
        adjudication[0]["candidate_answers"][0]["independent_reviews"]["A"]
    )
    assert _independent_view(final) == _independent_view(adjudication)
    consensus = clean_consensus(final)
    assert "independent_reviews" not in consensus[0]
    assert "independent_reviews" not in consensus[0]["candidate_answers"][0]
    assert validate_completed(master, consensus)["passed"]


def test_independent_review_mutation_is_detectable() -> None:
    adjudication = prepare_adjudication(_master(), _completed("A"), _completed("B"))
    mutated = copy.deepcopy(adjudication)
    mutated[0]["independent_reviews"]["A"]["notes"] = "changed"
    assert _independent_view(mutated) != _independent_view(adjudication)

