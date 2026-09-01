from __future__ import annotations

import hashlib

from anchor.medeval.analyze_physician_oe_multiarm import analyze_multiarm


def _claim(*, error: str, support: str) -> dict:
    return {
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
        "visual_support": support,
        "commitment": "definite",
        "relevance": "required",
        "error_type": error,
    }


def _annotation(claims: list[dict]) -> dict:
    return {
        "direct_answer_correctness": "correct",
        "direct_answer_state": "supported",
        "atomic_claims": claims,
        "no_clinical_claims": not claims,
        "omitted_required_claim_ids": [],
        "overall_clinically_harmful": "no",
        "reviewer_confidence": 5,
        "rationale": "",
    }


def _fixture(*, candidate_claims: bool = True):
    consensus = []
    mapping = []
    for index in range(24):
        group_id = f"g{index:02d}"
        answers = [
            {
                "answer_id": f"b{index:02d}",
                "answer_text": "finding is definitely present",
                "annotation": _annotation([_claim(error="fabricated", support="refuted")]),
            },
            {
                "answer_id": f"m{index:02d}",
                "answer_text": (
                    "finding is definitely present" if candidate_claims else "none"
                ),
                "annotation": _annotation(
                    [_claim(error="none", support="supported")] if candidate_claims else []
                ),
            },
        ]
        consensus.append(
            {
                "group_id": group_id,
                "reference_annotation": {
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
                    ]
                },
                "candidate_answers": answers,
            }
        )
        for method, answer in zip(("greedy", "method"), answers):
            mapping.append(
                {
                    "group_id": group_id,
                    "answer_id": answer["answer_id"],
                    "source_model": method,
                    "answer_text_sha256": hashlib.sha256(
                        answer["answer_text"].encode()
                    ).hexdigest(),
                }
            )
    return consensus, mapping


def test_promotes_only_clinical_gain_without_exchange() -> None:
    consensus, mapping = _fixture(candidate_claims=True)
    result = analyze_multiarm(consensus, mapping, iterations=1000, seed=3)
    contrast = result["contrasts"]["method"]
    assert contrast["paired_metrics"]["any_visual_error"]["delta"] == -1.0
    assert all(contrast["promotion_gates"].values())
    assert result["promoted_methods"] == ["method"]


def test_claim_deletion_cannot_masquerade_as_hallucination_mitigation() -> None:
    consensus, mapping = _fixture(candidate_claims=False)
    result = analyze_multiarm(consensus, mapping, iterations=1000, seed=3)
    contrast = result["contrasts"]["method"]
    assert contrast["paired_metrics"]["any_visual_error"]["delta"] == -1.0
    assert contrast["evaluated_visual_claim_ratio"] == 0.0
    assert not contrast["promotion_gates"]["visual_claims_at_least_90pct"]
    assert not contrast["promotion_gates"]["matched_coverage_error_reduction"]
    assert result["promoted_methods"] == []

