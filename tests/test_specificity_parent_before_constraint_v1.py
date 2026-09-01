import json

from anchor.corrected_sgta.audit_specificity_parent_before_constraint_v1 import (
    audit,
    classify_parent_realization,
)


def _candidate(
    *,
    edge_id: str,
    answer: str,
    parent: str,
    child: str,
    constraint: str,
    edge_type: str = "subtype",
) -> dict:
    return {
        "case_id": "CASE-" + edge_id,
        "edge_id": edge_id,
        "edge_type": edge_type,
        "answer_span": answer,
        "parent_proposal": parent,
        "child_proposal": child,
        "added_constraint_proposal": constraint,
        "image_relpath": edge_id + ".jpg",
        "modality_stratum": "XR",
        "anatomy_stratum": "thorax",
    }


def test_completed_observed_parent_is_strictly_before_constraint():
    parent = "A pulmonary opacity is present."
    child = "This finding suggests pneumonia."
    row = classify_parent_realization(
        _candidate(
            edge_id="strict",
            answer=parent + " " + child,
            parent=parent,
            child=child,
            constraint="suggests pneumonia.",
        )
    )
    assert row["strict_parent_before_constraint"] is True
    assert row["realization_state"] == "strict_sentence_closed_parent_before_constraint"


def test_deleted_modifier_parent_is_counterfactual_not_observed_prefix():
    child = "A large pulmonary opacity is present."
    row = classify_parent_realization(
        _candidate(
            edge_id="deleted",
            answer=child,
            parent="A pulmonary opacity is present.",
            child=child,
            constraint="large",
            edge_type="size_morph",
        )
    )
    assert row["exact_parent_before_constraint"] is False
    assert row["realization_state"] == "counterfactual_parent_only"


def test_incomplete_surface_prefix_is_reported_but_not_certified():
    child = "The opacity is suggestive of pneumonia."
    row = classify_parent_realization(
        _candidate(
            edge_id="incomplete",
            answer=child,
            parent="The opacity is",
            child=child,
            constraint="suggestive of pneumonia.",
        )
    )
    assert row["exact_parent_before_constraint"] is True
    assert row["strict_parent_before_constraint"] is False
    assert row["realization_state"] == "exact_surface_parent_but_not_sentence_closed"


def test_current_pack_fails_closed_without_reading_physician_outcomes():
    result = audit(
        __import__("pathlib").Path(
            "corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2/"
            "candidates.blinded.jsonl"
        )
    )
    assert result["status"] == "no_go_current_pack"
    assert result["outcome_blind_contract"]["physician_reviews_read"] is False
    assert result["state_counts"] == {
        "counterfactual_parent_only": 76,
        "exact_surface_parent_but_not_sentence_closed": 27,
        "strict_sentence_closed_parent_before_constraint": 24,
    }
    assert result["strict_parent_summaries"]["dev"]["cases"] == 8
    assert result["strict_parent_summaries"]["test"]["cases"] == 14
    assert result["strict_parent_summaries"]["dev"]["repeated_exact_constraint_blocks"] == 0
    assert result["strict_parent_summaries"]["test"]["repeated_exact_constraint_blocks"] == 0
    assert result["scientific_naming_gate"]["crossing_authorized"] is False
