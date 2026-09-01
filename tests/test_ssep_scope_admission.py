import json

from anchor.corrected_sgta.ssep_scope_admission import (
    ADMISSION_RULES,
    build_minimal_pair,
    extract_candidates,
    finding_mentions,
    summarize,
)


def test_negated_coordination_is_candidate_not_truth():
    rows = extract_candidates("No focal consolidation or pleural effusion is seen.")
    assert len(rows) == 1
    assert rows[0]["operator"] == "negated_coordination"
    assert rows[0]["parser_status"] == "candidate_only_not_human_truth"
    assert [s["finding"] for s in rows[0]["siblings"]] == ["consolidation", "pleural_effusion"]


def test_already_distributive_negation_is_not_shared_scope():
    assert extract_candidates("No focal consolidation; no pleural effusion.") == []


def test_two_sibling_minimal_pair_preserves_frozen_mechanical_invariants():
    mentions = finding_mentions("No pneumothorax or pleural effusion.")
    pair = build_minimal_pair(mentions, "negated_coordination")
    assert pair is not None
    assert pair["ordered_claims"] == ["pneumothorax", "pleural_effusion"]
    assert pair["claim_count_equal"]
    assert pair["whitespace_word_count_equal"]
    assert pair["human_naturalness"] is None
    assert pair["human_semantic_equivalence"] is None


def test_reference_does_not_count_as_second_model_and_human_gate_fails_closed():
    candidate = {
        "source_id": "hulu_mimic_report",
        "source_kind": "model",
        "model_id": "hulu",
        "task": ADMISSION_RULES["formal_task"],
        "operator": "negated_coordination",
        "sentence": "No pneumothorax or pleural effusion.",
        "siblings": [
            {"finding": "pneumothorax"},
            {"finding": "pleural_effusion"},
        ],
        "minimal_pair_candidate": {
            "parser_constructible": True,
            "shared": "No pneumothorax or pleural effusion.",
            "distributive": "No pneumothorax; no pleural effusion.",
        },
    }
    reference = {
        **candidate,
        "source_id": "mimic_reference_report",
        "source_kind": "reference",
        "model_id": None,
        "task": "mimic_study_report_reference",
    }
    summary = summarize([candidate] * 100 + [reference] * 100, [])
    assert summary["formal_model_census"]["by_model"] == {"hulu": 100}
    assert not summary["mechanical_gates"]["two_models_with_minimum_cases"]
    assert summary["decision"] == "NO_GO"
    assert not summary["minimal_pair_model_run_authorized"]


def test_mixed_negation_does_not_become_shared_scope():
    assert extract_candidates("There is atelectasis but no pleural effusion.")[0]["operator"] == "contrastive_scope"
