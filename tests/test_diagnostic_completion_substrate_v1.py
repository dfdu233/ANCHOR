from anchor.corrected_sgta.audit_diagnostic_completion_substrate_v1 import (
    audit,
    extract_events,
)
from anchor.corrected_sgta.audit_pragmatic_commitment_pairs_v1 import audit_pairs
from anchor.corrected_sgta.prepare_pragmatic_commitment_confirmatory_v1 import (
    balanced_split,
)
from anchor.corrected_sgta.run_huatuo_pragmatic_commitment_generation_v1 import (
    record_key,
    validate_substrate,
)


def test_extracts_patient_specific_diagnostic_completion():
    text = "Bilateral patchy opacities, suggesting possible pneumonia or pulmonary edema."
    events = extract_events(text)
    assert [(row["edge_id"], row["commitment"]) for row in events] == [
        ("opacity_to_pneumonia", "uncertain"),
        ("opacity_to_edema", "uncertain"),
    ]


def test_excludes_generic_educational_differential():
    text = (
        "A pleural effusion can be clinically relevant as it may indicate "
        "underlying medical conditions such as pneumonia or malignancy."
    )
    assert extract_events(text) == []


def test_excludes_negated_child():
    text = "A focal opacity is not consistent with pneumonia."
    assert extract_events(text) == []


def test_excludes_no_signs_of_child():
    text = "There are lung opacities but no signs of consolidation that would suggest pneumonia."
    assert extract_events(text) == []


def test_rule_out_language_is_not_a_positive_connector():
    text = "A pulmonary nodule requires further evaluation to rule out lung tumor."
    assert extract_events(text) == []


def test_audit_refuses_sparse_current_events_for_replay():
    generations = [
        {
            "image_id": "a",
            "prompt_condition": "neutral",
            "text": "An opacity suggests possible pneumonia.",
        },
        {
            "image_id": "b",
            "prompt_condition": "neutral",
            "text": "An opacity suggests pneumonia.",
        },
    ]
    labels = {
        "a": {
            "Lung Opacity": 3,
            "Infiltration": 0,
            "Consolidation": 0,
            "Nodule/Mass": 0,
            "Pneumonia": 0,
            "Edema": 0,
            "Atelectasis": 0,
            "Lung tumor": 0,
        },
        "b": {
            "Lung Opacity": 3,
            "Infiltration": 0,
            "Consolidation": 0,
            "Nodule/Mass": 0,
            "Pneumonia": 3,
            "Edema": 0,
            "Atelectasis": 0,
            "Lung tumor": 0,
        },
    }
    result = audit(
        generations,
        labels,
        condition="neutral",
        minimum_events_per_extreme=2,
        minimum_edge_types=1,
    )
    assert result["strict_transition_events"] == 2
    assert result["gates"]["confirmatory_hidden_state_replay_authorized"] is False
    assert result["target_key_role_overlap"]["pneumonia"] == {
        "parent_seen_child_0of3": 1,
        "parent_seen_child_3of3": 1,
    }


def test_pair_audit_requires_same_observation_and_refuses_discovery_n():
    generations = [
        {
            "image_id": "a",
            "prompt_condition": "neutral",
            "generated_token_count": 12,
            "text": "Bilateral patchy opacities, suggesting possible pneumonia.",
        },
        {
            "image_id": "a",
            "prompt_condition": "existential",
            "generated_token_count": 11,
            "text": "Bilateral patchy opacities, which are consistent with pneumonia.",
        },
    ]
    votes = {
        "a": {
            "Lung Opacity": 3,
            "Infiltration": 0,
            "Consolidation": 0,
            "Nodule/Mass": 0,
            "Pneumonia": 1,
            "Edema": 0,
            "Atelectasis": 0,
            "Lung tumor": 0,
        }
    }
    result = audit_pairs(
        generations,
        votes,
        reference_condition="neutral",
        focused_condition="existential",
        maximum_sentence_word_gap=4,
        minimum_pairs=2,
    )
    assert result["admitted_pairs"] == 1
    assert result["directions"] == {"up": 1}
    assert result["gates"]["confirmatory_mechanism_authorized"] is False


def test_confirmatory_split_balances_each_reader_stratum():
    rows = [
        {
            "image_id": f"i{index}",
            "lung_opacity_votes": 1 + index % 3,
            "pneumonia_votes": index % 4,
        }
        for index in range(41)
    ]
    assigned, diagnostics = balanced_split(rows, seed=7319)
    assert len(assigned) == len(rows)
    assert {row["experiment_split"] for row in assigned} == {"dev", "test"}
    for counts in diagnostics.values():
        assert abs(counts.get("dev", 0) - counts.get("test", 0)) <= 1


def test_generation_key_is_condition_specific():
    assert record_key("image", "neutral") != record_key("image", "existential")
