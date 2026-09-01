from __future__ import annotations

from anchor.corrected_sgta.build_pubmedvision_source_semantic_admission_v1 import (
    FINDINGS,
    classify_finding,
    classify_question,
    is_generic_alignment_prompt,
    select_review_rows,
    source_group,
    source_split,
)


def spec(name: str):
    return next(value for value in FINDINGS if value.finding == name)


def test_unmentioned_is_never_inferred_negative_from_normal_language() -> None:
    text = "Normal chest x-ray with no acute findings and a sharp costophrenic angle."
    for finding in FINDINGS:
        assert classify_finding(text, finding)["state"] == "unmentioned"


def test_pleural_effusion_four_states_are_conservative() -> None:
    finding = spec("pleural_effusion")
    assert classify_finding("A small pleural effusion is present.", finding)["state"] == "positive"
    assert classify_finding("No pleural effusion.", finding)["state"] == "negative"
    assert classify_finding("No definite pleural effusion.", finding)["state"] == "uncertain"
    assert classify_finding("Pleural effusion cannot be excluded.", finding)["state"] == "uncertain"
    assert classify_finding("No change in the small pleural effusion.", finding)["state"] == "positive"
    assert classify_finding("There is a dull costophrenic angle.", finding)["state"] == "unmentioned"


def test_exact_aliases_and_non_alias_proxies() -> None:
    assert classify_finding("There is cardiomegaly.", spec("cardiomegaly"))["state"] == "positive"
    assert classify_finding("The cardiothoracic ratio is 60%.", spec("cardiomegaly"))["state"] == "unmentioned"
    assert classify_finding("Pulmonary fibrosis is present.", spec("pulmonary_fibrosis"))["state"] == "positive"
    assert classify_finding("Linear pulmonary scarring is present.", spec("pulmonary_fibrosis"))["state"] == "unmentioned"
    assert classify_finding("The thoracic aorta is ectatic.", spec("aortic_enlargement"))["state"] == "positive"


def test_question_is_a_separate_presupposition_channel() -> None:
    finding = spec("pleural_effusion")
    neutral = classify_question("Does this radiograph show a pleural effusion?", finding)
    asserted = classify_question("Describe the extent of the pleural effusion.", finding)
    assert neutral["presupposition"] == "neutral_query"
    assert asserted["presupposition"] == "presupposes_positive"


def test_source_split_is_group_deterministic() -> None:
    assert source_group("images/pmc_123_0.jpg") == "pmc_123"
    assert source_group("images/pmc_123_4.jpg") == "pmc_123"
    assert source_split("pmc_123") == source_split("pmc_123")
    assert source_split("pmc_123") in {"source_train", "source_dev", "source_review"}


def test_generic_alignment_prompt_is_conservative() -> None:
    assert is_generic_alignment_prompt("Analyze the image in a comprehensive and detailed manner.")
    assert is_generic_alignment_prompt("What is depicted in the images?")
    assert not is_generic_alignment_prompt("What is the extent of the pleural effusion?")


def test_unrelated_hedge_outside_clause_does_not_pollute_mention() -> None:
    finding = spec("pleural_effusion")
    text = "Pneumonia is likely. A pleural effusion is present."
    assert classify_finding(text, finding)["state"] == "positive"


def test_conflicting_temporal_mentions_fail_conservatively_to_uncertain() -> None:
    finding = spec("pleural_effusion")
    text = "There was a prior pleural effusion; no pleural effusion is present now."
    assert classify_finding(text, finding)["state"] == "uncertain"


def test_source_group_split_is_identical_across_stages() -> None:
    group = source_group("images/pmc_987_0.jpg")
    alignment_split = source_split(group)
    instruction_split = source_split(source_group("images/pmc_987_4.jpg"))
    assert alignment_split == instruction_split


def test_review_selection_preserves_minimum_finding_quota() -> None:
    rows = []
    for stage in ("alignment", "instruction_tuning"):
        for finding in FINDINGS:
            for state in ("positive", "negative", "uncertain", "unmentioned"):
                for index in range(5):
                    rows.append(
                        {
                            "source_split": "source_review",
                            "stage": stage,
                            "finding": finding.finding,
                            "assistant_state": state,
                            "response_unit_id": f"{stage}:{finding.finding}:{state}:{index}",
                        }
                    )
    selected = select_review_rows(rows, size=160, seed=3)
    counts = {finding.finding: 0 for finding in FINDINGS}
    for row in selected:
        counts[row["finding"]] += 1
    assert len(selected) == 160
    assert set(counts.values()) == {40}
