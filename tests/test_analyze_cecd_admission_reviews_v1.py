from anchor.corrected_sgta.analyze_cecd_admission_reviews_v1 import (
    EXPECTED_CANDIDATE_PROMPTS,
    EXPECTED_NONBASELINE_RENDERS,
    analyze,
)
from anchor.corrected_sgta.run_cecd_factorial_v1 import IDENTITY_RENDER_NAME


def _mapping():
    return {
        "clinical_pairs": [
            {"pair_id": f"{transform}-{i}", "transform": transform}
            for transform in EXPECTED_NONBASELINE_RENDERS
            for i in range(20)
        ] + [
            {"pair_id": f"id{i}", "transform": IDENTITY_RENDER_NAME}
            for i in range(12)
        ],
        "language_items": [
            {"item_id": f"{prompt}-{i}", "candidate_prompt": prompt}
            for prompt in EXPECTED_CANDIDATE_PROMPTS
            for i in range(4)
        ],
    }


def _clinical(mapping, changed=()):
    return [
        {
            "pair_id": row["pair_id"],
            "support_state_same_supported_refuted_undetermined": "no" if row["pair_id"] in changed else "yes",
            "lesion_visibility": "unchanged",
            "clinically_interchangeable": "yes",
            "unable_to_judge": "no",
        }
        for row in mapping["clinical_pairs"]
    ]


def _language(mapping, value="yes"):
    return [
        {"item_id": row["item_id"], "same_clinical_proposition": value, "same_speech_act": value, "same_certainty_demand": value, "same_answer_space": value}
        for row in mapping["language_items"]
    ]


def test_admission_requires_both_clinicians_and_both_language_roles():
    mapping = _mapping()
    good = _clinical(mapping)
    language = _language(mapping)
    result = analyze(mapping=mapping, clinical_reviews=[good, good], template_review=language, language_review=language)
    assert result["passed"]
    assert result["cecd_model_scoring_authorized"]


def test_one_reviewers_render_change_above_five_percent_fails():
    mapping = _mapping()
    good = _clinical(mapping)
    # Two of twenty primary pairs is 10%; identity controls remain clean.
    transform = EXPECTED_NONBASELINE_RENDERS[0]
    bad = _clinical(mapping, changed={f"{transform}-0", f"{transform}-1"})
    language = _language(mapping)
    result = analyze(mapping=mapping, clinical_reviews=[good, bad], template_review=language, language_review=language)
    assert not result["passed"]
    assert not result["cecd_model_scoring_authorized"]


def test_admission_rejects_partial_render_set_even_when_two_families_pass():
    mapping = _mapping()
    retained = set(EXPECTED_NONBASELINE_RENDERS[:2]) | {IDENTITY_RENDER_NAME}
    mapping["clinical_pairs"] = [
        row for row in mapping["clinical_pairs"] if row["transform"] in retained
    ]
    good = _clinical(mapping)
    language = _language(mapping)
    result = analyze(
        mapping=mapping,
        clinical_reviews=[good, good],
        template_review=language,
        language_review=language,
    )
    assert result["passed"] is False
    assert result["cecd_model_scoring_authorized"] is False
    assert result["science_grid_contract"]["render_set_exact"] is False


def test_admission_rejects_partial_prompt_set():
    mapping = _mapping()
    mapping["language_items"] = [
        row
        for row in mapping["language_items"]
        if row["candidate_prompt"] == EXPECTED_CANDIDATE_PROMPTS[0]
    ]
    good = _clinical(mapping)
    language = _language(mapping)
    result = analyze(
        mapping=mapping,
        clinical_reviews=[good, good],
        template_review=language,
        language_review=language,
    )
    assert result["passed"] is False
    assert result["cecd_model_scoring_authorized"] is False
    assert result["science_grid_contract"]["prompt_set_exact"] is False
