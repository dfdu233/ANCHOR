from anchor.corrected_sgta.audit_information_mismatch import (
    NON_VISUAL,
    TIER_A,
    TIER_B,
    labels_for,
    split_sentences,
)


def test_explicit_prior_and_history_are_tier_a() -> None:
    prior = "Compared with the prior study, the edema has improved."
    history = "The findings are consistent with the patient's history of renal disease."
    assert labels_for(prior, TIER_A) == ["prior_image"]
    assert labels_for(history, TIER_A) == ["clinical_history"]


def test_weak_temporal_and_recommendation_do_not_inflate_tier_a() -> None:
    weak = "Stable bibasilar reticular markings."
    recommendation = "Recommend CT follow-up."
    assert not labels_for(weak, TIER_A)
    assert labels_for(weak, TIER_B) == ["implicit_temporal"]
    assert not labels_for(recommendation, TIER_A)
    assert labels_for(recommendation, NON_VISUAL) == ["management_or_followup"]


def test_other_modality_is_tier_a_and_sentence_split_is_stable() -> None:
    report = "A nodule was seen on prior CT. No focal opacity.\nNo effusion."
    sentences = split_sentences(report)
    assert len(sentences) == 3
    assert labels_for(sentences[0], TIER_A) == ["other_test"]


def test_spatial_comparison_and_requested_labs_are_not_prior_evidence() -> None:
    spatial = "The opacity is more prominent on the left compared to the right."
    requested = "Correlation with laboratory results may be useful."
    explicit_prior = "Compared to the previous radiograph, edema has improved."
    etiologic_history = "Scarring could be due to previous infections."
    assert not labels_for(spatial, TIER_A)
    assert not labels_for(requested, TIER_A)
    assert not labels_for(etiologic_history, TIER_A)
    assert labels_for(explicit_prior, TIER_A) == ["prior_image"]
