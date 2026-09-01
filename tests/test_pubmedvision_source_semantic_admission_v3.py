from __future__ import annotations

from anchor.corrected_sgta.build_pubmedvision_source_semantic_admission_v3 import (
    DEFAULT_ONTOLOGY,
    classify_finding_v3,
    eligibility_from_counts,
    load_ontology,
    raw_counts,
    raw_question_counts,
    select_blind_review,
)


def _spec(name: str):
    _, specs = load_ontology(DEFAULT_ONTOLOGY)
    return next(spec for spec in specs if spec.finding == name)


def test_prefrozen_ontology_has_25_atomic_claims() -> None:
    ontology, specs = load_ontology(DEFAULT_ONTOLOGY)
    assert ontology["schema_version"] == "pubmedvision-source-atomic-ontology-v3"
    assert len(specs) == 25
    assert not {"other_lesion", "other_diseases", "no_finding"} & {
        spec.finding for spec in specs
    }


def test_emphysema_bare_term_has_boundaries() -> None:
    spec = _spec("emphysema")
    assert classify_finding_v3("There is emphysema.", spec)["state"] == "positive"
    assert classify_finding_v3("An emphysematous bulla is present.", spec)["state"] == "unmentioned"


def test_calcification_requires_explicit_lung_context() -> None:
    spec = _spec("calcification")
    assert classify_finding_v3("Aortic valve calcification is present.", spec)["state"] == "unmentioned"
    assert classify_finding_v3("Pulmonary calcification is present.", spec)["state"] == "positive"


def test_unmentioned_is_not_negative_in_broad_ontology() -> None:
    _, specs = load_ontology(DEFAULT_ONTOLOGY)
    for spec in specs:
        assert classify_finding_v3("No acute abnormality is identified.", spec)["state"] == "unmentioned"


def test_distributed_list_negation_is_generic_and_adversative_resets_scope() -> None:
    effusion = _spec("pleural_effusion")
    consolidation = _spec("consolidation")
    assert classify_finding_v3(
        "The lungs are clear without any consolidation, masses, or pleural effusion.",
        effusion,
    )["state"] == "negative"
    assert classify_finding_v3(
        "The lungs are clear without any consolidation, masses, or pleural effusion.",
        consolidation,
    )["state"] == "negative"
    assert classify_finding_v3(
        "There is no consolidation, but a pleural effusion is present.", effusion
    )["state"] == "positive"
    assert classify_finding_v3(
        "There is an absence of other abnormalities such as pleural effusion.", effusion
    )["state"] == "negative"
    assert classify_finding_v3(
        "There is no change in the small pleural effusion.", effusion
    )["state"] == "positive"


def test_differential_example_is_uncertain_not_definite() -> None:
    pneumonia = _spec("pneumonia")
    assert classify_finding_v3(
        "This pattern can occur in conditions such as pneumonia or pulmonary edema.",
        pneumonia,
    )["state"] == "uncertain"
    assert classify_finding_v3("There is right lower lobe pneumonia.", pneumonia)["state"] == "positive"


def test_do_not_temporal_resolution_and_adjectival_hedge_are_not_positive() -> None:
    pneumothorax = _spec("pneumothorax")
    effusion = _spec("pleural_effusion")
    cardiomegaly = _spec("cardiomegaly")
    consolidation = _spec("consolidation")
    assert classify_finding_v3(
        "The findings do not indicate fluid levels or pneumothoraces.", pneumothorax
    )["state"] == "negative"
    assert classify_finding_v3(
        "The right pleural effusion observed previously has resolved.", effusion
    )["state"] == "uncertain"
    assert classify_finding_v3(
        "Cardiac enlargement is not significantly present.", cardiomegaly
    )["state"] == "uncertain"
    assert classify_finding_v3(
        "There is a suspicious consolidation in the upper lung.", consolidation
    )["state"] == "uncertain"
    assert classify_finding_v3(
        "The differential might include a pleural effusion.", effusion
    )["state"] == "uncertain"
    assert classify_finding_v3(
        "The image appears to show focal consolidation.", consolidation
    )["state"] == "uncertain"
    assert classify_finding_v3(
        "The right lung seems to have dense consolidation.", consolidation
    )["state"] == "uncertain"


def test_eligibility_is_exactly_generic_alignment_train_and_dev() -> None:
    claims = ["a", "b", "c"]
    template = {
        claim: {
            split: {state: 0 for state in ("positive", "negative", "uncertain", "unmentioned")}
            for split in ("source_train", "source_dev", "source_review")
        }
        for claim in claims
    }
    template["a"]["source_train"]["positive"] = 20
    template["a"]["source_dev"]["positive"] = 5
    template["b"]["source_train"]["positive"] = 19
    template["b"]["source_dev"]["positive"] = 100
    template["c"]["source_train"]["positive"] = 100
    template["c"]["source_dev"]["positive"] = 4
    counts = {"alignment_generic": template}
    result = eligibility_from_counts(counts, claims)
    assert [row["claim_id"] for row in result if row["automatic_count_eligible"]] == ["a"]


def test_raw_counts_never_merges_alignment_or_instruction_domains() -> None:
    rows = [
        {"finding": "x", "source_split": "source_train", "stage": "alignment", "review_domain": "alignment_generic", "assistant_state": "positive"},
        {"finding": "x", "source_split": "source_train", "stage": "alignment", "review_domain": None, "assistant_state": "positive"},
        {"finding": "x", "source_split": "source_train", "stage": "instruction_tuning", "review_domain": "instruction_tuning", "assistant_state": "positive"},
    ]
    counts = raw_counts(rows, ["x"])
    assert counts["alignment_all"]["x"]["source_train"]["positive"] == 2
    assert counts["alignment_generic"]["x"]["source_train"]["positive"] == 1
    assert counts["instruction_tuning"]["x"]["source_train"]["positive"] == 1


def test_question_presupposition_counts_are_stage_and_split_separate() -> None:
    rows = [
        {"finding": "x", "source_split": "source_train", "stage": "alignment", "question_presupposition": "none"},
        {"finding": "x", "source_split": "source_train", "stage": "instruction_tuning", "question_presupposition": "presupposes_positive"},
    ]
    counts = raw_question_counts(rows, ["x"])
    assert counts["alignment"]["x"]["source_train"]["none"] == 1
    assert counts["alignment"]["x"]["source_train"]["presupposes_positive"] == 0
    assert counts["instruction_tuning"]["x"]["source_train"]["presupposes_positive"] == 1


def _review_row(claim: str, domain: str, state: str, index: int) -> dict:
    stage = "alignment" if domain == "alignment_generic" else "instruction_tuning"
    return {
        "finding": claim,
        "review_domain": domain,
        "assistant_state": state,
        "source_split": ("source_train", "source_dev", "source_review")[index % 3],
        "response_unit_id": f"{claim}:{domain}:{state}:{index}",
        "stage": stage,
    }


def test_review_is_eligible_only_stage_separate_and_stratified() -> None:
    rows = []
    for claim in ("eligible", "excluded"):
        for domain in ("alignment_generic", "instruction_tuning"):
            for state in ("positive", "negative", "uncertain", "unmentioned"):
                for index in range(40):
                    rows.append(_review_row(claim, domain, state, index))
    selected, plan = select_blind_review(rows, {"eligible"}, seed=3)
    assert {row["finding"] for row in selected} == {"eligible"}
    assert len(selected) == 2 * 4 * 30
    assert {row["review_domain"] for row in selected} == {
        "alignment_generic",
        "instruction_tuning",
    }
    assert plan["eligible"]["alignment_generic"]["positive_precision_zero_error_bound_capable"]
    assert all(row["design_weight"] == 40 / 30 for row in selected)


def test_sparse_review_stratum_is_censused_not_padded() -> None:
    rows = [_review_row("eligible", "alignment_generic", "positive", index) for index in range(7)]
    selected, plan = select_blind_review(rows, {"eligible"}, seed=9)
    assert len(selected) == 7
    stratum = plan["eligible"]["alignment_generic"]["predicted_state_strata"]["positive"]
    assert stratum["census"]
    assert stratum["sample_n"] == stratum["population_n"] == 7
    assert not plan["eligible"]["alignment_generic"]["positive_precision_zero_error_bound_capable"]


def test_review_selection_is_deterministic() -> None:
    rows = [_review_row("eligible", "alignment_generic", "positive", index) for index in range(50)]
    first, _ = select_blind_review(rows, {"eligible"}, seed=11)
    second, _ = select_blind_review(list(reversed(rows)), {"eligible"}, seed=11)
    assert [row["response_unit_id"] for row in first] == [row["response_unit_id"] for row in second]
