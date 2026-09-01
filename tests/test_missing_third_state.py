import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from corrected_sgta.clinical_claims import (
    ClinicalClaim,
    bounded_state,
    claims_to_fixed_oe_rows,
    commitment_bounded_claims,
    evidence_bounded_commitment_projection,
    evidence_conserving_claim_exchange,
    epistemic_coordinates,
    epistemic_state,
    evaluate_claim_rows,
    evaluate_oe_claim_rows,
    evaluate_oe_methods_matched_coverage,
    paired_clinical_selectivity,
    oe_prediction_axes,
    polarity_preserving_commitment_claims,
    reader_calibrated_state_distribution,
    reader_state,
    simplex_logits,
    support_commitment_decomposition,
    tristate_logits,
    validate_oe_reference_provenance,
)
from corrected_sgta.prepare_vindr_reader_manifest import (
    balanced_subset,
    build_oe_listing_records,
    build_records,
    experiment_split,
    reader_effect_summary,
    select_ontology_columns,
    read_votes,
)
from corrected_sgta.prepare_vindr_selectivity_triplets import build_triplets
from corrected_sgta.fit_clinical_response_aligner import probability_metrics
from corrected_sgta.fit_reader_agreement_gate import (
    agreement_metrics,
    main as reader_agreement_main,
    paired_cluster_bootstrap_continuous,
    paired_cluster_bootstrap_increment,
)
from corrected_sgta.fit_reader_adjusted_support import (
    adjust_rows,
    fit_reader_effects,
    infer_item_map,
)
from corrected_sgta.analyze_clinical_selectivity import directional_admission_gates
from corrected_sgta.apply_reader_calibrated_projection import (
    project_row,
    validate_mechanism_authorization,
)
from corrected_sgta.authorize_reader_grounded_projection import (
    authorization_fingerprint,
    build_authorization,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    auc_or_none,
    deterministic_orthogonal_direction,
    freeze_or_validate_config,
    intervention_coordinate_changes,
    null_commitment_direction,
    norm_matched_direction_subtraction,
    orthogonalized_unit_direction,
    sha256_file,
    validate_global_null_sidecar,
)
from corrected_sgta.screen_clinical_presupposition import (
    analyze as analyze_clinical_presupposition,
)
from corrected_sgta.radgraph_claims import claims_from_radgraph


def test_reader_state_preserves_both_disagreement_bins():
    assert reader_state(0, 3) == "refuted"
    assert reader_state(1, 3) == "undetermined"
    assert reader_state(2, 3) == "undetermined"
    assert reader_state(3, 3) == "supported"


def test_local_auroc_is_tie_aware_and_has_no_sklearn_dependency():
    assert auc_or_none([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]) == 1.0
    assert auc_or_none([0, 1], [0.0, 0.0]) == 0.5
    assert auc_or_none([1, 1], [0.0, 1.0]) is None


def test_probe_resume_preserves_original_fingerprint_and_rejects_drift(tmp_path: Path):
    path = tmp_path / "config.json"
    original = {"model": "m", "created_at": "old", "command": "first", "fingerprint": "f"}
    freeze_or_validate_config(original, path, resume=False)
    resumed = freeze_or_validate_config(
        {**original, "created_at": "new", "command": "second", "fingerprint": "new"},
        path,
        resume=True,
    )
    assert resumed["fingerprint"] == "f"
    with pytest.raises(ValueError, match="config drift"):
        freeze_or_validate_config({**original, "model": "other"}, path, resume=True)


def test_vindr_formal_manifest_uses_the_frozen_claim_ontology():
    selected, excluded = select_ontology_columns(
        ["Pleural effusion", "No finding", "Nodule/Mass", "Other diseases"],
        {"pleural_effusion", "nodule_mass"},
    )
    assert selected == ["Pleural effusion", "Nodule/Mass"]
    assert excluded == ["No finding", "Other diseases"]


def test_clinical_selectivity_rewards_target_change_over_nuisance_drift():
    result = paired_clinical_selectivity(2.0, 1.8, -1.0, 1.0, 0.0)
    assert result["signed_clinical_change"] == 3.0
    assert result["absolute_clinical_change"] == 3.0
    assert abs(result["absolute_nuisance_change"] - 0.2) < 1e-12
    assert abs(result["clinical_selectivity_gap"] - 2.8) < 1e-12
    assert abs(result["unsigned_selectivity_gap"] - 2.8) < 1e-12
    assert result["unsigned_responsive"] == 1.0
    assert result["misdirected_responsive"] == 0.0


def test_clinical_selectivity_exposes_visually_sensitive_but_wrong_model():
    result = paired_clinical_selectivity(0.5, -0.5, 0.75, 1.0, 0.0)
    assert result["absolute_nuisance_change"] == 1.0
    assert result["signed_clinical_change"] == -0.25
    assert result["clinical_selectivity_gap"] == -1.25


def test_unsigned_response_can_falsely_look_grounded_in_wrong_direction():
    result = paired_clinical_selectivity(0.5, 0.45, 1.25, 1.0, 0.0)
    assert result["absolute_clinical_change"] == 0.75
    assert result["unsigned_selectivity_gap"] == 0.7
    assert result["signed_clinical_change"] == -0.75
    assert result["unsigned_responsive"] == 1.0
    assert result["misdirected_responsive"] == 1.0


def test_reader_response_metrics_keep_the_disagreement_state():
    result = probability_metrics(
        positive_votes=[0, 1, 2, 3],
        reader_counts=[3, 3, 3, 3],
        scores=[-4.0, -0.2, 0.2, 4.0],
    )
    assert result["tristate_accuracy"] == 1.0
    assert result["disagreement_overcommitment_rate"] == 0.0
    assert result["unanimous_fabrication_rate"] == 0.0
    assert result["unanimous_omission_rate"] == 0.0


def test_reader_calibrated_distribution_keeps_polarity_and_clarity_independent():
    distribution = reader_calibrated_state_distribution(
        support_probability=0.8,
        clarity_probability=0.25,
    )
    assert abs(distribution["supported"] - 0.2) < 1e-12
    assert abs(distribution["refuted"] - 0.05) < 1e-12
    assert abs(distribution["undetermined"] - 0.75) < 1e-12
    assert abs(sum(distribution.values()) - 1.0) < 1e-12


def test_evidence_projection_caps_commitment_without_changing_definite_odds():
    evidence = reader_calibrated_state_distribution(0.8, 0.3)
    projected, audit = evidence_bounded_commitment_projection(
        {"supported": 0.8, "refuted": 0.1, "undetermined": 0.1},
        evidence,
    )
    assert abs(projected["supported"] + projected["refuted"] - 0.3) < 1e-12
    assert abs(projected["undetermined"] - 0.7) < 1e-12
    assert abs(projected["supported"] / projected["refuted"] - 8.0) < 1e-12
    assert audit["commitment_capped"] is True
    assert audit["polarity_clipped_to_boundary"] is False
    assert audit["projected_state"] == "undetermined"


def test_evidence_projection_hedges_a_polarity_contradiction_without_flipping_it():
    evidence = reader_calibrated_state_distribution(0.9, 0.8)
    projected, audit = evidence_bounded_commitment_projection(
        {"supported": 0.05, "refuted": 0.9, "undetermined": 0.05},
        evidence,
        polarity_margin=0.2,
    )
    assert abs(projected["supported"] - projected["refuted"]) < 1e-12
    assert audit["polarity_clipped_to_boundary"] is True
    assert audit["projected_state"] == "undetermined"


def test_evidence_projection_leaves_a_compliant_clear_claim_unchanged():
    decoder = {"supported": 0.75, "refuted": 0.05, "undetermined": 0.2}
    evidence = reader_calibrated_state_distribution(0.9, 0.9)
    projected, audit = evidence_bounded_commitment_projection(decoder, evidence)
    assert projected == decoder
    assert audit["commitment_capped"] is False
    assert audit["polarity_clipped_to_boundary"] is False
    assert audit["forward_kl_from_decoder"] == 0.0


def test_formal_projection_requires_image_disjoint_reader_vote_calibration():
    row = {
        "image_id": "test-image",
        "finding": "Pleural effusion",
        "decoder_probabilities": {
            "supported": 0.8,
            "refuted": 0.1,
            "undetermined": 0.1,
        },
        "calibrated_support_probability": 0.8,
        "calibrated_clarity_probability": 0.3,
        "calibration_provenance": {
            "formal_reference": True,
            "reference_source": "vindr_reader_votes",
            "calibration_split": "dev",
            "image_disjoint_from_target": True,
            "support_calibrator_sha256": "a" * 64,
            "clarity_calibrator_sha256": "b" * 64,
            "calibration_manifest_sha256": "c" * 64,
            "ontology_sha256": "d" * 64,
        },
    }
    projected = project_row(row, 0.0, 0.0, formal=True)
    assert projected["finding"] == "pleural_effusion"
    assert projected["prediction_state"] == "undetermined"
    expected_hashes = {
        "support_calibrator_sha256": "a" * 64,
        "clarity_calibrator_sha256": "b" * 64,
        "calibration_manifest_sha256": "c" * 64,
        "ontology_sha256": "e" * 64,
    }
    try:
        project_row(
            row,
            0.0,
            0.0,
            formal=True,
            expected_hashes=expected_hashes,
        )
    except ValueError as error:
        assert "hashes do not match files" in str(error)
    else:
        raise AssertionError("formal projection accepted hashes from different files")
    invalid = dict(row)
    invalid["calibration_provenance"] = {
        **row["calibration_provenance"],
        "calibration_split": "test",
    }
    try:
        project_row(invalid, 0.0, 0.0, formal=True)
    except ValueError as error:
        assert "inadmissible calibration provenance" in str(error)
    else:
        raise AssertionError("formal projection accepted test-fitted calibration")
    unhashed = dict(row)
    unhashed["calibration_provenance"] = {
        key: value
        for key, value in row["calibration_provenance"].items()
        if key != "ontology_sha256"
    }
    try:
        project_row(unhashed, 0.0, 0.0, formal=True)
    except ValueError as error:
        assert "missing or invalid calibration hashes" in str(error)
    else:
        raise AssertionError("formal projection accepted unhashed calibration")


def test_directional_admission_requires_ordering_selectivity_and_finding_majority():
    passing = {
        "reader_support_spearman_bootstrap": {
            "estimate": 0.4,
            "ci_low": 0.1,
            "ci_high": 0.6,
        },
        "clinical_selectivity_gap": {
            "estimate": 0.3,
            "ci_low": 0.05,
            "ci_high": 0.5,
        },
        "by_finding": {
            "effusion": {
                "anchor_vote_bin_counts": {str(vote): 12 for vote in range(4)},
                "reader_support_spearman_bootstrap": {
                    "ci_low": 0.04,
                },
                "clinical_selectivity_gap_bootstrap": {"ci_low": 0.02},
            },
            "opacity": {
                "anchor_vote_bin_counts": {str(vote): 12 for vote in range(4)},
                "reader_support_spearman_bootstrap": {
                    "ci_low": 0.03,
                },
                "clinical_selectivity_gap_bootstrap": {"ci_low": 0.01},
            },
            "nodule": {
                "anchor_vote_bin_counts": {str(vote): 12 for vote in range(4)},
                "reader_support_spearman_bootstrap": {
                    "ci_low": -0.01,
                },
                "clinical_selectivity_gap_bootstrap": {"ci_low": 0.01},
            },
        },
    }
    gates, summary = directional_admission_gates(passing, True, True, 10)
    assert gates["directional_admission_authorized"] is True
    assert summary["passed_findings"] == ["effusion", "opacity"]
    passing["clinical_selectivity_gap"]["ci_low"] = -0.01
    gates, _ = directional_admission_gates(passing, True, True, 10)
    assert gates["directional_admission_authorized"] is False


def test_projection_authorization_binds_all_mechanism_gates_to_one_model():
    model_id = "huatuo-7b"
    directional = {
        "model_id": model_id,
        "experiment_split": "test",
        "test_layer_selected_without_test_labels": True,
        "formal_reference": True,
        "mechanism_gates": {"directional_admission_authorized": True},
    }
    tetrad = {
        "model_id": model_id,
        "formal_reference": True,
        "mechanism_gates": {"observational_erasure_authorized": True},
    }
    clarity = {
        "model_id": model_id,
        "formal_reference": True,
        "mechanism_gates": {"measurement_authorized": True},
    }
    hashes = {
        "directional_admission": "a" * 64,
        "tetrad_erasure": "b" * 64,
        "clarity_gate": "c" * 64,
        "support_calibrator": "d" * 64,
        "boundary_classification": "e" * 64,
    }
    boundary = {
        "method_branch_authorized": True,
        "model_gate": {
            "huatuo-7b": {"strict_majority": True},
            "hulu-4b": {"strict_majority": True},
        },
    }
    authorization = build_authorization(
        model_id, directional, tetrad, clarity, boundary, hashes
    )
    assert authorization["reader_grounded_projection_authorized"] is True
    assert authorization["fingerprint"] == authorization_fingerprint(authorization)
    assert (
        validate_mechanism_authorization(authorization, "c" * 64, "d" * 64)
        == model_id
    )

    wrong_model = build_authorization(
        "other-model", directional, tetrad, clarity, boundary, hashes
    )
    assert wrong_model["reader_grounded_projection_authorized"] is False
    try:
        validate_mechanism_authorization(authorization, "c" * 64, "e" * 64)
    except ValueError as error:
        assert "support calibrator" in str(error)
    else:
        raise AssertionError("cross-model support calibrator was accepted")
    tampered = {**authorization, "model_id": "hulu-4b"}
    try:
        validate_mechanism_authorization(tampered, "c" * 64)
    except ValueError as error:
        assert "modified" in str(error)
    else:
        raise AssertionError("tampered authorization was accepted")


def test_presupposition_screen_requires_bidirectional_clinical_errors_at_matched_length():
    rows = []
    for model_id in ("huatuo-7b", "hulu-4b"):
        for item_index in range(4):
            for condition in ("neutral", "existential", "negative_obligation"):
                rows.append(
                    {
                        "item_id": f"image-{item_index}",
                        "model_id": model_id,
                        "prompt_condition": condition,
                        "token_count": 40 + int(condition != "neutral"),
                        "claim_universe_sha256": "a" * 64,
                        "supported_claim_count": 10,
                        "refuted_claim_count": 10,
                        "positive_claim_error_count": 2 if condition == "existential" else 0,
                        "negative_claim_error_count": 2 if condition == "negative_obligation" else 0,
                        "omitted_supported_claim_count": 1,
                        "formal_reference": True,
                        "adjudication_source": "physician",
                        "automatic_labeler_only": False,
                        "ground_truth_used_for_generation_or_selection": False,
                    }
                )
    result = analyze_clinical_presupposition(
        rows,
        bootstrap_draws=100,
        seed=42,
        minimum_pairs=4,
        minimum_models=2,
        maximum_absolute_length_gap=2,
        maximum_relative_length_gap=0.1,
    )
    assert result["clinical_presupposition_amplification_survives"] is True
    assert result["passed_models"] == ["huatuo-7b", "hulu-4b"]

    for row in rows:
        if row["prompt_condition"] == "existential":
            row["token_count"] = 100
    result = analyze_clinical_presupposition(
        rows,
        bootstrap_draws=100,
        seed=42,
        minimum_pairs=4,
        minimum_models=2,
        maximum_absolute_length_gap=2,
        maximum_relative_length_gap=0.1,
    )
    assert result["clinical_presupposition_amplification_survives"] is False


def test_agreement_gate_changes_only_commitment_not_polarity():
    result = agreement_metrics(
        positive_votes=[0, 1, 2, 3],
        reader_counts=[3, 3, 3, 3],
        agreement_logits=[4.0, -4.0, -4.0, 4.0],
        polarity_scores=[-2.0, -0.2, 0.2, 2.0],
    )
    assert result["tri_state_accuracy"] == 1.0
    assert result["clear_case_accuracy"] == 1.0
    assert result["disagreement_overcommitment_rate"] == 0.0
    assert result["unanimous_fabrication_rate"] == 0.0
    assert result["unanimous_omission_rate"] == 0.0


def test_evidence_conserving_exchange_swaps_claims_without_shrinking():
    draft = [
        ClinicalClaim("cardiomegaly", uncertainty="uncertain"),
        ClinicalClaim("pleural_effusion"),
        ClinicalClaim("history", provenance="context"),
    ]
    revised, audit = evidence_conserving_claim_exchange(
        draft,
        {
            "cardiomegaly": -0.2,
            "pleural_effusion": 0.9,
            "pneumothorax": 1.4,
        },
        ("cardiomegaly", "pleural_effusion", "pneumothorax"),
        minimum_exchange_margin=0.5,
    )
    assert len(revised) == len(draft)
    assert [claim.finding for claim in revised] == [
        "pneumothorax",
        "pleural_effusion",
        "history",
    ]
    assert revised[0].uncertainty == "uncertain"
    assert revised[2] == draft[2]
    assert sum(row["action"] == "exchanged" for row in audit) == 1


def test_evidence_conserving_exchange_requires_a_real_score_gain():
    draft = [ClinicalClaim("cardiomegaly"), ClinicalClaim("pleural_effusion")]
    revised, audit = evidence_conserving_claim_exchange(
        draft,
        {"cardiomegaly": 0.4, "pleural_effusion": 0.8, "pneumothorax": 0.6},
        ("cardiomegaly", "pleural_effusion", "pneumothorax"),
        minimum_exchange_margin=0.3,
    )
    assert revised == draft
    assert all(row["action"] == "retained" for row in audit)


def test_evidence_conserving_exchange_can_fix_both_fabrication_and_omission_at_fixed_coverage():
    ontology = ("cardiomegaly", "pleural_effusion", "pneumothorax")
    supports = {"cardiomegaly": 0.0, "pleural_effusion": 1.0, "pneumothorax": 1.0}
    scores = {"cardiomegaly": -0.5, "pleural_effusion": 1.0, "pneumothorax": 1.5}
    metadata = {
        finding: {
            "reference_observability": "image_grounded",
            "reference_relevance": "required",
        }
        for finding in ontology
    }
    baseline = [ClinicalClaim("cardiomegaly"), ClinicalClaim("pleural_effusion")]
    exchanged, _ = evidence_conserving_claim_exchange(
        baseline, scores, ontology, minimum_exchange_margin=0.5
    )
    baseline_rows, _ = claims_to_fixed_oe_rows(
        "report-1", "baseline", baseline, supports, scores, metadata
    )
    exchange_rows, _ = claims_to_fixed_oe_rows(
        "report-1", "ecce", exchanged, supports, scores, metadata
    )
    evaluation = evaluate_oe_methods_matched_coverage(
        baseline_rows + exchange_rows, baseline_method="baseline"
    )
    assert evaluation["natural_positive_content_counts"] == {"baseline": 2, "ecce": 2}
    assert evaluation["natural"]["baseline"]["fabricated_claim_rate"] == 0.5
    assert evaluation["natural"]["ecce"]["fabricated_claim_rate"] == 0.0
    assert evaluation["natural"]["baseline"][
        "required_unanimous_positive_omission_rate"
    ] == 0.5
    assert evaluation["natural"]["ecce"][
        "required_unanimous_positive_omission_rate"
    ] == 0.0
    assert evaluation["coverage_guard_pass"]["ecce"] is True
    assert evaluation["omission_nonincrease_pass"]["ecce"] is True


def test_support_commitment_gap_decomposes_exactly():
    result = support_commitment_decomposition(
        reader_support=1 / 3,
        predicted_support=0.8,
        prediction_state="supported",
    )
    assert abs(
        result["signed_total_gap"]
        - result["signed_language_transfer_gap"]
        - result["signed_visual_support_gap"]
    ) < 1e-12
    assert abs(result["decomposition_residual"]) < 1e-12
    assert result["signed_language_transfer_gap"] > 0
    assert result["signed_visual_support_gap"] > 0


def test_bounded_decoding_hedges_instead_of_deleting_and_adds_omissions():
    draft = [ClinicalClaim("Pleural effusion", anatomy="left")]
    claims, audit = commitment_bounded_claims(
        draft,
        {"pleural_effusion": 0.1, "pneumothorax": 2.0},
        ["Pleural effusion", "Pneumothorax"],
        tau=0.6,
        add_threshold=1.5,
        required_findings=["Pneumothorax"],
    )
    assert len(claims) == 2
    assert claims[0].uncertainty == "uncertain"
    assert claims[0].anatomy == "left"
    assert claims[1].finding == "pneumothorax"
    assert claims[1].state == "supported"
    assert [row["action"] for row in audit] == [
        "bounded_to_undetermined",
        "added_omitted_high_support",
    ]


def test_high_support_optional_claim_is_not_forced_into_report():
    claims, audit = commitment_bounded_claims(
        [],
        {"pleural_effusion": 2.0},
        ["Pleural effusion"],
        tau=0.6,
        add_threshold=1.5,
        required_findings=[],
    )
    assert claims == []
    assert audit == [
        {
            "finding": "pleural_effusion",
            "action": "not_added_without_required_relevance",
            "draft_state": "absent_from_draft",
            "final_state": "unmentioned",
            "evidence": 2.0,
        }
    ]


def test_reader_agreement_rewrite_changes_only_certainty():
    draft = [
        ClinicalClaim("Pleural effusion", polarity="present", anatomy="left"),
        ClinicalClaim(
            "Pneumothorax", polarity="absent", uncertainty="uncertain"
        ),
        ClinicalClaim("Treatment", provenance="knowledge"),
    ]
    revised, audit = polarity_preserving_commitment_claims(
        draft,
        {"pleural_effusion": 0.2, "pneumothorax": 0.9, "treatment": 0.9},
        clear_threshold=0.5,
    )
    assert len(revised) == len(draft)
    assert [claim.key for claim in revised] == [claim.key for claim in draft]
    assert [claim.polarity for claim in revised] == [claim.polarity for claim in draft]
    assert [claim.state for claim in revised] == [
        "undetermined",
        "refuted",
        "unobservable",
    ]
    assert [row["action"] for row in audit] == [
        "hedged",
        "unhedged",
        "outside_image_grounded_gate",
    ]


def test_hedging_preserves_positive_content_axis():
    axes = oe_prediction_axes(
        {
            "emitted": True,
            "prediction_state": "undetermined",
            "prediction_polarity": "present",
            "prediction_uncertainty": "uncertain",
        }
    )
    assert axes == {"polarity": "present", "uncertainty": "uncertain"}


def test_emitted_undetermined_claim_cannot_erase_its_polarity():
    try:
        oe_prediction_axes(
            {"emitted": True, "prediction_state": "undetermined"}
        )
    except ValueError as error:
        assert "hedging must not erase" in str(error)
    else:
        raise AssertionError("polarity-free hedged OE claim was accepted")


def test_evaluator_exposes_short_answer_and_all_negative_cheats():
    rows = [
        {"reader_support": 1.0, "prediction_state": "refuted"},
        {"reader_support": 0.0, "prediction_state": "refuted"},
        {"reader_support": 1 / 3, "prediction_state": "undetermined"},
        {"reader_support": 2 / 3, "prediction_state": "supported"},
    ]
    result = evaluate_claim_rows(rows)
    assert result["coverage"] == 0.75
    assert result["unanimous_positive_omission_rate"] == 1.0
    assert result["disagreement_overcommitment_rate"] == 0.5
    assert result["mean_absolute_support_commitment_gap"] > 0
    assert result["mean_overcommitment_strength_gap"] > 0


def test_vindr_manifest_counts_and_balancing(tmp_path: Path):
    path = tmp_path / "image_labels_train.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "rad_ID", "Effusion"])
        writer.writeheader()
        for positives in range(4):
            for image_index in range(2):
                image_id = f"i{positives}{image_index}"
                for reader in range(3):
                    writer.writerow(
                        {
                            "image_id": image_id,
                            "rad_ID": f"r{reader}",
                            "Effusion": int(reader < positives),
                        }
                    )
    votes, findings, _, _ = read_votes(path)
    records, statistics = build_records(votes, findings, "https://example.test/train")
    assert statistics["Effusion"] == {"0/3": 2, "1/3": 2, "2/3": 2, "3/3": 2}
    selected = balanced_subset(records, {"Effusion"}, per_bin=1, seed=42)
    assert len(selected) == 4
    assert {row["reader_state"] for row in selected} == {
        "refuted",
        "undetermined",
        "supported",
    }
    assert all(str(row["dicom_url"]).endswith(".dicom") for row in selected)
    assert all(
        row["reader_votes"]
        == [
            {"rad_id": "r0", "vote": int(0 < row["positive_votes"])},
            {"rad_id": "r1", "vote": int(1 < row["positive_votes"])},
            {"rad_id": "r2", "vote": int(2 < row["positive_votes"])},
        ]
        for row in selected
    )
    audit = reader_effect_summary(records)
    assert audit["reader_identity_preserved"] is True
    assert audit["unique_readers"] == 3
    assert audit["images_per_reader"] == {"r0": 8, "r1": 8, "r2": 8}
    assert audit["per_finding_reader_counts"]["effusion"]["r0"] == {
        "positive": 6,
        "total": 8,
    }


def test_reader_effect_audit_rejects_aggregate_vote_tampering():
    row = {
        "image_id": "image-1",
        "finding": "effusion",
        "positive_votes": 2,
        "reader_count": 3,
        "reader_votes": [
            {"rad_id": "r0", "vote": 0},
            {"rad_id": "r1", "vote": 0},
            {"rad_id": "r2", "vote": 1},
        ],
    }
    try:
        reader_effect_summary([row])
    except ValueError as error:
        assert "aggregate positive_votes disagree" in str(error)
    else:
        raise AssertionError("tampered aggregate vote count was accepted")


def test_vindr_oe_manifest_expands_every_image_to_complete_ontology():
    rows = []
    for image_id, votes_by_finding in {
        "image-a": {"Effusion": 3, "Pneumothorax": 1},
        "image-b": {"Effusion": 0, "Pneumothorax": 2},
    }.items():
        for finding, votes in votes_by_finding.items():
            rows.append(
                {
                    "dataset": "vindr-cxr-1.0.0",
                    "reference_source": "vindr_reader_votes",
                    "formal_reference": True,
                    "image_id": image_id,
                    "finding": finding.lower(),
                    "finding_source_name": finding,
                    "positive_votes": votes,
                    "reader_count": 3,
                    "reader_support": votes / 3,
                }
            )
    expanded = build_oe_listing_records(
        rows,
        {"image-a", "image-b"},
        {"Effusion", "Pneumothorax"},
        seed=42,
        dev_fraction=0.25,
    )
    assert len(expanded) == 4
    assert {
        (row["image_id"], row["finding"], row["reference_relevance"])
        for row in expanded
    } == {
        ("image-a", "effusion", "required"),
        ("image-a", "pneumothorax", "optional"),
        ("image-b", "effusion", "out_of_scope"),
        ("image-b", "pneumothorax", "optional"),
    }
    assert all(
        row["reference_contract_version"] == "missing-third-state-claims-v8"
        for row in expanded
    )


def test_experiment_split_is_image_level_and_deterministic():
    first = experiment_split("image-1", seed=42, dev_fraction=0.25)
    assert first in {"dev", "test"}
    assert experiment_split("image-1", seed=42, dev_fraction=0.25) == first


def test_vindr_selectivity_triplets_are_balanced_and_image_disjoint():
    rows = []
    for votes in range(4):
        for index in range(6):
            rows.append(
                {
                    "image_id": f"v{votes}-{index}",
                    "finding": "pleural_effusion",
                    "positive_votes": votes,
                    "reader_count": 3,
                    "reader_support": votes / 3,
                    "reference_source": "vindr_reader_votes",
                    "formal_reference": True,
                    "experiment_split": "dev",
                    "dicom_relpath": f"train/v{votes}-{index}.dicom",
                    "dicom_metadata": {
                        "view_position": "pa",
                        "manufacturer": "unit_test",
                        "manufacturer_model": "unit_test",
                        "rows": 2048 + index,
                        "columns": 2048,
                        "aspect_ratio": 2048 / (2048 + index),
                    },
                }
            )
    triplets, summary = build_triplets(
        rows, seed=42, match_manufacturer=True, max_triplets_per_bin=None
    )
    assert summary["triplets"] == 8
    assert summary["records"] == 24
    assert summary["role_counts"] == {
        "anchor": 8,
        "same_state_swap": 8,
        "opposite_state_swap": 8,
    }
    images = {row["image_id"] for row in triplets}
    assert len(images) == len(triplets)
    for triplet_id in {row["triplet_id"] for row in triplets}:
        members = [row for row in triplets if row["triplet_id"] == triplet_id]
        anchor = next(row for row in members if row["swap_role"] == "anchor")
        same = next(row for row in members if row["swap_role"] == "same_state_swap")
        opposite = next(
            row for row in members if row["swap_role"] == "opposite_state_swap"
        )
        assert anchor["positive_votes"] == same["positive_votes"]
        assert anchor["positive_votes"] + opposite["positive_votes"] == 3


def test_tristate_boundary_is_symmetric():
    assert bounded_state(2.0, 0.5) == "supported"
    assert bounded_state(-2.0, 0.5) == "refuted"
    assert bounded_state(0.0, 0.5) == "undetermined"


def test_claim_simplex_has_independent_polarity_and_commitment_coordinates():
    logits = simplex_logits(polarity=1.25, commitment=-0.4)
    coordinates = epistemic_coordinates(logits)
    assert abs(coordinates["polarity"] - 1.25) < 1e-12
    assert abs(coordinates["commitment"] + 0.4) < 1e-12
    shifted = {key: value + 17.0 for key, value in logits.items()}
    shifted_coordinates = epistemic_coordinates(shifted)
    assert abs(shifted_coordinates["polarity"] - coordinates["polarity"]) < 1e-12
    assert abs(shifted_coordinates["commitment"] - coordinates["commitment"]) < 1e-12


def test_legacy_signed_evidence_collapses_claim_plane_to_v_curve():
    tau = 0.6
    for evidence in (-2.0, -0.2, 0.0, 0.7, 3.0):
        coordinates = epistemic_coordinates(tristate_logits(evidence, tau))
        assert abs(coordinates["polarity"] - evidence) < 1e-12
        assert abs(coordinates["commitment"] - (abs(evidence) - tau)) < 1e-12


def test_same_polarity_can_mean_ignorance_or_supported_evidence():
    assert epistemic_state(1.0, -0.2, 0.5, 0.0) == "ignorance"
    assert epistemic_state(1.0, 0.8, 0.5, 0.0) == "supported"
    assert epistemic_state(0.1, 0.8, 0.5, 0.0) == "conflict"


def test_activation_subtraction_is_directional_and_norm_matched():
    hidden = torch.tensor([3.0, 4.0, 0.0])
    direction = torch.tensor([1.0, 0.0, 0.0])
    changed, audit = norm_matched_direction_subtraction(hidden, direction, 0.1)
    assert abs(float(changed.norm()) - float(hidden.norm())) < 1e-6
    assert audit["direction_projection_after"] < audit["direction_projection_before"]
    assert abs(audit["relative_step_l2"] - 0.1) < 1e-6


def test_random_control_is_deterministic_and_orthogonal():
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    first = deterministic_orthogonal_direction(target, "case-1", 42)
    second = deterministic_orthogonal_direction(target, "case-1", 42)
    assert torch.equal(first, second)
    assert abs(float(torch.dot(first, target / target.norm()))) < 1e-6
    assert abs(float(first.norm()) - 1.0) < 1e-6


def test_claim_plane_direction_preserves_polarity_to_first_order():
    commitment_gradient = torch.tensor([1.0, 2.0, 1.0])
    polarity_gradient = torch.tensor([1.0, 0.0, 0.0])
    direction, audit = orthogonalized_unit_direction(
        commitment_gradient, polarity_gradient
    )
    assert abs(float(torch.dot(direction, polarity_gradient))) < 1e-6
    assert abs(float(direction.norm()) - 1.0) < 1e-6
    assert audit["fraction_target_gradient_retained"] < 1.0
    assert abs(audit["target_preserve_cosine_after"]) < 1e-6


def test_claim_plane_gradient_restores_autograd_inside_inference_mode():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = torch.nn.Embedding(3, 4)
            self.model = type("Decoder", (), {"norm": torch.nn.Identity()})()
            with torch.no_grad():
                self.output.weight.copy_(
                    torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                        ]
                    )
                )

        def get_output_embeddings(self):
            return self.output

    bot = type("Bot", (), {"model": DummyModel()})()
    null_hidden = tuple(torch.randn(1, 2, 4) for _ in range(3))
    with torch.inference_mode():
        direction, polarity, audit = null_commitment_direction(
            bot,
            null_hidden,
            layer=1,
            ids={"supported": 0, "refuted": 1, "undetermined": 2},
        )
    assert abs(float(torch.dot(direction, polarity))) < 1e-6
    assert audit["orthogonal_component_l2"] > 0


def test_intervention_audit_separates_commitment_effect_from_polarity_leakage():
    row = {
        "measurement": {
            "final_layer": 3,
            "trajectory": {
                "3": {"real_logits": simplex_logits(1.0, 2.0)}
            },
            "activation_intervention": {
                "targeted": {"logits": simplex_logits(1.0, 1.0)},
                "random_orthogonal": {"logits": simplex_logits(1.2, 2.1)},
            },
        }
    }
    changes = intervention_coordinate_changes(row)
    assert abs(changes["targeted_polarity_change"]) < 1e-12
    assert changes["targeted_commitment_change"] == -1.0
    assert abs(changes["random_polarity_change"] - 0.2) < 1e-12
    assert abs(changes["random_commitment_change"] - 0.1) < 1e-12
    assert changes["targeted_polarity_sign_flip"] == 0.0


def test_random_control_can_preserve_target_and_polarity_subspace():
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    polarity = torch.tensor([0.0, 1.0, 0.0, 0.0])
    random = deterministic_orthogonal_direction(
        target, "two-constraints", 42, additional_directions=(polarity,)
    )
    assert abs(float(torch.dot(random, target))) < 1e-6
    assert abs(float(torch.dot(random, polarity))) < 1e-6


def test_cluster_bootstrap_detects_new_information_beyond_final_scalar():
    labels = [0, 0, 1, 1, 0, 1]
    baseline = [0.0] * len(labels)
    candidate = [-4.0, -3.0, 4.0, 3.0, -2.0, 2.0]
    result = paired_cluster_bootstrap_increment(
        labels,
        baseline,
        candidate,
        [f"image-{index}" for index in range(len(labels))],
        draws=500,
        seed=42,
    )
    assert result["auroc_gain"]["estimate"] == 0.5
    assert result["auroc_gain"]["ci_low"] > 0
    assert result["brier_gain"]["estimate"] > 0
    assert result["brier_gain"]["ci_low"] > 0


def test_reader_gate_layer_selection_never_uses_test(
    tmp_path: Path, monkeypatch
):
    manifest_path = tmp_path / "manifest.jsonl"
    adjusted_path = tmp_path / "reader_adjusted.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "gate.json"
    manifest_rows = []
    raw_rows = []
    adjusted_rows = []
    for split in ("dev", "test"):
        for repeat in range(2):
            for votes in range(4):
                image_id = f"{split}-{repeat}-{votes}"
                manifest_rows.append(
                    {
                        "finding": "effusion",
                        "image_id": image_id,
                        "positive_votes": votes,
                        "reader_count": 3,
                        "experiment_split": split,
                        "formal_reference": True,
                        "reference_source": "vindr_reader_votes",
                        "reader_ids": ["r0", "r1", "r2"],
                        "reader_votes": [
                            {"rad_id": "r0", "vote": int(votes > 0)},
                            {"rad_id": "r1", "vote": int(votes > 1)},
                            {"rad_id": "r2", "vote": int(votes > 2)},
                        ],
                    }
                )
                clear = votes in {0, 3}
                adjusted_rows.append(
                    {
                        **manifest_rows[-1],
                        "reader_adjusted_clarity": float(clear),
                        "reader_adjusted_reference_role": "sensitivity_only",
                    }
                )
                polarity = -2.0 if votes < 1.5 else 2.0
                # Dev selects layer 1.  Test alone would select layer 2, which
                # catches accidental test-set layer selection.
                layer_one_commitment = (3.0 if clear else -3.0) if split == "dev" else 0.0
                layer_two_commitment = (-3.0 if clear else 3.0) if split == "dev" else (3.0 if clear else -3.0)
                trajectory = {}
                for layer, commitment in (
                    (1, layer_one_commitment),
                    (2, layer_two_commitment),
                    (3, 0.0),
                ):
                    trajectory[str(layer)] = {
                        "real_logits": simplex_logits(polarity, commitment),
                        "baseline_state": "supported" if polarity > 0 else "refuted",
                    }
                raw_rows.append(
                    {
                        "finding": "effusion",
                        "image_id": image_id,
                        "status": "ok",
                        "measurement": {"trajectory": trajectory},
                    }
                )
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    adjusted_path.write_text(
        "".join(json.dumps(row) + "\n" for row in adjusted_rows), encoding="utf-8"
    )
    raw_path.write_text(
        "".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "fit_reader_agreement_gate.py",
            "--manifest",
            str(manifest_path),
            "--reader-adjusted-manifest",
            str(adjusted_path),
            "--raw",
            str(raw_path),
            "--output",
            str(output_path),
            "--steps",
            "100",
            "--bootstrap-draws",
            "100",
            "--min-test-per-class",
            "2",
        ],
    )
    reader_agreement_main()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["selected_early_layer"] == 1
    assert result["formal_reference"] is True
    assert result["reader_effect_control"]["reader_identity_preserved"] is True
    assert result["reader_effect_control"]["reader_vocab_fit_on_dev"] == [
        "r0",
        "r1",
        "r2",
    ]
    assert result["reader_effect_control"]["unseen_test_readers"] == []
    assert result["reader_effect_control"]["unseen_test_findings"] == []
    assert result["reader_adjusted_sensitivity"]["provided"] is True
    assert (
        result["reader_adjusted_sensitivity"]
        ["selected_layer_inherited_from_raw_dev_gate"]
        == 1
    )
    assert result["bootstrap_unit"] == "image_id"
    assert result["provenance"]["test_used_for_selection_or_fitting"] is False
    assert result["per_finding_heldout_tests"]["effusion"]["qualified"] is True
    assert result["per_finding_heldout_tests"]["effusion"]["test_vote_bin_counts"] == {
        "0/3": 2,
        "1/3": 2,
        "2/3": 2,
        "3/3": 2,
    }
    assert result["finding_majority_gate"]["qualified_findings"] == ["effusion"]
    calibrator = result["selected_clarity_calibrator"]
    assert calibrator["fit_split"] == "dev"
    assert calibrator["normalization"]["fit_split"] == "dev"
    assert len(calibrator["normalization"]["feature_mean"]) == 2
    assert len(result["locked_test_predictions"]) == 8


def test_matched_coverage_prevents_short_answer_hallucination_win():
    supports = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0}
    rows = []
    baseline_scores = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    for claim_id, support in supports.items():
        rows.append(
            {
                "method": "baseline",
                "claim_id": claim_id,
                "reader_support": support,
                "emitted": claim_id in {"a", "b", "c"},
                "prediction_state": "supported" if claim_id in {"a", "b", "c"} else "undetermined",
                "assertion_score": baseline_scores[claim_id],
            }
        )
        rows.append(
            {
                "method": "terse",
                "claim_id": claim_id,
                "reader_support": support,
                "emitted": claim_id == "a",
                "prediction_state": "supported" if claim_id == "a" else "undetermined",
                "assertion_score": 5.0 if claim_id == "a" else -1.0,
            }
        )
    result = evaluate_oe_methods_matched_coverage(rows, "baseline")
    assert result["matched_claim_count"] == 1
    assert result["matched"]["baseline"]["positive_claim_hallucination_rate"] == 0.0
    assert result["matched"]["terse"]["positive_claim_hallucination_rate"] == 0.0
    assert not result["coverage_guard_pass"]["terse"]
    assert result["natural"]["terse"]["unanimous_positive_omission_rate"] == 0.5


def test_report_omission_counts_only_task_required_supported_claims():
    rows = [
        {
            "claim_id": "study-1:required_finding",
            "reader_support": 1.0,
            "reference_relevance": "required",
            "emitted": True,
            "prediction_state": "supported",
            "assertion_score": 3.0,
        },
        {
            "claim_id": "study-1:optional_finding",
            "reader_support": 1.0,
            "reference_relevance": "optional",
            "emitted": False,
            "prediction_state": "undetermined",
            "assertion_score": 2.0,
        },
    ]
    result = evaluate_oe_claim_rows(rows)
    assert result["unanimous_positive_omission_rate"] == 0.5
    assert result["required_unanimous_positive_omission_rate"] == 0.0
    assert result["required_unanimous_positive_recall"] == 1.0
    assert result["optional_unanimous_positive_mention_rate"] == 0.0


def test_empty_and_uniform_negative_outputs_fail_open_ended_guards():
    rows = []
    for method in ("baseline", "empty", "negative"):
        for claim_id, support in (("a", 1.0), ("b", 0.0)):
            emitted = method == "baseline" and claim_id == "a"
            state = "supported" if emitted else "undetermined"
            if method == "negative":
                emitted, state = True, "refuted"
            rows.append(
                {
                    "method": method,
                    "claim_id": claim_id,
                    "reader_support": support,
                    "emitted": emitted,
                    "prediction_state": state,
                    "assertion_score": 1.0 if emitted else -1.0,
                }
            )
    result = evaluate_oe_methods_matched_coverage(rows, "baseline")
    assert not result["matched_valid"]
    assert result["anti_cheat"]["empty"]["empty_or_refusal"]
    assert result["anti_cheat"]["negative"]["uniform_negative"]
    assert not result["omission_nonincrease_pass"]["empty"]


def test_uniform_uncertainty_cannot_masquerade_as_oe_mitigation():
    rows = []
    for method in ("baseline", "all_uncertain"):
        for claim_id, votes in (("a", 3), ("b", 2), ("c", 1), ("d", 0)):
            uncertain = method == "all_uncertain"
            state = "undetermined" if uncertain or votes == 2 else (
                "supported" if votes == 3 else "refuted"
            )
            rows.append(
                {
                    "method": method,
                    "claim_id": claim_id,
                    "reader_support": votes / 3,
                    "emitted": True,
                    "prediction_state": state,
                    "prediction_polarity": "present" if votes >= 2 else "absent",
                    "prediction_uncertainty": "uncertain" if state == "undetermined" else "definite",
                    "assertion_score": 0.0 if uncertain else (3.0 if votes >= 2 else -3.0),
                }
            )
    result = evaluate_oe_methods_matched_coverage(rows, "baseline")
    uncertain = result["natural"]["all_uncertain"]
    assert result["anti_cheat"]["all_uncertain"]["uniform_uncertain"]
    assert uncertain["disagreement_overcommitment_rate"] == 0.0
    assert uncertain["clear_case_accuracy"] == 0.0
    assert not result["clear_case_non_degradation_pass"]["all_uncertain"]
    assert not result["tristate_non_degradation_pass"]["all_uncertain"]


def test_hedged_fabrication_remains_a_positive_content_error():
    rows = [
        {
            "claim_id": "image-1:effusion",
            "reader_support": 0.0,
            "emitted": True,
            "prediction_state": "undetermined",
            "prediction_polarity": "present",
            "prediction_uncertainty": "uncertain",
            "assertion_score": 0.0,
        }
    ]
    result = evaluate_oe_claim_rows(rows)
    assert result["positive_assertion_count"] == 1
    assert result["hedged_positive_claim_count"] == 1
    assert result["fabricated_claim_rate"] == 1.0
    assert result["fabricated_hedged_positive_count"] == 1
    assert result["legacy_third_state_positive_count"] == 0
    assert result["legacy_third_state_fabricated_claim_rate"] is None


def test_commitment_only_rewrite_cannot_claim_content_hallucination_reduction():
    reference = {
        "effusion": {
            "reference_source": "vindr_reader_votes",
            "reference_contract_version": "missing-third-state-claims-v8",
            "reference_observability": "image_grounded",
            "reference_relevance": "required",
            "positive_votes": 0,
            "reader_count": 3,
        }
    }
    draft = [ClinicalClaim("effusion")]
    hedged, _ = polarity_preserving_commitment_claims(
        draft, {"effusion": 0.0}, clear_threshold=0.5
    )
    rows = []
    for method, claims in (("baseline", draft), ("hedged", hedged)):
        method_rows, _ = claims_to_fixed_oe_rows(
            "image-1",
            method,
            claims,
            {"effusion": 0.0},
            {"effusion": 1.0},
            reference,
        )
        rows.extend(method_rows)
    result = evaluate_oe_methods_matched_coverage(rows, "baseline")
    assert result["matched_claim_count"] == 1
    assert result["natural"]["baseline"]["fabricated_claim_rate"] == 1.0
    assert result["natural"]["hedged"]["fabricated_claim_rate"] == 1.0


def test_radgraph_claim_parser_preserves_finding_location_attribute_and_uncertainty():
    annotation = {
        "text": "There may be a small left pleural effusion.",
        "entities": {
            "1": {
                "tokens": "small",
                "label": "Observation::uncertain",
                "start_ix": 4,
                "end_ix": 4,
                "relations": [["modify", "3"]],
            },
            "2": {
                "tokens": "left",
                "label": "Anatomy::definitely present",
                "start_ix": 5,
                "end_ix": 5,
                "relations": [],
            },
            "3": {
                "tokens": "pleural effusion",
                "label": "Observation::uncertain",
                "start_ix": 6,
                "end_ix": 7,
                "relations": [["located_at", "2"]],
            },
        },
    }
    claims, audit = claims_from_radgraph(
        annotation, {"pleural_effusion": ["pleural effusion"]}
    )
    assert [claim.to_dict() for claim in claims] == [
        {
            "finding": "pleural_effusion",
            "polarity": "present",
            "uncertainty": "uncertain",
            "anatomy": "left",
            "attributes": ["small"],
            "provenance": "image_grounded",
            "state": "undetermined",
        }
    ]
    assert audit["unmatched_observations"] == []
    assert audit["orphan_anatomy_entities"] == []


def test_radgraph_claim_parser_does_not_guess_unknown_or_etiologic_claims():
    annotation = {
        "text": "Opacity suggestive of pneumonia and a mystery sign.",
        "entities": {
            "1": {
                "tokens": "opacity",
                "label": "Observation::definitely present",
                "start_ix": 0,
                "end_ix": 0,
                "relations": [["suggestive_of", "2"]],
            },
            "2": {
                "tokens": "pneumonia",
                "label": "Observation::definitely present",
                "start_ix": 3,
                "end_ix": 3,
                "relations": [],
            },
            "3": {
                "tokens": "mystery sign",
                "label": "Observation::definitely present",
                "start_ix": 6,
                "end_ix": 7,
                "relations": [],
            },
        },
    }
    claims, audit = claims_from_radgraph(
        annotation,
        {"lung_opacity": ["opacity"], "pneumonia": ["pneumonia"]},
    )
    assert [claim.finding for claim in claims] == ["lung_opacity", "pneumonia"]
    assert claims[0].state == "supported"
    assert claims[1].provenance == "knowledge"
    assert claims[1].state == "unobservable"
    assert audit["unmatched_observations"] == [
        {"root_entity_id": "3", "phrase": "mystery sign", "reason": "no_ontology_match"}
    ]


def test_radgraph_claim_parser_exposes_equal_length_ontology_ambiguity():
    annotation = {
        "text": "Mass.",
        "entities": {
            "1": {
                "tokens": "mass",
                "label": "Observation::definitely present",
                "start_ix": 0,
                "end_ix": 0,
                "relations": [],
            }
        },
    }
    claims, audit = claims_from_radgraph(
        annotation, {"lung_mass": ["mass"], "mediastinal_mass": ["mass"]}
    )
    assert claims == []
    assert audit["unmatched_observations"][0]["reason"] == "ambiguous_ontology_match"


def test_formal_oe_reference_requires_raw_reader_votes_and_frozen_contract():
    rows = [
        {
            "method": method,
            "claim_id": "image-1:effusion",
            "reader_support": 2 / 3,
            "positive_votes": 2,
            "reader_count": 3,
            "reference_source": "vindr_reader_votes",
            "reference_contract_version": "missing-third-state-claims-v8",
            "reference_observability": "image_grounded",
            "reference_relevance": "required",
            "reader_ids": ["r0", "r1", "r2"],
            "reader_votes": [
                {"rad_id": "r0", "vote": 1},
                {"rad_id": "r1", "vote": 1},
                {"rad_id": "r2", "vote": 0},
            ],
        }
        for method in ("baseline", "cbd")
    ]
    audit = validate_oe_reference_provenance(rows)
    assert audit["status"] == "valid"
    assert audit["automatic_truth_allowed"] is False


def test_formal_oe_reference_rejects_aggregate_only_vindr_votes():
    rows = [
        {
            "claim_id": "image-1:effusion",
            "reader_support": 2 / 3,
            "positive_votes": 2,
            "reader_count": 3,
            "reference_source": "vindr_reader_votes",
            "reference_contract_version": "missing-third-state-claims-v8",
            "reference_observability": "image_grounded",
            "reference_relevance": "required",
        }
    ]
    try:
        validate_oe_reference_provenance(rows)
    except ValueError as error:
        assert "requires three reader-level votes" in str(error)
    else:
        raise AssertionError("aggregate-only VinDr reference was accepted")


def test_formal_oe_reference_rejects_llm_judge_truth():
    rows = [
        {
            "claim_id": "image-1:effusion",
            "reader_support": 1.0,
            "reference_source": "llm_judge",
            "reference_contract_version": "some-prompt",
            "reference_observability": "image_grounded",
            "reference_relevance": "required",
        }
    ]
    try:
        validate_oe_reference_provenance(rows)
    except ValueError as error:
        assert "cannot define truth" in str(error)
    else:
        raise AssertionError("LLM judge reference was incorrectly accepted")


def test_formal_oe_reference_requires_explicit_task_relevance():
    rows = [
        {
            "claim_id": "image-1:effusion",
            "reader_support": 1.0,
            "reference_source": "vindr_reader_votes",
            "reference_contract_version": "missing-third-state-claims-v8",
            "reference_observability": "image_grounded",
            "positive_votes": 3,
            "reader_count": 3,
        }
    ]
    try:
        validate_oe_reference_provenance(rows)
    except ValueError as error:
        assert "missing reference_relevance" in str(error)
    else:
        raise AssertionError("formal OE reference accepted implicit relevance")


def test_claim_projection_keeps_fixed_universe_and_exposes_unmapped_and_conflicts():
    claims = [
        ClinicalClaim("effusion"),
        ClinicalClaim("effusion", polarity="absent"),
        ClinicalClaim("treatment recommendation", provenance="knowledge"),
    ]
    reference = {
        finding: {
            "reference_source": "vindr_reader_votes",
            "reference_contract_version": "missing-third-state-claims-v8",
            "reference_observability": "image_grounded",
            "reference_relevance": "required",
            "positive_votes": votes,
            "reader_count": 3,
        }
        for finding, votes in {"effusion": 2, "pneumothorax": 0}.items()
    }
    rows, audit = claims_to_fixed_oe_rows(
        "image-1",
        "baseline",
        claims,
        {"effusion": 2 / 3, "pneumothorax": 0.0},
        {"effusion": 2.0, "pneumothorax": -1.0},
        reference,
    )
    assert len(rows) == 2
    assert rows[0]["emitted"] and rows[0]["prediction_state"] == "undetermined"
    assert rows[0]["prediction_polarity"] == "conflict"
    assert not rows[1]["emitted"]
    assert audit["duplicate_state_conflicts"][0]["finding"] == "effusion"
    assert not audit["adjudication_complete"]
    assert audit["out_of_ontology_claims"][0]["finding"] == "treatment_recommendation"


def test_global_null_requires_dev_sidecar_and_rejects_plumbing_by_default(tmp_path: Path):
    vector = tmp_path / "global_null.npy"
    np.save(vector, np.array([1.0, 2.0], dtype=np.float32), allow_pickle=False)
    vector.with_suffix(".json").write_text(
        json.dumps(
            {
                "split_requirement": "dev only",
                "vector_sha256": sha256_file(vector),
                "plumbing_only": True,
            }
        )
    )
    try:
        validate_global_null_sidecar(vector, allow_plumbing=False)
    except ValueError as error:
        assert "inadmissible" in str(error)
    else:
        raise AssertionError("plumbing-only null was incorrectly admitted")
    metadata = validate_global_null_sidecar(vector, allow_plumbing=True)
    assert metadata["split_requirement"] == "dev only"


def test_reader_adjustment_distinguishes_panel_bias_from_image_support():
    liberal_votes = (("l0", 1), ("l1", 1), ("l2", 1))
    conservative_votes = (("c0", 1), ("c1", 1), ("c2", 1))
    biases = {"l0": 2.0, "l1": 2.0, "l2": 2.0, "c0": -2.0, "c1": -2.0, "c2": -2.0}
    liberal_residual, _ = infer_item_map(liberal_votes, 0.0, biases, 1.0)
    conservative_residual, _ = infer_item_map(conservative_votes, 0.0, biases, 1.0)
    assert conservative_residual > liberal_residual


def test_continuous_clarity_bootstrap_detects_incremental_signal():
    result = paired_cluster_bootstrap_continuous(
        targets=[0.0, 0.0, 1.0, 1.0] * 8,
        baseline_probabilities=[0.5] * 32,
        candidate_probabilities=[0.1, 0.1, 0.9, 0.9] * 8,
        clusters=[f"image-{index}" for index in range(32)],
        draws=500,
        seed=42,
    )
    assert result["brier_gain"]["estimate"] > 0
    assert result["brier_gain"]["ci_low"] > 0
    assert result["mae_gain"]["ci_low"] > 0


def test_reader_effects_are_fit_on_dev_and_frozen_for_test():
    rng = np.random.default_rng(17)
    readers = [f"r{index}" for index in range(6)]
    planted = np.asarray([-1.2, -0.8, -0.4, 0.4, 0.8, 1.2])
    rows = []
    for index in range(180):
        split = "dev" if index < 150 else "test"
        latent = (-1.5, 0.0, 1.5)[index % 3]
        selected = sorted({readers[(index + offset * 2) % 6] for offset in range(3)})
        reader_votes = []
        for reader in selected:
            probability = 1.0 / (
                1.0 + np.exp(-(latent + planted[int(reader[1:])]))
            )
            reader_votes.append(
                {"rad_id": reader, "vote": int(rng.random() < probability)}
            )
        rows.append(
            {
                "image_id": f"image-{index}",
                "finding": "effusion",
                "positive_votes": sum(item["vote"] for item in reader_votes),
                "reader_count": 3,
                "reader_ids": [item["rad_id"] for item in reader_votes],
                "reader_votes": reader_votes,
                "experiment_split": split,
            }
        )
    model = fit_reader_effects(rows, steps=700, seed=4)
    estimated = np.asarray([model["reader_bias"][reader] for reader in readers])
    assert np.corrcoef(planted, estimated)[0, 1] > 0.8
    adjusted = adjust_rows(rows, model)
    assert all(
        row["reader_adjusted_inference"] == "joint_dev_fit"
        for row in adjusted[:150]
    )
    assert all(
        row["reader_adjusted_inference"]
        == "test_vote_map_with_frozen_dev_effects"
        for row in adjusted[150:]
    )
    assert all(row["reader_adjusted_reference_role"] == "sensitivity_only" for row in adjusted)
