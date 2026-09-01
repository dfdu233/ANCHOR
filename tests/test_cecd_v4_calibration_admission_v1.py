import copy
import json
from pathlib import Path

import numpy as np
import pytest

from corrected_sgta.cecd_v4_calibration_admission_v1 import (
    ContractError,
    NON_AUTHORIZING,
    assess_b0_regularity,
    assess_confirmation,
    fit_dev_admission,
    strict_calibrated_probabilities,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cecd_v4_calibration_admission_contract_20260803.json"
RENDERS = ("canonical", "render_b", "render_c")
PROMPTS = ("prompt_a", "prompt_b", "prompt_c")


def _config(*, minimum=4, draws=59):
    value = json.loads(CONFIG_PATH.read_text())
    value["directional_admission"]["minimum_per_vote_bin"] = minimum
    value["directional_admission"]["bootstrap_draws"] = draws
    value["calibration"]["folds"] = 2
    value["denominator_regularity"]["bootstrap_draws"] = draws
    return value


def _logit_fields(score):
    logits = {"supported": float(score) / 2, "refuted": -float(score) / 2, "undetermined": 0.0}
    raw = np.asarray(list(logits.values()))
    probability = np.exp(raw - raw.max())
    probability /= probability.sum()
    return {
        "signed_score": float(score),
        "commitment_score": abs(float(score)) / 2,
        "tristate_logits": logits,
        "tristate_entropy": float(-np.sum(probability * np.log(probability))),
    }


def _record(image, vote, render, prompt, score, *, patient=None):
    row = {
        "model": "ModelA",
        "image_id": image,
        "patient_id": patient or f"patient-{image}",
        "finding": "effusion",
        "reader_votes": vote,
        "render_id": render,
        "prompt_id": prompt,
        "acquisition_view": "PA",
        "input_prompt_length_tokens": 8,
        "answer_length_tokens": 1,
        **_logit_fields(score),
    }
    return row


def _payload(split, *, reversal=False, extreme_confirmation=False, replicates=6):
    rng = np.random.default_rng(104 if split == "dev_fit" else 205)
    records = []
    for vote in range(4):
        centers = (-2.4, -0.8, 0.8, 2.4) if split == "dev_fit" else (-2.2, -0.7, 0.7, 2.2)
        center = centers[vote]
        if reversal and vote == 2:
            center = -1.1
        for replicate in range(replicates):
            image = f"{split}-v{vote}-r{replicate}"
            base = center + rng.normal(0.0, 0.08 if split == "dev_fit" else 0.015)
            if extreme_confirmation:
                base += 20.0
            row_effect = np.asarray([0.0, 0.08, -0.05])
            column_effect = np.asarray([0.0, -0.07, 0.04])
            interaction = np.outer([1.0, -0.5, -0.5], [1.0, -0.5, -0.5]) * 0.01
            score = base + row_effect[:, None] + column_effect[None, :] + interaction
            for r, render in enumerate(RENDERS):
                for p, prompt in enumerate(PROMPTS):
                    records.append(_record(image, vote, render, prompt, score[r, p]))
            for p, prompt in enumerate(PROMPTS):
                records.append(_record(image, vote, "identity", prompt, score[0, p]))
            records.append(_record(image, vote, "canonical", "prompt_duplicate", score[0, 0]))
    return {
        "schema_version": "clinical-equivalence-factorial-v1",
        "split": split,
        "source_manifest_split": "dev" if split == "dev_fit" else "confirmation",
        "frozen_before_outputs": True,
        "score_definition": "fp32_yes_minus_no_logit",
        "primary_renders": list(RENDERS),
        "primary_prompts": list(PROMPTS),
        "baseline_render": "canonical",
        "baseline_prompt": "prompt_a",
        "identity_render": "identity",
        "duplicate_prompt": "prompt_duplicate",
        "records": records,
    }


def test_raw_four_bin_direction_and_swap_gate_precede_isotonic_shape():
    config = _config()
    good = fit_dev_admission(_payload("dev_fit"), config)
    diagnostic = good["directional_admission"]["ModelA"]["effusion"]
    means = diagnostic["raw_directionality"]["raw_bin_means"]
    assert [means[str(value)] for value in range(4)] == sorted(means.values())
    assert diagnostic["raw_directionality"]["strict_raw_bin_order"] is True
    assert diagnostic["same_support_image_swap"]["n_pairs"] == 12
    assert diagnostic["gates"]["passed"] is True
    assert good["calibrators"]["ModelA"]["effusion"]["raw_directional_admission_passed"] is True

    bad = fit_dev_admission(_payload("dev_fit", reversal=True), config)
    rejected = bad["directional_admission"]["ModelA"]["effusion"]
    assert rejected["raw_directionality"]["strict_raw_bin_order"] is False
    assert rejected["gates"]["passed"] is False
    # An increasing isotonic object may still exist diagnostically, but its
    # forced monotonicity cannot make raw admission pass.
    assert bad["calibrators"]["ModelA"]["effusion"]["raw_directional_admission_passed"] is False


def test_all_scientific_thresholds_are_hash_bound_config_and_tamper_fails():
    config = _config()
    bundle = fit_dev_admission(_payload("dev_fit"), config)
    changed = copy.deepcopy(config)
    changed["directional_admission"]["minimum_spearman_rank_correlation"] = 0.9
    with pytest.raises(ContractError, match="confirmation config differs"):
        assess_confirmation(_payload("confirmation_locked"), bundle, changed)

    malformed = copy.deepcopy(config)
    malformed["reader_evidence"]["named_reader_loro_computed"] = True
    with pytest.raises(ContractError, match="cannot claim named-reader LORO"):
        validate_config(malformed)


def test_confirmation_is_apply_only_and_never_silently_clips():
    config = _config()
    bundle = fit_dev_admission(_payload("dev_fit"), config)
    report = assess_confirmation(_payload("confirmation_locked"), bundle, config)
    assert report["status"] == NON_AUTHORIZING
    assert report["authorized"] is False
    assert report["apply_only_no_refit"] is True
    assert report["all_required_score_families_inside_frozen_dev_support"] is True
    assert all(
        family["clipped_values"] == 0 and family["extrapolated_values"] == 0
        for family in report["calibration_support"]["ModelA"]["effusion"].values()
    )
    assert report["b0_multiplier_pairing"]["same_stream_key_for_every_model"] is True

    outside = assess_confirmation(
        _payload("confirmation_locked", extreme_confirmation=True), bundle, config
    )
    assert outside["all_required_score_families_inside_frozen_dev_support"] is False
    actual = outside["calibration_support"]["ModelA"]["effusion"]["actual"]
    assert actual["outside_fraction"] == 1.0
    assert actual["clipped_values"] == 0
    assert outside["b0_regularity"]["ModelA"]["status"] == "not_computed_due_to_calibration_support_failure"
    calibrator = bundle["calibrators"]["ModelA"]["effusion"]
    with pytest.raises(ContractError, match="clipping is forbidden"):
        strict_calibrated_probabilities(calibrator, [float(calibrator["support_max"]) + 1.0])


def test_missing_required_confirmation_family_fails_closed():
    config = _config()
    config["calibration"]["required_confirmation_score_families"].append("haar")
    bundle = fit_dev_admission(_payload("dev_fit"), config)
    with pytest.raises(ContractError, match="missing required confirmation score families"):
        assess_confirmation(_payload("confirmation_locked"), bundle, config)


def test_b0_near_zero_and_unstable_denominators_fail_closed():
    config = _config(draws=199)
    strata = [f"f{index % 4}|{index % 4}" for index in range(40)]
    clusters = [f"patient-{index}" for index in range(40)]
    near_zero = assess_b0_regularity(
        np.full(40, 1e-10), strata, clusters, config, stream_key="near-zero"
    )
    assert near_zero["gates"]["ratio_authorizable_from_denominator_only"] is False
    assert near_zero["failure_policy"] == "ratio_non_authorizing_keep_absolute_theta_only"

    stable = assess_b0_regularity(
        np.full(40, 0.2), strata, clusters, config, stream_key="stable"
    )
    assert stable["gates"]["ratio_authorizable_from_denominator_only"] is True
    assert stable["bootstrap"]["coefficient_of_variation"] < 1e-12


def test_aggregate_votes_are_never_reported_as_named_reader_loro():
    config = _config()
    bundle = fit_dev_admission(_payload("dev_fit"), config)
    boundary = bundle["reader_inference_boundary"]
    assert boundary["input_used"] == "aggregate_vote_count_0_1_2_3_only"
    assert boundary["named_reader_loro_computed"] is False
    assert boundary["aggregate_vote_substitutes_for_named_reader_loro"] is False
    assert boundary["promotion_requires_separate_hash_bound_named_reader_loro"] is True

    report = assess_confirmation(_payload("confirmation_locked"), bundle, config)
    assert report["reader_inference_boundary"]["named_reader_loro_gate"] == "absent_fail_closed"
    assert report["promotion_effect"] == "none"


def test_dev_confirmation_cluster_overlap_fails_closed():
    config = _config()
    dev = _payload("dev_fit")
    bundle = fit_dev_admission(dev, config)
    confirmation = _payload("confirmation_locked")
    dev_image = dev["records"][0]["image_id"]
    old_image = confirmation["records"][0]["image_id"]
    for row in confirmation["records"]:
        if row["image_id"] == old_image:
            row["image_id"] = dev_image
            row["patient_id"] = f"patient-{dev_image}"
    with pytest.raises(ContractError, match="dev/confirmation cluster overlap"):
        assess_confirmation(confirmation, bundle, config)


def test_cluster_mode_drift_fails_before_patient_overlap_can_be_hidden():
    config = _config()
    bundle = fit_dev_admission(_payload("dev_fit"), config)
    confirmation = _payload("confirmation_locked")
    for row in confirmation["records"]:
        row["patient_id"] = ""
    with pytest.raises(ContractError, match="global cluster mode differs"):
        assess_confirmation(confirmation, bundle, config)
