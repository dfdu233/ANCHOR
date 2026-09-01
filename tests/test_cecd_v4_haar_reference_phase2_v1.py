import copy
import json
from pathlib import Path

import numpy as np
import pytest

from corrected_sgta.build_cecd_v4_haar_reference_phase2_v1 import (
    INFERENCE_BOUNDARY,
    MACRO_STATISTIC,
    Phase2ReferenceError,
    audit_reference_mcse,
    bind_mcse_evaluation_trace,
    build_reference_plan,
    spectral_haar_antithetic_pair,
    validate_phase2_config,
)
from corrected_sgta.validate_cecd_v4_promotion_phase1_v1 import (
    EXPECTED_CONTROLS,
    EXPECTED_FINDINGS,
    EXPECTED_MODELS,
    EXPECTED_PROMPTS,
    EXPECTED_RENDERS,
    build_phase1_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE1_CONFIG = json.loads(
    (ROOT / "configs/cecd_v4_promotion_phase1_contract_20260803.json").read_text()
)
PHASE2_CONFIG = json.loads(
    (ROOT / "configs/cecd_v4_haar_reference_phase2_contract_20260803.json").read_text()
)


def _score_fields(score):
    logits = {"supported": score / 2, "refuted": -score / 2, "undetermined": 0.0}
    values = np.asarray(list(logits.values()), dtype=float)
    probability = np.exp(values - values.max())
    probability /= probability.sum()
    return {
        "signed_score": float(score),
        "commitment_score": abs(float(score)) / 2,
        "tristate_logits": logits,
        "tristate_entropy": float(-np.sum(probability * np.log(probability))),
    }


def _row(model, image, patient, finding, vote, render, prompt, score):
    return {
        "model": model,
        "image_id": image,
        "patient_id": patient,
        "finding": finding,
        "reader_votes": vote,
        "render_id": render,
        "prompt_id": prompt,
        "acquisition_view": "PA",
        "input_prompt_length_tokens": 11,
        "answer_length_tokens": 1,
        **_score_fields(score),
    }


def _payload(stage, quota=1):
    prefix = "dev" if stage == "dev_fit" else "confirmation"
    records = []
    for vote in range(4):
        for replicate in range(quota):
            image = f"{prefix}-v{vote}-n{replicate}"
            patient = f"{prefix}-patient-v{vote}-n{replicate}"
            for model in EXPECTED_MODELS:
                for finding_index, finding in enumerate(EXPECTED_FINDINGS):
                    for render_index, render in enumerate(EXPECTED_RENDERS):
                        for prompt_index, prompt in enumerate(EXPECTED_PROMPTS):
                            score = vote - 1.5 + 0.03 * render_index - 0.02 * prompt_index
                            score += 0.01 * render_index * prompt_index + 0.001 * finding_index
                            records.append(
                                _row(model, image, patient, finding, vote, render, prompt, score)
                            )
                    for prompt_index, prompt in enumerate(EXPECTED_PROMPTS):
                        score = vote - 1.5 - 0.02 * prompt_index + 0.001 * finding_index
                        records.append(
                            _row(
                                model, image, patient, finding, vote,
                                EXPECTED_CONTROLS["identity_render"], prompt, score,
                            )
                        )
                    records.append(
                        _row(
                            model, image, patient, finding, vote,
                            EXPECTED_CONTROLS["baseline_render"],
                            EXPECTED_CONTROLS["duplicate_prompt"],
                            vote - 1.5 + 0.001 * finding_index,
                        )
                    )
    return {
        "schema_version": "clinical-equivalence-factorial-v1",
        "split": stage,
        "source_manifest_split": "dev" if stage == "dev_fit" else "confirmation",
        "frozen_before_outputs": True,
        "patient_provenance": {
            "schema_version": "verified-external-patient-mapping-v1",
            "verified_before_model_outputs": True,
            "source_manifest_sha256": ("c" if stage == "dev_fit" else "d") * 64,
        },
        "score_definition": "fp32_yes_minus_no_logit",
        "primary_renders": list(EXPECTED_RENDERS),
        "primary_prompts": list(EXPECTED_PROMPTS),
        **EXPECTED_CONTROLS,
        "records": records,
    }


@pytest.fixture()
def phase2_plan():
    phase1_config = copy.deepcopy(PHASE1_CONFIG)
    phase1_config["stage_contract"]["dev_fit"]["exact_orbits_per_model_finding_vote"] = 1
    phase1_config["stage_contract"]["confirmation_locked"]["exact_orbits_per_model_finding_vote"] = 1
    phase1_config["cluster_contract"]["minimum_unique_patient_clusters_per_model_finding_vote"] = {
        "dev_fit": 1, "confirmation_locked": 1,
    }
    phase1_config["bootstrap"].update(
        draws=5,
        minimum_ess_fraction=0.001,
        maximum_single_cluster_weight_fraction=1.0,
        maximum_single_cluster_orbit_contribution=1.0,
    )
    dev = _payload("dev_fit")
    confirmation = _payload("confirmation_locked")
    phase1 = build_phase1_contract(dev, confirmation, phase1_config)
    return phase1, confirmation, build_reference_plan(phase1, confirmation, PHASE2_CONFIG)


def _center(value):
    value = np.asarray(value, dtype=float)
    return value - value.mean(axis=0, keepdims=True) - value.mean(axis=1, keepdims=True) + value.mean()


def _audit(values, plan, b0, *, initial=None, model=EXPECTED_MODELS[0],
           calibrator_sha256="e" * 64):
    trace = bind_mcse_evaluation_trace(
        values, plan=plan, model=model, orbit_order=plan.shared_orbit_keys,
        macro_statistic=MACRO_STATISTIC, calibrator_sha256=calibrator_sha256,
        b0_point=0.2, b0_bootstrap=b0,
    )
    return audit_reference_mcse(
        values, plan=plan, model=model, orbit_order=plan.shared_orbit_keys,
        macro_statistic=MACRO_STATISTIC, calibrator_sha256=calibrator_sha256,
        evaluation_trace=trace, b0_point=0.2, b0_bootstrap=b0,
        config=PHASE2_CONFIG, initial_audit=initial,
    )


def test_frozen_contract_is_exact_4096_antithetic_and_never_a_randomization_test(phase2_plan):
    validate_phase2_config(PHASE2_CONFIG)
    _, _, plan = phase2_plan
    artifact = plan.artifact
    assert artifact["authorized"] is False
    assert artifact["authorizer_implemented"] is False
    assert artifact["real_results_inspected"] is False
    assert artifact["gpu_required"] is False
    assert artifact["schedule"]["initial_independent_draws"] == 2048
    assert artifact["schedule"]["initial_reference_count"] == 4096
    assert artifact["schedule"]["maximum_reference_count"] == 8192
    assert artifact["inference_boundary"].startswith("singular-spectrum-preserving")
    assert INFERENCE_BOUNDARY == "stress_reference_not_randomization_test"


def test_shared_keyed_pair_is_deterministic_centered_ispectral_and_antithetic(phase2_plan):
    _, _, plan = phase2_plan
    orbit = plan.shared_orbit_keys[0]
    interaction = _center(np.arange(15, dtype=float).reshape(5, 3) ** 2)
    positive, negative = spectral_haar_antithetic_pair(
        interaction, plan=plan, shared_orbit=orbit, independent_draw=17,
    )
    repeated = spectral_haar_antithetic_pair(
        interaction, plan=plan, shared_orbit=orbit, independent_draw=17,
    )
    other = spectral_haar_antithetic_pair(
        interaction, plan=plan, shared_orbit=orbit, independent_draw=18,
    )
    assert np.array_equal(positive, repeated[0])
    assert np.array_equal(negative, repeated[1])
    assert np.allclose(negative, -positive)
    assert np.allclose(positive.sum(axis=0), 0, atol=1e-9)
    assert np.allclose(positive.sum(axis=1), 0, atol=1e-9)
    assert np.allclose(
        np.linalg.svd(interaction, compute_uv=False),
        np.linalg.svd(positive, compute_uv=False),
        rtol=1e-9, atol=1e-12,
    )
    assert not np.allclose(positive, other[0])


def test_plan_fails_on_phase1_tamper_or_unpaired_confirmation(phase2_plan):
    phase1, confirmation, _ = phase2_plan
    phase1.artifact["authorized"] = True
    with pytest.raises(Phase2ReferenceError):
        build_reference_plan(phase1, confirmation, PHASE2_CONFIG)


def test_reference_seed_binding_is_outcome_blind_to_changed_scores(phase2_plan):
    phase1, confirmation, original = phase2_plan
    phase1_config = copy.deepcopy(PHASE1_CONFIG)
    phase1_config["stage_contract"]["dev_fit"]["exact_orbits_per_model_finding_vote"] = 1
    phase1_config["stage_contract"]["confirmation_locked"]["exact_orbits_per_model_finding_vote"] = 1
    phase1_config["cluster_contract"]["minimum_unique_patient_clusters_per_model_finding_vote"] = {
        "dev_fit": 1, "confirmation_locked": 1,
    }
    phase1_config["bootstrap"].update(
        draws=5, minimum_ess_fraction=0.001,
        maximum_single_cluster_weight_fraction=1.0,
        maximum_single_cluster_orbit_contribution=1.0,
    )
    changed_dev = _payload("dev_fit")
    changed_confirmation = copy.deepcopy(confirmation)
    for payload in (changed_dev, changed_confirmation):
        for row in payload["records"]:
            row.update(_score_fields(row["signed_score"] + 7.0))
    changed_phase1 = build_phase1_contract(
        changed_dev, changed_confirmation, phase1_config,
    )
    changed = build_reference_plan(changed_phase1, changed_confirmation, PHASE2_CONFIG)
    assert phase1.artifact["binding_sha256"] != changed_phase1.artifact["binding_sha256"]
    assert original.binding_sha256 == changed.binding_sha256
    assert original.artifact["seed_binding_excludes_scores_logits_losses_and_reader_votes"] is True


def test_reference_seed_is_invariant_to_vote_permutation_but_strata_remain_audited(phase2_plan):
    phase1, confirmation, original = phase2_plan
    config = copy.deepcopy(PHASE1_CONFIG)
    config["stage_contract"]["dev_fit"]["exact_orbits_per_model_finding_vote"] = 1
    config["stage_contract"]["confirmation_locked"]["exact_orbits_per_model_finding_vote"] = 1
    config["cluster_contract"]["minimum_unique_patient_clusters_per_model_finding_vote"] = {
        "dev_fit": 1, "confirmation_locked": 1,
    }
    config["bootstrap"].update(
        draws=5, minimum_ess_fraction=0.001,
        maximum_single_cluster_weight_fraction=1.0,
        maximum_single_cluster_orbit_contribution=1.0,
    )
    changed_dev = _payload("dev_fit")
    changed_confirmation = copy.deepcopy(confirmation)
    for payload in (changed_dev, changed_confirmation):
        for row in payload["records"]:
            row["reader_votes"] = 3 - row["reader_votes"]
    changed_phase1 = build_phase1_contract(changed_dev, changed_confirmation, config)
    changed = build_reference_plan(changed_phase1, changed_confirmation, PHASE2_CONFIG)
    assert phase1.artifact["binding_sha256"] != changed_phase1.artifact["binding_sha256"]
    assert original.binding_sha256 == changed.binding_sha256
    assert original.shared_orbit_keys == changed.shared_orbit_keys
    assert original.artifact["shared_orbits"][
        "reader_votes_excluded_from_seed_but_retained_in_pairing_audit"
    ] is True


def test_initial_mcse_uses_pairs_and_only_frozen_threshold_triggers_doubling(phase2_plan):
    _, _, plan = phase2_plan
    b0 = np.full(256, 0.2)
    x = np.linspace(-0.1, 0.1, 2048)
    exact_cancel = np.column_stack((x, -x))
    low = _audit(exact_cancel, plan, b0)
    assert low["reference_count"] == 4096
    assert low["independent_antithetic_pairs"] == 2048
    assert low["pair_mean_mcse"] < 1e-15
    assert low["decision"] == "precision_complete_at_4096"
    assert low["doubling_triggered"] is False

    high_values = np.column_stack((x, x))
    high = _audit(high_values, plan, b0)
    assert high["mcse_over_B0"] > 0.005
    assert high["decision"] == "double_once_to_8192"
    assert high["doubling_triggered"] is True


def test_8192_requires_triggered_unchanged_prefix_and_is_a_hard_cap(phase2_plan):
    _, _, plan = phase2_plan
    b0 = np.full(256, 0.2)
    x = np.linspace(-0.1, 0.1, 2048)
    initial_values = np.column_stack((x, x))
    initial = _audit(initial_values, plan, b0)
    extra = np.column_stack((x[::-1], x[::-1]))
    full = np.vstack((initial_values, extra))
    final = _audit(full, plan, b0, initial=initial)
    assert final["reference_count"] == 8192
    assert final["decision"] == "precision_complete_at_8192_cap_no_further_doubling"
    assert final["further_doubling_allowed"] is False

    with pytest.raises(Phase2ReferenceError, match="requires its frozen initial audit"):
        _audit(full, plan, b0)
    low_initial = _audit(np.column_stack((x, -x)), plan, b0)
    with pytest.raises(Phase2ReferenceError, match="not triggered"):
        _audit(
            np.vstack((np.column_stack((x, -x)), extra)), plan, b0,
            initial=low_initial,
        )
    changed = full.copy()
    changed[0, 0] += 1e-3
    with pytest.raises(Phase2ReferenceError, match="prefix differs"):
        _audit(changed, plan, b0, initial=initial)
    with pytest.raises(Phase2ReferenceError, match="exactly 4096 or 8192"):
        _audit(np.zeros((4097, 2)), plan, b0, initial=initial)


def test_near_zero_or_unstable_b0_fails_closed_before_precision_decision(phase2_plan):
    _, _, plan = phase2_plan
    values = np.zeros((2048, 2))
    with pytest.raises(Phase2ReferenceError, match="near-zero floor"):
        bind_mcse_evaluation_trace(
            values, plan=plan, model=EXPECTED_MODELS[0],
            orbit_order=plan.shared_orbit_keys, macro_statistic=MACRO_STATISTIC,
            calibrator_sha256="e" * 64, b0_point=1e-5,
            b0_bootstrap=np.full(256, 0.2),
        )
    unstable = np.full(256, 0.2)
    unstable[:8] = 1.01e-4
    with pytest.raises(Phase2ReferenceError, match="unstable relative"):
        bind_mcse_evaluation_trace(
            values, plan=plan, model=EXPECTED_MODELS[0],
            orbit_order=plan.shared_orbit_keys, macro_statistic=MACRO_STATISTIC,
            calibrator_sha256="e" * 64, b0_point=0.2, b0_bootstrap=unstable,
        )
    with pytest.raises(Phase2ReferenceError, match="incomplete"):
        bind_mcse_evaluation_trace(
            values, plan=plan, model=EXPECTED_MODELS[0],
            orbit_order=plan.shared_orbit_keys, macro_statistic=MACRO_STATISTIC,
            calibrator_sha256="e" * 64, b0_point=0.2,
            b0_bootstrap=np.full(99, 0.2),
        )


def test_mcse_rejects_arbitrary_array_and_context_reuse(phase2_plan):
    _, _, plan = phase2_plan
    b0 = np.full(256, 0.2)
    values = np.zeros((2048, 2))
    trace = bind_mcse_evaluation_trace(
        values, plan=plan, model=EXPECTED_MODELS[0],
        orbit_order=plan.shared_orbit_keys, macro_statistic=MACRO_STATISTIC,
        calibrator_sha256="e" * 64, b0_point=0.2, b0_bootstrap=b0,
    )
    arbitrary = values.copy()
    arbitrary[0, 0] = 0.25
    with pytest.raises(Phase2ReferenceError, match="arbitrary or modified"):
        audit_reference_mcse(
            arbitrary, plan=plan, model=EXPECTED_MODELS[0],
            orbit_order=plan.shared_orbit_keys, macro_statistic=MACRO_STATISTIC,
            calibrator_sha256="e" * 64, evaluation_trace=trace,
            b0_point=0.2, b0_bootstrap=b0, config=PHASE2_CONFIG,
        )
    with pytest.raises(Phase2ReferenceError, match="context binding drift"):
        audit_reference_mcse(
            values, plan=plan, model=EXPECTED_MODELS[1],
            orbit_order=plan.shared_orbit_keys, macro_statistic=MACRO_STATISTIC,
            calibrator_sha256="e" * 64, evaluation_trace=trace,
            b0_point=0.2, b0_bootstrap=b0, config=PHASE2_CONFIG,
        )
    with pytest.raises(Phase2ReferenceError, match="orbit order differs"):
        bind_mcse_evaluation_trace(
            values, plan=plan, model=EXPECTED_MODELS[0],
            orbit_order=tuple(reversed(plan.shared_orbit_keys)),
            macro_statistic=MACRO_STATISTIC, calibrator_sha256="e" * 64,
            b0_point=0.2, b0_bootstrap=b0,
        )
def test_contract_drift_and_uncentered_interaction_fail_closed(phase2_plan):
    drift = copy.deepcopy(PHASE2_CONFIG)
    drift["inference_boundary"] = "haar_randomization_test"
    with pytest.raises(Phase2ReferenceError, match="not a randomization test"):
        validate_phase2_config(drift)
    _, _, plan = phase2_plan
    with pytest.raises(Phase2ReferenceError, match="not in the frozen centered"):
        spectral_haar_antithetic_pair(
            np.arange(15, dtype=float).reshape(5, 3),
            plan=plan, shared_orbit=plan.shared_orbit_keys[0], independent_draw=0,
        )
