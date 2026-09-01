import copy

import numpy as np
import pytest

from corrected_sgta.analyze_cecd_product_attributable_risk_v4_prototype import (
    ContractError,
    NON_AUTHORIZING,
    apply_confirmation,
    cell_coordinate_interaction,
    fit_dev_calibration,
    object_sha256,
    spectral_haar_interaction,
)


RENDERS = ("canonical", "render_b", "render_c")
PROMPTS = ("prompt_a", "prompt_b", "prompt_c")


def _logit_fields(score):
    logits = {
        "supported": float(score) / 2,
        "refuted": -float(score) / 2,
        "undetermined": 0.0,
    }
    value = np.asarray(list(logits.values()))
    probability = np.exp(value - value.max())
    probability /= probability.sum()
    return {
        "signed_score": float(score),
        "commitment_score": abs(float(score)) / 2,
        "tristate_logits": logits,
        "tristate_entropy": float(-np.sum(probability * np.log(probability))),
    }


def _payload(mode, split, *, seed=4, replicates=12):
    rng = np.random.default_rng(seed)
    records = []
    row_axis = np.asarray([-1.0, 0.0, 1.0])
    column_axis = np.asarray([-1.0, 0.0, 1.0])
    for vote in range(4):
        truth = -1.0 if vote < 2 else 1.0
        for replicate in range(replicates):
            image = f"{split}-v{vote}-n{replicate}"
            base = (-3.0, -0.65, 0.65, 3.0)[vote] + rng.normal(0, 0.08)
            # Vary the additive landscape per image.  This makes a localized
            # harmful interaction distinguishable from exchangeable energy.
            magnitude = (0.8, 1.1) if vote in (0, 3) else (0.15, 0.35)
            ra = rng.uniform(*magnitude) * rng.choice((-1.0, 1.0))
            ca = rng.uniform(*magnitude) * rng.choice((-1.0, 1.0))
            additive = base + ra * row_axis[:, None] + ca * column_axis[None, :]
            if mode == "additive":
                interaction = np.zeros((3, 3))
            elif mode == "generic_random":
                interaction = rng.normal(size=(3, 3))
                interaction -= interaction.mean(axis=0, keepdims=True)
                interaction -= interaction.mean(axis=1, keepdims=True)
                interaction += interaction.mean()
                interaction *= 1.1 / max(np.linalg.norm(interaction), 1e-9)
            elif mode == "reader_localized_harm":
                if vote in (0, 3):
                    # Outcome-blind adversarial fixture: among centered
                    # matrices of fixed energy, choose the one that maximizes
                    # a smooth reader-distribution loss for this additive
                    # landscape.  Force J[0,0]=0 so the canonical calibration
                    # cell cannot learn the injected product orientation.
                    anchor = np.outer([2.0, -1.0, -1.0], [2.0, -1.0, -1.0])
                    target = float(vote == 3)
                    additive_p = 1 / (1 + np.exp(-additive))
                    best = None
                    best_loss = -np.inf
                    for _ in range(512):
                        candidate = rng.normal(size=(3, 3))
                        candidate -= candidate.mean(axis=0, keepdims=True)
                        candidate -= candidate.mean(axis=1, keepdims=True)
                        candidate += candidate.mean()
                        candidate -= (candidate[0, 0] / anchor[0, 0]) * anchor
                        norm = np.linalg.norm(candidate)
                        if norm < 1e-9:
                            continue
                        candidate *= 7.0 / norm
                        probability = 1 / (1 + np.exp(-(additive + candidate)))
                        loss = np.mean((probability - target) ** 2 - (additive_p - target) ** 2)
                        if loss > best_loss:
                            best_loss = loss
                            best = candidate
                    interaction = best
                else:
                    interaction = np.zeros((3, 3))
            else:
                raise ValueError(mode)
            score = additive + interaction
            for r, render in enumerate(RENDERS):
                for p, prompt in enumerate(PROMPTS):
                    records.append(_record(image, vote, render, prompt, score[r, p]))
            for p, prompt in enumerate(PROMPTS):
                records.append(_record(image, vote, "canonical_identity", prompt, score[0, p]))
            records.append(_record(image, vote, "canonical", "prompt_a_duplicate", score[0, 0]))
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
        "identity_render": "canonical_identity",
        "duplicate_prompt": "prompt_a_duplicate",
        "records": records,
    }


def _record(image, vote, render, prompt, score):
    return {
        "model": "Huatuo",
        "image_id": image,
        "patient_id": "patient-" + image,
        "finding": "effusion",
        "reader_votes": vote,
        "render_id": render,
        "prompt_id": prompt,
        "acquisition_view": "PA",
        "input_prompt_length_tokens": 9,
        "answer_length_tokens": 1,
        **_logit_fields(score),
    }


def _fit(mode="additive"):
    return fit_dev_calibration(
        _payload(mode, "dev_fit", seed=10), folds=2,
        null_draws=39, bootstrap_draws=59, seed=123,
    )


def _two_models(payload):
    result = copy.deepcopy(payload)
    duplicate = copy.deepcopy(result["records"])
    for row in duplicate:
        row["model"] = "Hulu"
    result["records"].extend(duplicate)
    return result


def test_spectral_haar_stays_centered_and_preserves_singular_values():
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(4, 3))
    value = raw - raw.mean(axis=0, keepdims=True)
    value -= value.mean(axis=1, keepdims=True)
    value += value.mean()
    rotated = spectral_haar_interaction(value, rng)
    assert np.allclose(rotated.sum(axis=0), 0, atol=1e-10)
    assert np.allclose(rotated.sum(axis=1), 0, atol=1e-10)
    assert np.allclose(np.linalg.svd(rotated, compute_uv=False), np.linalg.svd(value, compute_uv=False))
    coordinate = cell_coordinate_interaction(value, np.random.default_rng(9))
    assert np.allclose(coordinate.sum(axis=0), 0, atol=1e-10)
    assert np.allclose(coordinate.sum(axis=1), 0, atol=1e-10)
    assert np.linalg.norm(coordinate) == pytest.approx(np.linalg.norm(value))


def test_pure_additive_grid_has_zero_product_risk_and_never_authorizes():
    bundle = _fit("additive")
    result = apply_confirmation(_payload("additive", "confirmation_locked", seed=12), bundle)
    assert result["status"] == NON_AUTHORIZING
    assert result["authorized"] is False
    assert result["observed_product_risk"]["brier_delta"] == pytest.approx(0.0, abs=1e-14)
    assert result["observed_product_risk"]["soft_bernoulli_nll_delta"] == pytest.approx(0.0, abs=1e-14)
    assert result["primary_pael"]["models"]["Huatuo"]["point"] == pytest.approx(0.0, abs=1e-14)


def test_generic_random_interaction_does_not_beat_matched_energy_null():
    bundle = _fit("generic_random")
    result = apply_confirmation(
        _payload("generic_random", "confirmation_locked", seed=10), bundle
    )
    matched = result["null_diagnostics"]["matched_orbit"]["brier_delta"]
    assert matched["null_percentile"] < 0.95
    assert matched["approximate_reference_tail_fraction_greater_equal"] > 0.05


def test_reader_localized_harm_has_positive_pael_and_beats_matched_null():
    bundle = fit_dev_calibration(
        _payload("reader_localized_harm", "dev_fit", seed=10, replicates=32),
        folds=2, null_draws=39, bootstrap_draws=99, seed=123,
    )
    result = apply_confirmation(
        _payload("reader_localized_harm", "confirmation_locked", seed=10, replicates=32), bundle
    )
    pael = result["primary_pael"]["models"]["Huatuo"]
    matched = result["null_diagnostics"]["matched_orbit"]["brier_delta"]
    assert pael["point"] > 0
    assert pael["ci95"][0] > 0
    assert matched["excess"] > 0
    assert matched["null_percentile"] >= 0.95


def test_overlap_refit_tamper_seed_change_and_cell_deletion_fail_closed():
    dev = _payload("additive", "dev_fit", seed=10)
    bundle = fit_dev_calibration(dev, folds=2, null_draws=19, bootstrap_draws=19, seed=8)

    with pytest.raises(ContractError, match="apply-only"):
        apply_confirmation(dev, bundle)

    overlap = _payload("additive", "confirmation_locked", seed=12)
    old = overlap["records"][0]["image_id"]
    new = dev["records"][0]["image_id"]
    for row in overlap["records"]:
        if row["image_id"] == old:
            row["image_id"] = new
            row["patient_id"] = "patient-" + new
    with pytest.raises(ContractError, match="overlap"):
        apply_confirmation(overlap, bundle)

    tampered = copy.deepcopy(bundle)
    tampered["calibrators"]["Huatuo"]["effusion"]["y_thresholds"][0] += 0.01
    with pytest.raises(ContractError, match="modified after freezing"):
        apply_confirmation(_payload("additive", "confirmation_locked", seed=12), tampered)

    changed_seed = copy.deepcopy(bundle)
    changed_seed["randomization"]["seed"] += 1
    with pytest.raises(ContractError, match="modified after freezing"):
        apply_confirmation(_payload("additive", "confirmation_locked", seed=12), changed_seed)

    stale_source = copy.deepcopy(bundle)
    stale_source["source_sha256"] = "0" * 64
    del stale_source["bundle_sha256"]
    stale_source["bundle_sha256"] = object_sha256(stale_source)
    with pytest.raises(ContractError, match="current v4 module"):
        apply_confirmation(_payload("additive", "confirmation_locked", seed=12), stale_source)

    deleted = _payload("additive", "confirmation_locked", seed=12)
    del deleted["records"][0]
    with pytest.raises(ContractError, match="incomplete factorial orbit"):
        apply_confirmation(deleted, bundle)


def test_incomplete_confirmation_matching_fails_closed():
    bundle = _fit("generic_random")
    tiny = _payload("generic_random", "confirmation_locked", seed=42, replicates=1)
    result = apply_confirmation(tiny, bundle)
    assert result["null_diagnostics"]["matched_orbit"]["available"] is False
    assert "incomplete confirmation matched-orbit stratum" in result["null_diagnostics"]["matched_orbit"]["reason"]
    assert result["primary_pael"]["models"]["Huatuo"]["decision"] == "not_computed_non_authorizing_prototype"


def test_two_models_share_identical_whole_image_bootstrap_multipliers():
    dev = _two_models(_payload("generic_random", "dev_fit", seed=10))
    bundle = fit_dev_calibration(
        dev, folds=2, null_draws=19, bootstrap_draws=19, seed=44
    )
    confirmation = _two_models(
        _payload("generic_random", "confirmation_locked", seed=10)
    )
    result = apply_confirmation(confirmation, bundle)
    primary = result["primary_pael"]
    assert set(primary["models"]) == {"Huatuo", "Hulu"}
    assert primary["shared_cluster_multiplier_plan"]["same_weights_used_for_every_model"] is True
    assert len(primary["shared_cluster_multiplier_plan"]["plan_sha256"]) == 64
