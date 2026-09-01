import json

import numpy as np
import pytest

from corrected_sgta.analyze_clinical_equivalence_composition_defect_v1 import (
    behavioral_pid_mmi,
    ContractError,
    MARGINAL_FEATURES,
    _fit_predict,
    analyze,
    apply_confirmation_stage,
    fit_dev_stage,
    load_inputs,
    runner_rows_to_payload,
    two_way_centered,
    validate_payload,
)


RENDERS = ("canonical", "native_linear", "center_plus")
PROMPTS = ("is_there", "does_show", "can_be_seen")
FINDINGS = ("aortic", "cardiomegaly", "effusion", "fibrosis")


def consistent_scores(signed: float) -> dict:
    logits = {
        "supported": signed / 2.0,
        "refuted": -signed / 2.0,
        "undetermined": 0.0,
    }
    values = np.asarray(list(logits.values()), dtype=float)
    probability = np.exp(values - values.max())
    probability /= probability.sum()
    return {
        "polarity": signed,
        "commitment": abs(signed) / 2.0,
        "tristate_entropy": float(
            -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)))
        ),
        "logits": logits,
    }


def synthetic_payload(mode: str, *, models=("Huatuo",), seed: int = 7):
    rng = np.random.default_rng(seed)
    records = []
    render_axis = np.asarray([-1.0, 0.0, 1.0])
    prompt_axis = np.asarray([-1.0, 0.0, 1.0])
    for model in models:
        for finding in FINDINGS:
            for vote in range(4):
                for replicate in range(10):
                    image_id = f"{finding}-{vote}-{replicate}"
                    acquisition_view = "AP" if replicate == 9 else "PA"
                    ordinary = (vote - 1.5) + rng.normal(scale=0.03)
                    main = 0.04 * render_axis[:, None] + 0.03 * prompt_axis[None, :]
                    if mode == "additive":
                        interaction = np.zeros((3, 3))
                        base = ordinary
                    elif mode == "nonlinear_no_new_error":
                        # A real mixed derivative exists, but all clear errors
                        # are already determined by the clean score.  The
                        # interaction is too small to create or repair one.
                        base = 2.0 * (vote - 1.5)
                        if vote in (0, 3) and replicate % 4 == 0:
                            base *= -0.5
                        orientation = -1.0 if replicate % 2 else 1.0
                        interaction = 0.6 * orientation * np.outer(render_axis, prompt_axis)
                    elif mode == "true_interaction":
                        base = (vote - 1.5) * rng.uniform(0.4, 1.6)
                        # A random centered interaction prevents cell position
                        # or unsigned instability magnitude from revealing its
                        # clinically harmful direction.  The signed mixed
                        # derivative still reconstructs that missing direction.
                        a, b, c = rng.normal(size=3)
                        d = -a - b - c
                        interaction = np.asarray(
                            [
                                [0.0, -(a + c), -(b + d)],
                                [-(a + b), a, b],
                                [-(c + d), c, d],
                            ]
                        )
                        interaction *= rng.uniform(1.5, 3.5) / np.sqrt(
                            np.mean(interaction**2)
                        )
                    else:
                        raise ValueError(mode)
                    score = base + main + interaction
                    commitment = np.abs(score)
                    for render_index, render in enumerate(RENDERS):
                        for prompt_index, prompt in enumerate(PROMPTS):
                            records.append(
                                {
                                    "model": model,
                                    "image_id": image_id,
                                    "finding": finding,
                                    "reader_votes": vote,
                                    "render_id": render,
                                    "prompt_id": prompt,
                                    "signed_score": float(score[render_index, prompt_index]),
                                    "commitment_score": float(commitment[render_index, prompt_index]),
                                    "acquisition_view": acquisition_view,
                                    "tristate_entropy": 0.4,
                                    # Optional scalar sensitivity control. It
                                    # is explicitly not an exact Treble run.
                                    "crossmodal_direct_effect_scalar_surrogate": float(main[render_index, prompt_index]),
                                }
                            )
                    # Exact image and prompt duplicates are required controls.
                    for prompt_index, prompt in enumerate(PROMPTS):
                        records.append(
                            {
                                "model": model,
                                "image_id": image_id,
                                "finding": finding,
                                "reader_votes": vote,
                                "render_id": "canonical_identity",
                                "prompt_id": prompt,
                                "signed_score": float(score[0, prompt_index]),
                                "commitment_score": float(commitment[0, prompt_index]),
                                "acquisition_view": acquisition_view,
                                "tristate_entropy": 0.4,
                                "crossmodal_direct_effect_scalar_surrogate": float(main[0, prompt_index]),
                            }
                        )
                    records.append(
                        {
                            "model": model,
                            "image_id": image_id,
                            "finding": finding,
                            "reader_votes": vote,
                            "render_id": "canonical",
                            "prompt_id": "is_there_duplicate",
                            "signed_score": float(score[0, 0]),
                            "commitment_score": float(commitment[0, 0]),
                            "acquisition_view": acquisition_view,
                            "tristate_entropy": 0.4,
                            "crossmodal_direct_effect_scalar_surrogate": float(main[0, 0]),
                        }
                    )
    prompt_lengths = {
        "is_there": 9,
        "does_show": 11,
        "can_be_seen": 13,
        "is_there_duplicate": 9,
    }
    for row in records:
        signed = float(row["signed_score"])
        row["tristate_logits"] = {
            "supported": signed / 2.0,
            "refuted": -signed / 2.0,
            "undetermined": 0.0,
        }
        values = np.asarray(list(row["tristate_logits"].values()), dtype=float)
        probability = np.exp(values - values.max())
        probability /= probability.sum()
        row["tristate_entropy"] = float(
            -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)))
        )
        row["commitment_score"] = abs(signed) / 2.0
        row["input_prompt_length_tokens"] = prompt_lengths[row["prompt_id"]]
        row["answer_length_tokens"] = 1
    return {
        "schema_version": "clinical-equivalence-factorial-v1",
        "split": "dev",
        "frozen_before_outputs": True,
        "score_definition": "fp32_yes_minus_no_logit",
        "primary_renders": list(RENDERS),
        "primary_prompts": list(PROMPTS),
        "baseline_render": "canonical",
        "baseline_prompt": "is_there",
        "identity_render": "canonical_identity",
        "duplicate_prompt": "is_there_duplicate",
        "records": records,
    }


def test_two_way_centering_removes_both_additive_main_effects():
    row = np.asarray([-2.0, 0.0, 3.0])[:, None]
    column = np.asarray([-1.0, 4.0, 7.0])[None, :]
    assert np.allclose(two_way_centered(11.0 + row + column), 0.0, atol=1e-12)


def test_three_stage_dev_fit_and_confirmation_are_apply_only_and_image_disjoint():
    dev = synthetic_payload("true_interaction", models=("Huatuo", "Hulu"))
    dev["split"] = "dev_fit"
    dev["source_manifest_split"] = "dev"
    fit = fit_dev_stage(dev, folds=4, draws=20, seed=17)
    assert fit["gate"]["authorized_for_method_level_treble_adapter_run"] is False

    confirmation = synthetic_payload(
        "true_interaction", models=("Huatuo", "Hulu"), seed=19
    )
    confirmation["split"] = "confirmation_locked"
    confirmation["source_manifest_split"] = "confirmation"
    for row in confirmation["records"]:
        row["image_id"] = "locked-" + row["image_id"]
    result = apply_confirmation_stage(confirmation, fit, draws=20, seed=23)
    assert result["stage_label"] == "confirmation_locked"
    assert all(
        model["dev_predictor_refit_on_confirmation"] is False
        for model in result["models"].values()
    )
    assert result["gate"]["authorized_for_hidden_state_stage"] is False

    confirmation["records"][0]["image_id"] = dev["records"][0]["image_id"]
    # One changed cell first causes an incomplete orbit; changing the entire
    # orbit reaches the intended whole-image leakage guard.
    original = "locked-" + dev["records"][0]["image_id"]
    leaked = dev["records"][0]["image_id"]
    for row in confirmation["records"]:
        if row["image_id"] == original:
            row["image_id"] = leaked
    with pytest.raises(ContractError, match="whole-image leakage"):
        apply_confirmation_stage(confirmation, fit, draws=5, seed=23)


def test_pure_additive_factorial_fails_interaction_gate():
    result = analyze(synthetic_payload("additive"), folds=4, draws=80, seed=11)
    model = result["models"]["Huatuo"]
    assert model["interaction_rms_reader_equivalents"]["point"] < 1e-10
    assert model["model_screen_pass"] is False
    assert result["gate"]["authorized_for_hidden_state_stage"] is False


def test_nonlinearity_without_incremental_error_information_fails():
    result = analyze(
        synthetic_payload("nonlinear_no_new_error"), folds=4, draws=80, seed=13
    )
    model = result["models"]["Huatuo"]
    assert model["interaction_rms_reader_equivalents"]["point"] >= 0.25
    increment = model["clear_reader_polarity_error"][
        "cecd_vs_marginals_plus_full_orbit"
    ]
    assert increment["delta_auc"] < 0.03
    assert model["model_screen_pass"] is False


def test_external_fold_leakage_is_rejected_fail_closed():
    payload = synthetic_payload("additive")
    image_id = payload["records"][0]["image_id"]
    same_image = [row for row in payload["records"] if row["image_id"] == image_id]
    same_image[0]["fold_id"] = "fold-0"
    same_image[1]["fold_id"] = "fold-1"
    with pytest.raises(ContractError, match="leaks one image"):
        validate_payload(payload)


def test_readout_and_fixed_length_controls_are_hash_bound_fail_closed():
    payload = synthetic_payload("additive")
    payload["records"][0]["answer_length_tokens"] = 2
    with pytest.raises(ContractError, match="must equal one"):
        validate_payload(payload)

    payload = synthetic_payload("additive")
    payload["records"][0]["tristate_logits"]["supported"] += 0.25
    with pytest.raises(ContractError, match="signed_score disagrees"):
        validate_payload(payload)


def test_behavioral_pid_is_mmi_control_not_hidden_pid():
    probabilities = np.full((3, 5, 3), 1.0 / 3.0)
    result = behavioral_pid_mmi(probabilities)
    assert result["mmi_synergy_nats"] == pytest.approx(0.0)
    assert np.allclose(result["local_synergy_excess_nats"], 0.0)
    assert "not hidden-state PID" in result["guardrail"]


def test_fold_local_encoder_safely_ignores_unseen_acquisition_view():
    def row(index, view, target):
        item = {name: float(index % 3) for name in MARGINAL_FEATURES}
        item.update(
            finding="effusion", acquisition_view=view,
            polarity_error=target,
        )
        return item

    train = [row(index, "PA", index % 2) for index in range(8)]
    test = [row(9, "UNSEEN_VALID_CATEGORY", 0), row(10, "UNSEEN_VALID_CATEGORY", 1)]
    prediction = _fit_predict(train, test, MARGINAL_FEATURES, seed=3)
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()


def test_native_runner_15_plus_3_plus_1_jsonl_contract_is_accepted():
    wrapper = synthetic_payload("additive")
    runner_rows = []
    for row in wrapper["records"]:
        converted = {
            "contract_version": "clinical-equivalence-factorial-v1",
            "model": row["model"],
            "image_id": row["image_id"],
            "finding": row["finding"],
            "positive_votes": row["reader_votes"],
            "render_id": row["render_id"],
            "prompt_id": row["prompt_id"],
            "signed_score": row["signed_score"],
            "commitment_score": row["commitment_score"],
            "acquisition_view": row["acquisition_view"],
            "tristate_entropy": row["tristate_entropy"],
            "tristate_logits": row["tristate_logits"],
            "raw_prompt_token_count": row["input_prompt_length_tokens"],
            "answer_length_tokens": row["answer_length_tokens"],
            "status": "ok",
        }
        if row["render_id"] == "canonical_identity":
            converted.update(
                cell_role="identity_image_control",
                cell_id=f"control_identity_image__{row['prompt_id']}",
                reference_cell_id=f"science__canonical__{row['prompt_id']}",
            )
        elif row["prompt_id"] == "is_there_duplicate":
            converted.update(
                cell_role="exact_duplicate_prompt_control",
                cell_id="control_duplicate_prompt__is_there_duplicate",
                reference_cell_id="science__canonical__is_there",
            )
        else:
            converted.update(
                cell_role="science_factorial",
                cell_id=f"science__{row['render_id']}__{row['prompt_id']}",
                reference_cell_id=None,
            )
        runner_rows.append(converted)
    adapted = runner_rows_to_payload(runner_rows)
    contract = validate_payload(adapted)
    assert contract["baseline_render"] == "canonical"
    assert contract["baseline_prompt"] == "is_there"
    assert contract["identity_render"] == "canonical_identity"
    assert contract["duplicate_prompt"] == "is_there_duplicate"


def test_runner_packer_payload_loads_and_runs_analyzer_end_to_end(tmp_path):
    from corrected_sgta.run_cecd_factorial_v1 import (
        cell_specs,
        pack_factorial_rows,
        safe_record_key,
        shard_path,
    )

    selected = []
    for vote in range(4):
        for replicate in range(3):
            individual = [1] * vote + [0] * (3 - vote)
            row = {
                "image_id": f"packed-{vote}-{replicate}",
                "finding": "pleural_effusion",
                "positive_votes": vote,
                "reader_votes": individual,
                "reader_support": vote / 3,
            }
            selected.append(row)
            record_key = safe_record_key(row)
            for index, spec in enumerate(cell_specs(row["finding"])):
                target = shard_path(tmp_path / "cell_shards", record_key, spec.cell_id)
                target.parent.mkdir(parents=True, exist_ok=True)
                shard = {
                    "status": "ok",
                    "config_fingerprint": "cfg",
                    "record_key": record_key,
                    "image_id": row["image_id"],
                    "finding": row["finding"],
                    "positive_votes": vote,
                    "individual_reader_votes": individual,
                    "reader_support": vote / 3,
                    "acquisition_view": "pa",
                    "cell_id": spec.cell_id,
                    "cell_role": spec.role,
                    "reference_cell_id": spec.reference_cell_id,
                    "render_name": spec.render_name,
                    "render_pixel_sha256": "a" * 64,
                    "prompt_name": spec.prompt_name,
                    "prompt_text_sha256": __import__("hashlib").sha256(
                        spec.prompt_text.encode()
                    ).hexdigest(),
                    "raw_prompt_token_count": 12,
                    "scores": consistent_scores(float(vote - 1.5)),
                }
                target.write_text(json.dumps(shard))
    pack_factorial_rows(tmp_path, {"fingerprint": "cfg", "model": "huatuo:fake"}, selected)
    wrapper_path = tmp_path / "factorial_payload.json"
    loaded = load_inputs([wrapper_path])
    result = analyze(loaded, folds=2, draws=10, seed=5)
    assert result["status"] == "complete"
    assert result["contract"]["excluded_orbit_count"] == 0
    assert result["models"]["huatuo:fake"]["n_orbits"] == 12


def test_one_invalid_science_cell_excludes_exactly_one_complete_orbit():
    payload = synthetic_payload("additive")
    failed = next(
        row for row in payload["records"]
        if row["render_id"] == "center_plus" and row["prompt_id"] == "does_show"
    )
    failed["valid"] = False
    failed["signed_score"] = None
    failed["commitment_score"] = None
    failed["invalid_reasons"] = ["per_case_computational_guard_failed"]
    contract = validate_payload(payload)
    assert len(contract["excluded_orbits"]) == 1
    excluded = contract["excluded_orbits"][0]
    assert excluded["image_id"] == failed["image_id"]
    assert excluded["finding"] == failed["finding"]
    assert excluded["policy"] == "whole_orbit_excluded_no_imputation"
    assert excluded["reasons"] == ["per_case_computational_guard_failed"]
    assert len(contract["by_orbit"]) == 159
    assert all(
        key[1:] != (failed["image_id"], failed["finding"])
        for key in contract["by_orbit"]
    )


def test_reader_grounded_interaction_explained_by_generic_stability_fails():
    result = analyze(
        synthetic_payload("true_interaction", models=("Huatuo", "Hulu")),
        folds=4,
        draws=100,
        seed=17,
    )
    assert result["estimand"]["type"] == "two-way centered interaction / discrete mixed derivative"
    assert "not an algebraic commutator" in result["estimand"]["guardrail"]
    assert result["gate"]["authorized_for_hidden_state_stage"] is False
    assert result["gate"]["authorized_for_method_level_treble_adapter_run"] is False
    assert result["gate"]["behavioral_phenomenon_confirmed_on_locked_test"] is False
    assert result["gate"]["oral_baseline_closure_established"] is False
    assert result["exact_treble_method_collision"]["status"] == "not_authorized"
    assert result["exact_treble_method_collision"]["hidden_state_authorized"] is False
    for model in result["models"].values():
        assert model["interaction_rms_reader_equivalents"]["point"] >= 0.25
        ladder = model["clear_reader_polarity_error"][
            "behavioral_incremental_collision_ladder"
        ]
        comparison = ladder["behavioral_mmi_pid_style_synergy"]
        assert comparison["baseline_auc"] > 0.99
        assert comparison["delta_auc"] < 0.03
        assert model["clear_reader_polarity_error"]["clinically_harmful_direction"]["ci95"][0] > 0
        assert model["same_direction_findings"] < 3
        assert model["identity_controls"]["maximum_rms_re"] == pytest.approx(0.0)
        assert model["reader_disagreement_separate_axis"]["gate_role"] == "diagnostic_only"
        assert model["model_screen_pass"] is False


def test_no_scalar_payload_cannot_rescue_a_behavioral_no_go():
    payload = synthetic_payload("true_interaction", models=("Huatuo", "Hulu"))
    for row in payload["records"]:
        legacy = row.pop("crossmodal_direct_effect_scalar_surrogate", None)
        row["treble_nde_score"] = legacy  # must be ignored, never exact Treble
    result = analyze(payload, folds=4, draws=40, seed=19)
    assert result["gate"]["authorized_for_hidden_state_stage"] is False
    assert result["gate"]["authorized_for_method_level_treble_adapter_run"] is False
    assert all(
        not model["gate_components"]["crossmodal_direct_effect_scalar_surrogate_complete"]
        for model in result["models"].values()
    )
    assert result["exact_treble_method_collision"]["hidden_state_authorized"] is False
