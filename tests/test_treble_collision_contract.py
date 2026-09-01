import numpy as np
import pytest

from corrected_sgta.treble_collision_contract import (
    DUAL_SEMANTICS_METHODS,
    DUAL_SEMANTICS_OUTCOME_SCHEMA,
    DUAL_SEMANTICS_PREFLIGHT_SCHEMA,
    DUAL_SEMANTICS_THRESHOLDS,
    DUAL_SEMANTICS_VARIANTS,
    EXTERNAL_OUTCOME_SCHEMA,
    TREBLE_REPOSITORY_COMMIT,
    TrebleContractError,
    paper_representation_deltas,
    proceedings_compute_ledger,
    released_code_compute_ledger,
    released_code_norm_preserving_shift,
    released_code_representation_deltas,
    source_audit,
    validate_dual_semantics_envelope_outcome,
    validate_dual_semantics_preflight_contract,
    validate_external_method_outcome,
)


def _dual_semantics_payload():
    baseline = {
        "ce_overcommitment_rate": 0.30,
        "ce_clear_accuracy": 0.90,
        "oe_hallucination_rate": 0.25,
        "oe_omission_rate": 0.10,
        "oe_claim_coverage": 0.80,
        "oe_mean_claims": 5.0,
        "oe_mean_length": 100.0,
        "oe_refusal_rate": 0.0,
        "reader_brier": 0.20,
    }
    values = {
        "unmitigated": baseline,
        "cecd_interaction_projection": {
            **baseline,
            "ce_overcommitment_rate": 0.20,
            "ce_clear_accuracy": 0.895,
            "oe_hallucination_rate": 0.18,
            "oe_omission_rate": 0.09,
            "reader_brier": 0.18,
        },
        "treble_proceedings": {
            **baseline,
            "ce_overcommitment_rate": 0.23,
            "oe_hallucination_rate": 0.21,
            "reader_brier": 0.19,
        },
        "treble_released": {
            **baseline,
            "ce_overcommitment_rate": 0.22,
            "oe_hallucination_rate": 0.20,
            "reader_brier": 0.18,
        },
        "full_orbit": {
            **baseline,
            "ce_overcommitment_rate": 0.225,
            "oe_hallucination_rate": 0.205,
            "reader_brier": 0.19,
        },
        "render_only": {**baseline, "ce_overcommitment_rate": 0.26},
        "prompt_only": {**baseline, "ce_overcommitment_rate": 0.27},
        "random_norm": {**baseline, "ce_overcommitment_rate": 0.29},
        "sign_permuted": {**baseline, "ce_overcommitment_rate": 0.28},
        "main_effect_removal": {**baseline, "ce_overcommitment_rate": 0.25},
    }
    assert set(values) == set(DUAL_SEMANTICS_METHODS)
    bootstrap = {}
    for control in ("treble_proceedings", "treble_released", "full_orbit"):
        bootstrap[f"cecd_vs_{control}"] = {
            "ce_overcommitment_control_minus_cecd": {
                "point": 0.02,
                "ci_lower": 0.005,
                "ci_upper": 0.04,
                "replicates": 10_000,
                "unit": "cluster_id",
            },
            "oe_hallucination_control_minus_cecd": {
                "point": 0.02,
                "ci_lower": 0.004,
                "ci_upper": 0.04,
                "replicates": 10_000,
                "unit": "cluster_id",
            },
            "reader_brier_control_minus_cecd": {
                "point": 0.005,
                "ci_lower": 0.0,
                "ci_upper": 0.02,
                "replicates": 10_000,
                "unit": "cluster_id",
            },
        }
    proceedings_calibration = proceedings_compute_ledger()
    released_calibration = released_code_compute_ledger().__dict__
    fingerprints = {}
    ledgers = {}
    paired = {}
    intervals = {}
    for index, family in enumerate(("huatuo", "hulu"), 1):
        fingerprints[family] = {
            "model_id": f"{family}:frozen",
            "checkpoint_sha256": str(index) * 64,
            "processor_sha256": str(index + 1) * 64,
            "template_sha256": str(index + 2) * 64,
            "generation_contract_sha256": str(index + 3) * 64,
            "hook_contract_sha256": str(index + 4) * 64,
            "vision_token_transport_contract_sha256": str(index + 5) * 64,
        }
        ledgers[family] = {
            "treble_proceedings": dict(proceedings_calibration),
            "treble_released": dict(released_calibration),
            "target_examples": 100,
            "cecd_target_generation_forwards": 400,
            "full_orbit_target_generation_forwards": 400,
        }
        paired[family] = {
            "n_clusters": {"ce": 100, "oe": 100},
            "methods": {method: dict(metrics) for method, metrics in values.items()},
        }
        intervals[family] = {
            name: {metric: dict(estimate) for metric, estimate in metrics.items()}
            for name, metrics in bootstrap.items()
        }
    return {
        "schema_version": DUAL_SEMANTICS_OUTCOME_SCHEMA,
        "source_repo_commit": TREBLE_REPOSITORY_COMMIT,
        "reproduction_fidelity": "dual_semantics_common_protocol_envelope",
        "paper_native_claimed": False,
        "exact_reproduction_claimed": False,
        "implementation_origin": "independent_clean_room_from_public_equations_and_audited_arithmetic",
        "redistribution_policy": "local_evaluation_only_no_official_source_or_demo_redistribution",
        "variants": {name: dict(spec) for name, spec in DUAL_SEMANTICS_VARIANTS.items()},
        "model_fingerprints": fingerprints,
        "calibration_split": "dev",
        "evaluation_split": "locked_test",
        "calibration_manifest_sha256": "a" * 64,
        "evaluation_manifest_sha256": "b" * 64,
        "record_keys_sha256": "c" * 64,
        "claim_contract_sha256": "d" * 64,
        "preflight_sha256": "e" * 64,
        "compute_ledger": ledgers,
        "paired_method_metrics": paired,
        "paired_cluster_bootstrap": intervals,
        "collision_verdict": "cecd_survives_dual_semantics_envelope",
    }


def _dual_semantics_preflight():
    outcome = _dual_semantics_payload()
    return {
        "schema_version": DUAL_SEMANTICS_PREFLIGHT_SCHEMA,
        "frozen_before_method_outputs": True,
        "source_repo_commit": outcome["source_repo_commit"],
        "reproduction_fidelity": outcome["reproduction_fidelity"],
        "paper_native_claimed": False,
        "exact_reproduction_claimed": False,
        "implementation_origin": outcome["implementation_origin"],
        "redistribution_policy": outcome["redistribution_policy"],
        "variants": outcome["variants"],
        "model_fingerprints": outcome["model_fingerprints"],
        "stage1_analysis_sha256": "1" * 64,
        "stage1_input_gate_sha256": "2" * 64,
        "admission_sha256": "3" * 64,
        "calibration_split": "dev",
        "evaluation_split": "locked_test",
        "calibration_manifest_sha256": outcome["calibration_manifest_sha256"],
        "evaluation_manifest_sha256": outcome["evaluation_manifest_sha256"],
        "record_keys_sha256": outcome["record_keys_sha256"],
        "claim_contract_sha256": outcome["claim_contract_sha256"],
        "methods": list(DUAL_SEMANTICS_METHODS),
        "primary_envelope_controls": [
            "treble_proceedings",
            "treble_released",
            "full_orbit",
        ],
        "method_metrics": [
            "ce_overcommitment_rate",
            "ce_clear_accuracy",
            "oe_hallucination_rate",
            "oe_omission_rate",
            "oe_claim_coverage",
            "oe_mean_claims",
            "oe_mean_length",
            "oe_refusal_rate",
            "reader_brier",
        ],
        "thresholds": dict(DUAL_SEMANTICS_THRESHOLDS),
        "bootstrap_replicates": 10_000,
        "bootstrap_unit": "cluster_id",
        "compute_ledger": outcome["compute_ledger"],
        "method_output_root": "/tmp/cecd-dual-semantics-locked-output",
    }


def test_proceedings_and_released_vision_cross_modal_deltas_are_not_silently_equated():
    original = np.array([1.0, 2.0])
    hallucinated = np.array([4.0, 8.0])
    black = np.array([3.0, 5.0])
    no_image = np.array([1.0, 1.0])
    noisy = np.array([2.0, 4.0])
    paper = paper_representation_deltas(
        original_vision=original,
        corrupted_vision_mean=hallucinated,
        original_text=original,
        hallucinated_text=hallucinated,
        black_image_text_state=black,
        no_image_text_state=no_image,
    )
    released = released_code_representation_deltas(
        original_vision=original,
        gaussian_step500_vision_mean=hallucinated,
        factual_caption_text_state=original,
        hallucinated_caption_text_state=hallucinated,
        no_image_text_state=no_image,
        gaussian_step200_image_text_state_mean=noisy,
    )
    np.testing.assert_allclose(paper["vision"], -released["vision"])
    np.testing.assert_allclose(paper["text"], released["text"])
    assert not np.allclose(paper["cross_modal"], released["cross_modal"])


def test_released_shift_preserves_each_activation_norm_and_uses_point_one_step():
    x = np.array([[[3.0, 4.0], [5.0, 12.0]]])
    direction = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    shifted = released_code_norm_preserving_shift(x, [direction], [0.9])
    np.testing.assert_allclose(
        np.linalg.norm(shifted, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-12
    )
    expected_first = np.array([0.6, 0.8]) + 0.1 * 0.9 * np.array([1.0, 0.0])
    expected_first /= np.linalg.norm(expected_first)
    np.testing.assert_allclose(shifted[0, 0], 5.0 * expected_first)


def test_default_compute_ledger_keeps_forward_types_separate():
    ledger = released_code_compute_ledger()
    assert ledger.vision_counterfactual_encoder_forwards == 2500
    assert ledger.vision_original_encoder_forwards == 50
    assert ledger.text_pair_multimodal_forwards == 100
    assert ledger.cross_degraded_multimodal_forwards == 2500
    assert ledger.cross_no_image_language_forwards == 50
    assert ledger.total_image_bearing_forwards == 5150
    assert ledger.target_generation_forwards_per_example == 1
    paper = proceedings_compute_ledger()
    assert paper["cross_black_image_multimodal_forwards"] == 50
    assert paper["total_image_bearing_forwards"] == 2700
    assert paper["total_image_bearing_forwards"] < ledger.total_image_bearing_forwards


def test_public_source_audit_blocks_exact_reproduction_and_scalar_nde_substitution():
    audit = source_audit()
    assert audit["reproduction_authorized"] is False
    assert audit["paper"] == "2025.findings-emnlp.1000"
    assert audit["doi"] == "10.18653/v1/2025.findings-emnlp.1000"
    assert "proceedings_vision_delta_order_disagrees_with_released_source" in audit["blockers"]
    assert "released_method_has_no_per_claim_scalar_nde_output" in audit["blockers"]
    assert any("two-way centered interaction" in item for item in audit["forbidden_substitutions"])


def test_external_exact_v1_gate_rejects_unresolved_and_self_attested_resolution():
    payload = {
        "schema_version": EXTERNAL_OUTCOME_SCHEMA,
        "source_repo_commit": TREBLE_REPOSITORY_COMMIT,
        "reproduction_fidelity": "blocked_unresolved_paper_code_semantics",
        "model_fingerprint": "model-hash",
        "calibration_split": "dev",
        "evaluation_split": "locked_test",
        "record_keys_sha256": "a" * 64,
        "compute_ledger": {"image_bearing_forwards": 5150},
        "paired_method_metrics": {"primary_delta": 0.04},
        "paired_cluster_bootstrap": {"primary_delta_ci95": [0.01, 0.07]},
        "collision_verdict": "direct_collision",
    }
    with pytest.raises(TrebleContractError, match="remain unresolved"):
        validate_external_method_outcome(payload)
    payload["reproduction_fidelity"] = "paper_and_code_semantics_resolved"
    with pytest.raises(TrebleContractError, match="self-attestation cannot resolve"):
        validate_external_method_outcome(payload)


def test_external_gate_rejects_empty_scientific_evidence():
    payload = {
        "schema_version": EXTERNAL_OUTCOME_SCHEMA,
        "source_repo_commit": TREBLE_REPOSITORY_COMMIT,
        "reproduction_fidelity": "paper_and_code_semantics_resolved",
        "model_fingerprint": "model-hash",
        "calibration_split": "dev",
        "evaluation_split": "locked_test",
        "record_keys_sha256": "a" * 64,
        "compute_ledger": {},
        "paired_method_metrics": {},
        "paired_cluster_bootstrap": {},
        "collision_verdict": "cecd_survives",
    }
    with pytest.raises(TrebleContractError, match="cannot be empty"):
        validate_external_method_outcome(payload)


def test_shape_and_zero_vector_fail_closed():
    with pytest.raises(TrebleContractError, match="equal shape"):
        paper_representation_deltas(
            original_vision=np.zeros(2),
            corrupted_vision_mean=np.zeros(3),
            original_text=np.zeros(2),
            hallucinated_text=np.zeros(2),
            black_image_text_state=np.zeros(2),
            no_image_text_state=np.zeros(2),
        )
    with pytest.raises(TrebleContractError, match="zero vector"):
        released_code_norm_preserving_shift(np.zeros((1, 2)), [np.ones((1, 2))], [0.9])


def test_dual_semantics_envelope_can_survive_without_impersonating_exact_treble():
    result = validate_dual_semantics_envelope_outcome(_dual_semantics_payload())
    assert result["valid"] is True
    assert result["computed_collision_verdict"] == "cecd_survives_dual_semantics_envelope"
    assert result["cecd_treble_envelope_advantage_established"] is True
    assert result["cecd_causal_claim_authorized"] is False
    assert result["full_method_gate_authorized"] is False
    assert result["oral_baseline_closure_authorized"] is False
    assert "no_official_compatible_dynamic_or_multimodal_activation_baseline" in result[
        "method_closure_limitations"
    ]
    assert result["paper_native_treble_reproduced"] is False
    assert result["exact_treble_reproduced"] is False
    assert result["paper_claim_authorized"] is False


def test_dual_semantics_preflight_freezes_outcome_blind_method_and_threshold_closure():
    validate_dual_semantics_preflight_contract(_dual_semantics_preflight())

    payload = _dual_semantics_preflight()
    payload["frozen_before_method_outputs"] = False
    with pytest.raises(TrebleContractError, match="before any output"):
        validate_dual_semantics_preflight_contract(payload)

    payload = _dual_semantics_preflight()
    payload["thresholds"]["oe_claim_count_absolute_difference_max"] = 1.0
    with pytest.raises(TrebleContractError, match="thresholds drifted"):
        validate_dual_semantics_preflight_contract(payload)

    payload = _dual_semantics_preflight()
    payload["primary_envelope_controls"].remove("full_orbit")
    with pytest.raises(TrebleContractError, match="control envelope is incomplete"):
        validate_dual_semantics_preflight_contract(payload)


def test_dual_semantics_envelope_rejects_false_exactness_or_semantic_blending():
    payload = _dual_semantics_payload()
    payload["paper_native_claimed"] = True
    with pytest.raises(TrebleContractError, match="cannot claim paper-native"):
        validate_dual_semantics_envelope_outcome(payload)

    payload = _dual_semantics_payload()
    payload["variants"]["treble_released"]["vision_delta"] = "silently_blended_sign"
    with pytest.raises(TrebleContractError, match="semantics are not exact"):
        validate_dual_semantics_envelope_outcome(payload)


def test_dual_semantics_envelope_requires_two_models_and_every_control():
    payload = _dual_semantics_payload()
    del payload["paired_method_metrics"]["hulu"]
    with pytest.raises(TrebleContractError, match="bind Huatuo and Hulu"):
        validate_dual_semantics_envelope_outcome(payload)

    payload = _dual_semantics_payload()
    del payload["paired_method_metrics"]["huatuo"]["methods"]["full_orbit"]
    with pytest.raises(TrebleContractError, match="method closure is incomplete"):
        validate_dual_semantics_envelope_outcome(payload)


def test_dual_semantics_envelope_recomputes_no_exchange_and_ci_verdicts():
    payload = _dual_semantics_payload()
    payload["paired_method_metrics"]["huatuo"]["methods"][
        "cecd_interaction_projection"
    ]["oe_mean_claims"] = 4.0
    payload["collision_verdict"] = "collision_or_no_specific_advantage"
    result = validate_dual_semantics_envelope_outcome(payload)
    assert result["cecd_causal_claim_authorized"] is False
    assert "cecd_interaction_projection_claim_count_not_fixed" in result[
        "family_failures"
    ]["huatuo"]

    payload = _dual_semantics_payload()
    estimate = payload["paired_cluster_bootstrap"]["hulu"][
        "cecd_vs_treble_released"
    ]["oe_hallucination_control_minus_cecd"]
    estimate["ci_lower"] = 0.0
    with pytest.raises(TrebleContractError, match="declared collision verdict"):
        validate_dual_semantics_envelope_outcome(payload)
    payload["collision_verdict"] = "collision_or_no_specific_advantage"
    result = validate_dual_semantics_envelope_outcome(payload)
    assert result["cecd_causal_claim_authorized"] is False


def test_dual_semantics_envelope_rejects_fake_hash_and_collapsed_compute_ledger():
    payload = _dual_semantics_payload()
    payload["record_keys_sha256"] = "z" * 64
    with pytest.raises(TrebleContractError, match="lowercase SHA-256"):
        validate_dual_semantics_envelope_outcome(payload)

    payload = _dual_semantics_payload()
    payload["compute_ledger"]["huatuo"]["treble_released"] = {
        "total_forwards": 5150
    }
    with pytest.raises(TrebleContractError, match="heterogeneous"):
        validate_dual_semantics_envelope_outcome(payload)
