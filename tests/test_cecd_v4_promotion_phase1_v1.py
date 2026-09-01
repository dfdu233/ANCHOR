import copy
import json
from pathlib import Path

import numpy as np
import pytest

from corrected_sgta.validate_cecd_v4_promotion_phase1_v1 import (
    EXPECTED_CONTROLS,
    EXPECTED_FINDINGS,
    EXPECTED_MODELS,
    EXPECTED_PROMPTS,
    EXPECTED_RENDERS,
    Phase1ContractError,
    PATIENT_MANIFEST_VERSION,
    build_phase1_contract,
    object_sha256,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = json.loads(
    (ROOT / "configs/cecd_v4_promotion_phase1_contract_20260803.json").read_text()
)


def _test_config():
    config = copy.deepcopy(FORMAL_CONFIG)
    config["stage_contract"]["dev_fit"]["exact_orbits_per_model_finding_vote"] = 2
    config["stage_contract"]["confirmation_locked"]["exact_orbits_per_model_finding_vote"] = 3
    config["cluster_contract"]["minimum_unique_patient_clusters_per_model_finding_vote"] = {
        "dev_fit": 2, "confirmation_locked": 3,
    }
    config["bootstrap"].update(
        draws=31,
        minimum_ess_fraction=0.02,
        maximum_single_cluster_weight_fraction=0.95,
        maximum_single_cluster_orbit_contribution=0.95,
    )
    return config


def _score_fields(score):
    logits = {
        "supported": score / 2,
        "refuted": -score / 2,
        "undetermined": 0.0,
    }
    values = np.asarray(list(logits.values()), dtype=float)
    probability = np.exp(values - values.max())
    probability /= probability.sum()
    return {
        "signed_score": float(score),
        "commitment_score": abs(float(score)) / 2,
        "tristate_logits": logits,
        "tristate_entropy": float(-np.sum(probability * np.log(probability))),
    }


def _payload(stage, quota):
    records = []
    prefix = "dev" if stage == "dev_fit" else "confirmation"
    for vote in range(4):
        for replicate in range(quota):
            # The same image deliberately carries every finding and both
            # models, exercising the one-global-cluster requirement.
            image = f"{prefix}-v{vote}-n{replicate}"
            patient = f"{prefix}-patient-v{vote}-n{replicate}"
            for model in EXPECTED_MODELS:
                for finding in EXPECTED_FINDINGS:
                    score = vote - 1.5
                    for render in EXPECTED_RENDERS:
                        for prompt in EXPECTED_PROMPTS:
                            records.append(
                                _row(model, image, patient, finding, vote, render, prompt, score)
                            )
                    for prompt in EXPECTED_PROMPTS:
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
                            EXPECTED_CONTROLS["duplicate_prompt"], score,
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
            "source_manifest_sha256": ("a" if stage == "dev_fit" else "b") * 64,
        },
        "score_definition": "fp32_yes_minus_no_logit",
        "primary_renders": list(EXPECTED_RENDERS),
        "primary_prompts": list(EXPECTED_PROMPTS),
        **EXPECTED_CONTROLS,
        "records": records,
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


def _external_patient_manifests(config, dev, confirmation):
    output = {}
    for stage, payload in (
        ("dev_fit", dev), ("confirmation_locked", confirmation),
    ):
        mapping = {}
        for row in payload["records"]:
            mapping.setdefault(row["image_id"], row["patient_id"])
        manifest = {
            "schema_version": PATIENT_MANIFEST_VERSION,
            "stage": stage,
            "frozen_before_model_outputs": True,
            "records": [
                {"image_id": image, "patient_id": mapping[image]}
                for image in sorted(mapping)
            ],
        }
        payload["patient_provenance"]["source_manifest_sha256"] = object_sha256(
            manifest
        )
        config["cluster_contract"]["patient_provenance_anchor"][
            "stage_manifest_sha256"
        ][stage] = object_sha256(manifest)
        output[stage] = manifest
    return output


@pytest.fixture()
def exact_inputs():
    config = _test_config()
    return config, _payload("dev_fit", 2), _payload("confirmation_locked", 3)


def _drop_orbit(payload, model, image, finding):
    payload["records"] = [
        row for row in payload["records"]
        if not (
            row["model"] == model and row["image_id"] == image
            and row["finding"] == finding
        )
    ]


def test_formal_config_freezes_exact_scientific_dimensions_and_20_60_quota():
    validate_config(FORMAL_CONFIG)
    assert FORMAL_CONFIG["stage_contract"]["dev_fit"]["exact_orbits_per_model_finding_vote"] == 20
    assert FORMAL_CONFIG["stage_contract"]["confirmation_locked"]["exact_orbits_per_model_finding_vote"] == 60
    assert len(FORMAL_CONFIG["models"]) == 2
    assert len(FORMAL_CONFIG["findings"]) * len(FORMAL_CONFIG["reader_vote_bins"]) == 16
    assert len(FORMAL_CONFIG["science_grid"]["renders"]) * len(FORMAL_CONFIG["science_grid"]["prompts"]) == 15
    assert FORMAL_CONFIG["cluster_contract"]["patient_provenance_anchor"][
        "stage_manifest_sha256"
    ] == {"dev_fit": None, "confirmation_locked": None}


def test_exact_closure_pairing_and_strictly_positive_shared_plan(exact_inputs):
    config, dev, confirmation = exact_inputs
    manifests = _external_patient_manifests(config, dev, confirmation)
    result = build_phase1_contract(
        dev, confirmation, config, external_patient_manifests=manifests
    )
    artifact = result.artifact
    assert artifact["authorized"] is False
    assert artifact["haar_implemented"] is False
    assert artifact["calibration_implemented"] is False
    assert artifact["closure"]["exact_model_orbit_pairing"] is True
    assert artifact["closure"]["exact_strata_per_model"] == 16
    assert artifact["cluster_identity"]["global_mode"] == "patient"
    assert np.all(result.multipliers > 0)
    assert np.allclose(result.multipliers.mean(axis=1), 1.0)
    audit = artifact["shared_multiplier_plan"]["audit"]
    assert audit["rejected_draws"] == 0
    assert audit["rejection_rate"] == 0
    assert audit["effective_cluster_count"]["min"] > 0
    assert audit["maximum_single_cluster_orbit_contribution"]["max"] < 0.95
    image = confirmation["records"][0]["image_id"]
    cluster = result.image_to_cluster[image]
    # There is exactly one multiplier column for an image across all four
    # findings and both models, not eight separately sampled pseudo-clusters.
    assert result.cluster_order.count(cluster) == 1
    assert audit["static_orbits_per_cluster"]["min"] == 8


@pytest.mark.parametrize("fault", ["missing_stratum", "wrong_finding", "single_model", "three_by_three", "quota_minus_one"])
def test_p0_1_exact_closure_faults_fail_closed(exact_inputs, fault):
    config, dev, confirmation = exact_inputs
    payload = copy.deepcopy(dev)
    if fault == "missing_stratum":
        payload["records"] = [
            row for row in payload["records"]
            if not (row["finding"] == EXPECTED_FINDINGS[0] and row["reader_votes"] == 0)
        ]
    elif fault == "wrong_finding":
        payload["records"][0]["finding"] = "not_frozen"
    elif fault == "single_model":
        payload["records"] = [row for row in payload["records"] if row["model"] == EXPECTED_MODELS[0]]
    elif fault == "three_by_three":
        payload["primary_renders"] = payload["primary_renders"][:3]
    elif fault == "quota_minus_one":
        target = payload["records"][0]
        for model in EXPECTED_MODELS:
            _drop_orbit(payload, model, target["image_id"], target["finding"])
    with pytest.raises(Phase1ContractError):
        build_phase1_contract(payload, confirmation, config)


def test_two_model_orbit_pairing_mismatch_fails_even_when_quota_is_unchanged(exact_inputs):
    config, dev, confirmation = exact_inputs
    payload = copy.deepcopy(dev)
    target = payload["records"][0]
    old_image = target["image_id"]
    finding = target["finding"]
    for row in payload["records"]:
        if row["model"] == EXPECTED_MODELS[1] and row["image_id"] == old_image and row["finding"] == finding:
            row["image_id"] = "model-specific-unpaired-image"
            row["patient_id"] = "model-specific-unpaired-patient"
    with pytest.raises(Phase1ContractError, match="exact orbit pairing"):
        build_phase1_contract(payload, confirmation, config)


def test_conflicting_patient_mapping_fails_but_partial_mapping_uses_global_image_mode(exact_inputs):
    config, dev, confirmation = exact_inputs
    conflict = copy.deepcopy(dev)
    conflict["records"][0]["patient_id"] = "contradictory-patient"
    with pytest.raises(Phase1ContractError, match="conflicting patient mapping"):
        build_phase1_contract(conflict, confirmation, config)

    partial = copy.deepcopy(dev)
    partial["records"][0].pop("patient_id")
    result = build_phase1_contract(partial, confirmation, config)
    assert result.artifact["cluster_identity"]["global_mode"] == "image"
    assert result.artifact["cluster_identity"]["no_mixed_fallback"] is True
    assert all(value.startswith("image:") for value in result.image_to_cluster.values())


def test_unverified_patient_ids_are_explicitly_non_authorizing_image_clusters(exact_inputs):
    config, dev, confirmation = exact_inputs
    dev.pop("patient_provenance")
    result = build_phase1_contract(dev, confirmation, config)
    identity = result.artifact["cluster_identity"]
    assert identity["global_mode"] == "image"
    gate = identity["patient_cluster_gate"]
    assert gate["patient_cluster_inference_eligible"] is False
    assert gate["status"] == "diagnostic_image_cluster_only_non_authorizing"
    assert gate["provenance"]["failure_reasons"] == [
        "dev_fit:no_trusted_patient_manifest_anchor_in_config",
        "confirmation_locked:no_trusted_patient_manifest_anchor_in_config",
    ]


def test_verified_single_patient_cluster_fails_frozen_per_stratum_minimum(exact_inputs):
    config, dev, confirmation = exact_inputs
    for payload in (dev, confirmation):
        for row in payload["records"]:
            row["patient_id"] = "one-patient"
    # Avoid triggering the cross-stage patient check first so this adversary
    # exercises the per-model/finding/vote patient-cluster gate directly.
    for row in confirmation["records"]:
        row["patient_id"] = "one-confirmation-patient"
    manifests = _external_patient_manifests(config, dev, confirmation)
    with pytest.raises(Phase1ContractError, match="patient-cluster minimum failed"):
        build_phase1_contract(
            dev, confirmation, config, external_patient_manifests=manifests
        )


def test_fabricated_patient_hash_and_image_as_patient_cannot_authorize(exact_inputs):
    config, dev, confirmation = exact_inputs
    for payload in (dev, confirmation):
        for row in payload["records"]:
            row["patient_id"] = row["image_id"]
        payload["patient_provenance"]["source_manifest_sha256"] = "f" * 64
    result = build_phase1_contract(dev, confirmation, config)
    gate = result.artifact["cluster_identity"]["patient_cluster_gate"]
    assert result.artifact["cluster_identity"]["global_mode"] == "image"
    assert gate["patient_cluster_inference_eligible"] is False
    assert gate["provenance"]["external_manifest_content_recomputed"] is False

    # Even a perfectly self-consistent, re-signed manifest cannot promote
    # patient mode when the pre-output config has no trusted source anchor.
    manifests = _external_patient_manifests(config, dev, confirmation)
    config["cluster_contract"]["patient_provenance_anchor"][
        "stage_manifest_sha256"
    ] = {"dev_fit": None, "confirmation_locked": None}
    with pytest.raises(Phase1ContractError, match="lacks a pre-output config anchor"):
        build_phase1_contract(
            dev, confirmation, config, external_patient_manifests=manifests
        )


def test_supplied_patient_manifest_content_or_seal_mismatch_fails_closed(exact_inputs):
    config, dev, confirmation = exact_inputs
    manifests = _external_patient_manifests(config, dev, confirmation)
    broken = copy.deepcopy(manifests)
    broken["dev_fit"]["records"][0]["patient_id"] = "tampered-patient"
    with pytest.raises(Phase1ContractError, match="does not exactly reproduce"):
        build_phase1_contract(
            dev, confirmation, config, external_patient_manifests=broken
        )

    broken = copy.deepcopy(manifests)
    dev["patient_provenance"]["source_manifest_sha256"] = "e" * 64
    with pytest.raises(Phase1ContractError, match="content seal mismatch"):
        build_phase1_contract(
            dev, confirmation, config, external_patient_manifests=broken
        )


def test_image_and_declared_patient_cross_split_leakage_fail_closed(exact_inputs):
    config, dev, confirmation = exact_inputs
    image_leak = copy.deepcopy(confirmation)
    old_image = image_leak["records"][0]["image_id"]
    dev_image = dev["records"][0]["image_id"]
    for row in image_leak["records"]:
        if row["image_id"] == old_image:
            row["image_id"] = dev_image
    with pytest.raises(Phase1ContractError, match="crosses dev/confirmation"):
        build_phase1_contract(dev, image_leak, config)

    patient_leak = copy.deepcopy(confirmation)
    old_patient = patient_leak["records"][0]["patient_id"]
    dev_patient = dev["records"][0]["patient_id"]
    for row in patient_leak["records"]:
        if row["patient_id"] == old_patient:
            row["patient_id"] = dev_patient
    with pytest.raises(Phase1ContractError, match="patient .* crosses"):
        build_phase1_contract(dev, patient_leak, config)


def test_multiplier_plan_is_deterministic_keyed_and_thresholds_fail_closed(exact_inputs):
    config, dev, confirmation = exact_inputs
    first = build_phase1_contract(dev, confirmation, config)
    second = build_phase1_contract(dev, confirmation, config)
    assert np.array_equal(first.multipliers, second.multipliers)
    assert first.artifact["shared_multiplier_plan"]["multiplier_sha256"] == second.artifact["shared_multiplier_plan"]["multiplier_sha256"]

    impossible = copy.deepcopy(config)
    impossible["bootstrap"]["minimum_ess_fraction"] = 0.999999
    with pytest.raises(Phase1ContractError, match="ESS"):
        build_phase1_contract(dev, confirmation, impossible)


def test_rng_trace_is_invariant_to_scores_and_vote_relabeling(exact_inputs):
    config, dev, confirmation = exact_inputs
    baseline = build_phase1_contract(dev, confirmation, config)

    changed_scores_dev = copy.deepcopy(dev)
    changed_scores_confirmation = copy.deepcopy(confirmation)
    for payload in (changed_scores_dev, changed_scores_confirmation):
        for row in payload["records"]:
            row.update(_score_fields(row["signed_score"] + 13.0))
    changed_scores = build_phase1_contract(
        changed_scores_dev, changed_scores_confirmation, config,
    )
    assert baseline.artifact["binding_sha256"] != changed_scores.artifact["binding_sha256"]
    assert baseline.artifact["rng_design_identity_sha256"] == changed_scores.artifact[
        "rng_design_identity_sha256"
    ]
    assert np.array_equal(baseline.multipliers, changed_scores.multipliers)

    relabeled_dev = copy.deepcopy(dev)
    relabeled_confirmation = copy.deepcopy(confirmation)
    for payload in (relabeled_dev, relabeled_confirmation):
        for row in payload["records"]:
            row["reader_votes"] = 3 - row["reader_votes"]
    relabeled = build_phase1_contract(relabeled_dev, relabeled_confirmation, config)
    assert baseline.artifact["binding_sha256"] != relabeled.artifact["binding_sha256"]
    assert baseline.artifact["rng_design_identity_sha256"] == relabeled.artifact[
        "rng_design_identity_sha256"
    ]
    assert np.array_equal(baseline.multipliers, relabeled.multipliers)
    assert baseline.artifact["shared_multiplier_plan"][
        "continuous_multiplier_trace"
    ] == relabeled.artifact["shared_multiplier_plan"]["continuous_multiplier_trace"]
