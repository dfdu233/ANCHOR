import copy
import hashlib
import inspect

import numpy as np
import pytest

from corrected_sgta.cecd_semantic_boundary_proximity_control_v1 import (
    BASE_FEATURES,
    ContractError,
    NON_AUTHORIZING,
    POSTHOC_DESCRIPTOR_FIELDS,
    _prejoint_features,
    apply_confirmation_control,
    fit_dev_control,
    object_sha256,
)


RENDERS = ("canonical", "style_b")
PROMPTS = ("neutral", "paraphrase_b")


def _sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def _embedding(distance, nuisance):
    # The frozen proxy direction is the x axis. Keep unit norm while adding an
    # off-axis coordinate so the substrate exercises its geometry, not logits.
    distance = float(np.clip(distance, -0.95, 0.95))
    y = np.sqrt(max(1.0 - distance**2, 0.0))
    if nuisance < 0:
        y = -y
    return [distance, float(y)]


def _payload(split, *, prefix, n=96):
    records = []
    for index in range(n):
        image = f"{prefix}-image-{index:03d}"
        patient = f"{prefix}-patient-{index:03d}"
        sign = 1.0 if index % 2 == 0 else -1.0
        crossing = index % 4 in (0, 1)
        d00 = sign * (0.62 + 0.03 * (index % 3))
        d10 = sign * (
            (0.12 + 0.01 * (index % 3)) if crossing
            else (0.51 + 0.01 * (index % 3))
        )
        d01 = sign * (
            (0.10 + 0.01 * (index % 3)) if crossing
            else (0.47 + 0.01 * (index % 3))
        )
        d11 = -sign * (0.08 + 0.01 * (index % 4)) if crossing else sign * (0.24 + 0.01 * (index % 4))
        values = {
            ("canonical", "neutral"): d00,
            ("style_b", "neutral"): d10,
            ("canonical", "paraphrase_b"): d01,
            ("style_b", "paraphrase_b"): d11,
        }
        for render, prompt in values:
            records.append({
                "model": "SyntheticVLM",
                "image_id": image,
                "patient_id": patient,
                "finding": "effusion",
                "render_id": render,
                "prompt_id": prompt,
                "source_image_sha256": _sha(f"source:{image}"),
                "transformed_image_sha256": _sha(f"transformed:{image}:{render}"),
                "exact_prompt_sha256": _sha(f"prompt:effusion:{prompt}"),
                "proposition_sha256": _sha("proposition:effusion"),
                "embedding": _embedding(values[(render, prompt)], -1 if index % 3 == 0 else 1),
            })
    return {
        "schema_version": "cecd-semantic-boundary-proximity-input-v2",
        "split": split,
        "source_manifest_split": "dev" if split == "dev_fit" else "confirmation",
        "source_manifest_sha256": _sha(f"manifest:{prefix}:{split}"),
        "frozen_before_reader_outcomes": True,
        "baseline_render": "canonical",
        "baseline_prompt": "neutral",
        "primary_renders": list(RENDERS),
        "primary_prompts": list(PROMPTS),
        "model_provenance": {
            "SyntheticVLM": {
                "checkpoint_sha256": _sha("checkpoint-v1"),
                "tokenizer_sha256": _sha("tokenizer-v1"),
                "processor_sha256": _sha("processor-v1"),
                "code_revision_sha256": _sha("model-code-v1"),
            }
        },
        "representation_spec": {
            "SyntheticVLM": {
                "layer_id": "fusion_block_12_output",
                "token_selector": "last_frozen_prefix_token",
                "pooling": "none",
                "normalization": "l2_at_analysis",
                "dtype": "float32",
                "extraction_code_sha256": _sha("extractor-v1"),
            },
        },
        "transform_spec_sha256": {name: _sha(f"transform:{name}") for name in RENDERS},
        "prompt_spec_sha256": {name: _sha(f"prompt-spec:{name}") for name in PROMPTS},
        "proxies": [{
            "model": "SyntheticVLM",
            "finding": "effusion",
            "proxy_source_sha256": _sha("text-proxy-source-v1"),
            "present_embeddings": [[1.0, 0.0], [0.98, 0.02]],
            "refuted_embeddings": [[-1.0, 0.0], [-0.98, 0.02]],
        }],
        "records": records,
    }


def _fit():
    return fit_dev_control(_payload("dev_fit", prefix="dev"))


def _embedding_3d(distance, nuisance):
    distance = float(np.clip(distance, -0.95, 0.95))
    radius = np.sqrt(max(1.0 - distance**2, 0.0))
    sign = -1.0 if nuisance < 0 else 1.0
    return [distance, sign * float(0.8 * radius), float(0.6 * radius)]


def _add_second_model(payload):
    payload["model_provenance"]["SecondVLM"] = {
        "checkpoint_sha256": _sha("second-checkpoint-v1"),
        "tokenizer_sha256": _sha("second-tokenizer-v1"),
        "processor_sha256": _sha("second-processor-v1"),
        "code_revision_sha256": _sha("second-model-code-v1"),
    }
    payload["representation_spec"]["SecondVLM"] = {
        "layer_id": "fusion_block_9_output",
        "token_selector": "mean_visual_prefix",
        "pooling": "mean",
        "normalization": "l2_at_analysis",
        "dtype": "bfloat16",
        "extraction_code_sha256": _sha("second-extractor-v1"),
    }
    payload["proxies"].append({
        "model": "SecondVLM",
        "finding": "effusion",
        "proxy_source_sha256": _sha("second-text-proxy-source-v1"),
        "present_embeddings": [[1.0, 0.0, 0.0], [0.98, 0.02, 0.0]],
        "refuted_embeddings": [[-1.0, 0.0, 0.0], [-0.98, 0.02, 0.0]],
    })
    second_records = copy.deepcopy(payload["records"])
    for row in second_records:
        row["model"] = "SecondVLM"
        distance = float(row["embedding"][0])
        row["embedding"] = _embedding_3d(
            distance, -1 if int(row["image_id"].rsplit("-", 1)[1]) % 5 == 0 else 1
        )
    payload["records"].extend(second_records)
    return payload


def test_dev_fit_and_held_out_apply_are_reader_free_continuous_and_non_authorizing():
    bundle = _fit()
    result = apply_confirmation_control(
        _payload("confirmation_locked", prefix="confirmation"), bundle
    )
    assert bundle["status"] == NON_AUTHORIZING
    assert result["status"] == NON_AUTHORIZING
    assert bundle["authorized"] is False and result["authorized"] is False
    assert bundle["reader_or_clinical_outcomes_used"] is False
    assert result["reader_or_clinical_outcomes_used"] is False
    assert bundle["binary_decision_or_threshold"] is None
    assert result["binary_decision_or_threshold"] is None
    assert result["fit_or_refit_on_confirmation"] is False
    assert result["held_out_orbit_evaluation"] is True
    assert result["reader_free_metrics"]["overall"]["auroc"] > 0.8
    assert set(result["reader_free_metrics"]["by_model_finding"]) == {
        "SyntheticVLM\x1feffusion"
    }
    assert set(result["reader_free_metrics"]["by_transform_pair"]) == {
        "style_b\x1fparaphrase_b"
    }
    assert len(result["scored_product_comparisons"]) == 96
    first = result["scored_product_comparisons"][0]
    assert set(first["features"]) == set(BASE_FEATURES)
    assert set(first["posthoc_joint_endpoint_descriptor"]) == set(
        POSTHOC_DESCRIPTOR_FIELDS
    )
    assert 0.0 <= first["boundary_proximity_risk"] <= 1.0
    assert "reader_votes" not in str(result).lower()


def test_api_and_artifacts_expose_no_tunable_decision_threshold():
    assert "threshold" not in inspect.signature(fit_dev_control).parameters
    assert "threshold" not in inspect.signature(apply_confirmation_control).parameters
    bundle = _fit()
    assert bundle["threshold_selection"].startswith("prohibited")
    fixed = bundle["fixed_fits_by_model_finding"]["SyntheticVLM\x1feffusion"]
    assert fixed["family"] == "l2_logistic_regression"
    assert fixed["C"] == 1.0
    assert fixed["fit_scope"] == "model_by_finding"
    assert set(BASE_FEATURES).isdisjoint(POSTHOC_DESCRIPTOR_FIELDS)
    assert set(inspect.signature(_prejoint_features).parameters) == {
        "h00", "h10", "h01", "d00", "d10", "d01"
    }
    assert bundle["predictor_covariate_contract"]["allowed_cells"] == [
        "h00", "h10", "h01"
    ]
    assert bundle["predictor_covariate_contract"]["prohibited_predictor_cell"] == "h11"


def test_reader_clinical_and_outcome_fields_fail_closed_at_any_depth():
    for key, value in (
        ("reader_votes", 2),
        ("ground_truth", "present"),
        ("pael", 0.4),
        ("threshold", 0.5),
    ):
        payload = _payload("dev_fit", prefix=f"forbidden-{key}")
        payload["records"][0][key] = value
        with pytest.raises(ContractError, match="reader/outcome field is prohibited"):
            fit_dev_control(payload)


def test_dev_confirmation_image_patient_and_byte_overlap_fail_closed():
    dev = _payload("dev_fit", prefix="dev")
    bundle = fit_dev_control(dev)
    for mode in ("identity", "patient", "source", "transformed"):
        confirmation = _payload("confirmation_locked", prefix=f"confirmation-{mode}")
        target_image = confirmation["records"][0]["image_id"]
        dev_first = dev["records"][0]
        for row in confirmation["records"]:
            if row["image_id"] != target_image:
                continue
            if mode == "identity":
                row["image_id"] = dev_first["image_id"]
            elif mode == "patient":
                row["patient_id"] = dev_first["patient_id"]
            elif mode == "source":
                row["source_image_sha256"] = dev_first["source_image_sha256"]
            else:
                matching = next(
                    item for item in dev["records"]
                    if item["render_id"] == row["render_id"]
                )
                row["transformed_image_sha256"] = matching["transformed_image_sha256"]
        expected = {
            "identity": "images overlap",
            "patient": "patients overlap",
            "source": "source_image_sha256 overlap",
            "transformed": "transformed_image_sha256 overlap",
        }[mode]
        with pytest.raises(ContractError, match=expected):
            apply_confirmation_control(confirmation, bundle)


def test_duplicate_cell_and_duplicate_image_bytes_fail_closed():
    duplicate_cell = _payload("dev_fit", prefix="duplicate-cell")
    duplicate_cell["records"].append(copy.deepcopy(duplicate_cell["records"][0]))
    with pytest.raises(ContractError, match="duplicate product cell"):
        fit_dev_control(duplicate_cell)

    duplicate_bytes = _payload("dev_fit", prefix="duplicate-bytes")
    first_image = duplicate_bytes["records"][0]["image_id"]
    second_image = next(row["image_id"] for row in duplicate_bytes["records"] if row["image_id"] != first_image)
    first_source = duplicate_bytes["records"][0]["source_image_sha256"]
    for row in duplicate_bytes["records"]:
        if row["image_id"] == second_image:
            row["source_image_sha256"] = first_source
    with pytest.raises(ContractError, match="duplicate source image bytes"):
        fit_dev_control(duplicate_bytes)


def test_incomplete_orbit_and_cross_model_orbit_drift_fail_closed():
    incomplete = _payload("dev_fit", prefix="incomplete")
    del incomplete["records"][0]
    with pytest.raises(ContractError, match="incomplete factorial orbit"):
        fit_dev_control(incomplete)

    drift = _payload("dev_fit", prefix="model-drift")
    drift["model_provenance"]["SecondVLM"] = copy.deepcopy(
        drift["model_provenance"]["SyntheticVLM"]
    )
    drift["representation_spec"]["SecondVLM"] = copy.deepcopy(
        drift["representation_spec"]["SyntheticVLM"]
    )
    drift["proxies"].append({**copy.deepcopy(drift["proxies"][0]), "model": "SecondVLM"})
    duplicate = copy.deepcopy(drift["records"][:-4])
    for row in duplicate:
        row["model"] = "SecondVLM"
    drift["records"].extend(duplicate)
    with pytest.raises(ContractError, match="models do not share exact held-out orbit identities"):
        fit_dev_control(drift)


def test_bundle_tamper_and_checkpoint_proxy_or_transform_drift_fail_closed():
    bundle = _fit()
    confirmation = _payload("confirmation_locked", prefix="confirmation")

    tampered = copy.deepcopy(bundle)
    tampered["fixed_fits_by_model_finding"]["SyntheticVLM\x1feffusion"][
        "coefficient"
    ][0] += 0.1
    with pytest.raises(ContractError, match="modified after dev freeze"):
        apply_confirmation_control(confirmation, tampered)

    for mode in ("checkpoint", "proxy", "transform", "extractor"):
        changed = copy.deepcopy(confirmation)
        if mode == "checkpoint":
            changed["model_provenance"]["SyntheticVLM"]["checkpoint_sha256"] = _sha("changed")
        elif mode == "proxy":
            changed["proxies"][0]["present_embeddings"][0] = [0.8, 0.2]
        elif mode == "transform":
            changed["transform_spec_sha256"]["style_b"] = _sha("changed")
        else:
            changed["representation_spec"]["SyntheticVLM"]["extraction_code_sha256"] = _sha("changed")
        with pytest.raises(ContractError, match="drift"):
            apply_confirmation_control(changed, bundle)


def test_confirmation_cannot_be_refit_and_same_manifest_cannot_be_reused():
    dev = _payload("dev_fit", prefix="dev")
    bundle = fit_dev_control(dev)
    confirmation = _payload("confirmation_locked", prefix="confirmation")
    with pytest.raises(ContractError, match="apply-only split violation"):
        fit_dev_control(confirmation)

    confirmation["source_manifest_sha256"] = dev["source_manifest_sha256"]
    with pytest.raises(ContractError, match="same source manifest"):
        apply_confirmation_control(confirmation, bundle)


def test_dev_requires_both_generic_crossing_classes_without_outcome_repair():
    payload = _payload("dev_fit", prefix="one-class")
    # Replace every joint cell by its clean polarity, leaving no generic
    # boundary crossing. The fixed family fails rather than tuning a cutoff.
    clean_by_image = {
        row["image_id"]: row["embedding"]
        for row in payload["records"]
        if row["render_id"] == "canonical" and row["prompt_id"] == "neutral"
    }
    for row in payload["records"]:
        if row["render_id"] == "style_b" and row["prompt_id"] == "paraphrase_b":
            row["embedding"] = clean_by_image[row["image_id"]]
    with pytest.raises(ContractError, match="patient clusters per class"):
        fit_dev_control(payload)


def test_bundle_and_result_seals_are_deterministic_for_identical_input():
    dev = _payload("dev_fit", prefix="dev")
    first = fit_dev_control(dev)
    second = fit_dev_control(copy.deepcopy(dev))
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert object_sha256({k: v for k, v in first.items() if k != "bundle_sha256"}) == first["bundle_sha256"]

    confirmation = _payload("confirmation_locked", prefix="confirmation")
    result_a = apply_confirmation_control(confirmation, first)
    result_b = apply_confirmation_control(copy.deepcopy(confirmation), second)
    assert result_a["result_sha256"] == result_b["result_sha256"]


def test_h11_mediator_changes_cannot_change_covariates_or_predictions():
    """Adversarially move h11 while holding every pre-joint cell fixed."""
    bundle = _fit()
    original = _payload("confirmation_locked", prefix="mediator-original")
    changed = copy.deepcopy(original)
    for row in changed["records"]:
        if row["render_id"] != "style_b" or row["prompt_id"] != "paraphrase_b":
            continue
        distance = float(row["embedding"][0])
        # Preserve the endpoint polarity but rotate the off-axis h11 component.
        row["embedding"] = _embedding(distance, -1 if row["embedding"][1] >= 0 else 1)
    first = apply_confirmation_control(original, bundle)
    second = apply_confirmation_control(changed, bundle)
    first_rows = first["scored_product_comparisons"]
    second_rows = second["scored_product_comparisons"]
    assert [row["features"] for row in first_rows] == [
        row["features"] for row in second_rows
    ]
    assert [row["boundary_proximity_risk"] for row in first_rows] == [
        row["boundary_proximity_risk"] for row in second_rows
    ]
    assert [row["generic_joint_boundary_crossing_endpoint"] for row in first_rows] == [
        row["generic_joint_boundary_crossing_endpoint"] for row in second_rows
    ]
    assert any(
        left["posthoc_joint_endpoint_descriptor"]
        != right["posthoc_joint_endpoint_descriptor"]
        for left, right in zip(first_rows, second_rows)
    )


def test_h11_endpoint_flip_changes_target_but_not_predicted_risk():
    bundle = _fit()
    original = _payload("confirmation_locked", prefix="endpoint-original")
    flipped = copy.deepcopy(original)
    for row in flipped["records"]:
        if row["render_id"] == "style_b" and row["prompt_id"] == "paraphrase_b":
            row["embedding"][0] *= -1.0
    first = apply_confirmation_control(original, bundle)
    second = apply_confirmation_control(flipped, bundle)
    first_rows = first["scored_product_comparisons"]
    second_rows = second["scored_product_comparisons"]
    assert [row["boundary_proximity_risk"] for row in first_rows] == [
        row["boundary_proximity_risk"] for row in second_rows
    ]
    assert all(
        left["generic_joint_boundary_crossing_endpoint"]
        != right["generic_joint_boundary_crossing_endpoint"]
        for left, right in zip(first_rows, second_rows)
    )


def test_predictor_contract_rejects_mediator_feature_even_with_valid_reseal():
    bundle = _fit()
    tampered = copy.deepcopy(bundle)
    tampered["predictor_covariate_contract"]["allowed_cells"].append("h11")
    del tampered["bundle_sha256"]
    tampered["bundle_sha256"] = object_sha256(tampered)
    with pytest.raises(ContractError, match="pre-joint predictor covariate contract"):
        apply_confirmation_control(
            _payload("confirmation_locked", prefix="mediator-contract"), tampered
        )


def test_models_may_use_distinct_representation_specs_and_dimensions_but_fit_separately():
    dev = _add_second_model(_payload("dev_fit", prefix="two-model-dev"))
    bundle = fit_dev_control(dev)
    assert set(bundle["fixed_fits_by_model_finding"]) == {
        "SyntheticVLM\x1feffusion", "SecondVLM\x1feffusion"
    }
    assert bundle["frozen_contract"]["representation_spec"]["SyntheticVLM"] != (
        bundle["frozen_contract"]["representation_spec"]["SecondVLM"]
    )
    confirmation = _add_second_model(
        _payload("confirmation_locked", prefix="two-model-confirmation")
    )
    result = apply_confirmation_control(confirmation, bundle)
    assert set(result["reader_free_metrics"]["by_model_finding"]) == {
        "SyntheticVLM\x1feffusion", "SecondVLM\x1feffusion"
    }


def test_class_floor_counts_patient_clusters_not_comparison_rows():
    payload = _payload("dev_fit", prefix="cluster-floor", n=12)
    payload["primary_renders"].append("style_c")
    payload["primary_prompts"].append("paraphrase_c")
    payload["transform_spec_sha256"]["style_c"] = _sha("transform:style_c")
    payload["prompt_spec_sha256"]["paraphrase_c"] = _sha("prompt-spec:paraphrase_c")
    grouped = {}
    for row in payload["records"]:
        grouped[(row["image_id"], row["render_id"], row["prompt_id"])] = row
    additions = []
    for image in sorted({row["image_id"] for row in payload["records"]}):
        for render, prompt, source_render, source_prompt in (
            ("style_c", "neutral", "style_b", "neutral"),
            ("canonical", "paraphrase_c", "canonical", "paraphrase_b"),
            ("style_b", "paraphrase_c", "style_b", "paraphrase_b"),
            ("style_c", "paraphrase_b", "style_b", "paraphrase_b"),
            ("style_c", "paraphrase_c", "style_b", "paraphrase_b"),
        ):
            row = copy.deepcopy(grouped[(image, source_render, source_prompt)])
            row["render_id"] = render
            row["prompt_id"] = prompt
            row["transformed_image_sha256"] = _sha(f"transformed:{image}:{render}")
            row["exact_prompt_sha256"] = _sha(f"prompt:effusion:{prompt}")
            additions.append(row)
    payload["records"].extend(additions)
    # There are 24 comparison rows in each endpoint class, but only six
    # distinct patients in each.  Row multiplication must not pass admission.
    with pytest.raises(ContractError, match="distinct patient clusters per class"):
        fit_dev_control(payload)
