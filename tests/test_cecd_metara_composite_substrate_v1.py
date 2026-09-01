from __future__ import annotations

import copy
import math

import pytest

from anchor.corrected_sgta.validate_cecd_metara_composite_substrate_v1 import (
    ADMISSION_SCHEMA_VERSION,
    ANSWER_SPACE,
    REAL_ADAPTER_INPUT_SCHEMA,
    SCHEMA_VERSION,
    SCIENTIFIC_STATUS,
    TRANSFORM_REGISTRY_SCHEMA_VERSION,
    SubstrateError,
    object_sha256,
    text_sha256,
    validate_and_extract,
)


MODELS = ["model-a", "model-b"]


def digest(seed: str) -> str:
    return text_sha256(seed)


def proposition(finding: str) -> dict[str, str]:
    base = {
        "canonical_claim_id": f"claim:{finding}",
        "finding": finding,
        "answer_space": ANSWER_SPACE,
        "supported_token": "present",
        "refuted_token": "absent",
        "undetermined_token": "uncertain",
    }
    return {**base, "mapping_sha256": object_sha256(base)}


def transform_registry_entry(axis: str, family: str) -> dict[str, str]:
    canonical = {
        "axis": axis,
        "family": family,
        "implementation_name": f"fixture.{axis}.{family}",
        "implementation_version": "1.0.0",
        "implementation_sha256": digest(f"implementation:{axis}:{family}"),
    }
    return {**canonical, "entry_sha256": object_sha256(canonical)}


TRANSFORM_ENTRIES = [
    transform_registry_entry("image", "local_contrast"),
    transform_registry_entry("question", "active_passive_paraphrase"),
    transform_registry_entry("image", "background_style"),
    transform_registry_entry("question", "clinical_synonym_paraphrase"),
]


def transform_entry(axis: str, family: str) -> dict[str, str]:
    return next(
        entry
        for entry in TRANSFORM_ENTRIES
        if entry["axis"] == axis and entry["family"] == family
    )


def transform_registry() -> dict:
    entries = sorted(
        copy.deepcopy(TRANSFORM_ENTRIES),
        key=lambda entry: (entry["axis"], entry["family"]),
    )
    canonical = {
        "schema_version": TRANSFORM_REGISTRY_SCHEMA_VERSION,
        "entries": entries,
    }
    return {**canonical, "registry_sha256": object_sha256(canonical)}


def rehash_admission(unit_row: dict) -> None:
    artifact = unit_row["admission_artifact"]
    artifact["artifact_sha256"] = object_sha256(
        {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    )


def cell(role: str, image_seed: str, prompt: str, logits: tuple[float, ...]) -> dict:
    return {
        "role": role,
        "image_sha256": digest(image_seed),
        "prompt_text": prompt,
        "prompt_sha256": text_sha256(prompt),
        "tristate_logits": dict(zip(("support", "refute", "undetermined"), logits)),
    }


def unit(
    *,
    stage: str,
    model: str,
    patient_id: str,
    image_id: str,
    image_family: str,
    question_family: str,
    image_transform_id: str,
    question_transform_id: str,
) -> dict:
    canonical_prompt = "Is pleural effusion present?"
    transformed_prompt = "Does this radiograph show pleural effusion?"
    clean_logits = (3.0, 0.0, -1.0)
    # Both single-axis cells remain support; the joint cell flips to refute.
    model_offset = 0.1 if model == "model-b" else 0.0
    image_binding = transform_entry("image", image_family)
    question_binding = transform_entry("question", question_family)
    image_transform = {
        "id": image_transform_id,
        "family": image_family,
        "implementation_sha256": image_binding["implementation_sha256"],
        "registry_entry_sha256": image_binding["entry_sha256"],
        "parameters": {"strength": 0.2},
    }
    question_transform = {
        "id": question_transform_id,
        "family": question_family,
        "implementation_sha256": question_binding["implementation_sha256"],
        "registry_entry_sha256": question_binding["entry_sha256"],
        "parameters": {"template": 2},
    }
    cells = {
        "clean": cell("clean", f"{stage}:source", canonical_prompt, clean_logits),
        "image_only": cell(
            "image_only",
            f"{stage}:transformed",
            canonical_prompt,
            (2.2 + model_offset, 0.5, -0.5),
        ),
        "question_only": cell(
            "question_only",
            f"{stage}:source",
            transformed_prompt,
            (2.0 + model_offset, 0.7, -0.4),
        ),
        "joint": cell(
            "joint",
            f"{stage}:transformed",
            transformed_prompt,
            (0.0, 2.5 + model_offset, -0.5),
        ),
    }
    admission_base = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "artifact_id": f"admission:{stage}:{image_transform_id}:{question_transform_id}",
        "patient_id": patient_id,
        "image_id": image_id,
        "finding": "pleural_effusion",
        "canonical_claim_id": "claim:pleural_effusion",
        "source_image_sha256": cells["clean"]["image_sha256"],
        "transformed_image_sha256": cells["image_only"]["image_sha256"],
        "canonical_prompt_sha256": cells["clean"]["prompt_sha256"],
        "transformed_prompt_sha256": cells["question_only"]["prompt_sha256"],
        "image_transform_registry_entry_sha256": image_binding["entry_sha256"],
        "question_transform_registry_entry_sha256": question_binding["entry_sha256"],
        "pathology_preservation": {
            "decision": "preserved",
            "review_blinded_to_model_outputs": True,
            "review_protocol_sha256": digest("pathology-review-v1"),
            "finding_evidence_preserved": True,
            "anatomy_visibility_preserved": True,
            "attribute_visibility_preserved": True,
        },
        "proposition_preservation": {
            "decision": "preserved",
            "review_blinded_to_model_outputs": True,
            "review_protocol_sha256": digest("proposition-review-v1"),
            "canonical_proposition_preserved": True,
            "speech_act_preserved": True,
            "certainty_demand_preserved": True,
            "answer_space_preserved": True,
            "output_grammar_preserved": True,
        },
    }
    return {
        "unit_id": f"{stage}:{model}:{image_transform_id}:{question_transform_id}",
        "model": model,
        "patient_id": patient_id,
        "image_id": image_id,
        "finding": "pleural_effusion",
        "image_transform": image_transform,
        "question_transform": question_transform,
        "proposition_mapping": proposition("pleural_effusion"),
        "admission_artifact": {
            **admission_base,
            "artifact_sha256": object_sha256(admission_base),
        },
        "cells": cells,
    }


def valid_payload() -> dict:
    dev_units = [
        unit(
            stage="dev",
            model=model,
            patient_id="dev-patient",
            image_id="dev-image",
            image_family="local_contrast",
            question_family="active_passive_paraphrase",
            image_transform_id="dev-render-1",
            question_transform_id="dev-question-1",
        )
        for model in MODELS
    ]
    confirmation_units = [
        unit(
            stage="confirmation",
            model=model,
            patient_id="confirmation-patient",
            image_id="confirmation-image",
            image_family="background_style",
            question_family="clinical_synonym_paraphrase",
            image_transform_id="confirmation-render-1",
            question_transform_id="confirmation-question-1",
        )
        for model in MODELS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_status": SCIENTIFIC_STATUS,
        "authorized": False,
        "execution_scope": {
            "synthetic_only": True,
            "real_results_read": False,
            "gpu_used": False,
            "reader_labels_used": False,
        },
        "models": MODELS,
        "findings": ["pleural_effusion"],
        "finding_stage_quotas": {
            "dev_fit": {"pleural_effusion": 1},
            "confirmation_locked": {"pleural_effusion": 1},
        },
        "transform_registry": transform_registry(),
        "real_adapter_input_schema": copy.deepcopy(REAL_ADAPTER_INPUT_SCHEMA),
        "stages": {
            "dev_fit": {
                "declared_image_transform_families": ["local_contrast"],
                "declared_question_transform_families": [
                    "active_passive_paraphrase"
                ],
                "units": dev_units,
            },
            "confirmation_locked": {
                "declared_image_transform_families": ["background_style"],
                "declared_question_transform_families": [
                    "clinical_synonym_paraphrase"
                ],
                "units": confirmation_units,
            },
        },
    }


def test_valid_substrate_is_exact_paired_label_free_and_non_authorizing() -> None:
    result = validate_and_extract(valid_payload())

    assert result["authorized"] is False
    assert result["gpu_authorized"] is False
    assert result["real_results_read"] is False
    assert result["reader_labels_used"] is False
    assert result["counts"]["dev_fit"] == {
        "units": 2,
        "paired_scientific_orbits": 1,
        "unique_patients": 1,
        "unique_images_by_finding": {"pleural_effusion": 1},
    }
    assert result["counts"]["confirmation_locked"] == {
        "units": 2,
        "paired_scientific_orbits": 1,
        "unique_patients": 1,
        "unique_images_by_finding": {"pleural_effusion": 1},
    }
    assert all(
        row["features"]["singles_stable_but_joint_flips"]
        for row in result["feature_rows"]
    )
    assert all(
        row["features"]["probability_interaction_l2"] > 0
        for row in result["feature_rows"]
    )
    assert "not a faithful MetaRA reproduction" in result["paper_faithfulness_claim"]
    assert result["scientific_interpretation"] == {
        "interaction_role": "collision_control_readout_only",
        "standalone_metric_novelty": False,
        "standalone_mechanism_evidence": False,
        "pael_absorption_established": False,
        "absorption_requires_separate_preregistered_external_analysis": True,
    }
    assert result["real_adapter_contract"]["execution_enabled"] is False
    assert result["real_adapter_contract"]["adapter_implemented"] is False
    assert result == validate_and_extract(valid_payload())


def test_missing_fourth_cell_fails_closed() -> None:
    payload = valid_payload()
    del payload["stages"]["dev_fit"]["units"][0]["cells"]["joint"]
    with pytest.raises(SubstrateError, match="cells keys differ"):
        validate_and_extract(payload)


def test_single_axis_and_joint_must_reuse_exact_inputs() -> None:
    payload = valid_payload()
    target = payload["stages"]["dev_fit"]["units"][0]["cells"]["joint"]
    target["image_sha256"] = digest("wrong-joint-image")
    with pytest.raises(SubstrateError, match="does not reuse image-only bytes"):
        validate_and_extract(payload)

    payload = valid_payload()
    target = payload["stages"]["dev_fit"]["units"][0]["cells"]["image_only"]
    target["prompt_text"] = "A different question"
    target["prompt_sha256"] = text_sha256(target["prompt_text"])
    with pytest.raises(SubstrateError, match="image-only cell changed the question"):
        validate_and_extract(payload)


def test_missing_or_extra_model_breaks_exact_orbit_pairing() -> None:
    payload = valid_payload()
    payload["stages"]["confirmation_locked"]["units"].pop()
    with pytest.raises(SubstrateError, match="not paired exactly"):
        validate_and_extract(payload)


def test_cross_model_metadata_mismatch_is_rejected() -> None:
    payload = valid_payload()
    second = payload["stages"]["dev_fit"]["units"][1]
    second["admission_artifact"]["artifact_id"] = "different-admission"
    admission = second["admission_artifact"]
    admission["artifact_sha256"] = object_sha256(
        {key: value for key, value in admission.items() if key != "artifact_sha256"}
    )
    with pytest.raises(
        SubstrateError, match="canonical anchor|input metadata differs across models"
    ):
        validate_and_extract(payload)


@pytest.mark.parametrize("axis", ["image", "question"])
def test_confirmation_transform_families_must_be_held_out(axis: str) -> None:
    payload = valid_payload()
    if axis == "image":
        family = "local_contrast"
        payload["stages"]["confirmation_locked"][
            "declared_image_transform_families"
        ] = [family]
        for row in payload["stages"]["confirmation_locked"]["units"]:
            row["image_transform"]["family"] = family
            binding = transform_entry("image", family)
            row["image_transform"]["implementation_sha256"] = binding[
                "implementation_sha256"
            ]
            row["image_transform"]["registry_entry_sha256"] = binding[
                "entry_sha256"
            ]
            row["admission_artifact"][
                "image_transform_registry_entry_sha256"
            ] = binding["entry_sha256"]
            rehash_admission(row)
    else:
        family = "active_passive_paraphrase"
        payload["stages"]["confirmation_locked"][
            "declared_question_transform_families"
        ] = [family]
        for row in payload["stages"]["confirmation_locked"]["units"]:
            row["question_transform"]["family"] = family
            binding = transform_entry("question", family)
            row["question_transform"]["implementation_sha256"] = binding[
                "implementation_sha256"
            ]
            row["question_transform"]["registry_entry_sha256"] = binding[
                "entry_sha256"
            ]
            row["admission_artifact"][
                "question_transform_registry_entry_sha256"
            ] = binding["entry_sha256"]
            rehash_admission(row)
    with pytest.raises(SubstrateError, match="transform families are not held out"):
        validate_and_extract(payload)


def test_prompt_proposition_and_outcome_tampering_fail_closed() -> None:
    payload = valid_payload()
    payload["stages"]["dev_fit"]["units"][0]["cells"]["clean"][
        "prompt_sha256"
    ] = digest("wrong")
    with pytest.raises(SubstrateError, match="prompt_sha256 mismatch"):
        validate_and_extract(payload)

    payload = valid_payload()
    payload["stages"]["dev_fit"]["units"][0]["proposition_mapping"][
        "supported_token"
    ] = "yes"
    with pytest.raises(SubstrateError, match="mapping_sha256 mismatch"):
        validate_and_extract(payload)

    payload = valid_payload()
    payload["stages"]["dev_fit"]["units"][0]["reader_votes"] = 3
    with pytest.raises(SubstrateError, match="forbidden outcome key"):
        validate_and_extract(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (("synthetic_only", False), ("real_results_read", True), ("gpu_used", True)),
)
def test_execution_scope_cannot_expand(field: str, value: bool) -> None:
    payload = valid_payload()
    payload["execution_scope"][field] = value
    with pytest.raises(SubstrateError, match="synthetic CPU-only label-free"):
        validate_and_extract(payload)


def test_stage_images_and_source_bytes_must_be_disjoint() -> None:
    payload = valid_payload()
    for row in payload["stages"]["confirmation_locked"]["units"]:
        row["image_id"] = "dev-image"
        row["admission_artifact"]["image_id"] = "dev-image"
        rehash_admission(row)
    with pytest.raises(SubstrateError, match="image IDs cross"):
        validate_and_extract(payload)

    payload = valid_payload()
    dev_source = payload["stages"]["dev_fit"]["units"][0]["cells"]["clean"][
        "image_sha256"
    ]
    for row in payload["stages"]["confirmation_locked"]["units"]:
        row["cells"]["clean"]["image_sha256"] = dev_source
        row["cells"]["question_only"]["image_sha256"] = dev_source
        row["admission_artifact"]["source_image_sha256"] = dev_source
        rehash_admission(row)
    with pytest.raises(SubstrateError, match="source image bytes cross"):
        validate_and_extract(payload)


def test_transform_id_cannot_change_parameters_within_stage() -> None:
    payload = valid_payload()
    payload["stages"]["dev_fit"]["units"][1]["image_transform"]["parameters"] = {
        "strength": 0.4
    }
    with pytest.raises(SubstrateError, match="changes metadata"):
        validate_and_extract(payload)


def test_same_transform_application_must_reuse_exact_bytes_and_prompt() -> None:
    payload = valid_payload()
    extra = copy.deepcopy(payload["stages"]["dev_fit"]["units"])
    for row in extra:
        row["unit_id"] += ":second-question"
        row["question_transform"]["id"] = "dev-question-2"
        row["admission_artifact"][
            "artifact_id"
        ] = "admission:dev:dev-render-1:dev-question-2"
        for role in ("question_only", "joint"):
            row["cells"][role]["prompt_text"] = "Is there evidence of pleural effusion?"
            row["cells"][role]["prompt_sha256"] = text_sha256(
                row["cells"][role]["prompt_text"]
            )
        row["admission_artifact"]["transformed_prompt_sha256"] = row["cells"][
            "question_only"
        ]["prompt_sha256"]
        rehash_admission(row)
    payload["stages"]["dev_fit"]["units"].extend(extra)
    # One model now claims that the same image transformation produced new bytes.
    for changed in payload["stages"]["dev_fit"]["units"][-2:]:
        changed["cells"]["image_only"]["image_sha256"] = digest(
            "inconsistent-transformed-bytes"
        )
        changed["cells"]["joint"]["image_sha256"] = digest(
            "inconsistent-transformed-bytes"
        )
        changed["admission_artifact"]["transformed_image_sha256"] = changed[
            "cells"
        ]["image_only"]["image_sha256"]
        rehash_admission(changed)
    with pytest.raises(SubstrateError, match="image transform application.*not exact"):
        validate_and_extract(payload)


def test_transform_pairs_must_form_exact_cartesian_product() -> None:
    payload = valid_payload()
    original = payload["stages"]["dev_fit"]["units"]
    additions = []
    for model in MODELS:
        additions.append(
            unit(
                stage="dev",
                model=model,
                patient_id="dev-patient",
                image_id="dev-image",
                image_family="local_contrast",
                question_family="active_passive_paraphrase",
                image_transform_id="dev-render-2",
                question_transform_id="dev-question-1",
            )
        )
        additions.append(
            unit(
                stage="dev",
                model=model,
                patient_id="dev-patient",
                image_id="dev-image",
                image_family="local_contrast",
                question_family="active_passive_paraphrase",
                image_transform_id="dev-render-1",
                question_transform_id="dev-question-2",
            )
        )
    original.extend(additions)
    with pytest.raises(SubstrateError, match="lacks the exact image-by-question"):
        validate_and_extract(payload)


def test_family_rename_cannot_hide_reused_implementation() -> None:
    payload = valid_payload()
    registry = payload["transform_registry"]
    dev_entry = next(
        entry
        for entry in registry["entries"]
        if entry["axis"] == "image" and entry["family"] == "local_contrast"
    )
    confirmation_entry = next(
        entry
        for entry in registry["entries"]
        if entry["axis"] == "image" and entry["family"] == "background_style"
    )
    confirmation_entry["family"] = "renamed_background"
    confirmation_entry["implementation_sha256"] = dev_entry[
        "implementation_sha256"
    ]
    confirmation_entry["entry_sha256"] = object_sha256(
        {
            key: value
            for key, value in confirmation_entry.items()
            if key != "entry_sha256"
        }
    )
    canonical_registry = {
        "schema_version": registry["schema_version"],
        "entries": sorted(
            registry["entries"], key=lambda entry: (entry["axis"], entry["family"])
        ),
    }
    registry["entries"] = canonical_registry["entries"]
    registry["registry_sha256"] = object_sha256(canonical_registry)
    payload["stages"]["confirmation_locked"][
        "declared_image_transform_families"
    ] = ["renamed_background"]
    for row in payload["stages"]["confirmation_locked"]["units"]:
        row["image_transform"]["family"] = "renamed_background"
        row["image_transform"]["implementation_sha256"] = confirmation_entry[
            "implementation_sha256"
        ]
        row["image_transform"]["registry_entry_sha256"] = confirmation_entry[
            "entry_sha256"
        ]
        row["admission_artifact"][
            "image_transform_registry_entry_sha256"
        ] = confirmation_entry["entry_sha256"]
        rehash_admission(row)
    with pytest.raises(SubstrateError, match="renamed across families|family rename"):
        validate_and_extract(payload)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("pathology_preservation", "finding_evidence_preserved"),
        ("proposition_preservation", "speech_act_preserved"),
    ],
)
def test_admission_content_is_verified_even_after_rehash(
    section: str, field: str
) -> None:
    payload = valid_payload()
    row = payload["stages"]["dev_fit"]["units"][0]
    row["admission_artifact"][section][field] = False
    rehash_admission(row)
    with pytest.raises(SubstrateError, match=f"{field} must be true"):
        validate_and_extract(payload)


def test_admission_cannot_be_swapped_between_orbits_after_valid_rehash() -> None:
    payload = valid_payload()
    row = payload["stages"]["dev_fit"]["units"][0]
    row["admission_artifact"]["canonical_claim_id"] = "claim:cardiomegaly"
    rehash_admission(row)
    with pytest.raises(SubstrateError, match="does not bind the scientific orbit"):
        validate_and_extract(payload)


def test_patient_split_is_independent_of_image_id_split() -> None:
    payload = valid_payload()
    for row in payload["stages"]["confirmation_locked"]["units"]:
        row["patient_id"] = "dev-patient"
        row["admission_artifact"]["patient_id"] = "dev-patient"
        rehash_admission(row)
    with pytest.raises(SubstrateError, match="patient IDs cross"):
        validate_and_extract(payload)


def test_each_finding_must_meet_each_stage_quota() -> None:
    payload = valid_payload()
    payload["finding_stage_quotas"]["confirmation_locked"][
        "pleural_effusion"
    ] = 2
    with pytest.raises(SubstrateError, match="quota requires 2"):
        validate_and_extract(payload)


def test_extreme_finite_logits_have_finite_kl_features() -> None:
    payload = valid_payload()
    for stage in payload["stages"].values():
        for row in stage["units"]:
            row["cells"]["clean"]["tristate_logits"] = {
                "support": 1_000_000.0,
                "refute": -1_000_000.0,
                "undetermined": -1_000_000.0,
            }
            row["cells"]["image_only"]["tristate_logits"] = {
                "support": -1_000_000.0,
                "refute": 1_000_000.0,
                "undetermined": -1_000_000.0,
            }
    result = validate_and_extract(payload)
    kl_keys = [
        "symmetric_kl_clean_to_image_only",
        "symmetric_kl_clean_to_question_only",
        "symmetric_kl_clean_to_joint",
    ]
    assert all(
        math.isfinite(row["features"][key])
        for row in result["feature_rows"]
        for key in kl_keys
    )


def test_logit_range_and_real_adapter_execution_fail_closed() -> None:
    payload = valid_payload()
    payload["stages"]["dev_fit"]["units"][0]["cells"]["clean"][
        "tristate_logits"
    ]["support"] = 1_000_001.0
    with pytest.raises(SubstrateError, match="numerical stability contract"):
        validate_and_extract(payload)

    payload = valid_payload()
    payload["real_adapter_input_schema"]["execution_enabled"] = True
    with pytest.raises(SubstrateError, match="frozen, disabled schema handoff"):
        validate_and_extract(payload)
