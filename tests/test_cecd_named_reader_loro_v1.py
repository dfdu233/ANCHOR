import copy
import json
from pathlib import Path

import numpy as np
import pytest

from corrected_sgta.cecd_named_reader_loro_v1 import (
    ANNOTATION_PROTOCOL,
    IDENTITY_CONTRACT,
    MANIFEST_VERSION,
    NamedReaderLOROError,
    apply_named_reader_loro_confirmation,
    fail_closed_record,
    fit_named_reader_loro_dev,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = json.loads(
    (ROOT / "configs/cecd_named_reader_loro_contract_v1.json").read_text()
)

MODELS = ("model-a", "model-b")
FINDINGS = ("finding-a", "finding-b")
READERS = ("R8", "R9", "R10")
RENDERS = ("render-a", "render-b")
PROMPTS = ("prompt-a", "prompt-b")


def _config():
    value = copy.deepcopy(FORMAL_CONFIG)
    value["models"] = list(MODELS)
    value["findings"] = list(FINDINGS)
    value["minimum_dev_orbits_per_model_finding"] = 8
    value["minimum_confirmation_orbits_per_model_finding"] = 8
    value["bootstrap"]["draws"] = 31
    return value


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


def _inputs(stage, n=16):
    source_split = "dev" if stage == "dev_fit" else "confirmation"
    prefix = "d" if stage == "dev_fit" else "c"
    records = []
    named_records = []
    # All eight named-reader response patterns occur twice.  This makes every
    # held-out reader target identifiable without using that reader in fit.
    patterns = [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
        (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1),
    ]
    for index in range(n):
        image = f"{prefix}-image-{index}"
        patient = f"{prefix}-patient-{index}"
        votes = dict(zip(READERS, patterns[index % len(patterns)]))
        aggregate = sum(votes.values())
        for finding_index, finding in enumerate(FINDINGS):
            named_records.append({
                "reference_source": "vindr_reader_votes",
                "image_id": image,
                "patient_id": patient,
                "finding": finding,
                "reader_count": 3,
                "positive_votes": aggregate,
                "reader_ids": list(READERS),
                "reader_votes": [
                    {"rad_id": reader, "vote": vote} for reader, vote in votes.items()
                ],
            })
            for model_index, model in enumerate(MODELS):
                base = aggregate - 1.5 + 0.03 * finding_index + 0.02 * model_index
                for render_index, render in enumerate(RENDERS):
                    for prompt_index, prompt in enumerate(PROMPTS):
                        interaction = 0.32 * render_index * prompt_index
                        score = base + 0.07 * render_index - 0.04 * prompt_index + interaction
                        records.append(_cell(model, image, patient, finding, aggregate, render, prompt, score))
                for prompt_index, prompt in enumerate(PROMPTS):
                    score = base - 0.04 * prompt_index
                    records.append(_cell(model, image, patient, finding, aggregate, "identity", prompt, score))
                records.append(_cell(model, image, patient, finding, aggregate, RENDERS[0], "duplicate", base))
    payload = {
        "schema_version": "clinical-equivalence-factorial-v1",
        "split": stage,
        "source_manifest_split": source_split,
        "frozen_before_outputs": True,
        "score_definition": "fp32_yes_minus_no_logit",
        "primary_renders": list(RENDERS),
        "primary_prompts": list(PROMPTS),
        "baseline_render": RENDERS[0],
        "baseline_prompt": PROMPTS[0],
        "identity_render": "identity",
        "duplicate_prompt": "duplicate",
        "records": records,
    }
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "split": source_split,
        "reader_identity": dict(IDENTITY_CONTRACT),
        "annotation_protocol": dict(ANNOTATION_PROTOCOL),
        "records": named_records,
    }
    return payload, manifest


def _cell(model, image, patient, finding, aggregate, render, prompt, score):
    return {
        "model": model,
        "image_id": image,
        "patient_id": patient,
        "finding": finding,
        "reader_votes": aggregate,
        "render_id": render,
        "prompt_id": prompt,
        "acquisition_view": "PA",
        "input_prompt_length_tokens": 11,
        "answer_length_tokens": 1,
        **_score_fields(score),
    }


@pytest.fixture()
def exact_inputs():
    return _config(), *_inputs("dev_fit"), *_inputs("confirmation_locked")


def test_formal_contract_is_pre_result_and_non_authorizing():
    validated = validate_config(FORMAL_CONFIG)
    assert validated["authorized"] is False
    assert validated["frozen_before_real_outputs"] is True
    assert validated["named_readers"] == ["R8", "R9", "R10"]
    assert validated["bootstrap"]["unit"] == "patient"


def test_dev_fit_excludes_each_named_reader_and_confirmation_is_apply_only(exact_inputs):
    config, dev, dev_named, confirmation, confirmation_named = exact_inputs
    bundle = fit_named_reader_loro_dev(dev, dev_named, config)
    assert bundle["authorized"] is False
    assert bundle["fit_split"] == "dev"
    assert bundle["apply_split"] == "confirmation"
    assert "confirmation" not in bundle["dev_cecd_input_sha256"]
    for audit in bundle["exclusion_audit"].values():
        assert audit["heldout_outcomes_used_as_calibration_target"] == 0
        assert audit["nonheldout_named_votes_used"] == 2 * audit["n_fit"]

    result = apply_named_reader_loro_confirmation(
        confirmation, confirmation_named, bundle, config
    )
    assert result["authorized"] is False
    assert result["identifiability"]["named_rad_ID_preserved"] is True
    assert result["identifiability"]["heldout_reader_vote_excluded_from_calibration_target"] is True
    assert result["identifiability"]["upstream_selection_may_use_aggregate_vote_including_heldout"] is True
    assert set(result["primary"]["models"]) == set(MODELS)
    assert len(result["by_model_finding_heldout_reader"]) == len(MODELS) * len(FINDINGS) * 3
    assert result["primary"]["bootstrap_draws"] == 31


def test_heldout_dev_vote_cannot_change_its_own_loro_calibrator(exact_inputs):
    config, dev, dev_named, _, _ = exact_inputs
    original = fit_named_reader_loro_dev(dev, dev_named, config)
    changed_payload = copy.deepcopy(dev)
    changed_named = copy.deepcopy(dev_named)
    # Change the held-out outcome on one image only.  The R8 calibration must
    # remain byte-identical because its target is formed exclusively from R9
    # and R10; the remaining LORO fits retain directional identifiability.
    target_image = changed_named["records"][0]["image_id"]
    for row in changed_named["records"]:
        if row["image_id"] != target_image:
            continue
        old = next(item["vote"] for item in row["reader_votes"] if item["rad_id"] == "R8")
        for item in row["reader_votes"]:
            if item["rad_id"] == "R8":
                item["vote"] = 1 - old
        row["positive_votes"] += 1 if old == 0 else -1
    aggregates = {
        (row["image_id"], row["finding"]): row["positive_votes"]
        for row in changed_named["records"]
    }
    for row in changed_payload["records"]:
        row["reader_votes"] = aggregates[(row["image_id"], row["finding"])]
    changed = fit_named_reader_loro_dev(changed_payload, changed_named, config)
    for model in MODELS:
        for finding in FINDINGS:
            assert changed["calibrators"][model][finding]["R8"] == original["calibrators"][model][finding]["R8"]


@pytest.mark.parametrize("fault", ["aggregate", "positions", "duplicate", "two_readers", "wrong_panel"])
def test_aggregate_and_anonymous_or_invalid_reader_structures_fail_closed(exact_inputs, fault):
    config, dev, dev_named, _, _ = exact_inputs
    broken = copy.deepcopy(dev_named)
    row = broken["records"][0]
    if fault == "aggregate":
        row["reader_votes"] = row["positive_votes"]
    elif fault == "positions":
        row["reader_votes"] = [0, 0, 0]
    elif fault == "duplicate":
        row["reader_votes"][1]["rad_id"] = row["reader_votes"][0]["rad_id"]
    elif fault == "two_readers":
        row["reader_votes"] = row["reader_votes"][:2]
    elif fault == "wrong_panel":
        row["reader_votes"][0]["rad_id"] = "R11"
    with pytest.raises(NamedReaderLOROError):
        fit_named_reader_loro_dev(dev, broken, config)


def test_identity_provenance_and_aggregate_crosscheck_fail_closed(exact_inputs):
    config, dev, dev_named, _, _ = exact_inputs
    anonymous = copy.deepcopy(dev_named)
    anonymous["reader_identity"]["source_field"] = "vote_position"
    with pytest.raises(NamedReaderLOROError, match="identity provenance"):
        fit_named_reader_loro_dev(dev, anonymous, config)

    unsupported_independence = copy.deepcopy(dev_named)
    unsupported_independence["annotation_protocol"]["independent_reader_judgments"] = False
    with pytest.raises(NamedReaderLOROError, match="do not by themselves establish independent"):
        fit_named_reader_loro_dev(dev, unsupported_independence, config)

    mismatch = copy.deepcopy(dev_named)
    mismatch["records"][0]["positive_votes"] += 1
    with pytest.raises(NamedReaderLOROError, match="aggregate fields disagree"):
        fit_named_reader_loro_dev(dev, mismatch, config)


def test_missing_patient_identity_is_explicitly_unidentifiable(exact_inputs):
    config, dev, dev_named, _, _ = exact_inputs
    dev_named["records"][0].pop("patient_id")
    with pytest.raises(NamedReaderLOROError, match="patient-level leakage.*not identifiable") as caught:
        fit_named_reader_loro_dev(dev, dev_named, config)
    record = fail_closed_record(caught.value, "dev_fit")
    assert record["identifiable"] is False
    assert record["statistics_emitted"] is False
    assert record["authorized"] is False
    assert "patient-level leakage" in record["reason"]


@pytest.mark.parametrize("leak", ["image", "patient"])
def test_dev_confirmation_image_and_patient_leakage_fail_closed(exact_inputs, leak):
    config, dev, dev_named, confirmation, confirmation_named = exact_inputs
    bundle = fit_named_reader_loro_dev(dev, dev_named, config)
    if leak == "image":
        old = confirmation_named["records"][0]["image_id"]
        new = dev_named["records"][0]["image_id"]
        for row in confirmation_named["records"]:
            if row["image_id"] == old:
                row["image_id"] = new
        for row in confirmation["records"]:
            if row["image_id"] == old:
                row["image_id"] = new
    else:
        old = confirmation_named["records"][0]["patient_id"]
        new = dev_named["records"][0]["patient_id"]
        for row in confirmation_named["records"]:
            if row["patient_id"] == old:
                row["patient_id"] = new
        for row in confirmation["records"]:
            if row["patient_id"] == old:
                row["patient_id"] = new
    with pytest.raises(NamedReaderLOROError, match=f"{leak} crosses"):
        apply_named_reader_loro_confirmation(confirmation, confirmation_named, bundle, config)


def test_dev_and_confirmation_stage_roles_and_bundle_seal_cannot_drift(exact_inputs):
    config, dev, dev_named, confirmation, confirmation_named = exact_inputs
    with pytest.raises(NamedReaderLOROError, match="dev_fit must use truthful dev"):
        fit_named_reader_loro_dev(confirmation, confirmation_named, config)
    bundle = fit_named_reader_loro_dev(dev, dev_named, config)
    tampered = copy.deepcopy(bundle)
    tampered["calibrators"][MODELS[0]][FINDINGS[0]][READERS[0]]["y_thresholds"][0] = 0.9
    with pytest.raises(NamedReaderLOROError, match="bundle_sha256 mismatch"):
        apply_named_reader_loro_confirmation(confirmation, confirmation_named, tampered, config)


def test_reader_vote_order_is_not_reader_identity(exact_inputs):
    config, dev, dev_named, confirmation, confirmation_named = exact_inputs
    baseline = fit_named_reader_loro_dev(dev, dev_named, config)
    reordered = copy.deepcopy(dev_named)
    for row in reordered["records"]:
        row["reader_votes"] = list(reversed(row["reader_votes"]))
        row["reader_ids"] = list(reversed(row["reader_ids"]))
    candidate = fit_named_reader_loro_dev(dev, reordered, config)
    assert candidate["calibrators"] == baseline["calibrators"]
    result = apply_named_reader_loro_confirmation(
        confirmation, confirmation_named, candidate, config
    )
    assert result["identifiability"]["aggregate_or_anonymous_position_substitution"] is False
