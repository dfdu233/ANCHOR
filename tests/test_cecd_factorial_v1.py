from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

import corrected_sgta.run_cecd_factorial_v1 as cecd_factorial
from corrected_sgta.run_cecd_factorial_v1 import (
    BASELINE_VIEW,
    FROZEN_FINDINGS,
    IDENTITY_RENDER_NAME,
    SCIENCE_RENDER_NAMES,
    audit_is_admitted,
    cell_specs,
    compare_next_token_readouts,
    execution_contract,
    fp32_tristate_readout,
    pack_factorial_rows,
    selection,
    STAGE_SPECS,
    prompts_for,
    safe_record_key,
    shard_path,
    tokenizer_audit,
    valid_completed_cell,
    valid_render_audit_shard,
    individual_reader_votes,
    full_model_artifact_fingerprint,
    python_source_tree_fingerprint,
)


def test_three_stage_selection_contract_is_exact_and_hash_disjoint():
    selections = {
        stage: selection(stage, spec["manifest_split"], spec["per_bin"])
        for stage, spec in STAGE_SPECS.items()
    }
    assert {stage: len(rows) for stage, rows in selections.items()} == {
        "pilot_screen": 160,
        "dev_fit": 320,
        "confirmation_locked": 960,
    }
    images = {
        stage: {str(row["image_id"]) for row in rows}
        for stage, rows in selections.items()
    }
    assert images["pilot_screen"].isdisjoint(images["dev_fit"])
    assert images["pilot_screen"].isdisjoint(images["confirmation_locked"])
    assert images["dev_fit"].isdisjoint(images["confirmation_locked"])


def test_stage_selection_rejects_split_or_n_drift():
    with pytest.raises(ValueError, match="requires manifest_split"):
        selection("dev_fit", "pilot", 20)
    with pytest.raises(ValueError, match="requires manifest_split"):
        selection("confirmation_locked", "confirmation", 20)


class FakeTokenizer:
    """Minimal deterministic tokenizer for contract tests; no model download."""

    fixed = {"Yes": 7, "No": 11, "Maybe": 13}

    def encode(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        if text in self.fixed:
            return [self.fixed[text]]
        return [100 + sum(token.encode()) % 997 for token in text.split()]


class FakeModelScorer:
    """Deterministic fake next-token model used to audit duplicate controls."""

    def score(self, pixel_hash: str, prompt: str):
        seed = sum(pixel_hash.encode()) + sum(prompt.encode())
        hidden = torch.tensor([seed % 17, len(prompt), 1.0], dtype=torch.float32)
        weight = torch.zeros(20, 3, dtype=torch.float32)
        weight[7] = torch.tensor([1.0, 0.0, 0.0])
        weight[11] = torch.tensor([0.0, 1.0, 0.0])
        weight[13] = torch.tensor([0.0, 0.0, 1.0])
        return fp32_tristate_readout(
            hidden,
            weight,
            {"supported": 7, "refuted": 11, "undetermined": 13},
        )


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


def test_formal_cli_without_admission_fails_before_model_or_output(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "must-not-exist"

    def model_path_tripwire(_family):
        raise AssertionError("model path was touched before admission rejection")

    monkeypatch.setattr(cecd_factorial, "model_defaults", model_path_tripwire)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_cecd_factorial_v1", "--output-dir", str(output)],
    )
    with pytest.raises(RuntimeError, match="requires --admission-result"):
        cecd_factorial.main()
    assert not output.exists()


def test_admission_free_engineering_audit_can_never_authorize_science() -> None:
    contract = execution_contract(admission=None, engineering_render_audit=True)
    assert contract == {
        "execution_mode": "engineering_render_audit_only",
        "scientific_status": "engineering_render_audit_only_no_scientific_authorization",
        "clinical_equivalence_established": False,
        "cecd_model_scoring_authorized": False,
        "scientific_artifact_authorized": False,
    }
    with pytest.raises(RuntimeError, match="requires --admission-result"):
        execution_contract(admission=None, engineering_render_audit=False)


def test_reader_vote_records_normalize_and_fail_closed() -> None:
    row = {
        "reader_votes": [
            {"rad_id": "R10", "vote": 0},
            {"rad_id": "R8", "vote": 1},
            {"rad_id": "R9", "vote": 0},
        ],
        "reader_count": 3,
        "positive_votes": 1,
    }
    assert individual_reader_votes(row) == [0, 1, 0]
    row["positive_votes"] = 2
    with __import__("pytest").raises(ValueError, match="disagree"):
        individual_reader_votes(row)


def test_three_templates_preserve_frozen_proposition_and_speech_act() -> None:
    for finding in FROZEN_FINDINGS:
        prompts = prompts_for(finding)
        assert len(prompts) == 3
        assert len({item["text"] for item in prompts}) == 3
        assert {item["proposition"] for item in prompts} == {f"present({finding})"}
        assert {item["speech_act"] for item in prompts} == {
            "polar_diagnostic_question"
        }
        assert all(item["text"].endswith("Yes, No, or Maybe.") for item in prompts)


def test_tokenization_audit_is_complete_and_fingerprinted() -> None:
    audit = tokenizer_audit(FakeTokenizer())
    assert audit["verbalizer_token_ids"] == {
        "supported": 7,
        "refuted": 11,
        "undetermined": 13,
    }
    assert len(audit["rows"]) == 12
    assert len(audit["fingerprint"]) == 64
    assert all(len(row["token_ids_sha256"]) == 64 for row in audit["rows"])


def test_full_model_fingerprint_binds_custom_runtime_assets(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    runtime = tmp_path / "modeling_custom.py"
    runtime.write_text("VALUE = 1\n")
    first = full_model_artifact_fingerprint(tmp_path)
    runtime.write_text("VALUE = 2\n")
    second = full_model_artifact_fingerprint(tmp_path)
    assert first["fingerprint"] != second["fingerprint"]
    assert first["weight_content_hashes"][0]["name"] == "model.safetensors"
    assert first["non_weight_runtime_asset_hashes"][0]["name"] == "modeling_custom.py"


def test_external_python_runtime_tree_is_content_hash_bound(tmp_path) -> None:
    source = tmp_path / "runtime.py"
    source.write_text("VALUE = 1\n")
    first = python_source_tree_fingerprint(tmp_path)
    source.write_text("VALUE = 2\n")
    second = python_source_tree_fingerprint(tmp_path)
    assert first["fingerprint"] != second["fingerprint"]


def test_factorial_has_15_science_cells_and_separate_duplicate_controls() -> None:
    specs = cell_specs("pleural_effusion")
    science = [spec for spec in specs if spec.role == "science_factorial"]
    image_controls = [spec for spec in specs if spec.role == "identity_image_control"]
    prompt_controls = [
        spec for spec in specs if spec.role == "exact_duplicate_prompt_control"
    ]
    assert len(specs) == 19
    assert len(science) == 15
    assert {spec.render_name for spec in science} == set(SCIENCE_RENDER_NAMES)
    assert len(image_controls) == 3
    assert {spec.render_name for spec in image_controls} == {IDENTITY_RENDER_NAME}
    assert len(prompt_controls) == 1
    duplicate = prompt_controls[0]
    reference = next(spec for spec in science if spec.cell_id == duplicate.reference_cell_id)
    assert duplicate.prompt_text == reference.prompt_text
    assert duplicate.render_name == reference.render_name == BASELINE_VIEW


def test_fake_model_duplicate_image_and_prompt_controls_are_exact() -> None:
    scorer = FakeModelScorer()
    specs = cell_specs("cardiomegaly")
    pixel_hashes = {name: f"hash-{name}" for name in SCIENCE_RENDER_NAMES}
    # The lossless duplicate must expose exactly the baseline pixels.
    pixel_hashes[IDENTITY_RENDER_NAME] = pixel_hashes[BASELINE_VIEW]
    outputs = {
        spec.cell_id: scorer.score(pixel_hashes[spec.render_name], spec.prompt_text)
        for spec in specs
    }
    for spec in specs:
        if spec.reference_cell_id is not None:
            assert outputs[spec.cell_id]["logits"] == outputs[spec.reference_cell_id]["logits"]


def test_per_case_invalid_render_is_missing_not_baseline_substitution(tmp_path) -> None:
    render = {
        "name": "center_plus_0p05w",
        "audit": {"clinical_guard_pass": False, "pixel_sha256": "b" * 64},
    }
    admitted, reasons = audit_is_admitted(render)
    assert not admitted
    assert reasons == ["per_case_computational_guard_failed"]

    spec = next(
        spec
        for spec in cell_specs("aortic_enlargement")
        if spec.render_name == "center_plus_0p05w"
    )
    path = tmp_path / "missing.json"
    payload = {
        "status": "missing_invalid_render",
        "config_fingerprint": "cfg",
        "record_key": "claim",
        "cell_id": spec.cell_id,
        "render_name": spec.render_name,
        "prompt_name": spec.prompt_name,
        "prompt_text_sha256": __import__("hashlib").sha256(
            spec.prompt_text.encode()
        ).hexdigest(),
        "cell_role": spec.role,
        "missing_reasons": reasons,
    }
    path.write_text(json.dumps(payload))
    assert valid_completed_cell(path, "cfg", "claim", spec)
    payload["scores"] = {"logits": {"supported": 0, "refuted": 0, "undetermined": 0}}
    path.write_text(json.dumps(payload))
    assert not valid_completed_cell(path, "cfg", "claim", spec)


def test_render_audit_resume_requires_exact_frozen_view_contract(tmp_path) -> None:
    path = tmp_path / "audit.json"
    names = [*SCIENCE_RENDER_NAMES, IDENTITY_RENDER_NAME]
    payload = {
        "status": "engineering_render_audit_only",
        "config_fingerprint": "cfg",
        "record_key": "claim",
        "renders": [
            {
                "name": name,
                "audit": {"pixel_sha256": "a" * 64, "clinical_guard_pass": True},
            }
            for name in names
        ],
    }
    path.write_text(json.dumps(payload))
    assert valid_render_audit_shard(path, "cfg", "claim")
    payload["renders"].pop()
    path.write_text(json.dumps(payload))
    assert not valid_render_audit_shard(path, "cfg", "claim")


def test_fp32_readout_escapes_bfloat16_quantization_floor() -> None:
    hidden = torch.tensor([1.0003, 0.9999], dtype=torch.float32)
    weights = torch.zeros(8, 2, dtype=torch.float32)
    weights[2] = torch.tensor([1.0, 0.0])
    weights[3] = torch.tensor([0.0, 1.0])
    weights[5] = torch.tensor([0.5, 0.5])
    result = fp32_tristate_readout(
        hidden,
        weights,
        {"supported": 2, "refuted": 3, "undetermined": 5},
    )
    assert np.isclose(result["polarity"], 0.0004, atol=1e-7)
    assert result["readout"].startswith("FP32")


def test_next_token_conformance_detects_position_or_path_mismatch() -> None:
    direct = {"logits": {"supported": 2.0, "refuted": 1.0, "undetermined": 0.5}}
    harmless_common_shift = {
        "logits": {"supported": 7.01, "refuted": 6.0, "undetermined": 5.49},
        "generated_token_id": 7,
        "generated_text": "Yes",
    }
    assert compare_next_token_readouts(direct, harmless_common_shift, 0.02)["passed"]
    wrong_position = {
        "logits": {"supported": 1.0, "refuted": 2.0, "undetermined": 0.5}
    }
    assert not compare_next_token_readouts(direct, wrong_position, 0.1)["passed"]


def test_deterministic_packer_emits_exact_analyzer_contract(tmp_path) -> None:
    manifest_row = {
        "image_id": "img1",
        "finding": "pleural_effusion",
        "positive_votes": 2,
        "reader_votes": [0, 1, 1],
        "reader_support": 2 / 3,
    }
    control_row = {
        "image_id": "img2",
        "finding": "pleural_effusion",
        "positive_votes": 0,
        "reader_votes": [0, 0, 0],
        "reader_support": 0.0,
    }
    manifest_rows = [manifest_row, control_row]
    admission = {
        "status": "passed_hash_bound",
        "analysis_sha256": "b" * 64,
    }
    config = {
        "fingerprint": "cfg",
        "model": "huatuo:fake",
        "clinical_admission": admission,
    }
    for manifest in manifest_rows:
        record_key = safe_record_key(manifest)
        for index, spec in enumerate(cell_specs(manifest["finding"])):
            target = shard_path(tmp_path / "cell_shards", record_key, spec.cell_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "status": "ok",
                "config_fingerprint": "cfg",
                "record_key": record_key,
                "image_id": manifest["image_id"],
                "finding": manifest["finding"],
                "positive_votes": manifest["positive_votes"],
                "individual_reader_votes": manifest["reader_votes"],
                "reader_support": manifest["reader_support"],
                "acquisition_view": "unknown",
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
                "scores": consistent_scores(float(index)),
            }
            target.write_text(json.dumps(payload))
    summary = pack_factorial_rows(tmp_path, config, manifest_rows)
    packed = [json.loads(line) for line in (tmp_path / "factorial_rows.jsonl").read_text().splitlines()]
    assert summary["contract_version"] == "clinical-equivalence-factorial-v1"
    assert summary["role_counts"] == {
        "science_factorial": 30,
        "identity_image_control": 6,
        "exact_duplicate_prompt_control": 2,
    }
    assert len(packed) == 38
    assert all(row["model"] == "huatuo:fake" for row in packed)
    assert {row["reader_votes"] for row in packed} == {0, 2}
    assert all("signed_score" in row and "commitment_score" in row for row in packed)
    wrapper = json.loads((tmp_path / "factorial_payload.json").read_text())
    assert wrapper["schema_version"] == "clinical-equivalence-factorial-v1"
    assert wrapper["primary_renders"] == list(SCIENCE_RENDER_NAMES)
    assert wrapper["baseline_render"] == BASELINE_VIEW
    assert wrapper["identity_render"] == IDENTITY_RENDER_NAME
    assert wrapper["clinical_equivalence_admission"] == admission
    assert {row["reader_votes"] for row in wrapper["records"]} == {0, 2}
    assert summary["factorial_payload_sha256"]
    from corrected_sgta.analyze_clinical_equivalence_composition_defect_v1 import (
        validate_payload,
    )

    validated = validate_payload(wrapper)
    assert len(validated["rows"]) == 38

    record_key = safe_record_key(manifest_row)
    first_spec = cell_specs(manifest_row["finding"])[0]
    first_path = shard_path(tmp_path / "cell_shards", record_key, first_spec.cell_id)
    first = json.loads(first_path.read_text())
    first["status"] = "missing_invalid_render"
    first["missing_reasons"] = ["per_case_computational_guard_failed"]
    first.pop("scores")
    first_path.write_text(json.dumps(first))
    incomplete = pack_factorial_rows(tmp_path, config, manifest_rows)
    assert incomplete["complete_orbit_count"] == 1
    assert incomplete["incomplete_orbit_count"] == 1
    assert incomplete["incomplete_orbits"][0]["invalid_cells"][0]["cell_id"] == first_spec.cell_id
    invalid_wrapper = json.loads((tmp_path / "factorial_payload.json").read_text())
    invalid_records = [row for row in invalid_wrapper["records"] if not row["valid"]]
    assert len(invalid_records) == 1
    assert invalid_records[0]["signed_score"] is None
    assert invalid_records[0]["exclusion_reasons"] == [
        "per_case_computational_guard_failed"
    ]
    invalid_contract = validate_payload(invalid_wrapper)
    assert invalid_contract["excluded_orbits"][0]["reasons"] == [
        "per_case_computational_guard_failed"
    ]
