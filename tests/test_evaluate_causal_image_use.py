from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from anchor.medeval.evaluate_causal_image_use import (
    PARSER_CONTRACT,
    CausalConditionRecord,
    ContractError,
    analyze_records,
    validate_case_conditions,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _case(
    *,
    model: str,
    finding: str,
    index: int,
    behavior: str,
    invalid_condition: str | None = None,
) -> list[CausalConditionRecord]:
    case_id = f"{model}-{finding}-{index:03d}"
    source_subject = _hash(f"subject-{model}-{index}")
    swap_subject = _hash(f"swap-subject-{model}-{index}")
    source_image = _hash(f"source-image-{model}-{index}")
    question = _hash(f"question-{finding}-{index}")
    prompt = _hash(f"prompt-{finding}-{index}")
    if behavior == "uses_image":
        decisions = {
            "original": "yes",
            "swap": "yes",
            "target_mask": "no",
            "irrelevant_mask": "yes",
        }
    elif behavior == "ignores_image":
        decisions = {condition: "yes" for condition in (
            "original", "swap", "target_mask", "irrelevant_mask"
        )}
    elif behavior == "unstable":
        decisions = {
            "original": "yes",
            "swap": "no",
            "target_mask": "no",
            "irrelevant_mask": "no",
        }
    else:
        raise ValueError(behavior)
    if invalid_condition is not None:
        decisions[invalid_condition] = "invalid"

    rows = []
    for condition, decision in decisions.items():
        masked = condition in {"target_mask", "irrelevant_mask"}
        rows.append(
            CausalConditionRecord(
                case_id=case_id,
                model_id=model,
                finding=finding,
                cluster_id=_hash(f"cluster-{model}-{index}"),
                view="AP" if index % 2 == 0 else "PA",
                condition=condition,
                ground_truth="yes",
                ground_truth_provenance="independent_clinical",
                decision=decision,
                raw_text_sha256=_hash(f"raw-{case_id}-{condition}-{decision}"),
                parser_version=PARSER_CONTRACT,
                question_sha256=question,
                prompt_sha256=prompt,
                reference_contract_sha256=_hash("reference-contract-v1"),
                swap_manifest_sha256=_hash("swap-manifest-v1"),
                source_image_sha256=source_image,
                condition_image_sha256=(
                    source_image
                    if condition == "original"
                    else _hash(f"condition-image-{case_id}-{condition}")
                ),
                source_subject_hash=source_subject,
                condition_subject_hash=(
                    swap_subject if condition == "swap" else source_subject
                ),
                swap_label_preserved=True,
                target_region_defined=True,
                irrelevant_region_defined=True,
                region_provenance="expert_box",
                mask_sha256=_hash(f"mask-{case_id}-{condition}") if masked else None,
                mask_area_pixels=100 if masked else None,
            )
        )
    return rows


def _records(models: list[str], behavior: str, n: int = 40) -> list[CausalConditionRecord]:
    return [
        row
        for model in models
        for index in range(n)
        for row in _case(
            model=model,
            finding="cardiomegaly",
            index=index,
            behavior=behavior,
        )
    ]


def _analyze(records: list[CausalConditionRecord], models: list[str]) -> dict:
    return analyze_records(
        records,
        target_models=models,
        target_findings=["cardiomegaly"],
        bootstrap_replicates=200,
        seed=17,
        minimum_eligible_cases=30,
        minimum_parse_rate=0.95,
    )


def test_two_model_stable_image_use_gate_is_backend_neutral_and_not_gpu_authority() -> None:
    models = ["huatuo", "hulu"]
    result = _analyze(_records(models, "uses_image"), models)
    for model in models:
        cell = result["model_finding_cells"][f"{model}::cardiomegaly"]
        assert cell["official_behavior_category"] == "uses_image"
        assert cell["causal_grounding_rate"]["point"] == 1.0
        assert cell["unrelated_image_answer_rate"]["point"] == 1.0
        assert cell["irrelevant_mask_stability"]["point"] == 1.0
        assert cell["pcem_stable_image_user_cell"] is True
    assert result["admission"]["pcem_image_use_gate_passed"] is True
    assert result["admission"]["representation_capture_authorized"] is False
    assert result["admission"]["image_download_authorized"] is False
    assert result["admission"]["gpu_authorized"] is False
    assert result["admission"]["paper_claim_authorized"] is False
    assert result == _analyze(_records(models, "uses_image"), models)


def test_ignores_image_and_unstable_are_not_admitted() -> None:
    ignored = _analyze(_records(["model-a"], "ignores_image"), ["model-a"])
    ignored_cell = ignored["model_finding_cells"]["model-a::cardiomegaly"]
    assert ignored_cell["official_behavior_category"] == "ignores_image"
    assert ignored_cell["pcem_stable_image_user_cell"] is False
    assert ignored["admission"]["pcem_image_use_gate_passed"] is False

    unstable = _analyze(_records(["model-a"], "unstable"), ["model-a"])
    unstable_cell = unstable["model_finding_cells"]["model-a::cardiomegaly"]
    assert unstable_cell["official_behavior_category"] == "unstable"
    assert unstable_cell["irrelevant_mask_stability"]["point"] == 0.0
    assert unstable_cell["pcem_stable_image_user_cell"] is False


def test_parse_rate_and_power_are_independent_admission_gates() -> None:
    records = _records(["model-a"], "uses_image")
    for index in range(3):
        target = next(
            row
            for row in records
            if row.case_id == f"model-a-cardiomegaly-{index:03d}"
            and row.condition == "target_mask"
        )
        records[records.index(target)] = replace(target, decision="invalid")
    result = _analyze(records, ["model-a"])
    cell = result["model_finding_cells"]["model-a::cardiomegaly"]
    assert cell["official_behavior_category"] == "uses_image"
    assert cell["parse_rates"]["target_mask"] == pytest.approx(37 / 40)
    assert cell["minimum_eligible_case_gate_passed"] is True
    assert cell["parse_gate_passed"] is False
    assert cell["pcem_stable_image_user_cell"] is False

    small = _analyze(_records(["model-a"], "uses_image", n=20), ["model-a"])
    small_cell = small["model_finding_cells"]["model-a::cardiomegaly"]
    assert small_cell["official_behavior_category"] == "uses_image"
    assert small_cell["minimum_eligible_case_gate_passed"] is False
    assert small_cell["pcem_stable_image_user_cell"] is False


def test_swap_and_mask_nuisance_contracts_fail_closed() -> None:
    rows = _case(
        model="model-a",
        finding="cardiomegaly",
        index=0,
        behavior="uses_image",
    )
    swap = next(row for row in rows if row.condition == "swap")
    bad_swap = [
        replace(row, condition_subject_hash=row.source_subject_hash)
        if row is swap
        else row
        for row in rows
    ]
    with pytest.raises(ContractError, match="different patient"):
        validate_case_conditions(bad_swap)

    irrelevant = next(row for row in rows if row.condition == "irrelevant_mask")
    bad_mask = [
        replace(row, mask_area_pixels=101) if row is irrelevant else row for row in rows
    ]
    with pytest.raises(ContractError, match="equal pixel area"):
        validate_case_conditions(bad_mask)


def test_mixed_or_unfrozen_parser_versions_are_rejected() -> None:
    records = _records(["model-a"], "uses_image")
    records[0] = replace(records[0], parser_version="legacy-keyword-parser")
    with pytest.raises(ContractError, match="mixed parser versions"):
        _analyze(records, ["model-a"])

    records = [
        replace(row, parser_version="legacy-keyword-parser")
        for row in _records(["model-a"], "uses_image")
    ]
    with pytest.raises(ContractError, match="parser version must be the frozen"):
        _analyze(records, ["model-a"])


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    [
        ("ground_truth_provenance", "model_derived", "ground_truth_provenance_gate_passed"),
        ("region_provenance", "automatic_unvalidated", "region_provenance_gate_passed"),
    ],
)
def test_untrusted_truth_or_region_can_be_diagnostic_but_not_admitted(
    field: str, value: str, gate: str
) -> None:
    records = [replace(row, **{field: value}) for row in _records(["model-a"], "uses_image")]
    result = _analyze(records, ["model-a"])
    cell = result["model_finding_cells"]["model-a::cardiomegaly"]
    assert cell["official_behavior_category"] == "uses_image"
    assert cell[gate] is False
    assert cell["pcem_stable_image_user_cell"] is False
    assert result["admission"]["pcem_image_use_gate_passed"] is False


@pytest.mark.parametrize("field", ["reference_contract_sha256", "swap_manifest_sha256"])
def test_mixed_reference_or_swap_contract_is_rejected(field: str) -> None:
    records = _records(["model-a"], "uses_image")
    records[0] = replace(records[0], **{field: _hash(f"different-{field}")})
    with pytest.raises(ContractError, match="one reference contract"):
        _analyze(records, ["model-a"])


def test_programmatic_records_are_validated_not_only_jsonl_records() -> None:
    records = _records(["model-a"], "uses_image")
    records[0] = replace(records[0], raw_text_sha256="not-a-sha256")
    with pytest.raises(ContractError, match="raw_text_sha256"):
        _analyze(records, ["model-a"])
