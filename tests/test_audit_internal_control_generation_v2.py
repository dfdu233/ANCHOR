import json
from pathlib import Path

from anchor.medeval.audit_internal_control_generation_v2 import expected_arm_names
from anchor.medeval.hashing import sha256_file


def test_v2_contract_expands_to_one_common_512_token_matrix() -> None:
    contract = json.loads(
        open("configs/unified_eval/internal_control_t3_execution_v2.json").read()
    )
    names = expected_arm_names(contract)
    assert names == [
        "greedy512",
        "sample_t02_p09_seed42",
        "sample_t07_p09_seed42",
        "sample_t10_p09_seed42",
        "sample_t07_p09_seed1042",
        "sample_t07_p09_seed2042",
        "sample_t07_p09_seed3042",
        "sample_t07_p09_seed4042",
        "replay_t07_p09_seed42",
    ]
    assert len(names) == len(set(names)) == 9
    assert all(
        arm.get("max_new_tokens") == 512
        for arm in contract["temperature_length_controls"]["arms"]
    )


def test_v2_repair_is_bound_to_outcome_free_failure_and_matching_clinical_contract() -> None:
    execution_path = Path("configs/unified_eval/internal_control_t3_execution_v2.json")
    contract = json.loads(execution_path.read_text())
    trigger = Path(contract["repair_trigger"]["artifact"])
    assert sha256_file(trigger) == contract["repair_trigger"]["artifact_sha256"]
    failure = json.loads(trigger.read_text())
    assert failure["reference_answers_used"] is False
    assert failure["clinical_labels_used"] is False
    assert failure["all_eligible"] is False

    clinical = json.loads(
        Path("configs/unified_eval/internal_control_t3_clinical_analysis_v2.json").read_text()
    )
    assert clinical["baseline"] == contract["clinical_analysis"]["baseline"]
    assert clinical["candidate_arms"] == contract["clinical_analysis"]["candidates"]
    assert clinical["repair_basis"]["execution_contract_sha256"] == sha256_file(
        execution_path
    )

    v1 = Path("corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t3_n120_v1.json")
    v2 = Path("corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t3_n120_v2.json")
    assert v1.read_bytes() == v2.read_bytes()
