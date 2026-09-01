import json
from pathlib import Path

from anchor.medeval.artifact_registry import append_qualification, qualification_for
from anchor.medeval.audit_internal_baseline_controls_v1 import audit
from anchor.medeval.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/unified_eval/internal_baseline_control_contract_v1.json"


def _evidence(path: Path) -> Path:
    methods = []
    for name in (
        "temperature_length_controls",
        "self_consistency",
        "calibrated_abstention",
    ):
        methods.append(
            {
                "name": name,
                "stages": {"T1": {"status": "pass"}},
            }
        )
    path.write_text(json.dumps({"methods": methods}))
    return path


def test_current_registry_is_fail_closed() -> None:
    result = audit(
        contract_path=CONTRACT,
        evidence_path=ROOT / "corrected_runs/unified_eval/provenance/method_evidence_ladder_v9.json",
        registry_path=ROOT / "corrected_runs/unified_eval/provenance/artifact_registry_v1.jsonl",
    )
    assert result["status"] == "partial_fail_closed"
    assert result["paper_control_claim_authorized"] is False
    assert result["summary"]["t1_pass"] == [
        "temperature_length_controls",
        "self_consistency",
        "calibrated_abstention",
    ]
    assert result["summary"]["t2_pass"] == [
        "temperature_length_controls",
        "self_consistency",
    ]
    assert result["summary"]["t2_missing"] == []
    assert result["summary"]["t2_failed"] == ["calibrated_abstention"]
    assert result["summary"]["full_pass"] == []


def test_temperature_t2_requires_hash_bound_disjoint_dev_and_test(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path / "evidence.json")
    registry = tmp_path / "registry.jsonl"
    payload = {
        "protocol_version": "temperature-length-control-t2-v1",
        "contract_sha256": sha256_file(CONTRACT),
        "method": "temperature_length_controls",
        "stage": "T2",
        "provenance": {"dev_manifest_sha256": "dev", "test_manifest_sha256": "test"},
        "design": {
            "development_grid_frozen_before_test": True,
            "test_labels_used_for_tuning": False,
            "temperature_top_p_grid": [[0.7, 0.9]],
        },
        "generation": {
            "generated_token_ids_recorded": True,
            "seed_ledger_complete": True,
            "sampling_activation_non_degenerate": True,
            "exact_qid_coverage": True,
            "stop_and_cap_provenance_complete": True,
            "posthoc_truncation_used": False,
        },
        "analysis": {"matched_length_plan_frozen": True},
    }
    artifact = tmp_path / "temperature.json"
    artifact.write_text(json.dumps(payload))
    append_qualification(
        registry,
        qualification_for(
            artifact,
            status="admissible",
            evaluator_version="test",
            evidence_scope="internal control qualification; temperature_length_controls; T2",
            reason="test fixture",
        ),
    )
    result = audit(contract_path=CONTRACT, evidence_path=evidence, registry_path=registry)
    assert result["summary"]["t2_pass"] == ["temperature_length_controls"]

    payload["provenance"]["test_manifest_sha256"] = "dev"
    artifact.write_text(json.dumps(payload))
    # Append a new event for the changed artifact; latest-by-artifact must fail it.
    append_qualification(
        registry,
        qualification_for(
            artifact,
            status="admissible",
            evaluator_version="test-2",
            evidence_scope="internal control qualification; temperature_length_controls; T2",
            reason="negative fixture",
        ),
    )
    result = audit(contract_path=CONTRACT, evidence_path=evidence, registry_path=registry)
    row = next(row for row in result["methods"] if row["name"] == "temperature_length_controls")
    assert row["stages"]["T2"]["status"] == "failed"
    assert any("not_distinct" in failure for failure in row["stages"]["T2"]["evidence"][0]["failures"])


def test_t3_cannot_pass_without_t2(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path / "evidence.json")
    registry = tmp_path / "registry.jsonl"
    artifact = tmp_path / "t3.json"
    artifact.write_text(
        json.dumps(
            {
                "protocol_version": "claim-self-consistency-t3-v1",
                "contract_sha256": sha256_file(CONTRACT),
                "method": "self_consistency",
                "stage": "T3",
                "clinical": {
                    "claim_scoring_complete": True,
                    "claim_coverage_matched": True,
                    "answer_budget_matched": True,
                    "omissions_counted": True,
                    "test_judge_used_for_selection": False,
                    "bootstrap_cluster_unit": "image",
                },
                "efficacy": {
                    "primary_improvement_passed": True,
                    "omission_increased": False,
                    "length_or_coverage_exchange": False,
                },
            }
        )
    )
    append_qualification(
        registry,
        qualification_for(
            artifact,
            status="admissible",
            evaluator_version="test",
            evidence_scope="internal control qualification; self_consistency; T3",
            reason="test fixture",
        ),
    )
    result = audit(contract_path=CONTRACT, evidence_path=evidence, registry_path=registry)
    row = next(row for row in result["methods"] if row["name"] == "self_consistency")
    assert row["stages"]["T3"]["status"] == "failed"
    assert row["stages"]["full"]["status"] == "not_authorized"
