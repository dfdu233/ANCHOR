from __future__ import annotations

from pathlib import Path

import pytest

from anchor.corrected_sgta import audit_cecd_execution_dag_v1 as dag


def test_live_audit_is_outcome_blind_and_exposes_all_current_handoff_gaps() -> None:
    result = dag.audit()
    assert result["passed"] is False
    assert result["scope"] == {
        "cpu_only": True,
        "gpu_touched": False,
        "human_returns_opened": False,
        "admission_decisions_opened": False,
        "model_outputs_opened": False,
        "evaluation_results_opened": False,
    }
    blockers = {row["id"] for row in result["blockers"]}
    assert {
        "DUAL_PREFLIGHT_PRODUCER_GAP",
        "DUAL_RUNNER_TRIGGER_GAP",
        "GPU0_LOCK_SPLIT_BRAIN",
        "LISTING_RECEIPT_PROVENANCE_UNCLOSED",
        "LISTING_UPSTREAM_CE_HASH_UNVERIFIED",
    } <= blockers
    listing = next(row for row in result["dag"] if row["node"] == "listing_scientific_admission")
    assert listing["dependency_type"] == "convergent_AND_not_cycle"
    assert result["terminology"]["system_command_permission"].startswith("already_unrestricted")


def test_static_record_refuses_every_forbidden_outcome_tree() -> None:
    for path in dag.FORBIDDEN_OUTCOME_PATHS:
        with pytest.raises(RuntimeError, match="forbidden read"):
            dag.static_record(path / "anything.json" if path.suffix == "" else path)


def test_current_control_branches_remain_fail_closed() -> None:
    controls = dag.audit()["controls"]
    assert controls["halp"]["probe_training_authorized"] is False
    assert controls["system_pih"]["control_execution_ready"] is False
    assert controls["reader_threshold_aliasing"]["mainline_gate_modification_authorized"] is False
    assert controls["reader_threshold_aliasing"]["bindings"]["dev_fit_input"] == "missing"
    assert controls["reader_threshold_aliasing"]["bindings"]["confirmation_locked_input"] == "missing"


def test_gpu0_lock_split_brain_is_explicit_and_has_one_required_override() -> None:
    contract = dag.audit()["gpu_lock_contract"]
    assert contract["single_lock_closed"] is False
    assert contract["dual_default"] != contract["canonical_gpu0_lock"]
    assert contract["dual_required_override"] == contract["canonical_gpu0_lock"]
