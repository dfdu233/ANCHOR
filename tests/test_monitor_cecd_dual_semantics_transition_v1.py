from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.monitor_cecd_dual_semantics_transition_v1 as monitor


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "stage_state": tmp_path / "state.json",
        "stage_analysis": tmp_path / "analysis.json",
        "stage_input_gate": tmp_path / "input_gate.json",
        "admission": tmp_path / "admission.json",
        "preflight": tmp_path / "preflight.json",
        "preflight_build": tmp_path / "preflight_build.json",
        "input_root": tmp_path / "inputs",
        "method_output_root": tmp_path / "method_outputs",
        "authorization": tmp_path / "authorization.json",
        "runner_handoff": tmp_path / "runner_handoff.json",
        "formal_job_state": tmp_path / "formal_job_state.json",
        "formal_job_log": tmp_path / "formal_job.log",
    }


def _advance(tmp_path: Path, monkeypatch, *, stage_result=None):
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        monitor.cecd_monitor,
        "validate_stage_result",
        lambda **_: stage_result,
    )
    return monitor.advance(**paths, root=tmp_path), paths


def _write_done_inputs(paths: dict[str, Path]) -> None:
    paths["stage_state"].write_text(
        json.dumps({"name": "cecd-three-stage-v3", "status": "done"}), encoding="utf-8"
    )
    for name in ("stage_analysis", "stage_input_gate", "admission"):
        paths[name].write_text("{}", encoding="utf-8")


def _passed() -> dict:
    return {
        "gate": {"authorized_for_method_level_treble_adapter_run": True}
    }


def test_transition_waits_without_stage1_and_reports_running_pid(
    tmp_path: Path, monkeypatch
) -> None:
    result, paths = _advance(tmp_path, monkeypatch)
    assert result["stage"] == "waiting_for_two_model_stage1"
    assert result["stage1_state_present"] is False

    paths["stage_state"].write_text(
        json.dumps({"name": "cecd-three-stage-v3", "status": "running", "child_pid": 123}), encoding="utf-8"
    )
    result = monitor.advance(**paths, root=tmp_path)
    assert result["stage"] == "waiting_for_two_model_stage1"
    assert result["stage1_child_pid"] == 123


def test_transition_rejects_explicit_legacy_pilot_as_dev_state(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths["stage_state"].write_text(
        json.dumps({"name": "cecd-two-model-stage1-v2", "status": "done"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="legacy pilot-as-dev"):
        monitor.advance(**paths, root=tmp_path)


def test_transition_fails_closed_on_stage_failure_or_missing_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    paths["stage_state"].write_text(
        json.dumps({"name": "cecd-three-stage-v3", "status": "failed", "exit_code": 9}), encoding="utf-8"
    )
    result = monitor.advance(**paths, root=tmp_path)
    assert result["stage"] == "two_model_stage1_failed_terminal"
    assert result["retry_authorized"] is False

    paths["stage_state"].write_text(
        json.dumps({"name": "cecd-three-stage-v3", "status": "done"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="lacks analysis"):
        monitor.advance(**paths, root=tmp_path)


def test_transition_terminates_behavioral_no_go_without_preflight_rescue(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    _write_done_inputs(paths)
    monkeypatch.setattr(
        monitor.cecd_monitor,
        "validate_stage_result",
        lambda **_: {
            "gate": {"authorized_for_method_level_treble_adapter_run": False}
        },
    )
    result = monitor.advance(**paths, root=tmp_path)
    assert result["stage"] == "two_model_stage1_no_go_terminal"
    assert result["controlled_method_comparison_authorized"] is False
    assert result["retry_authorized"] is False


def test_transition_builds_preflight_and_emits_runner_handoff_after_pass(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    _write_done_inputs(paths)
    monkeypatch.setattr(monitor.cecd_monitor, "validate_stage_result", lambda **_: _passed())
    calls = []

    def fake_build(**kwargs):
        calls.append(("build", kwargs))
        paths["preflight"].write_text("{}", encoding="utf-8")
        paths["preflight_build"].write_text("{}", encoding="utf-8")
        return {"fingerprint": "b" * 64}

    monkeypatch.setattr(monitor, "build_preflight", fake_build)
    monkeypatch.setattr(monitor, "authorize", lambda **_: {"fingerprint": "a" * 64})
    monkeypatch.setattr(
        monitor, "emit_runner_handoff", lambda **_: {"fingerprint": "h" * 64}
    )
    result = monitor.advance(**paths, root=tmp_path)
    assert [name for name, _ in calls] == ["build"]
    assert result["stage"] == "controlled_comparison_authorized_runner_handoff_ready"
    assert result["behavioral_gate_passed"] is True
    assert result["gpu_launched_by_monitor"] is False


def test_transition_binds_present_preflight_but_does_not_generalize_gpu_authority(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    _write_done_inputs(paths)
    paths["preflight"].write_text("{}", encoding="utf-8")
    monkeypatch.setattr(monitor.cecd_monitor, "validate_stage_result", lambda **_: _passed())
    monkeypatch.setattr(monitor, "build_preflight", lambda **_: {})
    monkeypatch.setattr(
        monitor, "emit_runner_handoff", lambda **_: {"fingerprint": "h" * 64}
    )
    calls = []

    def fake_authorize(**kwargs):
        calls.append(kwargs)
        return {"fingerprint": "f" * 64}

    monkeypatch.setattr(monitor, "authorize", fake_authorize)
    result = monitor.advance(**paths, root=tmp_path)
    assert len(calls) == 1
    assert result["stage"] == "controlled_comparison_authorized_runner_handoff_ready"
    assert result["controlled_method_comparison_authorized"] is True
    assert result["general_gpu_authorized"] is False
    assert result["paper_claim_authorized"] is False


def test_transition_reuses_valid_write_once_authorization_without_rebinding(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    _write_done_inputs(paths)
    paths["preflight"].write_text("{}", encoding="utf-8")
    paths["preflight_build"].write_text("{}", encoding="utf-8")
    paths["authorization"].write_text("{}", encoding="utf-8")
    monkeypatch.setattr(monitor.cecd_monitor, "validate_stage_result", lambda **_: _passed())
    monkeypatch.setattr(
        monitor,
        "_validate_existing_authorization",
        lambda **_: {"fingerprint": "e" * 64},
    )
    monkeypatch.setattr(monitor, "build_preflight", lambda **_: {})
    monkeypatch.setattr(
        monitor, "emit_runner_handoff", lambda **_: {"fingerprint": "h" * 64}
    )

    def tripwire(**_kwargs):
        raise AssertionError("existing write-once authorization must not be rebound")

    monkeypatch.setattr(monitor, "authorize", tripwire)
    result = monitor.advance(**paths, root=tmp_path)
    assert result["authorization_fingerprint"] == "e" * 64
    assert result["stage"] == "controlled_comparison_authorized_runner_handoff_ready"


def test_runner_handoff_is_ce_only_hash_bound_write_once_and_never_launches(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    for name in ("authorization", "preflight", "preflight_build"):
        paths[name].write_text(json.dumps({"fixture": name}), encoding="utf-8")
    monkeypatch.setattr(monitor, "validate_build_receipt", lambda **_: {})
    result = monitor.emit_runner_handoff(
        authorization=paths["authorization"],
        preflight=paths["preflight"],
        preflight_build=paths["preflight_build"],
        handoff=paths["runner_handoff"],
        formal_job_state=paths["formal_job_state"],
        formal_job_log=paths["formal_job_log"],
        root=tmp_path,
    )
    assert result["status"] == "ready_not_launched"
    assert result["execute_ce_only"] is True
    assert result["launched_by_transition_monitor"] is False
    assert result["oe_authorized"] is False
    assert result["canonical_gpu_lock"].endswith("gpu0-vindr-v2.lock")
    assert "CECD_DUAL_EXECUTE_CE_ONLY=1" in result["launch_command"]
    assert monitor.emit_runner_handoff(
        authorization=paths["authorization"],
        preflight=paths["preflight"],
        preflight_build=paths["preflight_build"],
        handoff=paths["runner_handoff"],
        formal_job_state=paths["formal_job_state"],
        formal_job_log=paths["formal_job_log"],
        root=tmp_path,
    ) == result
    paths["authorization"].write_text(json.dumps({"fixture": "drift"}), encoding="utf-8")
    with pytest.raises(Exception, match="write-once preflight collision"):
        monitor.emit_runner_handoff(
            authorization=paths["authorization"],
            preflight=paths["preflight"],
            preflight_build=paths["preflight_build"],
            handoff=paths["runner_handoff"],
            formal_job_state=paths["formal_job_state"],
            formal_job_log=paths["formal_job_log"],
            root=tmp_path,
        )
