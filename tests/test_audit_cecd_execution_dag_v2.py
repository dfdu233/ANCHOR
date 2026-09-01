from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.corrected_sgta import audit_cecd_execution_dag_v2 as dag


def test_live_v2_audit_preserves_outcome_blind_scope_and_future_waiting() -> None:
    result = dag.audit()
    assert result["scope"] == {
        "cpu_only": True,
        "gpu_touched": False,
        "human_returns_opened": False,
        "admission_decisions_opened": False,
        "model_outputs_opened": False,
        "evaluation_results_opened": False,
        "automatic_gpu_execution_claimed": False,
    }
    assert result["scientific_boundaries"]["future_input_absence_is_not_an_operational_gap"] is True
    assert result["scientific_boundaries"]["human_decisions_synthesized"] is False
    assert result["scientific_boundaries"]["oe_hidden_and_treble_variants_authorized"] is False
    assert result["assurance_boundary"]["zero_blockers_means_static_handoff_closure_only"] is True
    assert set(result["waiting_not_blockers"]) == {
        "clinical_human_returns",
        "listing_human_returns",
        "listing_human_adjudication",
    }
    assert result["generated_artifacts_not_misclassified_as_external_inputs"][
        "three_stage_input_gate"
    ]["classification"] == "generated_scientific_gate_not_external_input"


def test_source_contract_requires_every_marker(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("# alpha beta\n", encoding="utf-8")
    assert dag._source_contract(source, ("alpha", "beta"))["ready"] is True
    failed = dag._source_contract(source, ("alpha", "gamma"))
    assert failed["ready"] is False
    assert failed["markers"] == {"alpha": True, "gamma": False}


def test_source_contract_rejects_marker_only_semantic_fakes(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text(
        "# validate_gate write_once_json\n"
        "def build():\n"
        "    return 'validate_gate write_once_json'\n",
        encoding="utf-8",
    )
    result = dag._source_contract(
        source,
        ("validate_gate", "write_once_json"),
        required_calls={"build": ("validate_gate", "write_once_json")},
    )
    assert result["markers"] == {"validate_gate": True, "write_once_json": True}
    assert result["semantic_calls"]["build"] == {
        "validate_gate": False,
        "write_once_json": False,
    }
    assert result["ready"] is False


def test_monitor_detects_source_newer_than_live_process(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "active.json"
    registry.write_text(
        json.dumps({"active_jobs": ["example-monitor-v1"]}), encoding="utf-8"
    )
    state = dag.ROOT / "corrected_runs/detached_jobs/example-monitor-v1.json"
    source = tmp_path / "monitor.py"
    source.write_text("pass\n", encoding="utf-8")
    state.parent.mkdir(parents=True, exist_ok=True)
    original = state.read_bytes() if state.exists() else None
    try:
        state.write_text(json.dumps({
            "name": "example-monitor-v1",
            "status": "running",
            "child_pid": 424242,
            "started_at": "2000-01-01T00:00:00+00:00",
            "command": ["python", "monitor.py"],
        }), encoding="utf-8")
        monkeypatch.setattr(dag, "_pid_alive", lambda _: True)
        record = dag.monitor_record(
            "example-monitor-", "monitor.py", source, registry=registry
        )
        assert record["ready"] is False
        assert record["checks"]["source_loaded_after_last_edit"] is False
    finally:
        if original is None:
            state.unlink(missing_ok=True)
        else:
            state.write_bytes(original)


def test_monitor_rejects_alive_unrelated_pid_even_with_forged_state(
    tmp_path: Path, monkeypatch
) -> None:
    registry = tmp_path / "active.json"
    registry.write_text(
        json.dumps({"active_jobs": ["example-monitor-v1"]}), encoding="utf-8"
    )
    state = dag.ROOT / "corrected_runs/detached_jobs/example-monitor-v1.json"
    source = tmp_path / "monitor.py"
    source.write_text("pass\n", encoding="utf-8")
    state.parent.mkdir(parents=True, exist_ok=True)
    original = state.read_bytes() if state.exists() else None
    try:
        state.write_text(json.dumps({
            "name": "example-monitor-v1",
            "status": "running",
            "child_pid": 12345,
            "started_at": "2999-01-01T00:00:00+00:00",
            "cwd": str(dag.ROOT),
            "command": ["python", "monitor.py"],
        }), encoding="utf-8")
        monkeypatch.setattr(dag, "_pid_alive", lambda _: True)
        monkeypatch.setattr(
            dag, "_process_record",
            lambda *_: {"inspected": True, "alive": True, "identity_matches": False},
        )
        record = dag.monitor_record(
            "example-monitor-", "monitor.py", source, registry=registry
        )
        assert record["ready"] is False
        assert record["checks"]["live_process_identity_matches"] is False
    finally:
        if original is None:
            state.unlink(missing_ok=True)
        else:
            state.write_bytes(original)


def test_waiting_inputs_are_never_stat_or_opened(monkeypatch) -> None:
    original_exists = Path.exists

    def guarded_exists(path: Path) -> bool:
        if any(path.resolve() == blocked.resolve() for blocked in dag.FORBIDDEN_OUTCOME_PATHS):
            raise AssertionError(f"forbidden presence check: {path}")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    result = dag.audit()
    assert all(
        row["presence"] == "deliberately_not_inspected_outcome_blind"
        for row in result["waiting_not_blockers"].values()
    )


@pytest.mark.parametrize("path", dag.FORBIDDEN_OUTCOME_PATHS)
def test_v2_inherits_forbidden_outcome_fence(path: Path) -> None:
    with pytest.raises(RuntimeError, match="forbidden read"):
        dag.static_record(path / "unread.json" if not path.suffix else path)
