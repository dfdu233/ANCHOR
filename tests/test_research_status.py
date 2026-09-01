import json
from pathlib import Path

from scripts import research_status


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_detached_summary_downgrades_dead_running_state_to_stale(
    tmp_path: Path, monkeypatch
) -> None:
    _json(
        tmp_path / "configs/research_active_jobs.json",
        {"active_jobs": ["dead-active", "live-active", "done-active"]},
    )
    state_dir = tmp_path / "corrected_runs/detached_jobs"
    _json(
        state_dir / "dead-active.json",
        {"name": "dead-active", "status": "running", "child_pid": 111},
    )
    _json(
        state_dir / "live-active.json",
        {"name": "live-active", "status": "running", "child_pid": 222},
    )
    _json(
        state_dir / "done-active.json",
        {"name": "done-active", "status": "done", "child_pid": 333},
    )
    _json(
        state_dir / "dead-history.json",
        {"name": "dead-history", "status": "running", "child_pid": 444},
    )
    monkeypatch.setattr(research_status, "alive", lambda pid: pid == 222)

    rows, hidden = research_status.detached_job_rows(tmp_path)
    by_name = {row["name"]: row for row in rows}
    assert by_name["dead-active"] == {
        "name": "dead-active",
        "raw_status": "running",
        "status": "stale",
        "pid": 111,
        "live": False,
    }
    assert by_name["live-active"]["status"] == "running"
    assert by_name["live-active"]["live"] is True
    assert by_name["done-active"]["status"] == "done"
    assert hidden == 1


def test_detached_summary_falls_back_to_supervisor_when_child_is_absent(
    tmp_path: Path, monkeypatch
) -> None:
    _json(
        tmp_path / "configs/research_active_jobs.json",
        {"active_jobs": ["supervisor-only"]},
    )
    _json(
        tmp_path / "corrected_runs/detached_jobs/supervisor-only.json",
        {
            "name": "supervisor-only",
            "status": "starting",
            "pid": 555,
            "child_pid": None,
        },
    )
    monkeypatch.setattr(research_status, "alive", lambda pid: pid == 555)
    rows, hidden = research_status.detached_job_rows(tmp_path)
    assert rows[0]["pid"] == 555
    assert rows[0]["status"] == "starting"
    assert rows[0]["live"] is True
    assert hidden == 0


def test_running_state_without_workload_child_is_stale(
    tmp_path: Path, monkeypatch
) -> None:
    _json(
        tmp_path / "configs/research_active_jobs.json",
        {"active_jobs": ["orphan-supervisor"]},
    )
    _json(
        tmp_path / "corrected_runs/detached_jobs/orphan-supervisor.json",
        {
            "name": "orphan-supervisor",
            "status": "running",
            "pid": 777,
            "child_pid": None,
        },
    )
    monkeypatch.setattr(research_status, "alive", lambda pid: pid == 777)
    rows, hidden = research_status.detached_job_rows(tmp_path)
    assert rows[0]["pid"] is None
    assert rows[0]["status"] == "stale"
    assert rows[0]["live"] is False
    assert hidden == 0


def test_reader_boundary_frozen_rejection_overrides_missing_outputs_and_active_names(
    tmp_path: Path,
) -> None:
    _json(
        tmp_path / "configs/research_active_jobs.json",
        {
            "active_jobs": [
                "vindr-v2-hulu-dev-and-screens-v3",
                "vindr-v2-confirmation-collections-v3",
            ]
        },
    )
    _json(
        tmp_path / "corrected_runs/paper/iclr_oral_completion_audit_v1/audit.json",
        {
            "version": "iclr-oral-completion-audit-v1",
            "current_authoritative_verdict": "REJECT_CURRENT_READER_BOUNDARY_AND_PIVOT",
            "killed_branches": {
                "reader_early_erasure": "data-refuted at frozen Huatuo development gate"
            },
        },
    )
    statuses = dict(research_status.reader_boundary_progress(tmp_path))
    assert len(statuses) == 5
    assert all(value.startswith("terminal/rejected") for value in statuses.values())
    assert all("pending" not in value for value in statuses.values())
    assert all("0/1920" not in value for value in statuses.values())


def test_reader_boundary_absence_is_inactive_without_active_manifest_job(
    tmp_path: Path,
) -> None:
    _json(
        tmp_path / "configs/research_active_jobs.json",
        {"active_jobs": []},
    )
    partial = (
        tmp_path
        / "corrected_runs/vindr_v2/hidden_confirmation_huatuo_all_findings_v1/metadata.jsonl"
    )
    partial.parent.mkdir(parents=True)
    partial.write_text('{}\n{}\n', encoding="utf-8")
    statuses = dict(research_status.reader_boundary_progress(tmp_path))
    assert statuses["VinDr Hulu dev residual screen"] == "inactive/not-authorized"
    assert statuses["VinDr huatuo confirmation"] == "inactive/not-authorized"
    assert statuses["VinDr hulu confirmation"] == "inactive/not-authorized"
    assert statuses["VinDr two-model boundary synthesis"] == "inactive/not-authorized"
