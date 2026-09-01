import pytest

from scripts.research_watchdog import merged_job_names, recovery_decision


def test_watchdog_leaves_terminal_and_failed_jobs_for_audit() -> None:
    assert recovery_decision({"status": "done"}) == "terminal"
    assert recovery_decision({"status": "failed"}) == "failed"


def test_watchdog_refuses_malformed_recovery_command() -> None:
    row = {"status": "running", "pid": -1, "command": "python bad.py"}
    assert recovery_decision(row) == "invalid"


def test_watchdog_recovers_only_dead_nonterminal_job() -> None:
    row = {"status": "running", "pid": -1, "command": ["python", "job.py"]}
    assert recovery_decision(row) == "recover"


def test_required_jobs_survive_an_older_active_manifest() -> None:
    assert merged_job_names(
        {"active_jobs": ["historical", "shared"]},
        {"required_jobs": ["shared", "formal-successor"]},
    ) == ["historical", "shared", "formal-successor"]


def test_required_job_contract_is_validated() -> None:
    with pytest.raises(ValueError, match="required_jobs"):
        merged_job_names({"active_jobs": []}, {"required_jobs": "bad"})
