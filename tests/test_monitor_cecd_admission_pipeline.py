import json
import subprocess
from pathlib import Path

import pytest

import scripts.monitor_cecd_admission_pipeline as monitor


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _delivery(path: Path, browser_passed: bool = True) -> None:
    _json(path / "delivery_index.json", {"version": "fixture"})
    index_hash = monitor.sha256_file(path / "delivery_index.json")
    _json(
        path / "verification.json",
        {
            "version": monitor.EXPECTED_DELIVERY_VERIFICATION,
            "delivery_index_sha256": index_hash,
            "passed": True,
        },
    )
    _json(
        path / "browser_smoke.json",
        {
            "version": monitor.EXPECTED_BROWSER_SMOKE,
            "delivery_index_sha256": index_hash,
            "verification_sha256": monitor.sha256_file(path / "verification.json"),
            "passed": browser_passed,
        },
    )


def test_monitor_waits_for_verified_delivery_then_all_human_files(tmp_path: Path) -> None:
    delivery, inbox, output = tmp_path / "delivery", tmp_path / "inbox", tmp_path / "output"
    state = monitor.advance(tmp_path / "pack", delivery, inbox, output)
    assert state["stage"] == "waiting_for_verified_reviewer_deliveries"
    _delivery(delivery)
    state = monitor.advance(tmp_path / "pack", delivery, inbox, output)
    assert state["stage"] == "waiting_for_four_independent_returns"
    assert len(state["missing"]) == 8


def test_delivery_gate_rejects_failed_browser_smoke(tmp_path: Path) -> None:
    _delivery(tmp_path, browser_passed=False)
    with pytest.raises(RuntimeError, match="failed verifier or browser smoke"):
        monitor.delivery_ready(tmp_path)


def test_failed_stage_is_terminal_and_never_retried(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    state = tmp_path / f"corrected_runs/detached_jobs/{monitor.STAGE_JOB}.json"
    _json(
        state,
        {
            "name": monitor.STAGE_JOB,
            "status": "failed",
            "exit_code": 7,
        },
    )
    result = monitor.launch_or_monitor_stage(tmp_path / "analysis.json")
    assert result == {
        "stage": "two_model_stage1_failed_terminal",
        "stage_job": monitor.STAGE_JOB,
        "exit_code": 7,
        "retry_authorized": False,
    }


def test_legacy_huatuo_wrapper_is_hard_retired_before_any_work() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "scripts/run_cecd_huatuo_stage1_v1.sh")],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert "RETIRED" in result.stderr
    assert "no command was run" in result.stderr


def test_done_stage_without_input_gate_cannot_mark_complete(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    state = tmp_path / f"corrected_runs/detached_jobs/{monitor.STAGE_JOB}.json"
    _json(
        state,
        {"name": monitor.STAGE_JOB, "status": "done", "exit_code": 0},
    )
    result = tmp_path / "corrected_runs/vindr_v2/cecd_three_stage_v3/confirmation_locked.json"
    _json(result, {"status": "complete"})
    with pytest.raises(RuntimeError, match="input gate is missing"):
        monitor.launch_or_monitor_stage(tmp_path / "admission.json")


def test_stage_validator_binds_admission_raw_runs_and_frozen_statistics(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    admission = tmp_path / "admission.json"
    _json(admission, {"passed": True})
    analyzer = (
        tmp_path
        / "anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py"
    )
    analyzer.parent.mkdir(parents=True)
    analyzer.write_text("# frozen analyzer fixture\n")
    runs = []
    models = {}
    for family in ("huatuo", "hulu"):
        run_dir = tmp_path / family
        rows = run_dir / "factorial_rows.jsonl"
        rows.parent.mkdir(parents=True)
        rows.write_text(f'{{"family":"{family}"}}\n')
        digest = monitor.sha256_file(rows)
        model = f"{family}:fixture"
        runs.append(
            {
                "family": family,
                "run_dir": str(run_dir.resolve()),
                "factorial_rows_sha256": digest,
                "model": model,
            }
        )
        models[model] = {}
    gate = (
        tmp_path / "corrected_runs/vindr_v2/cecd_three_stage_v3/input_gate.json"
    )
    _json(
        gate,
        {
            "version": monitor.EXPECTED_STAGE_GATE,
            "status": "passed",
            "passed": True,
            "hidden_state_authorized": False,
            "legacy_pilot_as_dev_authorized": False,
            "admission": {"sha256": monitor.sha256_file(admission)},
            "runs": {"confirmation_locked": runs},
            "confirmation_locked": {},
        },
    )
    result_path = gate.with_name("confirmation_locked.json")
    _json(
        result_path,
        {
            "version": monitor.EXPECTED_STAGE_ANALYSIS,
            "status": "complete",
            "stage_label": "confirmation_locked",
            "source_manifest_split": "confirmation",
            "models": models,
            "provenance": {
                "code_sha256": monitor.sha256_file(analyzer),
                "seed": 42,
                "bootstrap_draws": 5000,
                "mode": "confirmation_locked",
            },
            "gate": {
                "name": "behavioral_confirmation_locked_v1",
                "confirmation_passing_models": list(models),
                "both_models_pass": True,
                "authorized_for_method_level_treble_adapter_run": True,
                "authorized_for_hidden_state_stage": False,
            },
            "exact_treble_method_collision": {"hidden_state_authorized": False},
        },
    )
    gate_payload = json.loads(gate.read_text())
    gate_payload["confirmation_locked"] = {
        "path": str(result_path.resolve()),
        "sha256": monitor.sha256_file(result_path),
    }
    _json(gate, gate_payload)
    assert monitor.validate_stage_result(
        result_path=result_path, admission=admission
    )["gate"]["authorized_for_method_level_treble_adapter_run"] is True
    result = json.loads(result_path.read_text())
    result["provenance"]["mode"] = "dev_fit"
    _json(result_path, result)
    gate_payload["confirmation_locked"]["sha256"] = monitor.sha256_file(result_path)
    _json(gate, gate_payload)
    with pytest.raises(RuntimeError, match="locked-confirmation provenance/contract mismatch"):
        monitor.validate_stage_result(result_path=result_path, admission=admission)
    result["provenance"]["mode"] = "confirmation_locked"
    _json(result_path, result)
    gate_payload["confirmation_locked"]["sha256"] = monitor.sha256_file(result_path)
    _json(gate, gate_payload)
    admission.write_text("changed\n")
    with pytest.raises(RuntimeError, match="three-stage input gate contract mismatch"):
        monitor.validate_stage_result(result_path=result_path, admission=admission)


def test_first_complete_human_bundle_is_frozen_across_polls(tmp_path: Path) -> None:
    completed = {}
    attestations = {}
    for role in monitor.ROLES:
        completed[role] = tmp_path / "inbox" / f"{role}.csv"
        attestations[role] = tmp_path / "inbox" / f"{role}.json"
        completed[role].parent.mkdir(parents=True, exist_ok=True)
        completed[role].write_text(f"completed-{role}\n")
        attestations[role].write_text(f'{{"role":"{role}"}}\n')
    frozen, frozen_attestations, lock = monitor.freeze_human_bundle(
        output=tmp_path / "output",
        completed=completed,
        attestations=attestations,
    )
    assert lock.is_file()
    assert set(frozen) == set(monitor.ROLES)
    assert set(frozen_attestations) == set(monitor.ROLES)
    completed["clinical_reviewer_1"].write_text("changed-after-first-freeze\n")
    with pytest.raises(RuntimeError, match="write-once collision"):
        monitor.freeze_human_bundle(
            output=tmp_path / "output",
            completed=completed,
            attestations=attestations,
        )


def test_source_pack_closure_rejects_working_artifact_drift(tmp_path: Path) -> None:
    pack, output = tmp_path / "pack", tmp_path / "output"
    pack.mkdir()
    for name in (
        "REVIEW_INSTRUCTIONS.md",
        "clinical_reviewer_1.csv",
        "clinical_reviewer_2.csv",
        "clinical_template_reviewer.csv",
        "language_annotator.csv",
        "manifest.json",
        "sealed_mapping.json",
        "selected_claims.sealed.jsonl",
    ):
        (pack / name).write_text(f"fixture-{name}\n")
    _json(
        output / "pack_integrity.json",
        {
            "protocol_version": "cecd-admission-pack-integrity-v1",
            "passed": True,
            "review_sheets_blank": True,
            "reviewer_visible_leakage_checks_passed": True,
            "pack": str(pack.resolve()),
            "manifest_sha256": monitor.sha256_file(pack / "manifest.json"),
            "sealed_mapping_sha256": monitor.sha256_file(pack / "sealed_mapping.json"),
        },
    )
    monitor.validate_pack_closure(pack, output)
    (pack / "clinical_reviewer_1.csv").write_text("drifted\n")
    with pytest.raises(RuntimeError, match="write-once collision"):
        monitor.validate_pack_closure(pack, output)


def test_existing_admission_analysis_is_replayed_not_trusted(
    tmp_path: Path, monkeypatch
) -> None:
    pack, delivery, output = tmp_path / "pack", tmp_path / "delivery", tmp_path / "output"
    pack.mkdir()
    (pack / "sealed_mapping.json").write_text("{}\n")
    completed = {}
    attestations = {}
    for role in monitor.ROLES:
        completed[role] = tmp_path / "inbox" / f"{role}.csv"
        attestations[role] = tmp_path / "inbox" / f"{role}.json"
        completed[role].parent.mkdir(parents=True, exist_ok=True)
        completed[role].write_text(f"completed-{role}\n")
        attestations[role].write_text(f'{{"role":"{role}"}}\n')
    verification, browser = delivery / "verification.json", delivery / "browser.json"
    _json(verification, {"passed": True})
    _json(browser, {"passed": True})
    def fake_pack_closure(_pack, target):
        assert (target / "human_return_bundle_lock.json").is_file()
        _json(target / "pack_source_lock.json", {"version": "fixture"})
        return {}

    monkeypatch.setattr(monitor, "validate_pack_closure", fake_pack_closure)
    monkeypatch.setattr(
        monitor,
        "validate_all",
        lambda **_kwargs: {
            "version": monitor.VALIDATION_VERSION,
            "status": "four_independent_returns_validated",
            "roles": [],
        },
    )
    monkeypatch.setattr(
        monitor,
        "delivery_ready",
        lambda _path: {
            "ready": True,
            "verification": monitor.record(verification),
            "browser_smoke": monitor.record(browser),
        },
    )

    def fake_run(command, **_kwargs):
        replay = Path(command[-1])
        clinical = [Path(command[command.index("--clinical-review") + 1])]
        second_index = command.index("--clinical-review", command.index("--clinical-review") + 1)
        clinical.append(Path(command[second_index + 1]))
        template = Path(command[command.index("--clinical-template-review") + 1])
        language = Path(command[command.index("--language-review") + 1])
        _json(
            replay,
            {
                "version": monitor.EXPECTED_ADMISSION_ANALYSIS,
                "status": "failed",
                "passed": False,
                "cecd_model_scoring_authorized": False,
                "provenance": {
                    "sealed_mapping": str((pack / "sealed_mapping.json").resolve()),
                    "sealed_mapping_sha256": monitor.sha256_file(pack / "sealed_mapping.json"),
                    "clinical_reviews": [monitor.record(path) for path in clinical],
                    "clinical_template_review": monitor.record(template),
                    "language_review": monitor.record(language),
                },
            },
        )
        return subprocess.CompletedProcess(command, 2, "", "")

    monkeypatch.setattr(monitor.subprocess, "run", fake_run)
    first = monitor.analyze_admission(
        pack=pack,
        delivery=delivery,
        output=output,
        completed=completed,
        attestations=attestations,
    )
    assert first["stage"] == "human_admission_failed_terminal"
    forged = json.loads((output / "analysis.json").read_text())
    forged["passed"] = True
    forged["status"] = "passed"
    _json(output / "analysis.json", forged)
    with pytest.raises(RuntimeError, match="write-once collision"):
        monitor.analyze_admission(
            pack=pack,
            delivery=delivery,
            output=output,
            completed=completed,
            attestations=attestations,
        )
