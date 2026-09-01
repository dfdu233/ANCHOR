import json
from pathlib import Path

import pytest

import anchor.corrected_sgta.run_vindr_cecd_listing_pipeline_v1 as module
from anchor.corrected_sgta.run_vindr_cecd_listing_pipeline_v1 import (
    DEFAULT_GPU_LOCK,
    MODELS,
    STAGES,
    SchedulerError,
    execute_scheduler,
    prepare_scheduler_handoff,
)
from anchor.corrected_sgta.validate_vindr_cecd_listing_scientific_admission_v1 import (
    sha256_file,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch):
    files = {
        name: tmp_path / name
        for name in ("receipt.json", "handoff.json", "upstream.json", "confirmation.json")
    }
    for path in files.values():
        _json(path, {})
    pack = tmp_path / "pack"
    _json(pack / "manifest.json", {})
    reference = tmp_path / "reference.jsonl"
    reference.write_text("{}\n", encoding="utf-8")
    experiment = tmp_path / "experiment.json"
    _json(experiment, {"reference_contract": {"reference_file_sha256": sha256_file(reference)}})
    calls = []

    def validate(**kwargs):
        calls.append(kwargs)
        return {"upstream": {"confirmation_path": files["confirmation.json"]}}

    monkeypatch.setattr(module, "validate_scientific_admission", validate)
    handoff = tmp_path / "scheduler_handoff.json"
    plan = prepare_scheduler_handoff(
        receipt=files["receipt.json"],
        expected_receipt_sha256=sha256_file(files["receipt.json"]),
        adjudication_handoff=files["handoff.json"],
        expected_adjudication_handoff_sha256=sha256_file(files["handoff.json"]),
        upstream_gate=files["upstream.json"],
        expected_upstream_gate_sha256=sha256_file(files["upstream.json"]),
        pack_dir=pack,
        experiment_manifest=experiment,
        reference=reference,
        output_root=tmp_path / "runs",
        handoff_path=handoff,
    )
    calls.clear()
    return handoff, sha256_file(handoff), plan, calls


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _complete(run_dir: Path, *, model: str, stage: str, admission_sha: str) -> None:
    shard = run_dir / "cell_shards" / "image" / "cell.json"
    _json(shard, {"fake": "completion-metadata-test-only"})
    manifest = {
        "model_id": model,
        "split": stage,
        "admission_sha256": admission_sha,
        "gpu_lock": str(DEFAULT_GPU_LOCK.resolve()),
    }
    manifest["fingerprint"] = module.canonical_json_sha256(manifest)
    _json(run_dir / "run_manifest.json", manifest)
    inventory = [{"path": str(shard.relative_to(run_dir)), "sha256": sha256_file(shard)}]
    completion = {
        "status": "complete_eligible_orbits_only",
        "scientific_model": True,
        "config_fingerprint": manifest["fingerprint"],
        "cell_shards": 1,
        "shard_inventory": inventory,
    }
    completion["fingerprint"] = module.canonical_json_sha256(completion)
    _json(run_dir / "completion.json", completion)


def test_prepare_is_inert_and_execute_is_serial_stage_gated_with_shared_lock(
    tmp_path: Path, monkeypatch
) -> None:
    handoff, digest, plan, validations = _fixture(tmp_path, monkeypatch)
    assert not Path(plan["output_root"]).exists()
    commands = []

    def runner(command):
        command = list(command)
        commands.append(command)
        _complete(
            Path(_argument(command, "--output-dir")),
            model=_argument(command, "--model"),
            stage=_argument(command, "--split"),
            admission_sha=plan["admission_receipt"]["sha256"],
        )

    result = execute_scheduler(
        handoff_path=handoff, expected_handoff_sha256=digest, command_runner=runner
    )
    assert result["status"] == "pilot_dev_confirmation_two_model_pipeline_complete"
    assert [
        f"{_argument(command, '--split')}:{_argument(command, '--model')}"
        for command in commands
    ] == [f"{stage}:{model}" for stage in STAGES for model in MODELS]
    assert all(
        _argument(command, "--gpu-lock") == str(DEFAULT_GPU_LOCK)
        for command in commands
    )
    # Initial, each stage, and immediately before each of six launches.
    assert len(validations) == 1 + len(STAGES) + len(STAGES) * len(MODELS)
    for stage in STAGES:
        gate = json.loads(
            (Path(plan["output_root"]) / "stage_gates" / f"{stage}.json").read_text()
        )
        assert gate["status"] == "two_models_hash_complete"
        assert [row["model"] for row in gate["models"]] == list(MODELS)
        assert gate["simultaneous_models_used"] is False

    validations.clear()
    resumed_commands = []
    execute_scheduler(
        handoff_path=handoff,
        expected_handoff_sha256=digest,
        command_runner=lambda command: resumed_commands.append(command),
    )
    assert resumed_commands == []
    assert len(validations) == 1 + len(STAGES)


def test_partial_completion_never_relaunches_or_overwrites(tmp_path: Path, monkeypatch) -> None:
    handoff, digest, plan, _ = _fixture(tmp_path, monkeypatch)
    run_dir = Path(plan["output_root"]) / "huatuo" / "pilot"
    _json(run_dir / "run_manifest.json", {"partial": True})
    calls = []
    with pytest.raises(SchedulerError, match="partial completion"):
        execute_scheduler(
            handoff_path=handoff,
            expected_handoff_sha256=digest,
            command_runner=lambda command: calls.append(command),
        )
    assert calls == []
    assert not (run_dir / "completion.json").exists()


def test_existing_tampered_completion_fails_instead_of_replaying(tmp_path: Path, monkeypatch) -> None:
    handoff, digest, plan, _ = _fixture(tmp_path, monkeypatch)
    run_dir = Path(plan["output_root"]) / "huatuo" / "pilot"
    _complete(
        run_dir,
        model="huatuo",
        stage="pilot",
        admission_sha=plan["admission_receipt"]["sha256"],
    )
    completion = json.loads((run_dir / "completion.json").read_text())
    completion["cell_shards"] = 99
    _json(run_dir / "completion.json", completion)
    calls = []
    with pytest.raises(SchedulerError, match="completion contract mismatch"):
        execute_scheduler(
            handoff_path=handoff,
            expected_handoff_sha256=digest,
            command_runner=lambda command: calls.append(command),
        )
    assert calls == []


def test_shard_hash_tamper_is_not_treated_as_resumable_absence(tmp_path: Path, monkeypatch) -> None:
    handoff, digest, plan, _ = _fixture(tmp_path, monkeypatch)
    run_dir = Path(plan["output_root"]) / "huatuo" / "pilot"
    _complete(
        run_dir,
        model="huatuo",
        stage="pilot",
        admission_sha=plan["admission_receipt"]["sha256"],
    )
    shard = next((run_dir / "cell_shards").rglob("*.json"))
    shard.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SchedulerError, match="shard hash"):
        execute_scheduler(
            handoff_path=handoff,
            expected_handoff_sha256=digest,
            command_runner=lambda _: pytest.fail("tampered run must not relaunch"),
        )
