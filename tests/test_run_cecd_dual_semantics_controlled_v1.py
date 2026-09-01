from __future__ import annotations

import fcntl
import hashlib
import json
import sys
from pathlib import Path

import pytest

import anchor.corrected_sgta.run_cecd_dual_semantics_controlled_v1 as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha(path), "bytes": path.stat().st_size}


def _write_worker(path: Path) -> None:
    path.write_text(
        r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def record(path):
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}

def canonical(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

p = argparse.ArgumentParser()
p.add_argument("--describe-runtime", action="store_true")
p.add_argument("--model-family", required=True)
p.add_argument("--preflight", type=Path, required=True)
p.add_argument("--authorization", type=Path)
p.add_argument("--run-contract", type=Path)
p.add_argument("--method")
p.add_argument("--output-dir", type=Path)
p.add_argument("--task", choices=("ce", "full"), default="full")
p.add_argument("--shared-cache-root", type=Path)
a = p.parse_args()
pre = json.loads(a.preflight.read_text())
family = a.model_family
if a.describe_runtime:
    root = a.preflight.parent
    payload = {
        "schema_version": "cecd-dual-semantics-runtime-descriptor-v1",
        "model_family": family,
        "model_id": pre["model_fingerprints"][family]["model_id"],
        "model_fingerprint": pre["model_fingerprints"][family],
        "python_executable": str(Path(sys.executable).resolve()),
        "runtime_versions": {"python": sys.version.split()[0], "backend": "synthetic-test-v1"},
        "source_files": [record(Path(__file__))],
        "input_bindings": {
            key: record(root / (key + ".jsonl"))
            for key in ("calibration_manifest", "evaluation_manifest", "record_keys", "claim_contract")
        },
    }
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0)

contract = json.loads(a.run_contract.read_text())
if a.task == "ce":
    if a.shared_cache_root is None:
        raise SystemExit(19)
    a.shared_cache_root.mkdir(parents=True, exist_ok=True)
    raw_manifest = a.shared_cache_root / "raw_cache_manifest.json"
    if not raw_manifest.exists():
        cell_files = []
        for i in range(30):
            record_key = f"record_{i}"
            for cell in ("h00", "h10", "h01", "h11"):
                cell_path = a.shared_cache_root / "records" / record_key / "cells" / (cell + ".json")
                cell_path.parent.mkdir(parents=True, exist_ok=True)
                cell_path.write_text(json.dumps({"record_key": record_key, "cell": cell}))
                cell_files.append({"record_key": record_key, "cell": cell, "file": record(cell_path)})
        raw = {
            "schema_version": "cecd-dual-semantics-formal-ce-raw-cache-v1",
            "status": "complete",
            "config_fingerprint": "c" * 64,
            "run_fingerprint": contract["fingerprint"],
            "model_family": family,
            "records": 30,
            "clusters": 30,
            "cells": 120,
            "cell_files": cell_files,
            "shared_across_methods": True,
        }
        raw["fingerprint"] = canonical(raw)
        raw_manifest.write_text(json.dumps(raw, sort_keys=True))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    ce_path = a.output_dir / "ce_rows.jsonl"
    ce_path.write_text("".join(
        json.dumps({"row": i, "family": family, "method": a.method}) + "\n"
        for i in range(30)
    ))
    descriptor = contract["runtime_descriptors"][family]
    body = {
        "schema_version": "cecd-dual-semantics-formal-ce-shard-v1",
        "status": "formal_ce_complete_oe_blocked",
        "run_fingerprint": contract["fingerprint"],
        "model_family": family,
        "model_id": contract["models"][family]["model_id"],
        "method": a.method,
        "task": "ce",
        "raw_cache_manifest": record(raw_manifest),
        "ce_output": {
            "path": ce_path.name,
            "sha256": sha(ce_path),
            "bytes": ce_path.stat().st_size,
        },
        "rows": 30,
        "clusters": 30,
        "worker_sha256": sha(Path(__file__)),
        "runtime_descriptor_sha256": canonical(descriptor),
        "oe_implemented": False,
        "hidden_intervention_implemented": False,
        "paper_native_treble_claimed": False,
    }
    body["completion_fingerprint"] = canonical(body)
    (a.output_dir / "ce_completion.json").write_text(json.dumps(body, sort_keys=True))
    raise SystemExit(0)
fail_once = a.preflight.parent / "fail_once.flag"
if a.method == "fail_once" and not fail_once.exists():
    fail_once.write_text("auditable synthetic failure\n")
    raise SystemExit(17)
a.output_dir.mkdir(parents=True, exist_ok=True)
outputs = {}
for task in ("ce", "oe"):
    path = a.output_dir / (task + ".jsonl")
    path.write_text(json.dumps({"task": task, "family": family, "method": a.method}) + "\n")
    outputs[task] = {
        "path": path.name,
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "rows": 40,
        "clusters": 40,
    }
descriptor = contract["runtime_descriptors"][family]
body = {
    "schema_version": "cecd-dual-semantics-arm-shard-v1",
    "status": "complete",
    "run_fingerprint": contract["fingerprint"],
    "model_family": family,
    "model_id": contract["models"][family]["model_id"],
    "method": a.method,
    "tasks": ["ce", "oe"],
    "task_outputs": outputs,
    "observed_compute_ledger": {"target_forwards": 40, "wall_seconds": 0.01},
    "worker_sha256": sha(Path(__file__)),
    "runtime_descriptor_sha256": canonical(descriptor),
}
body["completion_fingerprint"] = canonical(body)
(a.output_dir / "completion.json").write_text(json.dumps(body, sort_keys=True))
''',
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, methods=("alpha", "beta")) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for name in (
        "calibration_manifest",
        "evaluation_manifest",
        "record_keys",
        "claim_contract",
    ):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps({"input": name}) + "\n", encoding="utf-8")
        inputs[name] = path
    models = {}
    for index, family in enumerate(("huatuo", "hulu"), 1):
        models[family] = {
            "model_id": f"{family}:synthetic",
            "checkpoint_sha256": str(index) * 64,
            "processor_sha256": str(index + 1) * 64,
            "template_sha256": str(index + 2) * 64,
            "generation_contract_sha256": str(index + 3) * 64,
            "hook_contract_sha256": str(index + 4) * 64,
            "vision_token_transport_contract_sha256": str(index + 5) * 64,
        }
    output_root = tmp_path / "outputs"
    preflight = {
        "methods": list(methods),
        "model_fingerprints": models,
        "method_output_root": str(output_root.resolve()),
        "calibration_manifest_sha256": _sha(inputs["calibration_manifest"]),
        "evaluation_manifest_sha256": _sha(inputs["evaluation_manifest"]),
        "record_keys_sha256": _sha(inputs["record_keys"]),
        "claim_contract_sha256": _sha(inputs["claim_contract"]),
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    bound = {}
    for name in (
        "stage1_analysis", "stage1_input_gate", "admission", "preflight_build"
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"sealed_fixture": name}), encoding="utf-8")
        bound[name] = _record(path)
    authorization = {
        "version": runner.AUTHORIZATION_VERSION,
        "status": "controlled_dual_semantics_comparison_authorized",
        **bound,
        "preflight": _record(preflight_path),
        "method_output_root": str(output_root.resolve()),
        "allowed_methods": list(methods),
        "required_outcome_schema": runner.DUAL_SEMANTICS_OUTCOME_SCHEMA,
        "controlled_method_comparison_authorized": True,
        "cecd_hidden_state_intervention_authorized_only_inside_locked_comparison": True,
        "locked_test_behavioral_increment_confirmed": True,
        "full_method_gate_authorized": False,
        "oral_baseline_closure_authorized": False,
        "official_compatible_dynamic_activation_baseline_present": False,
        "general_hidden_state_stage_authorized": False,
        "paper_native_treble_authorized": False,
        "exact_treble_authorized": False,
        "general_gpu_authorized": False,
        "paper_claim_authorized": False,
        "method_outputs_consumed": False,
    }
    authorization["fingerprint"] = runner.canonical_sha256(authorization)
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    worker = tmp_path / "synthetic_worker.py"
    _write_worker(worker)
    return {
        "authorization": authorization_path,
        "preflight": preflight_path,
        "output_root": output_root,
        "worker": worker,
        "gpu_lock": tmp_path / runner.GPU_LOCK_RELATIVE,
        "root": tmp_path,
        "inputs": inputs,
    }


def _call(
    paths: dict,
    *,
    execute=False,
    execute_ce_only=False,
    resume_after_failure=False,
):
    return runner.prepare_or_run(
        authorization_path=paths["authorization"],
        preflight_path=paths["preflight"],
        worker=paths["worker"],
        huatuo_python=Path(sys.executable),
        hulu_python=Path(sys.executable),
        gpu_lock=paths["gpu_lock"],
        root=paths["root"],
        execute=execute,
        execute_ce_only=execute_ce_only,
        resume_after_failure=resume_after_failure,
    )


def test_refuses_before_authorization_without_touching_worker_or_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    paths["authorization"].unlink()
    monkeypatch.setattr(
        runner, "validate_dual_semantics_preflight_contract", lambda _payload: None
    )
    with pytest.raises(runner.ControlledRunError, match="not authorized"):
        _call(paths)
    assert not paths["output_root"].exists()
    assert not paths["gpu_lock"].exists()


def test_rejects_noncanonical_gpu_lock_before_authorization_or_worker(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    paths["gpu_lock"] = tmp_path / "gpu0-cecd-dual-semantics-v1.lock"
    monkeypatch.setattr(
        runner, "validate_dual_semantics_preflight_contract", lambda _payload: None
    )
    with pytest.raises(runner.ControlledRunError, match="GPU lock drift"):
        _call(paths)
    assert not paths["output_root"].exists()
    assert not paths["gpu_lock"].exists()


def test_prepare_hash_binds_runtime_inputs_models_code_and_generic_closure(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path, methods=("static", "dynamic", "joint"))
    monkeypatch.setattr(
        runner, "validate_dual_semantics_preflight_contract", lambda _payload: None
    )
    result = _call(paths)
    assert result["status"] == "prepared_outcome_blind_no_gpu_launched"
    assert result["methods"] == ["static", "dynamic", "joint"]
    assert result["arm_count"] == 6
    assert not paths["gpu_lock"].exists()
    contract = json.loads((paths["output_root"] / "run_contract.json").read_text())
    assert contract["general_gpu_authorized"] is False
    assert set(contract["runtime_descriptors"]) == {"huatuo", "hulu"}
    assert contract["source_files"]["worker"]["sha256"] == _sha(paths["worker"])
    assert len(contract["arm_order"]) == 2 * len(contract["methods"])


def test_authorization_method_mismatch_is_fail_closed(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path)
    authorization = json.loads(paths["authorization"].read_text())
    authorization["allowed_methods"] = ["alpha"]
    authorization.pop("fingerprint")
    authorization["fingerprint"] = runner.canonical_sha256(authorization)
    paths["authorization"].write_text(json.dumps(authorization))
    monkeypatch.setattr(
        runner, "validate_dual_semantics_preflight_contract", lambda _payload: None
    )
    with pytest.raises(runner.ControlledRunError, match="method order and closure"):
        _call(paths)


def test_executes_exact_two_model_method_closure_and_resume_skips_valid_arms(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        runner, "validate_dual_semantics_preflight_contract", lambda _payload: None
    )
    _call(paths)
    manifest = _call(paths, execute=True)
    assert manifest["status"] == "complete"
    assert manifest["arm_count"] == 4
    assert manifest["strict_model_method_closure"] is True
    before = {
        str(path.relative_to(paths["output_root"])): path.stat().st_mtime_ns
        for path in paths["output_root"].glob("arms/*/*/completion.json")
    }
    replay = _call(paths, execute=True)
    after = {
        str(path.relative_to(paths["output_root"])): path.stat().st_mtime_ns
        for path in paths["output_root"].glob("arms/*/*/completion.json")
    }
    assert replay == manifest
    assert before == after


def test_input_or_worker_drift_rejected_before_resume(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        runner, "validate_dual_semantics_preflight_contract", lambda _payload: None
    )
    _call(paths)
    paths["inputs"]["claim_contract"].write_text("drift\n")
    with pytest.raises(runner.ControlledRunError, match="does not match frozen preflight"):
        _call(paths, execute=True)

    paths = _fixture(tmp_path / "worker_drift")
    _call(paths)
    paths["worker"].write_text(paths["worker"].read_text() + "\n# drift\n")
    with pytest.raises(runner.ControlledRunError, match="run-contract drift"):
        _call(paths, execute=True)


def test_worker_failure_is_terminal_until_explicit_audited_resume(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path, methods=("ok", "fail_once"))
    monkeypatch.setattr(
        runner, "validate_dual_semantics_preflight_contract", lambda _payload: None
    )
    _call(paths)
    with pytest.raises(runner.ControlledRunError, match="failed-stop"):
        _call(paths, execute=True)
    failures = list((paths["output_root"] / "failures").glob("attempt_*.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["automatic_retry_authorized"] is False
    with pytest.raises(runner.ControlledRunError, match="resume-after-failure"):
        _call(paths, execute=True)
    manifest = _call(paths, execute=True, resume_after_failure=True)
    assert manifest["arm_count"] == 4
    assert len(list((paths["output_root"] / "failures").glob("attempt_*.json"))) == 1


def test_gpu_flock_collision_fails_before_arm_launch(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        runner, "validate_dual_semantics_preflight_contract", lambda _payload: None
    )
    _call(paths)
    paths["gpu_lock"].parent.mkdir(parents=True, exist_ok=True)
    paths["gpu_lock"].touch()
    with paths["gpu_lock"].open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(runner.ControlledRunError, match="GPU lock is already held"):
            _call(paths, execute=True)
    assert not list(paths["output_root"].glob("arms/*/*/completion.json"))


def test_ce_only_stage_runs_seven_logit_controls_from_one_cache_per_model(
    tmp_path: Path, monkeypatch
) -> None:
    methods = (
        "unmitigated",
        "full_orbit",
        "render_only",
        "prompt_only",
        "random_norm",
        "sign_permuted",
        "main_effect_removal",
        "cecd_interaction_projection",
        "treble_proceedings",
        "treble_released",
    )
    paths = _fixture(tmp_path, methods=methods)
    monkeypatch.setattr(
        runner, "validate_dual_semantics_preflight_contract", lambda _payload: None
    )
    _call(paths)
    manifest = _call(paths, execute_ce_only=True)

    assert manifest["status"] == "formal_ce_complete_oe_and_hidden_methods_blocked"
    assert manifest["arm_count"] == 14
    assert manifest["methods"] == list(runner.IMPLEMENTED_KERNEL_METHODS)
    assert manifest["blocked_methods"] == [
        "cecd_interaction_projection",
        "treble_proceedings",
        "treble_released",
    ]
    assert manifest["operation_space"] == "centered tri-state next-token logits"
    assert manifest["oe_implemented"] is False
    assert manifest["hidden_intervention_implemented"] is False
    assert manifest["paper_native_treble_claimed"] is False
    assert manifest["paper_claim_authorized"] is False
    assert not (paths["output_root"] / "run_manifest.json").exists()
    assert len(list(paths["output_root"].glob("ce_arms/*/*/ce_completion.json"))) == 14
    assert len(list(paths["output_root"].glob("shared_ce_cache/*/raw_cache_manifest.json"))) == 2
    for family in ("huatuo", "hulu"):
        hashes = {
            row["raw_cache_manifest_sha256"]
            for row in manifest["arms"]
            if row["model_family"] == family
        }
        assert len(hashes) == 1

    before = {
        str(path.relative_to(paths["output_root"])): path.stat().st_mtime_ns
        for path in paths["output_root"].glob("ce_arms/*/*/ce_completion.json")
    }
    replay = _call(paths, execute_ce_only=True)
    after = {
        str(path.relative_to(paths["output_root"])): path.stat().st_mtime_ns
        for path in paths["output_root"].glob("ce_arms/*/*/ce_completion.json")
    }
    assert replay == manifest
    assert after == before
