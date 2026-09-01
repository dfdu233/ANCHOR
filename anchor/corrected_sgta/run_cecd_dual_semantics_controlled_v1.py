#!/usr/bin/env python3
"""Fail-closed executor for the authorized CECD dual-semantics envelope.

The executor is deliberately outcome-blind.  It freezes and verifies the
scientific worker, both model runtimes, input bindings, model fingerprints and
the exact model x method closure before it launches an arm.  It then treats
each model/method arm as one crash-resumable atomic shard.  Clinical metrics
and the collision verdict remain a separate post-run responsibility.

This file does not grant authority.  It accepts only the narrow write-once
authorization emitted by ``authorize_cecd_dual_semantics_preflight_v1`` and
never treats the authorization's explicitly false general-GPU flag as a
general permission.  GPU use is limited to the controlled comparison named in
that artifact.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from anchor.corrected_sgta.authorize_cecd_dual_semantics_preflight_v1 import (
    VERSION as AUTHORIZATION_VERSION,
)
from anchor.corrected_sgta.cecd_dual_semantics_kernels_v1 import (
    IMPLEMENTED_KERNEL_METHODS,
)
from anchor.corrected_sgta.treble_collision_contract import (
    DUAL_SEMANTICS_OUTCOME_SCHEMA,
    validate_dual_semantics_preflight_contract,
)


VERSION = "cecd-dual-semantics-controlled-runner-v1"
RUNTIME_DESCRIPTOR_SCHEMA = "cecd-dual-semantics-runtime-descriptor-v1"
ARM_SHARD_SCHEMA = "cecd-dual-semantics-arm-shard-v1"
RUN_MANIFEST_SCHEMA = "cecd-dual-semantics-run-manifest-v1"
CE_STAGE_SHARD_SCHEMA = "cecd-dual-semantics-formal-ce-shard-v1"
CE_STAGE_MANIFEST_SCHEMA = "cecd-dual-semantics-formal-ce-stage-manifest-v1"
CE_RAW_CACHE_SCHEMA = "cecd-dual-semantics-formal-ce-raw-cache-v1"
CE_CELL_ORDER = ("h00", "h10", "h01", "h11")
ROOT = Path("/home/dbw/ANCHOR")
DEFAULT_AUTHORIZATION = (
    ROOT / "corrected_runs/vindr_v2/cecd_dual_semantics_v1/authorization.json"
)
DEFAULT_PREFLIGHT = ROOT / "configs/cecd_dual_semantics_preflight_v1.json"
GPU_LOCK_RELATIVE = Path("corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock")
DEFAULT_GPU_LOCK = ROOT / GPU_LOCK_RELATIVE
DEFAULT_HUATUO_PYTHON = Path("/opt/miniconda3/envs/huatuo/bin/python")
DEFAULT_HULU_PYTHON = Path("/home/dbw/.venvs/hulumed/bin/python")
HEX64 = set("0123456789abcdef")
TASKS = ("ce", "oe")


class ControlledRunError(RuntimeError):
    """Raised when the narrow controlled-comparison contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ControlledRunError(f"required regular file is missing or symlinked: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _validate_file_record(record: Any, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
        raise ControlledRunError(f"{label} file record has an invalid shape")
    path = Path(str(record["path"])).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or not _is_hex64(record["sha256"])
        or isinstance(record["bytes"], bool)
        or not isinstance(record["bytes"], int)
        or record["bytes"] < 0
        or path.stat().st_size != record["bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise ControlledRunError(f"{label} file binding drift: {path}")
    return path


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlledRunError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ControlledRunError(f"{label} must be a JSON object")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            raise ControlledRunError(f"write-once artifact collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _authorization_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "fingerprint"}


def validate_authorization_and_preflight(
    *, authorization_path: Path, preflight_path: Path, root: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Revalidate authority without parsing Stage-1 or any sealed outcome."""

    if not authorization_path.is_file():
        raise ControlledRunError(
            "controlled comparison is not authorized; refusing before worker/runtime/GPU access"
        )
    authorization = load_object(authorization_path, "write-once authorization")
    preflight = load_object(preflight_path, "outcome-blind preflight")
    validate_dual_semantics_preflight_contract(preflight)
    fingerprint = authorization.get("fingerprint")
    if not _is_hex64(fingerprint) or canonical_sha256(
        _authorization_body(authorization)
    ) != fingerprint:
        raise ControlledRunError("authorization fingerprint mismatch")
    required_false = (
        "general_hidden_state_stage_authorized",
        "paper_native_treble_authorized",
        "exact_treble_authorized",
        "general_gpu_authorized",
        "paper_claim_authorized",
        "method_outputs_consumed",
    )
    if (
        authorization.get("version") != AUTHORIZATION_VERSION
        or authorization.get("status")
        != "controlled_dual_semantics_comparison_authorized"
        or authorization.get("controlled_method_comparison_authorized") is not True
        or authorization.get(
            "cecd_hidden_state_intervention_authorized_only_inside_locked_comparison"
        )
        is not True
        or any(authorization.get(field) is not False for field in required_false)
        or authorization.get("required_outcome_schema") != DUAL_SEMANTICS_OUTCOME_SCHEMA
        or authorization.get("locked_test_behavioral_increment_confirmed") is not True
        or authorization.get("full_method_gate_authorized") is not False
        or authorization.get("oral_baseline_closure_authorized") is not False
        or authorization.get("official_compatible_dynamic_activation_baseline_present")
        is not False
    ):
        raise ControlledRunError("authorization is not the narrow controlled-comparison grant")

    bound_preflight = authorization.get("preflight")
    if not isinstance(bound_preflight, Mapping):
        raise ControlledRunError("authorization does not bind a preflight")
    _validate_file_record(bound_preflight, "authorization.preflight")
    if (
        bound_preflight.get("path") != str(preflight_path.resolve())
        or bound_preflight.get("sha256") != sha256_file(preflight_path)
    ):
        raise ControlledRunError("authorization binds a different preflight")
    for label in (
        "stage1_analysis", "stage1_input_gate", "admission", "preflight_build"
    ):
        _validate_file_record(authorization.get(label), f"authorization.{label}")

    methods = preflight.get("methods")
    allowed = authorization.get("allowed_methods")
    if (
        not isinstance(methods, list)
        or not methods
        or any(not isinstance(item, str) or not item for item in methods)
        or len(methods) != len(set(methods))
        or allowed != methods
    ):
        raise ControlledRunError(
            "authorization/preflight method order and closure are not exactly equal"
        )
    models = preflight.get("model_fingerprints")
    if not isinstance(models, Mapping) or set(models) != {"huatuo", "hulu"}:
        raise ControlledRunError("preflight must bind exactly Huatuo and Hulu")

    output_root = Path(str(preflight["method_output_root"])).resolve()
    repository = root.resolve()
    if output_root == repository or repository not in output_root.parents:
        raise ControlledRunError("method output root must be a narrow repository child")
    if authorization.get("method_output_root") != str(output_root):
        raise ControlledRunError("authorization/preflight output-root mismatch")
    return authorization, preflight, output_root


def _runtime_descriptor_command(
    *, python_path: Path, worker: Path, family: str, preflight_path: Path
) -> list[str]:
    return [
        str(python_path.resolve()),
        "-u",
        str(worker.resolve()),
        "--describe-runtime",
        "--model-family",
        family,
        "--preflight",
        str(preflight_path.resolve()),
    ]


def _run_json_command(command: Sequence[str], env: Mapping[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr[-4000:].strip()
        raise ControlledRunError(
            f"runtime descriptor exited {completed.returncode}: {stderr}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ControlledRunError("runtime descriptor did not emit one JSON object") from error
    if not isinstance(payload, dict):
        raise ControlledRunError("runtime descriptor must be a JSON object")
    return payload


def _validate_runtime_descriptor(
    *,
    descriptor: Mapping[str, Any],
    family: str,
    python_path: Path,
    worker: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "model_family",
        "model_id",
        "model_fingerprint",
        "python_executable",
        "runtime_versions",
        "source_files",
        "input_bindings",
    }
    if set(descriptor) != required:
        raise ControlledRunError(f"{family} runtime descriptor fields are not closed")
    expected_model = preflight["model_fingerprints"][family]
    if (
        descriptor["schema_version"] != RUNTIME_DESCRIPTOR_SCHEMA
        or descriptor["model_family"] != family
        or descriptor["model_id"] != expected_model["model_id"]
        or descriptor["model_fingerprint"] != expected_model
        or Path(str(descriptor["python_executable"])).resolve() != python_path.resolve()
    ):
        raise ControlledRunError(f"{family} runtime/model identity mismatch")
    versions = descriptor["runtime_versions"]
    if (
        not isinstance(versions, Mapping)
        or not versions
        or any(not isinstance(key, str) or not str(value) for key, value in versions.items())
    ):
        raise ControlledRunError(f"{family} runtime versions are incomplete")
    sources = descriptor["source_files"]
    if not isinstance(sources, list) or not sources:
        raise ControlledRunError(f"{family} runtime source closure is empty")
    source_paths = []
    for index, record in enumerate(sources):
        path = _validate_file_record(record, f"{family}.source_files[{index}]")
        source_paths.append(path)
    if len(set(source_paths)) != len(source_paths) or worker.resolve() not in set(source_paths):
        raise ControlledRunError(f"{family} source closure must include the worker exactly once")

    bindings = descriptor["input_bindings"]
    expected_hashes = {
        "calibration_manifest": preflight["calibration_manifest_sha256"],
        "evaluation_manifest": preflight["evaluation_manifest_sha256"],
        "record_keys": preflight["record_keys_sha256"],
        "claim_contract": preflight["claim_contract_sha256"],
    }
    if not isinstance(bindings, Mapping) or set(bindings) != set(expected_hashes):
        raise ControlledRunError(f"{family} input binding closure is incomplete")
    for label, expected_hash in expected_hashes.items():
        _validate_file_record(bindings[label], f"{family}.input_bindings.{label}")
        if bindings[label]["sha256"] != expected_hash:
            raise ControlledRunError(f"{family} {label} does not match frozen preflight")
    return json.loads(json.dumps(descriptor, sort_keys=True))


def controlled_environment(root: Path) -> dict[str, str]:
    keep = ("PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "TZ", "HF_HOME")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(
                (str(root.resolve()), str((root / "anchor").resolve()))
            ),
        }
    )
    return env


def build_candidate_contract(
    *,
    authorization_path: Path,
    preflight_path: Path,
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    output_root: Path,
    worker: Path,
    python_by_family: Mapping[str, Path],
    gpu_lock: Path,
    root: Path,
) -> dict[str, Any]:
    repository = root.resolve()
    worker = worker.resolve()
    if not worker.is_file() or repository not in worker.parents:
        raise ControlledRunError("formal worker must be a regular file inside the repository")
    if set(python_by_family) != {"huatuo", "hulu"}:
        raise ControlledRunError("runtime map must contain exactly Huatuo and Hulu")
    env = controlled_environment(root)
    descriptors: dict[str, Any] = {}
    runtime_executables: dict[str, Any] = {}
    for family in ("huatuo", "hulu"):
        python_path = python_by_family[family].resolve()
        runtime_executables[family] = file_record(python_path)
        raw = _run_json_command(
            _runtime_descriptor_command(
                python_path=python_path,
                worker=worker,
                family=family,
                preflight_path=preflight_path,
            ),
            env,
        )
        descriptors[family] = _validate_runtime_descriptor(
            descriptor=raw,
            family=family,
            python_path=python_path,
            worker=worker,
            preflight=preflight,
        )
    source_root = Path(__file__).resolve().parent
    source_paths = {
        "runner": Path(__file__).resolve(),
        "authorization_binder": source_root
        / "authorize_cecd_dual_semantics_preflight_v1.py",
        "collision_contract": source_root / "treble_collision_contract.py",
        "worker": worker,
    }
    contract: dict[str, Any] = {
        "version": VERSION,
        "status": "frozen_before_arm_outputs",
        "authorization": file_record(authorization_path),
        "authorization_fingerprint": authorization["fingerprint"],
        "preflight": file_record(preflight_path),
        "method_output_root": str(output_root),
        "models": json.loads(json.dumps(preflight["model_fingerprints"], sort_keys=True)),
        # The closure is read from the frozen preflight.  It is intentionally
        # not copied from the current ten-method Python constant so a future,
        # pre-output contract revision can add a required dynamic baseline.
        "methods": list(preflight["methods"]),
        "tasks": list(TASKS),
        "arm_order": [
            {"model_family": family, "method": method}
            for family in ("huatuo", "hulu")
            for method in preflight["methods"]
        ],
        "source_files": {name: file_record(path) for name, path in source_paths.items()},
        "runtime_executables": runtime_executables,
        "runtime_descriptors": descriptors,
        "controlled_environment": env,
        "gpu_lock": str(gpu_lock.resolve()),
        "narrow_gpu_scope": "only this hash-bound controlled comparison",
        "general_gpu_authorized": False,
        "paper_claim_authorized": False,
        "results_interpreted": False,
        "host_prepare_identity": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    contract["fingerprint"] = canonical_sha256(contract)
    return contract


def _validate_stored_contract(
    *, stored: Mapping[str, Any], candidate: Mapping[str, Any]
) -> str:
    fingerprint = stored.get("fingerprint")
    if not _is_hex64(fingerprint):
        raise ControlledRunError("stored run contract fingerprint is malformed")
    body = {key: value for key, value in stored.items() if key != "fingerprint"}
    if canonical_sha256(body) != fingerprint:
        raise ControlledRunError("stored run contract fingerprint mismatch")
    if stored != candidate:
        changed = sorted(
            key for key in set(stored) | set(candidate) if stored.get(key) != candidate.get(key)
        )
        raise ControlledRunError(f"refusing execution/resume after run-contract drift: {changed}")
    for name, record in stored["source_files"].items():
        _validate_file_record(record, f"run_contract.source_files.{name}")
    for family, record in stored["runtime_executables"].items():
        _validate_file_record(record, f"run_contract.runtime_executables.{family}")
    return fingerprint


@contextmanager
def exclusive_lock(path: Path, label: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ControlledRunError(f"{label} lock is already held: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _safe_relative_output(root: Path, text: Any, label: str) -> Path:
    if not isinstance(text, str) or not text or Path(text).is_absolute():
        raise ControlledRunError(f"{label} must be a nonempty relative path")
    path = (root / text).resolve()
    if root.resolve() not in path.parents or path.is_symlink():
        raise ControlledRunError(f"{label} escapes its arm directory")
    return path


def validate_arm_completion(
    *,
    arm_dir: Path,
    run_fingerprint: str,
    worker_sha256: str,
    runtime_descriptor_sha256: str,
    family: str,
    model_id: str,
    method: str,
) -> dict[str, Any]:
    completion_path = arm_dir / "completion.json"
    payload = load_object(completion_path, f"{family}/{method} completion")
    required = {
        "schema_version",
        "status",
        "run_fingerprint",
        "model_family",
        "model_id",
        "method",
        "tasks",
        "task_outputs",
        "observed_compute_ledger",
        "worker_sha256",
        "runtime_descriptor_sha256",
        "completion_fingerprint",
    }
    if set(payload) != required:
        raise ControlledRunError(f"{family}/{method} completion fields are not closed")
    declared = payload["completion_fingerprint"]
    body = {key: value for key, value in payload.items() if key != "completion_fingerprint"}
    if not _is_hex64(declared) or canonical_sha256(body) != declared:
        raise ControlledRunError(f"{family}/{method} completion fingerprint mismatch")
    if (
        payload["schema_version"] != ARM_SHARD_SCHEMA
        or payload["status"] != "complete"
        or payload["run_fingerprint"] != run_fingerprint
        or payload["model_family"] != family
        or payload["model_id"] != model_id
        or payload["method"] != method
        or payload["tasks"] != list(TASKS)
        or payload["worker_sha256"] != worker_sha256
        or payload["runtime_descriptor_sha256"] != runtime_descriptor_sha256
    ):
        raise ControlledRunError(f"{family}/{method} identity or protocol mismatch")
    outputs = payload["task_outputs"]
    if not isinstance(outputs, Mapping) or set(outputs) != set(TASKS):
        raise ControlledRunError(f"{family}/{method} CE/OE output closure is incomplete")
    for task in TASKS:
        record = outputs[task]
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "bytes",
            "rows",
            "clusters",
        }:
            raise ControlledRunError(f"{family}/{method}/{task} output record is malformed")
        output_path = _safe_relative_output(arm_dir, record["path"], f"{task}.path")
        expected_file = {key: record[key] for key in ("sha256", "bytes")}
        expected_file["path"] = str(output_path)
        _validate_file_record(expected_file, f"{family}/{method}/{task}")
        for field in ("rows", "clusters"):
            number = record[field]
            if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                raise ControlledRunError(f"{family}/{method}/{task}.{field} must be positive")
        if record["clusters"] < 30:
            raise ControlledRunError(f"{family}/{method}/{task} has fewer than 30 clusters")
    ledger = payload["observed_compute_ledger"]
    if (
        not isinstance(ledger, Mapping)
        or not ledger
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
            for value in ledger.values()
        )
    ):
        raise ControlledRunError(f"{family}/{method} compute ledger is missing or invalid")
    return payload


def _arm_command(
    *,
    python_path: Path,
    worker: Path,
    authorization_path: Path,
    preflight_path: Path,
    run_contract_path: Path,
    family: str,
    method: str,
    output_dir: Path,
) -> list[str]:
    return [
        str(python_path.resolve()),
        "-u",
        str(worker.resolve()),
        "--authorization",
        str(authorization_path.resolve()),
        "--preflight",
        str(preflight_path.resolve()),
        "--run-contract",
        str(run_contract_path.resolve()),
        "--model-family",
        family,
        "--method",
        method,
        "--output-dir",
        str(output_dir.resolve()),
    ]


def _ce_arm_command(
    *,
    python_path: Path,
    worker: Path,
    authorization_path: Path,
    preflight_path: Path,
    run_contract_path: Path,
    family: str,
    method: str,
    shared_cache_root: Path,
    output_dir: Path,
) -> list[str]:
    return [
        *_arm_command(
            python_path=python_path,
            worker=worker,
            authorization_path=authorization_path,
            preflight_path=preflight_path,
            run_contract_path=run_contract_path,
            family=family,
            method=method,
            output_dir=output_dir,
        ),
        "--task",
        "ce",
        "--shared-cache-root",
        str(shared_cache_root.resolve()),
    ]


def validate_ce_stage_completion(
    *,
    arm_dir: Path,
    run_fingerprint: str,
    worker_sha256: str,
    runtime_descriptor_sha256: str,
    family: str,
    model_id: str,
    method: str,
    expected_raw_cache_manifest: Path,
) -> dict[str, Any]:
    payload = load_object(arm_dir / "ce_completion.json", f"{family}/{method} CE completion")
    required = {
        "schema_version",
        "status",
        "run_fingerprint",
        "model_family",
        "model_id",
        "method",
        "task",
        "raw_cache_manifest",
        "ce_output",
        "rows",
        "clusters",
        "worker_sha256",
        "runtime_descriptor_sha256",
        "oe_implemented",
        "hidden_intervention_implemented",
        "paper_native_treble_claimed",
        "completion_fingerprint",
    }
    if set(payload) != required:
        raise ControlledRunError(f"{family}/{method} CE completion fields are not closed")
    declared = payload["completion_fingerprint"]
    body = {key: value for key, value in payload.items() if key != "completion_fingerprint"}
    if not _is_hex64(declared) or canonical_sha256(body) != declared:
        raise ControlledRunError(f"{family}/{method} CE completion fingerprint mismatch")
    if (
        payload["schema_version"] != CE_STAGE_SHARD_SCHEMA
        or payload["status"] != "formal_ce_complete_oe_blocked"
        or payload["run_fingerprint"] != run_fingerprint
        or payload["model_family"] != family
        or payload["model_id"] != model_id
        or payload["method"] != method
        or payload["task"] != "ce"
        or payload["worker_sha256"] != worker_sha256
        or payload["runtime_descriptor_sha256"] != runtime_descriptor_sha256
        or payload["oe_implemented"] is not False
        or payload["hidden_intervention_implemented"] is not False
        or payload["paper_native_treble_claimed"] is not False
    ):
        raise ControlledRunError(f"{family}/{method} CE identity/scope mismatch")
    raw_cache_path = _validate_file_record(
        payload["raw_cache_manifest"], f"{family}/{method}.raw_cache_manifest"
    )
    if raw_cache_path != expected_raw_cache_manifest.resolve():
        raise ControlledRunError(
            f"{family}/{method} raw cache escapes its frozen per-model cache"
        )
    ce_record = payload["ce_output"]
    if not isinstance(ce_record, Mapping) or set(ce_record) != {"path", "sha256", "bytes"}:
        raise ControlledRunError(f"{family}/{method}.ce_output record is malformed")
    ce_path = _safe_relative_output(arm_dir, ce_record["path"], "ce_output.path")
    _validate_file_record(
        {**ce_record, "path": str(ce_path)}, f"{family}/{method}.ce_output"
    )
    for label in ("rows", "clusters"):
        value = payload[label]
        if isinstance(value, bool) or not isinstance(value, int) or value < 30:
            raise ControlledRunError(f"{family}/{method} CE {label} must be >=30")
    return payload


def validate_shared_raw_cache_manifest(
    *, path: Path, family: str, run_fingerprint: str
) -> dict[str, Any]:
    payload = load_object(path, f"{family} formal CE raw cache manifest")
    required = {
        "schema_version",
        "status",
        "config_fingerprint",
        "run_fingerprint",
        "model_family",
        "records",
        "clusters",
        "cells",
        "cell_files",
        "shared_across_methods",
        "fingerprint",
    }
    if set(payload) != required:
        raise ControlledRunError(f"{family} raw-cache manifest fields are not closed")
    declared = payload["fingerprint"]
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if not _is_hex64(declared) or canonical_sha256(body) != declared:
        raise ControlledRunError(f"{family} raw-cache manifest fingerprint mismatch")
    records, clusters, cells = (
        payload["records"],
        payload["clusters"],
        payload["cells"],
    )
    if (
        payload["schema_version"] != CE_RAW_CACHE_SCHEMA
        or payload["status"] != "complete"
        or payload["run_fingerprint"] != run_fingerprint
        or payload["model_family"] != family
        or not _is_hex64(payload["config_fingerprint"])
        or isinstance(records, bool)
        or not isinstance(records, int)
        or records < 30
        or isinstance(clusters, bool)
        or not isinstance(clusters, int)
        or clusters < 30
        or isinstance(cells, bool)
        or not isinstance(cells, int)
        or cells != records * len(CE_CELL_ORDER)
        or payload["shared_across_methods"] is not True
    ):
        raise ControlledRunError(f"{family} raw-cache identity/scope mismatch")
    entries = payload["cell_files"]
    if not isinstance(entries, list) or len(entries) != cells:
        raise ControlledRunError(f"{family} raw-cache cell closure is incomplete")
    observed: dict[str, set[str]] = {}
    cache_root = path.resolve().parent
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or set(entry) != {"record_key", "cell", "file"}:
            raise ControlledRunError(f"{family} raw-cache cell {index} is malformed")
        record_key, cell = str(entry["record_key"]), str(entry["cell"])
        if not record_key or cell not in CE_CELL_ORDER:
            raise ControlledRunError(f"{family} raw-cache cell {index} identity is invalid")
        cell_path = _validate_file_record(
            entry["file"], f"{family}.raw_cache.cell_files[{index}]"
        )
        expected = cache_root / "records" / record_key / "cells" / f"{cell}.json"
        if cell_path != expected.resolve():
            raise ControlledRunError(f"{family} raw-cache cell {index} escapes its record")
        observed.setdefault(record_key, set()).add(cell)
    if len(observed) != records or any(cells != set(CE_CELL_ORDER) for cells in observed.values()):
        raise ControlledRunError(f"{family} raw-cache does not close every four-cell orbit")
    return payload


def _failure_record(
    *, output_root: Path, family: str, method: str, command: Sequence[str], error: BaseException
) -> Path:
    failures = output_root / "failures"
    failures.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (failures / f"attempt_{attempt:04d}.json").exists():
        attempt += 1
    path = failures / f"attempt_{attempt:04d}.json"
    payload = {
        "version": VERSION,
        "status": "failed_stop",
        "time": utc_now(),
        "model_family": family,
        "method": method,
        "command": list(command),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "automatic_retry_authorized": False,
        "partial_outputs_promoted": False,
    }
    write_once_json(path, payload)
    atomic_json(
        output_root / "state.json",
        {
            "version": VERSION,
            "status": "failed_stop",
            "last_failure": str(path.resolve()),
            "automatic_retry_authorized": False,
        },
    )
    return path


def _existing_failed_state(output_root: Path) -> dict[str, Any] | None:
    state_path = output_root / "state.json"
    if not state_path.is_file():
        return None
    state = load_object(state_path, "runner state")
    return state if state.get("status") == "failed_stop" else None


def execute_arms(
    *,
    contract: Mapping[str, Any],
    authorization_path: Path,
    preflight_path: Path,
    run_contract_path: Path,
    output_root: Path,
    worker: Path,
    python_by_family: Mapping[str, Path],
    gpu_lock: Path,
    resume_after_failure: bool,
) -> dict[str, Any]:
    failed_state = _existing_failed_state(output_root)
    if failed_state is not None and not resume_after_failure:
        raise ControlledRunError(
            "previous arm failed; explicit --resume-after-failure is required after log audit"
        )
    run_fingerprint = str(contract["fingerprint"])
    worker_sha256 = str(contract["source_files"]["worker"]["sha256"])
    completed_arms: list[dict[str, Any]] = []
    output_lock = output_root / ".controlled-run.lock"
    with exclusive_lock(output_lock, "controlled-run"), exclusive_lock(gpu_lock, "GPU"):
        atomic_json(
            output_root / "state.json",
            {
                "version": VERSION,
                "status": "running",
                "pid": os.getpid(),
                "run_fingerprint": run_fingerprint,
                "narrow_gpu_scope": "only this hash-bound controlled comparison",
                "general_gpu_authorized": False,
            },
        )
        env = dict(contract["controlled_environment"])
        for arm in contract["arm_order"]:
            family, method = arm["model_family"], arm["method"]
            model_id = str(contract["models"][family]["model_id"])
            descriptor_sha = canonical_sha256(contract["runtime_descriptors"][family])
            arm_dir = output_root / "arms" / family / method
            if arm_dir.is_dir():
                completion = validate_arm_completion(
                    arm_dir=arm_dir,
                    run_fingerprint=run_fingerprint,
                    worker_sha256=worker_sha256,
                    runtime_descriptor_sha256=descriptor_sha,
                    family=family,
                    model_id=model_id,
                    method=method,
                )
                completed_arms.append(
                    {
                        "model_family": family,
                        "method": method,
                        "completion": file_record(arm_dir / "completion.json"),
                        "completion_fingerprint": completion["completion_fingerprint"],
                    }
                )
                continue
            temporary = (
                output_root
                / "partial"
                / family
                / method
                / f"attempt_{int(time.time_ns())}_{os.getpid()}"
            )
            temporary.mkdir(parents=True, exist_ok=False)
            command = _arm_command(
                python_path=python_by_family[family],
                worker=worker,
                authorization_path=authorization_path,
                preflight_path=preflight_path,
                run_contract_path=run_contract_path,
                family=family,
                method=method,
                output_dir=temporary,
            )
            try:
                completed = subprocess.run(
                    command,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    check=False,
                )
                if completed.returncode != 0:
                    raise ControlledRunError(
                        f"worker exited {completed.returncode} for {family}/{method}"
                    )
                completion = validate_arm_completion(
                    arm_dir=temporary,
                    run_fingerprint=run_fingerprint,
                    worker_sha256=worker_sha256,
                    runtime_descriptor_sha256=descriptor_sha,
                    family=family,
                    model_id=model_id,
                    method=method,
                )
                arm_dir.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(arm_dir)
            except BaseException as error:
                failure = _failure_record(
                    output_root=output_root,
                    family=family,
                    method=method,
                    command=command,
                    error=error,
                )
                raise ControlledRunError(
                    f"controlled comparison failed-stop at {family}/{method}; {failure}"
                ) from error
            completed_arms.append(
                {
                    "model_family": family,
                    "method": method,
                    "completion": file_record(arm_dir / "completion.json"),
                    "completion_fingerprint": completion["completion_fingerprint"],
                }
            )
        expected = [
            (arm["model_family"], arm["method"]) for arm in contract["arm_order"]
        ]
        observed = [(arm["model_family"], arm["method"]) for arm in completed_arms]
        if observed != expected:
            raise ControlledRunError("completed model x method closure/order is incomplete")
        manifest: dict[str, Any] = {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "status": "complete",
            "run_fingerprint": run_fingerprint,
            "models": ["huatuo", "hulu"],
            "methods": list(contract["methods"]),
            "tasks": list(TASKS),
            "arms": completed_arms,
            "arm_count": len(completed_arms),
            "strict_model_method_closure": True,
            "results_interpreted": False,
            "collision_verdict_computed": False,
            "paper_claim_authorized": False,
        }
        manifest["fingerprint"] = canonical_sha256(manifest)
        write_once_json(output_root / "run_manifest.json", manifest)
        atomic_json(
            output_root / "state.json",
            {
                "version": VERSION,
                "status": "complete_pending_blinded_analysis",
                "run_fingerprint": run_fingerprint,
                "manifest": file_record(output_root / "run_manifest.json"),
                "results_interpreted": False,
                "paper_claim_authorized": False,
            },
        )
        return manifest


def execute_formal_ce_stage(
    *,
    contract: Mapping[str, Any],
    authorization_path: Path,
    preflight_path: Path,
    run_contract_path: Path,
    output_root: Path,
    worker: Path,
    python_by_family: Mapping[str, Path],
    gpu_lock: Path,
    resume_after_failure: bool,
) -> dict[str, Any]:
    """Run only the seven logit-space CE controls from one cache per model.

    This stage is deliberately not a complete controlled comparison.  It never
    invokes CECD hidden intervention, either Treble variant or OE, and it never
    creates ``run_manifest.json``.
    """

    failed_state = _existing_failed_state(output_root)
    if failed_state is not None and not resume_after_failure:
        raise ControlledRunError(
            "previous arm failed; explicit --resume-after-failure is required after log audit"
        )
    available = [
        method for method in contract["methods"] if method in IMPLEMENTED_KERNEL_METHODS
    ]
    if set(available) != set(IMPLEMENTED_KERNEL_METHODS) or len(available) != len(
        IMPLEMENTED_KERNEL_METHODS
    ):
        raise ControlledRunError("formal CE stage requires all seven frozen logit controls")
    blocked = [method for method in contract["methods"] if method not in available]
    required_blocked = {
        "cecd_interaction_projection",
        "treble_proceedings",
        "treble_released",
    }
    if not required_blocked <= set(blocked):
        raise ControlledRunError("CE stage must retain CECD and both Treble variants as blocked")

    run_fingerprint = str(contract["fingerprint"])
    worker_sha256 = str(contract["source_files"]["worker"]["sha256"])
    completed_arms: list[dict[str, Any]] = []
    output_lock = output_root / ".controlled-run.lock"
    with exclusive_lock(output_lock, "controlled-run"), exclusive_lock(gpu_lock, "GPU"):
        atomic_json(
            output_root / "state.json",
            {
                "version": VERSION,
                "status": "formal_ce_stage_running_oe_blocked",
                "pid": os.getpid(),
                "run_fingerprint": run_fingerprint,
                "general_gpu_authorized": False,
                "paper_claim_authorized": False,
            },
        )
        env = dict(contract["controlled_environment"])
        for family in ("huatuo", "hulu"):
            shared_cache = output_root / "shared_ce_cache" / family
            model_id = str(contract["models"][family]["model_id"])
            descriptor_sha = canonical_sha256(contract["runtime_descriptors"][family])
            for method in available:
                arm_dir = output_root / "ce_arms" / family / method
                if arm_dir.is_dir():
                    completion = validate_ce_stage_completion(
                        arm_dir=arm_dir,
                        run_fingerprint=run_fingerprint,
                        worker_sha256=worker_sha256,
                        runtime_descriptor_sha256=descriptor_sha,
                        family=family,
                        model_id=model_id,
                        method=method,
                        expected_raw_cache_manifest=shared_cache
                        / "raw_cache_manifest.json",
                    )
                else:
                    temporary = (
                        output_root
                        / "partial_ce"
                        / family
                        / method
                        / f"attempt_{int(time.time_ns())}_{os.getpid()}"
                    )
                    temporary.mkdir(parents=True, exist_ok=False)
                    command = _ce_arm_command(
                        python_path=python_by_family[family],
                        worker=worker,
                        authorization_path=authorization_path,
                        preflight_path=preflight_path,
                        run_contract_path=run_contract_path,
                        family=family,
                        method=method,
                        shared_cache_root=shared_cache,
                        output_dir=temporary,
                    )
                    try:
                        process = subprocess.run(
                            command,
                            env=env,
                            stdin=subprocess.DEVNULL,
                            check=False,
                        )
                        if process.returncode != 0:
                            raise ControlledRunError(
                                f"CE worker exited {process.returncode} for {family}/{method}"
                            )
                        completion = validate_ce_stage_completion(
                            arm_dir=temporary,
                            run_fingerprint=run_fingerprint,
                            worker_sha256=worker_sha256,
                            runtime_descriptor_sha256=descriptor_sha,
                            family=family,
                            model_id=model_id,
                            method=method,
                            expected_raw_cache_manifest=shared_cache
                            / "raw_cache_manifest.json",
                        )
                        arm_dir.parent.mkdir(parents=True, exist_ok=True)
                        temporary.replace(arm_dir)
                    except BaseException as error:
                        failure = _failure_record(
                            output_root=output_root,
                            family=family,
                            method=f"ce::{method}",
                            command=command,
                            error=error,
                        )
                        raise ControlledRunError(
                            f"formal CE stage failed-stop at {family}/{method}; {failure}"
                        ) from error
                completed_arms.append(
                    {
                        "model_family": family,
                        "method": method,
                        "completion": file_record(arm_dir / "ce_completion.json"),
                        "completion_fingerprint": completion["completion_fingerprint"],
                        "raw_cache_manifest_sha256": completion["raw_cache_manifest"][
                            "sha256"
                        ],
                    }
                )
            validate_shared_raw_cache_manifest(
                path=shared_cache / "raw_cache_manifest.json",
                family=family,
                run_fingerprint=run_fingerprint,
            )
        expected = [
            (family, method)
            for family in ("huatuo", "hulu")
            for method in available
        ]
        observed = [(row["model_family"], row["method"]) for row in completed_arms]
        if observed != expected:
            raise ControlledRunError("formal CE model x seven-control closure is incomplete")
        for family in ("huatuo", "hulu"):
            hashes = {
                row["raw_cache_manifest_sha256"]
                for row in completed_arms
                if row["model_family"] == family
            }
            if len(hashes) != 1:
                raise ControlledRunError(f"{family} CE arms did not share one raw-logit cache")
        manifest: dict[str, Any] = {
            "schema_version": CE_STAGE_MANIFEST_SCHEMA,
            "status": "formal_ce_complete_oe_and_hidden_methods_blocked",
            "run_fingerprint": run_fingerprint,
            "models": ["huatuo", "hulu"],
            "methods": available,
            "blocked_methods": blocked,
            "task": "ce",
            "operation_space": "centered tri-state next-token logits",
            "arms": completed_arms,
            "arm_count": len(completed_arms),
            "shared_cache_per_model": True,
            "oe_implemented": False,
            "hidden_intervention_implemented": False,
            "paper_native_treble_claimed": False,
            "full_run_manifest_written": False,
            "results_interpreted": False,
            "paper_claim_authorized": False,
        }
        manifest["fingerprint"] = canonical_sha256(manifest)
        write_once_json(output_root / "ce_stage_manifest.json", manifest)
        atomic_json(
            output_root / "state.json",
            {
                "version": VERSION,
                "status": "formal_ce_complete_oe_blocked",
                "run_fingerprint": run_fingerprint,
                "ce_stage_manifest": file_record(output_root / "ce_stage_manifest.json"),
                "full_run_manifest_written": False,
                "paper_claim_authorized": False,
            },
        )
        return manifest


def prepare_or_run(
    *,
    authorization_path: Path,
    preflight_path: Path,
    worker: Path,
    huatuo_python: Path,
    hulu_python: Path,
    gpu_lock: Path,
    root: Path = ROOT,
    execute: bool = False,
    execute_ce_only: bool = False,
    resume_after_failure: bool = False,
) -> dict[str, Any]:
    if execute and execute_ce_only:
        raise ControlledRunError("full execution and CE-only execution are mutually exclusive")
    canonical_gpu_lock = (root / GPU_LOCK_RELATIVE).resolve()
    if gpu_lock.resolve() != canonical_gpu_lock:
        raise ControlledRunError(
            f"GPU lock drift: all VinDr real runners must use {canonical_gpu_lock}"
        )
    authorization, preflight, output_root = validate_authorization_and_preflight(
        authorization_path=authorization_path,
        preflight_path=preflight_path,
        root=root,
    )
    # A failed run is terminal before any further worker/runtime descriptor is
    # executed.  Explicit audited recovery is the sole exception.
    if (execute or execute_ce_only) and _existing_failed_state(
        output_root
    ) is not None and not resume_after_failure:
        raise ControlledRunError(
            "previous arm failed; explicit --resume-after-failure is required after log audit"
        )
    python_by_family = {"huatuo": huatuo_python, "hulu": hulu_python}
    candidate = build_candidate_contract(
        authorization_path=authorization_path,
        preflight_path=preflight_path,
        authorization=authorization,
        preflight=preflight,
        output_root=output_root,
        worker=worker,
        python_by_family=python_by_family,
        gpu_lock=gpu_lock,
        root=root,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = output_root / "run_contract.json"
    if contract_path.exists():
        stored = load_object(contract_path, "write-once run contract")
        _validate_stored_contract(stored=stored, candidate=candidate)
        contract = stored
    else:
        write_once_json(contract_path, candidate)
        contract = candidate
    if not execute and not execute_ce_only:
        return {
            "version": VERSION,
            "status": "prepared_outcome_blind_no_gpu_launched",
            "run_contract": file_record(contract_path),
            "run_fingerprint": contract["fingerprint"],
            "models": ["huatuo", "hulu"],
            "methods": list(contract["methods"]),
            "arm_count": len(contract["arm_order"]),
            "general_gpu_authorized": False,
            "paper_claim_authorized": False,
        }
    executor = execute_formal_ce_stage if execute_ce_only else execute_arms
    return executor(
        contract=contract,
        authorization_path=authorization_path,
        preflight_path=preflight_path,
        run_contract_path=contract_path,
        output_root=output_root,
        worker=worker,
        python_by_family=python_by_family,
        gpu_lock=gpu_lock,
        resume_after_failure=resume_after_failure,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--huatuo-python", type=Path, default=DEFAULT_HUATUO_PYTHON)
    parser.add_argument("--hulu-python", type=Path, default=DEFAULT_HULU_PYTHON)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch hash-bound arms; omission only freezes the outcome-blind run contract",
    )
    parser.add_argument(
        "--execute-ce-only",
        action="store_true",
        help=(
            "run only the seven formal centered-logit CE controls from one shared "
            "cache per model; OE/CECD/Treble remain blocked"
        ),
    )
    parser.add_argument(
        "--resume-after-failure",
        action="store_true",
        help="continue valid shards only after the prior failure log has been audited",
    )
    args = parser.parse_args()
    result = prepare_or_run(
        authorization_path=args.authorization,
        preflight_path=args.preflight,
        worker=args.worker,
        huatuo_python=args.huatuo_python,
        hulu_python=args.hulu_python,
        gpu_lock=args.gpu_lock,
        execute=args.execute,
        execute_ce_only=args.execute_ce_only,
        resume_after_failure=args.resume_after_failure,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
