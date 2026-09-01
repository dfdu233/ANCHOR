#!/usr/bin/env python3
"""Persistently advance Specificity Ratchet without synthesizing clinical truth.

The monitor accepts only separately signed human returns whose bytes are stable
across two polls.  It then performs the frozen merge, prepares a provenance-
blinded adjudicator archive, validates a returned adjudication in a disposable
pack, compiles the replay manifest, runs CPU preflight, and advances the frozen
canary -> full capture -> replay -> analysis chain.  It never fills or repairs
a clinical field, combines signatures only after exact validation, or retries
a failed scientific job.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anchor.corrected_sgta.merge_specificity_ratchet_reviews_v1 import (
    _write_csv_once,
    merge_reviews,
)
from anchor.corrected_sgta.validate_specificity_ratchet_adjudication_v1 import (
    PROTOCOL_ID,
    validate_adjudication,
)
from anchor.medeval.hashing import sha256_file
from anchor.medeval.package_specificity_ratchet_adjudication import (
    package_adjudication,
)
from anchor.medeval.store import atomic_write_json


VERSION = "specificity-ratchet-clinical-pipeline-monitor-v1"
ROOT = Path("/home/dbw/ANCHOR")
DEFAULT_PACK = ROOT / "corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2"
DEFAULT_INBOX = (
    Path("/home/dbw/datasets/public/vqa_rad_hf/physician_review_returns")
    / "specificity_ratchet_v3"
)
DEFAULT_DELIVERY = (
    Path("/home/dbw/datasets/public/vqa_rad_hf/physician_review_deliveries")
    / "specificity_ratchet_v3_adjudicator"
)
DEFAULT_IMAGE_ROOT = Path("/home/dbw/datasets/public/vqa_rad_hf")
DEFAULT_OUTPUT = ROOT / "corrected_runs/specificity_ratchet/clinical_returns_v1"
DEFAULT_PACK_LOCK = ROOT / "configs/specificity_ratchet/source_pack_v2_lock.json"
DEFAULT_PARENT_STATE_AUDIT = (
    ROOT / "corrected_runs/specificity_ratchet/parent_before_constraint_audit_v1.json"
)
PARENT_STATE_AUDITOR = (
    ROOT / "anchor/corrected_sgta/audit_specificity_parent_before_constraint_v1.py"
)
CANARY_NAME = "specificity-ratchet-native-canary-v1"
FULL_CAPTURE_NAME = "specificity-ratchet-native-full-capture-v1"
REPLAY_NAME = "specificity-ratchet-visible-replay-v1"
ANALYSIS_NAME = "specificity-ratchet-visible-analysis-v1"
GPU_LOCK = ROOT / "corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_inbox_instructions(inbox: Path) -> None:
    path = inbox / "RETURN_FILES.md"
    if path.exists():
        return
    atomic_text(
        path,
        """# Specificity Ratchet clinical-return inbox

Copy each file under a temporary name, then atomically rename it to the exact
name only after the copy finishes. The monitor requires unchanged size and
SHA-256 over two polls and never creates a clinical label or signature.

Two independent physicians return:

1. `annotations.reviewer_1.completed.csv`
2. `reviewer_1.attestation.json`
3. `annotations.reviewer_2.completed.csv`
4. `reviewer_2.attestation.json`

After both pairs pass, the monitor creates a blinded adjudicator archive in
the external delivery directory named in its heartbeat. A third physician
returns:

5. `adjudication.completed.csv`
6. `adjudicator.attestation.json`

Do not place private provenance, model identity, automatic scores, benchmark
answers, patient identifiers, or coordinator-created attestations here.
""",
    )


def paths(pack: Path, inbox: Path, output: Path) -> dict[str, Path]:
    return {
        "review_1": inbox / "annotations.reviewer_1.completed.csv",
        "attest_1": inbox / "reviewer_1.attestation.json",
        "review_2": inbox / "annotations.reviewer_2.completed.csv",
        "attest_2": inbox / "reviewer_2.attestation.json",
        "adjudication": inbox / "adjudication.completed.csv",
        "attest_adjudicator": inbox / "adjudicator.attestation.json",
        "frozen": output / "frozen",
        "merged": output / "adjudication.with_reviews.csv",
        "merge_metadata": output / "review_merge.json",
        "working_pack": output / "adjudicated_pack_v1",
        "working_pack_lock": output / "adjudicated_pack_v1.lock.json",
        "manifest": output / "replay_manifest_v1/samples.jsonl",
        "manifest_metadata": output / "replay_manifest_v1/metadata.json",
        "preflight": output / "replay_manifest_v1/preflight.json",
        "canary_state": ROOT / f"corrected_runs/detached_jobs/{CANARY_NAME}.json",
        "canary_log": ROOT / f"corrected_runs/detached_jobs/{CANARY_NAME}.log",
        "canary_output": output / "native_capture_huatuo_dev_canary_v1",
        "full_capture_state": ROOT
        / f"corrected_runs/detached_jobs/{FULL_CAPTURE_NAME}.json",
        "full_capture_log": ROOT
        / f"corrected_runs/detached_jobs/{FULL_CAPTURE_NAME}.log",
        "full_capture_output": output / "native_capture_huatuo_all_v1",
        "replay_state": ROOT / f"corrected_runs/detached_jobs/{REPLAY_NAME}.json",
        "replay_log": ROOT / f"corrected_runs/detached_jobs/{REPLAY_NAME}.log",
        "replay_output": output / "visible_replay_huatuo_all_v1",
        "analysis_state": ROOT / f"corrected_runs/detached_jobs/{ANALYSIS_NAME}.json",
        "analysis_log": ROOT / f"corrected_runs/detached_jobs/{ANALYSIS_NAME}.log",
        "analysis_output": output / "visible_replay_huatuo_all_v1/analysis.json",
    }


def human_input_signatures(p: dict[str, Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for key in (
        "review_1",
        "attest_1",
        "review_2",
        "attest_2",
        "adjudication",
        "attest_adjudicator",
    ):
        path = p[key]
        if path.is_file():
            output[key] = {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return output


def _read_single_id(csv_path: Path, field: str) -> str:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or field not in reader.fieldnames:
            raise ValueError(f"{csv_path}: missing {field}")
        ids = {row.get(field, "").strip() for row in reader}
    if len(ids) != 1 or "" in ids:
        raise ValueError(f"{csv_path}: {field} must be one stable nonempty ID")
    return next(iter(ids))


def load_reviewer_attestation(path: Path, expected_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"protocol_id", "reviewer"}:
        raise ValueError(f"{path}: reviewer attestation top-level keys differ")
    if payload["protocol_id"] != PROTOCOL_ID:
        raise ValueError(f"{path}: protocol_id mismatch")
    record = payload["reviewer"]
    expected_keys = {
        "reviewer_id",
        "role",
        "independent_review",
        "blinded_to_private_provenance",
        "completed_at_utc",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError(f"{path}: reviewer attestation keys differ")
    if record["reviewer_id"] != expected_id:
        raise ValueError(f"{path}: reviewer ID differs from completed CSV")
    if record["role"] != "physician":
        raise ValueError(f"{path}: role must be physician")
    if record["independent_review"] is not True:
        raise ValueError(f"{path}: independent_review must be true")
    if record["blinded_to_private_provenance"] is not True:
        raise ValueError(f"{path}: blinded_to_private_provenance must be true")
    if not isinstance(record["completed_at_utc"], str) or not record["completed_at_utc"].strip():
        raise ValueError(f"{path}: completed_at_utc must be nonempty")
    return record


def load_adjudicator_attestation(path: Path, expected_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"protocol_id", "adjudicator"}:
        raise ValueError(f"{path}: adjudicator attestation top-level keys differ")
    if payload["protocol_id"] != PROTOCOL_ID:
        raise ValueError(f"{path}: protocol_id mismatch")
    record = payload["adjudicator"]
    expected_keys = {
        "adjudicator_id",
        "role",
        "blinded_to_private_provenance",
        "completed_at_utc",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError(f"{path}: adjudicator attestation keys differ")
    if record["adjudicator_id"] != expected_id:
        raise ValueError(f"{path}: adjudicator ID differs from completed CSV")
    if record["role"] != "physician":
        raise ValueError(f"{path}: role must be physician")
    if record["blinded_to_private_provenance"] is not True:
        raise ValueError(f"{path}: blinded_to_private_provenance must be true")
    if not isinstance(record["completed_at_utc"], str) or not record["completed_at_utc"].strip():
        raise ValueError(f"{path}: completed_at_utc must be nonempty")
    return record


def freeze_copy(source: Path, directory: Path, label: str) -> Path:
    digest = sha256_file(source)
    suffix = "".join(source.suffixes) or ".dat"
    target = directory / f"{label}.{digest[:16]}{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != digest:
            raise RuntimeError(f"frozen target hash mismatch: {target}")
        return target
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != digest:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"copy hash mismatch: {source}")
    os.replace(temporary, target)
    return target


def directory_closure(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise RuntimeError(f"closure root is not a directory: {directory}")
    closure: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlinks are forbidden in frozen closure: {path}")
        if path.is_file():
            closure[str(path.relative_to(directory))] = sha256_file(path)
    if not closure:
        raise RuntimeError(f"frozen closure is empty: {directory}")
    return closure


def validate_source_pack_lock(pack: Path, lock_path: Path) -> dict[str, Any]:
    if not lock_path.is_file():
        raise RuntimeError(f"missing frozen source-pack lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if set(lock) != {"version", "protocol_id", "files"}:
        raise RuntimeError("source-pack lock schema differs")
    if (
        lock["version"] != "specificity-ratchet-source-pack-lock-v1"
        or lock["protocol_id"] != PROTOCOL_ID
        or not isinstance(lock["files"], dict)
    ):
        raise RuntimeError("source-pack lock protocol differs")
    actual = directory_closure(pack)
    if actual != lock["files"]:
        missing = sorted(set(lock["files"]) - set(actual))
        unexpected = sorted(set(actual) - set(lock["files"]))
        changed = sorted(
            name
            for name in set(actual) & set(lock["files"])
            if actual[name] != lock["files"][name]
        )
        raise RuntimeError(
            "frozen source-pack closure mismatch: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    return lock


def load_parent_state_gate(audit_path: Path, pack: Path) -> dict[str, Any]:
    """Bind the scientific launch gate to one audited source pack and auditor."""

    if not audit_path.is_file():
        raise RuntimeError(f"missing parent-state audit: {audit_path}")
    if not PARENT_STATE_AUDITOR.is_file():
        raise RuntimeError(f"missing parent-state auditor: {PARENT_STATE_AUDITOR}")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_protocol = "specificity-ratchet-parent-before-constraint-audit-v1"
    if payload.get("protocol_id") != expected_protocol:
        raise RuntimeError("parent-state audit protocol differs")
    candidate_path = pack / "candidates.blinded.jsonl"
    candidate_sha256 = sha256_file(candidate_path)
    if payload.get("candidate_sha256") != candidate_sha256:
        raise RuntimeError("parent-state audit is stale for the frozen candidate pack")
    auditor_sha256 = sha256_file(PARENT_STATE_AUDITOR)
    if payload.get("source_sha256") != auditor_sha256:
        raise RuntimeError("parent-state audit was produced by a different auditor")
    outcome_blind = payload.get("outcome_blind_contract")
    if not isinstance(outcome_blind, dict) or any(
        outcome_blind.get(key) is not False
        for key in (
            "physician_reviews_read",
            "adjudication_read",
            "clinical_support_inferred",
        )
    ):
        raise RuntimeError("parent-state audit does not satisfy the outcome-blind contract")
    naming = payload.get("scientific_naming_gate")
    gates = payload.get("gates")
    if not isinstance(naming, dict) or not isinstance(gates, dict):
        raise RuntimeError("parent-state audit lacks scientific launch gates")
    crossing_authorized = naming.get("crossing_authorized") is True
    construct_certifiable = (
        gates.get("current_pack_surface_construct_certifiable") is True
    )
    return {
        "path": str(audit_path.resolve()),
        "sha256": sha256_file(audit_path),
        "protocol_id": expected_protocol,
        "status": payload.get("status"),
        "candidate_sha256": candidate_sha256,
        "auditor_path": str(PARENT_STATE_AUDITOR.resolve()),
        "auditor_sha256": auditor_sha256,
        "crossing_authorized": crossing_authorized,
        "construct_certifiable": construct_certifiable,
        "scientific_gpu_authorized": crossing_authorized and construct_certifiable,
    }


def substrate_no_go_state(parent_state_gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "substrate_no_go_terminal",
        "terminal_reason": "parent_state_crossing_not_identified",
        "parent_state_audit": parent_state_gate,
        "native_canary_authorized": False,
        "native_full_capture_authorized": False,
        "visible_replay_authorized": False,
        "scientific_gpu_authorized": False,
        "construct_pilot_returns_preserved": True,
        "clinical_return_templates_preserved": True,
        "retry_authorized": False,
    }


def _require_scientific_gate(parent_state_gate: dict[str, Any]) -> dict[str, Any] | None:
    if parent_state_gate.get("scientific_gpu_authorized") is not True:
        return substrate_no_go_state(parent_state_gate)
    return None


def _write_json_once_or_equal(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"write-once JSON collision: {path}")
        return
    atomic_text(path, rendered)


def _frozen_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _load_frozen_review_bundle(p: dict[str, Path]) -> dict[str, Path]:
    if not p["merge_metadata"].is_file():
        raise RuntimeError("merged review exists without write-once merge metadata")
    metadata = json.loads(p["merge_metadata"].read_text(encoding="utf-8"))
    if metadata.get("output_sha256") != sha256_file(p["merged"]):
        raise RuntimeError("merged review hash differs from merge metadata")
    records = metadata.get("frozen_human_inputs")
    expected = {"review_1", "review_2", "attest_1", "attest_2"}
    if not isinstance(records, dict) or set(records) != expected:
        raise RuntimeError("merge metadata lacks exact frozen human-input bundle")
    frozen_root = p["frozen"].resolve()
    output: dict[str, Path] = {}
    for name in sorted(expected):
        record = records[name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise RuntimeError(f"invalid frozen input record: {name}")
        path = Path(record["path"]).resolve()
        if path.parent != frozen_root or not path.is_file():
            raise RuntimeError(f"frozen input escaped or disappeared: {name}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen input hash mismatch: {name}")
        output[name] = path
    return output


def _seal_working_pack(p: dict[str, Path]) -> None:
    validated = validate_adjudication(p["working_pack"])
    payload = {
        "version": "specificity-ratchet-working-pack-lock-v1",
        "protocol_id": PROTOCOL_ID,
        "files": directory_closure(p["working_pack"]),
        "validated_input_sha256": validated.input_sha256,
        "clinical_truth_created_by_monitor": False,
    }
    _write_json_once_or_equal(p["working_pack_lock"], payload)


def _validate_working_pack_closure(p: dict[str, Path]) -> None:
    if not p["working_pack_lock"].is_file():
        raise RuntimeError("existing working pack lacks write-once closure lock")
    lock = json.loads(p["working_pack_lock"].read_text(encoding="utf-8"))
    if (
        lock.get("version") != "specificity-ratchet-working-pack-lock-v1"
        or lock.get("protocol_id") != PROTOCOL_ID
        or lock.get("clinical_truth_created_by_monitor") is not False
    ):
        raise RuntimeError("working-pack lock protocol differs")
    if directory_closure(p["working_pack"]) != lock.get("files"):
        raise RuntimeError("existing working-pack directory closure changed")
    validated = validate_adjudication(p["working_pack"])
    if validated.input_sha256 != lock.get("validated_input_sha256"):
        raise RuntimeError("existing working-pack adjudication hash closure changed")


def _copy_working_pack(
    *,
    source_pack: Path,
    reviewer_1: Path,
    reviewer_2: Path,
    adjudication: Path,
    attestations: dict[str, Any],
    target: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="specificity-adjudicated-") as temporary_name:
        temporary = Path(temporary_name) / "pack"
        shutil.copytree(source_pack, temporary)
        shutil.copyfile(reviewer_1, temporary / "annotations.reviewer_1.csv")
        shutil.copyfile(reviewer_2, temporary / "annotations.reviewer_2.csv")
        shutil.copyfile(adjudication, temporary / "adjudication.csv")
        (temporary / "physician_attestations.json").write_text(
            json.dumps(attestations, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_adjudication(temporary)
        if target.exists():
            for source in temporary.iterdir():
                destination = target / source.name
                if not destination.is_file() or sha256_file(destination) != sha256_file(source):
                    raise RuntimeError(f"working-pack collision: {destination}")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(target.name + ".tmp")
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(temporary, staged)
        os.replace(staged, target)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        )
    return result


def _compile_and_preflight(
    p: dict[str, Path], image_root: Path
) -> dict[str, Any]:
    if not p["manifest"].exists() and not p["manifest_metadata"].exists():
        _run(
            [
                sys.executable,
                "anchor/corrected_sgta/compile_specificity_ratchet_replay_manifest_v1.py",
                "--pack",
                str(p["working_pack"]),
                "--attestations",
                str(p["working_pack"] / "physician_attestations.json"),
                "--output",
                str(p["manifest"]),
                "--metadata-output",
                str(p["manifest_metadata"]),
            ]
        )
    if not p["manifest"].is_file() or not p["manifest_metadata"].is_file():
        raise RuntimeError("partial replay-manifest write requires audit")
    if not p["preflight"].exists():
        result = _run(
            [
                sys.executable,
                "anchor/corrected_sgta/specificity_ratchet_visible_replay_v1.py",
                "--manifest",
                str(p["manifest"]),
                "--metadata",
                str(p["manifest_metadata"]),
                "--native-capture",
                str(p["canary_output"] / "native_capture.json"),
                "--image-root",
                str(image_root),
                "--output-dir",
                str(p["canary_output"] / "preflight_unused"),
                "--split",
                "all",
                "--preflight-only",
            ]
        )
        payload = json.loads(result.stdout)
        if payload.get("status") != "preflight_passed" or payload.get("gpu_started") is not False:
            raise RuntimeError("unexpected CPU preflight result")
        _write_json_once_or_equal(p["preflight"], payload)
    return json.loads(p["preflight"].read_text(encoding="utf-8"))


def _launch_or_monitor_canary(
    p: dict[str, Path], image_root: Path
) -> dict[str, Any]:
    if not p["canary_state"].exists():
        command = [
            sys.executable,
            "scripts/start_detached_job.py",
            "--name",
            CANARY_NAME,
            "--log",
            str(p["canary_log"]),
            "--state",
            str(p["canary_state"]),
            "--",
            *_gpu_command(
                [
                    "env",
                    "PYTHONPATH=.",
                    "/opt/miniconda3/envs/huatuo/bin/python",
                    "anchor/corrected_sgta/capture_huatuo_specificity_native_v1.py",
                    "--manifest",
                    str(p["manifest"]),
                    "--metadata",
                    str(p["manifest_metadata"]),
                    "--image-root",
                    str(image_root),
                    "--output-dir",
                    str(p["canary_output"]),
                    "--split",
                    "dev",
                    "--limit-cases",
                    "1",
                    "--adapter-factory",
                    "anchor.corrected_sgta.huatuo_specificity_ratchet_adapter_v1:create_adapter",
                    "--adapter-config",
                    "configs/specificity_ratchet/huatuo_full_replay_v1.json",
                ]
            ),
        ]
        _run(command)
        return {"stage": "native_canary_launched", "canary_job": CANARY_NAME}
    state = _read_job_state(p["canary_state"], CANARY_NAME)
    status = state.get("status")
    if status in {"starting", "running"}:
        return {
            "stage": "native_canary_running",
            "canary_job": CANARY_NAME,
            "canary_child_pid": state.get("child_pid"),
        }
    canary_path = p["canary_output"] / "CANARY.json"
    if status == "failed":
        return {
            "stage": "native_canary_failed_terminal",
            "canary_job": CANARY_NAME,
            "exit_code": state.get("exit_code"),
            "retry_authorized": False,
        }
    if status != "done" or not canary_path.is_file():
        raise RuntimeError(f"unexpected canary state: {status!r}")
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    expected = {
        "status": "canary_passed",
        "manifest_sha256": sha256_file(p["manifest"]),
        "metadata_sha256": sha256_file(p["manifest_metadata"]),
        "target_model_family": "huatuogpt-vision-7b",
        "split": "dev",
        "n_captured_cases": 1,
        "n_identity_failures": 0,
        "direct_output_sequences_captured_for_every_selected_case": True,
    }
    mismatches = {
        key: {"expected": value, "observed": canary.get(key)}
        for key, value in expected.items()
        if canary.get(key) != value
    }
    if mismatches:
        return {
            "stage": "native_canary_failed_terminal",
            "canary_job": CANARY_NAME,
            "capture_status": canary.get("status"),
            "contract_mismatches": mismatches,
            "retry_authorized": False,
        }
    return {
        "stage": "native_canary_passed_ready_for_full_capture",
        "canary_job": CANARY_NAME,
        "canary_sha256": sha256_file(canary_path),
        "confirmatory_claim_authorized": False,
        "reason": "the frozen 70-case pack remains a bounded pilot",
    }


def _read_job_state(path: Path, expected_name: str) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("name") != expected_name:
        raise RuntimeError(f"detached state name mismatch: {path}")
    if state.get("status") not in {"starting", "running", "done", "failed"}:
        raise RuntimeError(f"unexpected detached state status: {path}")
    return state


def _launch_detached(
    *,
    name: str,
    log: Path,
    state: Path,
    child_command: list[str],
) -> None:
    _run(
        [
            sys.executable,
            "scripts/start_detached_job.py",
            "--name",
            name,
            "--log",
            str(log),
            "--state",
            str(state),
            "--",
            *child_command,
        ]
    )


def _gpu_command(command: list[str]) -> list[str]:
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    return ["flock", str(GPU_LOCK), *command]


def _monitor_full_capture(p: dict[str, Path], image_root: Path) -> dict[str, Any]:
    if not p["full_capture_state"].exists():
        _launch_detached(
            name=FULL_CAPTURE_NAME,
            log=p["full_capture_log"],
            state=p["full_capture_state"],
            child_command=_gpu_command(
                [
                    "env",
                    "PYTHONPATH=.",
                    "/opt/miniconda3/envs/huatuo/bin/python",
                    "anchor/corrected_sgta/capture_huatuo_specificity_native_v1.py",
                    "--manifest",
                    str(p["manifest"]),
                    "--metadata",
                    str(p["manifest_metadata"]),
                    "--image-root",
                    str(image_root),
                    "--output-dir",
                    str(p["full_capture_output"]),
                    "--split",
                    "all",
                    "--adapter-factory",
                    "anchor.corrected_sgta.huatuo_specificity_ratchet_adapter_v1:create_adapter",
                    "--adapter-config",
                    "configs/specificity_ratchet/huatuo_full_replay_v1.json",
                ]
            ),
        )
        return {"stage": "native_full_capture_launched", "job": FULL_CAPTURE_NAME}
    state = _read_job_state(p["full_capture_state"], FULL_CAPTURE_NAME)
    if state["status"] in {"starting", "running"}:
        return {
            "stage": "native_full_capture_running",
            "job": FULL_CAPTURE_NAME,
            "child_pid": state.get("child_pid"),
        }
    capture_path = p["full_capture_output"] / "native_capture.json"
    if state["status"] == "failed":
        capture_status = None
        if capture_path.is_file():
            capture_status = json.loads(capture_path.read_text()).get("status")
        return {
            "stage": "native_full_capture_failed_terminal",
            "job": FULL_CAPTURE_NAME,
            "exit_code": state.get("exit_code"),
            "capture_status": capture_status,
            "retry_authorized": False,
        }
    if not capture_path.is_file():
        raise RuntimeError("full capture job is done without native_capture.json")
    capture = json.loads(capture_path.read_text())
    expected = {
        "status": "complete_passed",
        "manifest_sha256": sha256_file(p["manifest"]),
        "metadata_sha256": sha256_file(p["manifest_metadata"]),
        "target_model_family": "huatuogpt-vision-7b",
        "split": "all",
        "n_identity_failures": 0,
        "direct_output_sequences_captured_for_every_selected_case": True,
    }
    mismatches = {
        key: {"expected": value, "observed": capture.get(key)}
        for key, value in expected.items()
        if capture.get(key) != value
    }
    cases = capture.get("cases")
    if (
        not isinstance(cases, list)
        or not cases
        or capture.get("n_captured_cases") != len(cases)
        or capture.get("n_manifest_cases_in_split") != len(cases)
        or any(case.get("identity_passed") is not True for case in cases)
    ):
        mismatches["cases"] = {"expected": "nonempty and all identity_passed", "observed": "invalid"}
    if mismatches:
        return {
            "stage": "native_full_capture_failed_terminal",
            "job": FULL_CAPTURE_NAME,
            "contract_mismatches": mismatches,
            "retry_authorized": False,
        }
    return {
        "stage": "native_full_capture_passed",
        "job": FULL_CAPTURE_NAME,
        "native_capture_sha256": sha256_file(capture_path),
    }


def _monitor_replay(p: dict[str, Path], image_root: Path) -> dict[str, Any]:
    if not p["replay_state"].exists():
        _launch_detached(
            name=REPLAY_NAME,
            log=p["replay_log"],
            state=p["replay_state"],
            child_command=_gpu_command(
                [
                    "env",
                    "PYTHONPATH=.",
                    "/opt/miniconda3/envs/huatuo/bin/python",
                    "anchor/corrected_sgta/specificity_ratchet_visible_replay_v1.py",
                    "--manifest",
                    str(p["manifest"]),
                    "--metadata",
                    str(p["manifest_metadata"]),
                    "--native-capture",
                    str(p["full_capture_output"] / "native_capture.json"),
                    "--image-root",
                    str(image_root),
                    "--output-dir",
                    str(p["replay_output"]),
                    "--split",
                    "all",
                    "--adapter-factory",
                    "anchor.corrected_sgta.huatuo_specificity_ratchet_adapter_v1:create_adapter",
                    "--adapter-config",
                    "configs/specificity_ratchet/huatuo_full_replay_v1.json",
                ]
            ),
        )
        return {"stage": "visible_replay_launched", "job": REPLAY_NAME}
    state = _read_job_state(p["replay_state"], REPLAY_NAME)
    if state["status"] in {"starting", "running"}:
        return {
            "stage": "visible_replay_running",
            "job": REPLAY_NAME,
            "child_pid": state.get("child_pid"),
        }
    if state["status"] == "failed":
        return {
            "stage": "visible_replay_failed_terminal",
            "job": REPLAY_NAME,
            "exit_code": state.get("exit_code"),
            "retry_authorized": False,
        }
    complete_path = p["replay_output"] / "COMPLETE.json"
    if not complete_path.is_file():
        raise RuntimeError("replay job is done without COMPLETE.json")
    complete = json.loads(complete_path.read_text())
    if (
        complete.get("status") != "complete"
        or complete.get("native_capture_enforced") is not True
        or not isinstance(complete.get("rows"), int)
        or complete["rows"] <= 0
    ):
        return {
            "stage": "visible_replay_failed_terminal",
            "job": REPLAY_NAME,
            "completion_contract": complete,
            "retry_authorized": False,
        }
    return {
        "stage": "visible_replay_passed",
        "job": REPLAY_NAME,
        "complete_sha256": sha256_file(complete_path),
    }


def _monitor_analysis(p: dict[str, Path]) -> dict[str, Any]:
    if not p["analysis_state"].exists():
        _launch_detached(
            name=ANALYSIS_NAME,
            log=p["analysis_log"],
            state=p["analysis_state"],
            child_command=[
                "env",
                "PYTHONPATH=.",
                sys.executable,
                "anchor/corrected_sgta/analyze_specificity_ratchet_visible_replay_v1.py",
                "--run-dir",
                str(p["replay_output"]),
                "--output",
                str(p["analysis_output"]),
                "--bootstrap-replicates",
                "5000",
                "--seed",
                "7319",
            ],
        )
        return {"stage": "visible_analysis_launched", "job": ANALYSIS_NAME}
    state = _read_job_state(p["analysis_state"], ANALYSIS_NAME)
    if state["status"] in {"starting", "running"}:
        return {
            "stage": "visible_analysis_running",
            "job": ANALYSIS_NAME,
            "child_pid": state.get("child_pid"),
        }
    if not p["analysis_output"].is_file():
        return {
            "stage": "visible_analysis_operational_failure_terminal",
            "job": ANALYSIS_NAME,
            "exit_code": state.get("exit_code"),
            "retry_authorized": False,
        }
    analysis = json.loads(p["analysis_output"].read_text())
    status = analysis.get("status")
    if status not in {"passed", "pilot_only", "underpowered", "failed"}:
        raise RuntimeError(f"unexpected analysis status: {status!r}")
    if (status == "passed") != (state["status"] == "done"):
        raise RuntimeError("analysis exit state disagrees with scientific status")
    return {
        "stage": "visible_analysis_terminal",
        "job": ANALYSIS_NAME,
        "scientific_status": status,
        "analysis_sha256": sha256_file(p["analysis_output"]),
        "retry_authorized": False,
        "confirmatory_claim_authorized": False,
        "reason": "the frozen 70-case pack remains a bounded pilot",
    }


def _advance_scientific_chain(
    p: dict[str, Path], image_root: Path, parent_state_gate: dict[str, Any]
) -> dict[str, Any]:
    refused = _require_scientific_gate(parent_state_gate)
    if refused is not None:
        return refused
    canary = _launch_or_monitor_canary(p, image_root)
    if canary["stage"] != "native_canary_passed_ready_for_full_capture":
        return canary
    full_capture = _monitor_full_capture(p, image_root)
    if full_capture["stage"] != "native_full_capture_passed":
        return full_capture
    replay = _monitor_replay(p, image_root)
    if replay["stage"] != "visible_replay_passed":
        return replay
    return _monitor_analysis(p)


def advance(
    *,
    pack: Path,
    delivery: Path,
    image_root: Path,
    p: dict[str, Path],
    parent_state_gate: dict[str, Any],
) -> dict[str, Any]:
    refused = _require_scientific_gate(parent_state_gate)
    if refused is not None:
        return refused
    if p["working_pack"].exists():
        _validate_working_pack_closure(p)
        _compile_and_preflight(p, image_root)
        return _advance_scientific_chain(p, image_root, parent_state_gate)

    if p["merged"].exists():
        required = ("adjudication", "attest_adjudicator")
        missing = [str(p[key]) for key in required if not p[key].is_file()]
        if missing:
            delivery_index = delivery / "adjudicator_delivery.json"
            return {
                "stage": "waiting_for_blinded_adjudication",
                "missing": missing,
                "adjudicator_delivery": str(delivery_index.resolve()),
                "adjudicator_delivery_sha256": sha256_file(delivery_index),
            }
        frozen_reviews = _load_frozen_review_bundle(p)
        adjudicator_id = _read_single_id(p["adjudication"], "adjudicator_id")
        adjudicator = load_adjudicator_attestation(
            p["attest_adjudicator"], adjudicator_id
        )
        reviewer_1_id = _read_single_id(frozen_reviews["review_1"], "reviewer_id")
        reviewer_2_id = _read_single_id(frozen_reviews["review_2"], "reviewer_id")
        if adjudicator_id in {reviewer_1_id, reviewer_2_id}:
            raise ValueError("adjudicator ID must differ from both independent reviewers")
        reviewer_1 = load_reviewer_attestation(
            frozen_reviews["attest_1"], reviewer_1_id
        )
        reviewer_2 = load_reviewer_attestation(
            frozen_reviews["attest_2"], reviewer_2_id
        )
        attestations = {
            "protocol_id": PROTOCOL_ID,
            "reviewers": [reviewer_1, reviewer_2],
            "adjudicator": adjudicator,
        }
        frozen_adjudication = freeze_copy(
            p["adjudication"], p["frozen"], "adjudication.completed"
        )
        frozen_adjudicator_attestation = freeze_copy(
            p["attest_adjudicator"], p["frozen"], "adjudicator.attestation"
        )
        adjudicator = load_adjudicator_attestation(
            frozen_adjudicator_attestation, adjudicator_id
        )
        attestations["adjudicator"] = adjudicator
        _copy_working_pack(
            source_pack=pack,
            reviewer_1=frozen_reviews["review_1"],
            reviewer_2=frozen_reviews["review_2"],
            adjudication=frozen_adjudication,
            attestations=attestations,
            target=p["working_pack"],
        )
        _seal_working_pack(p)
        return {
            "stage": "physician_adjudication_admitted",
            "working_pack": str(p["working_pack"].resolve()),
            "next": "compile_replay_manifest_and_cpu_preflight",
        }

    required = ("review_1", "attest_1", "review_2", "attest_2")
    missing = [str(p[key]) for key in required if not p[key].is_file()]
    if missing:
        return {"stage": "waiting_for_independent_reviews", "missing": missing}
    reviewer_1_id = _read_single_id(p["review_1"], "reviewer_id")
    reviewer_2_id = _read_single_id(p["review_2"], "reviewer_id")
    if reviewer_1_id == reviewer_2_id:
        raise ValueError("independent reviewer IDs must differ")
    reviewer_1 = load_reviewer_attestation(p["attest_1"], reviewer_1_id)
    reviewer_2 = load_reviewer_attestation(p["attest_2"], reviewer_2_id)
    frozen_review_1 = freeze_copy(p["review_1"], p["frozen"], "reviewer_1.completed")
    frozen_review_2 = freeze_copy(p["review_2"], p["frozen"], "reviewer_2.completed")
    frozen_attest_1 = freeze_copy(
        p["attest_1"], p["frozen"], "reviewer_1.attestation"
    )
    frozen_attest_2 = freeze_copy(
        p["attest_2"], p["frozen"], "reviewer_2.attestation"
    )
    header, rows, metadata = merge_reviews(
        candidates_path=pack / "candidates.blinded.jsonl",
        schema_path=pack / "annotation_schema.json",
        template_path=pack / "adjudication.csv",
        reviewer_1_path=frozen_review_1,
        reviewer_2_path=frozen_review_2,
    )
    _write_csv_once(p["merged"], header, rows)
    metadata.update(
        {
            "output": str(p["merged"].resolve()),
            "output_sha256": sha256_file(p["merged"]),
            "reviewer_attestation_sha256": {
                reviewer_1_id: sha256_file(p["attest_1"]),
                reviewer_2_id: sha256_file(p["attest_2"]),
            },
            "reviewer_attestations_validated_not_synthesized": True,
            "frozen_human_inputs": {
                "review_1": _frozen_record(frozen_review_1),
                "review_2": _frozen_record(frozen_review_2),
                "attest_1": _frozen_record(frozen_attest_1),
                "attest_2": _frozen_record(frozen_attest_2),
            },
        }
    )
    _write_json_once_or_equal(p["merge_metadata"], metadata)
    package = package_adjudication(
        pack_dir=pack,
        merged_csv=p["merged"],
        image_root=image_root,
        output_dir=delivery,
    )
    return {
        "stage": "blinded_adjudication_prepared",
        "adjudicator_delivery": package["archive"],
        "adjudicator_delivery_sha256": package["archive_sha256"],
        "next": "waiting_for_blinded_adjudication",
        "clinical_labels_synthesized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--adjudicator-delivery-dir", type=Path, default=DEFAULT_DELIVERY)
    parser.add_argument("--pack-lock", type=Path, default=DEFAULT_PACK_LOCK)
    parser.add_argument(
        "--parent-state-audit", type=Path, default=DEFAULT_PARENT_STATE_AUDIT
    )
    parser.add_argument("--heartbeat", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError("interval must be at least one second")
    heartbeat = args.heartbeat or args.output / "monitor.heartbeat.json"
    for required in (
        args.pack / "candidates.blinded.jsonl",
        args.pack / "annotation_schema.json",
        args.pack / "adjudication.csv",
        args.image_root / "test_images",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    validate_source_pack_lock(args.pack, args.pack_lock)
    parent_state_gate = load_parent_state_gate(args.parent_state_audit, args.pack)
    args.inbox.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    args.adjudicator_delivery_dir.mkdir(parents=True, exist_ok=True)
    write_inbox_instructions(args.inbox)
    resolved = paths(args.pack, args.inbox, args.output)
    previous_signatures: dict[str, dict[str, Any]] = {}
    while True:
        try:
            signatures = human_input_signatures(resolved)
            if signatures and signatures != previous_signatures:
                state: dict[str, Any] = {
                    "stage": "waiting_for_stable_human_inputs",
                    "required_unchanged_polls": 2,
                    "human_input_signatures": signatures,
                }
            else:
                state = advance(
                    pack=args.pack,
                    delivery=args.adjudicator_delivery_dir,
                    image_root=args.image_root,
                    p=resolved,
                    parent_state_gate=parent_state_gate,
                )
            payload = {
                "version": VERSION,
                "time": utc_now(),
                "pid": os.getpid(),
                "inbox": str(args.inbox.resolve()),
                "output": str(args.output.resolve()),
                "clinical_labels_synthesized": False,
                "attestations_synthesized": False,
                "private_provenance_exposed_to_reviewers": False,
                "confirmatory_claim_authorized": False,
                "parent_state_audit": parent_state_gate,
                **state,
            }
            previous_signatures = signatures
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            payload = {
                "version": VERSION,
                "time": utc_now(),
                "pid": os.getpid(),
                "inbox": str(args.inbox.resolve()),
                "output": str(args.output.resolve()),
                "stage": "input_or_transition_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "clinical_labels_synthesized": False,
                "attestations_synthesized": False,
                "private_provenance_exposed_to_reviewers": False,
                "confirmatory_claim_authorized": False,
                "parent_state_audit": parent_state_gate,
            }
        atomic_write_json(heartbeat, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        terminal = payload.get("stage") in {
            "native_canary_failed_terminal",
            "native_full_capture_failed_terminal",
            "visible_replay_failed_terminal",
            "visible_analysis_operational_failure_terminal",
            "visible_analysis_terminal",
            "substrate_no_go_terminal",
        }
        if args.once or terminal:
            return
        if payload.get("stage") in {
            "physician_adjudication_admitted",
            "blinded_adjudication_prepared",
            "native_canary_launched",
            "native_full_capture_launched",
            "visible_replay_launched",
            "visible_analysis_launched",
        }:
            continue
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
