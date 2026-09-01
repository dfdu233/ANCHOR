#!/usr/bin/env python3
"""Persistently advance CECD Stage 1 into a formal-CE launch handoff.

The monitor never consumes model outputs or synthesizes human returns.  After a
canonical three-stage GO, it deterministically builds the outcome-blind
preflight, binds the narrow authorization, and emits one write-once detached
formal-CE launch handoff.  Emitting the handoff does not launch model/GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anchor.corrected_sgta.authorize_cecd_dual_semantics_preflight_v1 import (
    VERSION as AUTHORIZATION_VERSION,
    authorize,
)
from anchor.corrected_sgta.build_cecd_dual_semantics_preflight_v1 import (
    CANONICAL_GPU_LOCK_RELATIVE,
    DEFAULT_HUATUO_MODEL,
    DEFAULT_HUATUO_SOURCE,
    DEFAULT_HULU_MODEL,
    build_preflight,
    file_record as build_file_record,
    validate_build_receipt,
    write_once_json,
)
import scripts.monitor_cecd_admission_pipeline as cecd_monitor


VERSION = "cecd-dual-semantics-transition-monitor-v3-operational-closure"
ROOT = Path("/home/dbw/ANCHOR")
DEFAULT_STAGE_STATE = ROOT / "corrected_runs/detached_jobs/cecd-three-stage-v3.json"
DEFAULT_STAGE_ANALYSIS = (
    ROOT / "corrected_runs/vindr_v2/cecd_three_stage_v3/confirmation_locked.json"
)
DEFAULT_STAGE_INPUT_GATE = (
    ROOT / "corrected_runs/vindr_v2/cecd_three_stage_v3/input_gate.json"
)
DEFAULT_ADMISSION = ROOT / "corrected_runs/vindr_v2/cecd_human_admission_v2/analysis.json"
DEFAULT_PREFLIGHT = ROOT / "configs/cecd_dual_semantics_preflight_v1.json"
DEFAULT_PREFLIGHT_BUILD = (
    ROOT / "corrected_runs/vindr_v2/cecd_dual_semantics_v1/preflight_build.json"
)
DEFAULT_INPUT_ROOT = ROOT / "configs/cecd_dual_semantics_inputs_v1"
DEFAULT_METHOD_OUTPUT_ROOT = (
    ROOT / "corrected_runs/vindr_v2/cecd_dual_semantics_v1/method_outputs"
)
DEFAULT_AUTHORIZATION = (
    ROOT / "corrected_runs/vindr_v2/cecd_dual_semantics_v1/authorization.json"
)
DEFAULT_HEARTBEAT = (
    ROOT / "corrected_runs/vindr_v2/cecd_dual_semantics_v1/monitor.heartbeat.json"
)
DEFAULT_RUNNER_HANDOFF = (
    ROOT / "corrected_runs/vindr_v2/cecd_dual_semantics_v1/formal_ce_launch_handoff.json"
)
DEFAULT_FORMAL_JOB_STATE = ROOT / "corrected_runs/detached_jobs/cecd-dual-formal-ce-v1.json"
DEFAULT_FORMAL_JOB_LOG = ROOT / "corrected_runs/detached_jobs/cecd-dual-formal-ce-v1.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def _validate_existing_authorization(
    *, authorization: Path, stage_analysis: Path, stage_input_gate: Path,
    admission: Path, preflight: Path, preflight_build: Path,
) -> dict[str, Any]:
    """Validate the write-once handoff without touching later method outputs."""

    payload = _load_object(authorization, "dual-semantics authorization")
    fingerprint = payload.get("fingerprint")
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    expected_fingerprint = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    records = {
        "stage1_analysis": stage_analysis,
        "stage1_input_gate": stage_input_gate,
        "admission": admission,
        "preflight": preflight,
        "preflight_build": preflight_build,
    }
    for key, path in records.items():
        record = payload.get(key)
        if (
            not isinstance(record, dict)
            or record.get("path") != str(path.resolve())
            or record.get("sha256") != cecd_monitor.sha256_file(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"existing authorization {key} binding mismatch")
    if (
        payload.get("version") != AUTHORIZATION_VERSION
        or payload.get("status") != "controlled_dual_semantics_comparison_authorized"
        or payload.get("controlled_method_comparison_authorized") is not True
        or payload.get("general_hidden_state_stage_authorized") is not False
        or payload.get("general_gpu_authorized") is not False
        or payload.get("paper_claim_authorized") is not False
        or payload.get("method_outputs_consumed") is not False
        or fingerprint != expected_fingerprint
    ):
        raise RuntimeError("existing dual-semantics authorization contract mismatch")
    return payload


def emit_runner_handoff(
    *, authorization: Path, preflight: Path, preflight_build: Path,
    handoff: Path, formal_job_state: Path, formal_job_log: Path, root: Path,
) -> dict[str, Any]:
    """Freeze the exact detached formal-CE command without executing it."""

    validate_build_receipt(
        receipt_path=preflight_build,
        stage_state=root / "corrected_runs/detached_jobs/cecd-three-stage-v3.json",
        stage_analysis=root / "corrected_runs/vindr_v2/cecd_three_stage_v3/confirmation_locked.json",
        stage_input_gate=root / "corrected_runs/vindr_v2/cecd_three_stage_v3/input_gate.json",
        admission=root / "corrected_runs/vindr_v2/cecd_human_admission_v2/analysis.json",
        preflight_path=preflight,
        root=root,
    )
    gpu_lock = (root / CANONICAL_GPU_LOCK_RELATIVE).resolve()
    worker = (root / "anchor/corrected_sgta/cecd_dual_semantics_worker_v1.py").resolve()
    shell_runner = (root / "scripts/run_cecd_dual_semantics_controlled_v1.sh").resolve()
    detached = (root / "scripts/start_detached_job.py").resolve()
    command = [
        str((root / ".venv-full/bin/python").resolve()),
        str(detached),
        "--name", "cecd-dual-formal-ce-v1",
        "--log", str(formal_job_log.resolve()),
        "--state", str(formal_job_state.resolve()),
        "--", "env",
        f"CECD_DUAL_WORKER={worker}",
        "CECD_DUAL_EXECUTE_CE_ONLY=1",
        "bash", str(shell_runner),
    ]
    payload: dict[str, Any] = {
        "schema_version": "cecd-dual-semantics-formal-ce-launch-handoff-v1",
        "status": "ready_not_launched",
        "authorization": build_file_record(authorization),
        "preflight": build_file_record(preflight),
        "preflight_build": build_file_record(preflight_build),
        "formal_scope": "seven implemented centered-logit CE controls only",
        "execute_ce_only": True,
        "oe_authorized": False,
        "hidden_intervention_authorized": False,
        "treble_variants_authorized": False,
        "canonical_gpu_lock": str(gpu_lock),
        "detached_job_state": str(formal_job_state.resolve()),
        "detached_job_log": str(formal_job_log.resolve()),
        "launch_command": command,
        "launched_by_transition_monitor": False,
        "raw_model_output_rows_consumed_by_handoff": False,
        "paper_claim_authorized": False,
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_once_json(handoff, payload)
    return payload


def advance(
    *,
    stage_state: Path,
    stage_analysis: Path,
    stage_input_gate: Path,
    admission: Path,
    preflight: Path,
    preflight_build: Path,
    input_root: Path,
    method_output_root: Path,
    authorization: Path,
    runner_handoff: Path,
    formal_job_state: Path,
    formal_job_log: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not stage_state.is_file():
        return {
            "stage": "waiting_for_two_model_stage1",
            "stage1_state_present": False,
        }
    state = _load_object(stage_state, "Stage-1 detached state")
    if state.get("name") != "cecd-three-stage-v3":
        raise RuntimeError("legacy pilot-as-dev Stage-1 state is not eligible")
    status = state.get("status")
    if status in {"starting", "running"}:
        return {
            "stage": "waiting_for_two_model_stage1",
            "stage1_state_present": True,
            "stage1_status": status,
            "stage1_child_pid": state.get("child_pid"),
        }
    if status == "failed":
        return {
            "stage": "two_model_stage1_failed_terminal",
            "stage1_status": "failed",
            "exit_code": state.get("exit_code"),
            "retry_authorized": False,
        }
    if status != "done":
        raise RuntimeError(f"unexpected Stage-1 detached status: {status!r}")
    if not stage_analysis.is_file() or not stage_input_gate.is_file() or not admission.is_file():
        raise RuntimeError("completed Stage 1 lacks analysis, input gate, or admission")
    result = cecd_monitor.validate_stage_result(
        result_path=stage_analysis.resolve(), admission=admission.resolve()
    )
    method_gate = result.get("gate", {}).get(
        "authorized_for_method_level_treble_adapter_run"
    )
    if method_gate is not True:
        return {
            "stage": "two_model_stage1_no_go_terminal",
            "behavioral_gate_passed": False,
            "controlled_method_comparison_authorized": False,
            "retry_authorized": False,
        }
    build_preflight(
        stage_state=stage_state,
        stage_analysis=stage_analysis,
        stage_input_gate=stage_input_gate,
        admission=admission,
        preflight_path=preflight,
        build_receipt=preflight_build,
        input_root=input_root,
        method_output_root=method_output_root,
        huatuo_model=DEFAULT_HUATUO_MODEL,
        hulu_model=DEFAULT_HULU_MODEL,
        huatuo_source_root=DEFAULT_HUATUO_SOURCE,
        root=root,
    )
    if authorization.is_file():
        bound = _validate_existing_authorization(
            authorization=authorization,
            stage_analysis=stage_analysis,
            stage_input_gate=stage_input_gate,
            admission=admission,
            preflight=preflight,
            preflight_build=preflight_build,
        )
    else:
        bound = authorize(
            stage1_analysis=stage_analysis,
            stage1_input_gate=stage_input_gate,
            admission=admission,
            preflight_path=preflight,
            preflight_build=preflight_build,
            output=authorization,
            root=root,
        )
    handoff = emit_runner_handoff(
        authorization=authorization,
        preflight=preflight,
        preflight_build=preflight_build,
        handoff=runner_handoff,
        formal_job_state=formal_job_state,
        formal_job_log=formal_job_log,
        root=root,
    )
    return {
        "stage": "controlled_comparison_authorized_runner_handoff_ready",
        "behavioral_gate_passed": True,
        "authorization": str(authorization.resolve()),
        "authorization_fingerprint": bound["fingerprint"],
        "preflight_build": str(preflight_build.resolve()),
        "runner_handoff": str(runner_handoff.resolve()),
        "runner_handoff_fingerprint": handoff["fingerprint"],
        "controlled_method_comparison_authorized": True,
        "general_gpu_authorized": False,
        "gpu_launched_by_monitor": False,
        "paper_claim_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-state", type=Path, default=DEFAULT_STAGE_STATE)
    parser.add_argument("--stage-analysis", type=Path, default=DEFAULT_STAGE_ANALYSIS)
    parser.add_argument("--stage-input-gate", type=Path, default=DEFAULT_STAGE_INPUT_GATE)
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--preflight-build", type=Path, default=DEFAULT_PREFLIGHT_BUILD)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--method-output-root", type=Path, default=DEFAULT_METHOD_OUTPUT_ROOT)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--runner-handoff", type=Path, default=DEFAULT_RUNNER_HANDOFF)
    parser.add_argument("--formal-job-state", type=Path, default=DEFAULT_FORMAL_JOB_STATE)
    parser.add_argument("--formal-job-log", type=Path, default=DEFAULT_FORMAL_JOB_LOG)
    parser.add_argument("--heartbeat", type=Path, default=DEFAULT_HEARTBEAT)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError("interval must be at least one second")
    while True:
        try:
            state = advance(
                stage_state=args.stage_state,
                stage_analysis=args.stage_analysis,
                stage_input_gate=args.stage_input_gate,
                admission=args.admission,
                preflight=args.preflight,
                preflight_build=args.preflight_build,
                input_root=args.input_root,
                method_output_root=args.method_output_root,
                authorization=args.authorization,
                runner_handoff=args.runner_handoff,
                formal_job_state=args.formal_job_state,
                formal_job_log=args.formal_job_log,
                root=ROOT,
            )
            payload = {
                "version": VERSION,
                "time": utc_now(),
                "pid": os.getpid(),
                "clinical_labels_synthesized": False,
                "attestations_synthesized": False,
                "method_outputs_consumed_before_preflight": False,
                **state,
            }
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            payload = {
                "version": VERSION,
                "time": utc_now(),
                "pid": os.getpid(),
                "stage": "input_or_transition_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "clinical_labels_synthesized": False,
                "attestations_synthesized": False,
                "controlled_method_comparison_authorized": False,
                "general_gpu_authorized": False,
                "paper_claim_authorized": False,
            }
        atomic_json(args.heartbeat, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        if args.once or payload.get("stage") in {
            "two_model_stage1_failed_terminal",
            "two_model_stage1_no_go_terminal",
        }:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
