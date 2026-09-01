#!/usr/bin/env python3
"""Outcome-blind audit of the repaired CECD execution DAG.

Version 2 preserves the historical v1 audit and checks operational closure
without treating absent future human/scientific inputs as infrastructure
failures.  It never opens reviewer returns, admissions, model outputs, or
evaluation results.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from anchor.corrected_sgta.audit_cecd_execution_dag_v1 import (
    DATA_ROOT,
    FORBIDDEN_OUTCOME_PATHS,
    ROOT,
    static_record,
)


VERSION = "cecd-outcome-blind-execution-dag-audit-v2-repaired-handoffs"
ACTIVE_JOBS = ROOT / "configs/research_active_jobs.json"
SHARED_GPU_LOCK = ROOT / "corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock"


def _stable_static_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read one allowed static file without accepting a concurrent rewrite."""

    before = static_record(path)
    if not before.get("present"):
        raise RuntimeError(f"static file is missing: {path}")
    raw = path.read_bytes()
    after = static_record(path)
    if (
        before != after
        or hashlib.sha256(raw).hexdigest() != after.get("sha256")
        or len(raw) != after.get("bytes")
    ):
        raise RuntimeError(f"static file changed while it was read: {path}")
    return raw, after


def _load_static_json(path: Path) -> dict[str, Any]:
    raw, _ = _stable_static_bytes(path)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"static JSON must be an object: {path}")
    return payload


def _pid_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _active_job(
    prefix: str, *, registry: Path = ACTIVE_JOBS,
    registry_payload: Mapping[str, Any] | None = None,
) -> str | None:
    payload = dict(registry_payload) if registry_payload is not None else _load_static_json(registry)
    jobs = payload.get("active_jobs")
    if not isinstance(jobs, list):
        raise RuntimeError("active-job registry lacks an active_jobs list")
    pattern = re.compile(rf"^{re.escape(prefix)}v[1-9][0-9]*$")
    matches = [str(name) for name in jobs if pattern.fullmatch(str(name))]
    if len(matches) != 1:
        return None
    return matches[0]


def monitor_record(
    prefix: str,
    command_fragment: str,
    source: Path,
    *,
    registry: Path = ACTIVE_JOBS,
    registry_payload: Mapping[str, Any] | None = None,
    allow_completed: bool = False,
) -> dict[str, Any]:
    name = _active_job(
        prefix, registry=registry, registry_payload=registry_payload
    )
    if name is None:
        return {"prefix": prefix, "registered_name": None, "ready": False}
    state_path = ROOT / f"corrected_runs/detached_jobs/{name}.json"
    if not state_path.is_file():
        return {
            "prefix": prefix,
            "registered_name": name,
            "state_present": False,
            "ready": False,
        }
    state = _load_static_json(state_path)
    command = state.get("command")
    command_text = " ".join(map(str, command)) if isinstance(command, list) else str(command)
    started_raw = state.get("started_at")
    try:
        started = datetime.fromisoformat(str(started_raw).replace("Z", "+00:00")).timestamp()
    except ValueError:
        started = 0.0
    source_record = static_record(source)
    source_not_newer = (
        bool(source_record.get("present")) and source.stat().st_mtime <= started
    )
    child_pid = state.get("child_pid")
    process = _process_record(child_pid, source) if state.get("status") == "running" else {
        "inspected": False,
        "reason": "state_not_running",
        "identity_matches": False,
    }
    completed_ok = bool(
        allow_completed
        and state.get("status") == "done"
        and state.get("exit_code") == 0
        and source_not_newer
    )
    checks = {
        "state_name_matches": state.get("name") == name,
        "running_or_allowed_completed": state.get("status") == "running" or completed_ok,
        "live_process_identity_matches": (
            process.get("identity_matches") is True if state.get("status") == "running"
            else completed_ok
        ),
        "command_matches": command_fragment in command_text,
        "source_loaded_after_last_edit": source_not_newer,
        "state_cwd_is_canonical": Path(str(state.get("cwd", ""))).resolve() == ROOT.resolve(),
        "state_start_matches_live_process": (
            abs(float(process.get("started_at_epoch")) - started) <= 5.0
            if state.get("status") == "running"
            and isinstance(process.get("started_at_epoch"), (int, float))
            else completed_ok
        ),
    }
    checks["state_cwd_is_canonical"] = bool(
        isinstance(state.get("cwd"), str)
        and state["cwd"]
        and Path(state["cwd"]).is_absolute()
        and Path(state["cwd"]).resolve() == ROOT.resolve()
    )
    return {
        "prefix": prefix,
        "registered_name": name,
        "state": static_record(state_path),
        "source": source_record,
        "child_pid": child_pid,
        "process": process,
        "completed_monitor_accepted": completed_ok,
        "checks": checks,
        "ready": all(checks.values()),
    }


def _proc_start_time(pid: int) -> float:
    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    # The comm field is parenthesized and may itself contain spaces.
    fields_after_comm = stat_text[stat_text.rfind(")") + 2 :].split()
    start_ticks = int(fields_after_comm[19])  # field 22; suffix begins at field 3
    clock_ticks = int(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    boot_time = next(
        int(line.split()[1])
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines()
        if line.startswith("btime ")
    )
    return boot_time + start_ticks / clock_ticks


def _process_record(pid: Any, source: Path) -> dict[str, Any]:
    if not _pid_alive(pid):
        return {"inspected": True, "alive": False, "identity_matches": False}
    try:
        proc = Path(f"/proc/{pid}")
        argv = [
            item.decode("utf-8", errors="replace")
            for item in (proc / "cmdline").read_bytes().split(b"\0")
            if item
        ]
        cwd = Path(os.readlink(proc / "cwd")).resolve()
        source_resolved = source.resolve()
        source_argv_match = any(
            ((cwd / item).resolve() if not Path(item).is_absolute() else Path(item).resolve())
            == source_resolved
            for item in argv
        )
        started = _proc_start_time(int(pid))
        identity = bool(
            cwd == ROOT.resolve()
            and source_argv_match
            and source.stat().st_mtime <= started
        )
        return {
            "inspected": True,
            "alive": True,
            "argv": argv,
            "cwd": str(cwd),
            "started_at_epoch": started,
            "source_argv_match": source_argv_match,
            "identity_matches": identity,
        }
    except (OSError, ValueError, IndexError, StopIteration):
        return {"inspected": True, "alive": True, "identity_matches": False}


def _call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        if isinstance(item.func, ast.Name):
            names.add(item.func.id)
        elif isinstance(item.func, ast.Attribute):
            names.add(item.func.attr)
    return names


def _source_contract(
    path: Path,
    markers: tuple[str, ...] = (),
    *,
    required_calls: Mapping[str, Sequence[str]] | None = None,
    required_dict_literals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        raw, record = _stable_static_bytes(path)
    except RuntimeError:
        record = static_record(path)
        return {"record": record, "markers": {}, "ready": False}
    text = raw.decode("utf-8")
    found = {marker: marker in text for marker in markers}
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return {
            "record": record, "markers": found, "semantic_calls": {},
            "syntax_valid": False, "ready": False,
        }
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    semantic: dict[str, dict[str, bool]] = {}
    for function, calls in (required_calls or {}).items():
        actual = _call_names(functions[function]) if function in functions else set()
        semantic[function] = {call: call in actual for call in calls}
    semantic_ready = all(
        checks and all(checks.values()) for checks in semantic.values()
    ) if semantic else True
    literal_checks: dict[str, bool] = {}
    for key, expected in (required_dict_literals or {}).items():
        literal_checks[key] = any(
            isinstance(item, ast.Dict)
            and any(
                isinstance(dict_key, ast.Constant)
                and dict_key.value == key
                and isinstance(dict_value, ast.Constant)
                and type(dict_value.value) is type(expected)
                and dict_value.value == expected
                for dict_key, dict_value in zip(item.keys, item.values)
            )
            for item in ast.walk(tree)
        )
    return {
        "record": record,
        "markers": found,
        "semantic_calls": semantic,
        "required_dict_literals": literal_checks,
        "syntax_valid": True,
        "ready": all(found.values()) and semantic_ready and all(literal_checks.values()),
    }


def _shell_gpu_lock_contract(path: Path) -> dict[str, Any]:
    try:
        raw, record = _stable_static_bytes(path)
    except RuntimeError:
        record = static_record(path)
        return {"record": record, "ready": False}
    executable = [
        line.strip() for line in raw.decode("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    lock_assign = any(
        re.fullmatch(
            r'lock="?/home/dbw/ANCHOR/corrected_runs/detached_jobs/locks/'
            r'gpu0-vindr-v2\.lock"?', line,
        )
        for line in executable
    )
    lock_open = any(
        re.fullmatch(
            r'exec\s+8>"?(?:/home/dbw/ANCHOR/)?corrected_runs/detached_jobs/locks/'
            r'gpu0-vindr-v2\.lock"?', line,
        )
        for line in executable
    ) or (lock_assign and any(re.fullmatch(r'exec\s+8>"?\$lock"?', line) for line in executable))
    flock_same_fd = any(re.fullmatch(r"flock(?:\s+-n)?\s+8", line) for line in executable)
    return {
        "record": record,
        "checks": {"opens_canonical_lock_on_fd8": lock_open, "flocks_fd8": flock_same_fd},
        "ready": lock_open and flock_same_fd,
    }


def _assignment_contains_literal(path: Path, name: str, literal: str) -> bool:
    try:
        raw, _ = _stable_static_bytes(path)
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    except (RuntimeError, SyntaxError, UnicodeDecodeError):
        return False
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        return any(
            isinstance(item, ast.Constant) and item.value == literal
            for item in ast.walk(node.value)
        )
    return False


def audit(*, root: Path = ROOT, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    if root.resolve() != ROOT.resolve() or data_root.resolve() != DATA_ROOT.resolve():
        raise RuntimeError("v2 audit is pinned to the canonical repository and data root")

    sources = {
        "clinical_transition": _source_contract(
            root / "scripts/monitor_cecd_admission_pipeline.py",
            ("freeze_human_bundle", "launch_or_monitor_stage", "retry_authorized"),
            required_calls={
                "analyze_admission": (
                    "freeze_human_bundle", "validate_pack_closure", "validate_all",
                    "require_cecd_authorization", "launch_or_monitor_stage",
                ),
                "launch_or_monitor_stage": ("run", "validate_stage_result"),
                "advance": ("delivery_ready", "analyze_admission"),
            },
        ),
        "dual_builder": _source_contract(
            root / "anchor/corrected_sgta/build_cecd_dual_semantics_preflight_v1.py",
            ("BUILD_RECEIPT_SCHEMA", "CANONICAL_GPU_LOCK_RELATIVE", "EXPECTED_SELECTION_HASHES"),
            required_calls={
                "build_preflight": (
                    "_validate_stage_state", "_validate_gate", "validate_stage_result",
                    "validate_dual_semantics_preflight_contract", "validate_build_receipt",
                    "write_once_json", "write_once_jsonl",
                ),
                "validate_build_receipt": ("file_record", "canonical_sha256"),
            },
        ),
        "dual_authorizer": _source_contract(
            root / "anchor/corrected_sgta/authorize_cecd_dual_semantics_preflight_v1.py",
            ("preflight_build", "controlled_dual_semantics_comparison_authorized"),
            required_calls={
                "authorize": (
                    "validate_build_receipt", "validate_stage_result",
                    "validate_dual_semantics_preflight_contract", "_write_once_or_equal",
                )
            },
        ),
        "dual_transition": _source_contract(
            root / "scripts/monitor_cecd_dual_semantics_transition_v1.py",
            ("build_preflight", "emit_runner_handoff", "gpu_launched_by_monitor"),
            required_calls={
                "advance": (
                    "validate_stage_result", "build_preflight", "authorize",
                    "emit_runner_handoff",
                ),
                "emit_runner_handoff": ("validate_build_receipt", "write_once_json"),
            },
            required_dict_literals={
                "gpu_launched_by_monitor": False,
                "launched_by_transition_monitor": False,
            },
        ),
        "dual_runner": _source_contract(
            root / "anchor/corrected_sgta/run_cecd_dual_semantics_controlled_v1.py",
            ("GPU_LOCK_RELATIVE", "execute_ce_only", "preflight_build"),
            required_calls={
                "prepare_or_run": (
                    "validate_authorization_and_preflight", "build_candidate_contract",
                    "resolve",
                ),
                "execute_formal_ce_stage": (
                    "exclusive_lock", "validate_ce_stage_completion",
                    "validate_shared_raw_cache_manifest",
                ),
            },
        ),
        "listing_handoff": _source_contract(
            root / "anchor/corrected_sgta/prepare_vindr_cecd_listing_adjudication_handoff_v1.py",
            ("adjudicator.attestation.template.json", "admission_receipt_created"),
            required_calls={
                "prepare_handoff": (
                    "validate_all", "verify", "_expected_source_records",
                    "_adjudication_template", "file_record",
                )
            },
            required_dict_literals={"admission_receipt_created": False},
        ),
        "listing_assembler": _source_contract(
            root / "anchor/corrected_sgta/analyze_vindr_cecd_listing_admission_v1.py",
            ("admission_assembler_source", "human_adjudication_rejected_terminal"),
            required_calls={
                "assemble_receipt": (
                    "validate_human_evidence", "validate_admit_eligibility",
                    "validate_upstream_binary_ce", "evidence_records", "_write_once",
                )
            },
        ),
        "listing_validator": _source_contract(
            root / "anchor/corrected_sgta/validate_vindr_cecd_listing_scientific_admission_v1.py",
            ("UPSTREAM_GATE_RELATIVE", "admission_validator_source", "admission_assembler_source"),
            required_calls={
                "validate_scientific_admission": (
                    "validate_human_evidence", "validate_admit_eligibility",
                    "validate_upstream_binary_ce", "validate_file_record",
                ),
                "validate_upstream_binary_ce": ("validate_file_record", "sha256_file"),
            },
        ),
        "listing_scheduler": _source_contract(
            root / "anchor/corrected_sgta/run_vindr_cecd_listing_pipeline_v1.py",
            ("explicit_execute_flag_required", "verify_stage_completion", "DEFAULT_GPU_LOCK"),
            required_calls={
                "prepare_scheduler_handoff": ("validate_scientific_admission", "_write_once"),
                "execute_scheduler": (
                    "validate_scheduler_handoff", "validate_scientific_admission",
                    "verify_stage_completion", "run",
                ),
            },
            required_dict_literals={
                "explicit_execute_flag_required": True,
                "model_or_gpu_launched_during_preparation": False,
            },
        ),
        "listing_runtime": _source_contract(
            root / "anchor/corrected_sgta/run_vindr_cecd_listing_runtime_v1.py",
            (
                "validate_scientific_admission", "DEFAULT_GPU_LOCK", "completion.json",
                "_require_canonical_gpu_lock",
            ),
            required_calls={
                "run_runtime": (
                    "_require_canonical_gpu_lock", "validate_admission_gate",
                    "lock_factory", "NativeListingAdapter",
                ),
                "gpu_flock": ("flock",),
            },
        ),
        "three_stage_gpu_shell": _shell_gpu_lock_contract(
            root / "scripts/run_cecd_three_stage_v3.sh"
        ),
        "system_pih_gpu_shell": _shell_gpu_lock_contract(
            root / "scripts/run_cecd_system_pih_native_eager_canaries_v1.sh"
        ),
    }

    registry_payload = _load_static_json(ACTIVE_JOBS)
    registry_before = static_record(ACTIVE_JOBS)
    monitors = {
        "clinical_transition": monitor_record(
            "cecd-clinical-admission-monitor-",
            "monitor_cecd_admission_pipeline.py",
            root / "scripts/monitor_cecd_admission_pipeline.py",
            registry_payload=registry_payload,
            allow_completed=True,
        ),
        "dual_transition": monitor_record(
            "cecd-dual-semantics-transition-monitor-",
            "monitor_cecd_dual_semantics_transition_v1.py",
            root / "scripts/monitor_cecd_dual_semantics_transition_v1.py",
            registry_payload=registry_payload,
        ),
        "listing_returns": monitor_record(
            "vindr-cecd-listing-returns-",
            "monitor_vindr_cecd_listing_returns_v1.py",
            root / "scripts/monitor_vindr_cecd_listing_returns_v1.py",
            registry_payload=registry_payload,
            allow_completed=True,
        ),
    }
    registry_after = static_record(ACTIVE_JOBS)

    blockers: list[dict[str, str]] = []
    for name, contract in sources.items():
        if not contract["ready"]:
            blockers.append({
                "id": f"{name.upper()}_STATIC_CONTRACT_OPEN",
                "severity": "fatal_before_transition",
                "detail": "required source or fail-closed marker is absent",
            })
    for name, monitor in monitors.items():
        if not monitor["ready"]:
            blockers.append({
                "id": f"{name.upper()}_MONITOR_STALE_OR_MISSING",
                "severity": "fatal_before_future_input_arrives",
                "detail": "registered detached monitor is absent, dead, mismatched, or older than its source",
            })
    if registry_before != registry_after:
        blockers.append({
            "id": "ACTIVE_JOB_REGISTRY_CHANGED_DURING_AUDIT",
            "severity": "retry_static_audit",
            "detail": "active-job registry changed while monitor identity was audited",
        })

    relative_lock = "corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock"
    absolute_lock = str(SHARED_GPU_LOCK)
    lock_checks = {
        "three_stage_shell": sources["three_stage_gpu_shell"]["ready"],
        "dual_builder_exact_constant": _assignment_contains_literal(
            root / "anchor/corrected_sgta/build_cecd_dual_semantics_preflight_v1.py",
            "CANONICAL_GPU_LOCK_RELATIVE", relative_lock,
        ),
        "dual_runner_exact_constant_and_use": (
            _assignment_contains_literal(
                root / "anchor/corrected_sgta/run_cecd_dual_semantics_controlled_v1.py",
                "GPU_LOCK_RELATIVE", relative_lock,
            )
            and sources["dual_runner"]["semantic_calls"]["prepare_or_run"]["resolve"]
            and sources["dual_runner"]["semantic_calls"]["execute_formal_ce_stage"]["exclusive_lock"]
        ),
        "listing_runtime_exact_constant_and_fail_closed_use": (
            _assignment_contains_literal(
                root / "anchor/corrected_sgta/run_vindr_cecd_listing_runtime_v1.py",
                "DEFAULT_GPU_LOCK", absolute_lock,
            )
            and sources["listing_runtime"]["semantic_calls"]["run_runtime"]["_require_canonical_gpu_lock"]
            and sources["listing_runtime"]["semantic_calls"]["gpu_flock"]["flock"]
        ),
        "listing_scheduler_uses_runtime_lock": sources["listing_scheduler"]["ready"],
        "system_pih_shell": sources["system_pih_gpu_shell"]["ready"],
    }
    if not all(lock_checks.values()):
        blockers.append({
            "id": "GPU0_LOCK_CLOSURE_OPEN",
            "severity": "fatal_before_gpu_launch",
            "detail": "one or more formal runners are not bound to the shared VinDr GPU0 lock",
        })

    future_inputs = {
        "clinical_human_returns": data_root / "cecd_admission_returns_v3",
        "listing_human_returns": data_root / "cecd_listing_admission_returns_v1",
        "listing_human_adjudication": root / (
            "corrected_runs/vindr_v2/cecd_listing_admission_returns_v1/"
            "human_adjudication_handoff_v1"
        ),
    }
    waiting = {
        name: {
            "path": str(path.resolve()),
            "presence": "deliberately_not_inspected_outcome_blind",
        }
        for name, path in future_inputs.items()
    }
    generated_artifacts = {
        "clinical_human_admission": {
            "path": str((root / "corrected_runs/vindr_v2/cecd_human_admission_v2/analysis.json").resolve()),
            "classification": "generated_after_genuine_human_returns",
            "presence": "deliberately_not_inspected_outcome_blind",
        },
        "three_stage_input_gate": {
            "path": str((root / "corrected_runs/vindr_v2/cecd_three_stage_v3/input_gate.json").resolve()),
            "classification": "generated_scientific_gate_not_external_input",
            "presence": "deliberately_not_inspected_outcome_blind",
        },
        "listing_human_admission": {
            "path": str((root / "corrected_runs/vindr_v2/cecd_listing_admission_v1/admission.json").resolve()),
            "classification": "generated_after_genuine_human_adjudication",
            "presence": "deliberately_not_inspected_outcome_blind",
        },
        "system_pih_canaries": {
            "path": str((root / "corrected_runs/vindr_v2/system_pih_control_v1").resolve()),
            "classification": "pending_non_mainline_control_runtime_artifact",
            "presence": "deliberately_not_inspected_outcome_blind",
        },
    }

    result: dict[str, Any] = {
        "version": VERSION,
        "status": (
            "static_handoffs_ready_waiting_genuine_inputs"
            if not blockers else "static_handoffs_blocked"
        ),
        "passed": not blockers,
        "scope": {
            "cpu_only": True,
            "gpu_touched": False,
            "human_returns_opened": False,
            "admission_decisions_opened": False,
            "model_outputs_opened": False,
            "evaluation_results_opened": False,
            "automatic_gpu_execution_claimed": False,
        },
        "assurance_boundary": {
            "marker_strings_alone_are_sufficient": False,
            "state_json_and_pid_liveness_alone_are_sufficient": False,
            "live_process_cmdline_cwd_and_start_time_checked": True,
            "static_call_structure_checked": True,
            "scientific_or_human_gate_readiness_claimed": False,
            "zero_blockers_means_static_handoff_closure_only": True,
        },
        "sources": sources,
        "monitors": monitors,
        "active_job_registry": registry_after,
        "gpu_lock_contract": {
            "canonical": str(SHARED_GPU_LOCK.resolve()),
            "checks": lock_checks,
            "closed": all(lock_checks.values()),
        },
        "waiting_not_blockers": waiting,
        "generated_artifacts_not_misclassified_as_external_inputs": generated_artifacts,
        "scientific_boundaries": {
            "future_input_absence_is_not_an_operational_gap": True,
            "human_decisions_synthesized": False,
            "model_execution_triggered": False,
            "listing_requires_human_AND_upstream_ce_go": True,
            "dual_handoff_is_ce_only": True,
            "oe_hidden_and_treble_variants_authorized": False,
        },
        "blockers": blockers,
    }
    result["fingerprint"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = audit()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(encoded, end="")
    if args.strict and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
