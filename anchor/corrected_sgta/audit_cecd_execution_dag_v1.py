#!/usr/bin/env python3
"""Outcome-blind, CPU-only audit of the CECD execution handoff DAG.

This audit deliberately does not open human returns, admission decisions,
model outputs, Stage-1 analyses, or evaluation results.  It verifies only
static source/configuration infrastructure and detached-monitor liveness.
The purpose is operational: expose a missing handoff before independent human
returns arrive without weakening any scientific gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


VERSION = "cecd-outcome-blind-execution-dag-audit-v1"
ROOT = Path("/home/dbw/ANCHOR")
DATA_ROOT = Path("/home/dbw/datasets/physionet/vindr-cxr/1.0.0")
SHARED_GPU0_LOCK = ROOT / "corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock"
DUAL_DEFAULT_GPU0_LOCK = (
    ROOT / "corrected_runs/detached_jobs/locks/gpu0-cecd-dual-semantics-v1.lock"
)

# These are intentionally never opened by this audit.  Presence is also not a
# readiness criterion because doing so could let a partial or stale outcome
# silently influence an outcome-blind infrastructure check.
FORBIDDEN_OUTCOME_PATHS = (
    DATA_ROOT / "cecd_admission_returns_v3",
    DATA_ROOT / "cecd_listing_admission_returns_v1",
    ROOT / "corrected_runs/vindr_v2/cecd_human_admission_v2/analysis.json",
    ROOT / "corrected_runs/vindr_v2/cecd_three_stage_v3/confirmation_locked.json",
    ROOT / "corrected_runs/vindr_v2/cecd_dual_semantics_v1/method_outputs",
    ROOT / "corrected_runs/vindr_v2/cecd_listing_runtime_v1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def static_record(path: Path) -> dict[str, Any]:
    """Hash one non-outcome infrastructure file."""

    resolved = path.resolve()
    for forbidden in FORBIDDEN_OUTCOME_PATHS:
        blocked = forbidden.resolve()
        if resolved == blocked or blocked in resolved.parents:
            raise RuntimeError(f"outcome-blind audit attempted forbidden read: {resolved}")
    if not resolved.is_file() or resolved.is_symlink():
        return {"path": str(resolved), "present": False}
    return {
        "path": str(resolved),
        "present": True,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _load_static_json(path: Path) -> dict[str, Any]:
    record = static_record(path)
    if not record["present"]:
        raise RuntimeError(f"static JSON is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def detached_monitor(name: str, expected_command_fragment: str) -> dict[str, Any]:
    state_path = ROOT / f"corrected_runs/detached_jobs/{name}.json"
    if not state_path.is_file():
        return {
            "name": name,
            "state_path": str(state_path),
            "present": False,
            "alive": False,
        }
    # Detached state contains only process/command metadata, never scientific
    # outcomes or human-return contents.
    state = json.loads(state_path.read_text(encoding="utf-8"))
    command = state.get("command", [])
    command_text = " ".join(str(item) for item in command) if isinstance(command, list) else str(command)
    child = state.get("child_pid")
    return {
        "name": name,
        "state_path": str(state_path.resolve()),
        "present": True,
        "state_name_matches": state.get("name") == name,
        "status": state.get("status"),
        "child_pid": child,
        "alive": _pid_alive(child),
        "command_matches": expected_command_fragment in command_text,
    }


def _binding_status(record: Any, *, root: Path = ROOT) -> str:
    if record is None:
        return "missing"
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
        return "schema_drift"
    path = Path(str(record["path"]))
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    static = static_record(path)
    if not static["present"]:
        return "file_missing"
    if static["sha256"] != record["sha256"] or static["bytes"] != record["bytes"]:
        return "hash_or_size_drift"
    return "bound"


def audit(*, root: Path = ROOT, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    if root.resolve() != ROOT.resolve() or data_root.resolve() != DATA_ROOT.resolve():
        raise RuntimeError("v1 audit is intentionally pinned to the canonical repository and data root")

    static_paths = {
        "clinical_pack_manifest": data_root / "cecd_admission_pack_v2/manifest.json",
        "clinical_pack_sealed_mapping": data_root / "cecd_admission_pack_v2/sealed_mapping.json",
        "clinical_delivery_index": data_root / (
            "cecd_admission_pack_v2_reviewer_deliveries_v3/delivery_index.json"
        ),
        "clinical_monitor": root / "scripts/monitor_cecd_admission_pipeline.py",
        "three_stage_runner": root / "scripts/run_cecd_three_stage_v3.sh",
        "three_stage_verifier": root / "anchor/corrected_sgta/verify_cecd_three_stage_v3.py",
        "dual_transition_monitor": root / "scripts/monitor_cecd_dual_semantics_transition_v1.py",
        "dual_authorizer": root / "anchor/corrected_sgta/authorize_cecd_dual_semantics_preflight_v1.py",
        "dual_runner": root / "anchor/corrected_sgta/run_cecd_dual_semantics_controlled_v1.py",
        "dual_worker": root / "anchor/corrected_sgta/cecd_dual_semantics_worker_v1.py",
        "dual_shell": root / "scripts/run_cecd_dual_semantics_controlled_v1.sh",
        "listing_pack_manifest": data_root / "cecd_listing_admission_pack_v1/manifest.json",
        "listing_experiment_manifest": root / (
            "corrected_runs/vindr_v2/cecd_ontology_listing_substrate_v1/experiment_manifest.json"
        ),
        "listing_reference": root / (
            "corrected_runs/vindr_v2/cecd_ontology_listing_substrate_v1/reference_images.jsonl"
        ),
        "listing_return_monitor": root / "scripts/monitor_vindr_cecd_listing_returns_v1.py",
        "listing_runtime": root / "anchor/corrected_sgta/run_vindr_cecd_listing_runtime_v1.py",
        "halp_preflight": root / "anchor/corrected_sgta/halp_three_plane_preflight_v1.py",
        "system_pih_plan": root / "configs/cecd_system_pih_control_preflight_v1.json",
        "reader_alias_plan": root / "configs/reader_threshold_aliasing_preflight_v1.json",
    }
    static = {name: static_record(path) for name, path in static_paths.items()}

    clinical_monitor = detached_monitor(
        "cecd-clinical-admission-monitor-v4", "monitor_cecd_admission_pipeline.py"
    )
    dual_monitor = detached_monitor(
        "cecd-dual-semantics-transition-monitor-v2",
        "monitor_cecd_dual_semantics_transition_v1.py",
    )
    listing_monitor = detached_monitor(
        "vindr-cecd-listing-returns-v1", "monitor_vindr_cecd_listing_returns_v1.py"
    )

    system_plan = _load_static_json(static_paths["system_pih_plan"])
    system_integrations = system_plan.get("runtime_integrations", {})
    system_selection = system_plan.get("pih_selection", {})
    system_ready = all(
        isinstance(system_integrations.get(model), Mapping)
        and system_integrations[model].get("status") == "ready"
        and isinstance(system_selection.get(model), Mapping)
        and system_selection[model].get("status") == "ready"
        for model in ("huatuo", "hulu")
    )
    reader_plan = _load_static_json(static_paths["reader_alias_plan"])
    reader_bindings = {
        name: _binding_status(record)
        for name, record in reader_plan.get("bindings", {}).items()
    }

    dual_preflight = root / "configs/cecd_dual_semantics_preflight_v1.json"
    listing_admission_producer = (
        root / "anchor/corrected_sgta/analyze_vindr_cecd_listing_admission_v1.py"
    )
    listing_scheduler = root / "scripts/run_vindr_cecd_listing_pipeline_v1.sh"

    blockers: list[dict[str, str]] = []
    if any(not record["present"] for record in static.values()):
        blockers.append({
            "id": "STATIC_SUBSTRATE_MISSING",
            "severity": "fatal",
            "detail": "one or more required source/config/manifest files are absent",
        })
    for label, record in (
        ("clinical", clinical_monitor), ("dual_transition", dual_monitor),
        ("listing_returns", listing_monitor),
    ):
        if not (record.get("alive") and record.get("command_matches") and record.get("state_name_matches")):
            blockers.append({
                "id": f"{label.upper()}_MONITOR_NOT_LIVE",
                "severity": "fatal",
                "detail": "canonical detached monitor is absent, stale, or command-mismatched",
            })
    if not dual_preflight.is_file():
        blockers.append({
            "id": "DUAL_PREFLIGHT_PRODUCER_GAP",
            "severity": "fatal_after_stage1_go",
            "detail": (
                "transition monitor waits for a Stage-1-hash-bound preflight, but no canonical "
                "builder/command currently materializes it"
            ),
        })
    blockers.append({
        "id": "DUAL_RUNNER_TRIGGER_GAP",
        "severity": "fatal_after_authorization",
        "detail": (
            "transition monitor writes/revalidates authorization only; it never prepares or "
            "launches the formal CE runner"
        ),
    })
    if DUAL_DEFAULT_GPU0_LOCK.resolve() != SHARED_GPU0_LOCK.resolve():
        blockers.append({
            "id": "GPU0_LOCK_SPLIT_BRAIN",
            "severity": "fatal_before_any_dual_gpu_launch",
            "detail": (
                "three-stage/listing use gpu0-vindr-v2.lock but dual runner defaults to a "
                "different GPU0 lock; pass the shared lock explicitly"
            ),
        })
    if not listing_admission_producer.is_file():
        blockers.append({
            "id": "LISTING_ADJUDICATION_ADMISSION_GAP",
            "severity": "fatal_after_listing_returns",
            "detail": (
                "listing monitor stops after structural validation; no canonical producer "
                "creates the independently adjudicated, upstream-CE-bound admission receipt"
            ),
        })
    blockers.append({
        "id": "LISTING_RECEIPT_PROVENANCE_UNCLOSED",
        "severity": "fatal_before_listing_admission",
        "detail": (
            "the listing runtime receipt schema checks admission booleans but does not require "
            "hash-bound reviewer returns, attestations, adjudication records, or an admission "
            "analyzer source record"
        ),
    })
    blockers.append({
        "id": "LISTING_UPSTREAM_CE_HASH_UNVERIFIED",
        "severity": "fatal_before_listing_gpu_launch",
        "detail": (
            "the listing runtime accepts any nonzero 64-hex upstream CE hash and does not bind "
            "it to the canonical three-stage input gate/locked-confirmation file"
        ),
    })
    if not listing_scheduler.is_file():
        blockers.append({
            "id": "LISTING_RUNNER_TRIGGER_GAP",
            "severity": "fatal_after_listing_admission",
            "detail": "no canonical scheduler launches Huatuo/Hulu pilot->dev->confirmation listing runs",
        })

    result = {
        "version": VERSION,
        "status": "blocked_handoffs_outcome_blind" if blockers else "static_dag_ready",
        "passed": not blockers,
        "scope": {
            "cpu_only": True,
            "gpu_touched": False,
            "human_returns_opened": False,
            "admission_decisions_opened": False,
            "model_outputs_opened": False,
            "evaluation_results_opened": False,
        },
        "static_records": static,
        "monitors": {
            "clinical_admission": clinical_monitor,
            "dual_transition": dual_monitor,
            "listing_returns": listing_monitor,
        },
        "dag": [
            {
                "node": "clinical_admission_v4",
                "parents": ["four_independent_clinical_returns", "verified_clinical_pack"],
                "on_go": "cecd_three_stage_v3",
                "on_no_go": "terminal_mechanism_no_go",
                "automatic": True,
            },
            {
                "node": "cecd_three_stage_v3",
                "parents": ["clinical_admission_v4"],
                "on_go": "dual_semantics_preflight_bind",
                "on_no_go": "terminal_cecd_no_go",
                "automatic": True,
            },
            {
                "node": "dual_semantics_preflight_bind",
                "parents": ["cecd_three_stage_v3_go", "frozen_method_contract"],
                "on_go": "dual_semantics_authorization",
                "on_no_go": "blocked_no_method_run",
                "automatic": False,
            },
            {
                "node": "dual_semantics_formal_ce",
                "parents": ["dual_semantics_authorization"],
                "on_go": "blinded_method_analysis",
                "on_no_go": "failed_stop_manual_resume_after_audit",
                "automatic": False,
            },
            {
                "node": "listing_scientific_admission",
                "parents": [
                    "four_independent_listing_returns_and_adjudication",
                    "cecd_three_stage_v3_go_hash",
                ],
                "dependency_type": "convergent_AND_not_cycle",
                "on_go": "listing_model_runs",
                "on_no_go": "terminal_listing_no_go",
                "automatic": False,
            },
            {
                "node": "listing_model_runs",
                "parents": ["listing_scientific_admission"],
                "order": ["pilot", "dev", "confirmation_locked"],
                "models": ["huatuo", "hulu"],
                "automatic": False,
            },
        ],
        "controls": {
            "halp": {
                "status": "cpu_source_compatible_only",
                "probe_training_authorized": False,
            },
            "system_pih": {
                "status": "ready" if system_ready else "blocked_runtime_integration_and_dev_selection",
                "control_execution_ready": system_ready,
            },
            "reader_threshold_aliasing": {
                "status": "blocked_until_bound_stage_and_listing_inputs",
                "bindings": reader_bindings,
                "mainline_gate_modification_authorized": False,
            },
        },
        "gpu_lock_contract": {
            "canonical_gpu0_lock": str(SHARED_GPU0_LOCK.resolve()),
            "three_stage": str(SHARED_GPU0_LOCK.resolve()),
            "listing": str(SHARED_GPU0_LOCK.resolve()),
            "dual_default": str(DUAL_DEFAULT_GPU0_LOCK.resolve()),
            "dual_required_override": str(SHARED_GPU0_LOCK.resolve()),
            "single_lock_closed": DUAL_DEFAULT_GPU0_LOCK.resolve() == SHARED_GPU0_LOCK.resolve(),
        },
        "terminology": {
            "system_command_permission": "already_unrestricted_and_not_part_of_this_DAG",
            "scientific_admission": "preregistered evidence validity gate; never an OS permission prompt",
            "general_gpu_authorized_false": (
                "scope guard against unrelated GPU experiments; it does not deny execution of the "
                "named hash-bound controlled comparison"
            ),
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
