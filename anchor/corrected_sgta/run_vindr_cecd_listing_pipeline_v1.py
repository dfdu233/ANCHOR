#!/usr/bin/env python3
"""Prepare or execute the hash-gated VinDr listing stage scheduler.

Preparation is CPU-only and launches nothing.  Execution is an explicit later
action, revalidates both admission roots before every stage, runs one model at
a time through the shared GPU lock, and requires a hash-complete two-model
stage gate before advancing pilot -> dev -> confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from corrected_sgta.prepare_vindr_reader_manifest import sha256_file
from corrected_sgta.run_vindr_cecd_listing_runtime_v1 import (
    DEFAULT_GPU_LOCK,
    canonical_json_sha256,
    write_once_json,
)
from corrected_sgta.validate_vindr_cecd_listing_scientific_admission_v1 import (
    file_record,
    validate_scientific_admission,
)


VERSION = "vindr-cecd-listing-detached-scheduler-handoff-v1"
STAGES = ("pilot", "dev", "confirmation")
MODELS = ("huatuo", "hulu")
ROOT = Path("/home/dbw/ANCHOR")


class SchedulerError(RuntimeError):
    pass


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    expected = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise SchedulerError(f"write-once scheduler collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(expected)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def prepare_scheduler_handoff(
    *, receipt: Path, expected_receipt_sha256: str,
    adjudication_handoff: Path, expected_adjudication_handoff_sha256: str,
    upstream_gate: Path, expected_upstream_gate_sha256: str,
    pack_dir: Path, experiment_manifest: Path, reference: Path,
    output_root: Path, handoff_path: Path,
) -> dict[str, Any]:
    validated = validate_scientific_admission(
        receipt_path=receipt,
        expected_receipt_sha256=expected_receipt_sha256,
        handoff_path=adjudication_handoff,
        expected_handoff_sha256=expected_adjudication_handoff_sha256,
        upstream_gate_path=upstream_gate,
        expected_upstream_gate_sha256=expected_upstream_gate_sha256,
        pack_manifest_path=pack_dir / "manifest.json",
        experiment_manifest_path=experiment_manifest,
    )
    experiment = json.loads(experiment_manifest.read_text(encoding="utf-8"))
    if sha256_file(reference) != experiment.get("reference_contract", {}).get("reference_file_sha256"):
        raise SchedulerError("listing reference does not match experiment manifest")
    plan = {
        "schema_version": VERSION,
        "status": "ready_for_explicit_detached_launch",
        "admission_receipt": file_record(receipt),
        "adjudication_handoff": file_record(adjudication_handoff),
        "upstream_binary_ce_input_gate": file_record(upstream_gate),
        "upstream_locked_confirmation": file_record(validated["upstream"]["confirmation_path"]),
        "pack_manifest": file_record(pack_dir / "manifest.json"),
        "experiment_manifest": file_record(experiment_manifest),
        "reference": file_record(reference),
        "output_root": str(output_root.resolve()),
        "shared_gpu_lock": str(DEFAULT_GPU_LOCK.resolve()),
        "models": list(MODELS),
        "stages": list(STAGES),
        "execution_order": [f"{stage}:{model}" for stage in STAGES for model in MODELS],
        "simultaneous_models_authorized": False,
        "stage_completion_hash_gate_required": True,
        "explicit_execute_flag_required": True,
        "model_or_gpu_launched_during_preparation": False,
    }
    plan["fingerprint"] = canonical_json_sha256(plan)
    _write_once(handoff_path, plan)
    return plan


def validate_scheduler_handoff(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise SchedulerError("scheduler handoff does not match externally pinned hash")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = payload.get("fingerprint")
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if (
        payload.get("schema_version") != VERSION
        or payload.get("status") != "ready_for_explicit_detached_launch"
        or payload.get("models") != list(MODELS)
        or payload.get("stages") != list(STAGES)
        or payload.get("execution_order") != [f"{stage}:{model}" for stage in STAGES for model in MODELS]
        or payload.get("shared_gpu_lock") != str(DEFAULT_GPU_LOCK.resolve())
        or payload.get("simultaneous_models_authorized") is not False
        or payload.get("explicit_execute_flag_required") is not True
        or payload.get("model_or_gpu_launched_during_preparation") is not False
        or fingerprint != canonical_json_sha256(body)
    ):
        raise SchedulerError("scheduler handoff contract drift")
    for name in (
        "admission_receipt", "adjudication_handoff", "upstream_binary_ce_input_gate",
        "upstream_locked_confirmation", "pack_manifest", "experiment_manifest", "reference",
    ):
        record = payload[name]
        path_value = Path(str(record.get("path", "")))
        if not path_value.is_file() or file_record(path_value) != record:
            raise SchedulerError(f"scheduler input hash drift: {name}")
    return payload


def verify_stage_completion(run_dir: Path, *, model: str, stage: str, admission_sha: str) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    completion_path = run_dir / "completion.json"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise SchedulerError(f"{stage}/{model}: completion artifacts absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    manifest_fp = manifest.get("fingerprint")
    completion_fp = completion.get("fingerprint")
    if (
        manifest.get("model_id") != model or manifest.get("split") != stage
        or manifest.get("admission_sha256") != admission_sha
        or manifest.get("gpu_lock") != str(DEFAULT_GPU_LOCK.resolve())
        or manifest_fp != canonical_json_sha256({k: v for k, v in manifest.items() if k != "fingerprint"})
        or completion.get("status") != "complete_eligible_orbits_only"
        or completion.get("scientific_model") is not True
        or completion.get("config_fingerprint") != manifest_fp
        or completion_fp != canonical_json_sha256({k: v for k, v in completion.items() if k != "fingerprint"})
    ):
        raise SchedulerError(f"{stage}/{model}: completion contract mismatch")
    inventory = completion.get("shard_inventory")
    if not isinstance(inventory, list) or len(inventory) != completion.get("cell_shards"):
        raise SchedulerError(f"{stage}/{model}: shard inventory malformed")
    for row in inventory:
        shard = run_dir / str(row.get("path", ""))
        if not shard.is_file() or sha256_file(shard) != row.get("sha256"):
            raise SchedulerError(f"{stage}/{model}: shard hash mismatch")
    return {
        "model": model, "stage": stage,
        "run_manifest": file_record(manifest_path),
        "completion": file_record(completion_path),
        "shard_inventory_fingerprint": canonical_json_sha256(inventory),
    }


def execute_scheduler(
    *, handoff_path: Path, expected_handoff_sha256: str,
    command_runner: Callable[[Sequence[str]], Any] | None = None,
) -> dict[str, Any]:
    plan = validate_scheduler_handoff(handoff_path, expected_handoff_sha256)
    receipt = Path(plan["admission_receipt"]["path"])
    adjudication = Path(plan["adjudication_handoff"]["path"])
    upstream = Path(plan["upstream_binary_ce_input_gate"]["path"])
    pack_dir = Path(plan["pack_manifest"]["path"]).parent
    experiment = Path(plan["experiment_manifest"]["path"])
    reference = Path(plan["reference"]["path"])
    def revalidate_roots() -> None:
        validate_scientific_admission(
            receipt_path=receipt,
            expected_receipt_sha256=plan["admission_receipt"]["sha256"],
            handoff_path=adjudication,
            expected_handoff_sha256=plan["adjudication_handoff"]["sha256"],
            upstream_gate_path=upstream,
            expected_upstream_gate_sha256=plan["upstream_binary_ce_input_gate"]["sha256"],
            pack_manifest_path=pack_dir / "manifest.json",
            experiment_manifest_path=experiment,
        )

    revalidate_roots()
    runner = command_runner or (lambda command: subprocess.run(command, check=True))
    output_root = Path(plan["output_root"])
    stage_gates = []
    for stage in STAGES:
        revalidate_roots()
        completed = []
        for model in MODELS:
            run_dir = output_root / model / stage
            manifest_present = (run_dir / "run_manifest.json").exists()
            completion_present = (run_dir / "completion.json").exists()
            if manifest_present != completion_present:
                raise SchedulerError(
                    f"{stage}/{model}: partial completion exists; audit required"
                )
            if manifest_present:
                row = verify_stage_completion(
                    run_dir, model=model, stage=stage,
                    admission_sha=plan["admission_receipt"]["sha256"],
                )
            else:
                # Revalidate immediately before every possible model/GPU launch.
                revalidate_roots()
                command = [
                    str(ROOT / ".venv-full/bin/python"), "-m",
                    "corrected_sgta.run_vindr_cecd_listing_runtime_v1", "run",
                    "--experiment-manifest", str(experiment), "--pack-dir", str(pack_dir),
                    "--admission", str(receipt), "--expected-admission-sha256",
                    plan["admission_receipt"]["sha256"], "--adjudication-handoff", str(adjudication),
                    "--expected-adjudication-handoff-sha256", plan["adjudication_handoff"]["sha256"],
                    "--upstream-binary-ce-gate", str(upstream),
                    "--expected-upstream-binary-ce-gate-sha256",
                    plan["upstream_binary_ce_input_gate"]["sha256"],
                    "--reference", str(reference), "--output-dir", str(run_dir),
                    "--model", model, "--split", stage, "--gpu-lock", str(DEFAULT_GPU_LOCK),
                ]
                runner(command)
                row = verify_stage_completion(
                    run_dir, model=model, stage=stage,
                    admission_sha=plan["admission_receipt"]["sha256"],
                )
            completed.append(row)
        gate = {
            "schema_version": "vindr-cecd-listing-two-model-stage-completion-v1",
            "stage": stage,
            "status": "two_models_hash_complete",
            "models": completed,
            "simultaneous_models_used": False,
        }
        gate["fingerprint"] = canonical_json_sha256(gate)
        gate_path = output_root / "stage_gates" / f"{stage}.json"
        write_once_json(gate_path, gate)
        stage_gates.append(file_record(gate_path))
    result = {
        "schema_version": "vindr-cecd-listing-scheduler-completion-v1",
        "status": "pilot_dev_confirmation_two_model_pipeline_complete",
        "scheduler_handoff_sha256": expected_handoff_sha256,
        "stage_gates": stage_gates,
        "simultaneous_models_used": False,
        "shared_gpu_lock": str(DEFAULT_GPU_LOCK.resolve()),
    }
    result["fingerprint"] = canonical_json_sha256(result)
    write_once_json(output_root / "scheduler_completion.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--receipt", type=Path, required=True)
    prepare.add_argument("--expected-receipt-sha256", required=True)
    prepare.add_argument("--adjudication-handoff", type=Path, required=True)
    prepare.add_argument("--expected-adjudication-handoff-sha256", required=True)
    prepare.add_argument("--upstream-gate", type=Path, required=True)
    prepare.add_argument("--expected-upstream-gate-sha256", required=True)
    prepare.add_argument("--pack-dir", type=Path, required=True)
    prepare.add_argument("--experiment-manifest", type=Path, required=True)
    prepare.add_argument("--reference", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--handoff", type=Path, required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("--handoff", type=Path, required=True)
    execute.add_argument("--expected-handoff-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_scheduler_handoff(
            receipt=args.receipt, expected_receipt_sha256=args.expected_receipt_sha256,
            adjudication_handoff=args.adjudication_handoff,
            expected_adjudication_handoff_sha256=args.expected_adjudication_handoff_sha256,
            upstream_gate=args.upstream_gate,
            expected_upstream_gate_sha256=args.expected_upstream_gate_sha256,
            pack_dir=args.pack_dir, experiment_manifest=args.experiment_manifest,
            reference=args.reference, output_root=args.output_root, handoff_path=args.handoff,
        )
    else:
        result = execute_scheduler(
            handoff_path=args.handoff, expected_handoff_sha256=args.expected_handoff_sha256
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
