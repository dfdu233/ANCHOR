#!/usr/bin/env python3
"""Persistently advance CECD from four real blinded returns to Stage 1.

The monitor never creates or repairs a review decision or attestation. Human
files must be byte-stable across two polls, pass the frozen v3 validator, and
be copied into hash-named immutable files before the sealed mapping is opened.
A valid analyzed admission failure or failed scientific job is terminal and is
never retried. Recoverable input/transition errors remain in a CPU-only polling
state and cannot authorize or launch model scoring.
"""

from __future__ import annotations

import argparse
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

from anchor.corrected_sgta.analyze_clinical_equivalence_composition_defect_v1 import (
    CONFIRMATION_VERSION,
)
from anchor.corrected_sgta.cecd_admission_gate import (
    EXPECTED_VERSION as EXPECTED_ADMISSION_ANALYSIS,
    require_cecd_authorization,
)
from anchor.corrected_sgta.verify_cecd_three_stage_v3 import (
    VERSION as THREE_STAGE_GATE_VERSION,
)
from anchor.medeval.hashing import sha256_file
from anchor.medeval.store import atomic_write_json
from anchor.medeval.validate_cecd_returns_v3 import VERSION as VALIDATION_VERSION
from anchor.medeval.validate_cecd_returns_v3 import validate_all


VERSION = "cecd-clinical-admission-pipeline-monitor-v4-three-stage"
ROOT = Path("/home/dbw/ANCHOR")
DEFAULT_PACK = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2"
)
DEFAULT_DELIVERY = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/"
    "cecd_admission_pack_v2_reviewer_deliveries_v3"
)
DEFAULT_INBOX = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_returns_v3"
)
DEFAULT_OUTPUT = ROOT / "corrected_runs/vindr_v2/cecd_human_admission_v2"
STAGE_JOB = "cecd-three-stage-v3"
EXPECTED_DELIVERY_VERIFICATION = "cecd-reviewer-delivery-verification-v3.1"
EXPECTED_BROWSER_SMOKE = "cecd-v3-offline-browser-smoke-v1"
EXPECTED_STAGE_GATE = THREE_STAGE_GATE_VERSION
EXPECTED_STAGE_ANALYSIS = CONFIRMATION_VERSION
ROLES = (
    "clinical_reviewer_1",
    "clinical_reviewer_2",
    "clinical_template_reviewer",
    "language_reviewer",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_inbox_instructions(inbox: Path) -> None:
    target = inbox / "RETURN_FILES.md"
    if target.exists():
        return
    atomic_text(
        target,
        """# CECD blinded human-return inbox

Copy under temporary names, then atomically rename only complete files. The
monitor requires unchanged size and SHA-256 over two polls and never creates a
clinical/language decision or signature.

Return exactly these files from four distinct independent reviewers:

1. `clinical_reviewer_1.completed.csv`
2. `clinical_reviewer_1.attestation.json`
3. `clinical_reviewer_2.completed.csv`
4. `clinical_reviewer_2.attestation.json`
5. `clinical_template_reviewer.completed.csv`
6. `clinical_template_reviewer.attestation.json`
7. `language_annotator.completed.csv`
8. `language_reviewer.attestation.json`

Do not include the sealed mapping, source identities, transform names, reader
votes, model outputs, benchmark answers, or coordinator-created attestations.
""",
    )


def return_paths(inbox: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    completed = {
        "clinical_reviewer_1": inbox / "clinical_reviewer_1.completed.csv",
        "clinical_reviewer_2": inbox / "clinical_reviewer_2.completed.csv",
        "clinical_template_reviewer": inbox / "clinical_template_reviewer.completed.csv",
        "language_reviewer": inbox / "language_annotator.completed.csv",
    }
    attestations = {
        role: inbox / f"{role}.attestation.json" for role in ROLES
    }
    return completed, attestations


def signatures(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        path.name: {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
        if path.is_file()
    }


def record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


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


def write_once_or_equal(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"write-once collision: {path}")
        return
    atomic_text(path, rendered)


def validate_pack_closure(pack: Path, output: Path) -> dict[str, Any]:
    """Freeze the exact blinded-pack substrate before opening its mapping."""

    integrity_path = output / "pack_integrity.json"
    if not integrity_path.is_file():
        raise RuntimeError("CECD pack integrity artifact is missing")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if (
        integrity.get("protocol_version") != "cecd-admission-pack-integrity-v1"
        or integrity.get("passed") is not True
        or integrity.get("review_sheets_blank") is not True
        or integrity.get("reviewer_visible_leakage_checks_passed") is not True
        or Path(str(integrity.get("pack", ""))).resolve() != pack.resolve()
        or integrity.get("manifest_sha256") != sha256_file(pack / "manifest.json")
        or integrity.get("sealed_mapping_sha256")
        != sha256_file(pack / "sealed_mapping.json")
    ):
        raise RuntimeError("CECD source-pack integrity contract mismatch")
    critical_names = (
        "REVIEW_INSTRUCTIONS.md",
        "clinical_reviewer_1.csv",
        "clinical_reviewer_2.csv",
        "clinical_template_reviewer.csv",
        "language_annotator.csv",
        "manifest.json",
        "sealed_mapping.json",
        "selected_claims.sealed.jsonl",
    )
    missing = [name for name in critical_names if not (pack / name).is_file()]
    if missing:
        raise RuntimeError(f"CECD source pack lacks critical files: {missing}")
    closure = {
        "version": "cecd-admission-source-pack-lock-v1",
        "status": "frozen_before_sealed_mapping_open",
        "pack": str(pack.resolve()),
        "critical_files": {name: record(pack / name) for name in critical_names},
        "integrity_artifact": record(integrity_path),
    }
    write_once_or_equal(output / "pack_source_lock.json", closure)
    return closure


def freeze_human_bundle(
    *,
    output: Path,
    completed: dict[str, Path],
    attestations: dict[str, Path],
) -> tuple[dict[str, Path], dict[str, Path], Path]:
    """Freeze the first complete eight-file bundle; later byte drift is terminal."""

    frozen_dir = output / "frozen"
    frozen_completed = {
        role: freeze_copy(path, frozen_dir, f"{role}.completed")
        for role, path in completed.items()
    }
    frozen_attestations = {
        role: freeze_copy(path, frozen_dir, f"{role}.attestation")
        for role, path in attestations.items()
    }
    bundle = {
        "version": "cecd-eight-file-human-return-lock-v1",
        "status": "all_roles_and_attestations_frozen_before_unblinding",
        "completed_files": {
            role: record(path) for role, path in frozen_completed.items()
        },
        "attestation_files": {
            role: record(path) for role, path in frozen_attestations.items()
        },
    }
    lock_path = output / "human_return_bundle_lock.json"
    write_once_or_equal(lock_path, bundle)
    return frozen_completed, frozen_attestations, lock_path


def delivery_ready(delivery: Path) -> dict[str, Any]:
    index = delivery / "delivery_index.json"
    verification = delivery / "verification.json"
    browser = delivery / "browser_smoke.json"
    missing = [str(path) for path in (index, verification, browser) if not path.is_file()]
    if missing:
        return {"ready": False, "missing": missing}
    verified = json.loads(verification.read_text(encoding="utf-8"))
    smoke = json.loads(browser.read_text(encoding="utf-8"))
    index_hash = sha256_file(index)
    verification_hash = sha256_file(verification)
    if (
        verified.get("version") != EXPECTED_DELIVERY_VERIFICATION
        or smoke.get("version") != EXPECTED_BROWSER_SMOKE
        or verified.get("delivery_index_sha256") != index_hash
        or smoke.get("delivery_index_sha256") != index_hash
        or smoke.get("verification_sha256") != verification_hash
        or verified.get("passed") is not True
        or smoke.get("passed") is not True
    ):
        raise RuntimeError("CECD reviewer delivery failed verifier or browser smoke")
    return {
        "ready": True,
        "verification": record(verification),
        "browser_smoke": record(browser),
    }


def launch_or_monitor_stage(analysis: Path) -> dict[str, Any]:
    state_path = ROOT / f"corrected_runs/detached_jobs/{STAGE_JOB}.json"
    log_path = ROOT / f"corrected_runs/detached_jobs/{STAGE_JOB}.log"
    if not state_path.exists():
        command = [
            sys.executable,
            "scripts/start_detached_job.py",
            "--name",
            STAGE_JOB,
            "--log",
            str(log_path),
            "--state",
            str(state_path),
            "--",
            "env",
            f"CECD_ADMISSION_RESULT={analysis.resolve()}",
            "bash",
            "scripts/run_cecd_three_stage_v3.sh",
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f"cannot launch CECD Stage 1: {result.stderr[-2000:]}")
        return {"stage": "two_model_stage1_launched", "stage_job": STAGE_JOB}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("name") != STAGE_JOB:
        raise RuntimeError("CECD detached Stage 1 state name mismatch")
    status = state.get("status")
    if status in {"starting", "running"}:
        return {
            "stage": "two_model_stage1_running",
            "stage_job": STAGE_JOB,
            "stage_child_pid": state.get("child_pid"),
        }
    if status == "failed":
        return {
            "stage": "two_model_stage1_failed_terminal",
            "stage_job": STAGE_JOB,
            "exit_code": state.get("exit_code"),
            "retry_authorized": False,
        }
    result_path = ROOT / "corrected_runs/vindr_v2/cecd_three_stage_v3/confirmation_locked.json"
    if status != "done" or not result_path.is_file():
        raise RuntimeError(f"unexpected CECD Stage 1 state: {status!r}")
    result = validate_stage_result(result_path=result_path, admission=analysis)
    method_authorized = result["gate"][
        "authorized_for_method_level_treble_adapter_run"
    ]
    return {
        "stage": "two_model_stage1_complete",
        "stage_job": STAGE_JOB,
        "analysis": str(result_path.resolve()),
        "analysis_sha256": sha256_file(result_path),
        "behavioral_gate_passed": method_authorized,
        "method_level_treble_authorized": method_authorized,
        "hidden_state_authorized": False,
        "next": (
            "resolve the frozen official-Treble collision gate; do not use a scalar surrogate"
            if method_authorized
            else "terminate CECD at the preregistered behavioral gate"
        ),
    }


def validate_stage_result(*, result_path: Path, admission: Path) -> dict[str, Any]:
    """Bind the locked confirmation to the v3 three-stage verifier."""

    gate_path = ROOT / "corrected_runs/vindr_v2/cecd_three_stage_v3/input_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("CECD Stage 1 input gate is missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("version") != EXPECTED_STAGE_GATE
        or gate.get("status") != "passed"
        or gate.get("passed") is not True
        or gate.get("hidden_state_authorized") is not False
        or gate.get("legacy_pilot_as_dev_authorized") is not False
        or gate.get("admission", {}).get("sha256") != sha256_file(admission)
        or gate.get("confirmation_locked", {}).get("path") != str(result_path.resolve())
        or gate.get("confirmation_locked", {}).get("sha256") != sha256_file(result_path)
    ):
        raise RuntimeError("CECD v3 three-stage input gate contract mismatch")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    provenance = result.get("provenance", {})
    expected_models = {
        row.get("model")
        for row in gate.get("runs", {}).get("confirmation_locked", [])
    }
    if (
        result.get("version") != EXPECTED_STAGE_ANALYSIS
        or result.get("status") != "complete"
        or result.get("stage_label") != "confirmation_locked"
        or result.get("source_manifest_split") != "confirmation"
        or provenance.get("code_sha256")
        != sha256_file(
            ROOT
            / "anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py"
        )
        or provenance.get("seed") != 42
        or provenance.get("bootstrap_draws") != 5000
        or provenance.get("mode") != "confirmation_locked"
        or set(result.get("models", {})) != expected_models
    ):
        raise RuntimeError("CECD locked-confirmation provenance/contract mismatch")
    stage_gate = result.get("gate", {})
    passing = stage_gate.get("confirmation_passing_models")
    authorized = stage_gate.get("authorized_for_method_level_treble_adapter_run")
    if (
        stage_gate.get("name") != "behavioral_confirmation_locked_v1"
        or not isinstance(passing, list)
        or len(passing) != len(set(passing))
        or not set(passing).issubset(expected_models)
        or stage_gate.get("both_models_pass") is not (set(passing) == expected_models)
        or authorized is not stage_gate.get("both_models_pass")
        or stage_gate.get("authorized_for_hidden_state_stage") is not False
        or result.get("exact_treble_method_collision", {}).get(
            "hidden_state_authorized"
        )
        is not False
    ):
        raise RuntimeError("CECD locked behavioral/hidden-state gate contract mismatch")
    return result


def analyze_admission(
    *,
    pack: Path,
    delivery: Path,
    output: Path,
    completed: dict[str, Path],
    attestations: dict[str, Path],
) -> dict[str, Any]:
    frozen_completed, frozen_attestations, bundle_lock_path = freeze_human_bundle(
        output=output,
        completed=completed,
        attestations=attestations,
    )
    # Only after all eight role/attestation files are hash-frozen may any code
    # open (even merely hash) the sealed mapping.
    validate_pack_closure(pack, output)
    validation = validate_all(
        pack_dir=pack,
        completed=frozen_completed,
        attestations=frozen_attestations,
    )
    if validation.get("version") != VALIDATION_VERSION:
        raise RuntimeError("unexpected CECD human-return validator version")
    validation.update(
        {
            "completed_files": {role: record(path) for role, path in frozen_completed.items()},
            "attestation_files": {
                role: record(path) for role, path in frozen_attestations.items()
            },
            "human_return_bundle_lock": record(bundle_lock_path),
        }
    )
    validation_path = output / "return_validation.json"
    write_once_or_equal(validation_path, validation)
    analysis_path = output / "analysis.json"
    with tempfile.TemporaryDirectory(prefix="cecd-admission-replay-", dir=output) as temporary:
        replay_path = Path(temporary) / "analysis.json"
        command = [
            sys.executable,
            "-m",
            "anchor.corrected_sgta.analyze_cecd_admission_reviews_v1",
            "--pack-dir",
            str(pack),
            "--clinical-review",
            str(frozen_completed["clinical_reviewer_1"]),
            "--clinical-review",
            str(frozen_completed["clinical_reviewer_2"]),
            "--clinical-template-review",
            str(frozen_completed["clinical_template_reviewer"]),
            "--language-review",
            str(frozen_completed["language_reviewer"]),
            "--output",
            str(replay_path),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode not in {0, 2} or not replay_path.is_file():
            raise RuntimeError(
                f"CECD admission analyzer failed ({result.returncode}): "
                f"{result.stderr[-3000:]}"
            )
        analysis = json.loads(replay_path.read_text(encoding="utf-8"))
    expected_provenance = {
        "sealed_mapping": str((pack / "sealed_mapping.json").resolve()),
        "sealed_mapping_sha256": sha256_file(pack / "sealed_mapping.json"),
        "clinical_reviews": [
            record(frozen_completed["clinical_reviewer_1"]),
            record(frozen_completed["clinical_reviewer_2"]),
        ],
        "clinical_template_review": record(
            frozen_completed["clinical_template_reviewer"]
        ),
        "language_review": record(frozen_completed["language_reviewer"]),
    }
    if analysis.get("provenance") != expected_provenance:
        raise RuntimeError("CECD admission analysis does not bind the current frozen returns")
    human = {
        "validation_artifact": record(validation_path),
        "attestations": {role: record(path) for role, path in frozen_attestations.items()},
    }
    if analysis.get("human_return_validation") not in (None, human):
        raise RuntimeError("CECD analysis human-return provenance collision")
    analysis["human_return_validation"] = human
    ready = delivery_ready(delivery)
    analysis["reviewer_delivery_validation"] = {
        "verification": ready["verification"],
        "browser_smoke": ready["browser_smoke"],
    }
    analysis["human_return_bundle_lock"] = record(bundle_lock_path)
    analysis["source_pack_lock"] = record(output / "pack_source_lock.json")
    write_once_or_equal(analysis_path, analysis)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("passed") is not True:
        return {
            "stage": "human_admission_failed_terminal",
            "analysis": str(analysis_path.resolve()),
            "analysis_sha256": sha256_file(analysis_path),
            "model_scoring_authorized": False,
            "retry_authorized": False,
        }
    require_cecd_authorization(analysis_path)
    return launch_or_monitor_stage(analysis_path)


def advance(pack: Path, delivery: Path, inbox: Path, output: Path) -> dict[str, Any]:
    readiness = delivery_ready(delivery)
    if not readiness["ready"]:
        return {"stage": "waiting_for_verified_reviewer_deliveries", **readiness}
    completed, attestations = return_paths(inbox)
    missing = [
        str(path)
        for path in [*completed.values(), *attestations.values()]
        if not path.is_file()
    ]
    if missing:
        return {"stage": "waiting_for_four_independent_returns", "missing": missing}
    return analyze_admission(
        pack=pack,
        delivery=delivery,
        output=output,
        completed=completed,
        attestations=attestations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--delivery", type=Path, default=DEFAULT_DELIVERY)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--heartbeat", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError("interval must be at least one second")
    for required in (args.pack / "manifest.json", args.pack / "sealed_mapping.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    args.inbox.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_inbox_instructions(args.inbox)
    heartbeat = args.heartbeat or args.output / "monitor.heartbeat.json"
    completed, attestations = return_paths(args.inbox)
    all_human_paths = [*completed.values(), *attestations.values()]
    prior_signatures: dict[str, dict[str, Any]] = {}
    while True:
        try:
            current = signatures(all_human_paths)
            if current and current != prior_signatures:
                state: dict[str, Any] = {
                    "stage": "waiting_for_stable_human_inputs",
                    "required_unchanged_polls": 2,
                    "human_input_signatures": current,
                }
            else:
                state = advance(args.pack, args.delivery, args.inbox, args.output)
            payload = {
                "version": VERSION,
                "time": utc_now(),
                "pid": os.getpid(),
                "pack": str(args.pack.resolve()),
                "delivery": str(args.delivery.resolve()),
                "inbox": str(args.inbox.resolve()),
                "output": str(args.output.resolve()),
                "clinical_or_language_labels_synthesized": False,
                "attestations_synthesized": False,
                "sealed_mapping_exposed_before_returns_locked": False,
                **state,
            }
            prior_signatures = current
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            payload = {
                "version": VERSION,
                "time": utc_now(),
                "pid": os.getpid(),
                "stage": "input_or_transition_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "clinical_or_language_labels_synthesized": False,
                "attestations_synthesized": False,
            }
        atomic_write_json(heartbeat, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        terminal = payload.get("stage") in {
            "human_admission_failed_terminal",
            "two_model_stage1_failed_terminal",
            "two_model_stage1_complete",
        }
        if args.once or terminal:
            return
        if payload.get("stage") == "two_model_stage1_launched":
            continue
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
