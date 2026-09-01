#!/usr/bin/env python3
"""Bind a frozen Treble/CECD comparison plan to a passed two-model Stage 1.

This is the missing pre-run half of the collision protocol.  It never evaluates
method outcomes.  It permits only one hash-bound controlled comparison after
the existing CECD validator reconstructs a passing Huatuo+Hulu Stage 1 and the
method-output directory is still empty.  Exact/paper-native Treble and general
hidden-state, GPU, or paper authorization remain false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from anchor.corrected_sgta.build_cecd_dual_semantics_preflight_v1 import (
    validate_build_receipt,
)
from anchor.corrected_sgta.treble_collision_contract import (
    DUAL_SEMANTICS_METHODS,
    DUAL_SEMANTICS_OUTCOME_SCHEMA,
    METHOD_CLOSURE_LIMITATIONS,
    validate_dual_semantics_preflight_contract,
)
import scripts.monitor_cecd_admission_pipeline as cecd_monitor


VERSION = "cecd-dual-semantics-runtime-authorization-v2-canonical-builder"
ROOT = Path("/home/dbw/ANCHOR")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required preflight input is missing: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_once_or_equal(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"write-once authorization collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def authorize(
    *,
    stage1_analysis: Path,
    stage1_input_gate: Path,
    admission: Path,
    preflight_path: Path,
    preflight_build: Path,
    output: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    for path in (
        stage1_analysis, stage1_input_gate, admission, preflight_path, preflight_build
    ):
        if not path.is_file():
            raise RuntimeError(f"required preflight input is missing: {path}")
    expected_gate = (
        cecd_monitor.ROOT
        / "corrected_runs/vindr_v2/cecd_three_stage_v3/input_gate.json"
    ).resolve()
    if stage1_input_gate.resolve() != expected_gate:
        raise RuntimeError("stage1_input_gate is not the gate reconstructed by the CECD validator")

    validate_build_receipt(
        receipt_path=preflight_build,
        stage_state=root / "corrected_runs/detached_jobs/cecd-three-stage-v3.json",
        stage_analysis=stage1_analysis,
        stage_input_gate=stage1_input_gate,
        admission=admission,
        preflight_path=preflight_path,
        root=root,
    )

    stage = cecd_monitor.validate_stage_result(
        result_path=stage1_analysis.resolve(), admission=admission.resolve()
    )
    gate = stage.get("gate", {})
    passing_models = gate.get("confirmation_passing_models")
    if (
        gate.get("authorized_for_method_level_treble_adapter_run") is not True
        or gate.get("authorized_for_hidden_state_stage") is not False
        or not isinstance(passing_models, list)
        or len(passing_models) != 2
        or len(set(passing_models)) != 2
    ):
        raise RuntimeError("two-model CECD locked confirmation did not authorize closest-work comparison")

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not isinstance(preflight, dict):
        raise RuntimeError("dual-semantics preflight must be a JSON object")
    validate_dual_semantics_preflight_contract(preflight)
    actual_hashes = {
        "stage1_analysis_sha256": sha256_file(stage1_analysis),
        "stage1_input_gate_sha256": sha256_file(stage1_input_gate),
        "admission_sha256": sha256_file(admission),
    }
    for field, digest in actual_hashes.items():
        if preflight[field] != digest:
            raise RuntimeError(f"preflight {field} does not bind the current file")
    fingerprint_models = {
        record["model_id"] for record in preflight["model_fingerprints"].values()
    }
    if set(passing_models) != fingerprint_models or set(stage.get("models", {})) != fingerprint_models:
        raise RuntimeError("preflight model fingerprints do not bind both passing Stage-1 models")

    method_output_root = Path(preflight["method_output_root"]).resolve()
    resolved_root = root.resolve()
    if method_output_root == resolved_root or resolved_root not in method_output_root.parents:
        raise RuntimeError("method output root must be a narrow path inside the repository")
    if output.resolve() == method_output_root or method_output_root in output.resolve().parents:
        raise RuntimeError("authorization artifact must remain outside the method-output root")
    if method_output_root.exists() and any(method_output_root.iterdir()):
        raise RuntimeError("method outputs already exist; outcome-blind preflight is too late")

    payload: dict[str, Any] = {
        "version": VERSION,
        "status": "controlled_dual_semantics_comparison_authorized",
        "stage1_analysis": file_record(stage1_analysis),
        "stage1_input_gate": file_record(stage1_input_gate),
        "admission": file_record(admission),
        "preflight": file_record(preflight_path),
        "preflight_build": file_record(preflight_build),
        "method_output_root": str(method_output_root),
        "allowed_methods": list(DUAL_SEMANTICS_METHODS),
        "required_outcome_schema": DUAL_SEMANTICS_OUTCOME_SCHEMA,
        "controlled_method_comparison_authorized": True,
        "comparison_scope": (
            "Treble dual-semantics plus factorial controls only; not full mitigation baseline closure"
        ),
        "static_activation_control_status": (
            "Treble proceedings/released common-protocol variants only; not exact paper-native"
        ),
        "official_compatible_dynamic_activation_baseline_present": False,
        "representation_level_pid_control_present": False,
        "locked_test_behavioral_increment_confirmed": True,
        "full_method_gate_authorized": False,
        "oral_baseline_closure_authorized": False,
        "method_closure_limitations": list(METHOD_CLOSURE_LIMITATIONS),
        "cecd_hidden_state_intervention_authorized_only_inside_locked_comparison": True,
        "general_hidden_state_stage_authorized": False,
        "paper_native_treble_authorized": False,
        "exact_treble_authorized": False,
        "general_gpu_authorized": False,
        "paper_claim_authorized": False,
        "method_outputs_consumed": False,
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _write_once_or_equal(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-analysis", type=Path, required=True)
    parser.add_argument("--stage1-input-gate", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = authorize(
        stage1_analysis=args.stage1_analysis,
        stage1_input_gate=args.stage1_input_gate,
        admission=args.admission,
        preflight_path=args.preflight,
        preflight_build=args.preflight_build,
        output=args.output,
        root=ROOT,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
