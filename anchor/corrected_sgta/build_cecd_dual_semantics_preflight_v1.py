#!/usr/bin/env python3
"""Build the canonical outcome-blind CECD dual-semantics preflight.

The builder is an authority-preserving bridge, not an analyzer.  It accepts
only a completed canonical three-stage-v3 detached job, the verifier's exact
input gate, its locked-confirmation artifact, and a genuine hash-bound clinical
admission.  It never opens model result rows.  Dev/confirmation claim manifests
are reconstructed from the frozen reader manifest and selection function.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from anchor.corrected_sgta.cecd_dual_semantics_ce_adapter_v1 import (
    FORMAL_CE_CLAIM_CONTRACT_SCHEMA,
    _record_key,
)
from anchor.corrected_sgta.cecd_dual_semantics_worker_v1 import (
    INPUT_SIDECAR_SCHEMA,
    compute_model_fingerprint,
)
from anchor.corrected_sgta.run_cecd_factorial_v1 import (
    DEFAULT_IMAGE_ROOT,
    FROZEN_SEED,
    canonical_json_sha256,
    selection,
)
from anchor.corrected_sgta.treble_collision_contract import (
    DUAL_SEMANTICS_METHODS,
    DUAL_SEMANTICS_PREFLIGHT_SCHEMA,
    DUAL_SEMANTICS_THRESHOLDS,
    DUAL_SEMANTICS_VARIANTS,
    METHOD_METRICS,
    PRIMARY_ENVELOPE_CONTROLS,
    TREBLE_REPOSITORY_COMMIT,
    proceedings_compute_ledger,
    released_code_compute_ledger,
    validate_dual_semantics_preflight_contract,
)
from anchor.corrected_sgta.verify_cecd_three_stage_v3 import (
    EXPECTED_SELECTION_HASHES,
    VERSION as THREE_STAGE_GATE_VERSION,
)
import scripts.monitor_cecd_admission_pipeline as cecd_monitor


VERSION = "cecd-dual-semantics-canonical-preflight-builder-v1"
BUILD_RECEIPT_SCHEMA = "cecd-dual-semantics-preflight-build-receipt-v1"
ROOT = Path("/home/dbw/ANCHOR")
CANONICAL_GPU_LOCK_RELATIVE = Path(
    "corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock"
)
DEFAULT_HUATUO_MODEL = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HULU_MODEL = Path("/home/dbw/models/Hulu-Med-4B")
DEFAULT_HUATUO_SOURCE = Path("/home/dbw/HuatuoGPT-Vision")


class PreflightBuildError(RuntimeError):
    """Raised before any formal model execution when authority drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PreflightBuildError(f"required regular file is missing or symlinked: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_build_receipt(
    *,
    receipt_path: Path,
    stage_state: Path,
    stage_analysis: Path,
    stage_input_gate: Path,
    admission: Path,
    preflight_path: Path,
    root: Path,
) -> dict[str, Any]:
    """Revalidate the complete write-once builder handoff without outcomes."""

    payload = load_object(receipt_path, "canonical preflight build receipt")
    fingerprint = payload.get("fingerprint")
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if not isinstance(fingerprint, str) or canonical_sha256(body) != fingerprint:
        raise PreflightBuildError("preflight build receipt fingerprint mismatch")
    required = {
        "schema_version", "version", "status", "stage_state", "stage1_analysis",
        "stage1_input_gate", "admission", "preflight", "input_sidecar",
        "input_bindings", "model_fingerprints",
        "stage_confirmation_model_provenance_sha256", "canonical_gpu_lock",
        "raw_model_outcome_rows_consumed_by_builder", "human_returns_synthesized",
        "gpu_or_model_execution_performed", "paper_claim_authorized", "fingerprint",
    }
    if set(payload) != required:
        raise PreflightBuildError("preflight build receipt fields are not closed")
    if (
        payload["schema_version"] != BUILD_RECEIPT_SCHEMA
        or payload["version"] != VERSION
        or payload["status"] != "complete_outcome_blind_no_model_execution"
        or payload["raw_model_outcome_rows_consumed_by_builder"] is not False
        or payload["human_returns_synthesized"] is not False
        or payload["gpu_or_model_execution_performed"] is not False
        or payload["paper_claim_authorized"] is not False
        or payload["canonical_gpu_lock"]
        != str((root / CANONICAL_GPU_LOCK_RELATIVE).resolve())
    ):
        raise PreflightBuildError("preflight build receipt scope/lock contract mismatch")
    expected_paths = {
        "stage_state": stage_state,
        "stage1_analysis": stage_analysis,
        "stage1_input_gate": stage_input_gate,
        "admission": admission,
        "preflight": preflight_path,
    }
    for label, path in expected_paths.items():
        if payload.get(label) != file_record(path):
            raise PreflightBuildError(f"preflight build receipt {label} binding drift")
    sidecar = preflight_path.with_name(f"{preflight_path.stem}.inputs.json")
    if payload.get("input_sidecar") != file_record(sidecar):
        raise PreflightBuildError("preflight build input-sidecar binding drift")
    bindings = payload.get("input_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "calibration_manifest", "evaluation_manifest", "record_keys", "claim_contract"
    }:
        raise PreflightBuildError("preflight build input closure is invalid")
    for label, record in bindings.items():
        if not isinstance(record, Mapping) or record != file_record(Path(str(record.get("path", "")))):
            raise PreflightBuildError(f"preflight build {label} binding drift")
    return payload


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightBuildError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PreflightBuildError(f"{label} must be a JSON object")
    return payload


def _write_once_text(path: Path, text: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            raise PreflightBuildError(f"write-once preflight collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_once_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_once_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _write_once_text(path, text)


def _validate_stage_state(path: Path, root: Path) -> dict[str, Any]:
    state = load_object(path, "three-stage detached state")
    command = state.get("command")
    command_text = " ".join(str(item) for item in command) if isinstance(command, list) else ""
    if (
        state.get("name") != "cecd-three-stage-v3"
        or state.get("status") != "done"
        or state.get("exit_code") != 0
        or "run_cecd_three_stage_v3.sh" not in command_text
        or Path(str(state.get("cwd", ""))).resolve() != root.resolve()
    ):
        raise PreflightBuildError("detached state is not a successful canonical three-stage-v3 job")
    return state


def _validate_gate(
    *, gate_path: Path, stage_analysis: Path, admission: Path, stage: Mapping[str, Any]
) -> dict[str, Any]:
    gate = load_object(gate_path, "three-stage input gate")
    stage_models = set(stage.get("models", {}))
    confirmation_runs = gate.get("runs", {}).get("confirmation_locked", [])
    gate_models = {
        str(row.get("model")) for row in confirmation_runs if isinstance(row, Mapping)
    }
    if (
        gate.get("version") != THREE_STAGE_GATE_VERSION
        or gate.get("status") != "passed"
        or gate.get("passed") is not True
        or gate.get("authorized_for_method_level_treble_adapter_run") is not True
        or gate.get("hidden_state_authorized") is not False
        or gate.get("legacy_pilot_as_dev_authorized") is not False
        or gate.get("admission", {}).get("path") != str(admission.resolve())
        or gate.get("admission", {}).get("sha256") != sha256_file(admission)
        or gate.get("confirmation_locked", {}).get("path")
        != str(stage_analysis.resolve())
        or gate.get("confirmation_locked", {}).get("sha256")
        != sha256_file(stage_analysis)
        or len(stage_models) != 2
        or gate_models != stage_models
    ):
        raise PreflightBuildError("three-stage-v3 gate/confirmation/admission binding drift")
    return gate


def _claim_rows(stage_label: str) -> list[dict[str, Any]]:
    rows = selection(stage_label)
    normalized = []
    for source in rows:
        row = dict(source)
        if not row.get("image_id") or not row.get("finding"):
            raise PreflightBuildError(f"{stage_label} contains an unidentifiable claim")
        row["cluster_id"] = str(row["image_id"])
        row["record_key"] = _record_key(row)
        normalized.append(row)
    keys = [str(row["record_key"]) for row in normalized]
    if len(keys) != len(set(keys)):
        raise PreflightBuildError(f"{stage_label} contains duplicate record keys")
    if canonical_json_sha256(keys) != EXPECTED_SELECTION_HASHES[stage_label]:
        raise PreflightBuildError(f"{stage_label} selection hash drift")
    return normalized


def _compute_ledger(target_examples: int) -> dict[str, Any]:
    calibration = {
        "treble_proceedings": proceedings_compute_ledger(),
        "treble_released": asdict(released_code_compute_ledger()),
        "target_examples": target_examples,
        "cecd_target_generation_forwards": 4 * target_examples,
        "full_orbit_target_generation_forwards": 4 * target_examples,
    }
    return {family: json.loads(json.dumps(calibration)) for family in ("huatuo", "hulu")}


def build_preflight(
    *,
    stage_state: Path,
    stage_analysis: Path,
    stage_input_gate: Path,
    admission: Path,
    preflight_path: Path,
    build_receipt: Path,
    input_root: Path,
    method_output_root: Path,
    huatuo_model: Path = DEFAULT_HUATUO_MODEL,
    hulu_model: Path = DEFAULT_HULU_MODEL,
    huatuo_source_root: Path = DEFAULT_HUATUO_SOURCE,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build all pre-output artifacts or verify byte-identical replay."""

    for path in (stage_state, stage_analysis, stage_input_gate, admission):
        if not path.is_file():
            raise PreflightBuildError(f"required authority input is missing: {path}")
    if build_receipt.is_file():
        return validate_build_receipt(
            receipt_path=build_receipt,
            stage_state=stage_state,
            stage_analysis=stage_analysis,
            stage_input_gate=stage_input_gate,
            admission=admission,
            preflight_path=preflight_path,
            root=root,
        )
    _validate_stage_state(stage_state, root)
    stage = cecd_monitor.validate_stage_result(
        result_path=stage_analysis.resolve(), admission=admission.resolve()
    )
    gate = _validate_gate(
        gate_path=stage_input_gate,
        stage_analysis=stage_analysis,
        admission=admission,
        stage=stage,
    )
    passing = stage.get("gate", {}).get("confirmation_passing_models")
    if (
        stage.get("gate", {}).get("authorized_for_method_level_treble_adapter_run")
        is not True
        or stage.get("gate", {}).get("authorized_for_hidden_state_stage") is not False
        or not isinstance(passing, list)
        or len(passing) != 2
        or len(set(passing)) != 2
    ):
        raise PreflightBuildError("locked confirmation is not a two-model method GO")

    repository = root.resolve()
    output_root = method_output_root.resolve()
    if output_root == repository or repository not in output_root.parents:
        raise PreflightBuildError("method output root must be a narrow repository child")
    if output_root.exists() and any(output_root.iterdir()):
        raise PreflightBuildError("formal method outputs already exist; preflight is too late")
    model_dirs = {"huatuo": huatuo_model.resolve(), "hulu": hulu_model.resolve()}
    fingerprints = {
        family: compute_model_fingerprint(family, path)
        for family, path in model_dirs.items()
    }
    if {record["model_id"] for record in fingerprints.values()} != set(passing):
        raise PreflightBuildError("current Huatuo/Hulu fingerprints do not bind both Stage-1 models")

    calibration_rows = _claim_rows("dev_fit")
    evaluation_rows = _claim_rows("confirmation_locked")
    input_paths = {
        "calibration_manifest": input_root / "calibration_manifest.jsonl",
        "evaluation_manifest": input_root / "evaluation_manifest.jsonl",
        "record_keys": input_root / "record_keys.json",
        "claim_contract": input_root / "claim_contract.json",
    }
    write_once_jsonl(input_paths["calibration_manifest"], calibration_rows)
    write_once_jsonl(input_paths["evaluation_manifest"], evaluation_rows)
    write_once_json(
        input_paths["record_keys"],
        {
            "schema_version": "cecd-dual-semantics-record-keys-v1",
            "record_keys": [row["record_key"] for row in evaluation_rows],
        },
    )
    write_once_json(
        input_paths["claim_contract"],
        {
            "schema_version": FORMAL_CE_CLAIM_CONTRACT_SCHEMA,
            "task": "fixed_claim_single_token_ce",
            "render_names": ["baseline_percentile", "native_linear"],
            "prompt_names": ["existential", "radiograph_subject"],
            "seed": FROZEN_SEED,
            "image_root": str(DEFAULT_IMAGE_ROOT),
            "minimum_clusters": 30,
        },
    )

    preflight = {
        "schema_version": DUAL_SEMANTICS_PREFLIGHT_SCHEMA,
        "frozen_before_method_outputs": True,
        "source_repo_commit": TREBLE_REPOSITORY_COMMIT,
        "reproduction_fidelity": "dual_semantics_common_protocol_envelope",
        "paper_native_claimed": False,
        "exact_reproduction_claimed": False,
        "implementation_origin": "independent_clean_room_from_public_equations_and_audited_arithmetic",
        "redistribution_policy": "local_evaluation_only_no_official_source_or_demo_redistribution",
        "variants": DUAL_SEMANTICS_VARIANTS,
        "model_fingerprints": fingerprints,
        "stage1_analysis_sha256": sha256_file(stage_analysis),
        "stage1_input_gate_sha256": sha256_file(stage_input_gate),
        "admission_sha256": sha256_file(admission),
        "calibration_split": "dev",
        "evaluation_split": "locked_test",
        "calibration_manifest_sha256": sha256_file(input_paths["calibration_manifest"]),
        "evaluation_manifest_sha256": sha256_file(input_paths["evaluation_manifest"]),
        "record_keys_sha256": sha256_file(input_paths["record_keys"]),
        "claim_contract_sha256": sha256_file(input_paths["claim_contract"]),
        "methods": list(DUAL_SEMANTICS_METHODS),
        "primary_envelope_controls": list(PRIMARY_ENVELOPE_CONTROLS),
        "method_metrics": list(METHOD_METRICS),
        "thresholds": DUAL_SEMANTICS_THRESHOLDS,
        "bootstrap_replicates": 10_000,
        "bootstrap_unit": "cluster_id",
        "compute_ledger": _compute_ledger(len(evaluation_rows)),
        "method_output_root": str(output_root),
    }
    validate_dual_semantics_preflight_contract(preflight)
    write_once_json(preflight_path, preflight)
    sidecar_path = preflight_path.with_name(f"{preflight_path.stem}.inputs.json")
    sidecar = {
        "schema_version": INPUT_SIDECAR_SCHEMA,
        "preflight_sha256": sha256_file(preflight_path),
        "model_dirs": {family: str(path) for family, path in model_dirs.items()},
        "huatuo_source_root": str(huatuo_source_root.resolve()),
        "input_bindings": {
            name: str(path.resolve()) for name, path in input_paths.items()
        },
    }
    write_once_json(sidecar_path, sidecar)

    confirmation_provenance = {
        str(row["family"]): str(row["model_provenance_sha256"])
        for row in gate["runs"]["confirmation_locked"]
    }
    receipt: dict[str, Any] = {
        "schema_version": BUILD_RECEIPT_SCHEMA,
        "version": VERSION,
        "status": "complete_outcome_blind_no_model_execution",
        "stage_state": file_record(stage_state),
        "stage1_analysis": file_record(stage_analysis),
        "stage1_input_gate": file_record(stage_input_gate),
        "admission": file_record(admission),
        "preflight": file_record(preflight_path),
        "input_sidecar": file_record(sidecar_path),
        "input_bindings": {name: file_record(path) for name, path in input_paths.items()},
        "model_fingerprints": fingerprints,
        "stage_confirmation_model_provenance_sha256": confirmation_provenance,
        "canonical_gpu_lock": str((root / CANONICAL_GPU_LOCK_RELATIVE).resolve()),
        "raw_model_outcome_rows_consumed_by_builder": False,
        "human_returns_synthesized": False,
        "gpu_or_model_execution_performed": False,
        "paper_claim_authorized": False,
    }
    receipt["fingerprint"] = canonical_sha256(receipt)
    write_once_json(build_receipt, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-state", type=Path, required=True)
    parser.add_argument("--stage-analysis", type=Path, required=True)
    parser.add_argument("--stage-input-gate", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--method-output-root", type=Path, required=True)
    parser.add_argument("--huatuo-model", type=Path, default=DEFAULT_HUATUO_MODEL)
    parser.add_argument("--hulu-model", type=Path, default=DEFAULT_HULU_MODEL)
    parser.add_argument("--huatuo-source-root", type=Path, default=DEFAULT_HUATUO_SOURCE)
    args = parser.parse_args()
    result = build_preflight(
        stage_state=args.stage_state,
        stage_analysis=args.stage_analysis,
        stage_input_gate=args.stage_input_gate,
        admission=args.admission,
        preflight_path=args.preflight,
        build_receipt=args.build_receipt,
        input_root=args.input_root,
        method_output_root=args.method_output_root,
        huatuo_model=args.huatuo_model,
        hulu_model=args.hulu_model,
        huatuo_source_root=args.huatuo_source_root,
        root=ROOT,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
