from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

import anchor.corrected_sgta.authorize_cecd_dual_semantics_preflight_v1 as module
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
)


def _sha(path: Path) -> str:
    return module.sha256_file(path)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "stage1": tmp_path / "stage1.json",
        "gate": tmp_path
        / "corrected_runs/vindr_v2/cecd_three_stage_v3/input_gate.json",
        "admission": tmp_path / "admission.json",
        "preflight_build": tmp_path / "preflight_build.json",
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fixture": name}), encoding="utf-8")
    return paths


def _preflight(tmp_path: Path, paths: dict[str, Path]) -> dict:
    fingerprints = {}
    ledger = {}
    proceedings_calibration = proceedings_compute_ledger()
    released_calibration = asdict(released_code_compute_ledger())
    for index, family in enumerate(("huatuo", "hulu"), 1):
        fingerprints[family] = {
            "model_id": f"{family}:frozen",
            "checkpoint_sha256": str(index) * 64,
            "processor_sha256": str(index + 1) * 64,
            "template_sha256": str(index + 2) * 64,
            "generation_contract_sha256": str(index + 3) * 64,
            "hook_contract_sha256": str(index + 4) * 64,
            "vision_token_transport_contract_sha256": str(index + 5) * 64,
        }
        ledger[family] = {
            "treble_proceedings": dict(proceedings_calibration),
            "treble_released": dict(released_calibration),
            "target_examples": 64,
            "cecd_target_generation_forwards": 256,
            "full_orbit_target_generation_forwards": 256,
        }
    return {
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
        "stage1_analysis_sha256": _sha(paths["stage1"]),
        "stage1_input_gate_sha256": _sha(paths["gate"]),
        "admission_sha256": _sha(paths["admission"]),
        "calibration_split": "dev",
        "evaluation_split": "locked_test",
        "calibration_manifest_sha256": "a" * 64,
        "evaluation_manifest_sha256": "b" * 64,
        "record_keys_sha256": "c" * 64,
        "claim_contract_sha256": "d" * 64,
        "methods": list(DUAL_SEMANTICS_METHODS),
        "primary_envelope_controls": list(PRIMARY_ENVELOPE_CONTROLS),
        "method_metrics": list(METHOD_METRICS),
        "thresholds": DUAL_SEMANTICS_THRESHOLDS,
        "bootstrap_replicates": 10_000,
        "bootstrap_unit": "cluster_id",
        "compute_ledger": ledger,
        "method_output_root": str((tmp_path / "method_outputs").resolve()),
    }


def _passed_stage() -> dict:
    models = {"huatuo:frozen": {}, "hulu:frozen": {}}
    return {
        "models": models,
        "gate": {
            "confirmation_passing_models": list(models),
            "authorized_for_method_level_treble_adapter_run": True,
            "authorized_for_hidden_state_stage": False,
        },
    }


def _authorize(tmp_path: Path, monkeypatch, *, stage=None):
    paths = _inputs(tmp_path)
    preflight = _preflight(tmp_path, paths)
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    monkeypatch.setattr(
        module.cecd_monitor,
        "validate_stage_result",
        lambda **_kwargs: _passed_stage() if stage is None else stage,
    )
    monkeypatch.setattr(module.cecd_monitor, "ROOT", tmp_path)
    monkeypatch.setattr(module, "validate_build_receipt", lambda **_: {})
    output = tmp_path / "authorization.json"
    result = module.authorize(
        stage1_analysis=paths["stage1"],
        stage1_input_gate=paths["gate"],
        admission=paths["admission"],
        preflight_path=preflight_path,
        preflight_build=paths["preflight_build"],
        output=output,
        root=tmp_path,
    )
    return result, paths, preflight_path, output


def test_runtime_binder_authorizes_only_locked_comparison_and_is_replay_stable(
    tmp_path: Path, monkeypatch
) -> None:
    result, paths, preflight_path, output = _authorize(tmp_path, monkeypatch)
    assert result["controlled_method_comparison_authorized"] is True
    assert result["official_compatible_dynamic_activation_baseline_present"] is False
    assert result["representation_level_pid_control_present"] is False
    assert result["locked_test_behavioral_increment_confirmed"] is True
    assert result["full_method_gate_authorized"] is False
    assert result["oral_baseline_closure_authorized"] is False
    assert result["cecd_hidden_state_intervention_authorized_only_inside_locked_comparison"] is True
    assert result["general_hidden_state_stage_authorized"] is False
    assert result["general_gpu_authorized"] is False
    assert result["paper_native_treble_authorized"] is False
    assert result["exact_treble_authorized"] is False
    assert result["paper_claim_authorized"] is False
    replay = module.authorize(
        stage1_analysis=paths["stage1"],
        stage1_input_gate=paths["gate"],
        admission=paths["admission"],
        preflight_path=preflight_path,
        preflight_build=paths["preflight_build"],
        output=output,
        root=tmp_path,
    )
    assert replay == result


def test_runtime_binder_rejects_failed_stage_or_model_alias(tmp_path: Path, monkeypatch) -> None:
    failed = _passed_stage()
    failed["gate"]["authorized_for_method_level_treble_adapter_run"] = False
    with pytest.raises(RuntimeError, match="did not authorize"):
        _authorize(tmp_path, monkeypatch, stage=failed)

    alias = _passed_stage()
    alias["models"] = {"huatuo:frozen": {}, "unexpected:model": {}}
    with pytest.raises(RuntimeError, match="do not bind"):
        _authorize(tmp_path / "alias", monkeypatch, stage=alias)


def test_runtime_binder_rejects_legacy_pilot_as_dev_gate_path(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _inputs(tmp_path)
    legacy = tmp_path / "corrected_runs/vindr_v2/cecd_two_model_stage1_v2/input_gate.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("{}", encoding="utf-8")
    preflight = _preflight(tmp_path, {**paths, "gate": legacy})
    preflight_path = tmp_path / "legacy-preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    monkeypatch.setattr(module.cecd_monitor, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="not the gate reconstructed"):
        module.authorize(
            stage1_analysis=paths["stage1"], stage1_input_gate=legacy,
            admission=paths["admission"], preflight_path=preflight_path,
            preflight_build=paths["preflight_build"],
            output=tmp_path / "authorization.json", root=tmp_path,
        )


def test_runtime_binder_rejects_hash_drift_and_late_freeze(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "validate_build_receipt", lambda **_: {})
    paths = _inputs(tmp_path)
    preflight = _preflight(tmp_path, paths)
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    paths["admission"].write_text("drift", encoding="utf-8")
    monkeypatch.setattr(module.cecd_monitor, "validate_stage_result", lambda **_: _passed_stage())
    monkeypatch.setattr(module.cecd_monitor, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="admission_sha256"):
        module.authorize(
            stage1_analysis=paths["stage1"],
            stage1_input_gate=paths["gate"],
            admission=paths["admission"],
            preflight_path=preflight_path,
            preflight_build=paths["preflight_build"],
            output=tmp_path / "authorization.json",
            root=tmp_path,
        )

    tmp_path = tmp_path / "late"
    tmp_path.mkdir()
    paths = _inputs(tmp_path)
    preflight = _preflight(tmp_path, paths)
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    method_root = Path(preflight["method_output_root"])
    method_root.mkdir()
    (method_root / "already_opened.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module.cecd_monitor, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="preflight is too late"):
        module.authorize(
            stage1_analysis=paths["stage1"],
            stage1_input_gate=paths["gate"],
            admission=paths["admission"],
            preflight_path=preflight_path,
            preflight_build=paths["preflight_build"],
            output=tmp_path / "authorization.json",
            root=tmp_path,
        )
