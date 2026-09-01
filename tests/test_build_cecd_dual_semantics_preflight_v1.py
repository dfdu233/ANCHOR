from __future__ import annotations

import json
from pathlib import Path

import pytest

import anchor.corrected_sgta.build_cecd_dual_semantics_preflight_v1 as builder


def _fingerprint(family: str) -> dict:
    offset = 1 if family == "huatuo" else 2
    return {
        "model_id": f"{family}:{'HuatuoGPT-Vision-7B' if family == 'huatuo' else 'Hulu-Med-4B'}",
        "checkpoint_sha256": str(offset) * 64,
        "processor_sha256": str(offset + 1) * 64,
        "template_sha256": str(offset + 2) * 64,
        "generation_contract_sha256": str(offset + 3) * 64,
        "hook_contract_sha256": str(offset + 4) * 64,
        "vision_token_transport_contract_sha256": str(offset + 5) * 64,
    }


def _fixture(tmp_path: Path, monkeypatch) -> dict:
    root = tmp_path
    authority = root / "corrected_runs/vindr_v2"
    stage_root = authority / "cecd_three_stage_v3"
    admission = authority / "cecd_human_admission_v2/analysis.json"
    state = root / "corrected_runs/detached_jobs/cecd-three-stage-v3.json"
    stage_analysis = stage_root / "confirmation_locked.json"
    gate_path = stage_root / "input_gate.json"
    for path in (admission, state, stage_analysis, gate_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    admission.write_text(json.dumps({"genuine": True}), encoding="utf-8")
    stage_analysis.write_text(json.dumps({"locked": True}), encoding="utf-8")
    state.write_text(
        json.dumps(
            {
                "name": "cecd-three-stage-v3",
                "status": "done",
                "exit_code": 0,
                "cwd": str(root.resolve()),
                "command": ["bash", "scripts/run_cecd_three_stage_v3.sh"],
            }
        ),
        encoding="utf-8",
    )
    fingerprints = {family: _fingerprint(family) for family in ("huatuo", "hulu")}
    models = {record["model_id"]: {} for record in fingerprints.values()}
    stage = {
        "models": models,
        "gate": {
            "confirmation_passing_models": list(models),
            "authorized_for_method_level_treble_adapter_run": True,
            "authorized_for_hidden_state_stage": False,
        },
    }
    gate = {
        "version": builder.THREE_STAGE_GATE_VERSION,
        "status": "passed",
        "passed": True,
        "authorized_for_method_level_treble_adapter_run": True,
        "hidden_state_authorized": False,
        "legacy_pilot_as_dev_authorized": False,
        "admission": {
            "path": str(admission.resolve()),
            "sha256": builder.sha256_file(admission),
        },
        "confirmation_locked": {
            "path": str(stage_analysis.resolve()),
            "sha256": builder.sha256_file(stage_analysis),
        },
        "runs": {
            "confirmation_locked": [
                {
                    "family": family,
                    "model": fingerprints[family]["model_id"],
                    "model_provenance_sha256": str(index) * 64,
                }
                for index, family in enumerate(("huatuo", "hulu"), 7)
            ]
        },
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    rows = {}
    for stage_label in ("dev_fit", "confirmation_locked"):
        rows[stage_label] = [
            {
                "image_id": f"{stage_label}_image_{index}",
                "finding": "pleural_effusion",
                "positive_votes": index % 4,
                "dicom_relpath": f"train/{stage_label}_image_{index}.dicom",
            }
            for index in range(30)
        ]
    monkeypatch.setattr(builder, "selection", lambda stage_label: rows[stage_label])
    monkeypatch.setattr(
        builder,
        "EXPECTED_SELECTION_HASHES",
        {
            stage_label: builder.canonical_json_sha256(
                [builder._record_key(row) for row in stage_rows]
            )
            for stage_label, stage_rows in rows.items()
        },
    )
    monkeypatch.setattr(
        builder.cecd_monitor, "validate_stage_result", lambda **_: stage
    )
    monkeypatch.setattr(
        builder,
        "compute_model_fingerprint",
        lambda family, _path: fingerprints[family],
    )
    huatuo_model = root / "models/HuatuoGPT-Vision-7B"
    hulu_model = root / "models/Hulu-Med-4B"
    source = root / "HuatuoGPT-Vision"
    for path in (huatuo_model, hulu_model, source):
        path.mkdir(parents=True)
    return {
        "stage_state": state,
        "stage_analysis": stage_analysis,
        "stage_input_gate": gate_path,
        "admission": admission,
        "preflight_path": root / "configs/cecd_dual_semantics_preflight_v1.json",
        "build_receipt": authority / "cecd_dual_semantics_v1/preflight_build.json",
        "input_root": root / "configs/cecd_dual_semantics_inputs_v1",
        "method_output_root": authority / "cecd_dual_semantics_v1/method_outputs",
        "huatuo_model": huatuo_model,
        "hulu_model": hulu_model,
        "huatuo_source_root": source,
        "root": root,
    }


def test_builder_closes_preflight_sidecar_receipt_and_replays(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result = builder.build_preflight(**paths)
    assert result["status"] == "complete_outcome_blind_no_model_execution"
    assert result["raw_model_outcome_rows_consumed_by_builder"] is False
    assert result["gpu_or_model_execution_performed"] is False
    assert result["canonical_gpu_lock"].endswith("gpu0-vindr-v2.lock")
    preflight = json.loads(paths["preflight_path"].read_text())
    assert set(preflight["model_fingerprints"]) == {"huatuo", "hulu"}
    assert preflight["evaluation_split"] == "locked_test"
    assert preflight["compute_ledger"]["huatuo"]["target_examples"] == 30
    sidecar = paths["preflight_path"].with_name(
        f"{paths['preflight_path'].stem}.inputs.json"
    )
    assert json.loads(sidecar.read_text())["preflight_sha256"] == builder.sha256_file(
        paths["preflight_path"]
    )
    assert builder.build_preflight(**paths) == result


def test_builder_receipt_rejects_authority_drift(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    builder.build_preflight(**paths)
    paths["admission"].write_text(json.dumps({"genuine": False}), encoding="utf-8")
    with pytest.raises(builder.PreflightBuildError, match="admission binding drift"):
        builder.build_preflight(**paths)


def test_builder_fails_before_model_fingerprint_on_noncanonical_state_or_no_go(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    paths["stage_state"].write_text(
        json.dumps({"name": "legacy", "status": "done", "exit_code": 0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        builder,
        "compute_model_fingerprint",
        lambda *_: (_ for _ in ()).throw(AssertionError("model fingerprint touched")),
    )
    with pytest.raises(builder.PreflightBuildError, match="not a successful canonical"):
        builder.build_preflight(**paths)

    paths = _fixture(tmp_path / "no_go", monkeypatch)
    monkeypatch.setattr(
        builder.cecd_monitor,
        "validate_stage_result",
        lambda **_: {
            "models": {"huatuo:HuatuoGPT-Vision-7B": {}, "hulu:Hulu-Med-4B": {}},
            "gate": {
                "confirmation_passing_models": [],
                "authorized_for_method_level_treble_adapter_run": False,
                "authorized_for_hidden_state_stage": False,
            },
        },
    )
    with pytest.raises(builder.PreflightBuildError, match="not a two-model method GO"):
        builder.build_preflight(**paths)
