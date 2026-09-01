from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from anchor.corrected_sgta.cecd_dual_semantics_kernels_v1 import (
    FactorialComponents,
    FactorialKernelError,
    IMPLEMENTED_KERNEL_METHODS,
    apply_factorial_control,
    factorial_components,
    reconstruct_cells,
    synthetic_adapter_conformance,
)
import anchor.corrected_sgta.cecd_dual_semantics_worker_v1 as worker
import anchor.corrected_sgta.run_cecd_dual_semantics_controlled_v1 as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_model(root: Path, family: str) -> Path:
    model = root / f"{family}-model"
    model.mkdir(parents=True)
    common = {
        "tokenizer.json": "{}",
        "tokenizer_config.json": json.dumps({"chat_template": f"{family}-chat"}),
        "model-00001-of-00001.safetensors": "tiny-weights",
    }
    for name, text in common.items():
        (model / name).write_text(text)
    if family == "huatuo":
        (model / "config.json").write_text(json.dumps({"num_hidden_layers": 28}))
        vision = model / "vit/clip_vit_large_patch14_336"
        vision.mkdir(parents=True)
        (vision / "config.json").write_text(json.dumps({"num_hidden_layers": 24}))
        (vision / "preprocessor_config.json").write_text("{}")
    else:
        (model / "config.json").write_text(
            json.dumps(
                {
                    "num_hidden_layers": 36,
                    "vision_encoder_config": {"num_hidden_layers": 27},
                }
            )
        )
        for name in (
            "chat_template.json",
            "preprocessor_config.json",
            "processing_hulumed.py",
            "image_processing_hulumed.py",
            "modeling_hulumed_qwen3.py",
        ):
            (model / name).write_text(f"{name}\n")
    return model


def _descriptor_fixture(tmp_path: Path) -> dict:
    models = {
        family: _tiny_model(tmp_path, family) for family in ("huatuo", "hulu")
    }
    huatuo_source = tmp_path / "huatuo-source"
    huatuo_source.mkdir()
    (huatuo_source / "runtime.py").write_text("RUNTIME = 'tiny'\n")
    inputs = {}
    for name in worker.INPUT_NAMES:
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps({"name": name}) + "\n")
        inputs[name] = path
    fingerprints = {
        family: worker.compute_model_fingerprint(family, path)
        for family, path in models.items()
    }
    preflight = {
        "model_fingerprints": fingerprints,
        "calibration_manifest_sha256": _sha(inputs["calibration_manifest"]),
        "evaluation_manifest_sha256": _sha(inputs["evaluation_manifest"]),
        "record_keys_sha256": _sha(inputs["record_keys"]),
        "claim_contract_sha256": _sha(inputs["claim_contract"]),
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight))
    sidecar = {
        "schema_version": worker.INPUT_SIDECAR_SCHEMA,
        "preflight_sha256": _sha(preflight_path),
        "model_dirs": {family: str(path) for family, path in models.items()},
        "huatuo_source_root": str(huatuo_source),
        "input_bindings": {name: str(path) for name, path in inputs.items()},
    }
    worker._sidecar_path(preflight_path).write_text(json.dumps(sidecar))
    return {
        "models": models,
        "inputs": inputs,
        "preflight": preflight_path,
        "fingerprints": fingerprints,
    }


def test_factorial_decomposition_and_seven_closed_form_controls() -> None:
    parts = FactorialComponents(
        grand=np.full((2, 8), 1.0),
        render=np.full((2, 8), 2.0),
        prompt=np.full((2, 8), 3.0),
        interaction=np.arange(16, dtype=float).reshape(2, 8) / 10.0,
    )
    orbit = reconstruct_cells(parts)
    recovered = factorial_components(orbit)
    for name in ("grand", "render", "prompt", "interaction"):
        np.testing.assert_allclose(getattr(recovered, name), getattr(parts, name))
    np.testing.assert_allclose(apply_factorial_control(orbit, "unmitigated"), orbit["h11"])
    np.testing.assert_allclose(apply_factorial_control(orbit, "full_orbit"), parts.grand)
    np.testing.assert_allclose(
        apply_factorial_control(orbit, "render_only"), (orbit["h01"] + orbit["h11"]) / 2
    )
    np.testing.assert_allclose(
        apply_factorial_control(orbit, "prompt_only"), (orbit["h10"] + orbit["h11"]) / 2
    )
    np.testing.assert_allclose(
        apply_factorial_control(orbit, "main_effect_removal"),
        parts.grand + parts.interaction,
    )
    assert set(IMPLEMENTED_KERNEL_METHODS) == {
        "unmitigated",
        "full_orbit",
        "render_only",
        "prompt_only",
        "random_norm",
        "sign_permuted",
        "main_effect_removal",
    }


def test_random_and_permuted_controls_match_only_interaction_energy() -> None:
    rng = np.random.default_rng(7)
    parts = FactorialComponents(*(rng.normal(size=(4, 32)) for _ in range(4)))
    orbit = reconstruct_cells(parts)
    base = parts.grand + parts.render + parts.prompt
    random = apply_factorial_control(orbit, "random_norm", seed=11) - base
    permuted = apply_factorial_control(orbit, "sign_permuted", seed=11) - base
    target_norm = np.linalg.norm(parts.interaction, axis=-1)
    np.testing.assert_allclose(np.linalg.norm(random, axis=-1), target_norm, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(permuted, axis=-1), target_norm, atol=1e-12)
    np.testing.assert_allclose(np.sum(random * parts.interaction, axis=-1), 0.0, atol=1e-12)
    np.testing.assert_array_equal(
        apply_factorial_control(orbit, "random_norm", seed=11),
        apply_factorial_control(orbit, "random_norm", seed=11),
    )


def test_kernels_fail_closed_on_bad_orbit_or_unimplemented_method() -> None:
    with pytest.raises(FactorialKernelError, match="exactly"):
        factorial_components({"h00": np.ones(2)})
    orbit = {key: np.ones(3) for key in ("h00", "h10", "h01", "h11")}
    orbit["h11"] = np.ones(4)
    with pytest.raises(FactorialKernelError, match="equal"):
        factorial_components(orbit)
    valid = {key: np.ones(3) for key in ("h00", "h10", "h01", "h11")}
    with pytest.raises(FactorialKernelError, match="not implemented"):
        apply_factorial_control(valid, "treble_proceedings")
    with pytest.raises(FactorialKernelError, match="dimension"):
        apply_factorial_control(
            {key: np.ones((2, 1)) for key in valid}, "random_norm"
        )


def test_synthetic_adapter_conformance_is_cpu_only_and_not_scientific_output() -> None:
    result = synthetic_adapter_conformance(42)
    assert result["passed"] is True
    assert result["gpu_used"] is False
    assert result["scientific_model_output"] is False
    assert result["methods"] == list(IMPLEMENTED_KERNEL_METHODS)


def test_real_descriptor_binds_model_inputs_source_and_cpu_runtime(tmp_path: Path) -> None:
    fixture = _descriptor_fixture(tmp_path)
    for family in ("huatuo", "hulu"):
        descriptor = worker.describe_runtime(
            family=family, preflight_path=fixture["preflight"]
        )
        assert descriptor["schema_version"] == worker.RUNTIME_DESCRIPTOR_SCHEMA
        assert descriptor["model_fingerprint"] == fixture["fingerprints"][family]
        assert descriptor["python_executable"] == str(Path(sys.executable).resolve())
        assert descriptor["runtime_versions"]["cuda_initialized"] == (
            "false_descriptor_is_cpu_only"
        )
        assert set(descriptor["input_bindings"]) == set(worker.INPUT_NAMES)
        source_paths = [row["path"] for row in descriptor["source_files"]]
        assert source_paths.count(str(Path(worker.__file__).resolve())) == 1
        assert all(Path(row["path"]).is_file() for row in descriptor["source_files"])


def test_descriptor_rejects_input_model_or_sidecar_drift(tmp_path: Path) -> None:
    fixture = _descriptor_fixture(tmp_path)
    fixture["inputs"]["claim_contract"].write_text("drift\n")
    with pytest.raises(worker.ScientificWorkerError, match="claim_contract hash"):
        worker.describe_runtime(family="huatuo", preflight_path=fixture["preflight"])

    fixture = _descriptor_fixture(tmp_path / "model")
    (fixture["models"]["hulu"] / "model-00001-of-00001.safetensors").write_text(
        "weight drift"
    )
    with pytest.raises(worker.ScientificWorkerError, match="model fingerprint"):
        worker.describe_runtime(family="hulu", preflight_path=fixture["preflight"])

    fixture = _descriptor_fixture(tmp_path / "sidecar")
    sidecar_path = worker._sidecar_path(fixture["preflight"])
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["preflight_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(worker.ScientificWorkerError, match="current preflight"):
        worker.describe_runtime(family="huatuo", preflight_path=fixture["preflight"])


def test_formal_methods_all_fail_before_output_or_model_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    authorization = tmp_path / "authorization.json"
    preflight = tmp_path / "preflight.json"
    run_contract = tmp_path / "run_contract.json"
    for path in (authorization, preflight):
        path.write_text("{}")
    contract = {
        "authorization_fingerprint": "a" * 64,
        "preflight": {"sha256": _sha(preflight)},
        "method_output_root": str((tmp_path / "outputs").resolve()),
        "models": {"huatuo": {}, "hulu": {}},
        "arm_order": [
            {"model_family": family, "method": method}
            for family in ("huatuo", "hulu")
            for method in (*IMPLEMENTED_KERNEL_METHODS, "treble_proceedings")
        ],
    }
    contract["fingerprint"] = worker.canonical_sha256(contract)
    run_contract.write_text(json.dumps(contract))
    monkeypatch.setattr(
        runner,
        "validate_authorization_and_preflight",
        lambda **_: (
            {"fingerprint": "a" * 64},
            {
                "methods": [
                    *IMPLEMENTED_KERNEL_METHODS,
                    "cecd_interaction_projection",
                    "treble_proceedings",
                    "treble_released",
                ]
            },
            (tmp_path / "outputs").resolve(),
        ),
    )
    for method in (
        *IMPLEMENTED_KERNEL_METHODS,
        "cecd_interaction_projection",
        "treble_proceedings",
        "treble_released",
    ):
        output = tmp_path / "outputs/partial/huatuo" / method
        with pytest.raises(worker.MethodNotImplementedError):
            worker.validate_formal_request(
                authorization=authorization,
                preflight=preflight,
                run_contract=run_contract,
                family="huatuo",
                method=method,
                output_dir=output,
            )
        assert not output.exists()

    for method in IMPLEMENTED_KERNEL_METHODS:
        output = tmp_path / "outputs/partial_ce/huatuo" / method
        context = worker.validate_formal_request(
            authorization=authorization,
            preflight=preflight,
            run_contract=run_contract,
            family="huatuo",
            method=method,
            output_dir=output,
            task="ce",
        )
        assert context["run_contract"]["fingerprint"] == contract["fingerprint"]
        assert not output.exists()

    with pytest.raises(worker.MethodNotImplementedError):
        worker.validate_formal_request(
            authorization=authorization,
            preflight=preflight,
            run_contract=run_contract,
            family="huatuo",
            method="treble_proceedings",
            output_dir=tmp_path / "outputs/partial_ce/huatuo/treble_proceedings",
            task="ce",
        )
