import json
from pathlib import Path

import pytest

import anchor.corrected_sgta.verify_cecd_three_stage_v3 as module
from anchor.corrected_sgta.analyze_clinical_equivalence_composition_defect_v1 import (
    CONFIRMATION_VERSION,
    DEV_FIT_VERSION,
)
from anchor.corrected_sgta.run_cecd_factorial_v1 import STAGE_SPECS


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _passing_model_result() -> dict:
    findings = ("aortic", "cardiomegaly", "effusion", "fibrosis")
    per_finding = {
        finding: {
            "delta_auc": 0.01,
            "image_cluster_bootstrap": {"delta_auc_ci95": [-0.01, 0.03]},
            "harmful_alignment": {"point": 0.1, "ci95": [0.01, 0.2]},
        }
        for finding in findings
    }
    components = {
        "pooled_delta_auc_point_at_least_0p03_and_ci_above_zero": True,
        "pooled_harmful_alignment_ci_above_zero": True,
        "interaction_rms_at_least_0p25_re_and_ci_above_zero": True,
        "identity_below_one_tenth": True,
        "all_reader_slopes_ci_above_zero": True,
        "heterogeneity_guard": True,
    }
    return {
        "pooled_four_finding_delta_auc": {
            "delta_auc": 0.04,
            "image_cluster_bootstrap": {"delta_auc_ci95": [0.01, 0.07]},
        },
        "pooled_harmful_alignment": {"point": 0.1, "ci95": [0.01, 0.2]},
        "interaction_rms_reader_equivalents": {
            "point": 0.3,
            "ci95": [0.2, 0.4],
        },
        "identity_controls": {"maximum_rms_re": 0.01, "below_one_tenth": True},
        "reader_slope_cluster_bootstrap": {
            finding: {"point": 1.0, "ci95": [0.5, 1.5]}
            for finding in findings
        },
        "per_finding": per_finding,
        "heterogeneity_guard": {
            "delta_positive_findings": 4,
            "harmful_alignment_positive_findings": 4,
            "no_finding_delta_at_or_below_minus_0p03": True,
            "no_finding_ci_strictly_below_zero": True,
            "passed": True,
        },
        "gate_components": components,
        "model_confirmation_pass": True,
    }


def _fixture(
    tmp_path: Path, monkeypatch, *, overlap: bool = False,
    model_drift: bool = False, analysis_code_drift: bool = False,
    empty_model_metrics: bool = False,
):
    admission = tmp_path / "admission.json"
    admission.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        module, "require_cecd_authorization", lambda _: {"version": "admission-v1"}
    )
    run_dirs = {
        family: {stage: tmp_path / family / stage for stage in STAGE_SPECS}
        for family in ("huatuo", "hulu")
    }
    fake_runs = {}
    for stage in STAGE_SPECS:
        fake_runs[stage] = []
        for family in ("huatuo", "hulu"):
            rows = run_dirs[family][stage] / "factorial_rows.jsonl"
            rows.parent.mkdir(parents=True, exist_ok=True)
            rows.write_text(f"{family}-{stage}\n", encoding="utf-8")
            images = [f"{stage}-image"]
            if overlap and stage == "confirmation_locked":
                images = ["dev_fit-image"]
            fake_runs[stage].append(
                {
                    "family": family,
                    "stage": stage,
                    "run_dir": str(run_dirs[family][stage].resolve()),
                    "model": f"{family}:frozen",
                    "claims": STAGE_SPECS[stage]["claims"],
                    "rows": STAGE_SPECS[stage]["claims"] * 19,
                    "factorial_rows": str(rows.resolve()),
                    "factorial_rows_sha256": module.sha256_file(rows),
                    "selection_keys_sha256": module.EXPECTED_SELECTION_HASHES[stage],
                    "image_ids": images,
                    "config_sha256": family[0] * 64,
                    "admission_sha256": module.sha256_file(admission),
                    "model_provenance_sha256": (
                        "f" * 64 if model_drift and family == "huatuo"
                        and stage == "confirmation_locked" else family[-1] * 64
                    ),
                    "next_token_conformance": {"sha256": stage[0] * 64},
                    "scientific_contract_sha256": stage[0] * 64,
                }
            )

    def fake_verify(*, family, stage, run_dir, admission):
        return next(row for row in fake_runs[stage] if row["family"] == family)

    monkeypatch.setattr(module, "verify_stage_run", fake_verify)
    models = {
        "huatuo:frozen": {} if empty_model_metrics else _passing_model_result(),
        "hulu:frozen": {} if empty_model_metrics else _passing_model_result(),
    }
    dev_inputs = {
        row["factorial_rows"]: row["factorial_rows_sha256"]
        for row in fake_runs["dev_fit"]
    }
    dev_fit_path = tmp_path / "dev_fit.json"
    _json(
        dev_fit_path,
        {
            "version": DEV_FIT_VERSION,
            "status": "dev_fit_complete_confirmation_not_opened",
            "stage_label": "dev_fit",
            "source_manifest_split": "dev",
            "models": models,
            "gate": {"authorized_for_method_level_treble_adapter_run": False},
            "provenance": {
                "input_sha256": dev_inputs,
                "code_sha256": (
                    "0" * 64 if analysis_code_drift
                    else module.sha256_file(module.ANALYZER_SOURCE)
                ),
                "mode": "dev_fit",
                "seed": 42,
                "folds": 5,
                "bootstrap_draws": 5000,
            },
        },
    )
    confirmation_inputs = {
        row["factorial_rows"]: row["factorial_rows_sha256"]
        for row in fake_runs["confirmation_locked"]
    }
    confirmation_path = tmp_path / "confirmation.json"
    _json(
        confirmation_path,
        {
            "version": CONFIRMATION_VERSION,
            "status": "complete",
            "stage_label": "confirmation_locked",
            "source_manifest_split": "confirmation",
            "models": models,
            "dev_fit_binding": {
                "path": str(dev_fit_path.resolve()),
                "sha256": module.sha256_file(dev_fit_path),
            },
            "gate": {
                "name": "behavioral_confirmation_locked_v1",
                "confirmation_passing_models": list(models),
                "both_models_pass": True,
                "authorized_for_method_level_treble_adapter_run": True,
                "authorized_for_hidden_state_stage": False,
                "behavioral_phenomenon_confirmed_on_locked_test": True,
            },
            "provenance": {
                "input_sha256": confirmation_inputs,
                "code_sha256": module.sha256_file(module.ANALYZER_SOURCE),
                "mode": "confirmation_locked",
                "seed": 42,
                "folds": 5,
                "bootstrap_draws": 5000,
            },
        },
    )
    monkeypatch.setattr(
        module,
        "_recompute_three_stage_analysis",
        lambda **_: json.loads(confirmation_path.read_text(encoding="utf-8")),
    )
    return admission, run_dirs, dev_fit_path, confirmation_path


def test_three_stage_verifier_binds_locked_result_and_revokes_legacy(
    tmp_path: Path, monkeypatch
) -> None:
    admission, runs, dev, confirmation = _fixture(tmp_path, monkeypatch)
    result = module.verify_three_stage(
        admission=admission, run_dirs=runs, dev_fit_path=dev,
        confirmation_path=confirmation, root=tmp_path,
    )
    assert result["passed"] is True
    assert result["legacy_pilot_as_dev_authorized"] is False
    assert result["authorized_for_method_level_treble_adapter_run"] is True
    assert result["hidden_state_authorized"] is False
    assert result["legacy_v1_v3_artifacts_authorized"] is False
    assert result["confirmation_locked"]["scientific_gate_authority"] == (
        "independent_raw_input_recomputation"
    )
    assert not any(result["whole_image_overlap_counts"].values())


def test_three_stage_verifier_rejects_whole_image_overlap(
    tmp_path: Path, monkeypatch
) -> None:
    admission, runs, dev, confirmation = _fixture(
        tmp_path, monkeypatch, overlap=True
    )
    with pytest.raises(RuntimeError, match="whole-image stage leakage"):
        module.verify_three_stage(
            admission=admission, run_dirs=runs, dev_fit_path=dev,
            confirmation_path=confirmation, root=tmp_path,
        )


def test_three_stage_verifier_rejects_model_weight_drift(
    tmp_path: Path, monkeypatch
) -> None:
    admission, runs, dev, confirmation = _fixture(
        tmp_path, monkeypatch, model_drift=True
    )
    with pytest.raises(RuntimeError, match="model weights change across stages"):
        module.verify_three_stage(
            admission=admission, run_dirs=runs, dev_fit_path=dev,
            confirmation_path=confirmation, root=tmp_path,
        )


def test_three_stage_verifier_rejects_analyzer_code_drift(
    tmp_path: Path, monkeypatch
) -> None:
    admission, runs, dev, confirmation = _fixture(
        tmp_path, monkeypatch, analysis_code_drift=True
    )
    with pytest.raises(RuntimeError, match="analysis code or frozen execution"):
        module.verify_three_stage(
            admission=admission, run_dirs=runs, dev_fit_path=dev,
            confirmation_path=confirmation, root=tmp_path,
        )


def test_three_stage_verifier_rejects_empty_metrics_with_asserted_true_gate(
    tmp_path: Path, monkeypatch
) -> None:
    admission, runs, dev, confirmation = _fixture(
        tmp_path, monkeypatch, empty_model_metrics=True
    )
    with pytest.raises(RuntimeError, match="model metrics are incomplete"):
        module.verify_three_stage(
            admission=admission,
            run_dirs=runs,
            dev_fit_path=dev,
            confirmation_path=confirmation,
            root=tmp_path,
        )


def test_independent_gate_rejects_asserted_component_that_disagrees_with_metrics() -> None:
    confirmation = {
        "models": {
            "huatuo:frozen": _passing_model_result(),
            "hulu:frozen": _passing_model_result(),
        },
        "gate": {
            "name": "behavioral_confirmation_locked_v1",
            "confirmation_passing_models": ["huatuo:frozen", "hulu:frozen"],
            "both_models_pass": True,
            "authorized_for_method_level_treble_adapter_run": True,
            "authorized_for_hidden_state_stage": False,
            "behavioral_phenomenon_confirmed_on_locked_test": True,
        },
    }
    confirmation["models"]["huatuo:frozen"]["gate_components"][
        "identity_below_one_tenth"
    ] = False
    with pytest.raises(RuntimeError, match="gate components disagree"):
        module._validate_recomputed_confirmation_gate(
            confirmation, {"huatuo:frozen", "hulu:frozen"}
        )


def test_exact_recomputation_rejects_tampered_artifact() -> None:
    recomputed = {"version": "v", "models": {"huatuo": {"pass": False}}}
    artifact = {"version": "v", "models": {"huatuo": {"pass": True}}}
    with pytest.raises(RuntimeError, match="independent raw-input recomputation"):
        module._require_exact_recomputation(
            label="locked-confirmation",
            artifact=artifact,
            recomputed=recomputed,
            ignored=set(),
        )


def test_recomputation_consumes_bound_raw_inputs_and_rebuilds_both_stages(
    tmp_path: Path, monkeypatch
) -> None:
    dev_paths = [tmp_path / "huatuo-dev.jsonl", tmp_path / "hulu-dev.jsonl"]
    confirmation_paths = [
        tmp_path / "huatuo-confirmation.jsonl",
        tmp_path / "hulu-confirmation.jsonl",
    ]
    for path in dev_paths + confirmation_paths:
        path.write_text("bound raw input\n", encoding="utf-8")
    calls = []

    def fake_load(paths):
        calls.append(("load", [Path(path) for path in paths]))
        return {"stage": "dev" if "dev" in paths[0].name else "confirmation"}

    def fake_fit(payload, *, folds, draws, seed):
        calls.append(("fit", payload, folds, draws, seed))
        return {"version": DEV_FIT_VERSION, "models": {"frozen": True}}

    def fake_apply(payload, dev_fit, *, draws, seed):
        calls.append(("apply", payload, dev_fit, draws, seed))
        return {
            "version": CONFIRMATION_VERSION,
            "models": {"huatuo": {"model_confirmation_pass": False}},
        }

    monkeypatch.setattr(module, "load_inputs", fake_load)
    monkeypatch.setattr(module, "fit_dev_stage", fake_fit)
    monkeypatch.setattr(module, "apply_confirmation_stage", fake_apply)
    runs = {
        "dev_fit": [{"factorial_rows": str(path)} for path in dev_paths],
        "confirmation_locked": [
            {"factorial_rows": str(path)} for path in confirmation_paths
        ],
    }
    dev_artifact = {
        "version": DEV_FIT_VERSION,
        "models": {"frozen": True},
        "provenance": {"ignored": True},
    }
    confirmation_artifact = {
        "version": CONFIRMATION_VERSION,
        "models": {"huatuo": {"model_confirmation_pass": False}},
        "provenance": {"ignored": True},
        "dev_fit_binding": {"ignored": True},
    }
    result = module._recompute_three_stage_analysis(
        runs=runs,
        dev_fit=dev_artifact,
        confirmation=confirmation_artifact,
    )
    assert result["version"] == CONFIRMATION_VERSION
    assert calls == [
        ("load", dev_paths),
        ("fit", {"stage": "dev"}, 5, 5000, 42),
        ("load", confirmation_paths),
        (
            "apply",
            {"stage": "confirmation"},
            {"version": DEV_FIT_VERSION, "models": {"frozen": True}},
            5000,
            42,
        ),
    ]


def test_legacy_v1_dev_fit_is_readable_json_but_cannot_authorize(
    tmp_path: Path, monkeypatch
) -> None:
    admission, runs, dev, confirmation = _fixture(tmp_path, monkeypatch)
    payload = json.loads(dev.read_text())
    payload["version"] = "clinical-equivalence-composition-defect-dev-fit-v1"
    dev.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="dev-fit artifact contract mismatch"):
        module.verify_three_stage(
            admission=admission,
            run_dirs=runs,
            dev_fit_path=dev,
            confirmation_path=confirmation,
            root=tmp_path,
        )
