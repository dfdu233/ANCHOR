import json
import hashlib
from pathlib import Path

import pytest

from anchor.corrected_sgta.run_cecd_factorial_v1 import (
    FROZEN_PER_BIN,
    FROZEN_SEED,
    FROZEN_VOTES,
    IDENTITY_RENDER_NAME,
    MEASUREMENT_NAME,
    PROMPT_TEMPLATES,
    SCIENCE_RENDER_NAMES,
    VERSION,
    FROZEN_FINDINGS,
    canonical_json_sha256,
    cell_specs,
    full_model_artifact_fingerprint,
    python_source_tree_fingerprint,
    sha256_file,
)
import anchor.corrected_sgta.verify_cecd_two_model_stage1_v2 as gate

from anchor.corrected_sgta.verify_cecd_two_model_stage1_v2 import verify_run


def _run(tmp_path: Path, family: str, admission: Path) -> Path:
    run = tmp_path / family
    run.mkdir(parents=True)
    model_dir = tmp_path / f"{family}_model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    external_runtime = None
    if family == "huatuo":
        runtime_root = tmp_path / "huatuo_runtime"
        runtime_root.mkdir(exist_ok=True)
        (runtime_root / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
        external_runtime = python_source_tree_fingerprint(runtime_root)
    config = {
        "version": VERSION,
        "measurement_name": MEASUREMENT_NAME,
        "dataset": "vindr-cxr-1.0.0-fixed-three-reader-panel",
        "manifest_sha256": "1" * 64,
        "bboxes_sha256": "2" * 64,
        "split": "pilot",
        "model_family": family,
        "model": f"{family}:fixture",
        "model_dir": str(model_dir.resolve()),
        "model_provenance": {
            "mode": "full_content_hash_including_all_weight_shards",
            "full_fingerprint": full_model_artifact_fingerprint(model_dir),
            "external_runtime_source": external_runtime,
        },
        "scientific_status": "human_admitted_behavioral_dev_screen",
        "clinical_equivalence_established": True,
        "execution_mode": "formal_human_admitted_model_scoring",
        "cecd_model_scoring_authorized": True,
        "scientific_artifact_authorized": True,
        "engineering_canary_max_claims": None,
        "active_claim_count": 160,
        "findings": list(FROZEN_FINDINGS),
        "votes": list(FROZEN_VOTES),
        "per_finding_vote_bin": FROZEN_PER_BIN,
        "seed": FROZEN_SEED,
        "frozen_claim_count": 160,
        "frozen_selection_keys_sha256": "3" * 64,
        "active_selection_keys_sha256": "3" * 64,
        "science_render_names": list(SCIENCE_RENDER_NAMES),
        "identity_render_name": IDENTITY_RENDER_NAME,
        "prompt_templates": [
            {"name": name, "template": template} for name, template in PROMPT_TEMPLATES
        ],
        "prompt_contract": {"frozen": True},
        "cells_per_claim": {
            "science": 15,
            "identity_image_controls": 3,
            "duplicate_prompt_controls": 1,
        },
        "missing_cell_policy": "record missing_invalid_render; never substitute baseline pixels or scores",
        "readout": "FP32 Yes/No/Maybe at exact next-token position",
        "next_token_conformance": {"frozen": True},
        "source_sha256": {"factorial_runner": "4" * 64},
        "clinical_admission": {
            "status": "passed_hash_bound",
            "analysis_path": str(admission.resolve()),
            "analysis_sha256": sha256_file(admission),
            "analysis_version": gate.ADMISSION_ANALYSIS_VERSION,
            "cecd_model_scoring_authorized": True,
        },
        "created_at": "ignored",
        "command": "ignored",
    }
    immutable = {
        key: value for key, value in config.items() if key not in {"created_at", "command"}
    }
    config["fingerprint"] = canonical_json_sha256(immutable)
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    config["next_token_conformance"] = {
        "required_before_scientific_scoring": True,
        "centered_tristate_logit_tolerance": 0.1,
        "choice_must_match": True,
    }
    immutable = {
        key: value for key, value in config.items() if key not in {"created_at", "command", "fingerprint"}
    }
    config["fingerprint"] = canonical_json_sha256(immutable)
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "next_token_conformance.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "config_fingerprint": config["fingerprint"],
                "model": config["model"],
                "render_id": "baseline_percentile",
                "prompt_id": "existential",
                "passed": True,
                "centered_tristate_max_abs_error": 0.01,
                "tolerance": 0.1,
                "direct_tristate_choice": "supported",
                "standard_tristate_choice": "supported",
                "direct_logits": {"supported": 2.0, "refuted": 1.0, "undetermined": 0.0},
                "standard_generation_logits": {"supported": 2.01, "refuted": 1.0, "undetermined": 0.0},
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for finding in FROZEN_FINDINGS:
        for index in range(40):
            for spec in cell_specs(finding):
                rows.append(
                    {
                        "contract_version": "clinical-equivalence-factorial-v1",
                        "config_fingerprint": config["fingerprint"],
                        "model": config["model"],
                        "image_id": f"{finding}-{index:02d}",
                        "finding": finding,
                        "render_id": spec.render_name,
                        "prompt_id": spec.prompt_name,
                        "cell_id": spec.cell_id,
                        "cell_role": spec.role,
                        "reference_cell_id": spec.reference_cell_id,
                        "prompt_text_sha256": hashlib.sha256(
                            spec.prompt_text.encode()
                        ).hexdigest(),
                        "status": "ok",
                    }
                )
    rows_path = run / "factorial_rows.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (run / "factorial_rows_manifest.json").write_text(
        json.dumps(
            {
                "config_fingerprint": config["fingerprint"],
                "factorial_rows_sha256": sha256_file(rows_path),
                "claims": 160,
                "rows": 3040,
                "complete_orbit_count": 160,
                "incomplete_orbit_count": 0,
            }
        ),
        encoding="utf-8",
    )
    return run


def test_two_model_input_unit_accepts_only_full_hash_bound_run(tmp_path: Path) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text("admitted", encoding="utf-8")
    run = _run(tmp_path, "huatuo", admission)
    result = verify_run(family="huatuo", run_dir=run, admission=admission)
    assert result["claims"] == 160
    assert result["rows"] == 3040
    assert result["admission_sha256"] == sha256_file(admission)

    admission.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="admission binding mismatch"):
        verify_run(family="huatuo", run_dir=run, admission=admission)


def test_input_gate_requires_hash_bound_next_token_conformance(tmp_path: Path) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text("admitted", encoding="utf-8")
    run = _run(tmp_path, "huatuo", admission)
    conformance = run / "next_token_conformance.json"
    conformance.unlink()
    with pytest.raises(RuntimeError, match="conformance artifact is missing"):
        verify_run(family="huatuo", run_dir=run, admission=admission)

    run = _run(tmp_path / "second", "huatuo", admission)
    conformance = run / "next_token_conformance.json"
    payload = json.loads(conformance.read_text())
    payload["passed"] = False
    conformance.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="conformance contract mismatch"):
        verify_run(family="huatuo", run_dir=run, admission=admission)


def test_input_gate_rehashes_executable_model_runtime(tmp_path: Path) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text("admitted", encoding="utf-8")
    run = _run(tmp_path, "hulu", admission)
    config = json.loads((run / "config.json").read_text())
    (Path(config["model_dir"]) / "config.json").write_text('{"drift": true}')
    with pytest.raises(RuntimeError, match="runtime asset hash drift"):
        verify_run(family="hulu", run_dir=run, admission=admission)


def test_input_gate_rejects_duplicate_cell_hidden_behind_valid_row_count(
    tmp_path: Path,
) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text("admitted", encoding="utf-8")
    run = _run(tmp_path, "huatuo", admission)
    rows_path = run / "factorial_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0]["cell_id"] = rows[1]["cell_id"]
    rows[0]["render_id"] = rows[1]["render_id"]
    rows[0]["prompt_id"] = rows[1]["prompt_id"]
    rows[0]["cell_role"] = rows[1]["cell_role"]
    rows[0]["reference_cell_id"] = rows[1]["reference_cell_id"]
    rows[0]["prompt_text_sha256"] = rows[1]["prompt_text_sha256"]
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = run / "factorial_rows_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["factorial_rows_sha256"] = sha256_file(rows_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="duplicate cell"):
        verify_run(family="huatuo", run_dir=run, admission=admission)


def test_input_gate_rejects_non_scientific_or_engineering_run_flags(
    tmp_path: Path,
) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text("admitted", encoding="utf-8")
    run = _run(tmp_path, "huatuo", admission)
    config_path = run / "config.json"
    config = json.loads(config_path.read_text())
    config["scientific_artifact_authorized"] = False
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="formal execution authorization mismatch"):
        verify_run(family="huatuo", run_dir=run, admission=admission)


def test_two_model_join_requires_one_shared_scientific_contract(
    tmp_path: Path, monkeypatch
) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text("admitted", encoding="utf-8")
    monkeypatch.setattr(
        gate,
        "require_cecd_authorization",
        lambda _path: {"version": gate.ADMISSION_ANALYSIS_VERSION},
    )
    calls = iter(
        [
            {
                "family": "huatuo",
                "model": "huatuo:a",
                "admission_sha256": sha256_file(admission),
                "scientific_contract_sha256": "1" * 64,
            },
            {
                "family": "hulu",
                "model": "hulu:b",
                "admission_sha256": sha256_file(admission),
                "scientific_contract_sha256": "2" * 64,
            },
        ]
    )
    monkeypatch.setattr(gate, "verify_run", lambda **_kwargs: next(calls))
    with pytest.raises(RuntimeError, match="do not share one frozen transform"):
        gate.verify_inputs(
            admission=admission,
            huatuo_dir=tmp_path / "huatuo",
            hulu_dir=tmp_path / "hulu",
        )
