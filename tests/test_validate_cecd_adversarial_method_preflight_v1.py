import json
from pathlib import Path

import pytest

from anchor.corrected_sgta.validate_cecd_adversarial_method_preflight_v1 import (
    COLLISION_SOURCES,
    PreflightError,
    sha256_file,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/cecd_adversarial_method_preflight_v1.json"


def _record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _ready_plan(tmp_path: Path) -> dict:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    source = tmp_path / "audited_control.py"
    source.write_text("# independently audited test control\n", encoding="utf-8")
    for record in payload["controls"].values():
        record.update(
            {
                "status": "ready",
                "implementation_fidelity": "independent_common_protocol",
                "models": {"huatuo": "ready", "hulu": "ready"},
                "source_files": [_record(source)],
            }
        )
    payload["controls"]["hallucxr_length_omission"][
        "implementation_fidelity"
    ] = "evaluation_guard"
    payload["controls"]["conrad_proper_score_confidence"][
        "implementation_fidelity"
    ] = "proper_score_common_protocol_control"
    frozen_bindings = {}
    for key in payload["bindings"]:
        binding = tmp_path / f"{key}.json"
        binding.write_text("{}\n", encoding="utf-8")
        frozen_bindings[key] = _record(binding)
    payload["bindings"] = frozen_bindings
    payload["method_output_root"] = str(tmp_path / "outputs")
    return payload


def test_current_plan_is_truthfully_blocked_mechanism_only() -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    result = validate_plan(payload, root=ROOT)
    assert result["passed"] is False
    assert result["mechanism_paper_scope_only"] is True
    assert result["mitigation_novelty_authorized"] is False
    assert set(result["controls"]) == set(COLLISION_SOURCES)
    assert any("system_mediated_yes_bias:not_implemented" in x for x in result["blockers"])


def test_ready_plan_authorizes_execution_but_never_paper_claim(tmp_path: Path) -> None:
    result = validate_plan(_ready_plan(tmp_path), root=tmp_path)
    assert result["passed"] is True
    assert result["full_method_execution_ready"] is True
    assert result["mitigation_novelty_authorized"] is False
    assert result["paper_claim_authorized"] is False
    assert result["three_stage_thresholds_modified"] is False


def test_ready_control_requires_both_models(tmp_path: Path) -> None:
    payload = _ready_plan(tmp_path)
    payload["controls"]["prompt_induced_heads"]["models"]["hulu"] = "incompatible"
    result = validate_plan(payload, root=tmp_path)
    assert result["passed"] is False
    assert "prompt_induced_heads:not_ready_on_both_models" in result["blockers"]


def test_rejects_occupied_novelty_claim(tmp_path: Path) -> None:
    payload = _ready_plan(tmp_path)
    payload["claim_boundaries"]["confidence_calibration_novelty_claimed"] = True
    with pytest.raises(PreflightError, match="novelty claim boundary"):
        validate_plan(payload, root=tmp_path)


def test_nonempty_output_root_is_too_late(tmp_path: Path) -> None:
    payload = _ready_plan(tmp_path)
    output = Path(payload["method_output_root"])
    output.mkdir(parents=True)
    (output / "result.json").write_text("{}\n", encoding="utf-8")
    result = validate_plan(payload, root=tmp_path)
    assert result["passed"] is False
    assert "method_output_root_not_empty_preflight_too_late" in result["blockers"]


def test_source_hash_drift_blocks(tmp_path: Path) -> None:
    payload = _ready_plan(tmp_path)
    payload["controls"]["cebc_evidence_bounded_editing"]["source_files"][0][
        "sha256"
    ] = "0" * 64
    result = validate_plan(payload, root=tmp_path)
    assert result["passed"] is False
    assert any("cebc_evidence_bounded_editing:source_hash_mismatch" in x for x in result["blockers"])
