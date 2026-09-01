import hashlib
import json
from pathlib import Path

import pytest

from anchor.corrected_sgta.cecd_admission_gate import (
    EXPECTED_AUTHORIZATION_BASIS,
    EXPECTED_VERSION,
    require_cecd_authorization,
)
from anchor.corrected_sgta.analyze_cecd_admission_reviews_v1 import (
    EXPECTED_CANDIDATE_PROMPTS,
    EXPECTED_NONBASELINE_RENDERS,
)
from anchor.corrected_sgta.run_cecd_factorial_v1 import (
    BASELINE_VIEW,
    IDENTITY_RENDER_NAME,
)


def _record(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _authorized_result(tmp_path: Path) -> Path:
    files = []
    for name in ("clinical_1.csv", "clinical_2.csv", "template.csv", "language.csv", "mapping.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files.append(path)
    roles = (
        "clinical_reviewer_1",
        "clinical_reviewer_2",
        "clinical_template_reviewer",
        "language_reviewer",
    )
    attestations = {}
    for index, role in enumerate(roles):
        attestation = tmp_path / f"{role}.attestation.json"
        attestation.write_text(json.dumps({"role": role}), encoding="utf-8")
        attestations[role] = _record(attestation)
    completed_files = {
        "clinical_reviewer_1": _record(files[0]),
        "clinical_reviewer_2": _record(files[1]),
        "clinical_template_reviewer": _record(files[2]),
        "language_reviewer": _record(files[3]),
    }
    bundle = tmp_path / "human_return_bundle_lock.json"
    bundle.write_text(
        json.dumps(
            {
                "version": "cecd-eight-file-human-return-lock-v1",
                "status": "all_roles_and_attestations_frozen_before_unblinding",
                "completed_files": completed_files,
                "attestation_files": attestations,
            }
        ),
        encoding="utf-8",
    )
    validation = tmp_path / "return_validation.json"
    validation.write_text(
        json.dumps(
            {
                "version": "cecd-blinded-human-return-validation-v3",
                "status": "four_independent_returns_validated",
                "roles": [
                    {"role": role, "reviewer_id": f"reviewer-{index}"}
                    for index, role in enumerate(roles)
                ],
                "completed_files": completed_files,
                "attestation_files": attestations,
                "human_return_bundle_lock": _record(bundle),
            }
        ),
        encoding="utf-8",
    )
    source_lock = tmp_path / "source_pack_lock.json"
    source_lock.write_text(
        json.dumps(
            {
                "version": "cecd-admission-source-pack-lock-v1",
                "status": "frozen_before_sealed_mapping_open",
            }
        ),
        encoding="utf-8",
    )
    delivery_verification = tmp_path / "delivery_verification.json"
    browser_smoke = tmp_path / "browser_smoke.json"
    delivery_verification.write_text("{}", encoding="utf-8")
    browser_smoke.write_text("{}", encoding="utf-8")
    payload = {
        "version": EXPECTED_VERSION,
        "status": "passed",
        "passed": True,
        "cecd_model_scoring_authorized": True,
        "maximum_change_rate": 0.05,
        "maximum_unable_rate": 0.10,
        "authorization_basis": EXPECTED_AUTHORIZATION_BASIS,
        "admitted_nonbaseline_renders": list(EXPECTED_NONBASELINE_RENDERS),
        "admitted_candidate_prompts": list(EXPECTED_CANDIDATE_PROMPTS),
        "science_grid_contract": {
            "baseline_render": BASELINE_VIEW,
            "required_nonbaseline_renders": list(EXPECTED_NONBASELINE_RENDERS),
            "required_candidate_prompts": list(EXPECTED_CANDIDATE_PROMPTS),
            "identity_render": IDENTITY_RENDER_NAME,
            "render_set_exact": True,
            "prompt_set_exact": True,
            "all_scored_cells_human_admitted": True,
        },
        "provenance": {
            "clinical_reviews": [_record(files[0]), _record(files[1])],
            "clinical_template_review": _record(files[2]),
            "language_review": _record(files[3]),
            "sealed_mapping": str(files[4]),
            "sealed_mapping_sha256": _record(files[4])["sha256"],
        },
        "human_return_validation": {
            "validation_artifact": _record(validation),
            "attestations": attestations,
        },
        "human_return_bundle_lock": _record(bundle),
        "source_pack_lock": _record(source_lock),
        "reviewer_delivery_validation": {
            "verification": _record(delivery_verification),
            "browser_smoke": _record(browser_smoke),
        },
    }
    result = tmp_path / "analysis.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    return result


def test_cecd_runtime_gate_accepts_complete_current_human_admission(tmp_path: Path) -> None:
    result = _authorized_result(tmp_path)
    assert require_cecd_authorization(result)["passed"] is True


def test_cecd_runtime_gate_rejects_absent_or_failed_admission(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="absent"):
        require_cecd_authorization(tmp_path / "missing.json")
    result = _authorized_result(tmp_path)
    payload = json.loads(result.read_text())
    payload["passed"] = False
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not pass"):
        require_cecd_authorization(result)


def test_cecd_runtime_gate_rejects_changed_reviewer_sheet(tmp_path: Path) -> None:
    result = _authorized_result(tmp_path)
    (tmp_path / "clinical_1.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale"):
        require_cecd_authorization(result)


def test_cecd_runtime_gate_rejects_asserted_pass_with_partial_science_grid(
    tmp_path: Path,
) -> None:
    result = _authorized_result(tmp_path)
    payload = json.loads(result.read_text())
    payload["admitted_nonbaseline_renders"] = list(EXPECTED_NONBASELINE_RENDERS[:2])
    payload["science_grid_contract"]["required_nonbaseline_renders"] = list(
        EXPECTED_NONBASELINE_RENDERS[:2]
    )
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly cover"):
        require_cecd_authorization(result)


def test_cecd_runtime_gate_keeps_legacy_v1_readable_but_non_authorizing(
    tmp_path: Path,
) -> None:
    result = _authorized_result(tmp_path)
    payload = json.loads(result.read_text())
    payload["version"] = "cecd-human-admission-analysis-v1"
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="version is not recognized"):
        require_cecd_authorization(result)
