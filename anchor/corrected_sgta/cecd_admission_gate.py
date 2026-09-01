"""Fail-closed runtime authorization for CECD model scoring.

The blinded human review is a scientific prerequisite, not an optional piece
of provenance.  Keeping this check in the executable path prevents a stale
watchdog manifest or a container restart from launching the factorial early.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from corrected_sgta.analyze_cecd_admission_reviews_v1 import (
    EXPECTED_CANDIDATE_PROMPTS,
    EXPECTED_NONBASELINE_RENDERS,
    VERSION as ADMISSION_ANALYSIS_VERSION,
)
from corrected_sgta.run_cecd_factorial_v1 import (
    BASELINE_VIEW,
    IDENTITY_RENDER_NAME,
)

EXPECTED_VERSION = ADMISSION_ANALYSIS_VERSION
EXPECTED_RETURN_VALIDATION_VERSION = "cecd-blinded-human-return-validation-v3"
EXPECTED_AUTHORIZATION_BASIS = (
    "four independent blinded role returns over clinical equivalence and language "
    "equivalence; pixel similarity is prohibited as clinical-admission evidence"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_record(record: Any, label: str) -> None:
    if not isinstance(record, dict):
        raise RuntimeError(f"CECD admission lacks {label} provenance")
    path = Path(str(record.get("path", "")))
    expected = str(record.get("sha256", ""))
    if not path.is_file() or len(expected) != 64 or _sha256(path) != expected:
        raise RuntimeError(f"CECD admission {label} provenance is missing or stale")


def require_cecd_authorization(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            "CECD model scoring is not authorized: blinded human-admission "
            f"analysis is absent ({path})"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != EXPECTED_VERSION:
        raise RuntimeError("CECD admission analysis version is not recognized")
    if payload.get("status") != "passed" or payload.get("passed") is not True:
        raise RuntimeError("CECD human-admission gate did not pass")
    if payload.get("cecd_model_scoring_authorized") is not True:
        raise RuntimeError("CECD admission does not authorize model scoring")
    if (
        payload.get("maximum_change_rate") != 0.05
        or payload.get("maximum_unable_rate") != 0.10
        or payload.get("authorization_basis") != EXPECTED_AUTHORIZATION_BASIS
    ):
        raise RuntimeError("CECD admission threshold or clinical-evidence basis drift")
    science_grid = payload.get("science_grid_contract")
    if (
        not isinstance(science_grid, dict)
        or science_grid.get("baseline_render") != BASELINE_VIEW
        or science_grid.get("identity_render") != IDENTITY_RENDER_NAME
        or science_grid.get("required_nonbaseline_renders")
        != list(EXPECTED_NONBASELINE_RENDERS)
        or science_grid.get("required_candidate_prompts")
        != list(EXPECTED_CANDIDATE_PROMPTS)
        or science_grid.get("render_set_exact") is not True
        or science_grid.get("prompt_set_exact") is not True
        or science_grid.get("all_scored_cells_human_admitted") is not True
        or payload.get("admitted_nonbaseline_renders")
        != list(EXPECTED_NONBASELINE_RENDERS)
        or payload.get("admitted_candidate_prompts")
        != list(EXPECTED_CANDIDATE_PROMPTS)
    ):
        raise RuntimeError(
            "CECD admission does not exactly cover the frozen runner science grid"
        )

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("CECD admission lacks reviewer provenance")
    clinical = provenance.get("clinical_reviews")
    if not isinstance(clinical, list) or len(clinical) != 2:
        raise RuntimeError("CECD admission requires exactly two clinical reviews")
    for index, record in enumerate(clinical, 1):
        _verify_record(record, f"clinical reviewer {index}")
    _verify_record(provenance.get("clinical_template_review"), "clinical template reviewer")
    _verify_record(provenance.get("language_review"), "language reviewer")
    _verify_record(
        {
            "path": provenance.get("sealed_mapping"),
            "sha256": provenance.get("sealed_mapping_sha256"),
        },
        "sealed mapping",
    )
    human = payload.get("human_return_validation")
    if not isinstance(human, dict):
        raise RuntimeError("CECD admission lacks v3 human-return validation")
    _verify_record(human.get("validation_artifact"), "human-return validation artifact")
    validation_path = Path(str(human["validation_artifact"]["path"]))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("version") != EXPECTED_RETURN_VALIDATION_VERSION:
        raise RuntimeError("CECD human-return validation version is not recognized")
    if validation.get("status") != "four_independent_returns_validated":
        raise RuntimeError("CECD does not contain four validated independent returns")
    roles = validation.get("roles")
    if not isinstance(roles, list) or len(roles) != 4:
        raise RuntimeError("CECD return validation does not contain four roles")
    reviewer_ids = [row.get("reviewer_id") for row in roles if isinstance(row, dict)]
    if len(reviewer_ids) != 4 or len(set(reviewer_ids)) != 4 or any(not value for value in reviewer_ids):
        raise RuntimeError("CECD return validation lacks four distinct reviewer IDs")
    attestations = human.get("attestations")
    if not isinstance(attestations, dict) or set(attestations) != {
        "clinical_reviewer_1",
        "clinical_reviewer_2",
        "clinical_template_reviewer",
        "language_reviewer",
    }:
        raise RuntimeError("CECD admission lacks all four attestation records")
    for role, record in attestations.items():
        _verify_record(record, f"{role} attestation")
    _verify_record(payload.get("human_return_bundle_lock"), "eight-file human-return lock")
    bundle_path = Path(str(payload["human_return_bundle_lock"]["path"]))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if (
        bundle.get("version") != "cecd-eight-file-human-return-lock-v1"
        or bundle.get("status")
        != "all_roles_and_attestations_frozen_before_unblinding"
        or bundle.get("completed_files") != validation.get("completed_files")
        or bundle.get("attestation_files") != validation.get("attestation_files")
        or validation.get("human_return_bundle_lock")
        != payload.get("human_return_bundle_lock")
    ):
        raise RuntimeError("CECD eight-file human-return lock contract mismatch")
    _verify_record(payload.get("source_pack_lock"), "source-pack lock")
    source_lock_path = Path(str(payload["source_pack_lock"]["path"]))
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if (
        source_lock.get("version") != "cecd-admission-source-pack-lock-v1"
        or source_lock.get("status") != "frozen_before_sealed_mapping_open"
    ):
        raise RuntimeError("CECD source-pack lock contract mismatch")
    delivery = payload.get("reviewer_delivery_validation")
    if not isinstance(delivery, dict):
        raise RuntimeError("CECD admission lacks reviewer-delivery validation")
    _verify_record(delivery.get("verification"), "reviewer-delivery verification")
    _verify_record(delivery.get("browser_smoke"), "reviewer-delivery browser smoke")
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    require_cecd_authorization(args.result)
    print(f"CECD model scoring authorized by {args.result.resolve()}")


if __name__ == "__main__":
    main()
