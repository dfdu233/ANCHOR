import csv
import json
from pathlib import Path

import pytest

from corrected_sgta.build_vindr_cecd_listing_admission_pack_v1 import (
    PROFESSIONAL_ROLE,
    RETURN_SCHEMA_VERSION,
    ROLES,
    VERSION,
    return_schema,
)
from corrected_sgta.package_vindr_cecd_listing_admission_deliveries_v1 import (
    role_kind,
    role_sheet,
)
from corrected_sgta.validate_vindr_cecd_listing_admission_returns_v1 import (
    validate_return,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _attestation(path: Path, role: str, reviewer_id: str = "reviewer-x") -> None:
    path.write_text(
        json.dumps(
            {
                "protocol_id": VERSION,
                "review_role": role,
                "reviewer": {
                    "reviewer_id": reviewer_id,
                    "professional_role": PROFESSIONAL_ROLE[role],
                    "independent_review": True,
                    "blinded_to_sealed_mapping": True,
                    "completed_at_utc": "2026-08-03T05:00:00Z",
                },
            }
        )
    )


def test_return_schema_freezes_four_isolated_roles() -> None:
    schema = return_schema()
    assert schema["schema_version"] == RETURN_SCHEMA_VERSION
    assert set(schema["roles"]) == set(ROLES)
    assert schema["attestation"]["four_distinct_reviewer_ids_required"] is True
    assert len(schema["changed_finding_ids"]["allowed_ids"]) == 14
    assert role_kind("clinical_reviewer_1") == "clinical"
    assert role_kind("clinical_template_reviewer") == "prompt"
    assert role_sheet("language_reviewer") == "language_reviewer.csv"


def test_clinical_return_requires_explicit_changed_finding_ids(tmp_path: Path) -> None:
    role = "clinical_reviewer_1"
    template = {
        "pair_id": "p1",
        "image_A": "images/p1_A.png",
        "image_B": "images/p1_B.png",
        "same_support_state_for_all_14": "",
        "visibility_change": "",
        "listing_interchangeable": "",
        "changed_finding_ids": "",
        "unable_to_judge": "",
        "comments": "",
    }
    _write_csv(tmp_path / f"{role}.csv", [template])
    completed = dict(template)
    completed.update(
        {
            "same_support_state_for_all_14": "no",
            "visibility_change": "A_clearer",
            "listing_interchangeable": "no",
            "changed_finding_ids": "pleural_effusion;pneumothorax",
            "unable_to_judge": "no",
        }
    )
    completed_path = tmp_path / "completed.csv"
    _write_csv(completed_path, [completed])
    attestation = tmp_path / "attestation.json"
    _attestation(attestation, role)
    result = validate_return(
        pack_dir=tmp_path,
        role=role,
        completed_path=completed_path,
        attestation_path=attestation,
    )
    assert result.rows == 1

    completed["changed_finding_ids"] = ""
    _write_csv(completed_path, [completed])
    with pytest.raises(ValueError, match="changed finding required"):
        validate_return(
            pack_dir=tmp_path,
            role=role,
            completed_path=completed_path,
            attestation_path=attestation,
        )


def test_prompt_return_enforces_unable_consistency_and_attestation(tmp_path: Path) -> None:
    role = "language_reviewer"
    template = {
        "item_id": "q1",
        "wording_A": "A",
        "wording_B": "B",
        "same_target_ontology": "",
        "same_inclusion_obligation": "",
        "same_speech_act": "",
        "same_certainty_demand": "",
        "same_answer_space": "",
        "same_output_grammar": "",
        "unable_to_judge": "",
        "comments": "",
    }
    _write_csv(tmp_path / f"{role}.csv", [template])
    completed = dict(template)
    for field in (
        "same_target_ontology",
        "same_inclusion_obligation",
        "same_speech_act",
        "same_certainty_demand",
        "same_answer_space",
        "same_output_grammar",
    ):
        completed[field] = "yes"
    completed["unable_to_judge"] = "no"
    completed_path = tmp_path / "completed.csv"
    _write_csv(completed_path, [completed])
    attestation = tmp_path / "attestation.json"
    _attestation(attestation, role)
    result = validate_return(
        pack_dir=tmp_path,
        role=role,
        completed_path=completed_path,
        attestation_path=attestation,
    )
    assert result.reviewer_id == "reviewer-x"

    completed["same_output_grammar"] = "unable"
    _write_csv(completed_path, [completed])
    with pytest.raises(ValueError, match="unable_to_judge inconsistent"):
        validate_return(
            pack_dir=tmp_path,
            role=role,
            completed_path=completed_path,
            attestation_path=attestation,
        )
