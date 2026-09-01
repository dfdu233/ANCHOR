from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anchor.corrected_sgta.audit_cecd_oe_report_transfer_substrate_v1 import (
    audit_current_substrate,
)
from anchor.corrected_sgta.validate_cecd_oe_report_transfer_pack_v1 import (
    PAIR_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ValidationError,
    content_signature,
    validate_pack,
    validate_pairs,
)


SHA = "a" * 64


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def truth(state: str, relevance: str = "optional") -> dict[str, object]:
    return {
        "state": state,
        "relevance": relevance,
        "reviewer_count": 2,
        "independent_reviews": True,
        "source_model_blinded": True,
        "reference_hidden": True,
        "image_inspected": True,
        "adjudicated": True,
        "adjudicator_is_third_radiologist": True,
        "truth_record_sha256": SHA,
    }


def normalized(finding: str, polarity: str = "present") -> dict[str, object]:
    return {
        "finding": finding,
        "polarity": polarity,
        "uncertainty": "definite",
        "anatomy": None,
        "attributes": [],
    }


def claim(claim_id: str, slot: int, state: str) -> dict[str, object]:
    value = normalized(f"finding_{state}_{slot}")
    target = f"Atomic assertion for {claim_id}."
    return {
        "claim_id": claim_id,
        "slot": slot,
        "text_span": target,
        "target_assertion": target,
        "target_assertion_sha256": hashlib.sha256(target.encode()).hexdigest(),
        "normalized_claim": value,
        "content_signature": content_signature(value),
        "claim_type": "visual",
        "commitment": "definite",
        "atomization": {"human_confirmed": True, "source_model_blinded": True},
        "truth": truth(state),
    }


def future_pack() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for patient_number in range(200):
        split = "dev" if patient_number < 80 else "test"
        patient_id = f"p{patient_number:08d}"
        study_id = f"s{patient_number:08d}"
        image_id = f"image-{patient_number:04d}"
        for task in ("oe_abnormality_listing", "report"):
            case_id = f"{task}-{patient_number:04d}"
            required = normalized("required_finding")
            drafts = []
            for model in ("huatuo", "hulu"):
                drafts.append(
                    {
                        "model_id": model,
                        "draft_id": f"{case_id}-{model}",
                        "generated_once_on_canonical_cell": True,
                        "generated_before_orbit_scoring": True,
                        "answer_text": f"Unique answer {model} {task} patient {patient_number} with clinical detail.",
                        "refused": False,
                        "cap_hit": False,
                        "covered_required_claim_ids": ["required-0"],
                        "claims": [
                            claim(f"{model}-{case_id}-supported", 0, "supported"),
                            claim(f"{model}-{case_id}-refuted", 1, "refuted"),
                            claim(f"{model}-{case_id}-undetermined", 2, "undetermined"),
                        ],
                    }
                )
            cases.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "image_id": image_id,
                    "image_sha256": SHA,
                    "split": split,
                    "task": task,
                    "required_set_reviewed": True,
                    "required_claims": [
                        {
                            "claim_id": "required-0",
                            "normalized_claim": required,
                            "truth": truth("supported", "required"),
                        }
                    ],
                    "drafts": drafts,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "frozen_before_orbit_scores": True,
        "efficacy_outputs_inspected": False,
        "models": ["huatuo", "hulu"],
        "claimed_tasks": ["oe_abnormality_listing", "report"],
        "equivalence_admission": {
            "frozen_before_orbit_scores": True,
            "outcome_blind": True,
            "clinically_admitted": True,
            "reviewer_count": 2,
            "independent_reviews": True,
            "adjudicated": True,
            "adjudicator_is_third_radiologist": True,
            "unique_images": 60,
            "artifact_sha256": SHA,
            "admitted_render_ids": ["r0", "r1", "r2"],
            "admitted_prompt_ids": ["p0", "p1", "p2"],
        },
        "orbit_contract": {
            "science_render_ids": ["r0", "r1", "r2"],
            "science_prompt_ids": ["p0", "p1", "p2"],
            "baseline_render_id": "r0",
            "identity_render_id": "r_identity",
            "duplicate_prompt_id": "p_duplicate",
            "score_definition": "length_normalized_teacher_forced_logprob",
            "target_policy": "exact_same_atomic_assertion_all_cells",
            "draft_policy": "generate_once_canonical_before_orbit",
            "complete_cells_per_claim": 13,
        },
        "cases": cases,
    }


def intervention_rows(pack_result: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for (case_id, model), draft in pack_result["draft_index"].items():
        claims = []
        for base in draft["claims"]:
            updated = {
                key: base[key]
                for key in ("finding", "polarity", "uncertainty", "anatomy", "attributes")
            }
            updated["uncertainty"] = "uncertain"
            claims.append(
                {
                    "claim_id": base["claim_id"],
                    "slot": base["slot"],
                    "normalized_claim": updated,
                }
            )
        rows.append(
            {
                "schema_version": PAIR_SCHEMA_VERSION,
                "case_id": case_id,
                "model_id": model,
                "draft_id": draft["draft_id"],
                "generated_with_content_lock": True,
                "answer_text": draft["answer_text"],
                "refused": draft["refused"],
                "cap_hit": draft["cap_hit"],
                "covered_required_claim_ids": draft["covered_required_claim_ids"],
                "claims": claims,
            }
        )
    return rows


def test_current_cache_shape_remains_strict_no_go(tmp_path: Path) -> None:
    manifest = tmp_path / "vqa.json"
    write_json(
        manifest,
        [
            {
                "qid": "q1",
                "image_sha256": SHA,
                "img_name": f"{SHA}.jpg",
                "answer": "right",
            }
        ],
    )
    vqa_outputs = []
    for model in ("huatuo", "hulu"):
        path = tmp_path / f"{model}.vqa.jsonl"
        write_jsonl(path, [{"question_id": "q1", "text": "Long answer."}])
        vqa_outputs.append((model, path))
    report_outputs = []
    for model in ("hulu", "llava"):
        path = tmp_path / f"{model}.report.jsonl"
        write_jsonl(
            path,
            [
                {
                    "qid": "image-1",
                    "img_name": "p10/p12345678/s12345678/image-1.jpg",
                    "greedy": {"text": "A generated report."},
                }
            ],
        )
        report_outputs.append((model, path))
    physician = tmp_path / "physician"
    write_json(
        physician / "review.metadata.json",
        {"n_model_assignments": 2, "n_answer_units": 2},
    )
    write_jsonl(
        physician / "review.template.jsonl",
        [
            {
                "group_id": "g1",
                "candidate_answers": [
                    {"annotation": {"atomic_claims": []}},
                    {"annotation": {"atomic_claims": []}},
                ],
            }
        ],
    )
    (physician / "clinical_returns_v1").mkdir()

    result = audit_current_substrate(
        manifest, vqa_outputs, report_outputs, physician
    )

    assert result["status"] == "strict_no_go_current_vqa_rad_mimic_physician_pack"
    assert result["gates"]["formal_oe_transfer_authorized"] is False
    assert result["gates"]["gpu_authorized"] is False
    assert result["sources"]["vqa_rad_oe"]["manifest"][
        "rows_with_patient_group_identity"
    ] == 0
    assert result["sources"]["mimic_report"]["patient_identity_recoverable"] is True
    assert result["sources"]["physician_oe_pack"]["completed_return_files"] == 0


def test_future_pack_and_fixed_content_pairs_pass() -> None:
    result = validate_pack(future_pack())
    assert result["formal_orbit_scoring_authorized"] is True
    assert result["efficacy_claim_authorized"] is False
    assert result["patient_counts"]["report"] == {"dev": 80, "test": 120}

    pair_result = validate_pairs(result, intervention_rows(result))
    assert pair_result["status"] == "content_conservation_pass"
    assert pair_result["positive_k_preserved"] is True
    assert pair_result["efficacy_claim_authorized"] is False


def test_patient_split_leak_is_rejected() -> None:
    pack = future_pack()
    pack["cases"][-1]["patient_id"] = "p00000000"
    with pytest.raises(ValidationError, match="leaks across dev/test"):
        validate_pack(pack)


def test_intervention_cannot_delete_a_claim() -> None:
    result = validate_pack(future_pack())
    rows = intervention_rows(result)
    rows[0]["claims"] = rows[0]["claims"][:-1]
    with pytest.raises(ValidationError, match="claim count/K changed"):
        validate_pairs(result, rows)


def test_intervention_cannot_hide_omission_by_coverage_change() -> None:
    result = validate_pack(future_pack())
    rows = intervention_rows(result)
    rows[0]["covered_required_claim_ids"] = []
    with pytest.raises(ValidationError, match="omission coverage changed"):
        validate_pairs(result, rows)
