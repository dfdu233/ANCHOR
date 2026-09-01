import csv
import json
from pathlib import Path

import pytest

import anchor.corrected_sgta.validate_vindr_cecd_listing_scientific_admission_v1 as module
import anchor.corrected_sgta.analyze_vindr_cecd_listing_admission_v1 as assembler_module
from anchor.corrected_sgta.build_vindr_cecd_listing_admission_pack_v1 import ROLES
from anchor.corrected_sgta.prepare_vindr_cecd_listing_adjudication_handoff_v1 import (
    CLINICAL_FINAL_FIELDS,
    PROMPT_FINAL_FIELDS,
    VERSION as HANDOFF_VERSION,
    canonical_sha256,
)
from anchor.corrected_sgta.validate_vindr_cecd_listing_scientific_admission_v1 import (
    ScientificAdmissionError,
    file_record,
    validate_scientific_admission,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def _inventory_record(path: Path, root: Path) -> dict:
    record = file_record(path)
    record["path"] = str(path.resolve().relative_to(root.resolve()))
    return record


def _human_fixture(tmp_path: Path, monkeypatch):
    handoff_dir = tmp_path / "handoff"
    frozen = handoff_dir / "frozen_returns"
    reviewer_ids = []
    inventory = []
    evidence_returns = {}
    for index, role in enumerate(ROLES):
        reviewer_ids.append(f"reviewer-{index}")
        completed = frozen / f"{role}.completed.csv"
        attestation = frozen / f"{role}.attestation.json"
        completed.parent.mkdir(parents=True, exist_ok=True)
        completed.write_text(f"role\n{role}\n", encoding="utf-8")
        _json(attestation, {"reviewer_id": reviewer_ids[-1]})
        inventory.extend(
            [_inventory_record(completed, handoff_dir), _inventory_record(attestation, handoff_dir)]
        )
        evidence_returns[role] = {
            "completed": file_record(completed),
            "attestation": file_record(attestation),
        }

    clinical_template = handoff_dir / "clinical_adjudication.template.csv"
    clinical_completed = handoff_dir / "clinical_adjudication.completed.csv"
    clinical_fields = [
        "pair_id", "image_A", "image_B",
        "review_A__same_support_state_for_all_14",
        "review_B__same_support_state_for_all_14",
        *CLINICAL_FINAL_FIELDS,
    ]
    clinical_blank = {
        "pair_id": "p1", "image_A": "a.png", "image_B": "b.png",
        "review_A__same_support_state_for_all_14": "yes",
        "review_B__same_support_state_for_all_14": "yes",
        **{field: "" for field in CLINICAL_FINAL_FIELDS},
    }
    clinical_final = dict(clinical_blank)
    clinical_final.update(
        {
            "adjudicated_same_support_state_for_all_14": "yes",
            "adjudicated_visibility_change": "unchanged",
            "adjudicated_listing_interchangeable": "yes",
            "adjudicated_changed_finding_ids": "",
            "adjudicated_unable_to_judge": "no",
            "adjudication_rationale": "Both blinded reviews support equivalence.",
        }
    )
    _csv(clinical_template, clinical_fields, clinical_blank)
    _csv(clinical_completed, clinical_fields, clinical_final)

    prompt_template = handoff_dir / "prompt_adjudication.template.csv"
    prompt_completed = handoff_dir / "prompt_adjudication.completed.csv"
    prompt_fields = [
        "item_id", "wording_A", "wording_B",
        "review_A__same_target_ontology", "review_B__same_target_ontology",
        *PROMPT_FINAL_FIELDS,
    ]
    prompt_blank = {
        "item_id": "q1", "wording_A": "A", "wording_B": "B",
        "review_A__same_target_ontology": "yes", "review_B__same_target_ontology": "yes",
        **{field: "" for field in PROMPT_FINAL_FIELDS},
    }
    prompt_final = dict(prompt_blank)
    prompt_final.update(
        {
            "adjudicated_same_target_ontology": "yes",
            "adjudicated_same_inclusion_obligation": "yes",
            "adjudicated_same_speech_act": "yes",
            "adjudicated_same_certainty_demand": "yes",
            "adjudicated_same_answer_space": "yes",
            "adjudicated_same_output_grammar": "yes",
            "adjudicated_unable_to_judge": "no",
            "adjudication_rationale": "The answer contract is unchanged.",
        }
    )
    _csv(prompt_template, prompt_fields, prompt_blank)
    _csv(prompt_completed, prompt_fields, prompt_final)
    inventory.extend(
        [_inventory_record(clinical_template, handoff_dir), _inventory_record(prompt_template, handoff_dir)]
    )
    handoff = {
        "schema_version": HANDOFF_VERSION,
        "status": "ready_for_human_adjudication",
        "handoff_inventory": inventory,
        "validated_return_summary": {
            "roles": [
                {"role": role, "reviewer_id": reviewer_ids[index]}
                for index, role in enumerate(ROLES)
            ]
        },
        "admission_receipt_created": False,
    }
    handoff["fingerprint"] = canonical_sha256(handoff)
    handoff_path = handoff_dir / "handoff.json"
    _json(handoff_path, handoff)

    attestation = handoff_dir / "adjudicator.attestation.json"
    _json(
        attestation,
        {
            "schema_version": module.VERSION,
            "handoff_fingerprint": handoff["fingerprint"],
            "human_admission_decision": "admit",
            "adjudicators": [
                {
                    "scope": "clinical", "adjudicator_id": "adjudicator-clinical",
                    "professional_role": "physician", "independent_adjudication": True,
                    "blinded_to_model_outputs": True, "completed_at_utc": "2026-08-03T12:00:00Z",
                },
                {
                    "scope": "prompt", "adjudicator_id": "adjudicator-language",
                    "professional_role": "language_expert", "independent_adjudication": True,
                    "blinded_to_model_outputs": True, "completed_at_utc": "2026-08-03T12:05:00Z",
                },
            ],
        },
    )
    reference = tmp_path / "reference.jsonl"
    reference.write_text("{}\n", encoding="utf-8")
    pack = tmp_path / "pack" / "manifest.json"
    experiment = tmp_path / "experiment.json"
    _json(pack, {})
    _json(experiment, {"reference_contract": {"reference_file_sha256": module.sha256_file(reference)}})
    upstream = tmp_path / "upstream.json"
    confirmation = tmp_path / "confirmation.json"
    _json(upstream, {})
    _json(confirmation, {})
    monkeypatch.setattr(
        module,
        "validate_upstream_binary_ce",
        lambda **_: {"confirmation_path": confirmation, "input_gate": {}, "confirmation": {}},
    )
    evidence = {
        "frozen_returns": evidence_returns,
        "clinical_adjudication_completed": file_record(clinical_completed),
        "prompt_adjudication_completed": file_record(prompt_completed),
        "adjudicator_attestation": file_record(attestation),
    }
    receipt = {
        "schema_version": module.VERSION,
        "status": "independently_admitted_for_model_scoring",
        "four_independent_human_returns_validated": True,
        "listing_render_equivalence_admitted": True,
        "listing_prompt_equivalence_admitted": True,
        "adjudication_complete": True,
        "human_admission_decision": "admit",
        "upstream_binary_ce_gate_authorized": True,
        "upstream_binary_ce_authorization_sha256": module.sha256_file(upstream),
        "model_scoring_authorized": True,
        "gpu_authorized": True,
        "model_outputs_read_for_admission": False,
        "authorized_model_ids": ["huatuo", "hulu"],
        "pack_manifest_sha256": module.sha256_file(pack),
        "experiment_manifest_sha256": module.sha256_file(experiment),
        "reference_file_sha256": module.sha256_file(reference),
        "computational_guard_failure_pair_ids_sha256": "f" * 64,
        "adjudication_handoff": file_record(handoff_path),
        "human_evidence": evidence,
        "admission_validator_source": file_record(module.SOURCE),
        "admission_assembler_source": file_record(
            module.SOURCE.with_name("analyze_vindr_cecd_listing_admission_v1.py")
        ),
        "upstream_binary_ce": {
            "input_gate": file_record(upstream),
            "confirmation_locked": file_record(confirmation),
        },
    }
    receipt_path = tmp_path / "receipt.json"
    _json(receipt_path, receipt)
    args = {
        "receipt_path": receipt_path,
        "expected_receipt_sha256": module.sha256_file(receipt_path),
        "handoff_path": handoff_path,
        "expected_handoff_sha256": module.sha256_file(handoff_path),
        "upstream_gate_path": upstream,
        "expected_upstream_gate_sha256": module.sha256_file(upstream),
        "pack_manifest_path": pack,
        "experiment_manifest_path": experiment,
    }
    return args, receipt, evidence, attestation


def _rewrite_receipt(args: dict, receipt: dict) -> None:
    _json(args["receipt_path"], receipt)
    args["expected_receipt_sha256"] = module.sha256_file(args["receipt_path"])


def test_eight_frozen_files_and_completed_human_adjudication_are_required(tmp_path, monkeypatch) -> None:
    args, receipt, evidence, _ = _human_fixture(tmp_path, monkeypatch)
    assert validate_scientific_admission(**args)["receipt"]["human_admission_decision"] == "admit"
    role = ROLES[0]
    frozen = Path(evidence["frozen_returns"][role]["completed"]["path"])
    frozen.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ScientificAdmissionError, match="file record/hash mismatch"):
        validate_scientific_admission(**args)
    frozen.write_text(f"role\n{role}\n", encoding="utf-8")
    del receipt["human_evidence"]["frozen_returns"][ROLES[-1]]
    _rewrite_receipt(args, receipt)
    with pytest.raises(ScientificAdmissionError, match="role closure"):
        validate_scientific_admission(**args)


def test_completed_adjudication_immutable_fields_cannot_drift(tmp_path, monkeypatch) -> None:
    args, receipt, evidence, _ = _human_fixture(tmp_path, monkeypatch)
    completed = Path(evidence["clinical_adjudication_completed"]["path"])
    with completed.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames or []), list(reader)
    rows[0]["image_A"] = "different.png"
    _csv(completed, fields, rows[0])
    receipt["human_evidence"]["clinical_adjudication_completed"] = file_record(completed)
    _rewrite_receipt(args, receipt)
    with pytest.raises(ScientificAdmissionError, match="immutable field"):
        validate_scientific_admission(**args)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"adjudicator_id": "reviewer-0"}, "distinct"),
        ({"completed_at_utc": "2026-08-03T12:00:00"}, "timezone"),
        ({"blinded_to_model_outputs": False}, "blinding"),
    ],
)
def test_adjudicator_identity_timezone_and_blinding_are_fail_closed(
    tmp_path, monkeypatch, mutation, match
) -> None:
    args, receipt, _, attestation = _human_fixture(tmp_path, monkeypatch)
    payload = json.loads(attestation.read_text())
    payload["adjudicators"][0].update(mutation)
    _json(attestation, payload)
    receipt["human_evidence"]["adjudicator_attestation"] = file_record(attestation)
    _rewrite_receipt(args, receipt)
    with pytest.raises(ScientificAdmissionError, match=match):
        validate_scientific_admission(**args)


def test_validator_source_copy_is_not_the_canonical_source(tmp_path, monkeypatch) -> None:
    args, receipt, _, _ = _human_fixture(tmp_path, monkeypatch)
    impostor = tmp_path / "validator_copy.py"
    impostor.write_bytes(module.SOURCE.read_bytes())
    receipt["admission_validator_source"] = file_record(impostor)
    _rewrite_receipt(args, receipt)
    with pytest.raises(ScientificAdmissionError, match="canonical validator"):
        validate_scientific_admission(**args)


@pytest.mark.parametrize(
    "decision,authorized,status",
    [
        ("admit", True, "independently_admitted_for_model_scoring"),
        ("reject", False, "human_adjudication_rejected_terminal"),
    ],
)
def test_receipt_assembler_only_copies_explicit_human_decision(
    tmp_path, monkeypatch, decision, authorized, status
) -> None:
    args, _, _, attestation = _human_fixture(tmp_path, monkeypatch)
    completed_attestation = attestation.with_name("adjudicator.attestation.completed.json")
    payload = json.loads(attestation.read_text())
    payload["human_admission_decision"] = decision
    _json(completed_attestation, payload)
    monkeypatch.setattr(
        assembler_module,
        "validate_upstream_binary_ce",
        lambda **_: {"confirmation_path": Path(args["upstream_gate_path"])},
    )
    output = tmp_path / f"{decision}.receipt.json"
    result = assembler_module.assemble_receipt(
        handoff_path=args["handoff_path"],
        expected_handoff_sha256=args["expected_handoff_sha256"],
        upstream_gate_path=args["upstream_gate_path"],
        expected_upstream_gate_sha256=args["expected_upstream_gate_sha256"],
        pack_manifest_path=args["pack_manifest_path"],
        experiment_manifest_path=args["experiment_manifest_path"],
        output=output,
    )
    assert result["human_admission_decision"] == decision
    assert result["status"] == status
    assert result["model_scoring_authorized"] is authorized
    assert result["gpu_authorized"] is authorized
    assert result["authorized_model_ids"] == (["huatuo", "hulu"] if authorized else [])


def test_blank_human_decision_cannot_produce_any_receipt(tmp_path, monkeypatch) -> None:
    args, _, _, attestation = _human_fixture(tmp_path, monkeypatch)
    completed_attestation = attestation.with_name("adjudicator.attestation.completed.json")
    payload = json.loads(attestation.read_text())
    payload["human_admission_decision"] = ""
    _json(completed_attestation, payload)
    monkeypatch.setattr(
        assembler_module,
        "validate_upstream_binary_ce",
        lambda **_: {"confirmation_path": Path(args["upstream_gate_path"])},
    )
    output = tmp_path / "blank.receipt.json"
    with pytest.raises(RuntimeError, match="explicit admit or reject"):
        assembler_module.assemble_receipt(
            handoff_path=args["handoff_path"],
            expected_handoff_sha256=args["expected_handoff_sha256"],
            upstream_gate_path=args["upstream_gate_path"],
            expected_upstream_gate_sha256=args["expected_upstream_gate_sha256"],
            pack_manifest_path=args["pack_manifest_path"],
            experiment_manifest_path=args["experiment_manifest_path"],
            output=output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "sheet_key,field,bad_value,match",
    [
        (
            "clinical_adjudication_completed",
            "adjudicated_listing_interchangeable",
            "no",
            "clinical adjudication",
        ),
        (
            "prompt_adjudication_completed",
            "adjudicated_same_output_grammar",
            "no",
            "prompt adjudication",
        ),
    ],
)
def test_top_level_admit_must_match_every_nonexempt_final_row(
    tmp_path, monkeypatch, sheet_key, field, bad_value, match
) -> None:
    args, _, evidence, attestation = _human_fixture(tmp_path, monkeypatch)
    completed_attestation = attestation.with_name("adjudicator.attestation.completed.json")
    _json(completed_attestation, json.loads(attestation.read_text()))
    completed = Path(evidence[sheet_key]["path"])
    with completed.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, row = list(reader.fieldnames or []), next(reader)
    row[field] = bad_value
    _csv(completed, fields, row)
    monkeypatch.setattr(
        assembler_module,
        "validate_upstream_binary_ce",
        lambda **_: {"confirmation_path": Path(args["upstream_gate_path"])},
    )
    output = tmp_path / "inconsistent-admit.json"
    with pytest.raises(RuntimeError, match=match):
        assembler_module.assemble_receipt(
            handoff_path=args["handoff_path"],
            expected_handoff_sha256=args["expected_handoff_sha256"],
            upstream_gate_path=args["upstream_gate_path"],
            expected_upstream_gate_sha256=args["expected_upstream_gate_sha256"],
            pack_manifest_path=args["pack_manifest_path"],
            experiment_manifest_path=args["experiment_manifest_path"],
            output=output,
        )
    assert not output.exists()
