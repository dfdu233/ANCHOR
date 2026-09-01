import csv
import json
from pathlib import Path

import pytest

import anchor.corrected_sgta.prepare_vindr_cecd_listing_adjudication_handoff_v1 as handoff_module
import anchor.corrected_sgta.run_vindr_cecd_listing_pipeline_v1 as pipeline_module
import anchor.corrected_sgta.validate_vindr_cecd_listing_scientific_admission_v1 as admission_module
from anchor.corrected_sgta.build_vindr_cecd_listing_admission_pack_v1 import (
    CLINICAL_DECISION_FIELDS,
    PROMPT_DECISION_FIELDS,
    ROLES,
)
from anchor.corrected_sgta.prepare_vindr_cecd_listing_adjudication_handoff_v1 import (
    HandoffError,
    prepare_handoff,
)
from anchor.corrected_sgta.run_vindr_cecd_listing_pipeline_v1 import (
    MODELS,
    STAGES,
    prepare_scheduler_handoff,
    validate_scheduler_handoff,
)
from anchor.corrected_sgta.validate_vindr_cecd_listing_scientific_admission_v1 import (
    EXPECTED_SELECTION_HASHES,
    ScientificAdmissionError,
    file_record,
    validate_scientific_admission,
    validate_upstream_binary_ce,
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


def _returns(tmp_path: Path):
    completed = {}
    attestations = {}
    clinical_fields = ["pair_id", "image_A", "image_B", *CLINICAL_DECISION_FIELDS]
    clinical_row = {
        "pair_id": "p1", "image_A": "a.png", "image_B": "b.png",
        "same_support_state_for_all_14": "yes", "visibility_change": "unchanged",
        "listing_interchangeable": "yes", "changed_finding_ids": "",
        "unable_to_judge": "no", "comments": "independent",
    }
    prompt_fields = ["item_id", "wording_A", "wording_B", *PROMPT_DECISION_FIELDS]
    prompt_row = {
        "item_id": "q1", "wording_A": "A", "wording_B": "B",
        "same_target_ontology": "yes", "same_inclusion_obligation": "yes",
        "same_speech_act": "yes", "same_certainty_demand": "yes",
        "same_answer_space": "yes", "same_output_grammar": "yes",
        "unable_to_judge": "no", "comments": "independent",
    }
    for index, role in enumerate(ROLES):
        completed[role] = tmp_path / "inbox" / f"{role}.completed.csv"
        attestations[role] = tmp_path / "inbox" / f"{role}.attestation.json"
        if role.startswith("clinical_reviewer_"):
            _csv(completed[role], clinical_fields, clinical_row)
        else:
            _csv(completed[role], prompt_fields, prompt_row)
        _json(attestations[role], {"reviewer": f"human-{index}"})
    return completed, attestations


def test_valid_returns_create_write_once_human_handoff_without_admission(
    tmp_path: Path, monkeypatch
) -> None:
    pack = tmp_path / "pack"
    _json(pack / "manifest.json", {"version": "fake-pack"})
    completed, attestations = _returns(tmp_path)
    monkeypatch.setattr(handoff_module, "verify", lambda _: {"passed": True})
    monkeypatch.setattr(
        handoff_module,
        "validate_all",
        lambda **_: {
            "status": "four_independent_returns_structurally_valid",
            "roles": [
                {"role": role, "reviewer_id": f"human-{index}", "rows": 1}
                for index, role in enumerate(ROLES)
            ],
            "admission_decision_computed": False,
        },
    )
    output = tmp_path / "handoff"
    first = prepare_handoff(
        pack_dir=pack, completed=completed, attestations=attestations, output_dir=output
    )
    assert first["status"] == "ready_for_human_adjudication"
    assert first["clinical_equivalence_decided"] is False
    assert first["prompt_equivalence_decided"] is False
    assert first["admission_receipt_created"] is False
    assert first["gpu_authorized"] is False
    assert not (output / "admission.json").exists()
    assert (output / "ADJUDICATION_REQUIRED.md").is_file()
    attestation_template = json.loads(
        (output / "adjudicator.attestation.template.json").read_text()
    )
    assert attestation_template["human_admission_decision"] == ""
    assert all(not row["adjudicator_id"] for row in attestation_template["adjudicators"])
    with (output / "clinical_adjudication.template.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["review_A__same_support_state_for_all_14"] == "yes"
    assert row["review_B__same_support_state_for_all_14"] == "yes"
    assert row["adjudicated_same_support_state_for_all_14"] == ""
    second = prepare_handoff(
        pack_dir=pack, completed=completed, attestations=attestations, output_dir=output
    )
    assert second == first
    completed["clinical_reviewer_1"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(HandoffError, match="does not match"):
        prepare_handoff(
            pack_dir=pack, completed=completed, attestations=attestations, output_dir=output
        )


def _canonical_upstream(root: Path, *, both_pass: bool = True):
    admission = root / admission_module.UPSTREAM_ADMISSION_RELATIVE
    dev = root / admission_module.UPSTREAM_DEV_FIT_RELATIVE
    confirmation = root / admission_module.UPSTREAM_CONFIRMATION_RELATIVE
    gate = root / admission_module.UPSTREAM_GATE_RELATIVE
    _json(admission, {"status": "human-admission"})
    _json(dev, {"status": "dev-fit"})
    _json(
        confirmation,
        {
            "version": admission_module.CONFIRMATION_VERSION,
            "status": "complete",
            "stage_label": "confirmation_locked",
            "gate": {
                "both_models_pass": both_pass,
                "authorized_for_method_level_treble_adapter_run": both_pass,
            },
        },
    )
    runs = {
        stage: [
            {
                "family": model,
                "stage": stage,
                "selection_keys_sha256": digest,
                "admission_sha256": admission_module.sha256_file(admission),
            }
            for model in ("huatuo", "hulu")
        ]
        for stage, digest in EXPECTED_SELECTION_HASHES.items()
    }
    _json(
        gate,
        {
            "version": admission_module.THREE_STAGE_VERSION,
            "status": "passed",
            "passed": True,
            "authorized_for_method_level_treble_adapter_run": both_pass,
            "admission": {
                "path": str(admission.resolve()),
                "sha256": admission_module.sha256_file(admission),
                "version": admission_module.BINARY_CE_ADMISSION_VERSION,
            },
            "dev_fit": {"path": str(dev.resolve()), "sha256": admission_module.sha256_file(dev)},
            "confirmation_locked": {
                "path": str(confirmation.resolve()),
                "sha256": admission_module.sha256_file(confirmation),
                "behavioral_gate_passed": both_pass,
            },
            "runs": runs,
        },
    )
    return gate, confirmation


def test_upstream_requires_canonical_path_exact_hash_confirmation_and_selection_closure(
    tmp_path: Path,
) -> None:
    gate, confirmation = _canonical_upstream(tmp_path)
    result = validate_upstream_binary_ce(
        input_gate_path=gate,
        expected_input_gate_sha256=admission_module.sha256_file(gate),
        root=tmp_path,
    )
    assert result["confirmation_path"] == confirmation.resolve()
    with pytest.raises(ScientificAdmissionError, match="hash mismatch"):
        validate_upstream_binary_ce(
            input_gate_path=gate, expected_input_gate_sha256="1" * 64, root=tmp_path
        )
    alias = tmp_path / "structurally-correct-alias.json"
    alias.write_bytes(gate.read_bytes())
    with pytest.raises(ScientificAdmissionError, match="canonical path"):
        validate_upstream_binary_ce(
            input_gate_path=alias,
            expected_input_gate_sha256=admission_module.sha256_file(alias),
            root=tmp_path,
        )
    payload = json.loads(gate.read_text())
    payload["runs"]["dev_fit"][0]["selection_keys_sha256"] = "0" * 64
    _json(gate, payload)
    with pytest.raises(ScientificAdmissionError, match="selection"):
        validate_upstream_binary_ce(
            input_gate_path=gate,
            expected_input_gate_sha256=admission_module.sha256_file(gate),
            root=tmp_path,
        )


def _minimal_receipt_fixture(tmp_path: Path, monkeypatch):
    pack = tmp_path / "pack" / "manifest.json"
    reference = tmp_path / "reference.jsonl"
    experiment = tmp_path / "experiment.json"
    handoff = tmp_path / "handoff.json"
    upstream = tmp_path / "upstream.json"
    confirmation = tmp_path / "confirmation.json"
    clinical = tmp_path / "clinical.completed.csv"
    prompt = tmp_path / "prompt.completed.csv"
    for path in (reference, handoff, upstream, confirmation, clinical, prompt):
        path.write_text("{}\n", encoding="utf-8")
    _json(pack, {})
    _json(experiment, {"reference_contract": {"reference_file_sha256": admission_module.sha256_file(reference)}})
    monkeypatch.setattr(admission_module, "validate_human_evidence", lambda **_: {})
    monkeypatch.setattr(admission_module, "validate_admit_eligibility", lambda **_: None)
    monkeypatch.setattr(
        admission_module,
        "validate_upstream_binary_ce",
        lambda **_: {"confirmation_path": confirmation, "input_gate": {}, "confirmation": {}},
    )
    receipt = {
        "schema_version": admission_module.VERSION,
        "status": "independently_admitted_for_model_scoring",
        "four_independent_human_returns_validated": True,
        "listing_render_equivalence_admitted": True,
        "listing_prompt_equivalence_admitted": True,
        "adjudication_complete": True,
        "human_admission_decision": "admit",
        "upstream_binary_ce_gate_authorized": True,
        "upstream_binary_ce_authorization_sha256": admission_module.sha256_file(upstream),
        "model_scoring_authorized": True,
        "gpu_authorized": True,
        "model_outputs_read_for_admission": False,
        "authorized_model_ids": ["huatuo", "hulu"],
        "pack_manifest_sha256": admission_module.sha256_file(pack),
        "experiment_manifest_sha256": admission_module.sha256_file(experiment),
        "reference_file_sha256": admission_module.sha256_file(reference),
        "computational_guard_failure_pair_ids_sha256": "f" * 64,
        "adjudication_handoff": file_record(handoff),
        "human_evidence": {
            "clinical_adjudication_completed": file_record(clinical),
            "prompt_adjudication_completed": file_record(prompt),
        },
        "admission_validator_source": file_record(admission_module.SOURCE),
        "admission_assembler_source": file_record(
            admission_module.SOURCE.with_name("analyze_vindr_cecd_listing_admission_v1.py")
        ),
        "upstream_binary_ce": {
            "input_gate": file_record(upstream),
            "confirmation_locked": file_record(confirmation),
        },
    }
    receipt_path = tmp_path / "receipt.json"
    _json(receipt_path, receipt)
    return receipt_path, handoff, upstream, pack, experiment


def test_receipt_requires_exact_canonical_validator_source_record(tmp_path: Path, monkeypatch) -> None:
    receipt, handoff, upstream, pack, experiment = _minimal_receipt_fixture(tmp_path, monkeypatch)
    admission_module.validate_scientific_admission(
        receipt_path=receipt, expected_receipt_sha256=admission_module.sha256_file(receipt),
        handoff_path=handoff, expected_handoff_sha256=admission_module.sha256_file(handoff),
        upstream_gate_path=upstream, expected_upstream_gate_sha256=admission_module.sha256_file(upstream),
        pack_manifest_path=pack, experiment_manifest_path=experiment,
    )
    payload = json.loads(receipt.read_text())
    impostor = tmp_path / "validator.py"
    impostor.write_text(admission_module.SOURCE.read_text(), encoding="utf-8")
    payload["admission_validator_source"] = file_record(impostor)
    _json(receipt, payload)
    with pytest.raises(ScientificAdmissionError, match="canonical validator"):
        admission_module.validate_scientific_admission(
            receipt_path=receipt, expected_receipt_sha256=admission_module.sha256_file(receipt),
            handoff_path=handoff, expected_handoff_sha256=admission_module.sha256_file(handoff),
            upstream_gate_path=upstream, expected_upstream_gate_sha256=admission_module.sha256_file(upstream),
            pack_manifest_path=pack, experiment_manifest_path=experiment,
        )


def test_scheduler_preparation_is_inert_ordered_and_write_once(tmp_path: Path, monkeypatch) -> None:
    receipt, handoff, upstream, pack_manifest, experiment = _minimal_receipt_fixture(tmp_path, monkeypatch)
    reference = tmp_path / "reference.jsonl"
    confirmation = tmp_path / "confirmation.json"
    monkeypatch.setattr(
        pipeline_module,
        "validate_scientific_admission",
        lambda **_: {"upstream": {"confirmation_path": confirmation}},
    )
    output = tmp_path / "scheduler.json"
    result = prepare_scheduler_handoff(
        receipt=receipt, expected_receipt_sha256=admission_module.sha256_file(receipt),
        adjudication_handoff=handoff,
        expected_adjudication_handoff_sha256=admission_module.sha256_file(handoff),
        upstream_gate=upstream, expected_upstream_gate_sha256=admission_module.sha256_file(upstream),
        pack_dir=pack_manifest.parent, experiment_manifest=experiment, reference=reference,
        output_root=tmp_path / "runs", handoff_path=output,
    )
    assert result["execution_order"] == [f"{stage}:{model}" for stage in STAGES for model in MODELS]
    assert result["model_or_gpu_launched_during_preparation"] is False
    assert result["simultaneous_models_authorized"] is False
    pinned = admission_module.sha256_file(output)
    assert validate_scheduler_handoff(output, pinned) == result
    assert not (tmp_path / "runs").exists()
