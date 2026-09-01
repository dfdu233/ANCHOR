import csv
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from anchor.corrected_sgta.merge_specificity_ratchet_reviews_v1 import (
    COPY_FIELDS,
    REVIEW_FIELDS,
)
from anchor.corrected_sgta.validate_specificity_ratchet_adjudication_v1 import (
    FINAL_FIELDS,
)
from scripts.monitor_specificity_ratchet_pipeline import (
    ANALYSIS_NAME,
    CANARY_NAME,
    FULL_CAPTURE_NAME,
    REPLAY_NAME,
    _advance_scientific_chain,
    advance,
    directory_closure,
    load_adjudicator_attestation,
    load_parent_state_gate,
    load_reviewer_attestation,
    paths,
    substrate_no_go_state,
    validate_source_pack_lock,
)
import scripts.monitor_specificity_ratchet_pipeline as monitor


AUTHORIZED_PARENT_STATE_GATE = {
    "crossing_authorized": True,
    "construct_certifiable": True,
    "scientific_gpu_authorized": True,
}


def _write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path):
    pack = tmp_path / "pack"
    inbox = tmp_path / "inbox"
    output = tmp_path / "output"
    delivery = tmp_path / "delivery"
    image_root = tmp_path / "images"
    pack.mkdir()
    inbox.mkdir()
    image_bytes = b"not-a-real-image-needed-only-for-archive-unit-test"
    digest = hashlib.sha256(image_bytes).hexdigest()
    image_relpath = f"test_images/{digest}.jpg"
    (image_root / "test_images").mkdir(parents=True)
    (image_root / image_relpath).write_bytes(image_bytes)
    candidates = [
        {
            "case_id": "case-1",
            "edge_id": "edge-1",
            "question": "What is present?",
            "image_relpath": image_relpath,
            "answer_span": "A left opacity is present.",
            "parent_proposal": "An opacity is present.",
            "child_proposal": "A left opacity is present.",
            "added_constraint_proposal": "left",
            "edge_type": "laterality",
            "modality_stratum": "XR",
            "anatomy_stratum": "thorax",
            "answer_length_stratum": "short_le_50",
            "observability_screen": "potentially_single_image_decidable",
            "prompt_requested_increment": True,
            "proposal_only": True,
        }
    ]
    (pack / "candidates.blinded.jsonl").write_text(
        json.dumps(candidates[0]) + "\n", encoding="utf-8"
    )
    schema = {
        "protocol_id": "specificity-ratchet-physician-pack-v2",
        "fields": {
            "edge_entailment_admitted": ["yes", "no", "uncertain"],
            "parent_visual_support": ["supported", "refuted", "undetermined", "unobservable"],
            "child_visual_support": ["supported", "refuted", "undetermined", "unobservable"],
            "increment_observability": [
                "observable_on_supplied_image",
                "requires_other_view_or_sequence",
                "requires_history_lab_pathology_or_prior",
                "fundamentally_nonvisual_knowledge",
                "uncertain",
            ],
            "logical_scope_preserved": ["yes", "no", "not_applicable"],
            "reviewer_confidence": ["low", "medium", "high"],
            "clinical_usefulness_if_backed_off": [
                "improves", "unchanged", "minor_loss", "major_loss", "uncertain"
            ],
            "clinically_harmful_if_wrong": ["no", "minor", "major", "uncertain"],
        },
    }
    (pack / "annotation_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    immutable = list(candidates[0])
    review_header = [*immutable, "reviewer_id", *COPY_FIELDS]
    blank = {
        **{
            key: "True" if value is True else "False" if value is False else str(value)
            for key, value in candidates[0].items()
        },
        "reviewer_id": "",
        **{field: "" for field in COPY_FIELDS},
    }
    for role in (1, 2):
        _write_csv(pack / f"annotations.reviewer_{role}.csv", review_header, [blank])
    copied = [
        f"r{role}_{field}"
        for field in [*REVIEW_FIELDS, "rationale"]
        for role in (1, 2)
    ]
    final = [
        *[f"final_{field}" for field in FINAL_FIELDS],
        "adjudicator_id",
        "disagreement_reason",
        "adjudication_rationale",
    ]
    _write_csv(
        pack / "adjudication.csv",
        ["case_id", "edge_id", *copied, *final],
        [{"case_id": "case-1", "edge_id": "edge-1"}],
    )
    return pack, inbox, output, delivery, image_root, schema, review_header, blank


def _complete_review(blank, schema, reviewer_id, child_state):
    row = dict(blank)
    row.update(
        reviewer_id=reviewer_id,
        edge_entailment_admitted="yes",
        parent_visual_support="supported",
        child_visual_support=child_state,
        increment_observability="observable_on_supplied_image",
        logical_scope_preserved="yes",
        reviewer_confidence="high",
        clinical_usefulness_if_backed_off="unchanged",
        clinically_harmful_if_wrong="no",
        rationale=f"Independent assessment by {reviewer_id}.",
    )
    assert all(row[field] in schema["fields"][field] for field in REVIEW_FIELDS)
    return row


def _reviewer_attestation(reviewer_id):
    return {
        "protocol_id": "specificity-ratchet-physician-pack-v2",
        "reviewer": {
            "reviewer_id": reviewer_id,
            "role": "physician",
            "independent_review": True,
            "blinded_to_private_provenance": True,
            "completed_at_utc": "2026-08-03T00:00:00Z",
        },
    }


def test_reviewer_stage_prepares_blinded_adjudication_archive(tmp_path):
    pack, inbox, output, delivery, image_root, schema, header, blank = _fixture(tmp_path)
    p = paths(pack, inbox, output)
    for role, child in ((1, "supported"), (2, "undetermined")):
        _write_csv(
            p[f"review_{role}"],
            header,
            [_complete_review(blank, schema, f"doctor-{role}", child)],
        )
        p[f"attest_{role}"].write_text(
            json.dumps(_reviewer_attestation(f"doctor-{role}")), encoding="utf-8"
        )
    state = advance(
        pack=pack,
        delivery=delivery,
        image_root=image_root,
        p=p,
        parent_state_gate=AUTHORIZED_PARENT_STATE_GATE,
    )
    assert state["stage"] == "blinded_adjudication_prepared"
    assert p["merged"].is_file()
    archive = Path(state["adjudicator_delivery"])
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
        form_name = next(name for name in names if name.endswith("ADJUDICATION_FORM.html"))
        form = handle.extractfile(form_name).read()
    assert not any("provenance" in name.lower() for name in names)
    assert not any("source_model" in name.lower() for name in names)
    assert b"http://" not in form and b"https://" not in form
    assert b"adjudicator.attestation.json" in form
    assert b"function reviewBlock" in form and b"Reviewer ${n}" in form
    waiting = advance(
        pack=pack,
        delivery=delivery,
        image_root=image_root,
        p=p,
        parent_state_gate=AUTHORIZED_PARENT_STATE_GATE,
    )
    assert waiting["stage"] == "waiting_for_blinded_adjudication"

    frozen_review_1_bytes = p["review_1"].read_bytes()
    with p["review_1"].open(newline="", encoding="utf-8") as handle:
        current_review_rows = list(csv.DictReader(handle))
    current_review_rows[0]["rationale"] = "Post-merge inbox replacement."
    _write_csv(p["review_1"], header, current_review_rows)
    replaced_attestation = json.loads(p["attest_1"].read_text())
    replaced_attestation["reviewer"]["completed_at_utc"] = "2026-08-03T01:00:00Z"
    p["attest_1"].write_text(json.dumps(replaced_attestation))

    with p["merged"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        adjudication_header = list(reader.fieldnames or [])
        adjudication_rows = list(reader)
    adjudication_rows[0].update(
        final_edge_entailment_admitted="yes",
        final_parent_visual_support="supported",
        final_child_visual_support="supported",
        final_increment_observability="observable_on_supplied_image",
        final_logical_scope_preserved="yes",
        final_clinical_usefulness_if_backed_off="unchanged",
        final_clinically_harmful_if_wrong="no",
        adjudicator_id="doctor-3",
        disagreement_reason="The readers differed on child support.",
        adjudication_rationale="The supplied image supports the constrained claim.",
    )
    _write_csv(p["adjudication"], adjudication_header, adjudication_rows)
    p["attest_adjudicator"].write_text(
        json.dumps(
            {
                "protocol_id": "specificity-ratchet-physician-pack-v2",
                "adjudicator": {
                    "adjudicator_id": "doctor-3",
                    "role": "physician",
                    "blinded_to_private_provenance": True,
                    "completed_at_utc": "2026-08-03T00:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )
    admitted = advance(
        pack=pack,
        delivery=delivery,
        image_root=image_root,
        p=p,
        parent_state_gate=AUTHORIZED_PARENT_STATE_GATE,
    )
    assert admitted["stage"] == "physician_adjudication_admitted"
    combined = json.loads(
        (p["working_pack"] / "physician_attestations.json").read_text()
    )
    assert [row["reviewer_id"] for row in combined["reviewers"]] == [
        "doctor-1",
        "doctor-2",
    ]
    assert combined["adjudicator"]["adjudicator_id"] == "doctor-3"
    assert (p["working_pack"] / "annotations.reviewer_1.csv").read_bytes() == frozen_review_1_bytes
    assert combined["reviewers"][0]["completed_at_utc"] == "2026-08-03T00:00:00Z"
    assert p["working_pack_lock"].is_file()

    schema_path = p["working_pack"] / "annotation_schema.json"
    schema_path.write_text("{}\n")
    with pytest.raises(RuntimeError, match="directory closure changed"):
        advance(
            pack=pack,
            delivery=delivery,
            image_root=image_root,
            p=p,
            parent_state_gate=AUTHORIZED_PARENT_STATE_GATE,
        )


def test_attestations_are_exact_and_id_bound(tmp_path):
    reviewer_path = tmp_path / "reviewer.json"
    reviewer_path.write_text(json.dumps(_reviewer_attestation("doctor-1")))
    assert load_reviewer_attestation(reviewer_path, "doctor-1")["role"] == "physician"
    with pytest.raises(ValueError, match="differs from completed CSV"):
        load_reviewer_attestation(reviewer_path, "doctor-2")

    adjudicator_path = tmp_path / "adjudicator.json"
    adjudicator_path.write_text(
        json.dumps(
            {
                "protocol_id": "specificity-ratchet-physician-pack-v2",
                "adjudicator": {
                    "adjudicator_id": "doctor-3",
                    "role": "physician",
                    "blinded_to_private_provenance": True,
                    "completed_at_utc": "2026-08-03T00:00:00Z",
                },
            }
        )
    )
    assert load_adjudicator_attestation(adjudicator_path, "doctor-3")["role"] == "physician"
    payload = json.loads(adjudicator_path.read_text())
    payload["adjudicator"]["blinded_to_private_provenance"] = False
    adjudicator_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must be true"):
        load_adjudicator_attestation(adjudicator_path, "doctor-3")


def test_source_pack_lock_refuses_edge_or_schema_drift(tmp_path):
    pack, *_ = _fixture(tmp_path)
    lock = tmp_path / "pack.lock.json"
    lock.write_text(
        json.dumps(
            {
                "version": "specificity-ratchet-source-pack-lock-v1",
                "protocol_id": "specificity-ratchet-physician-pack-v2",
                "files": directory_closure(pack),
            }
        )
    )
    validate_source_pack_lock(pack, lock)
    schema = json.loads((pack / "annotation_schema.json").read_text())
    schema["fields"]["child_visual_support"].append("automatically_inferred")
    (pack / "annotation_schema.json").write_text(json.dumps(schema))
    with pytest.raises(RuntimeError, match="annotation_schema.json"):
        validate_source_pack_lock(pack, lock)


def test_parent_state_gate_binds_audit_candidate_and_auditor_hash(tmp_path):
    pack, *_ = _fixture(tmp_path)
    auditor_sha256 = hashlib.sha256(monitor.PARENT_STATE_AUDITOR.read_bytes()).hexdigest()
    audit = tmp_path / "parent-state-audit.json"
    audit.write_text(
        json.dumps(
            {
                "protocol_id": "specificity-ratchet-parent-before-constraint-audit-v1",
                "status": "no_go_current_pack",
                "candidate_sha256": hashlib.sha256(
                    (pack / "candidates.blinded.jsonl").read_bytes()
                ).hexdigest(),
                "source_sha256": auditor_sha256,
                "outcome_blind_contract": {
                    "physician_reviews_read": False,
                    "adjudication_read": False,
                    "clinical_support_inferred": False,
                },
                "scientific_naming_gate": {"crossing_authorized": False},
                "gates": {"current_pack_surface_construct_certifiable": False},
            }
        )
    )
    gate = load_parent_state_gate(audit, pack)
    assert gate["sha256"] == hashlib.sha256(audit.read_bytes()).hexdigest()
    assert gate["candidate_sha256"] == hashlib.sha256(
        (pack / "candidates.blinded.jsonl").read_bytes()
    ).hexdigest()
    assert gate["scientific_gpu_authorized"] is False

    candidates = (pack / "candidates.blinded.jsonl")
    candidates.write_text(candidates.read_text() + "\n")
    with pytest.raises(RuntimeError, match="stale"):
        load_parent_state_gate(audit, pack)


def test_no_go_gate_is_terminal_before_any_scientific_command(tmp_path, monkeypatch):
    p = paths(tmp_path / "pack", tmp_path / "inbox", tmp_path / "output")
    gate = {
        "path": str(tmp_path / "audit.json"),
        "sha256": "a" * 64,
        "crossing_authorized": False,
        "construct_certifiable": False,
        "scientific_gpu_authorized": False,
    }
    monkeypatch.setattr(
        monitor,
        "_run",
        lambda command: pytest.fail(f"unexpected scientific command: {command}"),
    )
    state = advance(
        pack=tmp_path / "pack",
        delivery=tmp_path / "delivery",
        image_root=tmp_path / "images",
        p=p,
        parent_state_gate=gate,
    )
    assert state == substrate_no_go_state(gate)
    assert state["stage"] == "substrate_no_go_terminal"
    assert state["scientific_gpu_authorized"] is False
    assert state["construct_pilot_returns_preserved"] is True


def _scientific_chain_fixture(tmp_path: Path):
    output = tmp_path / "scientific"
    p = paths(tmp_path / "pack", tmp_path / "inbox", output)
    for key in (
        "canary_state",
        "canary_log",
        "full_capture_state",
        "full_capture_log",
        "replay_state",
        "replay_log",
        "analysis_state",
        "analysis_log",
    ):
        p[key] = tmp_path / "jobs" / p[key].name
    p["manifest"].parent.mkdir(parents=True)
    p["manifest"].write_text('{"sample_id":"sample-1"}\n')
    p["manifest_metadata"].write_text('{"status":"fixture"}\n')
    p["canary_state"].parent.mkdir(parents=True)
    p["canary_state"].write_text(
        json.dumps({"name": CANARY_NAME, "status": "done", "exit_code": 0})
    )
    p["canary_output"].mkdir(parents=True)
    (p["canary_output"] / "CANARY.json").write_text(
        json.dumps(
            {
                "status": "canary_passed",
                "manifest_sha256": hashlib.sha256(p["manifest"].read_bytes()).hexdigest(),
                "metadata_sha256": hashlib.sha256(
                    p["manifest_metadata"].read_bytes()
                ).hexdigest(),
                "target_model_family": "huatuogpt-vision-7b",
                "split": "dev",
                "n_captured_cases": 1,
                "n_identity_failures": 0,
                "direct_output_sequences_captured_for_every_selected_case": True,
            }
        )
    )
    return p


def _complete_full_capture(p):
    p["full_capture_state"].write_text(
        json.dumps({"name": FULL_CAPTURE_NAME, "status": "done", "exit_code": 0})
    )
    p["full_capture_output"].mkdir(parents=True)
    (p["full_capture_output"] / "native_capture.json").write_text(
        json.dumps(
            {
                "status": "complete_passed",
                "manifest_sha256": hashlib.sha256(p["manifest"].read_bytes()).hexdigest(),
                "metadata_sha256": hashlib.sha256(
                    p["manifest_metadata"].read_bytes()
                ).hexdigest(),
                "target_model_family": "huatuogpt-vision-7b",
                "split": "all",
                "n_manifest_cases_in_split": 1,
                "n_captured_cases": 1,
                "n_identity_failures": 0,
                "direct_output_sequences_captured_for_every_selected_case": True,
                "cases": [{"case_id": "case-1", "identity_passed": True}],
            }
        )
    )


def test_canary_pass_launches_full_capture_once(tmp_path, monkeypatch):
    p = _scientific_chain_fixture(tmp_path)
    commands = []
    monkeypatch.setattr(monitor, "_run", lambda command: commands.append(command))
    state = _advance_scientific_chain(
        p, tmp_path / "images", AUTHORIZED_PARENT_STATE_GATE
    )
    assert state == {"stage": "native_full_capture_launched", "job": FULL_CAPTURE_NAME}
    assert len(commands) == 1
    assert "flock" in commands[0]
    assert any(
        token.endswith("capture_huatuo_specificity_native_v1.py")
        for token in commands[0]
    )
    assert "--limit-cases" not in commands[0]


def test_stale_canary_never_authorizes_successor(tmp_path, monkeypatch):
    p = _scientific_chain_fixture(tmp_path)
    canary_path = p["canary_output"] / "CANARY.json"
    canary = json.loads(canary_path.read_text())
    canary["manifest_sha256"] = "0" * 64
    canary_path.write_text(json.dumps(canary))
    monkeypatch.setattr(
        monitor,
        "_run",
        lambda command: pytest.fail(f"unexpected successor launch: {command}"),
    )
    state = _advance_scientific_chain(
        p, tmp_path / "images", AUTHORIZED_PARENT_STATE_GATE
    )
    assert state["stage"] == "native_canary_failed_terminal"
    assert state["retry_authorized"] is False


def test_failed_full_capture_is_terminal_and_never_launches_replay(tmp_path, monkeypatch):
    p = _scientific_chain_fixture(tmp_path)
    p["full_capture_state"].write_text(
        json.dumps({"name": FULL_CAPTURE_NAME, "status": "failed", "exit_code": 2})
    )
    monkeypatch.setattr(
        monitor,
        "_run",
        lambda command: pytest.fail(f"unexpected successor launch: {command}"),
    )
    state = _advance_scientific_chain(
        p, tmp_path / "images", AUTHORIZED_PARENT_STATE_GATE
    )
    assert state["stage"] == "native_full_capture_failed_terminal"
    assert state["retry_authorized"] is False


def test_full_capture_pass_launches_replay_once(tmp_path, monkeypatch):
    p = _scientific_chain_fixture(tmp_path)
    _complete_full_capture(p)
    commands = []
    monkeypatch.setattr(monitor, "_run", lambda command: commands.append(command))
    state = _advance_scientific_chain(
        p, tmp_path / "images", AUTHORIZED_PARENT_STATE_GATE
    )
    assert state == {"stage": "visible_replay_launched", "job": REPLAY_NAME}
    assert len(commands) == 1
    assert "flock" in commands[0]
    assert any(
        token.endswith("specificity_ratchet_visible_replay_v1.py")
        for token in commands[0]
    )
    assert str(p["full_capture_output"] / "native_capture.json") in commands[0]


def test_replay_pass_launches_analysis_and_scientific_nonpass_is_terminal(
    tmp_path, monkeypatch
):
    p = _scientific_chain_fixture(tmp_path)
    _complete_full_capture(p)
    p["replay_state"].write_text(
        json.dumps({"name": REPLAY_NAME, "status": "done", "exit_code": 0})
    )
    p["replay_output"].mkdir(parents=True)
    (p["replay_output"] / "COMPLETE.json").write_text(
        json.dumps({"status": "complete", "native_capture_enforced": True, "rows": 3})
    )
    commands = []
    monkeypatch.setattr(monitor, "_run", lambda command: commands.append(command))
    launched = _advance_scientific_chain(
        p, tmp_path / "images", AUTHORIZED_PARENT_STATE_GATE
    )
    assert launched == {"stage": "visible_analysis_launched", "job": ANALYSIS_NAME}
    assert len(commands) == 1
    assert any(
        token.endswith("analyze_specificity_ratchet_visible_replay_v1.py")
        for token in commands[0]
    )

    p["analysis_state"].write_text(
        json.dumps({"name": ANALYSIS_NAME, "status": "failed", "exit_code": 1})
    )
    p["analysis_output"].write_text(json.dumps({"status": "pilot_only"}))
    commands.clear()
    terminal = _advance_scientific_chain(
        p, tmp_path / "images", AUTHORIZED_PARENT_STATE_GATE
    )
    assert terminal["stage"] == "visible_analysis_terminal"
    assert terminal["scientific_status"] == "pilot_only"
    assert terminal["confirmatory_claim_authorized"] is False
    assert commands == []


def test_done_capture_without_valid_artifact_fails_closed(tmp_path, monkeypatch):
    p = _scientific_chain_fixture(tmp_path)
    p["full_capture_state"].write_text(
        json.dumps({"name": FULL_CAPTURE_NAME, "status": "done", "exit_code": 0})
    )
    monkeypatch.setattr(
        monitor,
        "_run",
        lambda command: pytest.fail(f"unexpected successor launch: {command}"),
    )
    with pytest.raises(RuntimeError, match="without native_capture.json"):
        _advance_scientific_chain(
            p, tmp_path / "images", AUTHORIZED_PARENT_STATE_GATE
        )
