from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from anchor.medeval.evaluators import (
    evaluate_ce_generation,
    parse_leading_binary,
    require_evaluator,
)
from anchor.medeval.legacy import audit_legacy_answers
from anchor.medeval.queue import JobQueue
from anchor.medeval.registry import Registry
from anchor.medeval.schema import ClaimEvidence, PredictionRecord, RetrievalRecord
from anchor.medeval.store import PROTOCOL_ID, PredictionStore, RunManifest
from anchor.medeval.task import doc_to_messages, load_task_spec, materialize_samples
from corrected_sgta.evaluate_medheval_answers import (
    binary_inconsistency,
    normalize_binary_reference,
    parse_answer,
)


def manifest(method: str = "greedy") -> RunManifest:
    return RunManifest(
        protocol_id=PROTOCOL_ID,
        track="common_protocol",
        task={"name": "tiny", "kind": "ce_generation"},
        model={"name": "model", "checkpoint": "sha"},
        method={"name": method},
        generation={"max_new_tokens": 8},
        evaluator={"name": "ce_generation", "sha": "eval"},
        ordered_samples_sha256="samples",
        ordered_images_sha256="images",
        code={"commit": "commit", "diff_sha256": "diff"},
        runtime={"python": "test"},
    )


def test_registry_resolves_alias_and_rejects_duplicate():
    registry = Registry("test")
    registry.register("canonical", 7, aliases=("alias",))
    assert registry.resolve("alias") == 7
    with pytest.raises(ValueError):
        registry.register("other", 8, aliases=("alias",))


def test_common_claim_and_retrieval_contracts_are_fail_closed():
    evidence = ClaimEvidence(
        sample_id="a",
        finding="effusion",
        layer=14,
        logits={"supported": 1.0, "refuted": -1.0, "undetermined": 0.0},
        polarity=1.0,
        clarity=0.0,
    )
    assert evidence.polarity == 1.0
    retrieval = RetrievalRecord(
        sample_id="a",
        query="effusion",
        split_policy="patient_disjoint",
        index_version="v1",
        documents=({"doc_id": "d", "rank": 1, "score": 0.5, "sha256": "abc"},),
    )
    assert retrieval.documents[0]["doc_id"] == "d"
    with pytest.raises(ValueError):
        ClaimEvidence(
            sample_id="a",
            finding="effusion",
            layer=14,
            logits={"supported": 1.0},
            polarity=1.0,
            clarity=0.0,
        )


def test_task_yaml_and_message_snapshot(tmp_path: Path):
    data = tmp_path / "data.json"
    data.write_text(json.dumps([{
        "id": "a", "image": "x.png", "question": "Visible?", "answer": "yes"
    }]))
    config = tmp_path / "task.yaml"
    config.write_text(
        "name: tiny\nkind: ce_generation\n"
        f"dataset_path: {data}\nimage_root: {tmp_path}\n"
        "evaluator: ce_generation\nprompt_template: 'Question: {question}'\n"
        "sample_id_field: id\n"
    )
    samples = materialize_samples(load_task_spec(config))
    assert doc_to_messages(samples[0]) == [{
        "role": "user",
        "content": [
            {"type": "image", "path": "x.png"},
            {"type": "text", "text": "Question: Visible?"},
        ],
    }]


def test_prediction_store_strict_resume(tmp_path: Path):
    store = PredictionStore(tmp_path / "run", manifest())
    record = PredictionRecord(
        protocol_id=PROTOCOL_ID,
        run_fingerprint=manifest().fingerprint,
        sample_fingerprint="sample-sha",
        sample_id="a", cluster_id="patient-a", prediction="Yes",
        prompt="Question",
    )
    store.append(record)
    assert PredictionStore(tmp_path / "run", manifest()).cached("a", "sample-sha") == record
    with pytest.raises(ValueError, match="manifest fingerprint changed"):
        PredictionStore(tmp_path / "run", manifest("vcd"))


def test_ce_parser_is_leading_and_invalid_is_incorrect():
    assert parse_leading_binary("Answer: Yes, visible.") == "yes"
    assert parse_leading_binary("There is no finding; yes elsewhere") is None
    result = evaluate_ce_generation([
        {"sample_id": "1", "reference": "yes", "prediction": "Yes."},
        {"sample_id": "2", "reference": "no", "prediction": "Uncertain."},
    ])
    assert result["accuracy"] == 0.5
    assert result["valid_parse_rate"] == 0.5
    with pytest.raises(ValueError):
        require_evaluator("oe_vqa", "report_claims")


def test_medheval_ceg_parser_uses_leading_decision_only():
    parsed = parse_answer(
        "No, there is no pleural effusion present.", answer_type="binary"
    )
    assert parsed.labels == ("no",)
    assert binary_inconsistency("No, there is no pleural effusion present.")
    parsed = parse_answer("Yes, edema was not present before.", answer_type="binary")
    assert parsed.labels == ("yes",)
    assert binary_inconsistency("Yes, edema was not present before.")
    assert parse_answer("There is no edema.", answer_type="binary").labels is None


def test_binary_reference_accepts_leading_label_but_never_later_semantics():
    assert normalize_binary_reference("No (the silhouette is normal).") == "no"
    assert normalize_binary_reference("Yes, edema was not present before.") == "yes"
    assert normalize_binary_reference("There is no edema.") is None


def test_strict_ternary_parser_keeps_missing_third_state():
    assert parse_answer("Maybe.", answer_type="ternary").labels == ("maybe",)
    assert parse_answer("Uncertain based on this image.", answer_type="ternary").labels == ("maybe",)
    assert parse_answer("There may be edema.", answer_type="ternary").labels is None


def test_legacy_degenerate_complete_file_requires_rerun(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    path.write_text("".join(
        json.dumps({"question_id": str(index), "text": "The"}) + "\n"
        for index in range(25)
    ))
    result = audit_legacy_answers(path, [str(index) for index in range(25)])
    assert result["aligned"] is True
    assert result["grade"] == "C"
    assert result["action"] == "rerun"


def test_short_answer_vqa_does_not_treat_one_word_as_degenerate(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    values = ["left", "right", "frontal", "axial"] * 6
    path.write_text("".join(
        json.dumps({"question_id": str(index), "text": value}) + "\n"
        for index, value in enumerate(values)
    ))
    result = audit_legacy_answers(
        path,
        [str(index) for index in range(len(values))],
        allow_short_answers=True,
    )
    assert result["aligned"] is True
    assert result["degenerate_reasons"] == []
    assert result["grade"] == "B"


def test_full_run_reports_dominance_without_censoring_method(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    path.write_text("".join(
        json.dumps({"question_id": str(index), "text": "right"}) + "\n"
        for index in range(25)
    ))
    result = audit_legacy_answers(
        path,
        [str(index) for index in range(25)],
        allow_short_answers=True,
        enforce_behavioral_quality=False,
    )
    assert result["aligned"] is True
    assert result["degenerate_reasons"] == []
    assert result["behavioral_warnings"] == [
        "one_prediction_dominates_at_least_90_percent"
    ]
    assert result["grade"] == "B"


def test_short_oe_function_word_fragments_fail_qualification(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    values = ["The", "In", "This", "On"] * 6
    path.write_text("".join(
        json.dumps({"question_id": str(index), "text": value}) + "\n"
        for index, value in enumerate(values)
    ))
    result = audit_legacy_answers(
        path,
        [str(index) for index in range(len(values))],
        allow_short_answers=True,
    )
    assert result["aligned"] is True
    assert result["function_word_only_fraction"] == 1.0
    assert "function_word_only_predictions_at_least_50_percent" in result[
        "degenerate_reasons"
    ]


def test_queue_claim_heartbeat_finish_and_recover(tmp_path: Path):
    queue = JobQueue(tmp_path / "jobs.sqlite")
    first = queue.enqueue("one", "fp-one", ["true"], "out")
    assert queue.enqueue("duplicate", "fp-one", ["false"], "other") == first
    job = queue.claim("worker")
    assert job and job.id == first and job.attempts == 1
    queue.heartbeat(job.id, "worker")
    queue.finish(job.id, "worker", 0)
    assert queue.status()[0]["status"] == "done"

    second = queue.enqueue("two", "fp-two", ["true"], "out-two")
    stale = queue.claim("dead-worker")
    assert stale and stale.id == second
    queue.connection.execute(
        "UPDATE jobs SET heartbeat_at=? WHERE id=?", (time.time() - 1000, second)
    )
    assert queue.recover_stale(10) == [second]
    assert queue.status()[1]["status"] == "queued"
