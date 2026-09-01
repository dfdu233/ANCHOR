import json

import pytest

from anchor.medeval.artifact_registry import (
    append_qualification,
    latest_by_artifact,
    qualification_for,
)


def test_registry_is_append_only_and_idempotent(tmp_path):
    artifact = tmp_path / "answers.jsonl"
    artifact.write_text('{"text":"answer"}\n')
    registry = tmp_path / "registry.jsonl"
    value = qualification_for(
        artifact,
        status="identity_only",
        evaluator_version="oe-v2",
        evidence_scope="backend identity only",
        reason="generation cap hit",
    )
    first = append_qualification(registry, value)
    second = append_qualification(registry, value)
    assert first == second
    assert len(registry.read_text().splitlines()) == 1
    assert latest_by_artifact(registry)[str(artifact.resolve())]["status"] == "identity_only"


def test_registry_rejects_unknown_status(tmp_path):
    artifact = tmp_path / "answers.jsonl"
    artifact.write_text("{}\n")
    with pytest.raises(ValueError):
        qualification_for(
            artifact,
            status="trusted",
            evaluator_version="x",
            evidence_scope="x",
            reason="x",
        )


def test_registry_records_failed_cutoff_without_promoting_it(tmp_path):
    artifact = tmp_path / "qualification.json"
    artifact.write_text("{}\n")
    registry = tmp_path / "registry.jsonl"
    value = qualification_for(
        artifact,
        status="failed_cutoff",
        evaluator_version="gate-v1",
        evidence_scope="internal control qualification; control; T2",
        reason="executed but non-degeneracy failed",
    )
    row = append_qualification(registry, value)
    assert row["status"] == "failed_cutoff"
