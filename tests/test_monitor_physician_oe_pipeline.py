import json
from pathlib import Path

import pytest

import scripts.monitor_physician_oe_pipeline as monitor
from scripts.monitor_physician_oe_pipeline import (
    freeze_copy,
    human_input_signatures,
    load_attestation,
    paths,
)


def test_freeze_copy_is_hash_named_and_idempotent(tmp_path: Path):
    source = tmp_path / "return.jsonl"
    source.write_text("{}\n")
    first = freeze_copy(source, tmp_path / "frozen", "reviewer_A.completed")
    second = freeze_copy(source, tmp_path / "frozen", "reviewer_A.completed")
    assert first == second
    assert first.read_bytes() == source.read_bytes()
    assert first.name.startswith("reviewer_A.completed.")


def test_attestation_is_explicit_and_fail_closed(tmp_path: Path):
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "adjudicator_id": "blind-radiologist-3",
                "attest_model_blinded": True,
                "attest_no_private_mapping": True,
            }
        )
    )
    assert load_attestation(good)["adjudicator_id"] == "blind-radiologist-3"
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "adjudicator_id": "blind-radiologist-3",
                "attest_model_blinded": False,
                "attest_no_private_mapping": True,
            }
        )
    )
    with pytest.raises(ValueError, match="model_blinded"):
        load_attestation(bad)


def test_default_pipeline_paths_keep_mapping_out_of_inbox(tmp_path: Path):
    mapped = paths(tmp_path / "base", tmp_path / "inbox", tmp_path / "output")
    assert mapped["mapping"].parent == tmp_path / "base"
    assert mapped["return_a"].parent == tmp_path / "inbox"
    assert mapped["analysis"].parent == tmp_path / "output"


def test_human_input_signatures_bind_size_and_hash(tmp_path: Path):
    mapped = paths(tmp_path / "base", tmp_path / "inbox", tmp_path / "output")
    mapped["return_a"].parent.mkdir(parents=True)
    mapped["return_a"].write_text("first\n")
    first = human_input_signatures(mapped)
    assert set(first) == {"return_a"}
    mapped["return_a"].write_text("second\n")
    second = human_input_signatures(mapped)
    assert first != second
    assert second["return_a"]["size"] == len("second\n")


def test_existing_unqualified_analysis_cannot_mark_pipeline_complete(tmp_path: Path):
    mapped = paths(tmp_path / "base", tmp_path / "inbox", tmp_path / "output")
    mapped["analysis"].parent.mkdir(parents=True)
    mapped["analysis"].write_text(json.dumps({"protocol_version": "wrong"}))
    with pytest.raises(RuntimeError, match="protocol mismatch"):
        monitor.advance(mapped)


def test_analysis_validator_recomputes_all_input_hashes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    mapped = paths(tmp_path / "base", tmp_path / "inbox", tmp_path / "output")
    for key in ("master", "consensus", "provenance", "mapping"):
        mapped[key].parent.mkdir(parents=True, exist_ok=True)
        mapped[key].write_text(f"{key}\n")
    source_hashes = {}
    for key, relative in {
        "prepare_adjudication_source_sha256": "anchor/medeval/prepare_physician_oe_adjudication.py",
        "finalize_consensus_source_sha256": "anchor/medeval/finalize_physician_oe_consensus.py",
        "analysis_source_sha256": "anchor/medeval/analyze_physician_oe_multiarm.py",
    }.items():
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# {key}\n")
        source_hashes[key] = monitor.sha256_file(source)
    mapped["prereg"].write_text(
        json.dumps(
            {
                "protocol_version": "anchor-physician-oe-multiarm-prereg-v1",
                "frozen_before_physician_labels": True,
                "clinical_labels_inspected": False,
                "baseline": "greedy",
                "candidate_methods": ["method"],
                "bootstrap_iterations": 10000,
                "bootstrap_seed": 20260802,
                "provenance": {
                    "review_template_sha256": monitor.sha256_file(mapped["master"]),
                    "private_mapping_sha256": monitor.sha256_file(mapped["mapping"]),
                    **source_hashes,
                },
            }
        )
    )
    expected = {
        "template": str(mapped["master"].resolve()),
        "template_sha256": monitor.sha256_file(mapped["master"]),
        "consensus": str(mapped["consensus"].resolve()),
        "consensus_sha256": monitor.sha256_file(mapped["consensus"]),
        "consensus_provenance": str(mapped["provenance"].resolve()),
        "consensus_provenance_sha256": monitor.sha256_file(mapped["provenance"]),
        "mapping": str(mapped["mapping"].resolve()),
        "mapping_sha256": monitor.sha256_file(mapped["mapping"]),
    }
    mapped["analysis"].parent.mkdir(parents=True, exist_ok=True)
    mapped["analysis"].write_text(
        json.dumps(
            {
                "protocol_version": monitor.ANALYSIS_VERSION,
                "baseline": "greedy",
                "bootstrap_iterations": 10000,
                "seed": 20260802,
                "methods": ["greedy", "method"],
                "aggregates": {"greedy": {}, "method": {}},
                "contrasts": {"method": {"versus": "greedy"}},
                "promoted_methods": [],
                "provenance": expected,
            }
        )
    )
    assert monitor.validate_analysis_artifact(mapped)["methods"] == [
        "greedy",
        "method",
    ]
    mapped["consensus"].write_text("changed\n")
    with pytest.raises(RuntimeError, match="provenance/hash mismatch"):
        monitor.validate_analysis_artifact(mapped)
