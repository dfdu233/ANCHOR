import csv
import hashlib
import json
from pathlib import Path

import pytest

import anchor.corrected_sgta.analyze_clinical_template_attractors_v1 as audit


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fake_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "shards").mkdir(parents=True)
    (run / "errors").mkdir()
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "rad_id", "Pleural effusion", "Lung Opacity"],
        )
        writer.writeheader()
        for index in range(200):
            for reader_index, reader in enumerate(("R8", "R9", "R10")):
                writer.writerow(
                    {
                        "image_id": f"image-{index:03d}",
                        "rad_id": reader,
                        "Pleural effusion": int(reader_index < index % 4),
                        "Lung Opacity": int(reader_index < (index + 1) % 4),
                    }
                )
    ontology = tmp_path / "ontology.json"
    renderer = tmp_path / "renderer.py"
    runner = tmp_path / "runner.py"
    ontology.write_text("{}\n", encoding="utf-8")
    renderer.write_text("# renderer\n", encoding="utf-8")
    runner.write_text("# runner\n", encoding="utf-8")
    manifest = [
        {
            "item_id": f"image-{index:03d}",
            "image_id": f"image-{index:03d}",
            "claim_universe_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "selection_uses_reader_labels": False,
        }
        for index in range(200)
    ]
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest)
    (run / "selected_manifest.jsonl").write_text(manifest_text, encoding="utf-8")
    config = {
        "version": audit.EXPECTED_GENERATION_VERSION,
        "created_at": "ignored",
        "command": [str(runner)],
        "limit": 200,
        "split": "pilot",
        "formal_clinical_claim_evaluation": False,
        "labels_csv": str(labels),
        "labels_csv_sha256": audit.sha256_file(labels),
        "ontology": str(ontology),
        "ontology_sha256": audit.sha256_file(ontology),
        "renderer_source": str(renderer),
        "renderer_source_sha256": audit.sha256_file(renderer),
        "runner_sha256": audit.sha256_file(runner),
        "selected_manifest_sha256": audit.sha256_file(run / "selected_manifest.jsonl"),
        "prompt_conditions": [{"name": value} for value in audit.EXPECTED_CONDITIONS],
    }
    immutable = {key: value for key, value in config.items() if key not in {"created_at", "command"}}
    config["fingerprint"] = audit.canonical_sha(immutable)
    _write_json(run / "generation_config.json", config)
    conformance = {
        "passed": True,
        "fingerprint": config["fingerprint"],
        "direct_text": "same",
        "standard_inference_text": "same",
        "direct_generated_token_ids": [1],
        "direct_generated_token_count": 1,
    }
    _write_json(run / "generation_conformance.json", conformance)
    rows = []
    texts = {
        "neutral": "No pleural effusion or focal lung opacity.",
        "existential": "A possible lung opacity is present.",
        "negative_obligation": "There is no pleural effusion.",
    }
    for item in manifest:
        for condition in audit.EXPECTED_CONDITIONS:
            text = texts[condition]
            row = {
                "version": audit.EXPECTED_GENERATION_VERSION,
                "item_id": item["item_id"],
                "image_id": item["image_id"],
                "prompt_condition": condition,
                "text": text,
                "fingerprint": config["fingerprint"],
                "claim_universe_sha256": item["claim_universe_sha256"],
                "clinical_claim_evaluation_status": "pending_shared_audit",
                "automatic_labeler_used": False,
                "ground_truth_used_for_generation_or_selection": False,
                "generated_token_ids": [1, 2],
                "generated_token_count": 2,
                "visible_answer_token_count": 7,
                "max_new_tokens": 256,
                "hit_max_new_tokens": False,
            }
            rows.append(row)
            _write_json(run / "shards" / f"{audit._record_key(item['item_id'], condition)}.json", row)
    rows.sort(key=lambda row: (row["item_id"], row["prompt_condition"]))
    generations_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    (run / "generations.jsonl").write_text(generations_text, encoding="utf-8")
    _write_json(
        run / "generation_summary.json",
        {
            "status": "generation_complete_clinical_audit_pending",
            "items": 200,
            "generations": 600,
            "fingerprint": config["fingerprint"],
            "generations_sha256": audit.sha256_file(run / "generations.jsonl"),
            "clinical_claim_evaluation": "pending_shared_audit",
        },
    )
    return run


def test_frozen_normalization_and_narrow_claim_families():
    assert audit.normalized_template("Opacity 2.0 CM") == "opacity <num> cm"
    assert audit.word_count("one two-three") == 2
    assert audit.embedded_claim_memberships("No pleural effusion.")["negative_pleural_effusion"]
    assert not audit.embedded_claim_memberships("No pleural effusion.")["positive_pleural_effusion"]
    assert audit.embedded_claim_memberships("Possible lung opacity cannot be excluded.")["uncertain_lung_opacity"]
    assert audit.embedded_claim_memberships("Bilateral patchy opacities are present.")["positive_lung_opacity"]


def test_validation_requires_the_complete_hash_bound_cartesian_product(tmp_path):
    run = _fake_run(tmp_path)
    config, manifest, rows, conformance = audit.validate_generation_run(run)
    assert len(manifest) == 200
    assert len(rows) == 600
    assert conformance["passed"] is True
    next((run / "shards").glob("*.json")).unlink()
    with pytest.raises(ValueError, match="exactly 600 shards"):
        audit.validate_generation_run(run)


def test_end_to_end_is_atomic_diagnostic_and_does_not_authorize_confirmation(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    output = tmp_path / "analysis"
    monkeypatch.setattr(audit, "BOOTSTRAP_REPLICATES", 20)
    summary = audit.analyze(run, output)
    assert summary["integrity"]["generations"] == 600
    assert summary["scope"]["clinical_evaluator"] is False
    frozen = json.loads((output / "frozen_pilot_template_spec.json").read_text())
    assert frozen["confirmation_authorized"] is False
    assert frozen["confirmation_manifest_emitted"] is False
    gate = json.loads((output / "causal_lock_in_gate.json").read_text())
    assert gate["status"] == "not_run"
    assert len(gate["required_sequence"]) == 4
    assert (output / "COMPLETE.json").is_file()


def test_refuses_to_mix_outputs(tmp_path):
    output = tmp_path / "analysis"
    output.mkdir()
    (output / "foreign.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="refusing to mix"):
        audit.analyze(tmp_path / "missing", output)
