from __future__ import annotations

import json
from pathlib import Path

from anchor.medeval.audit_llava_mitigation_t3_generation_v1 import audit
from anchor.medeval.hashing import sha256_file


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _passing_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    manifest = repo / "manifest.json"
    rows = [
        {"qid": "q1", "image_sha256": "i1", "question": "Describe finding."},
        {"qid": "q2", "image_sha256": "i2", "question": "What finding?"},
    ]
    _write_json(manifest, rows)
    provenance = repo / "provenance.json"
    _write_json(provenance, {"reference_values_present": False})
    source = repo / "source.py"
    source.write_text("pass\n")
    contract = repo / "contract.json"
    bindings = {}
    for name, path in {
        "manifest": manifest,
        "manifest_provenance": provenance,
        "matrix_runner": source,
        "generation_port": source,
        "generation_trace": source,
        "vista_adapter": source,
        "model_config": source,
    }.items():
        bindings[name] = {"path": str(path), "sha256": sha256_file(path)}
    _write_json(contract, {
        "methods": ["greedy", "VISTA_off"],
        "reference_method": "greedy",
        "method_off_identity_pair": ["greedy", "VISTA_off"],
        "generation": {"limit": 2, "max_new_tokens": 8, "seed": 42},
        "operational_gates": {
            "method_off_token_exact_rate": 1.0,
            "maximum_cap_hit_rate": 0.05,
            "minimum_nonempty_rate": 0.95,
            "maximum_function_word_only_rate": 0.01,
            "minimum_explicit_sentence_terminal_rate": 0.95,
        },
        "source_bindings": bindings,
    })
    run_root = repo / "run"
    _write_json(run_root / "generation_contract.json", {
        "question_file_sha256": sha256_file(manifest),
        "max_new_tokens": 8,
        "keyword_stopping_enabled": False,
        "seed": 42,
    })
    answer_rows = [
        {
            "question_id": row["qid"],
            "text": "Clear finding.",
            "metadata": {
                "generated_token_ids": [10, 11],
                "raw_generated_token_ids": [10, 11, 2],
                "terminal_token_ids": [2],
                "stop_reason": "eos",
                "keyword_stopping_enabled": False,
            },
        }
        for row in rows
    ]
    for method in ("greedy", "VISTA_off"):
        path = run_root / "vqa_rad/official_test_oe_image_disjoint_n120/open_vqa" / method / "chunk_0000.answers.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in answer_rows))
    return repo, run_root, contract


def test_label_redacted_two_arm_fixture_passes(tmp_path) -> None:
    repo, run_root, contract = _passing_fixture(tmp_path)
    result = audit(run_root=run_root, execution_contract_path=contract, repository_root=repo)
    assert result["all_operational_gates_passed"] is True
    assert result["reference_answers_used"] is False
    assert result["method_off_identity"]["passed"] is True


def test_reference_leak_fails_closed(tmp_path) -> None:
    repo, run_root, contract = _passing_fixture(tmp_path)
    answer = (
        run_root
        / "vqa_rad/official_test_oe_image_disjoint_n120/open_vqa/greedy/chunk_0000.answers.jsonl"
    )
    rows = [json.loads(line) for line in answer.read_text().splitlines()]
    rows[0]["gt_ans"] = "leaked truth"
    answer.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = audit(run_root=run_root, execution_contract_path=contract, repository_root=repo)
    greedy = next(row for row in result["method_records"] if row["method"] == "greedy")
    assert result["all_operational_gates_passed"] is False
    assert greedy["reference_fields_absent"] is False
    assert "reference field leaked into raw generation output" in greedy["errors"]


def test_cap_hit_fails_closed(tmp_path) -> None:
    repo, run_root, contract = _passing_fixture(tmp_path)
    answer = (
        run_root
        / "vqa_rad/official_test_oe_image_disjoint_n120/open_vqa/greedy/chunk_0000.answers.jsonl"
    )
    rows = [json.loads(line) for line in answer.read_text().splitlines()]
    rows[0]["metadata"]["stop_reason"] = "max_new_tokens"
    answer.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = audit(run_root=run_root, execution_contract_path=contract, repository_root=repo)
    greedy = next(row for row in result["method_records"] if row["method"] == "greedy")
    assert result["all_operational_gates_passed"] is False
    assert greedy["cap_hit_rate"] == 0.5
    assert "cap-hit rate exceeds the frozen ceiling" in greedy["errors"]
