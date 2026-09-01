from __future__ import annotations

import json

from anchor.medeval.evaluate_oe_matrix import evaluate_matrix


def write_answers(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_matrix_is_strict_but_records_method_local_failure(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"qid": "a", "answer": "left", "image_sha256": "i1"},
        {"qid": "b", "answer": "right", "image_sha256": "i1"},
        {"qid": "c", "answer": "two", "image_sha256": "i2"},
    ]))
    root = tmp_path / "run"
    write_answers(root / "greedy" / "chunk_0000.answers.jsonl", [
        {"question_id": "a", "text": "wrong"},
        {"question_id": "b", "text": "right"},
        {"question_id": "c", "text": "2"},
    ])
    write_answers(root / "beam" / "chunk_0000.answers.jsonl", [
        {"question_id": "a", "text": "left"},
        {"question_id": "b", "text": "right"},
        {"question_id": "c", "text": "2"},
    ])
    result = evaluate_matrix(
        manifest_path=manifest,
        run_root=root,
        method_names=["greedy", "beam", "broken"],
        output_root=tmp_path / "evaluation",
        replicates=100,
        seed=3,
    )
    assert result["completed_methods"] == ["beam", "greedy"]
    assert "broken" in result["failed_methods"]
    assert result["methods"]["beam"]["paired_vs_greedy"]["metrics"]["normalized_exact"]["estimate"] == 1 / 3
    assert result["common_plumbing_valid"] is True
    assert result["scientifically_comparable_methods"] == ["beam", "greedy"]


def test_matrix_invalidates_common_port_with_function_word_fragments(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"qid": str(index), "answer": "right", "image_sha256": f"i{index}"}
        for index in range(20)
    ]))
    root = tmp_path / "run"
    write_answers(root / "greedy" / "chunk_0000.answers.jsonl", [
        {"question_id": str(index), "text": "The"}
        for index in range(20)
    ])
    result = evaluate_matrix(
        manifest_path=manifest,
        run_root=root,
        method_names=["greedy"],
        output_root=tmp_path / "evaluation",
        replicates=20,
        seed=3,
    )
    assert result["completed_methods"] == ["greedy"]
    assert result["common_plumbing_valid"] is False
    assert result["scientifically_comparable_methods"] == []
    assert "greedy_common_port_is_function_word_fragmented" in result[
        "validity_reasons"
    ]
