import json

from anchor.medeval.split_answers_by_model_v1 import split


def test_split_preserves_rows_and_qids(tmp_path) -> None:
    source = tmp_path / "answers.jsonl"
    source.write_text(
        "".join(
            json.dumps({"question_id": qid, "model_id": model, "text": qid}) + "\n"
            for model in ("a", "b")
            for qid in ("q1", "q2")
        )
    )
    result = split(source, tmp_path / "out")
    assert result["rows"] == 4
    assert result["outputs"]["a"]["unique_qids"] == 2
    assert result["outputs"]["b"]["rows"] == 2
