import json
from pathlib import Path

from anchor.medeval.rescore_ce_history import is_binary_answer_artifact, rescore


def test_rescore_ce_history_preserves_leading_decision(tmp_path: Path):
    source = tmp_path / "answers.jsonl"
    rows = [
        {
            "question_id": "a",
            "text": "No, there is no pleural effusion present.",
            "gt_ans": "No.",
        },
        {
            "question_id": "b",
            "text": "Yes, edema was not present previously.",
            "gt_ans": "Yes.",
        },
        {
            "question_id": "c",
            "text": "There is no pneumothorax.",
            "gt_ans": "No.",
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert is_binary_answer_artifact(rows)
    metrics, records = rescore(source, tmp_path)
    assert metrics["correct"] == 2
    assert metrics["valid"] == 2
    assert metrics["legacy_semantic_mismatch_count"] == 2
    assert metrics["legacy_semantic_ambiguous_count"] == 2
    assert metrics["rule_normalized_flip_count"] == 1
    assert metrics["artifact_status"] == "regenerate"
    assert records[2]["leading_decision"] is None
