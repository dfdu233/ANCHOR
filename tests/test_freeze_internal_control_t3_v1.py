from anchor.medeval.freeze_internal_control_t3_v1 import freeze_rows


def test_freeze_is_deterministic_one_question_per_image_and_outcome_blind() -> None:
    rows = [
        {"qid": "q1", "image_sha256": "i1", "answer": "a"},
        {"qid": "q2", "image_sha256": "i1", "answer": "b"},
        {"qid": "q3", "image_sha256": "i2", "answer": "c"},
    ]
    selected = freeze_rows(rows)
    mutated = [{**row, "answer": "opposite"} for row in rows]
    assert [row["qid"] for row in selected] == [row["qid"] for row in freeze_rows(mutated)]
    assert len(selected) == 2
    assert len({row["image_sha256"] for row in selected}) == 2
