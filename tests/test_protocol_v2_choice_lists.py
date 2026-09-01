from anchor.corrected_sgta.protocol_v2 import (
    build_prompt,
    choices_for_sample,
    task_kind,
)


def test_choice_list_uses_position_not_scientific_text_prefix() -> None:
    row = {
        "qid": "pmc-regression",
        "question": "Which organism?",
        "question_type": "multiple-choice",
        "choices": ["C. difficile", "D: Image A and C", "E. coli", "Other"],
        "answer": "A",
    }
    choices = choices_for_sample(row)
    assert choices.labels == ("A", "B", "C", "D")
    assert choices.texts[0] == "C. difficile"
    assert choices.texts[1] == "D: Image A and C"
    assert "A. C. difficile" in build_prompt(row)


def test_empty_choice_list_is_open_ended() -> None:
    row = {
        "qid": "mmmu-open-regression",
        "question": "Which arrow points to a hydrogen bond?",
        "question_type": "open",
        "choices": [],
        "answer": "C",
    }
    assert task_kind(row) == "open"
    assert build_prompt(row) == row["question"]
