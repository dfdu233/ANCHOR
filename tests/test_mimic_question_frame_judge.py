from anchor.corrected_sgta.judge_mimic_question_frame_opencode import (
    parse_content,
)


def test_judge_parser_accepts_fenced_json():
    rows = parse_content(
        '```json\n[{"id":"1","answer":"yes"}]\n```'
    )
    assert rows == [{"id": "1", "answer": "yes"}]
