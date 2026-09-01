import json

from anchor.medeval.audit_oe_generation_qualification_v1 import audit_qualification


def _write(path, caps):
    path.parent.mkdir(parents=True)
    rows = [
        {
            "question_id": f"q{index}",
            "text": "A complete clinical answer.",
            "metadata": {"hit_max_new_tokens": index < caps},
        }
        for index in range(20)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_cap_gate_is_reference_free_and_inclusive_at_five_percent(tmp_path) -> None:
    _write(tmp_path / "m" / "pass" / "answers.jsonl", caps=1)
    _write(tmp_path / "m" / "fail" / "answers.jsonl", caps=2)
    result = audit_qualification(
        tmp_path,
        models=["m"],
        arms=["pass", "fail"],
        expected_rows=20,
    )
    assert result["reference_answers_used"] is False
    assert result["records"][0]["eligible"] is True
    assert result["records"][1]["eligible"] is False
    assert result["all_eligible"] is False
