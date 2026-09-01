from anchor.medeval.build_physician_oe_deliveries import build_deliveries


def _template(n: int):
    return [
        {
            "group_id": f"g-{index}",
            "candidate_answers": [
                {"answer_id": f"a-{index}", "answer_text": "answer"}
            ],
        }
        for index in range(n)
    ]


def test_builds_two_independent_blinded_deliveries():
    result = build_deliveries(
        _template(24), calibration_groups=10, double_review_groups=24
    )
    assert set(result) == {"A", "B"}
    assert len(result["A"]) == len(result["B"]) == 24
    assert [row["review_phase"] for row in result["A"]].count("calibration") == 10
    assert [row["review_phase"] for row in result["A"]].count("double_review") == 14
    assert {row["reviewer_slot"] for row in result["A"]} == {"A"}
    assert {row["reviewer_slot"] for row in result["B"]} == {"B"}
    assert all("source_model" not in row for rows in result.values() for row in rows)
