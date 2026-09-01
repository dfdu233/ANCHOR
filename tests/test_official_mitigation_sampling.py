from corrected_sgta.run_official_mitigation import (
    balanced_binary_image_sample,
    claim_universe_image_sample,
)


def test_balanced_binary_image_sampling_is_deterministic_and_disjoint() -> None:
    rows = []
    for image in range(20):
        for label in ("Yes", "No"):
            rows.append({
                "qid": len(rows),
                "img_name": f"image-{image}.jpg",
                "answer": label,
                "modality": "CT" if image % 2 else "MRI",
                "hallucination_type": f"type_{image % 4}",
            })
    first = balanced_binary_image_sample(rows, cap=16, seed=42)
    second = balanced_binary_image_sample(rows, cap=16, seed=42)
    assert [row["qid"] for row in first] == [row["qid"] for row in second]
    assert len({row["img_name"] for row in first}) == 16
    assert sum(row["answer"] == "Yes" for row in first) == 8
    assert sum(row["answer"] == "No" for row in first) == 8


def test_balanced_binary_image_sampling_rejects_odd_cap() -> None:
    try:
        balanced_binary_image_sample([], cap=3, seed=1)
    except ValueError as error:
        assert "positive even" in str(error)
    else:
        raise AssertionError("odd cap should fail")


def test_claim_universe_sampling_keeps_all_binary_claims_per_image() -> None:
    rows = [
        {"qid": 1, "img_name": "a.jpg", "answer": "Yes"},
        {"qid": 2, "img_name": "a.jpg", "answer": "No"},
        {"qid": 3, "img_name": "a.jpg", "answer": "Left"},
        {"qid": 4, "img_name": "b.jpg", "answer": "No"},
        {"qid": 5, "img_name": "c.jpg", "answer": "Yes"},
    ]
    selected = claim_universe_image_sample(rows, image_cap=2, seed=11)
    names = {row["img_name"] for row in selected}
    assert len(names) == 2
    for name in names:
        assert {row["qid"] for row in selected if row["img_name"] == name} == {
            row["qid"] for row in rows
            if row["img_name"] == name and row["answer"] in {"Yes", "No"}
        }
