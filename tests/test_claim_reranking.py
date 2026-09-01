from corrected_sgta.analyze_claim_reranking import evaluate_image


def test_fixed_k_reranking_can_swap_false_for_true() -> None:
    rows = [
        {"question_id": 1, "image": "x", "truth": "no", "baseline": "yes", "scores": {"original_margin": 0, "null_margin": 0, "null_centered_margin": 0}},
        {"question_id": 2, "image": "x", "truth": "yes", "baseline": "no", "scores": {"original_margin": 2, "null_margin": 2, "null_centered_margin": 2}},
        {"question_id": 3, "image": "x", "truth": "yes", "baseline": "yes", "scores": {"original_margin": 1, "null_margin": 1, "null_centered_margin": 1}},
    ]
    result = evaluate_image(rows)
    assert result["baseline"]["n_predicted_claims"] == 2
    assert result["null_centered_margin"]["n_predicted_claims"] == 2
    assert result["null_centered_margin"]["tp"] == result["baseline"]["tp"] + 1
    assert result["null_centered_margin"]["fp"] == result["baseline"]["fp"] - 1
    assert result["null_centered_margin"]["fn"] == result["baseline"]["fn"] - 1
