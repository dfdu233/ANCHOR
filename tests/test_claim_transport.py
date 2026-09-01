from corrected_sgta.analyze_claim_transport import image_transport


def test_transport_conserves_claim_count_and_couples_fp_fn() -> None:
    rows = [
        {"question_id": 1, "image": "x", "truth": "no", "baseline": "yes", "method": "no"},
        {"question_id": 2, "image": "x", "truth": "yes", "baseline": "no", "method": "yes"},
        {"question_id": 3, "image": "x", "truth": "yes", "baseline": "yes", "method": "yes"},
    ]
    result = image_transport(rows, seed=1)
    assert result["baseline"]["n_predicted_claims"] == result["transport"]["n_predicted_claims"]
    assert result["transport_tp_delta"] == 1
    assert result["transport"]["fp"] == result["baseline"]["fp"] - 1
    assert result["transport"]["fn"] == result["baseline"]["fn"] - 1


def test_unpaired_deletion_is_rejected() -> None:
    rows = [
        {"question_id": 1, "image": "x", "truth": "no", "baseline": "yes", "method": "no"},
        {"question_id": 2, "image": "x", "truth": "yes", "baseline": "yes", "method": "invalid"},
    ]
    result = image_transport(rows, seed=2)
    assert result["changes"]["admitted_swaps"] == 0
    assert result["changes"]["unmatched_removals"] == 2
    assert result["transport"] == result["baseline"]
