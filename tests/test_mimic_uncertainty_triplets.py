from corrected_sgta.analyze_mimic_uncertainty_triplets import metrics, rank_auc


def test_rank_auc_with_ties() -> None:
    assert rank_auc([2.0, 1.0], [0.0]) == 1.0
    assert rank_auc([1.0], [1.0]) == 0.5


def test_third_state_bias_trades_uncertain_and_definite_commitment() -> None:
    rows = [
        {"state": "supported", "logits": {"supported": 3, "refuted": 0, "undetermined": 1}},
        {"state": "refuted", "logits": {"supported": 0, "refuted": 3, "undetermined": 1}},
        {"state": "undetermined", "logits": {"supported": 2, "refuted": 0, "undetermined": 1.5}},
    ]
    baseline = metrics(rows)
    calibrated = metrics(rows, uncertainty_bias=1.0)
    assert baseline["recall"]["undetermined"] == 0.0
    assert calibrated["recall"]["undetermined"] == 1.0
    assert calibrated["definite_accuracy"] == 1.0
