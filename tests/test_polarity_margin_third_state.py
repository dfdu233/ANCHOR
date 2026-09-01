from corrected_sgta.analyze_polarity_margin_third_state import (
    select_threshold,
    threshold_metrics,
    uncertainty_margin_auc,
)


def row(state: str, margin: float) -> dict:
    return {
        "state": state,
        "subject_id": f"{state}-{margin}",
        "logits": {"supported": margin / 2, "refuted": -margin / 2, "undetermined": -5.0},
    }


def test_small_opposing_claim_margin_recovers_third_state() -> None:
    rows = [row("supported", 3.0), row("refuted", -3.0), row("undetermined", 0.2)]
    result = threshold_metrics(rows, 0.5)
    assert result["accuracy"] == 1.0
    assert uncertainty_margin_auc(rows) == 1.0


def test_threshold_selection_respects_definite_no_harm_constraint() -> None:
    rows = [
        row("supported", 3.0),
        row("refuted", -3.0),
        row("undetermined", 0.2),
        row("undetermined", -0.3),
    ]
    threshold, result, feasible = select_threshold(rows, baseline_definite_accuracy=1.0)
    assert feasible
    assert threshold <= 3.0
    assert result["definite_accuracy"] == 1.0


def test_zero_threshold_reproduces_binary_argmax_including_ties() -> None:
    rows = [row("supported", 0.0), row("refuted", -1.0)]
    result = threshold_metrics(rows, 0.0)
    assert result["recall"]["supported"] == 1.0
    assert result["recall"]["refuted"] == 1.0
