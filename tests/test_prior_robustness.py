from corrected_sgta.analyze_prior_robustness import analyze_rows, auc


def _score(margin: float):
    return {
        "logits": {
            "supported": margin / 2.0,
            "refuted": -margin / 2.0,
            "undetermined": 0.0,
        }
    }


def _row(case_id: str, polarity: str, margins: tuple[float, float, float]):
    return {
        "case_id": case_id,
        "finding": "effusion",
        "reference_polarity": polarity,
        "status": "ok",
        "scores": {
            name: _score(value)
            for name, value in zip(("low", "neutral", "high"), margins)
        },
    }


def test_auc_handles_ties() -> None:
    assert auc([True, False], [1.0, 1.0]) == 0.5


def test_worst_prior_screen_detects_a_planted_gain() -> None:
    rows = [
        _row("p1", "positive", (2.0, 2.0, 2.0)),
        _row("p2", "positive", (1.5, 1.5, 1.5)),
        _row("n1", "negative", (-1.0, 3.0, -1.0)),
        _row("n2", "negative", (-2.0, 2.5, -2.0)),
    ]
    result = analyze_rows(rows, seed=3, draws=200)
    assert result["metrics"]["worst_prior"]["auroc"] == 1.0
    assert result["metrics"]["neutral"]["auroc"] == 0.0
    assert result["screening_gate_passed"] is True
