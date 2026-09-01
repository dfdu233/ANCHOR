from corrected_sgta.run_slake_prior_titration_probe import (
    pair_rows,
    stratified_contrast_bootstrap,
    stratified_mean_bootstrap,
)


def _score(margin: float):
    return {
        "logits": {
            "supported": margin / 2.0,
            "refuted": -margin / 2.0,
            "undetermined": 0.0,
        }
    }


def _row(case_id: str, polarity: str, margins: dict[str, float]):
    return {
        "case_id": case_id,
        "finding": "effusion",
        "reference_polarity": polarity,
        "status": "ok",
        "scores": {name: _score(value) for name, value in margins.items()},
    }


def test_additive_prior_shift_cancels_from_the_image_contrast() -> None:
    positive = _row("p", "positive", {"low": 0.0, "neutral": 2.0, "high": 4.0})
    negative = _row("n", "negative", {"low": -2.0, "neutral": 0.0, "high": 2.0})
    pair = pair_rows([positive, negative])[0]
    assert pair["contrast"] == {"low": 2.0, "neutral": 2.0, "high": 2.0}
    assert pair["low_to_high_interaction"] == 0.0
    assert pair["curvature"] == 0.0


def test_prior_gating_appears_as_an_interaction() -> None:
    positive = _row("p", "positive", {"low": 0.0, "neutral": 1.0, "high": 6.0})
    negative = _row("n", "negative", {"low": -2.0, "neutral": 0.0, "high": 2.0})
    pair = pair_rows([positive, negative])[0]
    assert pair["low_to_high_interaction"] == 2.0
    assert pair["curvature"] == 4.0


def test_stratified_contrast_does_not_depend_on_arbitrary_pairing() -> None:
    records = [
        _row("p1", "positive", {"low": 1.0, "neutral": 1.0, "high": 1.0}),
        _row("p2", "positive", {"low": 5.0, "neutral": 5.0, "high": 5.0}),
        _row("n1", "negative", {"low": -4.0, "neutral": -4.0, "high": -4.0}),
        _row("n2", "negative", {"low": 0.0, "neutral": 0.0, "high": 0.0}),
    ]
    result = stratified_contrast_bootstrap(
        records,
        lambda row: row["scores"]["neutral"]["logits"]["supported"]
        - row["scores"]["neutral"]["logits"]["refuted"],
        seed=7,
        draws=200,
    )
    assert result["estimate"] == 5.0


def test_real_image_prior_response_uses_all_strata_equally() -> None:
    records = [
        _row("p", "positive", {"low": 0.0, "neutral": 1.0, "high": 2.0}),
        _row("n", "negative", {"low": -2.0, "neutral": 0.0, "high": 2.0}),
    ]
    result = stratified_mean_bootstrap(
        records,
        lambda row: (
            row["scores"]["high"]["logits"]["supported"]
            - row["scores"]["high"]["logits"]["refuted"]
            - row["scores"]["low"]["logits"]["supported"]
            + row["scores"]["low"]["logits"]["refuted"]
        ),
        seed=7,
        draws=200,
    )
    assert result["estimate"] == 3.0
