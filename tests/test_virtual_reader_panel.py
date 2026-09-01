import json

import numpy as np
import pytest

from corrected_sgta.fit_virtual_reader_panel_v1 import (
    attach_population_weights,
    fit_reader_logistic,
    fit_multinomial_e_only,
    fit_multinomial_em_finding,
    load_feature_records,
    maybe_margin,
    poisson_binomial_three_state,
    predict_multinomial_e_only,
    predict_multinomial_em_finding,
    predict_reader_model,
)


PANEL = ("R8", "R9", "R10")


def test_poisson_binomial_collapses_exact_three_reader_counts():
    observed = poisson_binomial_three_state([0.2, 0.5, 0.8])
    assert np.allclose(observed, [0.08, 0.84, 0.08])
    assert observed.sum() == pytest.approx(1.0)


def test_coordinates_are_invariant_to_common_logit_shift():
    logits = np.asarray([1.0, 2.0, 4.5])
    shifted = logits + 73.0
    assert (logits[2] - logits[0]) == pytest.approx(shifted[2] - shifted[0])
    assert maybe_margin(logits) == pytest.approx(maybe_margin(shifted))
    assert maybe_margin(logits) == pytest.approx(2.0 - np.logaddexp(4.5, 1.0))


def _synthetic_rows(seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    biases = np.asarray([-0.9, 0.1, 0.8])
    for index, score in enumerate(np.linspace(-3.0, 3.0, 180)):
        finding = "effusion" if index % 2 else "nodule"
        finding_bias = 0.25 if finding == "effusion" else -0.25
        probabilities = 1.0 / (1.0 + np.exp(-(1.4 * score + finding_bias + biases)))
        votes = rng.binomial(1, probabilities)
        rows.append(
            {
                "record_key": f"{finding}:{index}",
                "image_id": f"image-{index}",
                "finding": finding,
                "positive_votes": int(votes.sum()),
                "reader_votes": votes,
                "logits": np.asarray([-score / 2, -abs(score), score / 2]),
                "signed_score": float(score),
                "maybe_margin": float(-abs(score) - np.logaddexp(score / 2, -score / 2)),
            }
        )
    return rows


def test_reader_model_recovers_shared_direction_and_centered_reader_order():
    rows = _synthetic_rows()
    model = fit_reader_logistic(
        rows,
        PANEL,
        include_reader_effects=True,
        include_maybe_margin=False,
        l2=1e-4,
    )
    assert all(value > 0 for value in model["score_slope_standardized_by_finding"].values())
    effects = model["reader_effects"]
    assert effects["R8"] < effects["R9"] < effects["R10"]
    assert sum(effects.values()) == pytest.approx(0.0, abs=1e-8)
    predictions = predict_reader_model(model, rows, PANEL)
    assert predictions.shape == (len(rows), 3)
    assert np.allclose(predictions.sum(axis=1), 1.0)
    assert predictions[-1, 2] > predictions[0, 2]


def test_flexible_panel_and_unconstrained_multinomial_are_valid_nested_predictions():
    rows = _synthetic_rows()
    flexible = fit_reader_logistic(
        rows,
        PANEL,
        include_reader_effects=True,
        include_maybe_margin=True,
        flexible_score=True,
        l2=1e-4,
    )
    multinomial = fit_multinomial_e_only(rows, l2=1e-4)
    strong = fit_multinomial_em_finding(rows, l2=1e-4)
    panel_probabilities = predict_reader_model(flexible, rows[:9], PANEL)
    multinomial_probabilities = predict_multinomial_e_only(multinomial, rows[:9])
    strong_probabilities = predict_multinomial_em_finding(strong, rows[:9])
    assert all(
        len(values) == 3
        for values in flexible["score_spline_knots_standardized_by_finding"].values()
    )
    assert panel_probabilities.shape == multinomial_probabilities.shape == (9, 3)
    assert np.allclose(panel_probabilities.sum(axis=1), 1.0)
    assert np.allclose(multinomial_probabilities.sum(axis=1), 1.0)
    assert np.allclose(strong_probabilities.sum(axis=1), 1.0)


def test_finding_specific_slopes_are_not_pooled_or_sign_constrained():
    rows = _synthetic_rows()
    for row in rows:
        if row["finding"] == "nodule":
            row["reader_votes"] = 1 - row["reader_votes"]
            row["positive_votes"] = int(row["reader_votes"].sum())
    model = fit_reader_logistic(
        rows,
        PANEL,
        include_reader_effects=True,
        include_maybe_margin=False,
        flexible_score=False,
        l2=1e-4,
    )
    slopes = model["score_slope_standardized_by_finding"]
    assert slopes["effusion"] > 0
    assert slopes["nodule"] < 0


def test_loader_reorders_votes_by_fixed_panel_and_uses_final_layer(tmp_path):
    directory = tmp_path / "dev"
    directory.mkdir()
    row = {
        "record_key": "effusion:image",
        "image_id": "image",
        "finding": "effusion",
        "positive_votes": 2,
        "reader_votes": [
            {"rad_id": "R10", "vote": 1},
            {"rad_id": "R8", "vote": 0},
            {"rad_id": "R9", "vote": 1},
        ],
        "experiment_split": "dev",
        "diagnostic_plain_logit_lens": {
            "7": {"supported": 10.0, "refuted": 0.0, "undetermined": 0.0},
            "28": {"supported": 4.0, "refuted": 1.0, "undetermined": 2.0},
        },
    }
    (directory / "metadata.jsonl").write_text(json.dumps(row) + "\n")
    loaded = load_feature_records(directory, "dev", PANEL)
    assert loaded[0]["reader_votes"].tolist() == [0, 1, 1]
    assert loaded[0]["logits"].tolist() == [1.0, 2.0, 4.0]
    assert loaded[0]["signed_score"] == 3.0


def test_population_weights_recover_availability_by_finding_vote_stratum():
    rows = [
        {
            "record_key": f"effusion:{vote}",
            "image_id": f"image-{vote}",
            "finding": "effusion",
            "positive_votes": vote,
        }
        for vote in range(4)
    ]
    summary = {
        "split_contract": {"quotas_per_finding_vote_bin": {"dev": 1}},
        "availability_before_sampling": {
            "effusion": {
                f"{vote}/3": {"dev": 10 * (vote + 1)} for vote in range(4)
            }
        }
    }
    weighted = attach_population_weights(rows, summary, "dev")
    raw = [row["population_weight_raw"] for row in weighted]
    assert raw == [10.0, 20.0, 30.0, 40.0]
    assert np.mean([row["population_weight"] for row in weighted]) == pytest.approx(1.0)


def test_loader_fails_closed_on_split_or_vote_mismatch(tmp_path):
    directory = tmp_path / "dev"
    directory.mkdir()
    row = {
        "record_key": "effusion:image",
        "image_id": "image",
        "finding": "effusion",
        "positive_votes": 3,
        "reader_votes": [
            {"rad_id": "R8", "vote": 0},
            {"rad_id": "R9", "vote": 1},
            {"rad_id": "R10", "vote": 1},
        ],
        "experiment_split": "confirmation",
        "diagnostic_plain_logit_lens": {
            "28": {"supported": 4.0, "refuted": 1.0, "undetermined": 2.0}
        },
    }
    (directory / "metadata.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        load_feature_records(directory, "dev", PANEL)
