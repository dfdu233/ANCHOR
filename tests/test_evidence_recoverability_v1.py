from corrected_sgta.analyze_evidence_recoverability_v1 import analyze, fit_threshold

import numpy as np


def _row(index: int, votes: int, values: list[float], finding: str = "effusion"):
    return {
        "record_key": f"{finding}:{index}",
        "image_id": f"image-{index}",
        "finding": finding,
        "positive_votes": votes,
        "diagnostic_plain_logit_lens": {
            str(layer): {"supported": value, "refuted": 0.0, "undetermined": -1.0}
            for layer, value in zip((1, 2, 3, 4), values)
        },
    }


def test_threshold_uses_positive_direction_without_flipping_axis():
    result = fit_threshold(
        np.asarray([-2.0, -1.0, 1.0, 2.0]), np.asarray([-1, -1, 1, 1])
    )
    assert result["dev_balanced_accuracy"] == 1.0
    assert -1.0 < result["threshold"] < 1.0


def test_raw_oracle_exposes_uniform_early_positive_bias():
    dev = [
        _row(0, 0, [1.0, 1.0, 1.0, -1.0]),
        _row(1, 0, [2.0, 2.0, 2.0, -1.0]),
        _row(2, 3, [3.0, 3.0, 3.0, 1.0]),
        _row(3, 3, [4.0, 4.0, 4.0, 1.0]),
    ]
    test = [
        _row(10, 0, [1.0, 1.0, 1.0, 1.0]),  # FP; raw early signs stay wrong
        _row(11, 3, [4.0, 4.0, 4.0, -1.0]),  # FN; raw early signs are right
        _row(12, 0, [1.5, 1.5, 1.5, -1.0]),  # same-truth null donor
        _row(13, 3, [3.5, 3.5, 3.5, 1.0]),   # same-truth null donor
    ]
    result = analyze(dev, test, draws=30, seed=3)
    assert result["recoverability"]["fp"]["native"]["estimate"] == 0.0
    assert result["recoverability"]["fn"]["native"]["estimate"] == 1.0
    assert result["role_counts"] == {"fn": 1, "fp": 1, "tn": 1, "tp": 1}


def test_convex_fusion_cannot_cross_when_all_signed_margins_are_nonpositive():
    signed_margins = np.asarray([-2.0, -0.5, 0.0])
    weights = np.asarray([0.2, 0.3, 0.5])
    assert np.all(weights >= 0) and np.isclose(weights.sum(), 1.0)
    assert float(weights @ signed_margins) <= 0.0


def test_shuffled_null_reports_observed_excess_and_preserves_trajectory_unit():
    dev = [
        _row(0, 0, [-2.0, -2.0, -2.0, -1.0]),
        _row(1, 0, [-1.0, -1.0, -1.0, -1.0]),
        _row(2, 3, [1.0, 1.0, 1.0, 1.0]),
        _row(3, 3, [2.0, 2.0, 2.0, 1.0]),
    ]
    test = [
        _row(10, 0, [-2.0, -2.0, -2.0, 1.0]),
        _row(11, 3, [2.0, 2.0, 2.0, -1.0]),
        _row(12, 0, [-1.0, -1.0, -1.0, -1.0]),
        _row(13, 3, [1.0, 1.0, 1.0, 1.0]),
    ]
    result = analyze(dev, test, draws=100, seed=7)
    null = result["recoverability"]["fp"]["finding_calibrated_shuffled_null"]
    assert null["observed"] == 1.0
    assert null["unit"] == "whole within-finding/same-truth sampled-layer trajectory"
    assert np.isclose(null["observed_minus_mean"], null["observed"] - null["mean"])
