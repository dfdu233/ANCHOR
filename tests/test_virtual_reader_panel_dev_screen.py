import numpy as np

from corrected_sgta.screen_virtual_reader_panel_dev_v1 import (
    crossfit_predictions,
    image_group_folds,
)
from tests.test_virtual_reader_panel import PANEL, _synthetic_rows


def test_image_group_folds_keep_repeated_image_claims_together():
    rows = _synthetic_rows()[:40]
    duplicate = dict(rows[0])
    duplicate["record_key"] += ":second-finding"
    duplicate["finding"] = "effusion"
    rows.append(duplicate)
    folds = image_group_folds(rows, 4, 17)
    indices = [index for index, row in enumerate(rows) if row["image_id"] == rows[0]["image_id"]]
    assert len(indices) == 2
    assert len(set(folds[indices])) == 1
    assert set(folds) == {0, 1, 2, 3}


def test_dev_predictions_are_complete_grouped_out_of_fold_probabilities():
    rows = _synthetic_rows()
    predictions, audits, fold_ids = crossfit_predictions(
        rows, PANEL, folds=3, seed=42, l2=1e-4
    )
    assert len(audits) == 3
    assert set(fold_ids) == {0, 1, 2}
    assert set(predictions) == {
        "dev_finding_only_empirical_prior",
        "dev_temperature_scaling",
        "M0_linear_e_reader_finding_threshold",
        "M1_flexible_e_virtual_reader_panel",
        "M2_flexible_e_maybe_interaction_panel",
        "M3_unconstrained_e_only_multinomial",
        "strong_em_finding_multinomial_calibration",
    }
    for probability in predictions.values():
        assert probability.shape == (len(rows), 3)
        assert np.isfinite(probability).all()
        assert np.allclose(probability.sum(axis=1), 1.0)
    assert all(
        set(audit["M0_score_slopes_by_finding"]) == {"effusion", "nodule"}
        for audit in audits
    )
