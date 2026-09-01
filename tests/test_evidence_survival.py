import numpy as np

from corrected_sgta.analyze_evidence_survival_v1 import summarize_record
from corrected_sgta.run_huatuo_evidence_survival_v1 import (
    deterministic_subset,
    square_grid_roi_mask,
)


def test_square_grid_roi_mask_accounts_for_square_padding_and_overlap():
    # A 100x50 image is vertically padded to 100x100.  This box occupies the
    # upper-left image quadrant and therefore overlaps only the lower half of
    # the first grid row and the left grid column.
    mask = square_grid_roi_mask(
        width=100,
        height=50,
        boxes=[{"x_min": 0, "y_min": 0, "x_max": 50, "y_max": 25}],
        token_count=4,
    )
    assert mask.tolist() == [True, False, False, False]


def test_square_grid_roi_mask_matches_integer_floor_padding_for_odd_difference():
    # Huatuo pastes a 100x51 image at y=(100-51)//2=24.  This sub-pixel box
    # remains just above the grid boundary; a mathematically centered 24.5
    # offset would incorrectly move it into the lower row.
    mask = square_grid_roi_mask(
        width=100,
        height=51,
        boxes=[{"x_min": 0, "y_min": 25.5, "x_max": 25, "y_max": 25.75}],
        token_count=4,
    )
    assert mask.tolist() == [True, False, False, False]


def test_deterministic_subset_is_repeatable_and_nested_across_doses():
    indices = np.arange(20)
    small = deterministic_subset(indices, 5, "case:roi", 0)
    large = deterministic_subset(indices, 12, "case:roi", 0)
    assert np.array_equal(small, deterministic_subset(indices, 5, "case:roi", 0))
    assert set(small).issubset(set(large))
    assert len(np.unique(large)) == 12


def test_survival_summary_uses_supported_minus_undetermined_for_positive_claim():
    row = {
        "record_key": "pleural_effusion:image",
        "image_id": "image",
        "finding": "pleural_effusion",
        "positive_votes": 3,
        "roi_tokens": 4,
        "baseline_logits": {"supported": 3.0, "refuted": 1.0, "undetermined": 2.0},
        "baseline_coordinates": {"polarity": 2.0, "commitment": 1.0},
        "interventions": [
            {
                "region": region,
                "dose": dose,
                "logits": {
                    "supported": supported,
                    "refuted": 1.0,
                    "undetermined": 2.0,
                },
                "coordinates": {"polarity": supported - 1.0, "commitment": supported - 2.0},
            }
            for region, supported in (("roi", 2.0), ("background", 3.0))
            for dose in (0.5, 1.0)
        ],
    }
    summary = summarize_record(row)
    assert summary["curves"]["roi"]["positive_commitment"] == [1.0, 0.0, 0.0]
    assert summary["curves"]["background"]["positive_commitment"] == [1.0, 1.0, 1.0]
    assert summary["roi_minus_background_survival_auc"] < 0


def test_survival_summary_excludes_positive_polarity_without_definite_commitment():
    row = {
        "record_key": "nodule_mass:image",
        "image_id": "image",
        "finding": "nodule_mass",
        "positive_votes": 2,
        "roi_tokens": 2,
        "baseline_logits": {"supported": 2.0, "refuted": 1.0, "undetermined": 3.0},
        "baseline_coordinates": {"polarity": 1.0, "commitment": 2.0},
        "interventions": [
            {
                "region": region,
                "dose": 1.0,
                "logits": {"supported": 2.0, "refuted": 1.0, "undetermined": 3.0},
                "coordinates": {"polarity": 1.0, "commitment": 2.0},
            }
            for region in ("roi", "background")
        ],
    }
    assert summarize_record(row)["directionally_admitted"] is False
