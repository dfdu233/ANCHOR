from __future__ import annotations

import numpy as np

from anchor.corrected_sgta.analyze_style_field_factorization import (
    displacement_factorization,
)


def make_features(delta: np.ndarray) -> np.ndarray:
    features = np.zeros(
        (delta.shape[0], delta.shape[1] + 2, delta.shape[2]),
        dtype=np.float64,
    )
    features[:, 2:] = delta
    return features


def test_style_only_field_is_fully_explained_by_style_offsets():
    style = np.asarray([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
    delta = np.broadcast_to(style, (5, *style.shape)).copy()
    result = displacement_factorization(make_features(delta))
    np.testing.assert_allclose(
        result["style_specific_offsets_explained"], 1.0
    )
    np.testing.assert_allclose(
        result["residual_after_optimal_style_offsets"], 0.0
    )


def test_case_only_field_is_not_explained_by_style_offsets_when_centered():
    case = np.asarray([[-1.0], [1.0], [-2.0], [2.0]])
    delta = np.broadcast_to(case[:, None], (4, 3, 1)).copy()
    result = displacement_factorization(make_features(delta))
    np.testing.assert_allclose(
        result["style_specific_offsets_explained"], 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        result["case_specific_offsets_explained"], 1.0
    )
