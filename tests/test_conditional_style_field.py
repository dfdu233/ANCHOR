import numpy as np

from anchor.corrected_sgta.analyze_conditional_style_field import (
    anova_factorization,
    crossed_cell_prediction,
    displacement_tensor,
    endpoint_projection_controls,
    kernel_ridge_predict,
    null_prior_projection,
)


def test_anova_recovers_additive_case_and_style_field() -> None:
    rng = np.random.default_rng(4)
    grand = rng.normal(size=(1, 1, 7))
    case = rng.normal(size=(5, 1, 7))
    case -= case.mean(axis=0, keepdims=True)
    style = rng.normal(size=(1, 4, 7))
    style -= style.mean(axis=1, keepdims=True)
    delta = grand + case + style
    result = anova_factorization(delta)
    assert np.isclose(result["interaction_energy_fraction"], 0.0, atol=1e-12)
    assert np.isclose(result["sum"], 1.0)


def test_crossed_cell_prediction_is_exact_for_additive_field() -> None:
    rng = np.random.default_rng(5)
    grand = rng.normal(size=(1, 1, 6))
    case = rng.normal(size=(6, 1, 6))
    style = rng.normal(size=(1, 3, 6))
    delta = grand + case + style
    result = crossed_cell_prediction(
        delta, [f"patient-{index}" for index in range(len(delta))]
    )
    assert np.isclose(result["crossed_cell_r2_zero"], 1.0)


def test_displacement_tensor_uses_real_view_as_origin() -> None:
    features = np.zeros((2, 5, 3))
    features[:, 0] = 2.0
    features[:, 2:] = 5.0
    assert np.all(displacement_tensor(features) == 3.0)


def test_kernel_ridge_predicts_linear_targets() -> None:
    x = np.arange(8, dtype=float)[:, None]
    y = np.concatenate([2 * x, -3 * x], axis=1)
    prediction = kernel_ridge_predict(x, y, x, 1e-8)
    assert np.allclose(prediction, y, atol=1e-5)


def test_null_projection_detects_exact_prior_directed_drift() -> None:
    features = np.zeros((4, 5, 3))
    features[:, 0] = np.asarray([1.0, 2.0, 3.0])
    features[:, 1] = 0.0
    features[:, 2:] = 0.5 * features[:, [0]]
    result = null_prior_projection(features, permutation_repeats=8)
    assert np.isclose(result["same_case_null_projection_energy"], 1.0)
    assert np.isclose(result["fraction_toward_null"], 1.0)


def test_endpoint_controls_report_centroid_and_null_directions() -> None:
    rng = np.random.default_rng(12)
    features = rng.normal(size=(5, 5, 4))
    result = endpoint_projection_controls(
        features, [f"patient-{index}" for index in range(5)]
    )
    assert 0 <= result["clean_centroid_projection_energy"] <= 1
    assert 0 <= result["null_projection_energy"] <= 1
    assert -1 <= result["median_null_centroid_direction_cosine"] <= 1
