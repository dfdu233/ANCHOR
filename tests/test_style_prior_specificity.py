import numpy as np

from anchor.corrected_sgta.analyze_style_prior_specificity import (
    exact_disease_assignment,
    patient_cluster_bootstrap_uniform_fraction,
    joint_nuisance_residual,
    patient_blocked_spectrum_permutation,
    patient_cluster_bootstrap_alignment,
    signature_spectrum,
    split_uniform_disease_axis,
)


def test_uniform_axis_split_is_orthogonal_and_exact() -> None:
    rng = np.random.default_rng(11)
    field = rng.normal(size=(5, 4, 6))
    uniform, contrast, fraction = split_uniform_disease_axis(field)
    assert np.allclose(field, uniform + contrast)
    assert 0 <= fraction <= 1
    axis = np.ones(6) / np.sqrt(6)
    assert np.allclose(np.einsum("...d,d->...", contrast, axis), 0)


def test_disease_assignment_prefers_matching_coordinates() -> None:
    rng = np.random.default_rng(13)
    basis = np.eye(4)
    first = np.stack([basis + rng.normal(scale=0.001, size=(4, 4))] * 5)
    second = np.stack([basis + rng.normal(scale=0.001, size=(4, 4))] * 5)
    result = exact_disease_assignment(first, second)
    assert result["matched_disease_mean_cosine"] > 0.99
    assert result["disease_assignment_exact_p"] < 0.2
    assert result["assignment_count"] == 24


def test_signature_spectrum_reports_rank_one() -> None:
    style = np.asarray([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])
    field = np.stack([style] * 7)
    result = signature_spectrum(field)
    assert result["energy_proportions"][0] > 0.999
    assert np.isclose(result["participation_effective_rank"], 1.0)


def test_patient_bootstrap_preserves_strong_paired_alignment() -> None:
    rng = np.random.default_rng(17)
    first = rng.normal(size=(12, 3, 3))
    second = first + rng.normal(scale=0.01, size=first.shape)
    result = patient_cluster_bootstrap_alignment(
        first,
        second,
        [f"p-{index}" for index in range(12)],
        repeats=50,
        seed=19,
    )
    assert (
        result["matched_style_mean_cosine"][
            "patient_cluster_bootstrap_95pct"
        ][0]
        > 0.9
    )
    assert (
        result["style_identity_assignment_margin"][
            "bootstrap_positive_fraction"
        ]
        > 0.9
    )


def test_spectrum_permutation_returns_bounded_p_values() -> None:
    rng = np.random.default_rng(23)
    field = rng.normal(size=(10, 4, 4))
    result = patient_blocked_spectrum_permutation(
        field,
        [f"p-{index}" for index in range(10)],
        repeats=20,
        seed=29,
    )
    for value in result["null"].values():
        assert 0 < value["one_sided_p"] <= 1


def test_uniform_fraction_bootstrap_detects_contrast_field() -> None:
    style = np.asarray([[1.0, -1.0], [-1.0, 1.0]])
    field = np.stack([style] * 8)
    result = patient_cluster_bootstrap_uniform_fraction(
        field,
        [f"p-{index}" for index in range(8)],
        repeats=20,
        seed=31,
    )
    assert result["observed"] < 1e-10
    assert result["patient_cluster_bootstrap_95pct"][1] < 1e-10


def test_joint_nuisance_residual_removes_radial_and_uniform_axes() -> None:
    rng = np.random.default_rng(37)
    evidence = rng.normal(size=(6, 5, 4))
    patients = [f"p-{index}" for index in range(6)]
    residual, fraction = joint_nuisance_residual(evidence, patients)
    assert residual.shape == (6, 3, 4)
    assert 0 <= fraction <= 1
    axis = np.ones(4) / 2
    assert np.allclose(np.einsum("csd,d->cs", residual, axis), 0)
