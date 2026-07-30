import numpy as np

from anchor.corrected_sgta.analyze_residual_style_signature import (
    anova_energy,
    case_center_style_residual,
    cross_model_signature,
    cross_patient_style_metrics,
    radial_residual,
)


def test_case_centering_removes_common_residual() -> None:
    residual = np.arange(48, dtype=np.float64).reshape(4, 4, 3)
    centered = case_center_style_residual(residual)
    assert np.allclose(centered.mean(axis=1), 0.0)
    assert np.allclose(
        centered[:, 1] - centered[:, 0],
        residual[:, 1] - residual[:, 0],
    )


def test_radial_residual_is_orthogonal() -> None:
    evidence = np.zeros((4, 4, 3), dtype=np.float64)
    evidence[:, 0] = np.arange(4)[:, None]
    evidence[:, 2] = evidence[:, 0] + np.asarray([1.0, 0.5, 0.0])
    evidence[:, 3] = evidence[:, 0] + np.asarray([0.5, 1.0, 0.0])
    residual, radial_fraction = radial_residual(
        evidence, [f"patient-{index}" for index in range(4)]
    )
    assert residual.shape == (4, 2, 3)
    assert 0 <= radial_fraction <= 1


def test_style_prototypes_identify_shared_signatures() -> None:
    rng = np.random.default_rng(3)
    prototypes = np.eye(3)
    residual = np.stack(
        [
            prototypes + rng.normal(scale=0.01, size=prototypes.shape)
            for _ in range(8)
        ]
    )
    patients = [f"patient-{index}" for index in range(8)]
    metrics = cross_patient_style_metrics(residual, patients)
    assert metrics["style_identification_accuracy"] == 1.0
    assert metrics["style_prototype_r2_zero"] > 0.9
    energies = anova_energy(residual)
    assert np.isclose(sum(energies.values()), 1.0)


def test_cross_model_exact_assignment_detects_matching_styles() -> None:
    rng = np.random.default_rng(9)
    first = np.stack([np.eye(3) + rng.normal(scale=0.01, size=(3, 3))] * 5)
    second = np.stack([np.eye(3) + rng.normal(scale=0.01, size=(3, 3))] * 5)
    result = cross_model_signature(first, second)
    assert result["matched_style_mean_cosine"] > 0.99
    assert result["style_assignment_exact_p"] < 0.2
    assert np.asarray(result["first_style_signature_vectors"]).shape == (3, 3)
