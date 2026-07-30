import numpy as np

from anchor.corrected_sgta.analyze_visual_evidence_chord_probe import (
    chord_projection,
    cross_validated_mse,
    split_half_style_reproducibility,
    style_case_variance_decomposition,
    susceptibility_correlations,
)
from anchor.corrected_sgta.run_visual_evidence_chord_probe import (
    candidate_sentences,
)


def test_chord_projection_recovers_scalar_and_zero_residual():
    null = np.asarray([1.0, -1.0, 0.5])
    real = np.asarray([3.0, 1.0, -0.5])
    styled = null + 0.25 * (real - null)
    alpha, predicted, residual = chord_projection(styled, real, null)
    assert np.isclose(alpha, 0.25)
    assert np.allclose(predicted, styled)
    assert np.allclose(residual, 0)


def test_style_offset_model_detects_reproducible_rotation():
    evidence = {}
    offset = np.asarray([0.5, -0.25])
    for index in range(5):
        null = np.asarray([0.1 * index, -0.2])
        real = null + np.asarray([1.0, 0.5])
        evidence[str(index)] = {
            "null": null,
            "real": real,
            "style_0": null + 0.6 * (real - null) + offset,
        }
    result = cross_validated_mse(evidence, ["style_0"])
    assert result["mean"]["style_offset"] < result["mean"]["chord"]


def test_diagonal_filter_recovers_concept_selective_attenuation():
    evidence = {}
    scale = np.asarray([0.2, 0.9])
    for index in range(8):
        null = np.asarray([0.01 * index, -0.03 * index])
        direction = np.asarray([1.0 + index / 10, 0.4 + index / 20])
        real = null + direction
        evidence[str(index)] = {
            "null": null,
            "real": real,
            "style_0": null + scale * direction,
        }
    result = cross_validated_mse(evidence, ["style_0"])
    assert result["mean"]["diagonal_filter"] < 1e-12
    assert result["mean"]["diagonal_filter"] < result["mean"]["chord"]


def test_candidate_sentences_are_complete_and_opposed():
    values = candidate_sentences("pleural effusion")
    assert values["positive"].endswith(".")
    assert "does not" in values["negative"]


def test_style_case_variance_decomposition_recovers_style_main_effect():
    evidence = {}
    style_offsets = {
        "style_0": np.asarray([1.0, 0.0]),
        "style_1": np.asarray([-1.0, 0.0]),
    }
    for index in range(5):
        real = np.asarray([0.1 * index, 0.2 * index])
        evidence[str(index)] = {
            "real": real,
            **{
                style: real + offset
                for style, offset in style_offsets.items()
            },
        }
    result = style_case_variance_decomposition(
        evidence, list(style_offsets)
    )
    fractions = result["fraction_of_centered_variance"]
    assert np.isclose(fractions["style"], 1.0)
    assert np.isclose(fractions["case"], 0.0)
    assert np.isclose(fractions["case_by_style"], 0.0)
    assert all(
        np.isclose(value["maximum_reducible_fraction"], 1.0)
        for value in result[
            "maximum_global_additive_correction_by_style"
        ].values()
    )


def test_susceptibility_detects_monotone_pixel_change():
    evidence = {}
    metrics = {}
    for index in range(5):
        real = np.asarray([1.0, -1.0])
        scale = 0.01 * (index + 1)
        evidence[str(index)] = {
            "real": real,
            "null": np.zeros(2),
            "style_0": real + np.asarray([scale, 0.0]),
        }
        metrics[(str(index), "style_0")] = {
            "mean_absolute_change": scale,
            "edge_correlation": 1.0 - scale,
        }
    result = susceptibility_correlations(
        evidence, metrics, ["style_0"]
    )
    assert np.isclose(
        result["correlations"]["mean_pixel_change"]["spearman_rho"],
        1.0,
    )


def test_split_half_certifies_shared_style_direction():
    evidence = {}
    for index in range(8):
        real = np.asarray([0.1 * index, -0.05 * index])
        evidence[str(index)] = {
            "real": real,
            "style_0": real + np.asarray([1.0, -0.5]),
        }
    result = split_half_style_reproducibility(
        evidence, ["style_0"], draws=100
    )
    style = result["by_style"]["style_0"]
    assert np.isclose(style["median_cosine"], 1.0)
    assert style["direction_certified"]
