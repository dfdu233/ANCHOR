from __future__ import annotations

import numpy as np

from anchor.corrected_sgta.analyze_layerwise_style_orbit import (
    linear_cka,
    participation_rank,
    per_case_susceptibility,
    variance_decomposition,
)
from anchor.corrected_sgta.run_layerwise_style_orbit_probe import parse_variant


def test_parse_variant_supports_base_and_checkpoint(tmp_path):
    assert parse_variant("base") == ("base", None)
    checkpoint = tmp_path / "merger.pt"
    checkpoint.write_bytes(b"checkpoint")
    name, path = parse_variant(f"matched={checkpoint}")
    assert name == "matched"
    assert path == checkpoint.resolve()


def test_susceptibility_is_zero_for_identical_style_views():
    features = np.zeros((3, 4, 5), dtype=np.float32)
    features[:, 0] = 1.0
    features[:, 1] = -1.0
    features[:, 2:] = features[:, [0]]
    np.testing.assert_allclose(per_case_susceptibility(features), 0.0)


def test_variance_decomposition_sums_to_one():
    generator = np.random.default_rng(7)
    features = generator.normal(size=(8, 5, 12))
    fractions = variance_decomposition(features)
    assert set(fractions) == {"case", "style", "case_by_style"}
    np.testing.assert_allclose(sum(fractions.values()), 1.0)


def test_participation_rank_detects_single_direction():
    generator = np.random.default_rng(3)
    direction = generator.normal(size=16)
    coefficients = generator.normal(size=(10, 4, 1))
    features = np.zeros((10, 6, 16), dtype=np.float64)
    features[:, 2:] = coefficients * direction
    assert participation_rank(features) < 1.01


def test_linear_cka_is_one_for_identical_matrices():
    generator = np.random.default_rng(11)
    values = generator.normal(size=(20, 6))
    np.testing.assert_allclose(linear_cka(values, values), 1.0)
