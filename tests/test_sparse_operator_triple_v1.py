import numpy as np

from anchor.corrected_sgta.audit_sparse_operator_triple_v1 import (
    h0_persistences,
    local_cfar,
    persistent_prominence,
)


def test_local_cfar_is_positive_affine_invariant() -> None:
    grid = np.arange(64, dtype=float).reshape(8, 8)
    grid[3, 4] += 20
    original = local_cfar(grid.reshape(-1), groups=1, side=8)
    transformed = local_cfar((3.25 * grid - 17).reshape(-1), groups=1, side=8)
    assert np.isclose(original, transformed)


def test_h0_lifetimes_are_translation_invariant_and_scale_equivariant() -> None:
    grid = np.zeros((5, 5), dtype=float)
    grid[1, 1] = 4
    grid[3, 3] = 3
    original = np.sort(h0_persistences(grid.reshape(-1), groups=1, side=5))
    translated = np.sort(h0_persistences((grid + 9).reshape(-1), groups=1, side=5))
    scaled = np.sort(h0_persistences((2 * grid).reshape(-1), groups=1, side=5))
    assert np.allclose(original, translated)
    assert np.allclose(2 * original, scaled)


def test_single_component_plateau_has_no_finite_persistence_signal() -> None:
    plateau = np.zeros((5, 5), dtype=float)
    plateau[1:4, 1:4] = 5
    assert persistent_prominence(plateau.reshape(-1), groups=1, side=5) == (0.0, 0.0)
