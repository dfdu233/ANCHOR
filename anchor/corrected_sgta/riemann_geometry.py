"""Small deterministic geometry utilities for ANCHOR-Riemann gates."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from corrected_sgta.anchor_transport import (
    deterministic_directions,
    sliced_wasserstein_squared,
)


def dirichlet_energy(
    trajectory: np.ndarray,
    metric: np.ndarray | None = None,
) -> float:
    """Discrete path energy of a token evidence trajectory.

    The default metric is Euclidean.  A caller may pass a positive-semidefinite
    source metric, e.g. a diagonal Fisher proxy estimated from source
    trajectories.  The function is intentionally tiny and auditable: it is an
    analysis energy, not a learned verifier.
    """

    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("trajectory must be a finite non-empty [tokens,dim] array")
    if values.shape[0] == 1:
        return 0.0
    diffs = np.diff(values, axis=0)
    if metric is None:
        energy = float(np.square(diffs).sum())
    else:
        matrix = np.asarray(metric, dtype=np.float64)
        if matrix.shape != (values.shape[1], values.shape[1]):
            raise ValueError("metric shape does not match trajectory dimension")
        if not np.isfinite(matrix).all():
            raise ValueError("metric must be finite")
        energy = float(np.einsum("ti,ij,tj->", diffs, matrix, diffs))
    if energy < -1e-12 or not math.isfinite(energy):
        raise FloatingPointError("invalid Dirichlet energy")
    return max(0.0, energy)


def diagonal_fisher_metric(
    source_trajectories: Sequence[np.ndarray],
    ridge: float = 1e-3,
) -> np.ndarray:
    """Return a diagonal source metric for token evidence coordinates.

    Coordinates that are stable on source correct trajectories get larger
    weight.  This is a conservative Fisher-style proxy: it avoids estimating a
    dense metric from small source banks while retaining the Riemannian story.
    """

    if ridge <= 0 or not math.isfinite(ridge):
        raise ValueError("ridge must be positive and finite")
    arrays = [np.asarray(value, dtype=np.float64) for value in source_trajectories]
    if not arrays:
        raise ValueError("source trajectories are required")
    pooled = np.concatenate(arrays, axis=0)
    if pooled.ndim != 2 or pooled.shape[0] < 2 or not np.isfinite(pooled).all():
        raise ValueError("source trajectories must form a finite [tokens,dim] array")
    variance = np.var(pooled, axis=0)
    weights = 1.0 / (variance + ridge)
    weights = weights / float(np.mean(weights))
    return np.diag(weights)


def nearest_manifold_distance(
    trajectory: np.ndarray,
    source_trajectories: Sequence[np.ndarray],
    *,
    directions: np.ndarray | None = None,
    quantiles: int = 32,
) -> tuple[float, int, list[float]]:
    """Distance from one empirical trajectory to a discrete source manifold."""

    if not source_trajectories:
        raise ValueError("source manifold is empty")
    if directions is None:
        directions = deterministic_directions(np.asarray(trajectory).shape[1])
    distances = [
        sliced_wasserstein_squared(
            trajectory, source, directions=directions, quantiles=quantiles
        )
        for source in source_trajectories
    ]
    index = int(np.argmin(np.asarray(distances, dtype=np.float64)))
    return float(distances[index]), index, [float(value) for value in distances]


def stable_ranked_permutation(count: int, seed: int) -> np.ndarray:
    """Deterministic random-manifold control."""

    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    return rng.permutation(count)


def zscore(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("zscore expects a finite vector")
    scale = float(array.std())
    if scale < 1e-12:
        return [0.0 for _ in array]
    return ((array - float(array.mean())) / scale).astype(float).tolist()


def riemann_code_identity() -> dict[str, Any]:
    return {
        "version": "anchor-riemann-geometry-v1",
        "geometry": [
            "token evidence empirical measures",
            "nearest sliced-Wasserstein source manifold distance",
            "diagonal Fisher-proxy Dirichlet energy",
        ],
        "target_labels_used_for_selection": False,
        "uses_canonical_label_logits_for_prediction": False,
    }
