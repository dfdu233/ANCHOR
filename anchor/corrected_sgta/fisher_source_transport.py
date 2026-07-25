"""Question-conditioned source transport in the decoder pullback geometry."""

from __future__ import annotations

import numpy as np
import torch


EPS = 1e-10


def pca_geometry(values: np.ndarray, explained_variance: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical mean and smallest PCA basis reaching the target variance."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < 2:
        raise ValueError("source features must have shape [N,D] with N >= 2")
    mean = array.mean(axis=0)
    centered = array - mean
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    energy = np.square(singular_values)
    cumulative = np.cumsum(energy) / max(float(energy.sum()), EPS)
    rank = int(np.searchsorted(cumulative, explained_variance) + 1)
    return mean.astype(np.float32), right[:rank].T.astype(np.float32)


def project(basis: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Orthogonally project a vector onto a column-orthonormal basis."""

    return basis @ (basis.T @ vector)


def fisher_matrix(probabilities: torch.Tensor) -> torch.Tensor:
    values = probabilities.float()
    return torch.diag(values) - torch.outer(values, values)


def question_conditioned_direction(
    jacobian: torch.Tensor,
    probabilities: torch.Tensor,
    basis: torch.Tensor,
    residual: torch.Tensor,
    dose_fraction: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a source-supported, decoder-visible direction with a fixed dose."""

    source_component = project(basis, residual.float())
    fisher = fisher_matrix(probabilities)
    response = jacobian @ source_component
    pullback = jacobian.T @ (fisher @ response)
    visible_component = project(basis, pullback)
    source_norm = source_component.norm()
    visible_norm = visible_component.norm()
    dose = dose_fraction * source_norm
    if float(source_norm) <= EPS or float(visible_norm) <= EPS:
        delta = torch.zeros_like(residual, dtype=torch.float32)
    else:
        delta = visible_component * (dose / visible_norm)
    alignment = torch.dot(delta, source_component)
    diagnostics = {
        "source_component_norm": float(source_norm.detach().cpu()),
        "visible_component_norm": float(visible_norm.detach().cpu()),
        "dose": float(delta.norm().detach().cpu()),
        "source_alignment": float(alignment.detach().cpu()),
        "fisher_energy": float(
            ((jacobian @ delta) @ fisher @ (jacobian @ delta)).detach().cpu()
        ),
    }
    return delta, diagnostics


def decoder_pullback_source_projection(
    jacobian: torch.Tensor,
    probabilities: torch.Tensor,
    tangent_basis: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project the source-normal residual onto its decoder-visible direction.

    ``tangent_basis`` spans the affine source manifold. The returned shift is
    the projection of the normal residual onto ``P_perp G P_perp r``, where
    ``G = J.T @ F @ J`` is the categorical Fisher pullback.
    """

    normal_residual = residual.float() - project(tangent_basis, residual.float())
    fisher = fisher_matrix(probabilities)
    response = jacobian @ normal_residual
    pullback = jacobian.T @ (fisher @ response)
    direction = pullback - project(tangent_basis, pullback)
    alignment = torch.dot(normal_residual, direction).clamp_min(0.0)
    squared_norm = torch.dot(direction, direction)
    if float(alignment) <= EPS or float(squared_norm) <= EPS:
        delta = torch.zeros_like(normal_residual)
    else:
        delta = direction * (alignment / (squared_norm + EPS))
    before = normal_residual.norm()
    after = (normal_residual - delta).norm()
    diagnostics = {
        "normal_residual_norm": float(before.detach().cpu()),
        "decoder_visible_norm": float(direction.norm().detach().cpu()),
        "alignment": float(alignment.detach().cpu()),
        "dose": float(delta.norm().detach().cpu()),
        "distance_after": float(after.detach().cpu()),
        "relative_closure": float(
            ((before - after) / before.clamp_min(EPS)).detach().cpu()
        ),
        "fisher_energy": float(
            ((jacobian @ delta) @ fisher @ (jacobian @ delta)).detach().cpu()
        ),
    }
    return delta, diagnostics


def equal_dose(vector: torch.Tensor, dose: float) -> torch.Tensor:
    norm = vector.float().norm()
    if float(norm) <= EPS or dose <= EPS:
        return torch.zeros_like(vector, dtype=torch.float32)
    return vector.float() * (dose / norm)


def source_closure(
    pooled: torch.Tensor, center: torch.Tensor, delta: torch.Tensor
) -> dict[str, float]:
    residual = center.float() - pooled.float()
    before = residual.norm()
    after = (residual - delta.float()).norm()
    return {
        "distance_before": float(before.detach().cpu()),
        "distance_after": float(after.detach().cpu()),
        "relative_closure": float(((before - after) / before.clamp_min(EPS)).detach().cpu()),
    }
