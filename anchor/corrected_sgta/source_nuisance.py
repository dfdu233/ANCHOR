"""Closed-form source nuisance subspace estimation and removal."""

from __future__ import annotations

import numpy as np
import torch


def fit_nuisance_subspace(
    clean_features: np.ndarray,
    shifted_features: np.ndarray,
    explained_variance: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a low-rank domain residual basis from paired source features.

    Rows in ``shifted_features`` may repeat each clean row for several
    label-preserving domain transforms. The residual matrix is intentionally
    not mean-centered: a systematic acquisition shift is itself nuisance.
    """
    clean = np.asarray(clean_features, dtype=np.float64)
    shifted = np.asarray(shifted_features, dtype=np.float64)
    if clean.ndim != 2 or shifted.ndim != 3:
        raise ValueError("expected clean [N,D] and shifted [N,K,D]")
    if clean.shape[0] != shifted.shape[0] or clean.shape[1] != shifted.shape[2]:
        raise ValueError("paired feature shapes do not match")
    if not 0.0 < explained_variance <= 1.0:
        raise ValueError("explained_variance must lie in (0,1]")
    if not np.isfinite(clean).all() or not np.isfinite(shifted).all():
        raise ValueError("features must be finite")

    residual = (shifted - clean[:, None, :]).reshape(-1, clean.shape[1])
    _, singular_values, right = np.linalg.svd(residual, full_matrices=False)
    energy = singular_values**2
    total = float(energy.sum())
    if total <= 0.0:
        raise ValueError("domain transforms produced a zero residual matrix")
    cumulative = np.cumsum(energy) / total
    rank = int(np.searchsorted(cumulative, explained_variance) + 1)
    basis = right[:rank].T
    diagnostics = {
        "rank": rank,
        "available_rank": int(len(singular_values)),
        "explained_variance": float(cumulative[rank - 1]),
        "residual_frobenius_norm": float(np.sqrt(total)),
        "orthogonality_error": float(
            np.linalg.norm(basis.T @ basis - np.eye(rank))
        ),
    }
    return (
        clean.mean(axis=0).astype(np.float32),
        basis.astype(np.float32),
        diagnostics,
    )


def remove_nuisance(
    features: torch.Tensor,
    source_mean: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Remove only the pooled displacement lying in the nuisance subspace."""
    squeeze = features.ndim == 2
    working = features.unsqueeze(0) if squeeze else features
    if working.ndim != 3:
        raise ValueError("expected projected tokens [T,D] or [B,T,D]")
    mean = source_mean.to(device=working.device, dtype=working.dtype).view(1, -1)
    directions = basis.to(device=working.device, dtype=working.dtype)
    pooled = working.mean(dim=1)
    coefficients = (pooled - mean) @ directions
    correction = coefficients @ directions.T
    projected = working - correction[:, None, :]
    return projected.squeeze(0) if squeeze else projected

