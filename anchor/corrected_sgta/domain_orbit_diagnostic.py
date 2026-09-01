"""Linear-algebra utilities for falsifying single-image domain-orbit removal.

This module is intentionally diagnostic.  It makes the feature-space basis and
the orbit-mean degeneracy explicit; it does not claim that an augmentation
orbit identifies a clinically correct canonical point.
"""

from __future__ import annotations

import torch


def fit_feature_basis(orbit: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return orbit mean, right-singular feature basis, and explained fractions.

    ``orbit`` has shape ``[views, tokens, width]``.  Residuals are stacked over
    views and corresponding spatial tokens, so the returned basis lives in the
    projector feature space ``R^width``.
    """

    if orbit.ndim != 3 or orbit.shape[0] < 2:
        raise ValueError("orbit must have shape [views>=2, tokens, width]")
    if not 0 < rank <= min(orbit.shape[0] * orbit.shape[1], orbit.shape[2]):
        raise ValueError("rank is outside the residual matrix dimensions")
    working = orbit.float()
    center = working.mean(dim=0)
    residual = (working - center).reshape(-1, working.shape[-1])
    _, singular, basis = torch.pca_lowrank(residual, q=rank, center=False)
    energy = residual.square().sum().clamp_min(1e-12)
    explained = singular.square() / energy
    return center, basis, explained


def canonicalize(original: torch.Tensor, center: torch.Tensor, basis: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Replace the selected coordinates of ``original`` by orbit-center coordinates."""

    if original.shape != center.shape or original.ndim != 2:
        raise ValueError("original and center must share [tokens, width] shape")
    if basis.ndim != 2 or basis.shape[0] != original.shape[1]:
        raise ValueError("basis must have shape [width, rank]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    delta = original.float() - center.float()
    projected = (delta @ basis.float()) @ basis.float().T
    return original.float() - float(alpha) * projected


def project_residuals(residuals: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if residuals.ndim != 3 or basis.ndim != 2 or residuals.shape[-1] != basis.shape[0]:
        raise ValueError("expected residuals [views,tokens,width] and basis [width,rank]")
    return residuals.float() - (residuals.float() @ basis.float()) @ basis.float().T


def heldout_attenuation(original: torch.Tensor, heldout: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of held-out orbit residual energy removed by the fitted basis."""

    if heldout.ndim != 3 or heldout.shape[1:] != original.shape:
        raise ValueError("heldout must have shape [views, tokens, width]")
    residual = heldout.float() - original.float().unsqueeze(0)
    before = residual.square().sum().clamp_min(1e-12)
    after = project_residuals(residual, basis).square().sum()
    return float((1.0 - after / before).cpu())


def degeneration_ratio(candidate: torch.Tensor, original: torch.Tensor, center: torch.Tensor) -> float:
    """Zero means the candidate is the orbit mean; one means it is the original."""

    denominator = (original.float() - center.float()).norm().clamp_min(1e-12)
    return float(((candidate.float() - center.float()).norm() / denominator).cpu())


def random_basis(width: int, rank: int, *, seed: int, device: torch.device) -> torch.Tensor:
    if not 0 < rank <= width:
        raise ValueError("rank must lie in [1,width]")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    matrix = torch.randn(width, rank, generator=generator, device=device, dtype=torch.float32)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis


def token_stability_gate(
    original: torch.Tensor,
    instability: torch.Tensor,
    fraction: float,
    alpha: float,
    *,
    mode: str = "unstable",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attenuate a fixed fraction of tokens and restore the global Frobenius norm.

    ``mode`` selects the most unstable, least unstable, or random tokens.  The
    latter two are mechanism controls with the same token count and attenuation.
    """

    if original.ndim != 2 or instability.shape != (original.shape[0],):
        raise ValueError("expected original [tokens,width] and instability [tokens]")
    if not 0.0 < fraction <= 1.0 or not 0.0 <= alpha <= 1.0:
        raise ValueError("fraction must be in (0,1] and alpha in [0,1]")
    count = max(1, int(round(original.shape[0] * fraction)))
    if mode == "unstable":
        indices = torch.topk(instability, count, largest=True).indices
    elif mode == "stable":
        indices = torch.topk(instability, count, largest=False).indices
    elif mode == "random":
        generator = torch.Generator(device=original.device)
        generator.manual_seed(seed)
        indices = torch.randperm(original.shape[0], generator=generator, device=original.device)[:count]
    else:
        raise ValueError(f"unknown token-selection mode: {mode}")
    mask = torch.zeros(original.shape[0], dtype=torch.bool, device=original.device)
    mask[indices] = True
    candidate = original.float().clone()
    candidate[mask] *= 1.0 - float(alpha)
    target_norm = original.float().norm().clamp_min(1e-12)
    candidate *= target_norm / candidate.norm().clamp_min(1e-12)
    return candidate, mask
