"""Numerically robust capped-simplex weights for source-guided token alignment."""

from __future__ import annotations

import torch

from corrected_sgta import models_token_transport_release2 as implementation


def robust_capped_unit_mean_weights(
    probabilities: torch.Tensor, cap: float
) -> torch.Tensor:
    """Return nonnegative weights with exact unit mean and maximum ``cap``."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be [batch,tokens]")
    if cap < 1.0:
        raise ValueError("cap must be at least one for unit-mean weights")
    output_dtype = probabilities.dtype
    values = probabilities.double().clamp_min(torch.finfo(torch.float64).tiny)
    values = values / values.sum(dim=1, keepdim=True)
    _, tokens = values.shape
    capped = torch.zeros_like(values, dtype=torch.bool)
    weights = torch.zeros_like(values)
    for _ in range(tokens + 1):
        free = ~capped
        free_mass = (values * free).sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        remaining = (
            float(tokens) - float(cap) * capped.sum(dim=1, keepdim=True)
        ).clamp_min(0.0)
        proposal = values * (remaining / free_mass)
        newly_capped = free & (proposal > float(cap))
        if not bool(newly_capped.any()):
            weights = torch.where(
                capped, torch.full_like(proposal, float(cap)), proposal
            )
            break
        capped = capped | newly_capped
    else:
        raise RuntimeError("robust capped weight allocation did not converge")
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("non-finite capped token weights")
    if bool((weights > float(cap) + 1e-10).any()):
        raise RuntimeError("token weight cap violated")
    if not bool(torch.allclose(weights.mean(dim=1), torch.ones(weights.shape[0], device=weights.device, dtype=weights.dtype), rtol=1e-9, atol=1e-9)):
        raise RuntimeError("token weights do not have unit mean")
    return weights.to(output_dtype)


TokenTransportMixinRelease3 = implementation.TokenTransportMixinRelease2


def load_token_transport_adapter_release3(name: str):
    implementation.capped_unit_mean_weights = robust_capped_unit_mean_weights
    return implementation.load_token_transport_adapter_release2(name)
