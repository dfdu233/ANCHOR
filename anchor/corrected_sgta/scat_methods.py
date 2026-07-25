"""SCA-T-compatible TIM adaptation for generative VLM hidden states.

SCA-T was designed for CLIP features and fixed text prototypes.  Here the
mathematically corresponding fixed-class surface is the last multimodal prompt
hidden state and semantic token rows of the language-model output embedding.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def fit_logit_scale(
    features: np.ndarray, prototypes: np.ndarray, reference_logits: np.ndarray
) -> float:
    """Fit SCA-T's one positive scale without using labels."""

    x = F.normalize(torch.as_tensor(features, dtype=torch.float32), dim=-1)
    w = F.normalize(torch.as_tensor(prototypes, dtype=torch.float32), dim=-1)
    cosine = x @ w.T
    cosine = cosine - cosine.mean(dim=-1, keepdim=True)
    target = torch.as_tensor(reference_logits, dtype=torch.float32)
    target = target - target.mean(dim=-1, keepdim=True)
    denominator = float((cosine * cosine).sum())
    if denominator <= 1e-12:
        raise ValueError("degenerate feature/prototype geometry")
    scale = float((cosine * target).sum()) / denominator
    return max(scale, 1e-4)


def tim_probabilities(
    features: np.ndarray,
    prototypes: np.ndarray,
    logit_scale: float,
    iterations: int = 100,
    learning_rate: float = 0.01,
    observed_marginal: np.ndarray | None = None,
    entropy_weight: float = 1.0,
    eps: float = 1e-3,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Apply the SCA-T repository's full-batch TIM/TIM(KL) objective.

    Passing ``observed_marginal=None`` reproduces TIM's uniform marginal
    entropy term.  Passing calibration class counts/proportions reproduces
    TIM(KL).  No labels enter the per-sample prediction loss.
    """

    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    device = torch.device(device)
    x = F.normalize(
        torch.as_tensor(features, dtype=torch.float32, device=device), dim=-1
    )
    initial = F.normalize(
        torch.as_tensor(prototypes, dtype=torch.float32, device=device), dim=-1
    )
    trainable = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([trainable], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, iterations)
    )
    if observed_marginal is None:
        marginal = torch.full(
            (initial.shape[0],), 1.0 / initial.shape[0], device=device
        )
    else:
        marginal = torch.as_tensor(
            observed_marginal, dtype=torch.float32, device=device
        )
        if marginal.numel() != initial.shape[0] or float(marginal.sum()) <= 0:
            raise ValueError("invalid observed marginal")
        marginal = marginal / marginal.sum()
    scale = float(logit_scale)
    for _ in range(iterations):
        normalized = F.normalize(trainable, dim=-1)
        logits = scale * (x @ normalized.T)
        # Upstream forms softmax and then divides by its mean. Large
        # generative-LM scales can underflow one class to exactly zero,
        # turning the otherwise valid KL term into inf or NaN. Compute the
        # equivalent probabilities and class marginal in log space.
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        log_class_marginal = torch.logsumexp(log_probabilities, dim=0) - math.log(
            x.shape[0]
        )
        class_marginal = log_class_marginal.exp()
        conditional_entropy = -torch.mean(
            torch.sum(probabilities * log_probabilities, dim=-1)
        )
        if observed_marginal is None:
            marginal_loss = torch.sum(class_marginal * log_class_marginal)
        else:
            positive = marginal > 0
            marginal_loss = torch.sum(
                marginal[positive]
                * (torch.log(marginal[positive]) - log_class_marginal[positive])
            )
        loss = marginal_loss + entropy_weight * conditional_entropy
        if not torch.isfinite(loss):
            raise RuntimeError("TIM produced a non-finite optimization loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
    with torch.no_grad():
        logits = scale * (x @ F.normalize(trainable, dim=-1).T)
        output = logits.softmax(-1).cpu().numpy()
    if not np.isfinite(output).all():
        raise RuntimeError("TIM produced non-finite probabilities")
    return output


def scale_as_log_parameter(scale: float) -> float:
    """Return the log-scale representation used by the upstream Adapter."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    return math.log(scale)
