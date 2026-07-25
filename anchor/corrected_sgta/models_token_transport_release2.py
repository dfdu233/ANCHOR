"""Source-guided token alignment with an exact capped, unit-mean weighting."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from corrected_sgta.models_alignment import HuluAlignmentAdapter, LlavaMedAlignmentAdapter


def capped_unit_mean_weights(probabilities: torch.Tensor, cap: float) -> torch.Tensor:
    """Scale probabilities to sum T under an exact per-token upper bound."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be [batch,tokens]")
    if cap < 1.0:
        raise ValueError("cap must be at least one for unit-mean weights")
    batch, tokens = probabilities.shape
    capped = torch.zeros_like(probabilities, dtype=torch.bool)
    weights = torch.zeros_like(probabilities)
    for _ in range(tokens + 1):
        free = ~capped
        free_mass = (probabilities * free).sum(dim=1, keepdim=True).clamp_min(1e-20)
        remaining = (
            float(tokens) - float(cap) * capped.sum(dim=1, keepdim=True)
        ).clamp_min(0.0)
        scale = remaining / free_mass
        proposal = probabilities * scale
        newly_capped = free & (proposal > float(cap))
        if not bool(newly_capped.any()):
            weights = torch.where(capped, torch.full_like(proposal, float(cap)), proposal)
            break
        capped = capped | newly_capped
    else:
        raise RuntimeError("capped weight allocation did not converge")
    return weights


class TokenTransportMixinRelease2:
    transport_temperature = 0.1
    transport_weight_cap = 4.0

    @contextmanager
    def source_transport(self, center: np.ndarray, beta: float):
        original_encode = self.model.encode_images
        center_array = np.asarray(center, dtype=np.float32)

        def hooked_encode(*args, **kwargs):
            features = original_encode(*args, **kwargs)
            center_unit = torch.as_tensor(
                center_array, device=features.device, dtype=features.dtype
            )
            center_unit = center_unit / center_unit.float().norm().clamp_min(1e-12).to(
                features.dtype
            )
            if features.ndim == 2:
                working = features.unsqueeze(0)
                squeeze = True
            elif features.ndim == 3:
                working = features
                squeeze = False
            else:
                raise RuntimeError(f"unexpected visual feature shape: {tuple(features.shape)}")
            mean = working.mean(dim=1, keepdim=True)
            target = center_unit.view(1, 1, -1) * mean.float().norm(
                dim=-1, keepdim=True
            ).to(working.dtype)
            similarities = F.cosine_similarity(
                working.float(), center_unit.float().view(1, 1, -1), dim=-1
            )
            probabilities = torch.softmax(
                similarities / float(self.transport_temperature), dim=1
            )
            weights = capped_unit_mean_weights(
                probabilities, float(self.transport_weight_cap)
            )
            transported = working + float(beta) * weights.unsqueeze(-1).to(
                working.dtype
            ) * (target - mean)
            return transported.squeeze(0) if squeeze else transported

        self.model.encode_images = hooked_encode
        try:
            yield
        finally:
            self.model.encode_images = original_encode

    def forward_ce_transport(
        self,
        image: Image.Image,
        prompt: str,
        labels: Sequence[str],
        center: np.ndarray,
        beta: float,
    ):
        with self.source_transport(center, beta):
            return self.forward_ce([image], prompt, labels)[0]

    def decode_ce_transport(
        self,
        image: Image.Image,
        prompt: str,
        center: np.ndarray,
        beta: float,
        max_new_tokens: int,
    ) -> str:
        with self.source_transport(center, beta):
            return self.decode_ce([image], prompt, max_new_tokens=max_new_tokens)[0]


class HuluTokenTransportAdapterRelease2(TokenTransportMixinRelease2, HuluAlignmentAdapter):
    pass


class LlavaTokenTransportAdapterRelease2(
    TokenTransportMixinRelease2, LlavaMedAlignmentAdapter
):
    pass


def load_token_transport_adapter_release2(name: str):
    if name == "hulu":
        return HuluTokenTransportAdapterRelease2()
    if name == "llava":
        return LlavaTokenTransportAdapterRelease2()
    raise ValueError(name)
