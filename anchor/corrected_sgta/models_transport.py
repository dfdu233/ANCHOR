"""Single-step source-center transport hooks for frozen medical VLMs."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from corrected_sgta.models_alignment import HuluAlignmentAdapter, LlavaMedAlignmentAdapter


def transported_mean(feature: np.ndarray, center: np.ndarray, beta: float) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    target = center / max(float(np.linalg.norm(center)), 1e-12)
    target = target * float(np.linalg.norm(feature))
    return feature + float(beta) * (target - feature)


class TransportMixin:
    @contextmanager
    def source_transport(self, center: np.ndarray, beta: float):
        original_encode = self.model.encode_images
        center_array = np.asarray(center, dtype=np.float32)

        def hooked_encode(*args, **kwargs):
            features = original_encode(*args, **kwargs)
            target_direction = torch.as_tensor(
                center_array, device=features.device, dtype=features.dtype
            )
            target_direction = target_direction / target_direction.float().norm().clamp_min(1e-12).to(features.dtype)
            if features.ndim == 3:
                mean = features.mean(dim=1, keepdim=True)
                target = target_direction.view(1, 1, -1) * mean.float().norm(dim=-1, keepdim=True).to(features.dtype)
            elif features.ndim == 2:
                mean = features.mean(dim=0, keepdim=True)
                target = target_direction.view(1, -1) * mean.float().norm(dim=-1, keepdim=True).to(features.dtype)
            else:
                raise RuntimeError(f"unexpected visual feature shape: {tuple(features.shape)}")
            return features + float(beta) * (target - mean)

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


class HuluTransportAdapter(TransportMixin, HuluAlignmentAdapter):
    pass


class LlavaTransportAdapter(TransportMixin, LlavaMedAlignmentAdapter):
    pass


def load_transport_adapter(name: str):
    if name == "hulu":
        return HuluTransportAdapter()
    if name == "llava":
        return LlavaTransportAdapter()
    raise ValueError(name)
