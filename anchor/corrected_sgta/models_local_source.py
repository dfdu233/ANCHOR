"""Local Source Bank token-prototype transport for frozen medical VLMs."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from corrected_sgta.models_alignment import HuluAlignmentAdapter
from corrected_sgta.models_surface import LlavaMedSurfaceAdapter


class LocalSourceTransportMixin:
    @contextmanager
    def local_source_transport(self, prototypes: np.ndarray, beta: float, confidence_power: float = 2.0):
        original_encode = self.model.encode_images
        proto_np = np.asarray(prototypes, dtype=np.float32)

        def hooked_encode(*args, **kwargs):
            features = original_encode(*args, **kwargs)
            squeeze = False
            if features.ndim == 2:
                working = features.unsqueeze(0)
                squeeze = True
            elif features.ndim == 3:
                working = features
            else:
                raise RuntimeError(f'unexpected visual feature shape: {tuple(features.shape)}')
            proto = torch.as_tensor(proto_np, device=working.device, dtype=working.dtype)
            token_norm = working.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            token_unit = working.float() / token_norm
            proto_unit = proto.float() / proto.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            sim = torch.matmul(token_unit, proto_unit.T)
            nearest_sim, nearest = sim.max(dim=-1)
            target_dir = proto_unit.index_select(0, nearest.reshape(-1)).reshape(*nearest.shape, proto.shape[-1])
            target = target_dir.to(working.dtype) * token_norm.to(working.dtype)
            # High-confidence local matches move more; uncertain tokens remain close to original.
            confidence = nearest_sim.clamp_min(0.0).pow(float(confidence_power)).unsqueeze(-1).to(working.dtype)
            transported = working + float(beta) * confidence * (target - working)
            return transported.squeeze(0) if squeeze else transported

        self.model.encode_images = hooked_encode
        try:
            yield
        finally:
            self.model.encode_images = original_encode

    def forward_ce_local_transport(
        self,
        image: Image.Image,
        prompt: str,
        labels: Sequence[str],
        prototypes: np.ndarray,
        beta: float,
        confidence_power: float = 2.0,
    ):
        with self.local_source_transport(prototypes, beta, confidence_power):
            return self.forward_ce([image], prompt, labels)[0]

    def decode_ce_local_transport(
        self,
        image: Image.Image,
        prompt: str,
        prototypes: np.ndarray,
        beta: float,
        max_new_tokens: int,
        confidence_power: float = 2.0,
    ) -> str:
        with self.local_source_transport(prototypes, beta, confidence_power):
            return self.decode_ce([image], prompt, max_new_tokens=max_new_tokens)[0]


class HuluLocalSourceAdapter(LocalSourceTransportMixin, HuluAlignmentAdapter):
    @torch.inference_mode()
    def visual_tokens(self, images: Sequence[Image.Image]) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []
        for image in images:
            inputs = self._inputs(image, "Describe the medical image briefly.")
            features = self.model.encode_images(
                inputs["pixel_values"],
                inputs["grid_sizes"],
                inputs["merge_sizes"],
            )
            if features.ndim == 2:
                tokens = features
            elif features.ndim == 3:
                tokens = features[0]
            else:
                raise RuntimeError(f"unexpected encoded shape: {tuple(features.shape)}")
            outputs.append(tokens.float().cpu().numpy())
            del features, inputs
        return outputs


class LlavaLocalSourceAdapter(LocalSourceTransportMixin, LlavaMedSurfaceAdapter):
    @torch.inference_mode()
    def visual_tokens(self, images: Sequence[Image.Image]) -> list[np.ndarray]:
        from llava.mm_utils import process_images

        image_tensor = process_images(list(images), self.image_processor, self.model.config)
        outputs: list[np.ndarray] = []
        if isinstance(image_tensor, list):
            tensors = [item.to(self.model.device, dtype=self.model.dtype).unsqueeze(0) for item in image_tensor]
        else:
            tensors = [item.unsqueeze(0).to(self.model.device, dtype=self.model.dtype) for item in image_tensor]
        for tensor in tensors:
            encoded = self.model.encode_images(tensor)
            if encoded.ndim == 2:
                tokens = encoded
            elif encoded.ndim == 3:
                tokens = encoded[0]
            else:
                raise RuntimeError(f'unexpected encoded shape: {tuple(encoded.shape)}')
            outputs.append(tokens.float().cpu().numpy())
            del encoded, tensor
        return outputs


def load_local_source_adapter(name: str):
    normalized = name.lower().replace("_", "-")
    if normalized in {"hulu", "hulu-med", "hulu-med-14b"}:
        return HuluLocalSourceAdapter()
    if normalized in {"llava", "llava-med", "llava-med-v1.5"}:
        return LlavaLocalSourceAdapter()
    raise ValueError(f"unknown local source adapter: {name}")
