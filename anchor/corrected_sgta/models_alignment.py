"""Model adapters extended with pooled visual-encoder features for DG alignment."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image

from corrected_sgta.models_surface import HuluSurfaceAdapter, LlavaMedSurfaceAdapter


class HuluAlignmentAdapter(HuluSurfaceAdapter):
    @torch.inference_mode()
    def visual_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        pooled = []
        for image in images:
            inputs = self._inputs(image, "Describe the medical image briefly.")
            features = self.model.encode_images(
                inputs["pixel_values"],
                inputs["grid_sizes"],
                inputs["merge_sizes"],
            )
            pooled.append(features.float().mean(dim=0).cpu().numpy())
            del features, inputs
        return np.stack(pooled)


class LlavaMedAlignmentAdapter(LlavaMedSurfaceAdapter):
    @torch.inference_mode()
    def visual_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        from llava.mm_utils import process_images

        image_tensor = process_images(
            list(images), self.image_processor, self.model.config
        )
        if isinstance(image_tensor, list):
            pooled = []
            for tensor in image_tensor:
                encoded = self.model.encode_images(
                    tensor.unsqueeze(0).to(self.model.device, dtype=self.model.dtype)
                )
                pooled.append(encoded[0].float().mean(dim=0).cpu().numpy())
            return np.stack(pooled)
        encoded = self.model.encode_images(
            image_tensor.to(self.model.device, dtype=self.model.dtype)
        )
        return encoded.float().mean(dim=1).cpu().numpy()


def load_alignment_adapter(name: str):
    normalized = name.lower().replace("_", "-")
    if normalized in {"hulu", "hulu-med", "hulu-med-14b"}:
        return HuluAlignmentAdapter()
    if normalized in {"llava", "llava-med", "llava-med-v1.5"}:
        return LlavaMedAlignmentAdapter()
    raise ValueError(f"unknown model adapter: {name}")
