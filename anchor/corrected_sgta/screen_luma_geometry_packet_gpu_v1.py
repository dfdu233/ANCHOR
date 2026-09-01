#!/usr/bin/env python3
"""GPU launcher for the frozen luma-geometry fatal screen.

The statistical protocol and image construction live in
``screen_luma_geometry_packet_v1``.  This launcher only moves the already
frozen BiomedCLIP encoders to CUDA so a baseline-safe method-search window can
finish the preregistered screen quickly.  It does not change the model,
features, probes, controls, or gate.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from anchor.corrected_sgta import screen_luma_geometry_packet_v1 as protocol


class CudaBiomedTower(protocol.BiomedTower):
    def __init__(self, root, threads):
        super().__init__(root, threads)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.device = torch.device("cuda:0")
        self.model.to(self.device)
        self.provenance["execution_device"] = str(self.device)

    def image(self, images: list[np.ndarray]) -> np.ndarray:
        with torch.inference_mode():
            tensor = self.tensor(images).to(self.device, non_blocking=True)
            features = self.model.encode_image(tensor, normalize=True)
        return features.cpu().numpy().astype(np.float32)

    def text_directions(self) -> np.ndarray:
        prompts = []
        for finding in protocol.FINDINGS:
            prompts.extend(
                [
                    f"a frontal chest radiograph showing {protocol.DISPLAY[finding]}",
                    f"a frontal chest radiograph without {protocol.DISPLAY[finding]}",
                ]
            )
        tokens = self.tokenizer(prompts, context_length=256).to(self.device)
        with torch.inference_mode():
            encoded = self.model.encode_text(tokens, normalize=True)
        encoded = encoded.cpu().numpy().reshape(len(protocol.FINDINGS), 2, -1)
        return encoded[:, 0] - encoded[:, 1]


if __name__ == "__main__":
    # The CPU protocol rejects an explicitly visible CUDA device.  Deleting
    # the selector keeps the process on the already assigned physical GPU;
    # the subclass above is the only behavioral change.
    os.environ["LUMA_GEOMETRY_ALLOW_GPU"] = "1"
    protocol.BiomedTower = CudaBiomedTower
    protocol.main()
