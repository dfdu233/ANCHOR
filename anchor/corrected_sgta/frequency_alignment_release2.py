"""Mean-preserving source spectral-shape alignment for medical images."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from corrected_sgta.frequency_alignment_v2 import resize_centered_amplitude


def feddg_frequency_interpolation_release2(
    image: Image.Image,
    target_amplitude: np.ndarray,
    low_frequency_ratio: float = 0.003,
    source_ratio: float = 0.0,
) -> Image.Image:
    """Align normalized spectral shape while exactly retaining image DC/mean.

    Raw DC magnitude mostly reflects display scaling rather than acquisition
    style. The target spectrum is therefore normalized to each image channel's
    DC before low-frequency interpolation. This keeps the FedDG phase-preserving
    intervention but avoids brightness shifts that confound clinical safety.
    """

    if not 0 <= low_frequency_ratio <= 0.5:
        raise ValueError("low_frequency_ratio must be in [0, 0.5]")
    if not 0 <= source_ratio <= 1:
        raise ValueError("source_ratio must be in [0, 1]")
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    chw = rgb.transpose(2, 0, 1)
    target = np.asarray(target_amplitude, dtype=np.float64)
    if target.ndim == 2:
        target = np.repeat(target[None, ...], 3, axis=0)
    if target.ndim != 3 or target.shape[0] not in (1, 3):
        raise ValueError(f"target amplitude must be [H,W] or [C,H,W], got {target.shape}")
    if target.shape[0] == 1:
        target = np.repeat(target, 3, axis=0)
    height, width = chw.shape[-2:]
    target_shifted = resize_centered_amplitude(target, height, width)
    fft = np.fft.fft2(chw, axes=(-2, -1))
    amplitude_shifted = np.fft.fftshift(np.abs(fft), axes=(-2, -1))
    center_h, center_w = height // 2, width // 2
    original_dc = amplitude_shifted[:, center_h, center_w]
    target_dc = target_shifted[:, center_h, center_w]
    target_shifted *= (
        original_dc / np.clip(target_dc, 1e-12, None)
    )[:, None, None]
    target_shifted[:, center_h, center_w] = original_dc
    radius = int(math.floor(min(height, width) * low_frequency_ratio))
    hs, he = center_h - radius, center_h + radius + 1
    ws, we = center_w - radius, center_w + radius + 1
    amplitude_shifted[:, hs:he, ws:we] = (
        source_ratio * amplitude_shifted[:, hs:he, ws:we]
        + (1.0 - source_ratio) * target_shifted[:, hs:he, ws:we]
    )
    amplitude_shifted[:, center_h, center_w] = original_dc
    amplitude = np.fft.ifftshift(amplitude_shifted, axes=(-2, -1))
    reconstructed = np.fft.ifft2(
        amplitude * np.exp(1j * np.angle(fft)), axes=(-2, -1)
    ).real
    out = np.clip(np.rint(reconstructed.transpose(1, 2, 0)), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")
