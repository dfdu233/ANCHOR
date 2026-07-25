"""Phase-preserving full-spectrum residual toward one model-source amplitude."""

from __future__ import annotations

import numpy as np
from PIL import Image

from corrected_sgta.frequency_alignment_release3 import resize_spectral_shape


def source_spectrum_alignment(
    image: Image.Image,
    target_amplitude: np.ndarray,
    low_frequency_ratio: float = 0.05,
    source_ratio: float = 0.0,
) -> Image.Image:
    """Blend all non-DC amplitude bins while retaining the target image phase.

    ``low_frequency_ratio`` is retained only as a compatibility argument for the
    frozen inference harness; in this module it is the full-spectrum residual
    weight ``alpha``. ``source_ratio`` must remain zero.
    """

    alpha = float(low_frequency_ratio)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("full-spectrum residual alpha must be in [0, 1]")
    if float(source_ratio) != 0.0:
        raise ValueError("source_ratio is unused and must be exactly zero")

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    chw = rgb.transpose(2, 0, 1)
    source = np.asarray(target_amplitude, dtype=np.float64)
    if source.ndim == 2:
        source = np.repeat(source[None, ...], 3, axis=0)
    if source.ndim != 3 or source.shape[0] not in (1, 3):
        raise ValueError(f"target amplitude must be [H,W] or [C,H,W], got {source.shape}")
    if source.shape[0] == 1:
        source = np.repeat(source, 3, axis=0)

    height, width = chw.shape[-2:]
    source_shifted = resize_spectral_shape(source, height, width)
    spectrum = np.fft.fft2(chw, axes=(-2, -1))
    target_shifted = np.fft.fftshift(np.abs(spectrum), axes=(-2, -1))
    center_h, center_w = height // 2, width // 2
    target_dc = target_shifted[:, center_h, center_w].copy()
    source_dc = source_shifted[:, center_h, center_w]
    source_shifted *= (target_dc / np.clip(source_dc, 1e-12, None))[:, None, None]

    blended = (1.0 - alpha) * target_shifted + alpha * source_shifted
    blended[:, center_h, center_w] = target_dc
    amplitude = np.fft.ifftshift(blended, axes=(-2, -1))
    reconstructed = np.fft.ifft2(
        amplitude * np.exp(1j * np.angle(spectrum)), axes=(-2, -1)
    ).real
    output = np.clip(np.rint(reconstructed.transpose(1, 2, 0)), 0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


# Compatibility name used by the reconstruction audit.
feddg_frequency_interpolation_source_spectrum = source_spectrum_alignment

