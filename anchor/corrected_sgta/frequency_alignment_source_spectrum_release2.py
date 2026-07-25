"""Hermitian-safe full-spectrum residual toward a model-source amplitude."""

from __future__ import annotations

import numpy as np
from PIL import Image

from corrected_sgta.frequency_alignment_release3 import resize_spectral_shape


def hermitian_project_shifted(amplitude: np.ndarray) -> np.ndarray:
    shifted = np.asarray(amplitude, dtype=np.float64)
    unshifted = np.fft.ifftshift(shifted, axes=(-2, -1))
    height, width = unshifted.shape[-2:]
    negative_h = (-np.arange(height)) % height
    negative_w = (-np.arange(width)) % width
    partner = np.take(np.take(unshifted, negative_h, axis=-2), negative_w, axis=-1)
    return np.fft.fftshift(0.5 * (unshifted + partner), axes=(-2, -1))


def source_spectrum_alignment_release2(
    image: Image.Image,
    target_amplitude: np.ndarray,
    low_frequency_ratio: float = 0.05,
    source_ratio: float = 0.0,
) -> Image.Image:
    """Blend every non-DC bin; compatibility `low_frequency_ratio` is alpha."""

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
    source_shifted = hermitian_project_shifted(
        resize_spectral_shape(source, height, width)
    )
    spectrum = np.fft.fft2(chw, axes=(-2, -1))
    target_shifted = np.fft.fftshift(np.abs(spectrum), axes=(-2, -1))
    center_h, center_w = height // 2, width // 2
    target_dc = target_shifted[:, center_h, center_w].copy()
    source_dc = source_shifted[:, center_h, center_w]
    source_shifted *= (target_dc / np.clip(source_dc, 1e-12, None))[:, None, None]
    source_shifted = hermitian_project_shifted(source_shifted)
    source_shifted[:, center_h, center_w] = target_dc
    blended = (1.0 - alpha) * target_shifted + alpha * source_shifted
    blended[:, center_h, center_w] = target_dc
    amplitude = np.fft.ifftshift(blended, axes=(-2, -1))
    reconstructed = np.fft.ifft2(
        amplitude * np.exp(1j * np.angle(spectrum)), axes=(-2, -1)
    )
    imaginary_max = float(np.max(np.abs(reconstructed.imag)))
    if imaginary_max > 1e-6:
        raise RuntimeError(f"Hermitian invariant failed: max imaginary={imaginary_max}")
    output = np.clip(
        np.rint(reconstructed.real.transpose(1, 2, 0)), 0, 255
    ).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


feddg_frequency_interpolation_source_spectrum_release2 = source_spectrum_alignment_release2

