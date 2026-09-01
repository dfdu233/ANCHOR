"""Minimal source-envelope calibration on model-visible image pixels."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
DEFAULT_SIZE = 336
DEFAULT_BINS = 64


def stable_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def model_visible_image(
    image: Image.Image,
    *,
    size: int = DEFAULT_SIZE,
    image_mean: tuple[float, float, float] = CLIP_IMAGE_MEAN,
) -> Image.Image:
    """Match Huatuo's square padding and CLIP resize before normalization."""
    image = image.convert("RGB")
    width, height = image.size
    side = max(width, height)
    background = tuple(int(value * 255) for value in image_mean)
    canvas = Image.new("RGB", (side, side), background)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    if side != size:
        canvas = canvas.resize((size, size), Image.Resampling.BICUBIC)
    return canvas


def gamma_style_shift(image: Image.Image, gamma: float) -> Image.Image:
    """Apply a deterministic intensity style shift on model-visible pixels."""
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if math.isclose(gamma, 1.0):
        return image.copy()
    array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    shifted = np.clip(array, 0.0, 1.0) ** gamma
    return Image.fromarray(
        np.rint(shifted * 255.0).astype(np.uint8), mode="RGB"
    )


def _radial_bin_map(height: int, width: int, bins: int) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float64)
    yy -= (height - 1) / 2.0
    xx -= (width - 1) / 2.0
    radius = np.sqrt(xx * xx + yy * yy)
    maximum = float(radius.max())
    mapped = np.floor(radius / max(maximum, 1e-12) * bins).astype(np.int32)
    return np.clip(mapped, 0, bins - 1)


def radial_log_amplitude(image: Image.Image, bins: int = DEFAULT_BINS) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    luminance = (
        0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    )
    spectrum = np.fft.fftshift(np.fft.fft2(luminance))
    log_amplitude = np.log(np.abs(spectrum) + 1e-8)
    mapping = _radial_bin_map(*luminance.shape, bins)
    descriptor = np.empty(bins, dtype=np.float64)
    for index in range(bins):
        values = log_amplitude[mapping == index]
        descriptor[index] = float(np.median(values)) if values.size else 0.0
    return descriptor.astype(np.float32)


def edge_correlation(left: Image.Image, right: Image.Image) -> float:
    def edges(image: Image.Image) -> np.ndarray:
        gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        gy, gx = np.gradient(gray)
        return np.hypot(gx, gy).reshape(-1)

    first = edges(left)
    second = edges(right)
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def structure_metrics(left: Image.Image, right: Image.Image) -> dict[str, float | None]:
    first = np.asarray(left.convert("RGB"), dtype=np.float64) / 255.0
    second = np.asarray(right.convert("RGB"), dtype=np.float64) / 255.0
    mse = float(np.mean((first - second) ** 2))
    return {
        "mse": mse,
        "psnr": None if mse <= 1e-15 else float(-10.0 * math.log10(mse)),
        "edge_correlation": edge_correlation(left, right),
    }


def _reconstruct_with_radial_delta(
    image: Image.Image,
    radial_delta: np.ndarray,
) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    chw = array.transpose(2, 0, 1)
    spectrum = np.fft.fftshift(np.fft.fft2(chw, axes=(-2, -1)), axes=(-2, -1))
    mapping = _radial_bin_map(chw.shape[-2], chw.shape[-1], len(radial_delta))
    gain = np.exp(np.asarray(radial_delta, dtype=np.float64)[mapping])
    transformed = np.abs(spectrum) * gain[None] * np.exp(1j * np.angle(spectrum))
    reconstructed = np.fft.ifft2(
        np.fft.ifftshift(transformed, axes=(-2, -1)), axes=(-2, -1)
    ).real
    output = np.clip(reconstructed.transpose(1, 2, 0), 0.0, 1.0)
    return Image.fromarray(np.rint(output * 255.0).astype(np.uint8), mode="RGB")


def radial_mean_calibration(
    image: Image.Image,
    center: np.ndarray,
    *,
    source_weight: float,
    active_bins: int | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    if not 0.0 <= source_weight <= 1.0:
        raise ValueError("source_weight must be in [0, 1]")
    current = radial_log_amplitude(image, len(center))
    delta = source_weight * (np.asarray(center, dtype=np.float64) - current)
    if active_bins is not None:
        if not 1 <= active_bins <= len(delta):
            raise ValueError("active_bins must be within the radial descriptor")
        delta[active_bins:] = 0.0
    output = _reconstruct_with_radial_delta(image, delta)
    return output, {
        "identity": bool(np.allclose(delta, 0.0)),
        "changed_band_count": int(np.count_nonzero(np.abs(delta) > 1e-8)),
        "mean_abs_log_gain": float(np.mean(np.abs(delta))),
        "max_abs_log_gain": float(np.max(np.abs(delta))),
        "active_bins": active_bins if active_bins is not None else len(delta),
        "structure": structure_metrics(image, output),
    }


def source_envelope_calibration(
    image: Image.Image,
    lower: np.ndarray,
    upper: np.ndarray,
    scale: np.ndarray,
    *,
    strength: float,
) -> tuple[Image.Image, dict[str, Any]]:
    if strength < 0.0:
        raise ValueError("strength must be non-negative")
    current = radial_log_amplitude(image, len(lower)).astype(np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    projected = np.clip(current, lower, upper)
    raw_delta = projected - current
    cap = strength * np.maximum(scale, 1e-6)
    delta = np.clip(raw_delta, -cap, cap)
    identity = bool(np.all(np.abs(delta) <= 1e-8))
    output = image.copy() if identity else _reconstruct_with_radial_delta(image, delta)
    return output, {
        "identity": identity,
        "outside_band_count": int(np.count_nonzero(np.abs(raw_delta) > 1e-8)),
        "changed_band_count": int(np.count_nonzero(np.abs(delta) > 1e-8)),
        "mean_abs_log_gain": float(np.mean(np.abs(delta))),
        "max_abs_log_gain": float(np.max(np.abs(delta))),
        "structure": structure_metrics(image, output),
    }


def load_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = ("mean", "median", "scale", "lower", "upper")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"bank is missing arrays: {missing}")
        bank = {key: np.asarray(payload[key]).copy() for key in required}
    lengths = {value.shape for value in bank.values()}
    if len(lengths) != 1 or next(iter(lengths)) != (DEFAULT_BINS,):
        raise ValueError(f"unexpected bank array shapes: {sorted(lengths)}")
    if not all(np.isfinite(value).all() for value in bank.values()):
        raise ValueError("bank contains non-finite values")
    return bank
