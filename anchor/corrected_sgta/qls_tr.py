"""Question-conditioned local-score trust-region primitives.

The source step is a KDE mean-shift update in a compact log-Fourier space.
Question conditioning is applied separately by limiting the categorical
Fisher/KL displacement of the frozen VLM output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PCAIndex:
    mean: np.ndarray
    components: np.ndarray
    coordinates: np.ndarray
    bandwidth: float
    median_nn_distance: float


def spectral_descriptor(
    image: Image.Image, *, image_size: int = 128, grid_size: int = 16
) -> np.ndarray:
    """Return a scale-normalized, centered log-amplitude descriptor."""

    gray = np.asarray(
        image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    amplitude = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
    log_amplitude = np.log1p(amplitude)
    block = image_size // grid_size
    if block * grid_size != image_size:
        raise ValueError("image_size must be divisible by grid_size")
    pooled = log_amplitude.reshape(grid_size, block, grid_size, block).mean((1, 3))
    pooled[grid_size // 2, grid_size // 2] = 0.0
    vector = pooled.reshape(-1)
    norm = np.linalg.norm(vector)
    return (vector / max(norm, 1e-12)).astype(np.float32)


def fit_pca_index(
    features: np.ndarray, *, variance: float = 0.90, rank_cap: int = 16
) -> PCAIndex:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("features must be [N,D] with N>=3")
    mean = values.mean(0)
    centered = values - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    energy = singular**2
    cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-12)
    rank = min(int(np.searchsorted(cumulative, variance) + 1), rank_cap, len(vt))
    components = vt[:rank]
    coordinates = centered @ components.T
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    sorted_distances = np.sort(distances, axis=1)
    neighbor_rank = min(7, sorted_distances.shape[1] - 1)
    bandwidth = float(np.median(sorted_distances[:, neighbor_rank]))
    median_nn = float(np.median(sorted_distances[:, 0]))
    return PCAIndex(
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        coordinates=coordinates.astype(np.float32),
        bandwidth=max(bandwidth, 1e-8),
        median_nn_distance=max(median_nn, 1e-8),
    )


def project_descriptor(feature: np.ndarray, index: PCAIndex) -> np.ndarray:
    return (np.asarray(feature) - index.mean) @ index.components.T


def kde_neighbors(
    feature: np.ndarray, index: PCAIndex, *, k: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return neighbor ids, normalized KDE weights, and mean-shift vector."""

    query = project_descriptor(feature, index)
    squared = np.sum((index.coordinates - query) ** 2, axis=1)
    ids = np.argsort(squared)[: min(k, len(squared))]
    logits = -squared[ids] / (2.0 * index.bandwidth**2)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    mean_shift = np.sum(weights[:, None] * index.coordinates[ids], axis=0) - query
    return ids, weights.astype(np.float32), mean_shift.astype(np.float32)


def categorical_js(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    eps = 1e-12
    return float(
        0.5 * np.sum(p * np.log((p + eps) / (midpoint + eps)))
        + 0.5 * np.sum(q * np.log((q + eps) / (midpoint + eps)))
    )


def fisher_quadratic(probabilities: np.ndarray, logit_delta: np.ndarray) -> float:
    """Categorical pullback Fisher energy in logit coordinates."""

    p = np.asarray(probabilities, dtype=np.float64)
    delta = np.asarray(logit_delta, dtype=np.float64)
    fisher = np.diag(p) - np.outer(p, p)
    return float(delta @ fisher @ delta)


def reconstruct_mean_shift_view(
    image: Image.Image,
    neighbor_images: list[Image.Image],
    weights: np.ndarray,
    *,
    blend: float,
    low_frequency_ratio: float = 0.10,
) -> Image.Image:
    """Apply a phase-preserving weighted local source amplitude update."""

    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0,1]")
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    height, width = rgb.shape[:2]
    chw = rgb.transpose(2, 0, 1)
    fft = np.fft.fft2(chw, axes=(-2, -1))
    shifted = np.fft.fftshift(np.abs(fft), axes=(-2, -1))
    target_logs = []
    for source in neighbor_images:
        array = np.asarray(
            source.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float64,
        ).transpose(2, 0, 1)
        source_amp = np.fft.fftshift(
            np.abs(np.fft.fft2(array, axes=(-2, -1))), axes=(-2, -1)
        )
        target_logs.append(np.log1p(source_amp))
    target = np.expm1(
        np.sum(np.asarray(weights)[:, None, None, None] * np.stack(target_logs), axis=0)
    )
    radius = max(1, int(math.floor(min(height, width) * low_frequency_ratio)))
    center_h, center_w = height // 2, width // 2
    hs, he = center_h - radius, center_h + radius + 1
    ws, we = center_w - radius, center_w + radius + 1
    shifted[:, hs:he, ws:we] = (
        (1.0 - blend) * shifted[:, hs:he, ws:we]
        + blend * target[:, hs:he, ws:we]
    )
    amplitude = np.fft.ifftshift(shifted, axes=(-2, -1))
    reconstructed = np.fft.ifft2(
        amplitude * np.exp(1j * np.angle(fft)), axes=(-2, -1)
    ).real
    output = np.clip(np.rint(reconstructed.transpose(1, 2, 0)), 0, 255).astype(
        np.uint8
    )
    return Image.fromarray(output, mode="RGB")
