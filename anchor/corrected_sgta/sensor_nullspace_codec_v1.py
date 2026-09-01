"""Legacy-preserving residual coding for frozen RGB vision encoders.

The codec maps a standard 8-bit grayscale observation and a signed sensor
residual to an RGB image.  RGB chroma carries the residual, while converting
the resulting integer RGB codeword back through the frozen luma functional
recovers the original grayscale value exactly at every pixel.

This module contains no labels, learned parameters, or model inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LUMA_BT709 = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


@dataclass(frozen=True)
class CodecAudit:
    legacy_mismatch_pixels: int
    payload_energy_requested: float
    payload_energy_transmitted: float
    payload_energy_ratio: float
    capacity_limited_fraction: float
    rgb_change_fraction: float


def project_to_null(vector: np.ndarray, functional: np.ndarray = LUMA_BT709) -> np.ndarray:
    """Project an RGB carrier into the null space of a linear functional."""

    value = np.asarray(vector, dtype=np.float64).reshape(3)
    weight = np.asarray(functional, dtype=np.float64).reshape(3)
    value = value - weight * float(weight @ value) / float(weight @ weight)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("carrier has no non-zero component in the luma null space")
    return value / norm


def _repair_integer_luma(
    rgb8: np.ndarray,
    target: np.ndarray,
    gray8: np.ndarray,
    functional: np.ndarray,
) -> np.ndarray:
    """Repair rare rounding mismatches by nearest feasible local codeword."""

    decoded = np.rint(np.einsum("...c,c->...", rgb8, functional)).astype(np.int16)
    bad = np.argwhere(decoded != gray8.astype(np.int16))
    if not len(bad):
        return rgb8

    offsets = []
    # Quantization of an exactly luma-preserving continuous target can differ
    # by at most one luma level.  A radius-three integer neighborhood is a
    # conservative, deterministic repair set.
    for dr in range(-3, 4):
        for dg in range(-3, 4):
            for db in range(-3, 4):
                offsets.append(np.asarray([dr, dg, db], dtype=np.int16))
    offsets.sort(key=lambda delta: (int(delta @ delta), tuple(abs(int(x)) for x in delta)))

    repaired = rgb8.copy()
    for index in bad:
        key = tuple(int(x) for x in index)
        base = repaired[key].astype(np.int16)
        desired = int(gray8[key])
        best = None
        best_cost = None
        for delta in offsets:
            candidate = base + delta
            if np.any(candidate < 0) or np.any(candidate > 255):
                continue
            if int(np.rint(float(functional @ candidate))) != desired:
                continue
            cost = float(np.sum((candidate.astype(np.float64) - target[key]) ** 2))
            if best_cost is None or cost < best_cost:
                best = candidate
                best_cost = cost
        if best is None:
            raise RuntimeError(f"no local luma-preserving RGB codeword for pixel {key}")
        repaired[key] = best.astype(np.uint8)
    return repaired


def encode_residual_rgb(
    gray8: np.ndarray,
    residual: np.ndarray,
    carrier: np.ndarray,
    gain: float,
    functional: np.ndarray = LUMA_BT709,
) -> tuple[np.ndarray, CodecAudit]:
    """Encode a sensor residual while exactly preserving the legacy gray image.

    ``gray8`` is the ordinary 8-bit image. ``residual`` is measured in the
    normalized [0,1] intensity coordinate before 8-bit quantization.  ``gain``
    is outcome-blind and controls carrier amplitude.  Per-pixel capacity
    scaling prevents clipping without rotating the carrier out of the null
    space.  A nearest-lattice repair then guarantees

        round(<functional, rgb8>) == gray8

    exactly for every output pixel.
    """

    q8 = np.asarray(gray8)
    r = np.asarray(residual, dtype=np.float64)
    if q8.dtype != np.uint8:
        raise TypeError("gray8 must have dtype uint8")
    if q8.shape != r.shape:
        raise ValueError("gray8 and residual must have identical shapes")
    if gain < 0:
        raise ValueError("gain must be non-negative")
    weight = np.asarray(functional, dtype=np.float64).reshape(3)
    if not np.isclose(float(weight.sum()), 1.0, atol=1e-12):
        raise ValueError("luma functional must sum to one")
    direction = project_to_null(carrier, weight)

    q = q8.astype(np.float64) / 255.0
    requested = gain * r[..., None] * direction
    capacity = np.ones(q.shape, dtype=np.float64)
    for channel in range(3):
        delta = requested[..., channel]
        positive = delta > 0
        negative = delta < 0
        channel_capacity = np.ones(q.shape, dtype=np.float64)
        channel_capacity[positive] = (1.0 - q[positive]) / delta[positive]
        channel_capacity[negative] = q[negative] / (-delta[negative])
        capacity = np.minimum(capacity, channel_capacity)
    capacity = np.clip(capacity, 0.0, 1.0)
    transmitted = requested * capacity[..., None]
    target01 = q[..., None] + transmitted
    if np.any(target01 < -1e-10) or np.any(target01 > 1.0 + 1e-10):
        raise RuntimeError("capacity calculation failed to prevent clipping")
    target8 = np.clip(target01, 0.0, 1.0) * 255.0
    rgb8 = np.rint(target8).astype(np.uint8)
    rgb8 = _repair_integer_luma(rgb8, target8, q8, weight)
    decoded = np.rint(np.einsum("...c,c->...", rgb8, weight)).astype(np.uint8)
    mismatches = int(np.count_nonzero(decoded != q8))
    if mismatches:
        raise RuntimeError(f"legacy luma invariant failed at {mismatches} pixels")

    requested_energy = float(np.sum(requested * requested))
    transmitted_energy = float(np.sum(transmitted * transmitted))
    ratio = transmitted_energy / requested_energy if requested_energy > 0 else 1.0
    base_rgb = np.repeat(q8[..., None], 3, axis=-1)
    audit = CodecAudit(
        legacy_mismatch_pixels=mismatches,
        payload_energy_requested=requested_energy,
        payload_energy_transmitted=transmitted_energy,
        payload_energy_ratio=ratio,
        capacity_limited_fraction=float(np.mean(capacity < 1.0 - 1e-12)),
        rgb_change_fraction=float(np.mean(np.any(rgb8 != base_rgb, axis=-1))),
    )
    return rgb8, audit


def decode_legacy_gray(
    rgb8: np.ndarray, functional: np.ndarray = LUMA_BT709
) -> np.ndarray:
    """Recover the exact legacy grayscale observation from an encoded RGB image."""

    value = np.asarray(rgb8)
    if value.dtype != np.uint8 or value.ndim < 3 or value.shape[-1] != 3:
        raise TypeError("rgb8 must be a uint8 array with a final RGB dimension")
    weight = np.asarray(functional, dtype=np.float64).reshape(3)
    return np.rint(np.einsum("...c,c->...", value, weight)).clip(0, 255).astype(np.uint8)
