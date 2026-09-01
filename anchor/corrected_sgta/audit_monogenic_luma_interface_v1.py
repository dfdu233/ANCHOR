#!/usr/bin/env python3
"""CPU-only interface audit for luma-preserving monogenic/Riesz coding.

This audit deliberately does not score a VLM.  It measures whether the clean
continuous construction survives the finite RGB interface that an existing
vision tower actually receives:

    X = f 1 + alpha ((R1 f) v1 + (R2 f) v2),   w^T v1 = w^T v2 = 0.

The audit reports (i) exact continuous luma error, (ii) the amount of Riesz
payload retained after per-pixel anti-clipping contraction, (iii) 8-bit luma
round-trip error, and (iv) how closely resize and the discrete Riesz transform
commute.  It uses no labels, model weights, or GPU inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pydicom


LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def luma_null_basis(weight: np.ndarray = LUMA) -> np.ndarray:
    """Return an orthonormal 3x2 basis for the Euclidean nullspace of weight."""

    _, _, vh = np.linalg.svd(np.asarray(weight, dtype=np.float64)[None, :])
    basis = vh[1:].T
    if not np.allclose(weight @ basis, 0.0, atol=1e-12):
        raise RuntimeError("failed to construct a luma-null basis")
    return basis


def render_dicom(path: Path, side: int) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    image = np.asarray(ds.pixel_array, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D image, got {image.shape}: {path}")
    finite = image[np.isfinite(image)]
    lo, hi = (float(value) for value in np.percentile(finite, [0.5, 99.5]))
    image = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = 1.0 - image
    return cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA).astype(np.float64)


def riesz_pair(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Periodic discrete first-order Riesz transform via its Fourier multiplier."""

    height, width = image.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.hypot(fx, fy)
    safe = np.where(radius == 0.0, 1.0, radius)
    spectrum = np.fft.fft2(image)
    r1 = np.fft.ifft2((-1j * fx / safe) * spectrum).real
    r2 = np.fft.ifft2((-1j * fy / safe) * spectrum).real
    return r1, r2


def anti_clip_capacity(base: np.ndarray, delta: np.ndarray, alpha: float) -> np.ndarray:
    """Largest per-pixel multiplier in [0,1] retaining an alpha-scaled payload."""

    requested = alpha * delta
    capacity = np.ones(base.shape, dtype=np.float64)
    for channel in range(3):
        value = requested[..., channel]
        positive = value > 0.0
        negative = value < 0.0
        bound = np.ones(base.shape, dtype=np.float64)
        bound[positive] = (1.0 - base[positive]) / value[positive]
        bound[negative] = base[negative] / (-value[negative])
        capacity = np.minimum(capacity, bound)
    return np.clip(capacity, 0.0, 1.0)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom > 1e-12 else float("nan")


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(array)),
        "median": float(np.nanmedian(array)),
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, default=Path("/workspace/vinbigdata/train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=32)
    parser.add_argument("--work-side", type=int, default=512)
    parser.add_argument("--model-side", type=int, default=336)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    args = parser.parse_args()

    paths = sorted(args.image_root.glob("*.dicom"))[: args.n_images]
    if not paths:
        raise FileNotFoundError(f"no DICOM files under {args.image_root}")
    basis = luma_null_basis()
    records: list[dict[str, object]] = []
    for path in paths:
        high = render_dicom(path, args.work_side)
        low = cv2.resize(high, (args.model_side, args.model_side), interpolation=cv2.INTER_AREA)
        # Quantize the legacy observation first: this is the actual 8-bit base
        # whose decoded luma we demand to preserve.
        gray8 = np.rint(np.clip(low, 0.0, 1.0) * 255.0).astype(np.uint8)
        base = gray8.astype(np.float64) / 255.0
        r1, r2 = riesz_pair(base)
        delta = r1[..., None] * basis[:, 0] + r2[..., None] * basis[:, 1]

        high_r1, high_r2 = riesz_pair(high)
        down_r1 = cv2.resize(high_r1, (args.model_side, args.model_side), interpolation=cv2.INTER_AREA)
        down_r2 = cv2.resize(high_r2, (args.model_side, args.model_side), interpolation=cv2.INTER_AREA)
        resize_record = {
            "r1_correlation": correlation(r1, down_r1),
            "r2_correlation": correlation(r2, down_r2),
            "relative_l2": float(
                np.sqrt(np.sum((r1 - down_r1) ** 2 + (r2 - down_r2) ** 2))
                / max(np.sqrt(np.sum(down_r1**2 + down_r2**2)), 1e-12)
            ),
        }

        arms: dict[str, object] = {}
        requested_energy_unit = float(np.sum(delta * delta))
        for alpha in args.alphas:
            capacity = anti_clip_capacity(base, delta, alpha)
            transmitted = alpha * capacity[..., None] * delta
            rgb = base[..., None] + transmitted
            continuous_luma = np.einsum("...c,c->...", rgb, LUMA)
            rgb8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            decoded8 = np.rint(np.einsum("...c,c->...", rgb8, LUMA)).astype(np.int16)
            target8 = gray8.astype(np.int16)
            transmitted_energy = float(np.sum(transmitted * transmitted))
            requested_energy = alpha * alpha * requested_energy_unit
            quantized_delta = rgb8.astype(np.float64) / 255.0 - base[..., None]
            quantized_energy = float(np.sum(quantized_delta * quantized_delta))
            arms[str(alpha)] = {
                "continuous_luma_max_abs_error": float(np.max(np.abs(continuous_luma - base))),
                "payload_energy_ratio_after_anti_clip": (
                    transmitted_energy / requested_energy if requested_energy > 0.0 else 1.0
                ),
                "capacity_limited_pixel_fraction": float(np.mean(capacity < 1.0 - 1e-12)),
                "payload_zeroed_pixel_fraction": float(np.mean(capacity <= 1e-12)),
                "uint8_luma_mismatch_fraction_without_lattice_repair": float(
                    np.mean(decoded8 != target8)
                ),
                "uint8_luma_mean_abs_error_levels": float(np.mean(np.abs(decoded8 - target8))),
                "uint8_rgb_changed_pixel_fraction": float(
                    np.mean(np.any(rgb8 != gray8[..., None], axis=-1))
                ),
                "uint8_payload_energy_ratio_to_continuous_transmitted": (
                    quantized_energy / transmitted_energy if transmitted_energy > 0.0 else 1.0
                ),
                "uint8_payload_correlation_with_continuous": correlation(
                    quantized_delta, transmitted
                ),
            }
        records.append({"image_id": path.stem, "resize_commutation": resize_record, "arms": arms})

    aggregate: dict[str, object] = {
        "resize_commutation": {
            key: summarize([float(r["resize_commutation"][key]) for r in records])
            for key in ("r1_correlation", "r2_correlation", "relative_l2")
        },
        "arms": {},
    }
    arm_keys = (
        "continuous_luma_max_abs_error",
        "payload_energy_ratio_after_anti_clip",
        "capacity_limited_pixel_fraction",
        "payload_zeroed_pixel_fraction",
        "uint8_luma_mismatch_fraction_without_lattice_repair",
        "uint8_luma_mean_abs_error_levels",
        "uint8_rgb_changed_pixel_fraction",
        "uint8_payload_energy_ratio_to_continuous_transmitted",
        "uint8_payload_correlation_with_continuous",
    )
    for alpha in args.alphas:
        name = str(alpha)
        aggregate["arms"][name] = {
            key: summarize([float(r["arms"][name][key]) for r in records]) for key in arm_keys
        }

    result = {
        "version": "monogenic-luma-interface-audit-v1",
        "n_images": len(records),
        "image_root": str(args.image_root),
        "work_side": args.work_side,
        "model_side": args.model_side,
        "luma": LUMA.tolist(),
        "null_basis": basis.tolist(),
        "scope": "CPU interface audit only; no labels, model inference, or mitigation claim",
        "mathematical_boundary": (
            "Riesz channels are deterministic functions of the same scalar image. Under ideal "
            "band-limited isotropic resizing they commute with resize and add no observation information."
        ),
        "aggregate": aggregate,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in result if k != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
