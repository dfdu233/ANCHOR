"""Source-aligned style, graph adaptation, and conformal utilities."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    out = np.exp(x)
    return out / np.sum(out, axis=axis, keepdims=True)


def entropy_weighted_fusion(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Fuse per-style logits using normalized inverse-entropy evidence."""

    probs = softmax_np(logits)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=-1)
    weights = softmax_np(-entropy / temperature, axis=0)
    return np.sum(weights[:, None] * np.asarray(logits), axis=0)


def feddg_frequency_interpolation(
    image: Image.Image,
    target_amplitude: np.ndarray,
    low_frequency_ratio: float = 0.003,
    source_ratio: float = 0.0,
) -> Image.Image:
    """FedDG-ELCFS frequency-space interpolation, generalized to a center.

    This follows ``freq_space_interpolation_demo.py``: only the centered
    low-frequency window is interpolated, phase is retained, and processing is
    channel-wise.  A 2-D external amplitude center is broadcast to RGB; a
    three-channel target preserves the reference implementation exactly.
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
        raise ValueError(
            f"target amplitude must be [H,W] or [C,H,W], got {target.shape}"
        )
    if target.shape[0] == 1:
        target = np.repeat(target, 3, axis=0)
    if target.shape[-2:] != chw.shape[-2:]:
        target = torch.from_numpy(target).unsqueeze(0).float()
        target = F.interpolate(
            target, size=chw.shape[-2:], mode="bilinear", align_corners=False
        )
        target = target.squeeze(0).numpy()

    fft = np.fft.fft2(chw, axes=(-2, -1))
    amplitude = np.fft.fftshift(np.abs(fft), axes=(-2, -1))
    target = np.fft.fftshift(target, axes=(-2, -1))
    height, width = chw.shape[-2:]
    radius = int(math.floor(min(height, width) * low_frequency_ratio))
    center_h, center_w = height // 2, width // 2
    hs, he = center_h - radius, center_h + radius + 1
    ws, we = center_w - radius, center_w + radius + 1
    amplitude[:, hs:he, ws:we] = (
        source_ratio * amplitude[:, hs:he, ws:we]
        + (1.0 - source_ratio) * target[:, hs:he, ws:we]
    )
    amplitude = np.fft.ifftshift(amplitude, axes=(-2, -1))
    reconstructed = np.fft.ifft2(
        amplitude * np.exp(1j * np.angle(fft)), axes=(-2, -1)
    ).real
    out = np.clip(reconstructed.transpose(1, 2, 0), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def gamma_transform(image: Image.Image, gamma: float) -> Image.Image:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    value = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    value = np.power(value, 1.0 / gamma)
    return Image.fromarray(np.clip(value * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def lame_rbf_affinity(
    features: torch.Tensor, knn: int = 5, force_symmetry: bool = False
) -> torch.Tensor:
    """The RBF affinity from the official LAME implementation."""

    features = F.normalize(features.float(), p=2, dim=-1)
    distance = torch.cdist(features, features, p=2)
    n_neighbors = min(knn, features.shape[0])
    sigma = distance.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1].mean()
    sigma = sigma.clamp_min(1e-12)
    kernel = torch.exp(-(distance**2) / (2 * sigma**2))
    if force_symmetry:
        kernel = 0.5 * (kernel + kernel.T)
    return kernel


def laplacian_optimization(
    probabilities: torch.Tensor,
    kernel: torch.Tensor,
    bound_lambda: float = 1.0,
    max_steps: int = 100,
    tolerance: float = 1e-8,
) -> torch.Tensor:
    """Official LAME CCCP update with its energy convergence criterion."""

    unary = -torch.log(probabilities.float().clamp_min(1e-10))
    y = (-unary).softmax(-1)
    old_energy = float("inf")
    for step in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(y)
        y = (-unary + pairwise).softmax(-1)
        energy = (
            unary * y - bound_lambda * pairwise * y + y * torch.log(y.clamp_min(1e-20))
        ).sum()
        current = float(energy)
        if step > 1 and abs(current - old_energy) <= tolerance * abs(old_energy):
            break
        old_energy = current
    return y


def lata_knn_affinity(features: torch.Tensor, knn: int = 5) -> torch.Tensor:
    """Sparse union-kNN RBF graph from LATA Eq. (4)."""

    features = F.normalize(features.float(), p=2, dim=-1)
    distance = torch.cdist(features, features, p=2)
    n = features.shape[0]
    k = min(knn, max(0, n - 1))
    if k == 0:
        return torch.zeros((n, n), dtype=features.dtype, device=features.device)
    neighbors = distance.topk(k=k + 1, largest=False).indices[:, 1:]
    directed = torch.zeros((n, n), dtype=torch.bool, device=features.device)
    directed.scatter_(1, neighbors, True)
    union = directed | directed.T
    neighbor_distances = distance[union]
    sigma = neighbor_distances.median().clamp_min(1e-12)
    weights = torch.exp(-(distance**2) / (sigma**2))
    return weights * union


def lata_refine(
    probabilities: torch.Tensor,
    features: torch.Tensor,
    gamma: float = 1.0,
    knn: int = 5,
    iterations: int = 10,
) -> torch.Tensor:
    """LATA Eq. (6), label-free (beta=0) on a joint fixed-class pool."""

    q = probabilities.float().clamp_min(1e-12)
    q = q / q.sum(-1, keepdim=True)
    graph = lata_knn_affinity(features, knn=knn)
    z = q.clone()
    log_q = q.log()
    for _ in range(iterations):
        z = torch.softmax(log_q + gamma * graph.matmul(z), dim=-1)
    return z


def conformal_quantile(scores: Iterable[float], alpha: float) -> float:
    """Finite-sample split-conformal higher quantile (Eq. 2 in LATA)."""

    values = np.sort(np.asarray(list(scores), dtype=np.float64))
    if not len(values):
        raise ValueError("calibration scores are empty")
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    return float(values[min(rank, len(values)) - 1])


def lac_sets(
    calibration_probs: np.ndarray,
    calibration_labels: np.ndarray,
    test_probs: np.ndarray,
    alpha: float,
):
    scores = (
        1.0 - calibration_probs[np.arange(len(calibration_labels)), calibration_labels]
    )
    threshold = conformal_quantile(scores, alpha)
    return (1.0 - test_probs) <= threshold, threshold


def aps_scores(
    probabilities: np.ndarray, uniforms: np.ndarray | None = None
) -> np.ndarray:
    """Randomized APS scores for every label, matching SCA-T/TorchCP."""

    probs = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(-probs, axis=-1)
    ordered = np.take_along_axis(probs, order, axis=-1)
    cumulative = np.cumsum(ordered, axis=-1)
    if uniforms is None:
        uniforms = np.ones_like(probs)
    randomized = cumulative - ordered * np.asarray(uniforms)
    inverse = np.argsort(order, axis=-1)
    return np.take_along_axis(randomized, inverse, axis=-1)


def aps_sets(
    calibration_probs: np.ndarray,
    calibration_labels: np.ndarray,
    test_probs: np.ndarray,
    alpha: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    cal_all = aps_scores(calibration_probs, rng.random(calibration_probs.shape))
    cal_scores = cal_all[np.arange(len(calibration_labels)), calibration_labels]
    threshold = conformal_quantile(cal_scores, alpha)
    test_scores = aps_scores(test_probs, rng.random(test_probs.shape))
    return test_scores <= threshold, threshold
