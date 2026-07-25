"""Core primitives for source-counterfactual evidence invariance.

The style bank is source-image-only. It stores robust log-spectrum centers and
never represents questions, answers, target images, or target statistics.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

VERSION = "anchor-dg-style-bank-v2"


def stable_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standardized_image(image: Image.Image, size: int) -> Image.Image:
    if size <= 0:
        raise ValueError("image size must be positive")
    return ImageOps.pad(
        image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS,
        color=(0, 0, 0), centering=(0.5, 0.5),
    )


def shifted_log_amplitude(image: Image.Image) -> np.ndarray:
    """Return log1p RGB FFT amplitude in shifted [C,H,W] layout."""
    array = np.asarray(image.convert("RGB"), dtype=np.float64).transpose(2, 0, 1)
    spectrum = np.fft.fftshift(np.fft.fft2(array, axes=(-2, -1)), axes=(-2, -1))
    return np.log1p(np.abs(spectrum)).astype(np.float32)


def intensity_statistics(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return array.mean(axis=(0, 1)), array.std(axis=(0, 1))


def smooth_radial_mask(shape: tuple[int, int], rho: float) -> np.ndarray:
    """Circular raised-cosine low-frequency mask with an untouched DC bin."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("invalid spectrum shape")
    if rho < 0 or rho > 0.5:
        raise ValueError("rho must be in [0, 0.5]")
    if rho == 0:
        return np.zeros(shape, dtype=np.float32)
    yy, xx = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    radius = max(1.0, rho * min(height, width))
    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = np.zeros(shape, dtype=np.float64)
    inside = distance < radius
    mask[inside] = 0.5 * (1.0 + np.cos(np.pi * distance[inside] / radius))
    mask[cy, cx] = 0.0
    return mask.astype(np.float32)


def transported_shifted_spectrum(
    image: Image.Image, target_log_amplitude: np.ndarray, rho: float, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return transported complex spectrum and the applied smooth mask."""
    if beta < 0 or beta > 1:
        raise ValueError("beta must be in [0,1]")
    target = np.asarray(target_log_amplitude, dtype=np.float64)
    if target.ndim != 3 or target.shape[0] != 3:
        raise ValueError("target log-amplitude must have shape [3,H,W]")
    prepared = image.convert("RGB")
    array = np.asarray(prepared, dtype=np.float64).transpose(2, 0, 1)
    height, width = array.shape[-2:]
    if target.shape[-2:] != (height, width):
        target = (
            F.interpolate(
                torch.from_numpy(target)[None].float(),
                size=(height, width), mode="bilinear", align_corners=False,
            )[0]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        unshifted = np.fft.ifftshift(target, axes=(-2, -1))
        conjugate_partner = np.roll(
            np.flip(unshifted, axis=(-2, -1)), shift=(1, 1), axis=(-2, -1)
        )
        target = np.fft.fftshift(
            0.5 * (unshifted + conjugate_partner), axes=(-2, -1)
        )
    spectrum = np.fft.fftshift(np.fft.fft2(array, axes=(-2, -1)), axes=(-2, -1))
    amplitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    log_amplitude = np.log1p(amplitude)
    mask = smooth_radial_mask(target.shape[-2:], rho)
    transported_log = log_amplitude + beta * mask[None] * (target - log_amplitude)
    transported_amplitude = np.expm1(transported_log).clip(min=0.0)
    transported = transported_amplitude * np.exp(1j * phase)
    cy, cx = target.shape[-2] // 2, target.shape[-1] // 2
    transported[:, cy, cx] = spectrum[:, cy, cx]
    return transported, mask


def transport_log_spectrum(
    image: Image.Image, target_log_amplitude: np.ndarray, rho: float, beta: float
) -> Image.Image:
    if beta == 0 or rho == 0:
        return image.convert("RGB").copy()
    spectrum, _ = transported_shifted_spectrum(image, target_log_amplitude, rho, beta)
    array = np.fft.ifft2(np.fft.ifftshift(spectrum, axes=(-2, -1)), axes=(-2, -1)).real
    array = np.clip(np.rint(array), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(array, mode="RGB")


@dataclass(frozen=True)
class StyleBank:
    domains: tuple[str, ...]
    log_amplitudes: dict[str, np.ndarray]
    rgb_means: dict[str, np.ndarray]
    rgb_stds: dict[str, np.ndarray]
    metadata: dict[str, object]
    path: Path | None = None

    def validate(self, heldout_domains: Iterable[str] = ()) -> None:
        if len(self.domains) < 2:
            raise ValueError("a counterfactual style bank requires at least two domains")
        if len(set(self.domains)) != len(self.domains):
            raise ValueError("style bank contains duplicate domains")
        leaked = set(heldout_domains).intersection(self.domains)
        if leaked:
            raise ValueError(f"held-out domain leaked into style bank: {sorted(leaked)}")
        for domain in self.domains:
            center = np.asarray(self.log_amplitudes[domain])
            if center.ndim != 3 or center.shape[0] != 3:
                raise ValueError(f"invalid log-amplitude shape for {domain}: {center.shape}")
            if not np.isfinite(center).all() or np.any(center < 0):
                raise ValueError(f"invalid log-amplitude values for {domain}")
            for name, values in (("rgb_mean", self.rgb_means[domain]), ("rgb_std", self.rgb_stds[domain])):
                values = np.asarray(values)
                if values.shape != (3,) or not np.isfinite(values).all():
                    raise ValueError(f"invalid {name} for {domain}")
        if self.metadata.get("version") != VERSION:
            raise ValueError("unsupported style-bank version")
        forbidden = {"answer", "answers", "question", "questions", "target_rows"}
        if forbidden.intersection(self.metadata):
            raise ValueError("style-bank metadata contains forbidden semantic fields")

    def log_amplitude(self, domain: str) -> np.ndarray:
        if domain not in self.log_amplitudes:
            raise KeyError(f"unknown style domain: {domain}")
        return self.log_amplitudes[domain]


def load_style_bank(path: Path, heldout_domains: Iterable[str] = ()) -> StyleBank:
    path = path.expanduser().resolve()
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"style bank or metadata missing: {path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("npz_sha256") != file_sha256(path):
        raise RuntimeError("style-bank checksum mismatch")
    with np.load(path, allow_pickle=False) as payload:
        domains = tuple(str(value) for value in payload["domains"].tolist())
        centers = {domain: payload[f"log_amplitude_{index}"].copy() for index, domain in enumerate(domains)}
        means = {domain: payload[f"rgb_mean_{index}"].copy() for index, domain in enumerate(domains)}
        stds = {domain: payload[f"rgb_std_{index}"].copy() for index, domain in enumerate(domains)}
    bank = StyleBank(domains, centers, means, stds, metadata, path)
    bank.validate(heldout_domains)
    return bank


def deterministic_other_domain(sample_domain: str, domains: Iterable[str], sample_id: str, seed: int, step: int = 0) -> str:
    candidates = sorted(set(domains) - {sample_domain})
    if not candidates:
        raise ValueError(f"no counterfactual domain exists for {sample_domain!r}")
    digest = stable_sha256({"id": sample_id, "seed": seed, "step": step})
    return candidates[int(digest[:16], 16) % len(candidates)]


def counterfactual_view(
    image: Image.Image, bank: StyleBank, source_domain: str, sample_id: str,
    seed: int, step: int, rho: float, beta: float,
) -> tuple[Image.Image, str]:
    target_domain = deterministic_other_domain(source_domain, bank.domains, sample_id, seed, step)
    return transport_log_spectrum(image, bank.log_amplitude(target_domain), rho, beta), target_domain


def edge_correlation(left: Image.Image, right: Image.Image) -> float:
    def edges(image: Image.Image) -> np.ndarray:
        gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        gx = np.diff(gray, axis=1, append=gray[:, -1:])
        gy = np.diff(gray, axis=0, append=gray[-1:, :])
        return np.hypot(gx, gy).reshape(-1)
    if left.size != right.size:
        right = right.resize(left.size, Image.Resampling.LANCZOS)
    a, b = edges(left), edges(right)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def gold_token_log_probabilities(answer_logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    if answer_logits.ndim != 2:
        raise ValueError("answer_logits must have shape [tokens, vocabulary]")
    token_ids = token_ids.reshape(-1).to(answer_logits.device, dtype=torch.long)
    if token_ids.numel() != answer_logits.shape[0]:
        raise ValueError("gold-token/logit length mismatch")
    if bool((token_ids < 0).any()) or bool((token_ids >= answer_logits.shape[1]).any()):
        raise ValueError("gold token id outside vocabulary")
    return F.log_softmax(answer_logits.float(), dim=-1).gather(1, token_ids[:, None]).squeeze(1)


def evidence_huber_loss(clean_logits: torch.Tensor, view_logits: torch.Tensor, token_ids: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    if clean_logits.shape != view_logits.shape:
        raise ValueError("clean/view answer logits differ in shape")
    clean = gold_token_log_probabilities(clean_logits, token_ids)
    view = gold_token_log_probabilities(view_logits, token_ids)
    return F.huber_loss(clean, view, delta=delta, reduction="mean")


def raw_logit_consistency_loss(clean_logits: torch.Tensor, view_logits: torch.Tensor) -> torch.Tensor:
    if clean_logits.shape != view_logits.shape:
        raise ValueError("clean/view answer logits differ in shape")
    clean_log = F.log_softmax(clean_logits.float(), dim=-1)
    view_log = F.log_softmax(view_logits.float(), dim=-1)
    clean_prob, view_prob = clean_log.exp(), view_log.exp()
    loss = 0.5 * (F.kl_div(clean_log, view_prob.detach(), reduction="batchmean") + F.kl_div(view_log, clean_prob.detach(), reduction="batchmean"))
    return loss.clamp_min(0.0)
