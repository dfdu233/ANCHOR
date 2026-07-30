"""Early-feature source-center calibration for Qwen2.5-VL.

The operator is intentionally non-parametric.  It acts immediately after
``visual.patch_embed`` and before Qwen's window reordering.  Every image is
reshaped using its own ``grid_thw`` entry; images are never concatenated into a
synthetic spatial grid.
"""
from __future__ import annotations

import hashlib
import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

CenterKind = Literal["log", "arithmetic"]


@dataclass(frozen=True)
class FeatureCenter:
    """A source amplitude prototype at one visual feature location."""

    amplitude: torch.Tensor
    kind: CenterKind
    grid_thw: tuple[int, int, int]
    count: int
    metadata: dict[str, Any]

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "FeatureCenter":
        return FeatureCenter(
            amplitude=self.amplitude.to(device=device, dtype=dtype),
            kind=self.kind,
            grid_thw=self.grid_thw,
            count=self.count,
            metadata=self.metadata,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_feature_center(center: FeatureCenter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "amplitude": center.amplitude.detach().float().cpu(),
        "kind": center.kind,
        "grid_thw": list(center.grid_thw),
        "count": center.count,
        "metadata": center.metadata,
    }
    torch.save(payload, path)
    manifest = {
        "path": str(path),
        "sha256": _sha256(path),
        "kind": center.kind,
        "grid_thw": list(center.grid_thw),
        "count": center.count,
        "shape": list(center.amplitude.shape),
        "metadata": center.metadata,
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_feature_center(path: Path, map_location: str | torch.device = "cpu") -> FeatureCenter:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    return FeatureCenter(
        amplitude=payload["amplitude"],
        kind=payload["kind"],
        grid_thw=tuple(int(value) for value in payload["grid_thw"]),
        count=int(payload["count"]),
        metadata=dict(payload.get("metadata") or {}),
    )


class StreamingAmplitudeCenter:
    """Numerically stable streaming arithmetic or log-amplitude mean."""

    def __init__(
        self,
        kind: CenterKind,
        grid_thw: tuple[int, int, int],
        channels: int,
        eps: float = 1e-6,
    ) -> None:
        self.kind = kind
        self.grid_thw = grid_thw
        self.channels = channels
        self.eps = eps
        t, h, w = grid_thw
        self.total = torch.zeros((t, channels, h, w), dtype=torch.float64)
        self.count = 0

    @torch.no_grad()
    def update(self, patch_tokens: torch.Tensor) -> None:
        """Update from ordered patch tokens shaped ``[T*H*W, C]``."""
        t, h, w = self.grid_thw
        expected = t * h * w
        if tuple(patch_tokens.shape) != (expected, self.channels):
            raise ValueError(
                f"patch token shape {tuple(patch_tokens.shape)} does not match "
                f"grid={self.grid_thw}, channels={self.channels}"
            )
        feature = patch_tokens.float().reshape(t, h, w, self.channels).permute(0, 3, 1, 2)
        amplitude = torch.fft.fft2(feature, dim=(-2, -1)).abs()
        statistic = torch.log(amplitude + self.eps) if self.kind == "log" else amplitude
        self.total += statistic.double().cpu()
        self.count += 1

    def finalize(self, metadata: dict[str, Any] | None = None) -> FeatureCenter:
        if self.count == 0:
            raise RuntimeError("cannot finalize an empty source center")
        mean = (self.total / self.count).float()
        amplitude = torch.exp(mean) if self.kind == "log" else mean
        return FeatureCenter(
            amplitude=amplitude,
            kind=self.kind,
            grid_thw=self.grid_thw,
            count=self.count,
            metadata=dict(metadata or {}),
        )


def calibrate_patch_tokens(
    patch_tokens: torch.Tensor,
    grid_thw: torch.Tensor,
    center: FeatureCenter,
    tau: float,
    apply_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Calibrate each image independently in the spatial Fourier domain.

    Args:
        patch_tokens: ordered tokens directly from ``patch_embed``, shape
            ``[sum(T_i H_i W_i), C]``.
        grid_thw: one ``[T,H,W]`` row per image.
        center: source amplitude center.  The first release intentionally
            requires an exact grid match instead of silently resizing centers.
        tau: interpolation strength in ``[0,1]``.
        apply_mask: optional boolean vector, one value per image.
    """
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0,1], got {tau}")
    if tau == 0.0:
        return patch_tokens
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError(f"grid_thw must have shape [N,3], got {tuple(grid_thw.shape)}")
    n_images = int(grid_thw.shape[0])
    if apply_mask is None:
        apply_mask = torch.ones(n_images, dtype=torch.bool, device=patch_tokens.device)
    else:
        apply_mask = apply_mask.to(device=patch_tokens.device, dtype=torch.bool)
        if apply_mask.numel() != n_images:
            raise ValueError("apply_mask must contain one value per image")

    source_amp = center.amplitude.to(device=patch_tokens.device, dtype=torch.float32)
    output: list[torch.Tensor] = []
    offset = 0
    channels = int(patch_tokens.shape[-1])
    for index, grid in enumerate(grid_thw.tolist()):
        t, h, w = (int(value) for value in grid)
        length = t * h * w
        tokens = patch_tokens[offset : offset + length]
        offset += length
        if not bool(apply_mask[index]):
            output.append(tokens)
            continue
        if (t, h, w) != center.grid_thw:
            raise ValueError(
                f"image grid {(t, h, w)} does not match center grid {center.grid_thw}; "
                "use aspect-preserving pad-to-grid preprocessing"
            )
        if tuple(source_amp.shape) != (t, channels, h, w):
            raise ValueError(
                f"center shape {tuple(source_amp.shape)} is incompatible with "
                f"grid={(t, h, w)}, channels={channels}"
            )
        feature = tokens.float().reshape(t, h, w, channels).permute(0, 3, 1, 2)
        spectrum = torch.fft.fft2(feature, dim=(-2, -1))
        amplitude = spectrum.abs()
        phase = spectrum / amplitude.clamp_min(eps)
        if center.kind == "log":
            mixed = torch.exp(
                (1.0 - tau) * torch.log(amplitude + eps)
                + tau * torch.log(source_amp + eps)
            )
        elif center.kind == "arithmetic":
            mixed = (1.0 - tau) * amplitude + tau * source_amp
        else:
            raise ValueError(f"unsupported center kind: {center.kind}")
        calibrated = torch.fft.ifft2(mixed * phase, dim=(-2, -1)).real
        calibrated = calibrated.permute(0, 2, 3, 1).reshape(length, channels)
        if not torch.isfinite(calibrated).all():
            raise FloatingPointError(
                f"non-finite calibrated patch tokens for image index={index}, "
                f"grid={(t, h, w)}, tau={tau}"
            )
        output.append(calibrated.to(dtype=tokens.dtype))
    if offset != int(patch_tokens.shape[0]):
        raise ValueError(
            f"grid_thw describes {offset} tokens, but patch_tokens has {patch_tokens.shape[0]}"
        )
    return torch.cat(output, dim=0)


def _resolve_visual(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    for _ in range(5):
        visual = getattr(current, "visual", None)
        if visual is not None:
            return visual
        for attribute in ("base_model", "model"):
            child = getattr(current, attribute, None)
            if child is not None and child is not current:
                current = child
                break
        else:
            break
    raise AttributeError("could not resolve Qwen visual tower from model wrapper")


def install_qwen_patch_center(model: torch.nn.Module) -> None:
    """Install the center operator without changing state-dict parameter names."""
    visual = _resolve_visual(model)
    if getattr(visual, "_anchor_center_installed", False):
        return

    visual._anchor_center_installed = True
    visual._anchor_feature_center = None
    visual._anchor_center_tau = 0.0
    visual._anchor_center_mask = None
    visual._anchor_capture_patch_tokens = False
    visual._anchor_captured_patch_tokens = None
    visual._anchor_native_support = None
    visual._anchor_native_support_mask = None
    visual._anchor_native_support_diagnostics = None

    def centered_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        hidden_states = self.patch_embed(hidden_states)
        if self._anchor_capture_patch_tokens:
            self._anchor_captured_patch_tokens = hidden_states.detach()
        if self._anchor_native_support is not None:
            from anchor.corrected_sgta.training_native_support import (
                project_patch_tokens_to_native_support,
            )

            hidden_states, diagnostics = project_patch_tokens_to_native_support(
                hidden_states,
                grid_thw,
                self._anchor_native_support,
                self._anchor_native_support_mask,
            )
            self._anchor_native_support_diagnostics = diagnostics
        elif self._anchor_feature_center is not None and self._anchor_center_tau > 0:
            hidden_states = calibrate_patch_tokens(
                hidden_states,
                grid_thw,
                self._anchor_feature_center,
                self._anchor_center_tau,
                self._anchor_center_mask,
            )

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :].reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :].reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        for layer_num, block in enumerate(self.blocks):
            current = cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
            hidden_states = block(
                hidden_states,
                cu_seqlens=current,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.merger(hidden_states)
        return hidden_states[torch.argsort(window_index), :]

    visual.forward = types.MethodType(centered_forward, visual)


def set_qwen_center_context(
    model: torch.nn.Module,
    center: FeatureCenter | None,
    tau: float = 0.0,
    apply_mask: torch.Tensor | None = None,
) -> None:
    install_qwen_patch_center(model)
    visual = _resolve_visual(model)
    visual._anchor_feature_center = center
    visual._anchor_center_tau = float(tau)
    visual._anchor_center_mask = apply_mask


def set_qwen_patch_capture(model: torch.nn.Module, enabled: bool) -> None:
    install_qwen_patch_center(model)
    visual = _resolve_visual(model)
    visual._anchor_capture_patch_tokens = bool(enabled)
    if not enabled:
        visual._anchor_captured_patch_tokens = None


def set_qwen_native_support_context(
    model: torch.nn.Module,
    support: Any | None,
    apply_mask: torch.Tensor | None = None,
) -> None:
    """Set the training-native support used by the next visual forward."""
    install_qwen_patch_center(model)
    visual = _resolve_visual(model)
    visual._anchor_native_support = support
    visual._anchor_native_support_mask = apply_mask
    visual._anchor_native_support_diagnostics = None


def get_qwen_native_support_diagnostics(
    model: torch.nn.Module,
) -> list[dict[str, float | bool]] | None:
    install_qwen_patch_center(model)
    return _resolve_visual(model)._anchor_native_support_diagnostics
