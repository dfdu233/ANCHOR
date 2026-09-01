"""Clean-room, cross-architecture port of Confidence-aware Visual Enhancement.

The attached CVE paper discloses the region quantiles and decoding equation,
but not the exact PIL operator strengths.  Those strengths are therefore
explicit parameters here and are recorded in every result.  CLIP-family ports
use the disclosed final-layer CLS-to-patch attention.  Architectures without a
CLS token use the norm of their final native visual tokens as a documented
spatial-importance proxy rather than pretending an exact CLS port exists.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from transformers import RepetitionPenaltyLogitsProcessor

from .cross_model_vcd import prefill_stream
from .models_oe import (
    Generation,
    HuatuoOEAdapter,
    HuluOEAdapter,
    LlavaMedOEAdapter,
    Qwen25VLOEAdapter,
)


def cve_logits(
    original: torch.Tensor,
    rebalanced: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 0.2,
) -> torch.Tensor:
    """Equation 7 from CVE, with shape and parameter checks."""

    if original.shape != rebalanced.shape:
        raise ValueError("CVE streams must have identical vocabulary shape")
    if alpha < 0 or beta < 0:
        raise ValueError("CVE alpha and beta must be non-negative")
    confidence_gap = original.max(dim=-1, keepdim=True).values - original
    return original + alpha * rebalanced - torch.relu(beta * confidence_gap)


def _square_map(values: torch.Tensor) -> np.ndarray:
    values = values.detach().float().cpu().flatten()
    side = math.isqrt(values.numel())
    if side * side != values.numel():
        raise ValueError(f"visual token count {values.numel()} is not square")
    array = values.reshape(side, side).numpy()
    array -= array.min()
    scale = float(array.max())
    return array / scale if scale > 0 else np.zeros_like(array)


@torch.inference_mode()
def _clip_attention(adapter: Any, image: Image.Image) -> np.ndarray:
    tensor = adapter._inputs(image, "visual evidence audit")[1] if isinstance(
        adapter, HuatuoOEAdapter
    ) else adapter._process_images([image])
    if isinstance(tensor, list):
        tensor = tensor[0]
    tensor = tensor.to(adapter.model.device, dtype=adapter.model.dtype)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    # LLaVA-NeXT any-resolution tensors contain a global tile first.  CVE's
    # disclosed single attention map is applied to that global view.
    tensor = tensor[:1]
    tower = adapter.model.get_vision_tower()
    backbone = getattr(tower, "vision_tower", tower)
    output = backbone(tensor, output_attentions=True, return_dict=True)
    attention = output.attentions[-1][0].float().mean(dim=0)
    # CLIP index zero is CLS; retain its outgoing patch distribution.
    return _square_map(attention[0, 1:])


@torch.inference_mode()
def _hulu_token_importance(adapter: HuluOEAdapter, image: Image.Image) -> np.ndarray:
    inputs = adapter._inputs(image, "visual evidence audit")
    encoder = adapter.model.get_vision_encoder()
    output = encoder(
        inputs["pixel_values"],
        inputs["grid_sizes"],
        inputs["merge_sizes"],
    )
    tokens = output[0] if isinstance(output, (list, tuple)) else output
    return _square_map(tokens.float().norm(dim=-1))


@torch.inference_mode()
def _qwen_token_importance(adapter: Qwen25VLOEAdapter, image: Image.Image) -> np.ndarray:
    inputs = adapter._inputs(image, "visual evidence audit")
    tokens = adapter.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    values = tokens.float().norm(dim=-1)
    grid = inputs["image_grid_thw"][0].tolist()
    merge = int(getattr(adapter.model.visual, "spatial_merge_size", 2))
    height, width = int(grid[1]) // merge, int(grid[2]) // merge
    if height * width != values.numel():
        return _square_map(values)
    array = values.detach().cpu().reshape(height, width).numpy()
    array -= array.min()
    scale = float(array.max())
    return array / scale if scale > 0 else np.zeros_like(array)


def visual_importance(adapter: Any, image: Image.Image) -> tuple[np.ndarray, str]:
    if isinstance(adapter, (HuatuoOEAdapter, LlavaMedOEAdapter)):
        return _clip_attention(adapter, image), "final_vision_cls_to_patch_attention"
    if isinstance(adapter, HuluOEAdapter):
        return _hulu_token_importance(adapter, image), "final_native_visual_token_norm_proxy"
    if isinstance(adapter, Qwen25VLOEAdapter):
        return _qwen_token_importance(adapter, image), "final_native_visual_token_norm_proxy"
    raise TypeError(f"CVE visual importance unavailable for {type(adapter).__name__}")


def rebalance_image(
    image: Image.Image,
    importance: np.ndarray,
    *,
    mid_quantile: float = 0.60,
    high_quantile: float = 0.95,
    sharpen_factor: float = 1.5,
    brighten_factor: float = 1.1,
    blur_radius: float = 2.0,
    darken_factor: float = 0.8,
) -> Image.Image:
    """Apply CVE's disclosed mid-enhance/high-suppress regional policy."""

    if not 0 <= mid_quantile < high_quantile <= 1:
        raise ValueError("CVE quantiles must satisfy 0 <= mid < high <= 1")
    base = image.convert("RGB")
    saliency = Image.fromarray(np.uint8(np.clip(importance, 0, 1) * 255)).resize(
        base.size, Image.Resampling.BILINEAR
    )
    values = np.asarray(saliency, dtype=np.float32) / 255.0
    mid_cut = float(np.quantile(values, mid_quantile))
    high_cut = float(np.quantile(values, high_quantile))
    mid_mask = Image.fromarray(np.uint8(((values >= mid_cut) & (values < high_cut)) * 255))
    high_mask = Image.fromarray(np.uint8((values >= high_cut) * 255))
    enhanced = ImageEnhance.Sharpness(base).enhance(sharpen_factor)
    enhanced = ImageEnhance.Brightness(enhanced).enhance(brighten_factor)
    suppressed = base.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    suppressed = ImageEnhance.Brightness(suppressed).enhance(darken_factor)
    return Image.composite(suppressed, Image.composite(enhanced, base, mid_mask), high_mask)


@torch.inference_mode()
def generate_cve(
    adapter: Any,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int,
    seed: int,
    alpha: float = 2.0,
    beta: float = 0.2,
) -> tuple[Generation, dict[str, Any]]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    importance, importance_source = visual_importance(adapter, image)
    rebalanced_image = rebalance_image(image, importance)
    original = prefill_stream(adapter, image, prompt)
    rebalanced = prefill_stream(adapter, rebalanced_image, prompt)
    generated: list[int] = []
    nll: list[float] = []
    changed_steps = 0
    delta: list[float] = []
    eos_value = getattr(adapter.model.generation_config, "eos_token_id", None)
    if eos_value is None:
        eos_value = adapter.tokenizer.eos_token_id
    eos_tokens = {
        int(value)
        for value in (eos_value if isinstance(eos_value, (list, tuple)) else [eos_value])
        if value is not None
    }
    penalty = 1.2 if isinstance(adapter, HuatuoOEAdapter) else 1.0
    repetition = RepetitionPenaltyLogitsProcessor(penalty) if penalty != 1.0 else None
    for index in range(max_new_tokens):
        clean_scores, balanced_scores = original.logits, rebalanced.logits
        if repetition is not None:
            prefix_ids = list(generated)
            if adapter.tokenizer.bos_token_id is not None:
                prefix_ids.insert(0, int(adapter.tokenizer.bos_token_id))
            prefix = torch.tensor([prefix_ids], dtype=torch.long, device=clean_scores.device)
            clean_scores = repetition(prefix, clean_scores.clone())
            balanced_scores = repetition(prefix, balanced_scores.clone())
        scores = cve_logits(clean_scores, balanced_scores, alpha=alpha, beta=beta)
        token = int(scores.argmax(dim=-1).item())
        changed_steps += int(token != int(clean_scores.argmax(dim=-1).item()))
        probabilities = torch.softmax(scores, dim=-1)
        nll.append(float(-probabilities[0, token].clamp_min(1e-12).log().item()))
        delta.append(float((scores - clean_scores).abs().mean().item()))
        if token in eos_tokens:
            break
        generated.append(token)
        if index + 1 == max_new_tokens:
            break
        original.logits, original.past = original.step(token, original.past)
        rebalanced.logits, rebalanced.past = rebalanced.step(token, rebalanced.past)
    text = adapter.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return Generation(
        text=text,
        uncertainty=float(np.mean(nll)) if nll else float("inf"),
        token_count=len(generated),
        token_ids=tuple(generated),
    ), {
        "method": "CVE",
        "implementation": "clean-room cross-architecture paper-formula port",
        "importance_source": importance_source,
        "alpha": alpha,
        "beta": beta,
        "mid_quantile": 0.60,
        "high_quantile": 0.95,
        "operator_strengths": {"sharpen": 1.5, "brighten": 1.1, "blur_radius": 2.0, "darken": 0.8},
        "operator_strength_status": "paper_unspecified_explicit_port_defaults",
        "generated_steps": len(generated),
        "decision_changed_steps": changed_steps,
        "mean_logit_delta": float(np.mean(delta)) if delta else 0.0,
    }
