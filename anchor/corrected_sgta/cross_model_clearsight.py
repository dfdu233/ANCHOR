"""ClearSight/VAF equations and a cross-backbone image-space diagnostic.

The attention intervention is exposed as a standalone kernel for exact hooks;
the common matrix uses a deterministic visual-signal image proxy when a
backend does not expose compatible attention tensors.  Every output audit
states which path was used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from .cross_model_cve import visual_importance
from .cross_model_vcd import prefill_stream
from .models_oe import Generation


def vaf_attention_logits(
    logits: torch.Tensor,
    *,
    text_query: torch.Tensor,
    visual_key: torch.Tensor,
    system_key: torch.Tensor,
    alpha: float = 0.15,
    beta: float = 0.10,
) -> torch.Tensor:
    """ClearSight Eq. (attention-logit VAF) with broadcast-safe masks."""
    if text_query.shape[-1] not in (1, logits.shape[-1]) or text_query.shape[-2] not in (1, logits.shape[-2]):
        raise ValueError("text_query mask must broadcast to attention logits")
    if visual_key.shape[-1] != logits.shape[-1] or system_key.shape[-1] != logits.shape[-1]:
        raise ValueError("key masks must match attention logits")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    enhance = text_query & visual_key
    suppress = text_query & system_key
    return logits + float(alpha) * enhance.to(logits.dtype) * logits - float(beta) * suppress.to(logits.dtype) * logits


def visual_signal_image(image: Image.Image, importance: np.ndarray, *, alpha: float = 0.15, beta: float = 0.10) -> Image.Image:
    """Image-space proxy: enhance high-signal regions and suppress weak context."""
    base = image.convert("RGB")
    sal = np.asarray(Image.fromarray(np.uint8(np.clip(importance, 0, 1) * 255)).resize(base.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    high = sal >= np.quantile(sal, 0.75)
    low = sal <= np.quantile(sal, 0.25)
    enhanced = ImageEnhance.Contrast(base).enhance(1.0 + 2.0 * float(alpha))
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.0 + 2.0 * float(alpha))
    suppressed = ImageEnhance.Brightness(base.filter(ImageFilter.GaussianBlur(1.0))).enhance(1.0 - min(0.5, float(beta)))
    out = Image.composite(enhanced, base, Image.fromarray(np.uint8(high * 255)))
    return Image.composite(suppressed, out, Image.fromarray(np.uint8(low * 255)))


@torch.inference_mode()
def generate_clearsight(adapter: Any, image: Image.Image, prompt: str, *, max_new_tokens: int, alpha: float = 0.15, beta: float = 0.10, seed: int = 42) -> tuple[Generation, dict[str, Any]]:
    importance, source = visual_importance(adapter, image)
    steered = visual_signal_image(image, importance, alpha=alpha, beta=beta)
    stream = prefill_stream(adapter, steered, prompt)
    generated: list[int] = []
    nll: list[float] = []
    eos_value = getattr(adapter.model.generation_config, "eos_token_id", None) or adapter.tokenizer.eos_token_id
    eos = {int(x) for x in (eos_value if isinstance(eos_value, (list, tuple)) else [eos_value]) if x is not None}
    for step in range(max_new_tokens):
        scores = stream.logits
        token = int(scores.argmax(dim=-1).item())
        probs = torch.softmax(scores, dim=-1)
        nll.append(float(-probs[0, token].clamp_min(1e-12).log().item()))
        if token in eos:
            break
        generated.append(token)
        if step + 1 == max_new_tokens:
            break
        stream.logits, stream.past = stream.step(token, stream.past)
    return Generation(adapter.tokenizer.decode(generated, skip_special_tokens=True).strip(), float(np.mean(nll)) if nll else float("inf"), len(generated), tuple(generated)), {
        "method": "ClearSight", "implementation": "clean-room image-space VAF proxy", "importance_source": source,
        "alpha": alpha, "beta": beta, "attention_hook": False, "generated_steps": len(generated),
    }
