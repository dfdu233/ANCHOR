"""Cross-architecture AvisC decoding kernel and image-space fallback.

The exact token mask is available to LLaVA hooks; other native adapters use a
documented image-space blind-region proxy so the same decision rule can be
evaluated across the complete model matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from .cross_model_cve import visual_importance
from .cross_model_vcd import prefill_stream
from .models_oe import Generation


def avisc_logits(
    original: torch.Tensor,
    masked_visual: torch.Tensor,
    *,
    alpha: float = 2.5,
    beta: float = 0.1,
) -> torch.Tensor:
    """AvisC contrast (1+alpha)l - alpha*l* with plausibility gate."""
    if original.shape != masked_visual.shape:
        raise ValueError("AvisC streams must have identical vocabulary shape")
    if alpha < 0 or not 0 < beta <= 1:
        raise ValueError("alpha must be non-negative and beta in (0,1]")
    cutoff = original.max(dim=-1, keepdim=True).values + float(np.log(beta))
    return ((1 + alpha) * original - alpha * masked_visual).masked_fill(original < cutoff, -torch.inf)


def blind_region_image(image: Image.Image, importance: np.ndarray, *, lamb: float = 1.0) -> Image.Image:
    """Keep only AvisC's blind (high-attention) region; fill context by mean."""
    if lamb < 0:
        raise ValueError("lambda must be non-negative")
    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    sal = np.asarray(Image.fromarray(np.uint8(np.clip(importance, 0, 1) * 255)).resize((base.shape[1], base.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    mask = sal > float(sal.mean() + lamb * sal.std())
    fill = np.full_like(base, base.reshape(-1, 3).mean(axis=0).astype(np.uint8))
    return Image.fromarray(np.where(mask[..., None], base, fill).astype(np.uint8), mode="RGB")


@torch.inference_mode()
def generate_avisc(adapter: Any, image: Image.Image, prompt: str, *, max_new_tokens: int, alpha: float = 2.5, beta: float = 0.1, lamb: float = 1.0, seed: int = 42) -> tuple[Generation, dict[str, Any]]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    importance, source = visual_importance(adapter, image)
    masked = blind_region_image(image, importance, lamb=lamb)
    original, masked_stream = prefill_stream(adapter, image, prompt), prefill_stream(adapter, masked, prompt)
    generated: list[int] = []
    nll: list[float] = []
    changed = 0
    eos_value = getattr(adapter.model.generation_config, "eos_token_id", None) or adapter.tokenizer.eos_token_id
    eos = {int(x) for x in (eos_value if isinstance(eos_value, (list, tuple)) else [eos_value]) if x is not None}
    for step in range(max_new_tokens):
        scores = avisc_logits(original.logits, masked_stream.logits, alpha=alpha, beta=beta)
        changed += int(int(scores.argmax()) != int(original.logits.argmax()))
        token = int(scores.argmax(dim=-1).item())
        probs = torch.softmax(scores, dim=-1)
        nll.append(float(-probs[0, token].clamp_min(1e-12).log().item()))
        if token in eos:
            break
        generated.append(token)
        if step + 1 == max_new_tokens:
            break
        original.logits, original.past = original.step(token, original.past)
        masked_stream.logits, masked_stream.past = masked_stream.step(token, masked_stream.past)
    return Generation(adapter.tokenizer.decode(generated, skip_special_tokens=True).strip(), float(np.mean(nll)) if nll else float("inf"), len(generated), tuple(generated)), {
        "method": "AvisC", "implementation": "clean-room cross-architecture image-space blind-region proxy", "importance_source": source,
        "alpha": alpha, "beta": beta, "lambda": lamb, "mask_space": "image_proxy", "generated_steps": len(generated), "decision_changed_steps": changed,
    }
