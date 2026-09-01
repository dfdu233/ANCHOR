"""Clean-room AGLA inference port for every native ANCHOR adapter.

AGLA combines the original and image-prompt-matched views and applies an
adaptive plausibility gate.  The paper leaves the exact augmentation strength
implementation-dependent; this module exposes that choice and records it in
the audit instead of silently calling an image-space heuristic official.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance

from .cross_model_cve import rebalance_image, visual_importance
from .cross_model_vcd import prefill_stream
from .models_oe import Generation


def agla_logits(
    original: torch.Tensor,
    augmented: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 0.5,
) -> torch.Tensor:
    """AGLA Eq. 3 plus its adaptive plausibility constraint (Eq. 4)."""
    if original.shape != augmented.shape:
        raise ValueError("AGLA streams must have identical vocabulary shape")
    if alpha < 0 or not 0 < beta <= 1:
        raise ValueError("alpha must be non-negative and beta in (0,1]")
    # The released implementation compares probabilities, which is equivalent
    # to this logit threshold and is numerically stable.
    cutoff = original.max(dim=-1, keepdim=True).values + float(np.log(beta))
    return (original + alpha * augmented).masked_fill(original < cutoff, -torch.inf)


def prompt_match_augment(
    image: Image.Image,
    importance: np.ndarray,
    *,
    strength: float = 1.35,
) -> Image.Image:
    """Deterministic image-space proxy for BLIP prompt matching.

    Importance is supplied by the native visual encoder.  Only the middle
    evidence band is enhanced, preserving global context and avoiding a
    method-specific object taxonomy.
    """
    if strength < 1:
        raise ValueError("augmentation strength must be >= 1")
    base = image.convert("RGB")
    saliency = Image.fromarray(np.uint8(np.clip(importance, 0, 1) * 255)).resize(
        base.size, Image.Resampling.BILINEAR
    )
    values = np.asarray(saliency, dtype=np.float32) / 255.0
    lo, hi = np.quantile(values, [0.35, 0.90])
    mask = Image.fromarray(np.uint8(((values >= lo) & (values <= hi)) * 255))
    enhanced = ImageEnhance.Contrast(base).enhance(float(strength))
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(float(strength))
    return Image.composite(enhanced, base, mask)


@torch.inference_mode()
def generate_agla(
    adapter: Any,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int,
    alpha: float = 2.0,
    beta: float = 0.5,
    seed: int = 42,
    augmented_image: Image.Image | None = None,
) -> tuple[Generation, dict[str, Any]]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if augmented_image is None:
        importance, source = visual_importance(adapter, image)
        augmented_image = prompt_match_augment(image, importance)
        implementation = "clean-room cross-architecture image-space prompt-match proxy"
        augmentation_status = "native_encoder_proxy"
        augmentation_strength: float | None = 1.35
    else:
        source = "BLIP-ITM-large GradCAM block 6"
        augmented_image = augmented_image.convert("RGB")
        implementation = "clean-room paper-formula port with precomputed BLIP-ITM prompt match"
        augmentation_status = "paper_disclosed_blip_itm_formula"
        augmentation_strength = None
    original = prefill_stream(adapter, image, prompt)
    augmented = prefill_stream(adapter, augmented_image, prompt)
    generated: list[int] = []
    nll: list[float] = []
    changed = 0
    eos_value = getattr(adapter.model.generation_config, "eos_token_id", None)
    if eos_value is None:
        eos_value = adapter.tokenizer.eos_token_id
    eos = {int(x) for x in (eos_value if isinstance(eos_value, (list, tuple)) else [eos_value]) if x is not None}
    for step in range(max_new_tokens):
        scores = agla_logits(original.logits, augmented.logits, alpha=alpha, beta=beta)
        clean = original.logits.argmax(dim=-1).item()
        token = int(scores.argmax(dim=-1).item())
        changed += int(token != clean)
        probs = torch.softmax(scores, dim=-1)
        nll.append(float(-probs[0, token].clamp_min(1e-12).log().item()))
        if token in eos:
            break
        generated.append(token)
        if step + 1 == max_new_tokens:
            break
        original.logits, original.past = original.step(token, original.past)
        augmented.logits, augmented.past = augmented.step(token, augmented.past)
    return Generation(adapter.tokenizer.decode(generated, skip_special_tokens=True).strip(), float(np.mean(nll)) if nll else float("inf"), len(generated), tuple(generated)), {
        "method": "AGLA",
        "implementation": implementation,
        "importance_source": source,
        "alpha": alpha, "beta": beta, "augmentation_strength": augmentation_strength,
        "augmentation_strength_status": augmentation_status,
        "generated_steps": len(generated), "decision_changed_steps": changed,
    }
