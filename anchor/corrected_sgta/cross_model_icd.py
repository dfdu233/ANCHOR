"""Architecture-portable Instruction Contrastive Decoding.

This is a disclosed port of the Apache-2.0 official ICD release.  It preserves
the released two-prompt logit contrast and adaptive plausibility constraint,
while each VLM retains its native image/prompt prefill and KV-cache path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import RepetitionPenaltyLogitsProcessor

from .cross_model_vcd import prefill_stream, vcd_logits
from .models_oe import Generation, HuatuoOEAdapter


DEFAULT_DISTURBANCE = (
    "You are a confused object detector. Provide only a fuzzy, unreliable "
    "impression of the image and avoid specific identification."
)


@torch.inference_mode()
def generate_icd(
    adapter: Any,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int,
    seed: int,
    alpha: float = 1.0,
    beta: float = 0.1,
    disturbance: str = DEFAULT_DISTURBANCE,
    repetition_penalty: float | None = None,
) -> tuple[Generation, dict[str, Any]]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not disturbance.strip():
        raise ValueError("ICD disturbance must be non-empty")
    if alpha == 0.0 and beta == 1.0:
        generation = adapter.generate_control(
            image,
            prompt,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        return generation, {
            "method": "ICD",
            "method_enabled": False,
            "architecture_port": type(adapter).__name__,
            "decode_loop": "native_method_off",
            "alpha": alpha,
            "beta": beta,
            "disturbance": None,
        }

    clean = prefill_stream(adapter, image, prompt)
    disturbed_prompt = disturbance.strip() + "\n\n" + prompt
    disturbed = prefill_stream(adapter, image, disturbed_prompt)
    generated: list[int] = []
    nll: list[float] = []
    changed_steps = 0
    contrast_l1: list[float] = []
    eos_value = getattr(adapter.model.generation_config, "eos_token_id", None)
    if eos_value is None:
        eos_value = adapter.tokenizer.eos_token_id
    eos_tokens = {
        int(value)
        for value in (
            eos_value if isinstance(eos_value, (list, tuple)) else [eos_value]
        )
        if value is not None
    }
    if repetition_penalty is None:
        repetition_penalty = 1.2 if isinstance(adapter, HuatuoOEAdapter) else 1.0
    repetition = (
        RepetitionPenaltyLogitsProcessor(repetition_penalty)
        if repetition_penalty != 1.0
        else None
    )
    for index in range(max_new_tokens):
        clean_scores = clean.logits
        disturbed_scores = disturbed.logits
        if repetition is not None:
            prefix_ids = list(generated)
            bos = adapter.tokenizer.bos_token_id
            if bos is not None:
                prefix_ids.insert(0, int(bos))
            prefix = torch.tensor(
                [prefix_ids], dtype=torch.long, device=clean_scores.device
            )
            clean_scores = repetition(prefix, clean_scores.clone())
            disturbed_scores = repetition(prefix, disturbed_scores.clone())
        scores = vcd_logits(
            clean_scores, disturbed_scores, alpha=alpha, beta=beta
        )
        token = int(scores.argmax(dim=-1).item())
        changed_steps += int(token != int(clean_scores.argmax(dim=-1).item()))
        probabilities = torch.softmax(scores, dim=-1)
        nll.append(float(-probabilities[0, token].clamp_min(1e-12).log().item()))
        finite = torch.isfinite(scores)
        contrast_l1.append(
            float((scores[finite] - clean_scores[finite]).abs().mean().item())
            if finite.any()
            else 0.0
        )
        if token in eos_tokens:
            break
        generated.append(token)
        if index + 1 == max_new_tokens:
            break
        clean.logits, clean.past = clean.step(token, clean.past)
        disturbed.logits, disturbed.past = disturbed.step(token, disturbed.past)

    text = adapter.tokenizer.decode(generated, skip_special_tokens=True).strip()
    generation = Generation(
        text=text,
        uncertainty=float(np.mean(nll)) if nll else float("inf"),
        token_count=len(generated),
        token_ids=tuple(generated),
    )
    return generation, {
        "method": "ICD",
        "method_enabled": True,
        "implementation": "official-formula cross-model port",
        "architecture_port": type(adapter).__name__,
        "decode_loop": "two_native_cached_prompt_streams",
        "alpha": alpha,
        "beta": beta,
        "disturbance": disturbance,
        "repetition_penalty": repetition_penalty,
        "generated_steps": len(generated),
        "decision_changed_steps": changed_steps,
        "mean_contrast_l1": float(np.mean(contrast_l1)) if contrast_l1 else 0.0,
    }
