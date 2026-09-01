"""Architecture ports of Visual Contrastive Decoding for medical VLMs.

The decision rule is kept identical across adapters:

    z_vcd = (1 + alpha) z_image - alpha z_noisy

and tokens outside VCD's adaptive plausibility constraint are masked.  Only
the multimodal prefill and cached one-token step are model-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList, RepetitionPenaltyLogitsProcessor

from .models_oe import (
    Generation,
    HuatuoOEAdapter,
    HuluOEAdapter,
    LlavaMedOEAdapter,
    Qwen25VLOEAdapter,
    _decode_generations,
)


@dataclass
class _Stream:
    logits: torch.Tensor
    past: Any
    step: Callable[[int, Any], tuple[torch.Tensor, Any]]


def add_diffusion_noise(
    tensor: torch.Tensor, noise_step: int = 500, *, generator: torch.Generator
) -> torch.Tensor:
    """VCD's released 1,000-step sigmoid schedule with explicit RNG state."""

    if not 0 <= noise_step < 1000:
        raise ValueError("noise_step must lie in [0, 999]")
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000, device=tensor.device))
    betas = betas * (0.5e-2 - 1e-5) + 1e-5
    alpha_bar = torch.cumprod(1 - betas, dim=0)[noise_step]
    noise = torch.randn(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        generator=generator,
    )
    return alpha_bar.sqrt() * tensor + (1 - alpha_bar).sqrt() * noise


def noise_visual_input(
    value: torch.Tensor | list[torch.Tensor],
    *,
    noise_step: int,
    generator: torch.Generator,
) -> torch.Tensor | list[torch.Tensor]:
    """Apply the same VCD schedule to fixed- or any-resolution inputs."""
    if isinstance(value, list):
        return [
            add_diffusion_noise(item, noise_step=noise_step, generator=generator)
            for item in value
        ]
    return add_diffusion_noise(value, noise_step=noise_step, generator=generator)


def vcd_logits(
    image_logits: torch.Tensor,
    noisy_logits: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> torch.Tensor:
    """Apply the released VCD contrast and adaptive plausibility constraint."""

    if image_logits.shape != noisy_logits.shape:
        raise ValueError("VCD streams must have identical vocabulary shape")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not 0 < beta <= 1:
        raise ValueError("beta must lie in (0, 1]")
    cutoff = image_logits.max(dim=-1, keepdim=True).values + float(np.log(beta))
    contrasted = (1 + alpha) * image_logits - alpha * noisy_logits
    return contrasted.masked_fill(image_logits < cutoff, -torch.inf)


def _huatuo_stream(
    adapter: HuatuoOEAdapter,
    input_ids: torch.Tensor,
    image_tensor: torch.Tensor,
) -> _Stream:
    model = adapter.model
    (
        _,
        position_ids,
        attention_mask,
        _,
        inputs_embeds,
        _,
    ) = model.prepare_inputs_labels_for_multimodal_new(
        input_ids, None, None, None, None, image_tensor
    )
    output = model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        return_dict=True,
    )

    def step(token: int, past: Any) -> tuple[torch.Tensor, Any]:
        token_ids = torch.tensor([[token]], dtype=torch.long, device=model.device)
        result = model(
            input_ids=token_ids,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        return result.logits[:, -1].float(), result.past_key_values

    return _Stream(output.logits[:, -1].float(), output.past_key_values, step)


def _hulu_stream(adapter: HuluOEAdapter, inputs: dict[str, Any]) -> _Stream:
    model = adapter.model
    # Hulu expands one image placeholder into many visual embeddings.  Native
    # ``generate`` performs this expansion before entering HF generation and
    # then carries the expanded attention mask through every cached step.  A
    # raw ``model(**inputs)`` prefill followed by mask-free token steps is
    # almost always numerically equivalent, but can flip a near-tied token;
    # reproduce the native state transition explicitly instead.
    (
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        _,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids=inputs.get("input_ids"),
        attention_mask=inputs.get("attention_mask"),
        position_ids=inputs.get("position_ids"),
        past_key_values=inputs.get("past_key_values"),
        labels=None,
        pixel_values=inputs.get("pixel_values"),
        grid_sizes=inputs.get("grid_sizes"),
        merge_sizes=inputs.get("merge_sizes"),
        modals=inputs.get("modals"),
    )
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        return_dict=True,
        num_logits_to_keep=1,
    )

    def step(token: int, past: Any) -> tuple[torch.Tensor, Any]:
        nonlocal attention_mask
        token_ids = torch.tensor([[token]], dtype=torch.long, device=model.device)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=-1,
        )
        step_position = attention_mask.long().cumsum(-1)[:, -1:] - 1
        cache_length = past.get_seq_length() if hasattr(past, "get_seq_length") else attention_mask.shape[-1] - 1
        cache_position = torch.tensor(
            [cache_length], dtype=torch.long, device=model.device
        )
        result = model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            position_ids=step_position,
            past_key_values=past,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
            num_logits_to_keep=1,
        )
        return result.logits[:, -1].float(), result.past_key_values

    return _Stream(output.logits[:, -1].float(), output.past_key_values, step)


def _qwen_stream(adapter: Qwen25VLOEAdapter, inputs: dict[str, Any]) -> _Stream:
    """Reproduce Qwen2.5-VL prefill and cached text steps for one image arm."""

    model = adapter.model
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    prepared = model.prepare_inputs_for_generation(
        input_ids,
        attention_mask=attention_mask,
        cache_position=cache_position,
        use_cache=True,
        pixel_values=inputs.get("pixel_values"),
        pixel_values_videos=inputs.get("pixel_values_videos"),
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        second_per_grid_ts=inputs.get("second_per_grid_ts"),
    )
    # ``prepare_inputs_for_generation`` already materializes ``use_cache`` in
    # current Qwen2.5-VL. Passing it again raises a duplicate-key TypeError.
    output = model(**prepared, return_dict=True, logits_to_keep=1)

    def step(token: int, past: Any) -> tuple[torch.Tensor, Any]:
        nonlocal attention_mask
        token_ids = torch.tensor([[token]], dtype=torch.long, device=model.device)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1
        )
        cache_length = past.get_seq_length()
        prepared_step = model.prepare_inputs_for_generation(
            token_ids,
            past_key_values=past,
            attention_mask=attention_mask,
            cache_position=torch.tensor([cache_length], device=model.device),
            use_cache=True,
        )
        result = model(**prepared_step, return_dict=True, logits_to_keep=1)
        return result.logits[:, -1].float(), result.past_key_values

    return _Stream(output.logits[:, -1].float(), output.past_key_values, step)


def _llava_stream(
    adapter: LlavaMedOEAdapter,
    input_ids: torch.Tensor,
    image_tensor: torch.Tensor | list[torch.Tensor],
    image_size: tuple[int, int],
) -> _Stream:
    """Prefill one LLaVA-Med visual arm and retain its native KV cache."""

    model = adapter.model
    (
        _,
        position_ids,
        attention_mask,
        _,
        inputs_embeds,
        _,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        None,
        None,
        None,
        image_tensor,
        image_sizes=[image_size],
    )
    output = model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        return_dict=True,
    )

    def step(token: int, past: Any) -> tuple[torch.Tensor, Any]:
        nonlocal attention_mask
        token_ids = torch.tensor([[token]], dtype=torch.long, device=model.device)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1
            )
        result = model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        return result.logits[:, -1].float(), result.past_key_values

    return _Stream(output.logits[:, -1].float(), output.past_key_values, step)


def prefill_stream(adapter: Any, image: Image.Image, prompt: str) -> _Stream:
    """Create a cached next-token stream for one native model/image/prompt view."""

    if isinstance(adapter, HuatuoOEAdapter):
        input_ids, image_tensor = adapter._inputs(image, prompt)
        return _huatuo_stream(adapter, input_ids, image_tensor)
    if isinstance(adapter, HuluOEAdapter):
        return _hulu_stream(adapter, adapter._inputs(image, prompt))
    if isinstance(adapter, LlavaMedOEAdapter):
        input_ids = adapter._prompt_ids(prompt).to(adapter.model.device)
        image_tensor = adapter._process_images([image])
        if isinstance(image_tensor, list):
            image_tensor = [
                value.to(adapter.model.device, dtype=adapter.model.dtype)
                for value in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(adapter.model.device, dtype=adapter.model.dtype)
        return _llava_stream(adapter, input_ids, image_tensor, image.size)
    if isinstance(adapter, Qwen25VLOEAdapter):
        return _qwen_stream(adapter, adapter._inputs(image, prompt))
    raise TypeError(f"cached stream is unavailable for {type(adapter).__name__}")


def _streams(
    adapter: Any,
    image: Image.Image,
    prompt: str,
    *,
    noise_step: int,
    seed: int,
) -> tuple[_Stream, _Stream]:
    generator = torch.Generator(device=adapter.model.device)
    generator.manual_seed(seed)
    if isinstance(adapter, HuatuoOEAdapter):
        input_ids, image_tensor = adapter._inputs(image, prompt)
        noisy = add_diffusion_noise(
            image_tensor, noise_step=noise_step, generator=generator
        )
        return (
            _huatuo_stream(adapter, input_ids, image_tensor),
            _huatuo_stream(adapter, input_ids, noisy),
        )
    if isinstance(adapter, HuluOEAdapter):
        inputs = adapter._inputs(image, prompt)
        noisy_inputs = dict(inputs)
        noisy_inputs["pixel_values"] = add_diffusion_noise(
            inputs["pixel_values"], noise_step=noise_step, generator=generator
        )
        return _hulu_stream(adapter, inputs), _hulu_stream(adapter, noisy_inputs)
    if isinstance(adapter, LlavaMedOEAdapter):
        input_ids = adapter._prompt_ids(prompt).to(adapter.model.device)
        image_tensor = adapter._process_images([image])
        if isinstance(image_tensor, list):
            image_tensor = [
                value.to(adapter.model.device, dtype=adapter.model.dtype)
                for value in image_tensor
            ]
            noisy = noise_visual_input(
                image_tensor, noise_step=noise_step, generator=generator
            )
        else:
            image_tensor = image_tensor.to(adapter.model.device, dtype=adapter.model.dtype)
            noisy = noise_visual_input(
                image_tensor, noise_step=noise_step, generator=generator
            )
        return (
            _llava_stream(adapter, input_ids, image_tensor, image.size),
            _llava_stream(adapter, input_ids, noisy, image.size),
        )
    if isinstance(adapter, Qwen25VLOEAdapter):
        inputs = adapter._inputs(image, prompt)
        noisy_inputs = dict(inputs)
        noisy_inputs["pixel_values"] = add_diffusion_noise(
            inputs["pixel_values"], noise_step=noise_step, generator=generator
        )
        return _qwen_stream(adapter, inputs), _qwen_stream(adapter, noisy_inputs)
    raise TypeError(f"VCD architecture port is unavailable for {type(adapter).__name__}")


class _HuluVCDProcessor(LogitsProcessor):
    """Contrast native Hulu generation scores with a cached noisy-image stream."""

    def __init__(self, noisy: _Stream, *, alpha: float, beta: float):
        self.noisy = noisy
        self.alpha = alpha
        self.beta = beta
        self.calls = 0
        self.changed = 0
        self.contrast_l1: list[float] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.calls:
            token = int(input_ids[0, -1].item())
            self.noisy.logits, self.noisy.past = self.noisy.step(token, self.noisy.past)
        clean = scores.float()
        contrasted = vcd_logits(
            clean, self.noisy.logits.float(), alpha=self.alpha, beta=self.beta
        )
        self.changed += int(
            contrasted.argmax(dim=-1).item() != clean.argmax(dim=-1).item()
        )
        finite = torch.isfinite(contrasted)
        if finite.any():
            self.contrast_l1.append(
                float((contrasted[finite] - clean[finite]).abs().mean().item())
            )
        self.calls += 1
        return contrasted


@torch.inference_mode()
def _generate_hulu_vcd_native_loop(
    adapter: HuluOEAdapter,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int,
    seed: int,
    alpha: float,
    beta: float,
    noise_step: int,
) -> tuple[Generation, dict[str, Any]]:
    """Keep Hulu's exact HF generation loop and alter only its token scores."""

    clean_inputs = adapter._inputs(image, prompt)
    generator = torch.Generator(device=adapter.model.device)
    generator.manual_seed(seed)
    noisy_inputs = dict(clean_inputs)
    noisy_inputs["pixel_values"] = add_diffusion_noise(
        clean_inputs["pixel_values"], noise_step=noise_step, generator=generator
    )
    processor = _HuluVCDProcessor(
        _hulu_stream(adapter, noisy_inputs), alpha=alpha, beta=beta
    )
    output = adapter.model.generate(
        **clean_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_return_sequences=1,
        num_beams=1,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=adapter.tokenizer.eos_token_id,
        logits_processor=LogitsProcessorList([processor]),
    )
    generation = _decode_generations(adapter.tokenizer, output, adapter.model)[0]
    audit = {
        "method": "VCD",
        "architecture_port": type(adapter).__name__,
        "decode_loop": "native_hf_generate_with_vcd_logits_processor",
        "alpha": alpha,
        "beta": beta,
        "noise_step": noise_step,
        "sample": False,
        "repetition_penalty": 1.0,
        "generated_steps": generation.token_count,
        "decision_changed_steps": processor.changed,
        "mean_contrast_l1": float(np.mean(processor.contrast_l1)) if processor.contrast_l1 else 0.0,
    }
    return generation, audit


@torch.inference_mode()
def generate_vcd(
    adapter: Any,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int,
    seed: int,
    alpha: float = 1.0,
    beta: float = 0.1,
    noise_step: int = 500,
    sample: bool = False,
    repetition_penalty: float | None = None,
) -> tuple[Generation, dict[str, Any]]:
    """Decode one response and return activation diagnostics for the port."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    # The released method's off state is the unmodified native decoder, not a
    # numerically reimplemented two-stream loop with zero contrast.  Near-tied
    # logits can otherwise diverge even though the algebraic coefficient is
    # zero, which would make T1 test numerical path differences instead of
    # method identity.
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
            "method": "VCD",
            "architecture_port": type(adapter).__name__,
            "decode_loop": "native_method_off",
            "method_enabled": False,
            "alpha": alpha,
            "beta": beta,
            "noise_step": None,
            "sample": False,
            "repetition_penalty": None,
            "generated_steps": generation.token_count,
            "decision_changed_steps": 0,
            "mean_contrast_l1": 0.0,
        }
    if isinstance(adapter, HuluOEAdapter):
        if sample:
            raise NotImplementedError("the qualified Hulu VCD port currently supports greedy decoding only")
        return _generate_hulu_vcd_native_loop(
            adapter,
            image,
            prompt,
            max_new_tokens=max_new_tokens,
            seed=seed,
            alpha=alpha,
            beta=beta,
            noise_step=noise_step,
        )
    clean, noisy = _streams(
        adapter, image, prompt, noise_step=noise_step, seed=seed
    )
    generator = torch.Generator(device=clean.logits.device)
    generator.manual_seed(seed + 1)
    generated: list[int] = []
    nll: list[float] = []
    changed_steps = 0
    contrast_l1: list[float] = []
    eos_value = getattr(adapter.model.generation_config, "eos_token_id", None)
    if eos_value is None:
        eos_value = adapter.tokenizer.eos_token_id
    eos_tokens = {int(value) for value in (eos_value if isinstance(eos_value, (list, tuple)) else [eos_value]) if value is not None}
    if repetition_penalty is None:
        repetition_penalty = 1.2 if isinstance(adapter, HuatuoOEAdapter) else 1.0
    repetition = (
        RepetitionPenaltyLogitsProcessor(repetition_penalty)
        if repetition_penalty != 1.0
        else None
    )
    for index in range(max_new_tokens):
        clean_scores = clean.logits
        noisy_scores = noisy.logits
        if repetition is not None:
            prefix_ids = list(generated)
            # Huatuo calls HF generation with ``inputs_embeds``. Transformers
            # initializes its bookkeeping sequence with BOS; in this Qwen2
            # checkpoint BOS equals EOS, so omitting it changes termination.
            bos = adapter.tokenizer.bos_token_id
            if bos is None:
                bos = getattr(adapter.model.generation_config, "bos_token_id", None)
            if bos is not None:
                prefix_ids.insert(0, int(bos))
            prefix = torch.tensor(
                [prefix_ids], dtype=torch.long, device=clean.logits.device
            )
            clean_scores = repetition(prefix, clean_scores.clone())
            noisy_scores = repetition(prefix, noisy_scores.clone())
        scores = vcd_logits(clean_scores, noisy_scores, alpha=alpha, beta=beta)
        clean_token = int(clean_scores.argmax(dim=-1).item())
        probabilities = torch.softmax(scores, dim=-1)
        if sample:
            token = int(
                torch.multinomial(probabilities, 1, generator=generator).item()
            )
        else:
            token = int(scores.argmax(dim=-1).item())
        changed_steps += int(token != clean_token)
        nll.append(float(-probabilities[0, token].clamp_min(1e-12).log().item()))
        finite = torch.isfinite(scores)
        contrast_l1.append(
            float((scores[finite] - clean.logits[finite]).abs().mean().item())
            if finite.any()
            else 0.0
        )
        if token in eos_tokens:
            break
        generated.append(token)
        if index + 1 == max_new_tokens:
            break
        clean.logits, clean.past = clean.step(token, clean.past)
        noisy.logits, noisy.past = noisy.step(token, noisy.past)
    text = adapter.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return (
        Generation(
            text=text,
            uncertainty=float(np.mean(nll)) if nll else float("inf"),
            token_count=len(generated),
            token_ids=tuple(generated),
        ),
        {
            "method": "VCD",
            "architecture_port": type(adapter).__name__,
            "alpha": alpha,
            "beta": beta,
            "noise_step": noise_step,
            "sample": sample,
            "repetition_penalty": repetition_penalty,
            "generated_steps": len(generated),
            "decision_changed_steps": changed_steps,
            "mean_contrast_l1": float(np.mean(contrast_l1)) if contrast_l1 else 0.0,
        },
    )
