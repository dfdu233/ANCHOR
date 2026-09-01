"""Native-generation DoLa ports for HuatuoGPT-Vision and Hulu-Med."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from types import MethodType
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from .models_oe import (
    Generation,
    HuatuoOEAdapter,
    HuluOEAdapter,
    LlavaMedOEAdapter,
    Qwen25VLOEAdapter,
    _decode_generations,
)


def _candidate_layers(model: Any, setting: str | list[int]) -> list[int]:
    config = model.config
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    final_layer = int(text_config.num_hidden_layers)
    tied = bool(getattr(config, "tie_word_embeddings", False))
    start = 0 if not tied else (2 if final_layer > 2 else max(0, final_layer - 1))
    if setting == "low":
        if start == final_layer // 2:
            return [start]
        stop = final_layer // 2 if final_layer <= 40 else 20
        return list(range(start, stop, 2))
    if setting == "high":
        start = final_layer // 2 if final_layer <= 40 else final_layer - 20
        return list(range(start, final_layer, 2))
    if isinstance(setting, list):
        return [int(layer) for layer in setting if int(layer) < final_layer]
    raise ValueError("dola_layers must be 'low', 'high', or a list of layers")


def _relative_top_filter(final: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    final = final.log_softmax(dim=-1)
    base = base.log_softmax(dim=-1)
    sorted_logits, _ = torch.sort(final, descending=True)
    minimum = sorted_logits[..., 0]
    threshold = torch.minimum(minimum, final.max(dim=-1).values + np.log(0.1)).unsqueeze(-1)
    mask = final < threshold
    return final.masked_fill(mask, -float("inf")), base.masked_fill(mask, -1e-3)


def _select_contrast(layers: list[int], candidates: dict[int, torch.Tensor], final: torch.Tensor) -> torch.Tensor:
    if not layers:
        raise ValueError("DoLa candidate layer set is empty")
    if len(layers) == 1:
        mature, premature = _relative_top_filter(final, candidates[layers[0]])
        return mature - premature
    stacked = torch.stack([candidates[layer] for layer in layers], dim=0)
    mature_prob = F.softmax(final, dim=-1)
    premature_prob = F.softmax(stacked, dim=-1)
    average = 0.5 * (mature_prob.unsqueeze(0) + premature_prob)
    mature_log = F.log_softmax(final, dim=-1).unsqueeze(0)
    premature_log = F.log_softmax(stacked, dim=-1)
    js = 0.5 * (
        F.kl_div(mature_log, average, reduction="none").mean(-1)
        + F.kl_div(premature_log, average, reduction="none").mean(-1)
    )
    chosen = layers[int(js.mean(-1).argmax().item())]
    mature, premature = _relative_top_filter(final, candidates[chosen])
    return mature - premature


@contextmanager
def _legacy_dola_forward(model: Any, setting: str | list[int]):
    """Inject official DoLa logits while retaining the model's native generate loop."""

    original = model.forward
    layers = _candidate_layers(model, setting)
    lm_head = model.get_output_embeddings()

    @wraps(original)
    def forward(this, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        output = original(*args, **kwargs)
        final = output.logits[:, -1:, :].float()
        candidates = {
            layer: lm_head(output.hidden_states[layer][:, -1:, :]).to(final.device)
            for layer in layers
        }
        output.logits[:, -1:, :] = _select_contrast(layers, candidates, final).to(output.logits.dtype)
        return output

    model.forward = MethodType(forward, model)
    try:
        yield layers
    finally:
        model.forward = original


@torch.inference_mode()
def generate_dola(
    adapter: Any,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int,
    seed: int,
    dola_layers: str | list[int] | None = "low",
) -> tuple[Generation, dict[str, Any]]:
    """Run Hugging Face's contrastive-layer decoder in each native VLM loop."""

    adapter._seed(seed)
    common = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_return_sequences": 1,
        "num_beams": 1,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": adapter.tokenizer.eos_token_id,
    }
    # Hulu's pinned Transformers fork exposes the official ``dola_layers``
    # generation mode. The Qwen2.5-VL runtime does not, so Qwen uses the same
    # official layer-contrast forward injection as Huatuo while retaining its
    # native multimodal ``generate`` loop.
    native_dola = dola_layers is not None and isinstance(adapter, HuluOEAdapter)
    if native_dola:
        common["dola_layers"] = dola_layers
    if isinstance(adapter, HuatuoOEAdapter):
        input_ids, images = adapter._inputs(image, prompt)
        context = _legacy_dola_forward(adapter.model, dola_layers) if dola_layers is not None else contextmanager(lambda: (yield []))()
        with context as selected_layers:
            output = adapter.model.generate(
                input_ids,
                images=images,
                min_new_tokens=1,
                repetition_penalty=1.2,
                eos_token_id=adapter.tokenizer.eos_token_id,
                **common,
            )
    elif isinstance(adapter, HuluOEAdapter):
        inputs = adapter._inputs(image, prompt)
        output = adapter.model.generate(**inputs, **common)
    elif isinstance(adapter, LlavaMedOEAdapter):
        input_ids = adapter._prompt_ids(prompt).to(adapter.model.device)
        image_tensor = adapter._process_images([image])
        if isinstance(image_tensor, list):
            image_tensor = [
                value.to(adapter.model.device, dtype=adapter.model.dtype)
                for value in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(adapter.model.device, dtype=adapter.model.dtype)
        context = _legacy_dola_forward(adapter.model, dola_layers) if dola_layers is not None else contextmanager(lambda: (yield []))()
        with context as selected_layers:
            output = adapter.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                attention_mask=torch.ones_like(input_ids, dtype=torch.long),
                **common,
            )
    elif isinstance(adapter, Qwen25VLOEAdapter):
        inputs = adapter._inputs(image, prompt)
        context = _legacy_dola_forward(adapter.model, dola_layers) if dola_layers is not None else contextmanager(lambda: (yield []))()
        with context as selected_layers:
            output = adapter.model.generate(**inputs, **common)
    else:
        raise TypeError(f"DoLa architecture port is unavailable for {type(adapter).__name__}")
    generation = _decode_generations(
        adapter.tokenizer,
        output,
        adapter.model,
        input_length=(inputs["input_ids"].shape[1] if isinstance(adapter, Qwen25VLOEAdapter) else None),
    )[0]
    return generation, {
        "method": "DoLa",
        "architecture_port": type(adapter).__name__,
        "decode_loop": "native_hf_generate",
        "dola_layers": dola_layers,
        "selected_candidate_layers": selected_layers if isinstance(adapter, (HuatuoOEAdapter, LlavaMedOEAdapter, Qwen25VLOEAdapter)) else dola_layers,
        "legacy_forward_port": isinstance(adapter, (HuatuoOEAdapter, LlavaMedOEAdapter, Qwen25VLOEAdapter)) and dola_layers is not None,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
