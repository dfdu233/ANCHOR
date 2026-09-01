"""Open-ended generation adapters for local Huatuo, Hulu-Med and LLaVA-Med.

The sampled sequence used by ConfGen contains only identically configured
temperature samples.  Greedy decoding is generated and reported separately.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .models import HuluAdapter, LlavaMedAdapter


@dataclass(frozen=True)
class Generation:
    text: str
    uncertainty: float
    token_count: int
    token_ids: tuple[int, ...] = ()


def geometric_log_pool(log_probabilities: torch.Tensor, beta: float) -> torch.Tensor:
    """KL-barycenter of one original and one or more source-guided views."""

    if log_probabilities.ndim != 2 or log_probabilities.shape[0] < 1:
        raise ValueError("expected [views, vocabulary] log probabilities")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    if log_probabilities.shape[0] == 1 or beta == 0.0:
        pooled = log_probabilities[0]
    else:
        pooled = (1.0 - beta) * log_probabilities[0]
        pooled = pooled + beta * log_probabilities[1:].mean(dim=0)
    return pooled - torch.logsumexp(pooled, dim=-1)


def _decode_generations(
    tokenizer: Any,
    output: Any,
    model: Any | None = None,
    input_length: int | None = None,
) -> list[Generation]:
    """Decode generation suffixes and processed-distribution mean NLL."""

    steps = len(output.scores)
    sequences = output.sequences
    if steps == 0:
        return [Generation("", float("inf"), 0) for _ in range(sequences.shape[0])]
    # Beam search may continue exploring after the ultimately selected beam has
    # emitted EOS. In that case ``len(output.scores)`` is longer than the
    # selected suffix, so taking the final ``steps`` tokens can start inside the
    # prompt or at padding. The prompt boundary is the only stable boundary.
    suffix = (
        sequences[:, input_length:]
        if input_length is not None and sequences.shape[1] > input_length
        else sequences[:, -steps:]
    )
    transition_scores = None
    beam_indices = getattr(output, "beam_indices", None)
    if beam_indices is not None and model is not None:
        # Beam-search score tensors are indexed by live beams, not returned
        # sequences.  Hugging Face's helper follows beam ancestry and prevents
        # silently assigning another beam's likelihood to the selected text.
        transition_scores = model.compute_transition_scores(
            sequences,
            output.scores,
            beam_indices,
            normalize_logits=True,
        )
    eos_values = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    if model is not None:
        model_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if isinstance(model_eos, int):
            eos_values.add(model_eos)
        elif model_eos is not None:
            eos_values.update(int(value) for value in model_eos)
    pad = tokenizer.pad_token_id
    generations: list[Generation] = []
    for row in range(suffix.shape[0]):
        ids: list[int] = []
        nll: list[float] = []
        # Some inputs-embeds beam implementations (Huatuo/Qwen2) prepend a
        # dummy pad/eos ID before the first generated token.  It is a sequence
        # initializer, not an empty answer; retain later EOS as the real stop.
        start = 0
        special = set(eos_values)
        if pad is not None:
            special.add(pad)
        while start < suffix.shape[1] and int(suffix[row, start]) in special:
            if not any(int(token) not in special for token in suffix[row, start + 1 :]):
                break
            start += 1
        for position in range(start, suffix.shape[1]):
            score_index = position if start == 0 else position - start
            token_tensor = suffix[row, position]
            if score_index >= steps or (
                transition_scores is not None and score_index >= transition_scores.shape[1]
            ):
                break
            token_id = int(token_tensor)
            if pad is not None and token_id == pad:
                break
            if token_id in eos_values:
                break
            ids.append(token_id)
            if transition_scores is not None:
                nll.append(float(-transition_scores[row, score_index].item()))
            else:
                token_log_probs = torch.log_softmax(
                    output.scores[score_index][row].float(), dim=-1
                )
                nll.append(float(-token_log_probs[token_id].item()))
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        uncertainty = float(np.mean(nll)) if nll else float("inf")
        generations.append(Generation(text, uncertainty, len(ids), tuple(ids)))
    return generations


class OEAdapterMixin:
    @staticmethod
    def _seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _generate_once(
        self,
        image: Image.Image,
        prompt: str,
        count: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        num_beams: int = 1,
    ) -> list[Generation]:
        raise NotImplementedError

    def generate_control(
        self,
        image: Image.Image,
        prompt: str,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        num_beams: int,
        max_new_tokens: int,
        seed: int,
    ) -> Generation:
        """Generate one canonical greedy, beam, or sampling control."""

        if num_beams < 1:
            raise ValueError("num_beams must be positive")
        if do_sample and temperature <= 0:
            raise ValueError("sampling requires positive temperature")
        self._seed(seed)
        return self._generate_once(
            image,
            prompt,
            1,
            do_sample,
            temperature,
            top_p,
            max_new_tokens,
            seed,
            num_beams=num_beams,
        )[0]

    def generate_oe(
        self,
        image: Image.Image,
        prompt: str,
        candidates: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        candidate_batch: int,
    ) -> tuple[Generation, list[Generation]]:
        self._seed(seed)
        greedy = self._generate_once(
            image, prompt, 1, False, temperature, top_p, max_new_tokens, seed
        )[0]
        sampled: list[Generation] = []
        # A fresh per-chunk seed makes resume independent of prior qids.
        for start in range(0, candidates, max(1, candidate_batch)):
            count = min(max(1, candidate_batch), candidates - start)
            chunk_seed = seed + 1 + start
            self._seed(chunk_seed)
            try:
                chunk = self._generate_once(
                    image,
                    prompt,
                    count,
                    True,
                    temperature,
                    top_p,
                    max_new_tokens,
                    chunk_seed,
                )
            except (RuntimeError, AssertionError, ValueError):
                # Some remote-code multimodal generators require batch size one.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                chunk = []
                for offset in range(count):
                    item_seed = chunk_seed + offset
                    self._seed(item_seed)
                    chunk.extend(
                        self._generate_once(
                            image,
                            prompt,
                            1,
                            True,
                            temperature,
                            top_p,
                            max_new_tokens,
                            item_seed,
                        )
                    )
            sampled.extend(chunk)
        if len(sampled) != candidates:
            raise RuntimeError(f"expected {candidates} candidates, got {len(sampled)}")
        return greedy, sampled


class HuatuoOEAdapter(OEAdapterMixin):
    """Huatuo's native serialization with auditable generated-token scores."""

    name = "huatuogpt-vision-7b"

    def __init__(
        self,
        model_path: Path = Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
        huatuo_root: Path = Path("/home/dbw/HuatuoGPT-Vision"),
    ):
        sys.path.insert(0, str(huatuo_root.resolve()))
        from cli import HuatuoChatbot  # type: ignore

        self.bot = HuatuoChatbot(str(model_path), device="cuda:0")
        self.bot.debug = False
        self.model = self.bot.model
        self.tokenizer = self.bot.tokenizer

    def _inputs(self, image: Image.Image, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.bot.input_moderation(prompt)
        text = self.bot.insert_image_placeholder(text, 1)
        conversation = self.bot.get_conv_without_history(text)
        input_ids = self.bot.preprocess(conversation, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(self.bot.device)
        image_tensors = torch.stack(self.bot.get_image_tensors([image]))
        image_tensors = image_tensors.to(dtype=torch.bfloat16, device=self.bot.device)
        return input_ids, image_tensors

    @torch.inference_mode()
    def _generate_once(
        self,
        image: Image.Image,
        prompt: str,
        count: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        num_beams: int = 1,
    ) -> list[Generation]:
        input_ids, image_tensors = self._inputs(image, prompt)
        kwargs: dict[str, Any] = {
            "images": image_tensors,
            "use_cache": True,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 1,
            "repetition_penalty": 1.2,
            "do_sample": do_sample,
            "num_return_sequences": count,
            "num_beams": num_beams,
            "return_dict_in_generate": True,
            "output_scores": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        output = self.model.generate(input_ids, **kwargs)
        # Huatuo's multimodal ``generate`` rebuilds the prompt as embeddings
        # and returns generated IDs only (unlike plain HF causal generation).
        result = _decode_generations(self.tokenizer, output, self.model)
        del output, input_ids, image_tensors
        return result

    def close(self) -> None:
        del self.bot
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HuluOEAdapter(OEAdapterMixin, HuluAdapter):
    @torch.inference_mode()
    def _generate_once(
        self,
        image: Image.Image,
        prompt: str,
        count: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        num_beams: int = 1,
    ) -> list[Generation]:
        inputs = self._inputs(image, prompt)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": count,
            "num_beams": num_beams,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        output = self.model.generate(**inputs, **kwargs)
        # Hulu's remote multimodal generation also returns generated IDs only.
        result = _decode_generations(self.tokenizer, output, self.model)
        del output, inputs
        return result


class LlavaMedOEAdapter(OEAdapterMixin, LlavaMedAdapter):
    @torch.inference_mode()
    def generate_dg_smoothed(
        self,
        images: list[Image.Image],
        prompt: str,
        beta: float,
        max_new_tokens: int,
    ) -> tuple[Generation, dict[str, float]]:
        """Greedy decode from the per-step KL barycenter of aligned views."""

        if not images:
            raise ValueError("at least the original image is required")
        count = len(images)
        input_ids = self._prompt_ids(prompt).repeat(count, 1).to(self.model.device)
        image_tensor = self._process_images(images)
        if isinstance(image_tensor, list):
            image_tensor = [
                item.to(self.model.device, dtype=self.model.dtype)
                for item in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=self.model.dtype)
        _, position_ids, attention_mask, _, inputs_embeds, _ = (
            self.model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                image_tensor,
                image_sizes=[image.size for image in images],
            )
        )
        output = self.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
        )
        generated: list[int] = []
        pooled_nll: list[float] = []
        view_js: list[float] = []
        eos = self.tokenizer.eos_token_id
        past = output.past_key_values
        hidden = output.last_hidden_state[:, -1]
        output_weight = self.model.get_output_embeddings().weight
        for step in range(max_new_tokens):
            logits = hidden.to(output_weight.dtype) @ output_weight.T
            log_probabilities = torch.log_softmax(logits.float(), dim=-1)
            pooled = geometric_log_pool(log_probabilities, beta)
            token = int(pooled.argmax().item())
            pooled_nll.append(float(-pooled[token].item()))
            mean_probability = log_probabilities.exp().mean(dim=0)
            mean_log = mean_probability.clamp_min(1e-12).log()
            js_value = (
                log_probabilities.exp()
                * (log_probabilities - mean_log.unsqueeze(0))
            ).sum(dim=-1).mean()
            view_js.append(max(0.0, float(js_value.item())))
            if eos is not None and token == eos:
                break
            generated.append(token)
            if step + 1 == max_new_tokens:
                break
            token_ids = torch.full(
                (count, 1), token, dtype=torch.long, device=self.model.device
            )
            past_length = int(
                past.get_seq_length()
                if hasattr(past, "get_seq_length")
                else past[0][0].shape[-2]
            )
            next_attention = torch.ones(
                (count, past_length + 1),
                dtype=torch.long,
                device=self.model.device,
            )
            next_position = torch.full(
                (count, 1),
                past_length,
                dtype=torch.long,
                device=self.model.device,
            )
            output = self.model.model(
                input_ids=token_ids,
                attention_mask=next_attention,
                position_ids=next_position,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
            hidden = output.last_hidden_state[:, -1]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        generation = Generation(
            text=text,
            uncertainty=float(np.mean(pooled_nll)) if pooled_nll else float("inf"),
            token_count=len(generated),
        )
        diagnostics = {
            "mean_view_js": float(np.mean(view_js)) if view_js else 0.0,
            "max_view_js": float(np.max(view_js)) if view_js else 0.0,
        }
        del output, past, hidden, input_ids, image_tensor
        return generation, diagnostics

    @torch.inference_mode()
    def generate_orbit_current_steered(
        self,
        images: list[Image.Image],
        prompt: str,
        beta: float,
        max_new_tokens: int,
        null_rgb: tuple[int, int, int] = (123, 117, 104),
    ) -> tuple[Generation, dict[str, float]]:
        """Greedy decode with source-style visual-current orbit steering.

        ``images`` must contain the original image first followed by one or
        more same-content style views. At each decoding step, this computes a
        visual current ``J_m = h(T_m x) - h(x_null)`` and applies the minimum
        hidden-state intervention moving the original current toward the orbit
        mean: ``h' = h_0 + beta * (mean_m J_m - J_0)``.

        The final output is still unconstrained natural-language generation; no
        class-label logits are used as the prediction interface.
        """

        if not images:
            raise ValueError("at least the original image is required")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must lie in [0,1]")
        count = len(images)
        null_images = [Image.new("RGB", image.size, null_rgb) for image in images]

        def initial_state(batch_images: list[Image.Image]):
            input_ids = self._prompt_ids(prompt).repeat(count, 1).to(self.model.device)
            image_tensor = self._process_images(batch_images)
            if isinstance(image_tensor, list):
                image_tensor = [
                    item.to(self.model.device, dtype=self.model.dtype)
                    for item in image_tensor
                ]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=self.model.dtype)
            _, position_ids, attention_mask, _, inputs_embeds, _ = (
                self.model.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    None,
                    None,
                    None,
                    None,
                    image_tensor,
                    image_sizes=[image.size for image in batch_images],
                )
            )
            output = self.model.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                return_dict=True,
            )
            return output, output.past_key_values, output.last_hidden_state[:, -1]

        view_output, view_past, view_hidden = initial_state(images)
        null_output, null_past, null_hidden = initial_state(null_images)

        generated: list[int] = []
        nll: list[float] = []
        current_norms: list[float] = []
        orbit_variances: list[float] = []
        steering_norms: list[float] = []
        eos = self.tokenizer.eos_token_id
        output_weight = self.model.get_output_embeddings().weight

        for step in range(max_new_tokens):
            currents = view_hidden.float() - null_hidden.float()
            orbit_mean = currents.mean(dim=0)
            original_current = currents[0]
            steering = beta * (orbit_mean - original_current)
            steered_hidden = view_hidden[0].float() + steering
            logits = steered_hidden.to(output_weight.dtype) @ output_weight.T
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            token = int(log_probs.argmax().item())
            nll.append(float(-log_probs[token].item()))
            current_norms.append(float(original_current.norm().item()))
            orbit_variances.append(
                float(((currents - orbit_mean) ** 2).sum(dim=-1).mean().item())
            )
            steering_norms.append(float(steering.norm().item()))
            if eos is not None and token == eos:
                break
            generated.append(token)
            if step + 1 == max_new_tokens:
                break

            token_ids = torch.full(
                (count, 1), token, dtype=torch.long, device=self.model.device
            )

            def next_state(past):
                past_length = int(
                    past.get_seq_length()
                    if hasattr(past, "get_seq_length")
                    else past[0][0].shape[-2]
                )
                attention = torch.ones(
                    (count, past_length + 1),
                    dtype=torch.long,
                    device=self.model.device,
                )
                position = torch.full(
                    (count, 1),
                    past_length,
                    dtype=torch.long,
                    device=self.model.device,
                )
                output = self.model.model(
                    input_ids=token_ids,
                    attention_mask=attention,
                    position_ids=position,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                return output, output.past_key_values, output.last_hidden_state[:, -1]

            view_output, view_past, view_hidden = next_state(view_past)
            null_output, null_past, null_hidden = next_state(null_past)

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        generation = Generation(
            text=text,
            uncertainty=float(np.mean(nll)) if nll else float("inf"),
            token_count=len(generated),
        )
        diagnostics = {
            "mean_visual_current_norm": float(np.mean(current_norms)) if current_norms else 0.0,
            "mean_orbit_current_variance": float(np.mean(orbit_variances)) if orbit_variances else 0.0,
            "max_orbit_current_variance": float(np.max(orbit_variances)) if orbit_variances else 0.0,
            "mean_steering_norm": float(np.mean(steering_norms)) if steering_norms else 0.0,
            "beta": float(beta),
            "view_count": int(count),
        }
        del view_output, view_past, view_hidden, null_output, null_past, null_hidden
        return generation, diagnostics

    @torch.inference_mode()
    def _generate_once(
        self,
        image: Image.Image,
        prompt: str,
        count: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        num_beams: int = 1,
    ) -> list[Generation]:
        input_ids = self._prompt_ids(prompt).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        # Use the adapter's deterministic center padding.  The upstream helper
        # randomly chooses between two one-pixel offsets for odd dimensions,
        # which makes method-off identity depend on process RNG history.
        image_tensor = self._process_images([image])
        if isinstance(image_tensor, list):
            image_tensor = [
                item.to(self.model.device, dtype=self.model.dtype)
                for item in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=self.model.dtype)
        kwargs = {
            "images": image_tensor,
            "image_sizes": [image.size],
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": count,
            "num_beams": num_beams,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        output = self.model.generate(input_ids, attention_mask=attention_mask, **kwargs)
        # LLaVA prepares multimodal prompt embeddings inside its generate
        # override and returns generated IDs only.
        result = _decode_generations(self.tokenizer, output, self.model)
        del output, input_ids, attention_mask, image_tensor
        return result


class LlavaNextOEAdapter(LlavaMedOEAdapter):
    """Official LLaVA-NeXT/LLaVA-1.6 Vicuna-7B any-resolution path."""

    name = "llava-v1.6-vicuna-7b"

    def __init__(
        self,
        model_path: Path = Path("/home/dbw/models/llava-v1.6-vicuna-7b"),
        llava_root: Path = Path("/home/dbw/SECOND/lmms-eval-vicuna/LLaVA-NeXT"),
        conv_mode: str = "vicuna_v1",
    ):
        sys.path.insert(0, str(llava_root.resolve()))
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                str(model_path),
                None,
                get_model_name_from_path(str(model_path)),
                device_map="auto",
                load_8bit=False,
                load_4bit=False,
            )
        )
        self.model.eval()
        self.conv_mode = conv_mode

    def _process_images(self, images: list[Image.Image]):
        from llava.mm_utils import process_images

        return process_images(images, self.image_processor, self.model.config)


class Qwen25VLOEAdapter(OEAdapterMixin):
    """Qwen2.5-VL's official processor/chat-template generation path."""

    name = "qwen2.5-vl-7b-instruct"

    def __init__(
        self,
        model_path: Path = Path("/home/dbw/models/Qwen2.5-VL-7B-Instruct"),
        max_pixels: int = 512 * 28 * 28,
    ):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_fast=False,
            min_pixels=256 * 28 * 28,
            max_pixels=max_pixels,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        ).eval()
        self.tokenizer = self.processor.tokenizer

    def _inputs(self, image: Image.Image, prompt: str):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        return self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

    @torch.inference_mode()
    def _generate_once(
        self,
        image: Image.Image,
        prompt: str,
        count: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        num_beams: int = 1,
    ) -> list[Generation]:
        inputs = self._inputs(image, prompt)
        kwargs: dict[str, Any] = {
            "do_sample": do_sample,
            "num_beams": num_beams,
            "num_return_sequences": count,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.model.generation_config.eos_token_id,
        }
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        output = self.model.generate(**inputs, **kwargs)
        result = _decode_generations(
            self.tokenizer,
            output,
            self.model,
            input_length=inputs["input_ids"].shape[1],
        )
        del output, inputs
        return result

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_oe_adapter(name: str, llava_conv_mode: str = "mistral_instruct"):
    normalized = name.lower().replace("_", "-")
    if normalized in {"huatuo", "huatuogpt", "huatuogpt-vision-7b"}:
        return HuatuoOEAdapter()
    if normalized in {"hulu", "hulu-med", "hulu-med-14b"}:
        return HuluOEAdapter()
    if normalized in {"llava", "llava-med", "llava-med-v1.5"}:
        return LlavaMedOEAdapter(conv_mode=llava_conv_mode)
    if normalized in {"llava16", "llava-1.6", "llava-v1.6", "llava-next"}:
        return LlavaNextOEAdapter()
    if normalized in {"qwen", "qwen2.5-vl", "qwen2.5-vl-7b-instruct"}:
        return Qwen25VLOEAdapter()
    raise ValueError(f"unknown model adapter: {name}")
