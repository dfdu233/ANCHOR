"""Open-ended generation adapters for local Hulu-Med and LLaVA-Med.

The sampled sequence used by ConfGen contains only identically configured
temperature samples.  Greedy decoding is generated and reported separately.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
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


def _decode_generations(tokenizer: Any, output: Any) -> list[Generation]:
    """Decode generation suffixes and processed-distribution mean NLL."""

    steps = len(output.scores)
    sequences = output.sequences
    if steps == 0:
        return [Generation("", float("inf"), 0) for _ in range(sequences.shape[0])]
    suffix = sequences[:, -steps:]
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    generations: list[Generation] = []
    for row in range(suffix.shape[0]):
        ids: list[int] = []
        nll: list[float] = []
        for step, token_tensor in enumerate(suffix[row]):
            token_id = int(token_tensor)
            if pad is not None and token_id == pad:
                break
            if eos is not None and token_id == eos:
                break
            token_log_probs = torch.log_softmax(
                output.scores[step][row].float(), dim=-1
            )
            ids.append(token_id)
            nll.append(float(-token_log_probs[token_id].item()))
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        uncertainty = float(np.mean(nll)) if nll else float("inf")
        generations.append(Generation(text, uncertainty, len(ids)))
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
    ) -> list[Generation]:
        raise NotImplementedError

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
    ) -> list[Generation]:
        inputs = self._inputs(image, prompt)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": count,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        output = self.model.generate(**inputs, **kwargs)
        result = _decode_generations(self.tokenizer, output)
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
    ) -> list[Generation]:
        from llava.mm_utils import process_images

        input_ids = self._prompt_ids(prompt).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        image_tensor = process_images([image], self.image_processor, self.model.config)
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
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        output = self.model.generate(input_ids, attention_mask=attention_mask, **kwargs)
        result = _decode_generations(self.tokenizer, output)
        del output, input_ids, attention_mask, image_tensor
        return result


def load_oe_adapter(name: str, llava_conv_mode: str = "mistral_instruct"):
    normalized = name.lower().replace("_", "-")
    if normalized in {"hulu", "hulu-med", "hulu-med-14b"}:
        return HuluOEAdapter()
    if normalized in {"llava", "llava-med", "llava-med-v1.5"}:
        return LlavaMedOEAdapter(conv_mode=llava_conv_mode)
    raise ValueError(f"unknown model adapter: {name}")
