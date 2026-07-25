"""Unified full-sequence model interface for ANCHOR.

Both adapters expose the same three operations: source/query embedding,
unrestricted candidate generation, and teacher-forced image/null evidence
trajectories. No task label space is passed to this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from corrected_sgta.anchor_transport import l2_normalize, validate_trajectory
from corrected_sgta.models import HULU_PATH, LLAVA_PATH
from corrected_sgta.models_oe import HuluOEAdapter, LlavaMedOEAdapter


NULL_RGB = (123, 117, 104)
IGNORE_INDEX = -100


@dataclass(frozen=True)
class SequenceEvidence:
    token_ids: list[int]
    token_text: list[str]
    image_token_log_probabilities: list[float]
    null_token_log_probabilities: list[float]
    trajectory: np.ndarray
    mean_image_log_probability: float
    eos_included: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "token_ids": self.token_ids,
            "token_text": self.token_text,
            "image_token_log_probabilities": self.image_token_log_probabilities,
            "null_token_log_probabilities": self.null_token_log_probabilities,
            "trajectory": self.trajectory.astype(float).tolist(),
            "mean_image_log_probability": self.mean_image_log_probability,
            "token_count": len(self.token_ids),
            "eos_included": self.eos_included,
        }


def _ensure_eos_supervision(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eos_token_id is None:
        raise RuntimeError("ANCHOR requires a tokenizer EOS token")
    supervised = labels.ne(IGNORE_INDEX)
    if bool((labels[supervised] == eos_token_id).any()):
        return input_ids, labels
    if int(input_ids[0, -1]) == eos_token_id:
        labels = labels.clone()
        labels[0, -1] = eos_token_id
        return input_ids, labels
    eos = torch.tensor([[eos_token_id]], dtype=input_ids.dtype)
    input_ids = torch.cat((input_ids.cpu(), eos), dim=1)
    labels = torch.cat(
        (labels.cpu(), torch.tensor([[eos_token_id]], dtype=labels.dtype)), dim=1
    )
    return input_ids, labels


def _selected_log_probabilities(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[int], np.ndarray]:
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(IGNORE_INDEX)
    token_ids = shifted_labels[mask]
    selected_logits = logits[:, :-1][mask].float()
    if token_ids.numel() == 0:
        raise RuntimeError("teacher forcing produced no supervised response tokens")
    values = -F.cross_entropy(selected_logits, token_ids, reduction="none")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("non-finite response-token log probabilities")
    return [int(value) for value in token_ids.detach().cpu()], values.detach().cpu().numpy()


def _make_evidence(
    tokenizer: Any,
    token_ids: list[int],
    image_logp: np.ndarray,
    null_logp: np.ndarray,
    max_sequence_tokens: int,
) -> SequenceEvidence:
    if token_ids != [int(value) for value in token_ids]:
        raise ValueError("token ids must be integers")
    if image_logp.shape != null_logp.shape or image_logp.shape != (len(token_ids),):
        raise ValueError("image/null token log-probability shapes differ")
    if len(token_ids) > max_sequence_tokens:
        raise ValueError(
            f"response has {len(token_ids)} tokens, exceeding fixed "
            f"T_max={max_sequence_tokens}"
        )
    eos_token_id = tokenizer.eos_token_id
    eos = np.asarray(
        [1.0 if token == eos_token_id else 0.0 for token in token_ids],
        dtype=np.float64,
    )
    if int(eos.sum()) != 1 or token_ids[-1] != eos_token_id:
        raise RuntimeError("the complete response trajectory must end in exactly one EOS")
    positions = (
        np.arange(1, len(token_ids) + 1, dtype=np.float64) / max_sequence_tokens
    )
    trajectory = np.column_stack(
        (-image_logp, image_logp - null_logp, positions, eos)
    )
    validate_trajectory(trajectory)
    return SequenceEvidence(
        token_ids=token_ids,
        token_text=[
            tokenizer.decode([token], skip_special_tokens=False)
            for token in token_ids
        ],
        image_token_log_probabilities=image_logp.astype(float).tolist(),
        null_token_log_probabilities=null_logp.astype(float).tolist(),
        trajectory=trajectory,
        mean_image_log_probability=float(image_logp.mean()),
        eos_included=True,
    )


class AnchorAdapterMixin:
    model_path: Path

    @torch.inference_mode()
    def _text_embedding(self, prompt: str) -> np.ndarray:
        tokens = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.model.device)
        if tokens.shape[1] == 0:
            raise ValueError("empty prompt cannot define a retrieval embedding")
        embedding = self.model.get_input_embeddings()(tokens)[0].float().mean(dim=0)
        return embedding.cpu().numpy()

    def _visual_embedding(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError

    def input_embedding(self, image: Image.Image, prompt: str) -> np.ndarray:
        visual = l2_normalize(self._visual_embedding(image))
        text = l2_normalize(self._text_embedding(prompt))
        # Equal modality weight, independent of their hidden widths.
        return np.concatenate((visual, text)) / np.sqrt(2.0)

    def sequence_evidence(
        self,
        image: Image.Image,
        prompt: str,
        response: str,
        max_sequence_tokens: int,
    ) -> SequenceEvidence:
        raise NotImplementedError

    def generate_candidates(
        self,
        image: Image.Image,
        prompt: str,
        *,
        candidate_budget: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        candidate_batch: int,
    ) -> list[dict[str, Any]]:
        if candidate_budget < 2:
            raise ValueError("candidate budget must include greedy plus samples")
        greedy, sampled = self.generate_oe(
            image=image,
            prompt=prompt,
            candidates=candidate_budget - 1,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
            candidate_batch=candidate_batch,
        )
        generations = [greedy, *sampled]
        if len(generations) != candidate_budget:
            raise RuntimeError("generation returned the wrong candidate count")
        output = []
        for index, generation in enumerate(generations):
            text = generation.text.strip()
            if generation.token_count >= max_new_tokens:
                raise RuntimeError(
                    f"candidate {index} reached max_new_tokens without EOS; "
                    "increase the fixed generation limit"
                )
            if not text:
                raise RuntimeError(f"empty generated candidate at index {index}")
            output.append(
                {
                    "candidate_id": f"candidate-{index}",
                    "acquisition_step": index,
                    "acquisition": "greedy" if index == 0 else "nucleus_sample",
                    "seed": seed if index == 0 else None,
                    "sampling_stream_index": (
                        None if index == 0 else index - 1
                    ),
                    "text": text,
                    "generation_mean_nll": float(generation.uncertainty),
                    "generation_token_count": int(generation.token_count),
                }
            )
        return output


class HuluAnchorAdapter(AnchorAdapterMixin, HuluOEAdapter):
    def __init__(self, model_path: Path = HULU_PATH):
        super().__init__(model_path=model_path)
        self.model_path = Path(model_path)

    @torch.inference_mode()
    def _visual_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self._inputs(image, "Describe the medical image briefly.")
        features = self.model.encode_images(
            inputs["pixel_values"], inputs["grid_sizes"], inputs["merge_sizes"]
        )
        value = features.float().mean(dim=0).cpu().numpy()
        del features, inputs
        return value

    def _teacher_inputs(
        self, image: Image.Image, prompt: str, response: str
    ) -> dict[str, Any]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": response},
        ]
        values = self.processor(
            images=[image],
            conversation=conversation,
            add_system_prompt=False,
            return_labels=True,
            return_tensors="pt",
        )
        input_ids, labels = _ensure_eos_supervision(
            values["input_ids"].unsqueeze(0)
            if values["input_ids"].ndim == 1
            else values["input_ids"],
            values["labels"].unsqueeze(0)
            if values["labels"].ndim == 1
            else values["labels"],
            self.tokenizer.eos_token_id,
        )
        values["input_ids"] = input_ids
        values["labels"] = labels
        if "attention_mask" in values:
            attention = values["attention_mask"]
            if attention.ndim == 1:
                attention = attention.unsqueeze(0)
            missing = input_ids.shape[1] - attention.shape[1]
            if missing < 0:
                raise RuntimeError("Hulu attention mask exceeds input length")
            if missing:
                attention = torch.cat(
                    (attention, torch.ones_like(input_ids[:, -missing:])), dim=1
                )
            values["attention_mask"] = attention
        for key, value in list(values.items()):
            if torch.is_tensor(value):
                if key == "pixel_values":
                    value = value.to(dtype=self.model.dtype)
                values[key] = value.to(self.model.device)
        return values

    @torch.inference_mode()
    def _token_logp(
        self, image: Image.Image, prompt: str, response: str
    ) -> tuple[list[int], np.ndarray]:
        inputs = self._teacher_inputs(image, prompt, response)
        labels = inputs.pop("labels")
        output = self.model(
            **inputs,
            use_cache=False,
            return_dict=True,
            num_logits_to_keep=0,
        )
        result = _selected_log_probabilities(output.logits, labels)
        del output, inputs, labels
        return result

    def sequence_evidence(
        self,
        image: Image.Image,
        prompt: str,
        response: str,
        max_sequence_tokens: int,
    ) -> SequenceEvidence:
        null_image = Image.new("RGB", image.size, NULL_RGB)
        image_ids, image_logp = self._token_logp(image, prompt, response)
        null_ids, null_logp = self._token_logp(null_image, prompt, response)
        if image_ids != null_ids:
            raise RuntimeError("Hulu image/null supervised token sequences differ")
        return _make_evidence(
            self.tokenizer, image_ids, image_logp, null_logp, max_sequence_tokens
        )


class LlavaMedAnchorAdapter(AnchorAdapterMixin, LlavaMedOEAdapter):
    def __init__(
        self, model_path: Path = LLAVA_PATH, conv_mode: str = "vicuna_v1"
    ):
        super().__init__(model_path=model_path, conv_mode=conv_mode)
        self.model_path = Path(model_path)

    @torch.inference_mode()
    def _visual_embedding(self, image: Image.Image) -> np.ndarray:
        tensors = self._process_images([image])
        if isinstance(tensors, list):
            tensors = tensors[0].unsqueeze(0)
        encoded = self.model.encode_images(
            tensors.to(self.model.device, dtype=self.model.dtype)
        )
        value = encoded[0].float().mean(dim=0).cpu().numpy()
        del encoded, tensors
        return value

    def _teacher_inputs(
        self, prompt: str, response: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from corrected_sgta.train_rule_dg_adapter import build_teacher_forcing

        input_ids, labels = build_teacher_forcing(self, prompt, response)
        return _ensure_eos_supervision(
            input_ids, labels, self.tokenizer.eos_token_id
        )

    @torch.inference_mode()
    def _token_logp(
        self, image: Image.Image, prompt: str, response: str
    ) -> tuple[list[int], np.ndarray]:
        input_ids, labels = self._teacher_inputs(prompt, response)
        ids = input_ids.to(self.model.device)
        labels = labels.to(self.model.device)
        pixels = self._process_images([image])
        if isinstance(pixels, list):
            pixels = [
                value.to(self.model.device, dtype=self.model.dtype)
                for value in pixels
            ]
        else:
            pixels = pixels.to(self.model.device, dtype=self.model.dtype)
        _, positions, mask, _, embeds, expanded_labels = (
            self.model.prepare_inputs_labels_for_multimodal(
                ids,
                None,
                None,
                None,
                labels,
                pixels,
                image_sizes=[image.size],
            )
        )
        output = self.model.model(
            input_ids=None,
            attention_mask=mask,
            position_ids=positions,
            inputs_embeds=embeds,
            use_cache=False,
            return_dict=True,
        )
        weight = self.model.get_output_embeddings().weight
        logits = output.last_hidden_state.to(weight.dtype) @ weight.T
        if expanded_labels is None:
            raise RuntimeError("LLaVA did not expand teacher-forcing labels")
        result = _selected_log_probabilities(logits, expanded_labels)
        del output, logits, pixels, ids, labels, embeds
        return result

    def sequence_evidence(
        self,
        image: Image.Image,
        prompt: str,
        response: str,
        max_sequence_tokens: int,
    ) -> SequenceEvidence:
        null_image = Image.new("RGB", image.size, NULL_RGB)
        image_ids, image_logp = self._token_logp(image, prompt, response)
        null_ids, null_logp = self._token_logp(null_image, prompt, response)
        if image_ids != null_ids:
            raise RuntimeError("LLaVA image/null supervised token sequences differ")
        return _make_evidence(
            self.tokenizer, image_ids, image_logp, null_logp, max_sequence_tokens
        )


def load_anchor_adapter(name: str, model_path: Path | None = None):
    normalized = name.lower().replace("_", "-")
    if normalized in {"hulu", "hulu-med", "hulu-med-14b"}:
        return HuluAnchorAdapter(model_path or HULU_PATH)
    if normalized in {"llava", "llava-med", "llava-med-v1.5"}:
        return LlavaMedAnchorAdapter(model_path or LLAVA_PATH)
    raise ValueError(f"unknown ANCHOR model adapter: {name}")
