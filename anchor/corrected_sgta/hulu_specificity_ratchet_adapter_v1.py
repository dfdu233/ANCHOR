#!/usr/bin/env python3
"""Inactive Hulu-Med replay plumbing for a future native Hulu substrate.

Hulu's native ``return_labels=True`` path supervises the assistant content and
the trailing ``<|im_end|>`` token.  The latter is a template token, not part of
the clinical target.  This adapter preserves the native rendered conversation,
processor token IDs, image-token preparation and model forward, but selects
only tokens whose contextual offsets cover the raw target text.  It refuses
any boundary, label, template or final-logit mismatch.  It is not currently
authorized for scientific use: Huatuo answers cannot establish spontaneous
Hulu behavior, and no physician-admitted Hulu-native full-answer substrate has
been built.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from .specificity_ratchet_teacher_forcing_v1 import (
    Condition,
    ContractError,
    RowExclusion,
    TeacherForcedTrace,
    validate_full_target_coverage,
)


VERSION = "hulu-specificity-ratchet-adapter-v1"
IGNORE_INDEX = -100
TEMPLATE_NAME = "hulu-med-4b-native-chat-template"
CURRENT_SCIENTIFIC_USE_AUTHORIZED = False


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def partition_hulu_assistant_target(
    *,
    rendered_assistant: str,
    generation_prompt: str,
    target: str,
    token_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
) -> dict[str, list[Any]]:
    """Select exact raw-target tokens from one native assistant message.

    Offsets are contextual character offsets over the rendered assistant
    message. Leading/trailing whitespace spill is allowed; spill into any
    non-whitespace template character is not.
    """

    if not target:
        raise RowExclusion("Hulu clinical target is empty")
    if len(token_ids) != len(offsets):
        raise ContractError("Hulu assistant token IDs and offsets differ in length")
    target_start = len(generation_prompt)
    target_end = target_start + len(target)
    if rendered_assistant[target_start:target_end] != target:
        raise ContractError("Hulu assistant rendering changed the raw target interval")
    if not rendered_assistant.startswith(generation_prompt):
        raise ContractError("Hulu assistant rendering changed the generation prompt")

    selected_ids: list[int] = []
    selected_offsets: list[tuple[int, int]] = []
    selected_assistant_indices: list[int] = []
    for index, (token_id, raw_offset) in enumerate(zip(token_ids, offsets)):
        if len(raw_offset) != 2:
            raise ContractError("Hulu tokenizer returned a malformed offset")
        start, end = int(raw_offset[0]), int(raw_offset[1])
        if start < 0 or end <= start or end > len(rendered_assistant):
            raise ContractError("Hulu tokenizer returned an invalid assistant offset")
        overlap_start, overlap_end = max(start, target_start), min(end, target_end)
        if overlap_end <= overlap_start:
            continue
        left_spill = rendered_assistant[start:overlap_start]
        right_spill = rendered_assistant[overlap_end:end]
        if any(not character.isspace() for character in left_spill + right_spill):
            raise RowExclusion(
                "Hulu target token spills into a non-whitespace template boundary"
            )
        selected_ids.append(int(token_id))
        selected_offsets.append(
            (overlap_start - target_start, overlap_end - target_start)
        )
        selected_assistant_indices.append(index)
    if not selected_ids:
        raise RowExclusion("Hulu target has no contextual content tokens")
    validate_full_target_coverage(target, selected_offsets, "unicode_character")
    return {
        "target_token_ids": selected_ids,
        "target_token_offsets": selected_offsets,
        "assistant_token_indices": selected_assistant_indices,
    }


class HuluSpecificityRatchetAdapter:
    """Native Hulu exact-context helper; not an authorized substrate."""

    def __init__(
        self,
        *,
        model_dir: Path,
        device_map: str = "auto",
        max_visual_tokens: int = 1024,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        if max_visual_tokens <= 0:
            raise ContractError("Hulu max_visual_tokens must be positive")
        self.model_dir = model_dir.resolve()
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_dir), trust_remote_code=True, local_files_only=True
        )
        self.processor.image_processor.max_tokens = max_visual_tokens
        self.tokenizer = self.processor.tokenizer
        self.model.eval()
        if not getattr(self.tokenizer, "is_fast", False):
            raise ContractError("Hulu adapter requires fast-tokenizer contextual offsets")
        self.blocks = self.model.model.layers
        self.layer_numbers = self._quartile_layers(len(self.blocks))
        self.layer_ids = [
            f"decoder_{number:02d}_of_{len(self.blocks):02d}"
            for number in self.layer_numbers
        ]
        chat_template = str(self.processor.chat_template)
        self.template_id = (
            f"{TEMPLATE_NAME}:{_sha256_bytes(chat_template.encode('utf-8'))[:16]}"
        )
        self._fingerprint = self._build_fingerprint(max_visual_tokens)

    @staticmethod
    def _quartile_layers(total: int) -> list[int]:
        if total < 4:
            raise ContractError("Hulu decoder exposes fewer than four layers")
        layers = [
            max(1, min(total, round(total * fraction)))
            for fraction in (0.25, 0.5, 0.75, 1.0)
        ]
        if len(set(layers)) != 4:
            raise ContractError("Hulu architecture-relative quartiles are not unique")
        return layers

    def _build_fingerprint(self, max_visual_tokens: int) -> dict[str, Any]:
        files = {}
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "processing_hulumed.py",
            "modeling_hulumed_qwen3.py",
            "tokenizer.json",
            "chat_template.json",
        ):
            path = self.model_dir / name
            if path.is_file():
                files[name] = _sha256_file(path)
        return {
            "adapter_version": VERSION,
            "model_family": "hulu-med-4b",
            "model_dir": str(self.model_dir),
            "model_files_sha256": files,
            "model_files_fingerprint": _canonical_sha256(files),
            "template_id": self.template_id,
            "generation_prompt_sha256": _sha256_bytes(
                str(self.processor.generation_prompt).encode("utf-8")
            ),
            "max_visual_tokens": max_visual_tokens,
            "layer_ids": self.layer_ids,
            "layer_fractions": [number / len(self.blocks) for number in self.layer_numbers],
            "target_selection_contract": (
                "native return_labels conversation; exact assistant contextual offsets; "
                "exclude supervised im_end/template tokens"
            ),
            "intermediate_logit_rule": "post-block residual -> native final norm -> native lm_head",
            "final_logit_rule": "selected ordinary CausalLM forward logits",
        }

    def fingerprint(self) -> dict[str, Any]:
        return dict(self._fingerprint)

    def _conversation(self, question: str, target: str, condition: Condition) -> list[dict[str, Any]]:
        user_content: list[dict[str, str]] = []
        if condition == "image":
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": question})
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target},
        ]

    def _prepare(
        self,
        *,
        image_path: Path | None,
        question: str,
        target: str,
        condition: Condition,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, dict[str, Any]]:
        if condition == "image":
            if image_path is None or not image_path.is_file():
                raise ContractError("Hulu image condition requires an existing image")
            image = Image.open(image_path).convert("RGB")
            image_hash = _sha256_file(image_path)
        else:
            if image_path is not None:
                raise ContractError("Hulu text-only condition received an image")
            image = None
            image_hash = None
        conversation = self._conversation(question, target, condition)
        processor_kwargs: dict[str, Any] = {
            "conversation": conversation,
            "return_labels": True,
            "add_system_prompt": False,
            "return_tensors": "pt",
        }
        if image is not None:
            processor_kwargs["images"] = [image]
        inputs = self.processor(**processor_kwargs)
        input_ids = inputs["input_ids"]
        processor_labels = inputs["labels"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            processor_labels = processor_labels.unsqueeze(0)
        if input_ids.shape[0] != 1 or processor_labels.shape != input_ids.shape:
            raise ContractError("Hulu processor did not return one aligned conversation")

        assistant_message = [{"role": "assistant", "content": target}]
        rendered_assistant = self.processor.apply_chat_template(
            assistant_message,
            tokenize=False,
            add_system_prompt=False,
            add_generation_prompt=False,
        )
        encoded_assistant = self.tokenizer(
            rendered_assistant,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        assistant_ids = [int(value) for value in encoded_assistant["input_ids"]]
        mapping = partition_hulu_assistant_target(
            rendered_assistant=rendered_assistant,
            generation_prompt=str(self.processor.generation_prompt),
            target=target,
            token_ids=assistant_ids,
            offsets=encoded_assistant["offset_mapping"],
        )
        if input_ids.shape[1] < len(assistant_ids):
            raise ContractError("Hulu conversation is shorter than its assistant message")
        assistant_start = input_ids.shape[1] - len(assistant_ids)
        expected_assistant = torch.tensor(
            assistant_ids, dtype=input_ids.dtype, device=input_ids.device
        )
        if not torch.equal(input_ids[0, assistant_start:], expected_assistant):
            raise ContractError("Hulu assistant tokenization is not the exact conversation suffix")
        target_positions = [
            assistant_start + int(index)
            for index in mapping["assistant_token_indices"]
        ]
        target_ids = torch.tensor(
            mapping["target_token_ids"], dtype=input_ids.dtype, device=input_ids.device
        )
        if not torch.equal(input_ids[0, target_positions], target_ids):
            raise ContractError("Hulu contextual target token IDs drifted")
        if not torch.equal(processor_labels[0, target_positions], target_ids):
            raise ContractError("Hulu target is not a subset of the native supervised slice")
        supervised = torch.nonzero(processor_labels[0].ne(IGNORE_INDEX), as_tuple=False).flatten()
        if bool((supervised < assistant_start).any()):
            raise ContractError("Hulu processor supervised tokens outside the assistant message")
        for position in supervised.tolist():
            assistant_index = int(position - assistant_start)
            if assistant_index in mapping["assistant_token_indices"]:
                continue
            start, end = encoded_assistant["offset_mapping"][assistant_index]
            target_start = len(str(self.processor.generation_prompt))
            target_end = target_start + len(target)
            if max(int(start), target_start) < min(int(end), target_end):
                raise ContractError("Hulu excluded native label overlaps the clinical target")

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[0, target_positions] = target_ids
        device = self.model.device
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if key == "pixel_values":
                    value = value.to(dtype=self.model.dtype)
                moved[key] = value.to(device)
            else:
                moved[key] = value
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        prepared_ids, attention, positions, _, embeddings, expanded_labels = (
            self.model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=moved.get("attention_mask"),
                position_ids=moved.get("position_ids"),
                labels=labels,
                pixel_values=moved.get("pixel_values"),
                grid_sizes=moved.get("grid_sizes"),
                merge_sizes=moved.get("merge_sizes"),
                modals=moved.get("modals"),
            )
        )
        if embeddings is None:
            if prepared_ids is None:
                raise ContractError("Hulu text-only preparation returned neither IDs nor embeddings")
            embeddings = self.model.get_model().embed_tokens(prepared_ids)
        if attention is None:
            attention = torch.ones(
                embeddings.shape[:2], dtype=torch.bool, device=embeddings.device
            )
        if expanded_labels is None or expanded_labels.shape[:2] != embeddings.shape[:2]:
            raise ContractError("Hulu multimodal preparation lost aligned target labels")
        prediction_mask = expanded_labels[:, 1:].ne(IGNORE_INDEX)
        expanded_target_ids = expanded_labels[:, 1:][prediction_mask]
        expected_target_ids = target_ids.to(expanded_target_ids.device)
        if not torch.equal(expanded_target_ids, expected_target_ids):
            raise RowExclusion("Hulu multimodal preparation changed target IDs/order")
        provenance = {
            "image_sha256": image_hash,
            "target_token_ids": list(mapping["target_token_ids"]),
            "target_token_offsets": list(mapping["target_token_offsets"]),
            "signed_input_ids_sha256": _sha256_bytes(
                ",".join(str(int(value)) for value in input_ids[0].tolist()).encode()
            ),
            "rendered_assistant_sha256": _sha256_bytes(
                rendered_assistant.encode("utf-8")
            ),
            "native_supervised_count": int(supervised.numel()),
            "clinical_target_count": len(target_positions),
        }
        return embeddings, attention, positions, prediction_mask, provenance

    def _forward_layers(
        self,
        *,
        embeddings: torch.Tensor,
        attention: torch.Tensor,
        positions: torch.Tensor | None,
        prediction_mask: torch.Tensor,
    ) -> list[list[float]]:
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for number in self.layer_numbers:
            def hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                *,
                layer_number: int = number,
            ) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_number] = hidden[:, :-1][prediction_mask].detach()

            handles.append(self.blocks[number - 1].register_forward_hook(hook))
        prediction_positions = torch.nonzero(
            prediction_mask[0], as_tuple=False
        ).flatten()
        if prediction_positions.numel() == 0:
            raise RowExclusion("Hulu target has no prediction positions")
        sequence_length = embeddings.shape[1]
        first_prediction = int(prediction_positions.min())
        logits_to_keep = sequence_length - first_prediction
        try:
            with torch.inference_mode():
                output = self.model(
                    input_ids=None,
                    attention_mask=attention,
                    position_ids=positions,
                    inputs_embeds=embeddings,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                    num_logits_to_keep=logits_to_keep,
                )
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(self.layer_numbers):
            raise ContractError("not every frozen Hulu decoder layer executed")
        kept_start = sequence_length - output.logits.shape[1]
        relative_positions = prediction_positions - kept_start
        if bool((relative_positions < 0).any()) or bool(
            (relative_positions >= output.logits.shape[1]).any()
        ):
            raise ContractError("Hulu reduced-logit window omitted a target prediction")
        standard_logits = output.logits[0, relative_positions].float()
        weight = self.model.get_output_embeddings().weight
        layer_logits = []
        for number in self.layer_numbers:
            normalized = self.model.model.norm(captured[number])
            layer_logits.append(F.linear(normalized.to(weight.dtype), weight).float())
        if not torch.allclose(
            layer_logits[-1], standard_logits, atol=2e-3, rtol=2e-3
        ):
            maximum = float((layer_logits[-1] - standard_logits).abs().max().item())
            raise ContractError(
                f"Hulu final logit lens does not match ordinary logits; max_abs={maximum:.6g}"
            )
        return [logits for logits in layer_logits]

    def score(
        self,
        *,
        image_path: Path | None,
        question: str,
        target: str,
        condition: Condition,
    ) -> TeacherForcedTrace:
        embeddings, attention, positions, prediction_mask, provenance = self._prepare(
            image_path=image_path,
            question=question,
            target=target,
            condition=condition,
        )
        layer_logits = self._forward_layers(
            embeddings=embeddings,
            attention=attention,
            positions=positions,
            prediction_mask=prediction_mask,
        )
        target_ids = torch.tensor(
            provenance["target_token_ids"],
            dtype=torch.long,
            device=layer_logits[0].device,
        )
        layer_gold = [
            F.log_softmax(logits, dim=-1)
            .gather(1, target_ids[:, None])[:, 0]
            .float()
            .cpu()
            .tolist()
            for logits in layer_logits
        ]
        serialization = {
            "adapter_version": VERSION,
            "template_id": self.template_id,
            "condition": condition,
            "question_sha256": _sha256_bytes(question.encode("utf-8")),
            "target_sha256": _sha256_bytes(target.encode("utf-8")),
            **provenance,
        }
        return TeacherForcedTrace(
            condition=condition,
            target=target,
            token_ids=list(provenance["target_token_ids"]),
            token_offsets=[tuple(pair) for pair in provenance["target_token_offsets"]],
            offset_unit="unicode_character",
            layer_ids=list(self.layer_ids),
            layer_gold_logp=layer_gold,
            serialized_input_sha256=_canonical_sha256(serialization),
            prompt_sha256=_sha256_bytes(question.encode("utf-8")),
            target_sha256=_sha256_bytes(target.encode("utf-8")),
            image_sha256=provenance["image_sha256"],
            template_id=self.template_id,
            contextual_offsets_certified=True,
        )


def build_adapter(config: dict[str, Any]) -> HuluSpecificityRatchetAdapter:
    if not CURRENT_SCIENTIFIC_USE_AUTHORIZED:
        raise ContractError(
            "Hulu scoring is not authorized for the Huatuo-sourced Specificity pack; "
            "a separate Hulu full-visible-answer substrate and native-ID gate are required"
        )
    allowed = {"model_dir", "device_map", "max_visual_tokens"}
    unknown = set(config) - allowed
    if unknown:
        raise ContractError(f"unknown Hulu adapter config keys: {sorted(unknown)}")
    return HuluSpecificityRatchetAdapter(
        model_dir=Path(config.get("model_dir", "/home/dbw/models/Hulu-Med-4B")),
        device_map=str(config.get("device_map", "auto")),
        max_visual_tokens=int(config.get("max_visual_tokens", 1024)),
    )
