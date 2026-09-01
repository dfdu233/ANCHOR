#!/usr/bin/env python3
"""Production Huatuo adapter for the frozen autoregressive lock-in probe.

The adapter follows Huatuo's generation serialization exactly up to the
assistant boundary, then tokenizes the complete teacher-forced assistant
payload in context (prefix + continuation + Huatuo's ``" \n"`` suffix).
Only continuation tokens receive labels.  Tokens whose leading whitespace
straddles the prefix/continuation boundary are assigned to the clinical text
they actually cover; a token crossing non-whitespace on both sides is refused.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from corrected_sgta.clinical_autoregressive_lockin_probe_v1 import (
    Condition,
    ContextualContinuationTrace,
    ContractError,
    GreedyGenerationTrace,
    PromptEndTrace,
    RowExclusion,
)
from corrected_sgta.run_huatuo_dicom_render_pilot_v1 import (
    canonical_json_sha256,
    model_artifact_fingerprint,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    dicom_to_pil,
    import_huatuo,
    sha256_file,
)


VERSION = "huatuo-clinical-autoregressive-lockin-adapter-v1"
IGNORE_INDEX = -100
TEMPLATE_ID = "huatuo-preprocess-huatuo-generation-boundary-v1"
ASSISTANT_SUFFIX = " \n"
PROMPT_END_CONTRACT = "last_expanded_prompt_token_before_first_assistant_response_token"
RENDER_CONTRACT = "dicom-percentile-0p5-99p5-monochrome-aware-v1"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _signed_ids_sha256(values: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in values).encode()
    return _sha(payload)


def _content_bytes(text: str, start: int, end: int) -> bool:
    return bool(text[start:end]) and any(not character.isspace() for character in text[start:end])


def partition_answer_tokens(
    *,
    answer_text: str,
    prefix: str,
    continuation: str,
    token_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
) -> dict[str, list[Any]]:
    """Map contextual answer tokens back to prefix and continuation spans.

    Fast-tokenizer offsets are character offsets.  A leading-space BPE token
    can begin in the trailing whitespace of ``prefix`` and end in
    ``continuation``.  Clipping that whitespace is exact for the clinical text;
    crossing non-whitespace from both segments is not and fails closed.
    """

    expected = prefix + continuation + ASSISTANT_SUFFIX
    if answer_text != expected or len(token_ids) != len(offsets):
        raise ContractError("assistant payload or tokenizer offset identity drifted")
    prefix_end = len(prefix)
    continuation_end = prefix_end + len(continuation)
    output: dict[str, list[Any]] = {
        "prefix_token_ids": [],
        "prefix_token_offsets": [],
        "continuation_token_ids": [],
        "continuation_token_offsets": [],
        "continuation_sequence_indices": [],
    }
    for sequence_index, (token_id, raw_offset) in enumerate(zip(token_ids, offsets)):
        if len(raw_offset) != 2:
            raise ContractError("tokenizer returned a malformed contextual offset")
        start, end = int(raw_offset[0]), int(raw_offset[1])
        if start < 0 or end < start or end > len(answer_text):
            raise ContractError("tokenizer returned an invalid contextual offset")
        prefix_overlap = (max(start, 0), min(end, prefix_end))
        continuation_overlap = (max(start, prefix_end), min(end, continuation_end))
        prefix_content = (
            prefix_overlap[1] > prefix_overlap[0]
            and _content_bytes(answer_text, *prefix_overlap)
        )
        continuation_content = (
            continuation_overlap[1] > continuation_overlap[0]
            and _content_bytes(answer_text, *continuation_overlap)
        )
        if prefix_content and continuation_content:
            raise RowExclusion(
                "one contextual token crosses non-whitespace prefix and continuation content"
            )
        if continuation_content:
            clipped = (
                continuation_overlap[0] - prefix_end,
                continuation_overlap[1] - prefix_end,
            )
            output["continuation_token_ids"].append(int(token_id))
            output["continuation_token_offsets"].append(clipped)
            output["continuation_sequence_indices"].append(sequence_index)
        elif prefix_content:
            output["prefix_token_ids"].append(int(token_id))
            output["prefix_token_offsets"].append(prefix_overlap)
        elif prefix_overlap[1] > prefix_overlap[0] and not continuation_content:
            # Preserve a standalone trailing-space token in the prefix length
            # control when contextual tokenization actually emits one.
            output["prefix_token_ids"].append(int(token_id))
            output["prefix_token_offsets"].append(prefix_overlap)
    if continuation and not output["continuation_token_ids"]:
        raise RowExclusion("continuation has no contextual tokenizer tokens")
    return output


class HuatuoLockinAdapter:
    """Exact Huatuo/VinDr implementation of the model-independent contract."""

    def __init__(
        self,
        *,
        model_dir: Path,
        huatuo_root: Path,
        device: str = "cuda:0",
    ) -> None:
        self.model_dir = model_dir.resolve()
        self.huatuo_root = huatuo_root.resolve()
        self.device = device
        klass = import_huatuo(self.huatuo_root)
        self.bot = klass(str(self.model_dir), device=device)
        self.bot.model.eval()
        if not getattr(self.bot.tokenizer, "is_fast", False):
            raise ContractError("Huatuo lock-in adapter requires exact fast-tokenizer offsets")
        self.blocks = self.bot.model.model.layers
        self.layer_numbers = self._quartile_layers(len(self.blocks))
        self.layer_ids = [
            f"decoder_{number:02d}_of_{len(self.blocks):02d}"
            for number in self.layer_numbers
        ]
        self.layer_fractions = [number / len(self.blocks) for number in self.layer_numbers]
        if not math.isclose(self.layer_fractions[-1], 1.0, abs_tol=1e-12):
            raise ContractError("decoder quartiles do not include the final layer")
        self._image_tensor_cache: dict[str, torch.Tensor] = {}
        self._artifact = model_artifact_fingerprint(self.model_dir)
        tokenizer_payload = {
            "tokenizer_json_sha256": (
                sha256_file(self.model_dir / "tokenizer.json")
                if (self.model_dir / "tokenizer.json").is_file()
                else None
            ),
            "vocab_size": len(self.bot.tokenizer),
            "bos_token_id": self.bot.tokenizer.bos_token_id,
            "eos_token_id": self.bot.tokenizer.eos_token_id,
            "pad_token_id": self.bot.tokenizer.pad_token_id,
        }
        self._tokenizer_fingerprint = canonical_json_sha256(tokenizer_payload)
        self._template_sha256 = _sha(
            (
                "<|user|>\\n{moderated optional-<image>\\n prompt.strip()}\\n"
                "<|assistant|>\\n{prefix}{continuation} \\n"
            ).encode()
        )

    @staticmethod
    def _quartile_layers(total: int) -> list[int]:
        if total < 4:
            raise ContractError("Huatuo decoder exposes fewer than four layers")
        values = [max(1, min(total, round(total * fraction))) for fraction in (0.25, 0.5, 0.75, 1.0)]
        if len(set(values)) != 4:
            raise ContractError("architecture-relative quartiles are not unique")
        return values

    @property
    def renderer_contract(self) -> str:
        """Declare the exact image decoding path used by this adapter.

        Subclasses may override this only when they also override
        ``_image_tensor``.  Keeping the declaration dynamic prevents a valid
        full-target adapter for public JPEG inputs from inheriting a false
        DICOM provenance tag.
        """

        return RENDER_CONTRACT

    @property
    def renderer_source_sha256(self) -> str:
        return sha256_file(Path(dicom_to_pil.__code__.co_filename).resolve())

    def fingerprint(self) -> dict[str, Any]:
        return {
            "adapter_version": VERSION,
            "model_family": "huatuogpt-vision-7b",
            "model_artifact_fingerprint": self._artifact["fingerprint"],
            "tokenizer_fingerprint": self._tokenizer_fingerprint,
            "chat_template_sha256": self._template_sha256,
            "multimodal_expansion_contract": (
                "Huatuo prepare_inputs_labels_for_multimodal_new; one signed -200 "
                "placeholder replaced by the native projected visual-token sequence"
            ),
            "prompt_end_position_contract": PROMPT_END_CONTRACT,
            "layer_logit_lens_contract": (
                "selected post-decoder residual -> native final norm -> native BF16 LM head -> "
                "FP32 log_softmax; final row taken from the same standard CausalLM forward"
            ),
            "generation_decode_contract": (
                "greedy-num_beams1-sampling_false-max_new_tokens256; min_new_tokens=1; "
                "repetition_penalty=1.2; native generation-only token IDs"
            ),
            "layer_ids": self.layer_ids,
            "layer_fractions": self.layer_fractions,
            "template_id": TEMPLATE_ID,
            "assistant_suffix": repr(ASSISTANT_SUFFIX),
            "renderer_contract": self.renderer_contract,
            "renderer_source_sha256": self.renderer_source_sha256,
        }

    def _prompt_ids(self, prompt: str, condition: Condition) -> torch.Tensor:
        moderated = self.bot.input_moderation(prompt)
        if moderated != prompt:
            raise RowExclusion("Huatuo input moderation changed the frozen prompt")
        text = (
            self.bot.insert_image_placeholder(moderated, 1)
            if condition == "image"
            else moderated
        )
        ids = self.bot.preprocess(
            self.bot.get_conv_without_history(text), return_tensors="pt"
        ).to(self.bot.model.device)
        placeholder_count = int(ids.eq(-200).sum())
        expected = 1 if condition == "image" else 0
        if placeholder_count != expected:
            raise ContractError("Huatuo prompt has the wrong image-placeholder count")
        return ids

    def _image_tensor(self, image_path: Path | None, condition: Condition) -> tuple[torch.Tensor | None, str | None]:
        if condition == "text_only":
            if image_path is not None:
                raise ContractError("text-only condition received an image")
            return None, None
        if image_path is None or not image_path.is_file():
            raise ContractError("image condition requires an existing DICOM")
        image_hash = sha256_file(image_path)
        tensor = self._image_tensor_cache.get(image_hash)
        if tensor is None:
            image = dicom_to_pil(image_path)
            tensor = torch.stack(self.bot.get_image_tensors([image])).to(
                device=self.bot.model.device, dtype=torch.bfloat16
            )
            self._image_tensor_cache[image_hash] = tensor
        return tensor, image_hash

    def _expand(
        self,
        ids: torch.Tensor,
        labels: torch.Tensor,
        image_tensor: torch.Tensor | None,
        condition: Condition,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if condition == "text_only":
            embeddings = self.bot.model.get_model().embed_tokens(ids).unsqueeze(0)
            attention = torch.ones((1, ids.numel()), dtype=torch.bool, device=ids.device)
            positions = torch.arange(ids.numel(), device=ids.device).unsqueeze(0)
            return embeddings, attention, positions, labels.unsqueeze(0)
        attention_raw = torch.ones_like(ids, dtype=torch.bool)
        _, positions, attention, _, embeddings, expanded_labels = (
            self.bot.model.prepare_inputs_labels_for_multimodal_new(
                [ids], None, [attention_raw], None, [labels], image_tensor
            )
        )
        if embeddings is None or expanded_labels is None or attention is None:
            raise ContractError("Huatuo multimodal expansion returned an incomplete batch")
        return embeddings, attention, positions, expanded_labels

    def _capture_forward(
        self,
        *,
        embeddings: torch.Tensor,
        attention: torch.Tensor,
        positions: torch.Tensor | None,
        prediction_mask: torch.Tensor | None,
    ) -> tuple[Any, dict[int, torch.Tensor]]:
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for number in self.layer_numbers:
            def hook(_module: Any, _inputs: Any, output: Any, *, layer_number: int = number) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                selected = hidden[:, -1] if prediction_mask is None else hidden[:, :-1][prediction_mask]
                captured[layer_number] = selected.detach()

            handles.append(self.blocks[number - 1].register_forward_hook(hook))
        try:
            with torch.inference_mode():
                output = self.bot.model(
                    input_ids=None,
                    attention_mask=attention,
                    position_ids=positions,
                    inputs_embeds=embeddings,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(self.layer_numbers):
            raise ContractError("not every frozen Huatuo decoder layer executed")
        return output, captured

    def prompt_end(
        self,
        *,
        image_path: Path | None,
        prompt: str,
        condition: Condition,
    ) -> PromptEndTrace:
        ids = self._prompt_ids(prompt, condition)
        labels = torch.full_like(ids, IGNORE_INDEX)
        image_tensor, image_hash = self._image_tensor(image_path, condition)
        embeddings, attention, positions, _ = self._expand(
            ids, labels, image_tensor, condition
        )
        _, captured = self._capture_forward(
            embeddings=embeddings,
            attention=attention,
            positions=positions,
            prediction_mask=None,
        )
        serialization = {
            "template_id": TEMPLATE_ID,
            "condition": condition,
            "signed_prompt_ids_sha256": _signed_ids_sha256(ids.tolist()),
            "image_sha256": image_hash,
            "renderer_contract": self.renderer_contract if image_hash else None,
        }
        return PromptEndTrace(
            condition=condition,
            prompt=prompt,
            layer_ids=list(self.layer_ids),
            layer_fractions=list(self.layer_fractions),
            layer_prompt_end_hidden=[
                captured[number][0].float().cpu().tolist()
                for number in self.layer_numbers
            ],
            serialized_prompt_sha256=canonical_json_sha256(serialization),
            prompt_sha256=_sha(prompt.encode()),
            image_sha256=image_hash,
            template_id=TEMPLATE_ID,
            prompt_end_position_contract=PROMPT_END_CONTRACT,
            first_response_token_consumed=False,
            multimodal_expansion_certified=True,
        )

    def generate(
        self,
        *,
        image_path: Path,
        prompt: str,
    ) -> GreedyGenerationTrace:
        ids = self._prompt_ids(prompt, "image")
        image_tensor, image_hash = self._image_tensor(image_path, "image")
        assert image_tensor is not None and image_hash is not None
        with torch.inference_mode():
            output = self.bot.model.generate(
                ids.unsqueeze(0),
                images=image_tensor,
                do_sample=False,
                num_beams=1,
                max_new_tokens=256,
                min_new_tokens=1,
                repetition_penalty=1.2,
                eos_token_id=self.bot.tokenizer.eos_token_id,
                pad_token_id=(
                    self.bot.tokenizer.pad_token_id
                    if self.bot.tokenizer.pad_token_id is not None
                    else self.bot.tokenizer.eos_token_id
                ),
                return_dict_in_generate=True,
                output_scores=False,
                use_cache=True,
            )
        generated_ids = [
            int(value) for value in output.sequences[0].detach().cpu().tolist()
        ]
        text = self.bot.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        serialization = {
            "template_id": TEMPLATE_ID,
            "condition": "image",
            "signed_prompt_ids_sha256": _signed_ids_sha256(ids.tolist()),
            "image_sha256": image_hash,
            "renderer_contract": self.renderer_contract,
        }
        return GreedyGenerationTrace(
            text=text,
            generated_token_ids=generated_ids,
            image_sha256=image_hash,
            prompt_sha256=_sha(prompt.encode()),
            serialized_prompt_sha256=canonical_json_sha256(serialization),
            template_id=TEMPLATE_ID,
            decode_contract="greedy-num_beams1-sampling_false-max_new_tokens256",
            hit_max_new_tokens=len(generated_ids) >= 256,
        )

    def score(
        self,
        *,
        image_path: Path | None,
        prompt: str,
        prefix: str,
        continuation: str,
        condition: Condition,
    ) -> ContextualContinuationTrace:
        prompt_ids = self._prompt_ids(prompt, condition)
        answer_text = prefix + continuation + ASSISTANT_SUFFIX
        encoded = self.bot.tokenizer(
            answer_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        answer_ids_list = [int(value) for value in encoded["input_ids"]]
        mapping = partition_answer_tokens(
            answer_text=answer_text,
            prefix=prefix,
            continuation=continuation,
            token_ids=answer_ids_list,
            offsets=encoded["offset_mapping"],
        )
        answer_ids = torch.tensor(
            answer_ids_list, dtype=torch.long, device=self.bot.model.device
        )
        full_ids = torch.cat((prompt_ids, answer_ids))
        labels = torch.full_like(full_ids, IGNORE_INDEX)
        continuation_indices = [
            prompt_ids.numel() + int(index)
            for index in mapping["continuation_sequence_indices"]
        ]
        labels[continuation_indices] = full_ids[continuation_indices]
        image_tensor, image_hash = self._image_tensor(image_path, condition)
        embeddings, attention, positions, expanded_labels = self._expand(
            full_ids, labels, image_tensor, condition
        )
        prediction_mask = expanded_labels[:, 1:].ne(IGNORE_INDEX)
        target_ids = expanded_labels[:, 1:][prediction_mask]
        expected_targets = torch.tensor(
            mapping["continuation_token_ids"],
            dtype=torch.long,
            device=target_ids.device,
        )
        if not torch.equal(target_ids, expected_targets):
            raise RowExclusion("multimodal expansion changed contextual continuation IDs/order")
        output, captured = self._capture_forward(
            embeddings=embeddings,
            attention=attention,
            positions=positions,
            prediction_mask=prediction_mask,
        )
        standard_logits = output.logits[:, :-1][prediction_mask].float()
        standard_gold = F.log_softmax(standard_logits, dim=-1).gather(
            1, target_ids[:, None]
        )[:, 0]
        layer_gold = []
        weight = self.bot.model.get_output_embeddings().weight
        for number in self.layer_numbers:
            if number == len(self.blocks):
                gold = standard_gold
            else:
                normalized = self.bot.model.model.norm(captured[number])
                logits = F.linear(normalized.to(weight.dtype), weight).float()
                gold = F.log_softmax(logits, dim=-1).gather(
                    1, target_ids[:, None]
                )[:, 0]
            layer_gold.append(gold.cpu().tolist())
        serialization = {
            "template_id": TEMPLATE_ID,
            "condition": condition,
            "signed_input_ids_sha256": _signed_ids_sha256(full_ids.tolist()),
            "prompt": prompt,
            "prefix": prefix,
            "continuation": continuation,
            "assistant_suffix": ASSISTANT_SUFFIX,
            "image_sha256": image_hash,
            "renderer_contract": self.renderer_contract if image_hash else None,
        }
        return ContextualContinuationTrace(
            condition=condition,
            prompt=prompt,
            prefix=prefix,
            continuation=continuation,
            prefix_token_ids=list(mapping["prefix_token_ids"]),
            prefix_token_offsets=list(mapping["prefix_token_offsets"]),
            continuation_token_ids=list(mapping["continuation_token_ids"]),
            continuation_token_offsets=list(mapping["continuation_token_offsets"]),
            offset_unit="unicode_character",
            layer_ids=list(self.layer_ids),
            layer_fractions=list(self.layer_fractions),
            layer_gold_logp=layer_gold,
            serialized_input_sha256=canonical_json_sha256(serialization),
            prompt_sha256=_sha(prompt.encode()),
            prefix_sha256=_sha(prefix.encode()),
            continuation_sha256=_sha(continuation.encode()),
            image_sha256=image_hash,
            template_id=TEMPLATE_ID,
            contextual_offsets_certified=True,
            final_layer_matches_standard_logits=True,
        )


def create_adapter(config: dict[str, Any]) -> HuatuoLockinAdapter:
    allowed = {"model_dir", "huatuo_root", "device"}
    unknown = set(config) - allowed
    if unknown:
        raise ContractError(f"unknown Huatuo adapter config keys: {sorted(unknown)}")
    return HuatuoLockinAdapter(
        model_dir=Path(config.get("model_dir", "/home/dbw/models/HuatuoGPT-Vision-7B")),
        huatuo_root=Path(config.get("huatuo_root", "/home/dbw/HuatuoGPT-Vision")),
        device=str(config.get("device", "cuda:0")),
    )
