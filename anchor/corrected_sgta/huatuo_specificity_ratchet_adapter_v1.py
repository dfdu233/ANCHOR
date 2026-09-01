#!/usr/bin/env python3
"""Huatuo adapter for full-visible-answer Specificity Ratchet replay.

The active runtime uses an empty prefix and scores the model's complete frozen
visible OE answer in Huatuo's native assistant context.  Constraint tokens are
localized inside that answer; automatically shortened parent/child strings are
never model inputs.  VQA-RAD inputs are decoded as public JPEG/PNG pixels,
never as DICOM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from corrected_sgta.clinical_autoregressive_lockin_probe_v1 import (
    ContractError as LockinContractError,
    RowExclusion as LockinRowExclusion,
)
from corrected_sgta.huatuo_lockin_adapter_v1 import (
    ASSISTANT_SUFFIX,
    IGNORE_INDEX,
    HuatuoLockinAdapter,
    partition_answer_tokens,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import sha256_file
from corrected_sgta.specificity_ratchet_teacher_forcing_v1 import (
    Condition,
    ContractError,
    RowExclusion,
    TeacherForcedTrace,
)


VERSION = "huatuo-specificity-ratchet-full-visible-answer-adapter-v1"
RENDER_CONTRACT = "pillow-public-jpeg-png-load-convert-rgb-v1"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


class HuatuoSpecificityRatchetAdapter(HuatuoLockinAdapter):
    """Exact full-visible-answer adapter for admitted VQA-RAD edges."""

    @property
    def renderer_contract(self) -> str:
        return RENDER_CONTRACT

    @property
    def renderer_source_sha256(self) -> str:
        return sha256_file(Path(__file__).resolve())

    def fingerprint(self) -> dict[str, Any]:
        payload = super().fingerprint()
        payload.update(
            {
                "adapter_version": VERSION,
                "scientific_runtime": "specificity-ratchet-visible-replay-runtime-v1",
                "target_serialization_contract": (
                    "empty-prefix plus the complete frozen visible Huatuo OE answer in "
                    "the native assistant payload; no isolated parent/child target"
                ),
                "isolated_parent_child_runtime_prohibited": True,
                "native_identity_decode_contract": (
                    "direct-output.sequences-greedy-num_beams1-max512-"
                    "min1-repetition_penalty1.2"
                ),
                "native_identity_token_contract": (
                    "raw output.sequences retained; only terminal eos/pad may be "
                    "removed before exact contextual-target-ID comparison"
                ),
                "visual_swap_contract": "exact native visual-token-count equality",
                "accepted_image_suffixes": sorted(ALLOWED_SUFFIXES),
            }
        )
        payload.pop("prompt_end_position_contract", None)
        payload.pop("generation_decode_contract", None)
        return payload

    def _image_tensor(
        self, image_path: Path | None, condition: Condition
    ) -> tuple[torch.Tensor | None, str | None]:
        if condition == "text_only":
            if image_path is not None:
                raise ContractError("text-only condition received an image")
            return None, None
        if image_path is None or not image_path.is_file():
            raise ContractError("image condition requires an existing public image")
        if image_path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ContractError(
                f"Specificity adapter refuses non-public-raster suffix: {image_path.suffix!r}"
            )
        image_hash = sha256_file(image_path)
        tensor = self._image_tensor_cache.get(image_hash)
        if tensor is None:
            try:
                with Image.open(image_path) as opened:
                    opened.load()
                    image = opened.convert("RGB")
            except (OSError, UnidentifiedImageError) as exc:
                raise ContractError(f"failed to decode admitted public image: {exc}") from exc
            tensors = self.bot.get_image_tensors([image])
            if len(tensors) != 1:
                raise ContractError("Huatuo image processor did not return exactly one tensor")
            tensor = torch.stack(tensors).to(
                device=self.bot.model.device, dtype=torch.bfloat16
            )
            self._image_tensor_cache[image_hash] = tensor
        return tensor, image_hash

    def score(
        self,
        *,
        image_path: Path | None,
        question: str,
        target: str,
        condition: Condition,
    ) -> TeacherForcedTrace:
        if not target.strip():
            raise RowExclusion("complete visible answer is blank")
        try:
            trace = super().score(
                image_path=image_path,
                prompt=question,
                prefix="",
                continuation=target,
                condition=condition,
            )
        except LockinRowExclusion as exc:
            raise RowExclusion(str(exc)) from exc
        except LockinContractError as exc:
            raise ContractError(str(exc)) from exc
        if trace.prefix_token_ids or trace.prefix_token_offsets:
            raise ContractError("empty-prefix full-answer path emitted prefix tokens")
        if not trace.final_layer_matches_standard_logits:
            raise ContractError("final-layer gold logits differ from native model logits")
        return TeacherForcedTrace(
            condition=condition,
            target=target,
            token_ids=list(trace.continuation_token_ids),
            token_offsets=list(trace.continuation_token_offsets),
            offset_unit=trace.offset_unit,
            layer_ids=list(trace.layer_ids),
            layer_gold_logp=[list(row) for row in trace.layer_gold_logp],
            serialized_input_sha256=trace.serialized_input_sha256,
            prompt_sha256=trace.prompt_sha256,
            target_sha256=trace.continuation_sha256,
            image_sha256=trace.image_sha256,
            template_id=trace.template_id,
            contextual_offsets_certified=trace.contextual_offsets_certified,
        )

    def visual_token_count(self, *, image_path: Path, question: str) -> int:
        """Return the exact native visual expansion length for swap matching."""

        prompt_ids = self._prompt_ids(question, "image")
        placeholder_count = int(prompt_ids.eq(-200).sum())
        if placeholder_count != 1:
            raise ContractError("Huatuo visual-length audit requires one image placeholder")
        labels = torch.full_like(prompt_ids, IGNORE_INDEX)
        image_tensor, _ = self._image_tensor(image_path, "image")
        embeddings, _, _, _ = self._expand(
            prompt_ids, labels, image_tensor, "image"
        )
        count = int(embeddings.shape[1] - (prompt_ids.numel() - placeholder_count))
        if count <= 0:
            raise ContractError("Huatuo native visual expansion has no visual tokens")
        return count

    def contextual_target_ids(self, *, target: str) -> list[int]:
        """Return IDs selected by the exact full-answer teacher-forcing path."""

        if not target.strip():
            raise RowExclusion("complete frozen visible answer is blank")
        answer_text = target + ASSISTANT_SUFFIX
        encoded = self.bot.tokenizer(
            answer_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        mapping = partition_answer_tokens(
            answer_text=answer_text,
            prefix="",
            continuation=target,
            token_ids=[int(value) for value in encoded["input_ids"]],
            offsets=encoded["offset_mapping"],
        )
        ids = [int(value) for value in mapping["continuation_token_ids"]]
        if not ids:
            raise RowExclusion("complete frozen visible answer has no contextual IDs")
        return ids

    def generate_native_identity(
        self,
        *,
        image_path: Path,
        question: str,
        seed: int,
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        """Regenerate exactly under the frozen canonical OE decode contract."""

        if max_new_tokens <= 0:
            raise ContractError("native identity max_new_tokens must be positive")
        prompt_ids = self._prompt_ids(question, "image")
        image_tensor, image_hash = self._image_tensor(image_path, "image")
        if image_tensor is None or image_hash is None:
            raise ContractError("native identity generation lost its image")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            output = self.bot.model.generate(
                prompt_ids.unsqueeze(0),
                images=image_tensor,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
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
        if not generated_ids:
            raise ContractError("native identity generation returned no output.sequences IDs")
        text = self.bot.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        if not text:
            raise ContractError("native identity generation decoded to empty text")
        return {
            "text": text,
            "direct_output_sequence_ids": generated_ids,
            "directly_captured_output_sequences": True,
            "terminal_special_token_ids": sorted(
                {
                    int(value)
                    for value in (
                        self.bot.tokenizer.eos_token_id,
                        self.bot.tokenizer.pad_token_id,
                    )
                    if value is not None
                }
            ),
            "image_sha256": image_hash,
            "seed": seed,
            "decode_contract": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 1,
                "repetition_penalty": 1.2,
            },
            "hit_max_new_tokens": len(generated_ids) >= max_new_tokens,
        }


def create_adapter(config: dict[str, Any]) -> HuatuoSpecificityRatchetAdapter:
    allowed = {"model_dir", "huatuo_root", "device"}
    unknown = set(config) - allowed
    if unknown:
        raise ContractError(f"unknown Huatuo adapter config keys: {sorted(unknown)}")
    try:
        return HuatuoSpecificityRatchetAdapter(
            model_dir=Path(config.get("model_dir", "/home/dbw/models/HuatuoGPT-Vision-7B")),
            huatuo_root=Path(config.get("huatuo_root", "/home/dbw/HuatuoGPT-Vision")),
            device=str(config.get("device", "cuda:0")),
        )
    except LockinRowExclusion as exc:
        raise RowExclusion(str(exc)) from exc
    except LockinContractError as exc:
        raise ContractError(str(exc)) from exc
