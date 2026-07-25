"""Case/whitespace-robust constrained-label adapters.

Each semantic class receives the maximum logit over an equal-sized set of
surface forms.  Max (rather than log-sum-exp) prevents tokenizer duplicate
forms from changing a class prior merely because they tokenize identically.
"""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image

from .models import CEForward, HuluAdapter, LlavaMedAdapter


class SurfaceFormMixin:
    def label_id_groups(self, labels: Sequence[str]) -> list[list[int]]:
        groups: list[list[int]] = []
        for label in labels:
            value = str(label)
            forms = (value, value.lower(), " " + value, " " + value.lower())
            ids: list[int] = []
            for form in forms:
                encoded = self.tokenizer.encode(form, add_special_tokens=False)
                if len(encoded) == 1 and encoded[0] not in ids:
                    ids.append(encoded[0])
            if not ids:
                raise ValueError(
                    f"no single-token surface form for {self.name}: {label!r}"
                )
            groups.append(ids)
        return groups


class HuluSurfaceAdapter(SurfaceFormMixin, HuluAdapter):
    @torch.inference_mode()
    def forward_ce(
        self, images: Sequence[Image.Image], prompt: str, labels: Sequence[str]
    ) -> list[CEForward]:
        groups = self.label_id_groups(labels)
        outputs: list[CEForward] = []
        for image in images:
            inputs = self._inputs(image, prompt)
            result = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
                num_logits_to_keep=1,
            )
            vocabulary_logits = result.logits[0, -1]
            class_logits = torch.stack(
                [vocabulary_logits[group].max() for group in groups]
            )
            vocabulary_log_probability = torch.log_softmax(
                vocabulary_logits.float(), dim=-1
            )
            sequence_nll = torch.stack(
                [-vocabulary_log_probability[group].max() for group in groups]
            )
            outputs.append(
                CEForward(
                    logits=class_logits.float().cpu().numpy(),
                    features=result.hidden_states[-1][0, -1].float().cpu().numpy(),
                    sequence_nll=sequence_nll.cpu().numpy(),
                )
            )
            del result, inputs
        return outputs


    @torch.inference_mode()
    def decode_ce(
        self, images: Sequence[Image.Image], prompt: str, max_new_tokens: int = 8
    ) -> list[str]:
        outputs = []
        for image in images:
            inputs = self._inputs(image, prompt)
            prompt_length = int(inputs["input_ids"].shape[1])
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            suffix = (
                generated[0, prompt_length:]
                if generated.shape[1] > prompt_length
                else generated[0]
            )
            outputs.append(self.tokenizer.decode(suffix, skip_special_tokens=True).strip())
            del generated, inputs
        return outputs


class LlavaMedSurfaceAdapter(SurfaceFormMixin, LlavaMedAdapter):
    @torch.inference_mode()
    def forward_ce(
        self, images: Sequence[Image.Image], prompt: str, labels: Sequence[str]
    ) -> list[CEForward]:
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
        base_output = self.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = base_output.last_hidden_state[:, -1]
        vocabulary_weight = self.model.get_output_embeddings().weight
        class_columns = []
        for group in self.label_id_groups(labels):
            surface_logits = (
                hidden.to(vocabulary_weight.dtype) @ vocabulary_weight[group].T
            )
            class_columns.append(surface_logits.max(-1).values)
        logits = torch.stack(class_columns, dim=-1)
        vocabulary_logits = hidden.to(vocabulary_weight.dtype) @ vocabulary_weight.T
        vocabulary_log_probability = torch.log_softmax(
            vocabulary_logits.float(), dim=-1
        )
        nll_columns = [
            -vocabulary_log_probability[:, group].max(-1).values
            for group in self.label_id_groups(labels)
        ]
        sequence_nll = torch.stack(nll_columns, dim=-1)
        return [
            CEForward(
                logits=logits[i].float().cpu().numpy(),
                features=hidden[i].float().cpu().numpy(),
                sequence_nll=sequence_nll[i].cpu().numpy(),
            )
            for i in range(count)
        ]


    @torch.inference_mode()
    def decode_ce(
        self, images: Sequence[Image.Image], prompt: str, max_new_tokens: int = 8
    ) -> list[str]:
        outputs = []
        for image in images:
            input_ids = self._prompt_ids(prompt).to(self.model.device)
            image_tensor = self._process_images([image])
            if isinstance(image_tensor, list):
                image_tensor = [
                    item.to(self.model.device, dtype=self.model.dtype) for item in image_tensor
                ]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=self.model.dtype)
            generated = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                attention_mask=torch.ones_like(input_ids, dtype=torch.long),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            suffix = generated[0, input_ids.shape[1]:] if generated.shape[1] > input_ids.shape[1] else generated[0]
            outputs.append(
                self.tokenizer.decode(suffix, skip_special_tokens=True).strip()
            )
            del generated, input_ids, image_tensor
        return outputs


def load_adapter(name: str):
    normalized = name.lower().replace("_", "-")
    if normalized in {"hulu", "hulu-med", "hulu-med-14b"}:
        return HuluSurfaceAdapter()
    if normalized in {"llava", "llava-med", "llava-med-v1.5"}:
        return LlavaMedSurfaceAdapter()
    raise ValueError(f"unknown model adapter: {name}")
