"""Local Hulu-Med and LLaVA-Med adapters with identical CE semantics."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


def _first_existing(*candidates: str) -> Path:
    """Resolve migrated model assets without silently accepting a missing path."""

    paths = [Path(value) for value in candidates]
    for path in paths:
        if path.exists():
            return path
    return paths[0]


HULU_PATH = _first_existing(
    "/home/dbw/models/Hulu-Med-4B",
    "/root/autodl-tmp/Hulu-Med/MedUniEval/datas/hub/"
    "models--ZJU-AI4H--Hulu-Med-14B/snapshots/"
    "b30d9161b8c23a79e20e1eca3891f63697531904",
)
LLAVA_PATH = _first_existing(
    "/home/dbw/models/LLaVA-Med-v1.5-mistral-7b",
    "/root/autodl-tmp/LLaVA-Med/microsoft/llava-med-v1.5-mistral-7b",
)
LLAVA_REPO = _first_existing(
    "/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5",
    "/root/autodl-tmp/LLaVA-Med",
)
LLAVA_IMAGE_PREPROCESS_VERSION = "deterministic-center-pad-v1"


def center_pad_image(image: Image.Image, background_color) -> Image.Image:
    """Match LLaVA's square padding without its random one-pixel offset."""
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    result = Image.new(image.mode, (side, side), background_color)
    result.paste(image, ((side - width) // 2, (side - height) // 2))
    return result


@dataclass
class CEForward:
    logits: np.ndarray
    features: np.ndarray
    sequence_nll: np.ndarray | None = None


class BaseAdapter:
    name: str

    def label_ids(self, labels: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for label in labels:
            encoded = self.tokenizer.encode(str(label), add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(
                    f"label must be exactly one token for {self.name}: {label!r} -> {encoded}"
                )
            ids.append(encoded[0])
        return ids

    def class_prototypes(self, labels: Sequence[str]) -> np.ndarray:
        ids = torch.tensor(
            self.label_ids(labels),
            device=self.model.get_output_embeddings().weight.device,
        )
        weight = self.model.get_output_embeddings().weight.index_select(0, ids)
        return weight.detach().float().cpu().numpy()

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HuluAdapter(BaseAdapter):
    name = "hulu-med-14b"

    def __init__(self, model_path: Path = HULU_PATH):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            # Hulu's pinned transformers 4.51 remote-code path forwards the
            # newer ``dtype`` kwarg into GenerationConfig, where a torch.dtype
            # is not JSON serializable.  The checkpoint's proven probe loader
            # uses the compatible legacy spelling.
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

    def _inputs(self, image: Image.Image, prompt: str):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor(
            images=[image],
            conversation=conversation,
            add_system_prompt=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                if key == "pixel_values":
                    value = value.to(dtype=self.model.dtype)
                inputs[key] = value.to(self.model.device)
        return inputs

    @torch.inference_mode()
    def forward_ce(
        self, images: Sequence[Image.Image], prompt: str, labels: Sequence[str]
    ) -> list[CEForward]:
        # Hulu's token-compression path is explicitly batch-size one, so style
        # variants are processed sequentially while keeping one loaded model.
        label_ids = self.label_ids(labels)
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
            last_logits = result.logits[0, -1, label_ids]
            last_hidden = result.hidden_states[-1][0, -1]
            outputs.append(
                CEForward(
                    logits=last_logits.float().cpu().numpy(),
                    features=last_hidden.float().cpu().numpy(),
                )
            )
            del result, inputs
        return outputs


class LlavaMedAdapter(BaseAdapter):
    name = "llava-med-v1.5-mistral-7b"

    def __init__(
        self, model_path: Path = LLAVA_PATH, conv_mode: str = "mistral_instruct"
    ):
        sys.path.insert(0, str(LLAVA_REPO))
        # The local checkpoint predates the transformers CVE gate; weights are
        # local safetensors. Keep the existing repository compatibility shim.
        import transformers.modeling_utils as modeling_utils
        import transformers.utils.import_utils as import_utils

        import_utils.check_torch_load_is_safe = lambda: None
        modeling_utils.check_torch_load_is_safe = lambda: None

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

    def _prompt_ids(self, prompt: str) -> torch.Tensor:
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        image_token = DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        conversation = conv_templates[self.conv_mode].copy()
        conversation.append_message(conversation.roles[0], image_token + "\n" + prompt)
        conversation.append_message(conversation.roles[1], None)
        text = conversation.get_prompt()
        return tokenizer_image_token(
            text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)

    def _process_images(self, images: Sequence[Image.Image]):
        processed = []
        pad = getattr(self.model.config, "image_aspect_ratio", None) == "pad"
        mean = tuple(int(value * 255) for value in self.image_processor.image_mean)
        for image in images:
            prepared = center_pad_image(image, mean) if pad else image
            tensor = self.image_processor.preprocess(
                prepared, return_tensors="pt"
            )["pixel_values"][0]
            processed.append(tensor)
        if all(tensor.shape == processed[0].shape for tensor in processed):
            return torch.stack(processed, dim=0)
        return processed

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
        label_ids = self.label_ids(labels)
        label_weight = self.model.get_output_embeddings().weight[label_ids]
        logits = hidden.to(label_weight.dtype) @ label_weight.T
        return [
            CEForward(
                logits=logits[i].float().cpu().numpy(),
                features=hidden[i].float().cpu().numpy(),
            )
            for i in range(count)
        ]


def load_adapter(name: str) -> BaseAdapter:
    normalized = name.lower().replace("_", "-")
    if normalized in {"hulu", "hulu-med", "hulu-med-14b"}:
        return HuluAdapter()
    if normalized in {"llava", "llava-med", "llava-med-v1.5"}:
        return LlavaMedAdapter()
    raise ValueError(f"unknown model adapter: {name}")
