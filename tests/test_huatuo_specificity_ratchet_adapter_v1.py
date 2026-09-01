import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from corrected_sgta.clinical_autoregressive_lockin_probe_v1 import (
    ContextualContinuationTrace,
)
from corrected_sgta.huatuo_lockin_adapter_v1 import HuatuoLockinAdapter
from corrected_sgta.huatuo_specificity_ratchet_adapter_v1 import (
    HuatuoSpecificityRatchetAdapter,
)
from corrected_sgta.specificity_ratchet_teacher_forcing_v1 import ContractError


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_full_target_wrapper_uses_empty_prefix_and_translates_exact_trace(monkeypatch):
    target = "A left lesion is present."
    observed = {}

    def fake_score(self, **kwargs):
        observed.update(kwargs)
        return ContextualContinuationTrace(
            condition=kwargs["condition"],
            prompt=kwargs["prompt"],
            prefix=kwargs["prefix"],
            continuation=kwargs["continuation"],
            prefix_token_ids=[],
            prefix_token_offsets=[],
            continuation_token_ids=[11, 12, 13, 14, 15],
            continuation_token_offsets=[(0, 1), (2, 6), (7, 13), (14, 16), (17, 25)],
            offset_unit="unicode_character",
            layer_ids=["decoder_07", "decoder_28"],
            layer_fractions=[0.25, 1.0],
            layer_gold_logp=[[-1.0] * 5, [-0.5] * 5],
            serialized_input_sha256=_sha(b"serialized"),
            prompt_sha256=_sha(kwargs["prompt"].encode()),
            prefix_sha256=_sha(b""),
            continuation_sha256=_sha(target.encode()),
            image_sha256=None,
            template_id="native-huatuo",
            contextual_offsets_certified=True,
            final_layer_matches_standard_logits=True,
        )

    monkeypatch.setattr(HuatuoLockinAdapter, "score", fake_score)
    adapter = HuatuoSpecificityRatchetAdapter.__new__(
        HuatuoSpecificityRatchetAdapter
    )
    trace = adapter.score(
        image_path=None,
        question="What is present?",
        target=target,
        condition="text_only",
    )
    assert observed["prefix"] == ""
    assert observed["continuation"] == target
    assert trace.target == target
    assert trace.target_sha256 == _sha(target.encode())
    assert trace.token_ids == [11, 12, 13, 14, 15]


def test_public_image_loader_converts_rgb_and_caches(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("L", (8, 6), color=127).save(image_path)
    calls = []

    class FakeBot:
        model = SimpleNamespace(device=torch.device("cpu"))

        def get_image_tensors(self, images):
            calls.append((images[0].mode, images[0].size))
            return [torch.ones((3, 4, 4), dtype=torch.float32)]

    adapter = HuatuoSpecificityRatchetAdapter.__new__(
        HuatuoSpecificityRatchetAdapter
    )
    adapter.bot = FakeBot()
    adapter._image_tensor_cache = {}
    first, first_hash = adapter._image_tensor(image_path, "image")
    second, second_hash = adapter._image_tensor(image_path, "image")
    assert calls == [("RGB", (8, 6))]
    assert first is second
    assert first.dtype == torch.bfloat16
    assert first_hash == second_hash


def test_public_image_loader_refuses_dicom_suffix(tmp_path):
    image_path = tmp_path / "sample.dicom"
    image_path.write_bytes(b"not-a-raster")
    adapter = HuatuoSpecificityRatchetAdapter.__new__(
        HuatuoSpecificityRatchetAdapter
    )
    adapter._image_tensor_cache = {}
    with pytest.raises(ContractError, match="non-public-raster suffix"):
        adapter._image_tensor(image_path, "image")


def test_fingerprint_declares_rgb_renderer_not_dicom():
    adapter = HuatuoSpecificityRatchetAdapter.__new__(
        HuatuoSpecificityRatchetAdapter
    )
    adapter._artifact = {"fingerprint": "model-fingerprint"}
    adapter._tokenizer_fingerprint = "tokenizer-fingerprint"
    adapter._template_sha256 = _sha(b"template")
    adapter.layer_ids = ["decoder_07", "decoder_28"]
    adapter.layer_fractions = [0.25, 1.0]
    fingerprint = adapter.fingerprint()
    assert fingerprint["renderer_contract"] == "pillow-public-jpeg-png-load-convert-rgb-v1"
    assert "dicom" not in fingerprint["renderer_contract"].lower()
    assert fingerprint["renderer_source_sha256"] == _sha(
        Path(
            "anchor/corrected_sgta/huatuo_specificity_ratchet_adapter_v1.py"
        ).read_bytes()
    )
    assert fingerprint["scientific_runtime"] == "specificity-ratchet-visible-replay-runtime-v1"
    assert fingerprint["isolated_parent_child_runtime_prohibited"] is True
    assert "output.sequences" in fingerprint["native_identity_token_contract"]


def test_contextual_target_ids_uses_exact_full_answer_payload():
    class FakeTokenizer:
        def __call__(self, text, add_special_tokens, return_offsets_mapping):
            assert text == "A finding. \n"
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            return {
                "input_ids": [10, 11, 12],
                "offset_mapping": [(0, 1), (2, 10), (10, 12)],
            }

    adapter = HuatuoSpecificityRatchetAdapter.__new__(
        HuatuoSpecificityRatchetAdapter
    )
    adapter.bot = SimpleNamespace(tokenizer=FakeTokenizer())
    assert adapter.contextual_target_ids(target="A finding.") == [10, 11]


def test_visual_token_count_uses_native_multimodal_expansion(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(image_path)
    adapter = HuatuoSpecificityRatchetAdapter.__new__(
        HuatuoSpecificityRatchetAdapter
    )
    monkeypatch.setattr(
        adapter,
        "_prompt_ids",
        lambda question, condition: torch.tensor([10, -200, 11, 12]),
    )
    monkeypatch.setattr(
        adapter,
        "_image_tensor",
        lambda path, condition: (torch.ones((1, 3, 2, 2)), "image-hash"),
    )
    monkeypatch.setattr(
        adapter,
        "_expand",
        lambda ids, labels, image, condition: (
            torch.ones((1, 9, 8)),
            torch.ones((1, 9), dtype=torch.bool),
            None,
            labels.unsqueeze(0),
        ),
    )
    assert adapter.visual_token_count(
        image_path=image_path, question="What is present?"
    ) == 6


def test_native_identity_generation_captures_output_sequences_directly(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(image_path)

    class FakeGenerateModel:
        def generate(self, input_ids, **kwargs):
            assert kwargs["max_new_tokens"] == 512
            return SimpleNamespace(sequences=torch.tensor([[21, 22, 23]]))

    class FakeTokenizer:
        eos_token_id = 99
        pad_token_id = 99

        def decode(self, ids, skip_special_tokens):
            return "A natural full answer."

    adapter = HuatuoSpecificityRatchetAdapter.__new__(
        HuatuoSpecificityRatchetAdapter
    )
    adapter.bot = SimpleNamespace(model=FakeGenerateModel(), tokenizer=FakeTokenizer())
    monkeypatch.setattr(
        adapter, "_prompt_ids", lambda question, condition: torch.tensor([1, -200, 2])
    )
    monkeypatch.setattr(
        adapter,
        "_image_tensor",
        lambda path, condition: (torch.ones((1, 3, 2, 2)), "image-hash"),
    )
    result = adapter.generate_native_identity(
        image_path=image_path,
        question="What is present?",
        seed=7,
        max_new_tokens=512,
    )
    assert result["text"] == "A natural full answer."
    assert result["direct_output_sequence_ids"] == [21, 22, 23]
    assert result["directly_captured_output_sequences"] is True
    assert result["terminal_special_token_ids"] == [99]
