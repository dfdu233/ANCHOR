import copy
import json
from types import SimpleNamespace

import pytest
import torch

from anchor.corrected_sgta.run_clinical_presupposition_generation_v1 import (
    COMMON_RESPONSE_FORM,
    exact_generate,
    prompt_contract,
    select_label_blind_images,
    surface_refusal,
    validate_shard,
)


def reference_rows():
    rows = []
    for image_id in ("a", "b", "c"):
        for finding_index, finding in enumerate(("cardiomegaly", "pleural_effusion")):
            votes = (ord(image_id) + finding_index) % 4
            rows.append(
                {
                    "image_id": image_id,
                    "dicom_relpath": f"train/{image_id}.dicom",
                    "experiment_split": "pilot",
                    "finding": finding,
                    "positive_votes": votes,
                    "reader_count": 3,
                    "reader_state": "undetermined",
                    "reference_contract_version": "test-v1",
                }
            )
    return rows


def test_prompts_are_distinct_pragmatic_tasks_with_shared_form():
    rows = prompt_contract()
    assert {row["name"] for row in rows} == {
        "neutral",
        "existential",
        "negative_obligation",
    }
    assert len({row["pragmatic_task"] for row in rows}) == 3
    assert all(COMMON_RESPONSE_FORM in row["prompt"] for row in rows)


def test_selection_membership_is_label_blind():
    rows = reference_rows()
    original = select_label_blind_images(rows, "pilot", 2, 42)
    changed = copy.deepcopy(rows)
    for row in changed:
        row["positive_votes"] = 3 - row["positive_votes"]
        row["reader_state"] = "changed"
    relabelled = select_label_blind_images(changed, "pilot", 2, 42)
    assert [row["item_id"] for row in original] == [row["item_id"] for row in relabelled]
    assert all(row["selection_uses_reader_labels"] is False for row in original)
    # The attached reference-universe hash should change even though selection cannot.
    assert [row["claim_universe_sha256"] for row in original] != [
        row["claim_universe_sha256"] for row in relabelled
    ]


def test_surface_refusal_is_explicitly_only_a_literal_diagnostic():
    matched = surface_refusal("I cannot diagnose this image; consult a radiologist.")
    assert matched["surface_refusal_match"] is True
    assert "diagnostic only" in matched["interpretation"]
    assert surface_refusal("No focal airspace opacity.")["surface_refusal_match"] is False


def test_shard_validation_rejects_truth_assignment():
    row = {
        "version": "clinical-presupposition-generation-only-v1",
        "item_id": "a",
        "image_id": "a",
        "prompt_condition": "neutral",
        "prompt": "p",
        "text": "answer",
        "generated_token_count": 1,
        "generated_token_ids": [1],
        "hit_max_new_tokens": False,
        "fingerprint": "fp",
        "claim_universe_sha256": "0" * 64,
        "clinical_claim_evaluation_status": "pending_shared_audit",
    }
    validate_shard(row, "a", "neutral", "fp")
    row["clinical_claim_evaluation_status"] = "correct"
    with pytest.raises(ValueError, match="must not assign clinical truth"):
        validate_shard(row, "a", "neutral", "fp")


def test_exact_generate_uses_model_sequence_not_decoded_text_retokenization():
    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 2

        def decode(self, ids, skip_special_tokens=True):
            assert ids == [17, 18, 2]
            return "decoded answer"

        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[999])

    class Model:
        def generate(self, input_ids, images, **kwargs):
            assert kwargs["return_dict_in_generate"] is True
            return SimpleNamespace(sequences=torch.tensor([[17, 18, 2]]))

    class Bot:
        device = "cpu"
        tokenizer = Tokenizer()
        model = Model()

        def input_moderation(self, text):
            return text

        def insert_image_placeholder(self, text, count):
            return "<image>\n" + text

        def get_conv_without_history(self, text):
            return [text]

        def preprocess(self, conversation, return_tensors):
            return torch.tensor([1, 2, 3])

        def get_image_tensors(self, images):
            return [torch.zeros(3, 2, 2)]

    result = exact_generate(
        Bot(), "prompt", object(), max_new_tokens=8, repetition_penalty=1.2
    )
    assert result["generated_token_ids"] == [17, 18, 2]
    assert result["generated_token_count"] == 3
    assert result["text"] == "decoded answer"
