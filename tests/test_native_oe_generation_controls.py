from argparse import Namespace
from types import SimpleNamespace

import torch

from anchor.corrected_sgta.models_oe import _decode_generations
from anchor.medeval.run_native_oe_vqa import generation_contract


def _args(**updates):
    values = {"decode_mode": "greedy", "num_beams": 1, "temperature": 1.0, "top_p": 1.0}
    values.update(updates)
    return Namespace(**values)


def test_generation_contracts_are_explicit_and_fail_closed():
    assert generation_contract(_args()) == {
        "do_sample": False,
        "num_beams": 1,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    assert generation_contract(_args(decode_mode="beam", num_beams=4))["num_beams"] == 4
    assert generation_contract(_args(decode_mode="sample", temperature=0.7))["do_sample"]
    for bad in (
        _args(decode_mode="beam", num_beams=1),
        _args(decode_mode="sample", num_beams=2),
        _args(decode_mode="sample", temperature=0.0),
    ):
        try:
            generation_contract(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid generation contract was accepted")


def test_beam_decoding_uses_selected_sequence_transition_scores():
    class Tokenizer:
        eos_token_id = 9
        pad_token_id = 0

        @staticmethod
        def decode(ids, skip_special_tokens=True):
            return " ".join(map(str, ids))

    class Model:
        @staticmethod
        def compute_transition_scores(sequences, scores, beam_indices, normalize_logits):
            assert normalize_logits is True
            return torch.tensor([[-0.25, -0.75]])

    output = SimpleNamespace(
        sequences=torch.tensor([[101, 4, 5]]),
        scores=(torch.zeros(3, 10), torch.zeros(3, 10)),
        beam_indices=torch.tensor([[0, 1, 1]]),
    )
    generation = _decode_generations(Tokenizer(), output, Model())[0]
    assert generation.text == "4 5"
    assert generation.token_ids == (4, 5)
    assert generation.uncertainty == 0.5


def test_beam_decoding_skips_inputs_embeds_dummy_eos_prefix():
    class Tokenizer:
        eos_token_id = 9
        pad_token_id = 9

        @staticmethod
        def decode(ids, skip_special_tokens=True):
            return " ".join(map(str, ids))

    class Model:
        generation_config = SimpleNamespace(eos_token_id=9)

        @staticmethod
        def compute_transition_scores(sequences, scores, beam_indices, normalize_logits):
            return torch.tensor([[-0.5, -0.25, 0.0]])

    output = SimpleNamespace(
        sequences=torch.tensor([[9, 4, 9]]),
        scores=(torch.zeros(2, 10), torch.zeros(2, 10), torch.zeros(2, 10)),
        beam_indices=torch.tensor([[0, 0, 0]]),
    )
    generation = _decode_generations(Tokenizer(), output, Model())[0]
    assert generation.text == "4"
    assert generation.token_ids == (4,)
    assert generation.uncertainty == 0.5
