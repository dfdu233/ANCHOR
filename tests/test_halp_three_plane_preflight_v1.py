from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import anchor.corrected_sgta.halp_three_plane_preflight_v1 as halp


class FakeBlock(torch.nn.Module):
    def __init__(self, increment: float):
        super().__init__()
        self.increment = increment
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        return (hidden_states + self.increment * self.scale,)


class FakeDecoder(torch.nn.Module):
    def __init__(self, layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [FakeBlock(float(index + 1)) for index in range(layers)]
        )
        self.norm = torch.nn.Identity()

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids,
        inputs_embeds,
        use_cache,
        output_hidden_states,
        return_dict,
    ):
        assert input_ids is None
        assert use_cache is False
        assert output_hidden_states is False
        assert return_dict is True
        hidden = inputs_embeds
        for block in self.layers:
            hidden = block(hidden)[0]
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


def _fake_audit(layers: int) -> dict:
    audit = {
        "model_family": "fake",
        "decoder_layer_count": layers,
        "fingerprint": "a" * 64,
    }
    return audit


def test_cpu_source_audit_closes_two_real_architectures_without_cuda() -> None:
    expected = {
        "huatuo": ("llava_qwen2", 28, 3584),
        "hulu": ("hulumed_qwen3", 36, 2560),
    }
    for family, (model_type, layers, width) in expected.items():
        audit = halp.cpu_source_audit(
            family=family, model_dir=halp.MODEL_DIRS[family]
        )
        assert audit["status"] == "cpu_source_audit_passed_no_model_or_cuda"
        assert audit["model_type"] == model_type
        assert audit["decoder_layer_count"] == layers
        assert audit["hidden_size"] == width
        assert len(audit["layers"]) == layers
        assert audit["layers"][0]["layer_number"] == 1
        assert audit["layers"][-1]["normalized_depth"] == 1.0
        assert audit["model_loaded"] is False
        assert audit["cuda_touched"] is False
        assert audit["official_halp_code_reproduction_claimed"] is False
        assert audit["cross_architecture_latent_semantics_identical_claimed"] is False
        assert audit["outcome_read"] is False
        assert audit["probe_training_authorized"] is False
        halp.validate_selection_policy(audit["selection_policy"])


def test_selection_policy_is_dev_group_cv_and_confirmation_apply_only() -> None:
    policy = halp.selection_policy()
    halp.validate_selection_policy(policy)
    assert policy["selection_split"] == "dev_only"
    assert policy["group_unit"] == "global_image_id"
    assert policy["confirmation_mode"] == "apply_only"
    tampered = dict(policy)
    tampered["confirmation_layer_selection"] = True
    tampered["fingerprint"] = halp.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "fingerprint"}
    )
    with pytest.raises(halp.HALPCompatibilityError, match="policy drift"):
        halp.validate_selection_policy(tampered)


def test_fake_hook_captures_exact_three_planes_and_all_post_block_layers() -> None:
    decoder = FakeDecoder(3)
    embeddings = torch.arange(1 * 6 * 4, dtype=torch.float32).reshape(1, 6, 4)
    visual_output = torch.arange(1 * 5 * 3, dtype=torch.float32).reshape(1, 5, 3)
    attention = torch.ones((1, 6), dtype=torch.bool)
    result = halp.capture_three_planes(
        decoder_model=decoder,
        embeddings=embeddings,
        attention_mask=attention,
        position_ids=None,
        visual_span=(1, 4),
        visual_only_output=visual_output,
        layers=(1, 2, 3),
    )
    np.testing.assert_allclose(result["visual_only"], visual_output[0].mean(0).numpy())
    cumulative = 0.0
    for layer in (1, 2, 3):
        cumulative += float(layer)
        np.testing.assert_allclose(
            result["decoder_vision_token"][layer], embeddings[0, 3].numpy() + cumulative
        )
        np.testing.assert_allclose(
            result["query_token"][layer], embeddings[0, 5].numpy() + cumulative
        )
    assert result["capture_audit"]["vision_token_index"] == 3
    assert result["capture_audit"]["query_token_index"] == 5
    assert result["capture_audit"]["one_forward_no_generation"] is True
    assert all(len(block._forward_hooks) == 0 for block in decoder.layers)


def test_fake_runtime_layer_hash_contract_is_ordered_and_drift_sensitive() -> None:
    decoder = FakeDecoder(2)
    first = halp.runtime_layer_contract(decoder, _fake_audit(2))
    second = halp.runtime_layer_contract(decoder, _fake_audit(2))
    assert first == second
    assert first["layer_count"] == 2
    assert [row["layer_number"] for row in first["layers"]] == [1, 2]
    with pytest.raises(halp.HALPCompatibilityError, match="layer count"):
        halp.runtime_layer_contract(decoder, _fake_audit(3))


def test_preprojector_hook_is_exact_once_and_nonmutating() -> None:
    module = torch.nn.Identity()
    value = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    with halp.PreProjectorVisionCapture(module) as capture:
        observed = module(value)
    assert torch.equal(observed, value)
    assert torch.equal(capture.one(), value)
    assert not module._forward_hooks


def test_capture_fails_closed_on_ambiguous_positions_or_layer_semantics() -> None:
    decoder = FakeDecoder(2)
    embeddings = torch.zeros((1, 5, 4))
    vision = torch.zeros((1, 3, 2))
    with pytest.raises(halp.HALPCompatibilityError, match="include the final"):
        halp.capture_three_planes(
            decoder_model=decoder,
            embeddings=embeddings,
            attention_mask=None,
            position_ids=None,
            visual_span=(1, 3),
            visual_only_output=vision,
            layers=(1,),
        )
    with pytest.raises(halp.HALPCompatibilityError, match="ambiguous"):
        halp.capture_three_planes(
            decoder_model=decoder,
            embeddings=embeddings,
            attention_mask=torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool),
            position_ids=None,
            visual_span=(1, 3),
            visual_only_output=vision,
            layers=(1, 2),
        )


def test_written_capture_cannot_masquerade_as_probe_or_halp_reproduction(
    tmp_path: Path,
) -> None:
    decoder = FakeDecoder(2)
    capture = halp.capture_three_planes(
        decoder_model=decoder,
        embeddings=torch.zeros((1, 5, 4)),
        attention_mask=None,
        position_ids=None,
        visual_span=(1, 3),
        visual_only_output=torch.ones((1, 3, 2)),
        layers=(1, 2),
    )
    audit = _fake_audit(2)
    contract = halp.runtime_layer_contract(decoder, audit)
    metadata = halp.write_engineering_capture(
        output_dir=tmp_path / "capture",
        family="fake",
        audit=audit,
        layer_contract=contract,
        capture=capture,
        input_identity={"record_key": "fake-no-outcome"},
    )
    assert metadata["probe_trained"] is False
    assert metadata["outcome_read"] is False
    assert metadata["dev_selection_performed"] is False
    assert metadata["official_halp_code_reproduction_claimed"] is False
    arrays = np.load(tmp_path / "capture/representations.npz", allow_pickle=False)
    assert set(arrays.files) == {
        "visual_only",
        "decoder_vision_token__layer_001",
        "decoder_vision_token__layer_002",
        "query_token__layer_001",
        "query_token__layer_002",
    }
    stored = json.loads((tmp_path / "capture/metadata.json").read_text())
    assert stored == metadata
