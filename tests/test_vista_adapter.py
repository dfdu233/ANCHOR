from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from corrected_sgta.vista_adapter import VistaRuntimeAdapter, _parse_layer_window


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.model = SimpleNamespace(layers=[object(), object(), object()])
        self.lm_head = nn.Identity()
        self.calls = 0

    def get_model(self):
        return self.model

    def forward(self, value, **kwargs):
        self.calls += 1
        states = tuple(value + index for index in range(4))
        return SimpleNamespace(logits=value + 10, hidden_states=states)


def test_method_off_is_a_literal_noop():
    model = _DummyModel()
    original_forward = model.forward
    with VistaRuntimeAdapter(model=model, enabled=False):
        observed = model(torch.tensor([1.0])).logits
    assert torch.equal(observed, torch.tensor([11.0]))
    assert model.forward == original_forward
    assert model.calls == 1


def test_enabled_adapter_requires_at_least_one_module():
    model = _DummyModel()
    with pytest.raises(ValueError, match="VSV, SLA, or both"):
        with VistaRuntimeAdapter(
            model=model,
            enabled=True,
            enable_vsv=False,
            enable_sla=False,
        ):
            pass


def test_sla_wrapper_mixes_selected_decoder_logits_and_restores():
    model = _DummyModel()
    adapter = VistaRuntimeAdapter(
        model=model,
        enabled=False,
        logits_layers="1,2",
        logits_alpha=0.25,
    )
    adapter.enabled = True
    adapter._install_sla()
    try:
        assert "value" in inspect.signature(model.forward).parameters
        output = model(torch.tensor([1.0]))
        # decoder states exclude the embedding state: [2, 3, 4], so layers
        # 1..2 average to 3.5; final logits are 11.
        assert torch.allclose(output.logits, torch.tensor([9.125]))
    finally:
        adapter._restore()
    assert torch.equal(model(torch.tensor([1.0])).logits, torch.tensor([11.0]))


def test_sla_restores_after_generation_exception():
    model = _DummyModel()
    adapter = VistaRuntimeAdapter(model=model, enabled=False, logits_layers="0,1")
    adapter.enabled = True
    adapter._install_sla()
    original = adapter._original_forward
    with pytest.raises(RuntimeError, match="generation-only"):
        model(torch.tensor([1.0]), labels=torch.tensor([1]))
    adapter._restore()
    assert model.forward == original


def test_restore_removes_adapter_created_transient_model_state():
    model = _DummyModel()
    adapter = VistaRuntimeAdapter(model=model, enabled=False)
    model.damro_mask_idx = None
    adapter._damro_attr_existed = False
    adapter._damro_attr_touched = True
    adapter._restore()
    assert not hasattr(model, "damro_mask_idx")


@pytest.mark.parametrize("spec", ["1", "2,1", "-1,1", "0,3"])
def test_layer_window_fails_closed(spec):
    with pytest.raises(ValueError):
        _parse_layer_window(spec, 3)
