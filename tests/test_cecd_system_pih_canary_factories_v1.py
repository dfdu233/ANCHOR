from __future__ import annotations

import json
import inspect
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from anchor.corrected_sgta import cecd_system_pih_canary_factories_v1 as factories
from anchor.corrected_sgta.cecd_system_pih_runtime_integration_v1 import (
    ExpandedRoleTokenProvenance,
    RuntimeIntegrationError,
)


class FakeHuatuoModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prepared = None

    def prepare_inputs_labels_for_multimodal_new(
        self, input_ids, position_ids, attention, past, labels, images
    ):
        self.prepared = (input_ids, position_ids, attention, past, labels, images)
        embeddings = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8)
        expanded_attention = torch.ones((1, 5), dtype=torch.bool)
        expanded_positions = torch.arange(5).reshape(1, 5)
        return None, expanded_positions, expanded_attention, None, embeddings, None


class FakeHuluModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(image_token_index=999)
        self.prepared = None

    def prepare_inputs_labels_for_multimodal(self, **kwargs):
        self.prepared = kwargs
        embeddings = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
        attention = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        positions = torch.arange(4).reshape(1, 4)
        return None, attention, positions, None, embeddings, None


class FakeHuatuoAdapter:
    def __init__(self) -> None:
        self.model = FakeHuatuoModel()
        self.image_seen = None
        self.prompt_seen = None

    def _inputs(self, image, prompt):
        self.image_seen = image
        self.prompt_seen = prompt
        return torch.tensor([[10, -200, 11]]), torch.ones((1, 3, 2, 2))


class FakeHuluAdapter:
    def __init__(self) -> None:
        self.model = FakeHuluModel()
        self.image_seen = None
        self.prompt_seen = None

    def _inputs(self, image, prompt):
        self.image_seen = image
        self.prompt_seen = prompt
        return {
            "input_ids": torch.tensor([[10, 999, 999, 11]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 2, 2)),
            "grid_sizes": torch.tensor([[1, 1, 2]]),
            "merge_sizes": torch.tensor([1]),
            "modals": ["image"],
        }


def _assert_bundle(bundle, family: str, expected_roles: tuple[str, ...]) -> None:
    assert set(bundle) == {"provenance", "forward_kwargs", "input_identity"}
    provenance = bundle["provenance"]
    assert isinstance(provenance, ExpandedRoleTokenProvenance)
    assert provenance.model_family == family
    assert provenance.role_provenance == expected_roles
    assert provenance.frozen_prefix_length == len(expected_roles)
    kwargs = bundle["forward_kwargs"]
    assert kwargs["input_ids"] is None
    assert kwargs["use_cache"] is False
    assert kwargs["return_dict"] is True
    assert tuple(kwargs["inputs_embeds"].shape[:2]) == (1, len(expected_roles))
    assert tuple(kwargs["attention_mask"].shape) == (1, len(expected_roles))
    assert "images" not in kwargs
    assert "pixel_values" not in kwargs
    identity = bundle["input_identity"]
    assert identity["model_family"] == family
    assert identity["cache_free"] is True
    assert identity["provenance_fingerprint"] == provenance.fingerprint


def test_huatuo_factories_reuse_native_adapter_and_expand_exact_provenance(
    monkeypatch,
) -> None:
    adapter = FakeHuatuoAdapter()
    monkeypatch.setattr(factories, "_load_huatuo_adapter", lambda: adapter)
    monkeypatch.setattr(factories, "_load_frozen_image", lambda: "frozen-image")
    model = factories.huatuo_model_factory()
    bundle = factories.huatuo_input_factory(model)
    _assert_bundle(
        bundle,
        "huatuo",
        ("user_text", "image", "image", "image", "user_text"),
    )
    assert adapter.image_seen == "frozen-image"
    assert adapter.prompt_seen == factories.FROZEN_PROMPT
    assert model.training is False
    raw_ids = model.prepared[0]
    assert len(raw_ids) == 1 and raw_ids[0].tolist() == [10, -200, 11]
    assert bundle["provenance"].spans.image == (1, 2, 3)


def test_hulu_factories_reuse_native_processor_and_preserve_exact_token_run(
    monkeypatch,
) -> None:
    adapter = FakeHuluAdapter()
    monkeypatch.setattr(factories, "_load_hulu_adapter", lambda: adapter)
    monkeypatch.setattr(factories, "_load_frozen_image", lambda: "frozen-image")
    model = factories.hulu_model_factory()
    bundle = factories.hulu_input_factory(model)
    _assert_bundle(
        bundle,
        "hulu",
        ("user_text", "image", "image", "user_text"),
    )
    assert adapter.image_seen == "frozen-image"
    assert adapter.prompt_seen == factories.FROZEN_PROMPT
    assert model.prepared["input_ids"].tolist() == [[10, 999, 999, 11]]
    assert bundle["provenance"].spans.image == (1, 2)


def test_factory_description_never_loads_an_adapter_or_touches_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        factories,
        "_load_huatuo_adapter",
        lambda: (_ for _ in ()).throw(AssertionError("model load forbidden")),
    )
    monkeypatch.setattr(
        factories,
        "_load_hulu_adapter",
        lambda: (_ for _ in ()).throw(AssertionError("model load forbidden")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA inspection forbidden")),
    )
    monkeypatch.setattr(
        factories,
        "file_record",
        lambda path: {
            "path": str(path),
            "sha256": factories.FROZEN_IMAGE_SHA256,
            "bytes": 1,
        },
    )
    description = factories.factory_description()
    assert description["status"] == "source_ready_explicit_canary_pending"
    assert description["model_loaded"] is False
    assert description["gpu_touched"] is False
    assert description["canary_run"] is False
    assert description["frozen_input"]["image"]["sha256"] == factories.FROZEN_IMAGE_SHA256


def test_public_factories_are_bound_to_this_source_not_torch_decorator() -> None:
    expected = Path(factories.__file__).resolve()
    for value in (
        factories.huatuo_model_factory,
        factories.huatuo_input_factory,
        factories.hulu_model_factory,
        factories.hulu_input_factory,
    ):
        assert Path(inspect.getsourcefile(value)).resolve() == expected


def test_input_factory_rejects_unbound_model() -> None:
    with pytest.raises(RuntimeIntegrationError, match="not bound"):
        factories.huatuo_input_factory(FakeHuatuoModel())
    with pytest.raises(RuntimeIntegrationError, match="not bound"):
        factories.hulu_input_factory(FakeHuluModel())


def test_hulu_rejects_noncontiguous_image_tokens(monkeypatch) -> None:
    adapter = FakeHuluAdapter()
    adapter._inputs = lambda image, prompt: {  # type: ignore[method-assign]
        "input_ids": torch.tensor([[999, 10, 999, 11]])
    }
    monkeypatch.setattr(factories, "_load_hulu_adapter", lambda: adapter)
    monkeypatch.setattr(factories, "_load_frozen_image", lambda: "frozen-image")
    model = factories.hulu_model_factory()
    with pytest.raises(RuntimeIntegrationError, match="not contiguous"):
        factories.hulu_input_factory(model)


def test_hulu_rejects_length_changing_visual_compression(monkeypatch) -> None:
    adapter = FakeHuluAdapter()
    monkeypatch.setattr(factories, "_load_hulu_adapter", lambda: adapter)
    monkeypatch.setattr(factories, "_load_frozen_image", lambda: "frozen-image")
    model = factories.hulu_model_factory()

    def compressed(**kwargs):
        del kwargs
        return None, None, None, None, torch.ones((1, 3, 8)), None

    model.prepare_inputs_labels_for_multimodal = compressed
    with pytest.raises(RuntimeIntegrationError, match="compression changed"):
        factories.hulu_input_factory(model)


def test_serial_handoff_is_explicit_locked_and_ordered() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/run_cecd_system_pih_native_eager_canaries_v1.sh").read_text()
    gate = script.index('CECD_EXPLICIT_CANARY_LAUNCH:-')
    lock = script.index('flock 8')
    huatuo = script.index('--family huatuo')
    hulu = script.index('--family hulu')
    assert gate < lock < huatuo < hulu
    assert "gpu0-vindr-v2.lock" in script
    assert "run-native-eager-canaries-v1" in script
    assert "huatuo_python=/opt/miniconda3/envs/huatuo/bin/python" in script
    assert "hulu_python=/home/dbw/.venvs/hulumed/bin/python" in script
    assert "env PYTHONPATH=/home/dbw/ANCHOR:/home/dbw/HuatuoGPT-Vision" in script
    assert "env PYTHONPATH=/home/dbw/ANCHOR" in script
    assert "export PYTHONPATH=" not in script

    handoff = json.loads(
        (root / "configs/cecd_system_pih_native_eager_canary_handoff_v1.json").read_text()
    )
    assert handoff["authorization"]["auto_launch"] is False
    assert handoff["authorization"]["gpu_touched_during_handoff"] is False
    assert handoff["scheduling"]["execution_order"] == ["huatuo", "hulu"]
    assert handoff["postconditions"]["preflight_must_remain_false_before_canary_audit"]


def test_query_chunked_v2_handoff_preserves_failed_historical_binding() -> None:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts/run_cecd_system_pih_native_eager_canaries_v2.sh"
    script = script_path.read_text()
    gate = script.index("run-native-eager-canaries-v2-query-chunked")
    lock = script.index("flock 8")
    huatuo = script.index("--family huatuo")
    hulu = script.index("--family hulu")
    assert gate < lock < huatuo < hulu
    assert "huatuo_native_eager_canary_v2_query_chunked.json" in script
    assert "hulu_native_eager_canary_v2_query_chunked.json" in script
    assert "expandable_segments:True" in script

    handoff = json.loads(
        (root / "configs/cecd_system_pih_native_eager_canary_handoff_v2.json").read_text()
    )
    disposition = json.loads(
        (
            root
            / "configs/cecd_system_pih_native_eager_canary_v2_disposition_20260803.json"
        ).read_text()
    )
    assert handoff["status"] == "source_ready_explicit_serial_launch_pending"
    assert disposition["post_lock_bindings_verified"] is True
    assert disposition["huatuo_artifact"]["max_absolute_error"] == 0.125
    assert disposition["huatuo_artifact"]["tolerance_passed"] is False
    assert disposition["scientific_authority"] is False
    assert disposition["frozen_handoff_sha256"] == hashlib.sha256(
        (root / "configs/cecd_system_pih_native_eager_canary_handoff_v2.json").read_bytes()
    ).hexdigest()
    assert handoff["authorization"]["auto_launch"] is False
    assert handoff["authorization"]["canary_only"] is True
    assert handoff["scheduling"]["execution_order"] == ["huatuo", "hulu"]
    assert handoff["memory_correction"]["changed_reduction_axis"] is False
    assert handoff["memory_correction"]["scientific_tolerance_unchanged"] is True
    # v2 is immutable historical evidence. Its source hashes intentionally no
    # longer match the v3 family-specific runtime.
    assert handoff["source_bindings"]["runtime"]["sha256"] == (
        "8c32e51e702d2cce9cc9b3830827961a99e74fa799ec158462e560b0c9bed79f"
    )


def test_family_specific_v3_handoff_is_hash_bound_and_non_overwriting() -> None:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts/run_cecd_system_pih_native_eager_canaries_v3.sh"
    script = script_path.read_text()
    gate = script.index("run-native-eager-canaries-v3-family-specific-query-shape")
    pre = script.index("--phase pre_lock")
    lock = script.index("flock 8")
    post = script.index("--phase post_lock")
    huatuo = script.index("--family huatuo")
    hulu = script.index("--family hulu")
    assert gate < pre < lock < post < huatuo < hulu
    assert "huatuo_native_eager_canary_v3_full_query.json" in script
    assert "hulu_native_eager_canary_v3_query_chunked.json" in script

    handoff = json.loads(
        (root / "configs/cecd_system_pih_native_eager_canary_handoff_v3.json").read_text()
    )
    disposition = json.loads(
        (
            root
            / "configs/cecd_system_pih_native_eager_canary_v3_disposition_20260803.json"
        ).read_text()
    )
    assert handoff["status"] == "source_ready_explicit_serial_launch_pending"
    assert handoff["authorization"]["auto_launch"] is False
    assert handoff["authorization"]["canary_only"] is True
    assert handoff["memory_correction"]["huatuo_query_shape"] == "full_query"
    assert handoff["memory_correction"]["hulu_query_chunk_size"] == 256
    assert disposition["post_lock_bindings_verified"] is True
    assert disposition["huatuo_artifact"]["tolerance_passed"] is True
    assert disposition["hulu_artifact"]["argmax_equal"] is True
    assert disposition["hulu_artifact"]["max_absolute_error"] == 3.0
    assert disposition["hulu_artifact"]["tolerance_passed"] is False
    assert disposition["scientific_authority"] is False
    # The handoff is immutable historical input. Result facts are append-only
    # in the disposition rather than written back into that frozen contract.
    assert disposition["frozen_handoff_sha256"] == hashlib.sha256(
        (root / "configs/cecd_system_pih_native_eager_canary_handoff_v3.json").read_bytes()
    ).hexdigest()
