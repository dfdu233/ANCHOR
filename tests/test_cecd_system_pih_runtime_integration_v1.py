from __future__ import annotations

from types import SimpleNamespace
import json
import os
from pathlib import Path
import subprocess
import textwrap

import pytest
import torch

from anchor.corrected_sgta import cecd_system_pih_runtime_integration_v1 as runtime
from anchor.corrected_sgta.validate_cecd_system_pih_control_preflight_v1 import (
    validate_plan,
)


def apply_rotary_pos_emb(query, key, cos, sin, position_ids=None):
    del cos, sin, position_ids
    return query, key


class FakeProjection(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        repeats = (self.out_features + tensor.shape[-1] - 1) // tensor.shape[-1]
        return tensor.repeat_interleave(repeats, dim=-1)[..., : self.out_features]


class FakeOProj(FakeProjection):
    pass


class FakeRotary(torch.nn.Module):
    def forward(self, value: torch.Tensor, seq_len: int | None = None):
        length = int(seq_len or value.shape[-2])
        shape = (1, 1, length, value.shape[-1])
        return torch.ones(shape, dtype=value.dtype), torch.zeros(shape, dtype=value.dtype)


class FakeQwen2Attention(torch.nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        geometry = runtime.MODEL_GEOMETRIES["huatuo"]
        width = geometry.num_query_heads * geometry.head_dim
        self.layer_idx = layer_idx
        self.num_heads = geometry.num_query_heads
        self.num_key_value_heads = geometry.num_key_value_heads
        self.head_dim = geometry.head_dim
        self.attention_dropout = 0.0
        self.q_proj = FakeProjection(width, width)
        self.k_proj = FakeProjection(width, geometry.num_key_value_heads * geometry.head_dim)
        self.v_proj = FakeProjection(width, geometry.num_key_value_heads * geometry.head_dim)
        self.o_proj = FakeOProj(width, width)
        self.rotary_emb = FakeRotary()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        return runtime._qwen2_forward(
            self,
            original_forward=self.forward,
            geometry=runtime.MODEL_GEOMETRIES["huatuo"],
            session=None,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )


class IdentityNorm(torch.nn.Module):
    def forward(self, tensor):
        return tensor


class FakeQwen3Attention(torch.nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        geometry = runtime.MODEL_GEOMETRIES["hulu"]
        hidden = 2560
        width = geometry.num_query_heads * geometry.head_dim
        self.layer_idx = layer_idx
        self.num_key_value_groups = geometry.kv_groups
        self.scaling = geometry.head_dim**-0.5
        self.attention_dropout = 0.0
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.q_proj = FakeProjection(hidden, width)
        self.k_proj = FakeProjection(hidden, geometry.num_key_value_heads * geometry.head_dim)
        self.v_proj = FakeProjection(hidden, geometry.num_key_value_heads * geometry.head_dim)
        self.o_proj = FakeOProj(width, hidden)
        self.q_norm = IdentityNorm()
        self.k_norm = IdentityNorm()

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        return runtime._qwen3_forward(
            self,
            original_forward=self.forward,
            geometry=runtime.MODEL_GEOMETRIES["hulu"],
            session=None,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )


class FakeLayer(torch.nn.Module):
    def __init__(self, attention: torch.nn.Module):
        super().__init__()
        self.self_attn = attention


class FakeDecoder(torch.nn.Module):
    def __init__(self, family: str):
        super().__init__()
        cls = FakeQwen2Attention if family == "huatuo" else FakeQwen3Attention
        self.layers = torch.nn.ModuleList(
            [FakeLayer(cls(index)) for index in range(runtime.EXPECTED_LAYERS[family])]
        )


class FakeCausalModel(torch.nn.Module):
    def __init__(self, family: str):
        super().__init__()
        self.family = family
        self.model = FakeDecoder(family)

    def forward(self, inputs_embeds, output_attentions=False, use_cache=False):
        del use_cache
        hidden = inputs_embeds
        all_weights = []
        for layer in self.model.layers:
            if self.family == "huatuo":
                hidden, weights, _ = layer.self_attn(
                    hidden_states=hidden,
                    output_attentions=output_attentions,
                    use_cache=False,
                )
            else:
                length = hidden.shape[1]
                dim = runtime.MODEL_GEOMETRIES["hulu"].head_dim
                position = (
                    torch.ones(1, length, dim, dtype=hidden.dtype),
                    torch.zeros(1, length, dim, dtype=hidden.dtype),
                )
                hidden, weights = layer.self_attn(
                    hidden_states=hidden,
                    position_embeddings=position,
                    attention_mask=None,
                    output_attentions=output_attentions,
                )
            all_weights.append(weights)
        return SimpleNamespace(logits=hidden[..., :17], attentions=all_weights)


def _provenance(family: str, visual: int = 3):
    if family == "huatuo":
        return runtime.build_huatuo_expanded_provenance(
            input_ids=(10, 11, -200, 12, 13),
            token_roles=("user_text", "user_text", "image", "user_text", "user_text"),
            image_placeholder_id=-200,
            projected_visual_token_count=visual,
        )
    return runtime.build_hulu_expanded_provenance(
        expanded_input_ids=(10, 11, *([151669] * visual), 12, 13),
        token_roles=("user_text", "user_text", *(("image",) * visual), "user_text", "user_text"),
        image_token_id=151669,
    )


def test_huatuo_and_hulu_expanded_provenance_is_exact_and_dynamic() -> None:
    huatuo = _provenance("huatuo", 7)
    hulu = _provenance("hulu", 5)
    assert huatuo.spans.image == tuple(range(2, 9))
    assert hulu.spans.image == tuple(range(2, 7))
    assert huatuo.spans.prefix_before_image_is_true_system is False
    assert len(huatuo.token_origins) == len(huatuo.role_provenance)
    assert huatuo.as_dict()["fingerprint"] == huatuo.fingerprint
    with pytest.raises(runtime.RuntimeIntegrationError, match="exactly one"):
        runtime.build_huatuo_expanded_provenance(
            input_ids=(1, -200, -200),
            token_roles=("user_text", "image", "image"),
            image_placeholder_id=-200,
            projected_visual_token_count=2,
        )
    with pytest.raises(runtime.RuntimeIntegrationError, match="disagree"):
        runtime.build_hulu_expanded_provenance(
            expanded_input_ids=(1, 151669, 2),
            token_roles=("user_text", "user_text", "user_text"),
            image_token_id=151669,
        )


@pytest.mark.parametrize("family,layer_index", [("huatuo", 21), ("hulu", 27)])
def test_clean_room_eager_patch_changes_only_last_frozen_query_attention_row(
    family: str, layer_index: int
) -> None:
    model = FakeCausalModel(family).eval()
    provenance = _provenance(family)
    hidden_width = 3584 if family == "huatuo" else 2560
    hidden = torch.linspace(
        -0.5, 0.5, steps=provenance.frozen_prefix_length * hidden_width
    ).reshape(1, provenance.frozen_prefix_length, hidden_width)
    original_forward = model.model.layers[layer_index].self_attn.forward
    with runtime.EagerAttentionPatchContext(
        model,
        family=family,
        provenance=provenance,
        system_layers=(),
        rotary_apply=apply_rotary_pos_emb,
    ):
        identity = model(inputs_embeds=hidden, output_attentions=True).attentions[layer_index]
    with runtime.EagerAttentionPatchContext(
        model,
        family=family,
        provenance=provenance,
        system_layers=(layer_index,),
        rotary_apply=apply_rotary_pos_emb,
    ) as context:
        patched = model(inputs_embeds=hidden, output_attentions=True).attentions[layer_index]
        assert len(context.sessions[layer_index].diagnostics) == 1
    assert model.model.layers[layer_index].self_attn.forward == original_forward
    last = provenance.frozen_prefix_length - 1
    source = list(provenance.spans.prefix_before_image)
    assert torch.equal(identity[..., :last, :], patched[..., :last, :])
    assert torch.count_nonzero(patched[..., last, source]) == 0
    assert torch.allclose(patched[..., last, :].sum(-1), torch.ones_like(patched[..., last, :].sum(-1)), atol=1e-6)
    if family == "hulu":
        assert model.model.layers[layer_index].self_attn.config._attn_implementation == "sdpa"


def test_system_session_prefill_then_decode_never_patches_generated_query() -> None:
    provenance = _provenance("huatuo")
    patch = runtime.PositionalPrefixAttentionPatch(provenance.spans, alpha=0.0)
    session = runtime.LastFrozenPrefixAttentionSession(patch)
    length = provenance.frozen_prefix_length
    prefill = torch.softmax(torch.randn(1, 28, length, length), dim=-1)
    session.apply(prefill)
    decode = torch.softmax(torch.randn(1, 28, 1, length + 1), dim=-1)
    assert torch.equal(session.apply(decode), decode)
    with pytest.raises(runtime.RuntimeIntegrationError, match="two full prefills"):
        session.apply(prefill)


def test_query_chunked_eager_matches_full_rows_and_patches_global_prefix_row() -> None:
    provenance = _provenance("hulu")
    geometry = runtime.MODEL_GEOMETRIES["hulu"]
    length = provenance.frozen_prefix_length
    generator = torch.Generator(device="cpu").manual_seed(9127)
    query = torch.randn(
        1, geometry.num_query_heads, length, geometry.head_dim, generator=generator
    )
    key = torch.randn(
        1, geometry.num_key_value_heads, length, geometry.head_dim, generator=generator
    )
    value = torch.randn(
        1, geometry.num_key_value_heads, length, geometry.head_dim, generator=generator
    )
    mask = torch.full((1, 1, length, length), torch.finfo(torch.float32).min)
    mask = torch.triu(mask, diagonal=1)

    full_session = runtime.LastFrozenPrefixAttentionSession(
        runtime.PositionalPrefixAttentionPatch(provenance.spans, alpha=0.0)
    )
    chunked_session = runtime.LastFrozenPrefixAttentionSession(
        runtime.PositionalPrefixAttentionPatch(provenance.spans, alpha=0.0)
    )
    full_output, full_weights = runtime._eager_attention_core(
        query=query,
        key=key,
        value=value,
        geometry=geometry,
        scaling=geometry.head_dim**-0.5,
        attention_mask=mask,
        training=False,
        dropout=0.0,
        session=full_session,
        return_weights=True,
        query_chunk_size=length,
    )
    chunked_output, chunked_weights = runtime._eager_attention_core(
        query=query,
        key=key,
        value=value,
        geometry=geometry,
        scaling=geometry.head_dim**-0.5,
        attention_mask=mask,
        training=False,
        dropout=0.0,
        session=chunked_session,
        return_weights=True,
        query_chunk_size=2,
    )
    assert torch.allclose(chunked_output, full_output, atol=1e-6, rtol=1e-6)
    assert torch.allclose(chunked_weights, full_weights, atol=1e-6, rtol=0.0)
    assert len(chunked_session.diagnostics) == 1
    assert chunked_session.prefill_seen is True

    no_weights_output, no_weights = runtime._eager_attention_core(
        query=query,
        key=key,
        value=value,
        geometry=geometry,
        scaling=geometry.head_dim**-0.5,
        attention_mask=mask,
        training=False,
        dropout=0.0,
        session=None,
        return_weights=False,
        query_chunk_size=2,
    )
    assert no_weights is None
    assert no_weights_output.shape == full_output.shape


@pytest.mark.parametrize("family", ["huatuo", "hulu"])
def test_qwen_forward_contract_gqa_mask_and_rotary_boundary(family: str) -> None:
    model = FakeCausalModel(family).eval()
    provenance = _provenance(family)
    attention = model.model.layers[0].self_attn
    width = 3584 if family == "huatuo" else 2560
    hidden = torch.randn(1, provenance.frozen_prefix_length, width)
    query_length = hidden.shape[1]
    mask = torch.zeros(1, 1, query_length, query_length)
    mask[..., -1] = torch.finfo(torch.float32).min
    calls = {"count": 0}

    def counted_rotary(query, key, cos, sin, position_ids=None):
        del cos, sin, position_ids
        calls["count"] += 1
        return query, key

    with runtime.EagerAttentionPatchContext(
        model,
        family=family,
        provenance=provenance,
        rotary_apply=counted_rotary,
    ):
        if family == "huatuo":
            result = attention(
                hidden_states=hidden,
                attention_mask=mask,
                output_attentions=True,
                use_cache=False,
            )
            assert isinstance(result, tuple) and len(result) == 3
            output, weights, cache = result
            assert cache is None
        else:
            head_dim = runtime.MODEL_GEOMETRIES[family].head_dim
            result = attention(
                hidden_states=hidden,
                position_embeddings=(
                    torch.ones(1, query_length, head_dim),
                    torch.zeros(1, query_length, head_dim),
                ),
                attention_mask=mask,
                past_key_value=None,
                output_attentions=True,
            )
            assert isinstance(result, tuple) and len(result) == 2
            output, weights = result
        geometry = runtime.MODEL_GEOMETRIES[family]
        assert output.shape == (1, query_length, width)
        assert weights.shape == (
            1,
            geometry.num_query_heads,
            query_length,
            query_length,
        )
        assert torch.count_nonzero(weights[..., -1]) == 0
        assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6)
    assert calls["count"] == 1


def test_pih_o_proj_hook_uses_width128_and_current_sample_prefill_decode_cache() -> None:
    model = FakeCausalModel("hulu").eval()
    provenance = _provenance("hulu")
    layer = model.model.layers[3].self_attn
    baseline_hooks = len(layer.o_proj._forward_pre_hooks)
    with runtime.PIHPreOProjPatchContext(
        model,
        family="hulu",
        provenance=provenance,
        selected_heads_by_layer={3: (0, 31)},
    ):
        prefix = provenance.frozen_prefix_length
        tensor = torch.arange(prefix * 4096, dtype=torch.float32).reshape(1, prefix, 4096)
        observed = layer.o_proj(tensor)
        expected_first = tensor[:, :, :128].mean(dim=1, keepdim=True).expand(1, prefix, 128)
        assert torch.equal(observed[..., :128], expected_first)
        decode = torch.full((1, 1, 4096), 999.0)
        decoded = layer.o_proj(decode)
        assert torch.equal(decoded[..., :128], expected_first[:, :1, :])
        assert runtime.derive_head_width(layer.o_proj.in_features, 32) == 128
    assert len(layer.o_proj._forward_pre_hooks) == baseline_hooks


def test_combined_context_restores_forward_and_hooks_after_exception() -> None:
    model = FakeCausalModel("huatuo").eval()
    provenance = _provenance("huatuo")
    attention = model.model.layers[0].self_attn
    original = attention.forward
    hooks = len(attention.o_proj._forward_pre_hooks)
    with pytest.raises(RuntimeError, match="sentinel"):
        with runtime.SystemPIHRuntimeContext(
            model,
            family="huatuo",
            provenance=provenance,
            system_layers=(21,),
            selected_heads_by_layer={0: (0,)},
            rotary_apply=apply_rotary_pos_emb,
        ):
            raise RuntimeError("sentinel")
    assert attention.forward == original
    assert len(attention.o_proj._forward_pre_hooks) == hooks
    with pytest.raises(runtime.RuntimeIntegrationError, match="query-head selection"):
        with runtime.SystemPIHRuntimeContext(
            model,
            family="huatuo",
            provenance=provenance,
            selected_heads_by_layer={0: (999,)},
            rotary_apply=apply_rotary_pos_emb,
        ):
            pass
    assert attention.forward == original
    assert len(attention.o_proj._forward_pre_hooks) == hooks


@pytest.mark.parametrize("family", ["huatuo", "hulu"])
def test_fake_native_vs_eager_canary_passes_without_intervention(family: str) -> None:
    model = FakeCausalModel(family).eval()
    provenance = _provenance(family)
    width = 3584 if family == "huatuo" else 2560
    inputs = torch.linspace(-0.2, 0.2, steps=provenance.frozen_prefix_length * width).reshape(
        1, provenance.frozen_prefix_length, width
    )
    result = runtime.run_model_native_vs_eager_canary(
        model=model,
        family=family,
        provenance=provenance,
        forward_kwargs={"inputs_embeds": inputs},
        rotary_apply=apply_rotary_pos_emb,
    )
    assert result.passed is True
    assert result.argmax_equal is True
    with pytest.raises(runtime.RuntimeIntegrationError, match="cache-free"):
        runtime.run_model_native_vs_eager_canary(
            model=model,
            family=family,
            provenance=provenance,
            forward_kwargs={"inputs_embeds": inputs, "past_key_values": object()},
            rotary_apply=apply_rotary_pos_emb,
        )


def test_runtime_description_is_source_only_and_never_claims_execution() -> None:
    for family in ("huatuo", "hulu"):
        result = runtime.runtime_description(family)
        assert result["model_loaded"] is False
        assert result["gpu_touched"] is False
        assert result["canary_run"] is False
        assert result["selected_heads_loaded"] is False
        assert result["official_code_copied"] is False
        assert result["head_width"] == 128
    assert runtime.runtime_description("hulu")["backend_transition"] == "native_sdpa_to_clean_room_eager"


@pytest.mark.parametrize(
    "interpreter",
    (
        "/opt/miniconda3/envs/huatuo/bin/python",
        "/home/dbw/.venvs/hulumed/bin/python",
    ),
)
def test_dash_m_cli_and_canonical_factory_share_exact_provenance_class(
    tmp_path: Path, interpreter: str
) -> None:
    """Regress the real ``python -m`` class-identity failure without a model."""

    factory = tmp_path / "cecd_fake_dash_m_factory.py"
    factory.write_text(
        textwrap.dedent(
            """
            class FakeCanaryResult:
                passed = True

                def as_dict(self):
                    return {"passed": True, "fixture": "dash-m-module-identity"}


            def build_model():
                from anchor.corrected_sgta import cecd_system_pih_runtime_integration_v1 as runtime
                runtime.run_model_native_vs_eager_canary = lambda **kwargs: FakeCanaryResult()
                return object()


            def build_input(model):
                del model
                from anchor.corrected_sgta import cecd_system_pih_runtime_integration_v1 as runtime
                provenance = runtime.build_huatuo_expanded_provenance(
                    input_ids=(10, -200, 11),
                    token_roles=("user_text", "image", "user_text"),
                    image_placeholder_id=-200,
                    projected_visual_token_count=2,
                )
                return {
                    "provenance": provenance,
                    "forward_kwargs": {},
                    "input_identity": {"fixture": "dash-m-module-identity"},
                }
            """
        ),
        encoding="utf-8",
    )
    output = tmp_path / (Path(interpreter).parent.parent.name + ".canary.json")
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{tmp_path}:{root}"
    completed = subprocess.run(
        [
            interpreter,
            "-m",
            "anchor.corrected_sgta.cecd_system_pih_runtime_integration_v1",
            "canary",
            "--family",
            "huatuo",
            "--model-factory",
            "cecd_fake_dash_m_factory:build_model",
            "--input-factory",
            "cecd_fake_dash_m_factory:build_input",
            "--output",
            str(output),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["status"] == "native_eager_canary_passed"
    assert artifact["result"]["fixture"] == "dash-m-module-identity"
    assert artifact["input_identity"]["fixture"] == "dash-m-module-identity"


def test_preflight_binds_real_source_but_canary_and_selected_heads_stay_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (root / "configs/cecd_system_pih_control_preflight_v1.json").read_text(
            encoding="utf-8"
        )
    )
    source = root / "anchor/corrected_sgta/cecd_system_pih_runtime_integration_v1.py"
    for family in ("huatuo", "hulu"):
        integration = plan["runtime_integrations"][family]
        assert integration["status"] == "source_integration_ready_canary_pending"
        assert integration["system_attention_runtime_patch"]["sha256"] == runtime.sha256_file(source)
        assert integration["pih_o_proj_runtime_integration"]["sha256"] == runtime.sha256_file(source)
        assert integration["native_eager_canary_artifact"] is None
        assert plan["pih_selection"][family]["selected_heads_artifact"] is None
        assert plan["pih_selection"][family]["random_heads_artifact"] is None
    result = validate_plan(plan, root=root)
    assert result["passed"] is False
    assert result["control_execution_ready"] is False
    assert result["per_model_runtime_integration_ready"] is False
    assert result["paper_native_reproduction_authorized"] is False
    for family in ("huatuo", "hulu"):
        assert f"{family}:runtime_integration_not_ready" in result["blockers"]
        assert f"{family}:native_eager_canary_artifact_missing" in result["blockers"]
        assert f"{family}:pih_selection_not_ready" in result["blockers"]
        assert f"{family}:system_attention_runtime_patch_missing" not in result["blockers"]
        assert f"{family}:pih_o_proj_runtime_integration_missing" not in result["blockers"]
