import json

import pytest
import torch

from anchor.corrected_sgta.cecd_dynamic_span_builder_v1 import (
    DynamicSpanError,
    build_expanded_prefix_spans,
)
from anchor.corrected_sgta.cecd_pih_dev_selection_schema_v1 import (
    SCHEMA_VERSION as SELECTION_SCHEMA,
    SelectionSchemaError,
    validate_outcome_blind_dev_manifest,
)
from anchor.corrected_sgta.cecd_pih_mean_ablation_v1 import (
    PIHHookError,
    PIHMeanAblationHook,
    derive_head_width,
    mean_ablate_pre_o_proj,
)
from anchor.corrected_sgta.cecd_positional_prefix_attention_v1 import (
    MODEL_GEOMETRIES,
    AttentionControlError,
    PositionalPrefixAttentionPatch,
    eager_gqa_attention_reference,
    equal_width_random_span,
    redistribute_post_softmax_attention,
    repeat_kv_for_gqa,
)
from anchor.corrected_sgta.cecd_system_numerical_canary_v1 import (
    NumericalCanaryError,
    compare_first_token_logits,
    run_first_token_canary,
)


def _spans(visual_length: int = 5):
    return build_expanded_prefix_spans(
        ["user_text"] * 3
        + ["image"] * visual_length
        + ["user_text"] * 4
        + ["user_text"] * 2,
        attention_mask=[1] * (visual_length + 7) + [0, 0],
        frozen_prefix_length=visual_length + 7,
    )


def test_dynamic_builder_is_role_aware_dynamic_and_excludes_padding() -> None:
    for visual_length in (2, 17, 576, 911):
        spans = _spans(visual_length)
        assert len(spans.image) == visual_length
        assert spans.system == ()
        assert spans.prefix_before_image == (0, 1, 2)
        assert spans.prefix_before_image_is_true_system is False
        assert set().union(*(set(v) for v in spans.role_partition().values())) == set(
            range(spans.prefix_length)
        )


def test_dynamic_builder_proves_true_system_only_from_roles() -> None:
    spans = build_expanded_prefix_spans(
        ["system", "system", "image", "image", "user_text"]
    )
    assert spans.prefix_before_image_is_true_system is True
    with pytest.raises(DynamicSpanError, match="contiguous"):
        build_expanded_prefix_spans(["user_text", "image", "user_text", "image"])
    with pytest.raises(DynamicSpanError, match="right padding"):
        build_expanded_prefix_spans(
            ["user_text", "image", "user_text"], attention_mask=[1, 0, 1]
        )
    with pytest.raises(DynamicSpanError, match="truncate"):
        build_expanded_prefix_spans(
            ["user_text", "image", "image", "user_text"],
            frozen_prefix_length=2,
        )


def _attention_fixture() -> torch.Tensor:
    generator = torch.Generator().manual_seed(41)
    scores = torch.randn(2, 4, 3, 12, generator=generator)
    return torch.softmax(scores.float(), dim=-1)


@pytest.mark.parametrize(
    "groups",
    [
        ((3, 4, 5, 6, 7), (8, 9, 10, 11)),
        ((3, 4, 5, 6, 7),),
        ((8, 9, 10, 11),),
    ],
)
def test_post_softmax_redistribution_conserves_mass_for_recipient_splits(groups) -> None:
    weights = _attention_fixture()
    transformed, diagnostics = redistribute_post_softmax_attention(
        weights,
        source_keys=(0, 1, 2),
        recipient_groups=groups,
        query_index=2,
        alpha=0.0,
    )
    assert diagnostics.mass_conserved is True
    assert diagnostics.max_mass_error <= 1e-6
    assert torch.allclose(
        transformed[:, :, 2].sum(-1), weights[:, :, 2].sum(-1), atol=1e-6
    )
    assert torch.count_nonzero(transformed[:, :, 2, :3]) == 0
    assert torch.equal(transformed[:, :, :2], weights[:, :, :2])


def test_alpha_one_is_exact_identity_and_source_zero_is_explicit_deficit() -> None:
    weights = _attention_fixture()
    identity, identity_diag = redistribute_post_softmax_attention(
        weights,
        source_keys=(0, 1, 2),
        recipient_groups=((3, 4, 5), (6, 7, 8, 9, 10, 11)),
        query_index=2,
        alpha=1.0,
    )
    assert torch.equal(identity, weights)
    assert identity_diag.mass_conserved
    zeroed, zero_diag = redistribute_post_softmax_attention(
        weights,
        source_keys=(0, 1, 2),
        recipient_groups=((3, 4, 5),),
        query_index=2,
        alpha=0.0,
        variant="source_zero",
    )
    assert zero_diag.mass_conserved is False
    removed = weights[:, :, 2, :3].sum(-1)
    assert torch.allclose(
        zeroed[:, :, 2].sum(-1), weights[:, :, 2].sum(-1) - removed, atol=1e-6
    )


def test_random_span_is_seeded_equal_width_and_patch_name_is_positional() -> None:
    first = equal_width_random_span(key_length=50, width=7, seed=20260803)
    second = equal_width_random_span(key_length=50, width=7, seed=20260803)
    assert first == second and len(first) == 7
    spans = _spans(5)
    patch = PositionalPrefixAttentionPatch(
        spans, recipient_mode="random_equal_width", random_seed=9
    )
    weights = torch.softmax(
        torch.randn(1, 4, spans.prefix_length, spans.prefix_length), dim=-1
    )
    transformed, diagnostics = patch(weights)
    assert transformed.shape == weights.shape
    assert diagnostics.mass_conserved
    assert spans.prefix_before_image_is_true_system is False


@pytest.mark.parametrize("family", ["huatuo", "hulu"])
def test_eager_reference_is_gqa_aware_for_both_target_geometries(family: str) -> None:
    geometry = MODEL_GEOMETRIES[family]
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, geometry.num_query_heads, 1, geometry.head_dim, generator=generator)
    key = torch.randn(1, geometry.num_key_value_heads, 12, geometry.head_dim, generator=generator)
    value = torch.randn(1, geometry.num_key_value_heads, 12, geometry.head_dim, generator=generator)
    expanded = repeat_kv_for_gqa(key, geometry)
    assert expanded.shape == (1, geometry.num_query_heads, 12, 128)
    output, weights, _ = eager_gqa_attention_reference(
        query, key, value, geometry=geometry
    )
    assert output.shape == (1, geometry.num_query_heads, 1, 128)
    assert weights.dtype == torch.float32
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6)


def test_repeat_kv_supports_fake_tensor_shape_execution() -> None:
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        pytest.skip("FakeTensorMode unavailable")
    geometry = MODEL_GEOMETRIES["hulu"]
    with FakeTensorMode():
        fake_kv = torch.empty(2, 8, 19, 128)
        repeated = repeat_kv_for_gqa(fake_kv, geometry)
        assert repeated.shape == (2, 32, 19, 128)


def test_zero_recipient_mass_and_overlap_fail_closed() -> None:
    weights = torch.zeros(1, 2, 1, 5)
    weights[..., :2] = 0.5
    with pytest.raises(AttentionControlError, match="recipient mass"):
        redistribute_post_softmax_attention(
            weights,
            source_keys=(0, 1),
            recipient_groups=((2, 3, 4),),
            query_index=0,
        )
    with pytest.raises(AttentionControlError, match="overlap"):
        redistribute_post_softmax_attention(
            torch.softmax(torch.randn(1, 2, 1, 5), -1),
            source_keys=(0, 1),
            recipient_groups=((1, 2),),
            query_index=0,
        )
    with pytest.raises(AttentionControlError, match="FP32"):
        redistribute_post_softmax_attention(
            torch.softmax(torch.randn(1, 2, 1, 5), -1).half(),
            source_keys=(0, 1),
            recipient_groups=((2, 3, 4),),
            query_index=0,
        )


def test_pih_width_uses_pre_o_proj_geometry_and_per_sample_prefix_mean() -> None:
    assert derive_head_width(3584, 28) == 128
    assert derive_head_width(4096, 32) == 128
    assert 2560 // 32 == 80
    tensor = torch.arange(1 * 4 * 4096, dtype=torch.float32).reshape(1, 4, 4096)
    transformed, means = mean_ablate_pre_o_proj(
        tensor,
        selected_heads=(0, 31),
        num_query_heads=32,
        frozen_prefix_length=3,
    )
    expected = tensor[:, :3, :128].mean(1, keepdim=True).expand(1, 4, 128)
    assert torch.equal(transformed[:, :, :128], expected)
    assert means[31].shape == (1, 1, 128)
    assert torch.equal(transformed[:, :, 128:256], tensor[:, :, 128:256])


def test_pih_hook_prefill_decode_lifecycle_has_no_cross_sample_mean() -> None:
    module = torch.nn.Linear(8, 8, bias=False)
    hook = PIHMeanAblationHook(selected_heads=(1,), num_query_heads=2)
    hook.begin_sample(frozen_prefix_length=3)
    prefill = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    prefill_out = hook(module, (prefill,))[0]
    cached_mean = prefill[:, :, 4:8].mean(1, keepdim=True)
    assert torch.equal(prefill_out[:, :, 4:8], cached_mean.expand(1, 3, 4))
    decode = torch.full((1, 1, 8), 999.0)
    decode_out = hook(module, (decode,))[0]
    assert torch.equal(decode_out[:, :, 4:8], cached_mean)
    hook.end_sample()
    with pytest.raises(PIHHookError, match="begin_sample"):
        hook(module, (decode,))
    with pytest.raises(PIHHookError, match="batch size"):
        mean_ablate_pre_o_proj(
            torch.zeros(2, 3, 8),
            selected_heads=(0,),
            num_query_heads=2,
            frozen_prefix_length=3,
        )


def test_pih_mean_ablation_supports_fake_tensor_without_batch_reduction() -> None:
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        pytest.skip("FakeTensorMode unavailable")
    with FakeTensorMode():
        tensor = torch.empty(1, 6, 4096)
        transformed, means = mean_ablate_pre_o_proj(
            tensor,
            selected_heads=(0, 31),
            num_query_heads=32,
            frozen_prefix_length=5,
        )
        assert transformed.shape == (1, 6, 4096)
        assert means[0].shape == (1, 1, 128)


def test_native_eager_canary_accepts_close_and_rejects_drift() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.8]])
    close = compare_first_token_logits(logits, logits + 1e-7)
    assert close.passed and close.argmax_equal
    result = run_first_token_canary(
        lambda x: x + 1e-7,
        lambda x: x,
        forward_kwargs={"x": logits},
    )
    assert result.passed
    with pytest.raises(NumericalCanaryError, match="failed"):
        run_first_token_canary(
            lambda x: x,
            lambda x: x.flip(-1),
            forward_kwargs={"x": logits},
        )


def _dev_manifest(family: str = "huatuo") -> dict:
    return {
        "schema_version": SELECTION_SCHEMA,
        "model_family": family,
        "split": "dev",
        "records": [
            {
                "record_id": f"r{index}",
                "image_id": f"i{index}",
                "patient_id": f"p{index // 2}",
                "prompt_pair_id": f"q{index}",
            }
            for index in range(4)
        ],
    }


def test_selection_runner_is_outcome_blind_schema_only() -> None:
    result = validate_outcome_blind_dev_manifest(_dev_manifest("hulu"))
    assert result["status"] == "schema_only_no_outcomes_no_selection"
    assert result["candidate_count"] == 36 * 32
    assert result["locked_test_scanned"] is False
    assert result["formal_head_selection_authorized"] is False
    assert result["selected_head_artifact_created"] is False
    contaminated = _dev_manifest()
    contaminated["records"][0]["outcome"] = "wrong"
    with pytest.raises(SelectionSchemaError, match="schema"):
        validate_outcome_blind_dev_manifest(contaminated)
    test_split = _dev_manifest()
    test_split["split"] = "test"
    with pytest.raises(SelectionSchemaError, match="dev"):
        validate_outcome_blind_dev_manifest(test_split)
