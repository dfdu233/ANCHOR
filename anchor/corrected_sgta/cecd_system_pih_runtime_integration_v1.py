#!/usr/bin/env python3
"""Clean-room Huatuo-Qwen2/Hulu-Qwen3 runtime integration for CECD controls.

The implementation binds two architecture-neutral interventions to the exact
target-model boundaries without copying either unlicensed paper repository:

* positional-prefix attention redistribution after FP32 softmax and before
  value aggregation, only on the last frozen-prefix query; and
* per-sample query-head mean ablation immediately before ``o_proj``.

All patches are instance-local context managers.  They restore original
forwards/hooks on ordinary exit and exceptions.  This module does not select
heads, load a checkpoint, authorize a run, or create a numerical-canary
artifact unless its explicit future ``canary`` command is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
import types
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


# ``python -m package.module`` executes this file as ``__main__``.  Factories
# subsequently import the canonical package name, and without this alias Python
# creates a second module object whose dataclass identities are incompatible
# with the CLI's classes.  Bind the canonical name before importing local
# dependencies or defining any runtime types.
if __name__ == "__main__" and __spec__ is not None and __spec__.name:
    canonical_name = __spec__.name
    current_module = sys.modules[__name__]
    existing_module = sys.modules.get(canonical_name)
    if existing_module is not None and existing_module is not current_module:
        raise RuntimeError(f"canonical module identity collision: {canonical_name}")
    sys.modules[canonical_name] = current_module

from .cecd_dynamic_span_builder_v1 import (
    ALLOWED_ROLES,
    ExpandedPrefixSpans,
    build_expanded_prefix_spans,
)
from .cecd_pih_mean_ablation_v1 import PIHMeanAblationHook, derive_head_width
from .cecd_positional_prefix_attention_v1 import (
    MODEL_GEOMETRIES,
    AttentionGeometry,
    PositionalPrefixAttentionPatch,
    RedistributionDiagnostics,
    redistribute_post_softmax_attention,
    repeat_kv_for_gqa,
)
from .cecd_system_numerical_canary_v1 import CanaryResult, compare_first_token_logits


VERSION = "cecd-system-pih-runtime-integration-v1"
PROVENANCE_SCHEMA = "cecd-expanded-role-token-provenance-v1"
CANARY_ARTIFACT_SCHEMA = "cecd-system-pih-native-eager-canary-artifact-v1"
ROOT = Path("/home/dbw/ANCHOR")
FAMILIES = ("huatuo", "hulu")
EXPECTED_LAYERS = {"huatuo": 28, "hulu": 36}
PRIMARY_SYSTEM_LAYERS = {
    "huatuo": tuple(range(21, 28)),
    "hulu": tuple(range(27, 36)),
}
EXPECTED_BACKENDS = {"huatuo": "eager", "hulu": "sdpa"}


class RuntimeIntegrationError(RuntimeError):
    """Raised before or during an invalid runtime intervention."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeIntegrationError(f"required regular file missing or symlinked: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


@dataclass(frozen=True)
class ExpandedRoleTokenProvenance:
    """Exact role and origin identity for one expanded batch-one prefix."""

    schema_version: str
    model_family: str
    role_provenance: tuple[str, ...]
    token_origins: tuple[str, ...]
    attention_mask: tuple[int, ...]
    frozen_prefix_length: int
    spans: ExpandedPrefixSpans
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "role_provenance": list(self.role_provenance),
            "token_origins": list(self.token_origins),
            "attention_mask": list(self.attention_mask),
            "frozen_prefix_length": self.frozen_prefix_length,
            "spans": asdict(self.spans),
        }
        payload["fingerprint"] = canonical_sha256(payload)
        return payload


def _finish_provenance(
    *,
    family: str,
    roles: Sequence[str],
    origins: Sequence[str],
    attention_mask: Sequence[bool | int] | None,
    frozen_prefix_length: int | None,
) -> ExpandedRoleTokenProvenance:
    if family not in FAMILIES:
        raise RuntimeIntegrationError(f"unsupported family: {family}")
    role_tuple = tuple(str(value) for value in roles)
    origin_tuple = tuple(str(value) for value in origins)
    if len(role_tuple) != len(origin_tuple) or not role_tuple:
        raise RuntimeIntegrationError("expanded roles and token origins must be nonempty/equal")
    mask = (
        tuple(1 for _ in role_tuple)
        if attention_mask is None
        else tuple(int(bool(value)) for value in attention_mask)
    )
    if len(mask) != len(role_tuple):
        raise RuntimeIntegrationError("expanded attention mask length mismatch")
    spans = build_expanded_prefix_spans(
        role_tuple,
        attention_mask=mask,
        frozen_prefix_length=frozen_prefix_length,
    )
    payload = {
        "schema_version": PROVENANCE_SCHEMA,
        "model_family": family,
        "role_provenance": list(role_tuple),
        "token_origins": list(origin_tuple),
        "attention_mask": list(mask),
        "frozen_prefix_length": spans.prefix_length,
        "spans": asdict(spans),
    }
    return ExpandedRoleTokenProvenance(
        schema_version=PROVENANCE_SCHEMA,
        model_family=family,
        role_provenance=role_tuple,
        token_origins=origin_tuple,
        attention_mask=mask,
        frozen_prefix_length=spans.prefix_length,
        spans=spans,
        fingerprint=canonical_sha256(payload),
    )


def build_huatuo_expanded_provenance(
    *,
    input_ids: Sequence[int],
    token_roles: Sequence[str],
    image_placeholder_id: int,
    projected_visual_token_count: int,
    attention_mask: Sequence[bool | int] | None = None,
    frozen_prefix_length: int | None = None,
) -> ExpandedRoleTokenProvenance:
    """Replace Huatuo's one image placeholder with exact projected-token roles."""

    ids = tuple(int(value) for value in input_ids)
    roles = tuple(str(value) for value in token_roles)
    if len(ids) != len(roles) or set(roles) - ALLOWED_ROLES:
        raise RuntimeIntegrationError("Huatuo token IDs/roles are misaligned or invalid")
    positions = [index for index, value in enumerate(ids) if value == image_placeholder_id]
    if len(positions) != 1:
        raise RuntimeIntegrationError("Huatuo requires exactly one image placeholder")
    if isinstance(projected_visual_token_count, bool) or projected_visual_token_count <= 0:
        raise RuntimeIntegrationError("projected visual-token count must be positive")
    placeholder = positions[0]
    if roles[placeholder] != "image":
        raise RuntimeIntegrationError("Huatuo image placeholder must have image provenance")
    expanded_roles = (
        roles[:placeholder]
        + ("image",) * projected_visual_token_count
        + roles[placeholder + 1 :]
    )
    origins = (
        tuple(f"source_token:{index}" for index in range(placeholder))
        + tuple(
            f"expanded_image:source={placeholder}:patch={patch}"
            for patch in range(projected_visual_token_count)
        )
        + tuple(f"source_token:{index}" for index in range(placeholder + 1, len(ids)))
    )
    expanded_mask = None
    if attention_mask is not None:
        original = tuple(int(bool(value)) for value in attention_mask)
        if len(original) != len(ids) or not original[placeholder]:
            raise RuntimeIntegrationError("Huatuo placeholder must be active in attention mask")
        expanded_mask = (
            original[:placeholder]
            + (1,) * projected_visual_token_count
            + original[placeholder + 1 :]
        )
    return _finish_provenance(
        family="huatuo",
        roles=expanded_roles,
        origins=origins,
        attention_mask=expanded_mask,
        frozen_prefix_length=frozen_prefix_length,
    )


def build_hulu_expanded_provenance(
    *,
    expanded_input_ids: Sequence[int],
    token_roles: Sequence[str],
    image_token_id: int,
    attention_mask: Sequence[bool | int] | None = None,
    frozen_prefix_length: int | None = None,
) -> ExpandedRoleTokenProvenance:
    """Bind Hulu's processor-materialized contiguous image-token run exactly."""

    ids = tuple(int(value) for value in expanded_input_ids)
    roles = tuple(str(value) for value in token_roles)
    if len(ids) != len(roles) or set(roles) - ALLOWED_ROLES:
        raise RuntimeIntegrationError("Hulu expanded IDs/roles are misaligned or invalid")
    image_positions = tuple(index for index, value in enumerate(ids) if value == image_token_id)
    if not image_positions or image_positions != tuple(
        range(image_positions[0], image_positions[-1] + 1)
    ):
        raise RuntimeIntegrationError("Hulu image-token IDs must form one nonempty contiguous run")
    if any((roles[index] == "image") != (index in set(image_positions)) for index in range(len(ids))):
        raise RuntimeIntegrationError("Hulu token roles disagree with exact image-token IDs")
    origins = tuple(
        f"expanded_image:token_index={index}:patch={index-image_positions[0]}"
        if index in set(image_positions)
        else f"source_token:{index}"
        for index in range(len(ids))
    )
    return _finish_provenance(
        family="hulu",
        roles=roles,
        origins=origins,
        attention_mask=attention_mask,
        frozen_prefix_length=frozen_prefix_length,
    )


class LastFrozenPrefixAttentionSession:
    """Apply one patch on prefill and never on cached decode tokens."""

    def __init__(self, patch: PositionalPrefixAttentionPatch) -> None:
        self.patch = patch
        self.prefill_seen = False
        self.diagnostics: list[RedistributionDiagnostics] = []

    def apply(self, weights: torch.Tensor) -> torch.Tensor:
        return self.apply_chunk(
            weights,
            query_start=0,
            total_query_length=weights.shape[-2],
        )

    def apply_chunk(
        self,
        weights: torch.Tensor,
        *,
        query_start: int,
        total_query_length: int,
    ) -> torch.Tensor:
        """Patch the frozen-prefix row without materializing the full Q x K map."""

        query_length, key_length = weights.shape[-2:]
        prefix = self.patch.spans.prefix_length
        if total_query_length == prefix and key_length == prefix:
            target = prefix - 1
            query_stop = query_start + query_length
            if not query_start <= target < query_stop:
                return weights
            if self.prefill_seen:
                raise RuntimeIntegrationError("a sample cannot execute two full prefills")
            source, recipients = self.patch._source_and_recipients(key_length)
            transformed, diagnostics = redistribute_post_softmax_attention(
                weights,
                source_keys=source,
                recipient_groups=recipients,
                query_index=target - query_start,
                alpha=self.patch.alpha,
                variant=self.patch.variant,
            )
            self.prefill_seen = True
            self.diagnostics.append(diagnostics)
            return transformed
        if total_query_length == 1 and query_length == 1 and key_length > prefix:
            if not self.prefill_seen:
                raise RuntimeIntegrationError("cached decode observed before patched prefill")
            return weights
        raise RuntimeIntegrationError(
            "unsupported chunked attention shape "
            f"chunk_Q={query_length}, total_Q={total_query_length}, "
            f"start={query_start}, K={key_length}, prefix={prefix}"
        )


def _eager_attention_core(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    geometry: AttentionGeometry,
    scaling: float,
    attention_mask: torch.Tensor | None,
    training: bool,
    dropout: float,
    session: LastFrozenPrefixAttentionSession | None,
    return_weights: bool = True,
    query_chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if training or float(dropout) != 0.0:
        raise RuntimeIntegrationError("common-protocol runtime is inference-only with dropout zero")
    geometry.validate()
    if query_chunk_size <= 0:
        raise RuntimeIntegrationError("query_chunk_size must be positive")
    expanded_key = repeat_kv_for_gqa(key, geometry)
    expanded_value = repeat_kv_for_gqa(value, geometry)
    key_transpose = expanded_key.transpose(-2, -1)
    total_query_length = query.shape[-2]
    output_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    # Preserve the target eager path's native-dtype score matmul. Only each
    # row-wise softmax result is promoted to FP32. Query chunking changes no
    # reduction axis and avoids Hulu's quadratic full-prefill allocation.
    for query_start in range(0, total_query_length, query_chunk_size):
        query_stop = min(total_query_length, query_start + query_chunk_size)
        query_chunk = query[..., query_start:query_stop, :]
        scores = torch.matmul(query_chunk, key_transpose) * float(scaling)
        if attention_mask is not None:
            mask = attention_mask[..., : expanded_key.shape[-2]]
            if mask.shape[-2] not in (1, total_query_length):
                raise RuntimeIntegrationError("attention mask query dimension is incompatible")
            if mask.shape[-2] == total_query_length:
                mask = mask[..., query_start:query_stop, :]
            mask = mask.to(scores.dtype)
            try:
                scores = scores + mask
            except RuntimeError as error:
                raise RuntimeIntegrationError("attention mask is not broadcast-compatible") from error
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        if session is not None:
            weights = session.apply_chunk(
                weights,
                query_start=query_start,
                total_query_length=total_query_length,
            )
        native_weights = weights.to(query.dtype)
        output_chunks.append(torch.matmul(native_weights, expanded_value))
        if return_weights:
            weight_chunks.append(native_weights)
    output = torch.cat(output_chunks, dim=-2)
    all_weights = torch.cat(weight_chunks, dim=-2) if return_weights else None
    return output, all_weights


def _forward_globals(original_forward: Callable[..., Any]) -> Mapping[str, Any]:
    function = getattr(original_forward, "__func__", original_forward)
    globals_dict = getattr(function, "__globals__", None)
    if not isinstance(globals_dict, Mapping):
        raise RuntimeIntegrationError("cannot resolve architecture helper globals")
    return globals_dict


def _qwen2_forward(
    module: torch.nn.Module,
    *,
    original_forward: Callable[..., Any],
    geometry: AttentionGeometry,
    session: LastFrozenPrefixAttentionSession | None,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
    del use_cache
    if "padding_mask" in kwargs:
        raise RuntimeIntegrationError("deprecated padding_mask is forbidden in frozen runtime")
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise RuntimeIntegrationError("Qwen2 runtime requires batch-one hidden states")
    batch, query_length, _ = hidden_states.shape
    query = module.q_proj(hidden_states).view(batch, query_length, geometry.num_query_heads, geometry.head_dim).transpose(1, 2)
    key = module.k_proj(hidden_states).view(batch, query_length, geometry.num_key_value_heads, geometry.head_dim).transpose(1, 2)
    value = module.v_proj(hidden_states).view(batch, query_length, geometry.num_key_value_heads, geometry.head_dim).transpose(1, 2)
    kv_length = key.shape[-2]
    if past_key_value is not None:
        if getattr(module, "layer_idx", None) is None:
            raise RuntimeIntegrationError("Qwen2 cached runtime requires layer_idx")
        kv_length += int(past_key_value.get_usable_length(kv_length, module.layer_idx))
    cos, sin = module.rotary_emb(value, seq_len=kv_length)
    apply_rotary = _forward_globals(original_forward).get("apply_rotary_pos_emb")
    if not callable(apply_rotary):
        raise RuntimeIntegrationError("Qwen2 apply_rotary_pos_emb helper is unavailable")
    query, key = apply_rotary(query, key, cos, sin, position_ids)
    if past_key_value is not None:
        key, value = past_key_value.update(
            key, value, module.layer_idx, {"sin": sin, "cos": cos}
        )
    output, weights = _eager_attention_core(
        query=query,
        key=key,
        value=value,
        geometry=geometry,
        scaling=geometry.head_dim**-0.5,
        attention_mask=attention_mask,
        training=bool(module.training),
        dropout=float(getattr(module, "attention_dropout", 0.0)),
        session=session,
        return_weights=output_attentions,
        # Huatuo's native backend is already eager. Preserve its full-Q GEMM
        # shape: real BF16 canary evidence showed that row chunking changes
        # rounding enough to fail the frozen native/eager tolerance.
        query_chunk_size=query_length,
    )
    output = output.transpose(1, 2).contiguous().reshape(
        batch, query_length, geometry.num_query_heads * geometry.head_dim
    )
    output = module.o_proj(output)
    return output, weights if output_attentions else None, past_key_value


def _qwen3_forward(
    module: torch.nn.Module,
    *,
    original_forward: Callable[..., Any],
    geometry: AttentionGeometry,
    session: LastFrozenPrefixAttentionSession | None,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: Any = None,
    cache_position: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise RuntimeIntegrationError("Qwen3 runtime requires batch-one hidden states")
    batch, query_length, _ = hidden_states.shape
    query = module.q_norm(
        module.q_proj(hidden_states).view(batch, query_length, geometry.num_query_heads, geometry.head_dim)
    ).transpose(1, 2)
    key = module.k_norm(
        module.k_proj(hidden_states).view(batch, query_length, geometry.num_key_value_heads, geometry.head_dim)
    ).transpose(1, 2)
    value = module.v_proj(hidden_states).view(
        batch, query_length, geometry.num_key_value_heads, geometry.head_dim
    ).transpose(1, 2)
    apply_rotary = _forward_globals(original_forward).get("apply_rotary_pos_emb")
    if not callable(apply_rotary):
        raise RuntimeIntegrationError("Qwen3 apply_rotary_pos_emb helper is unavailable")
    cos, sin = position_embeddings
    query, key = apply_rotary(query, key, cos, sin)
    if past_key_value is not None:
        key, value = past_key_value.update(
            key,
            value,
            module.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
    output_attentions = bool(kwargs.pop("output_attentions", False))
    output, weights = _eager_attention_core(
        query=query,
        key=key,
        value=value,
        geometry=geometry,
        scaling=float(getattr(module, "scaling", geometry.head_dim**-0.5)),
        attention_mask=attention_mask,
        training=bool(module.training),
        dropout=0.0 if not module.training else float(getattr(module, "attention_dropout", 0.0)),
        session=session,
        return_weights=output_attentions,
        # Hulu's ~16k-token full QxK eager map exceeds available memory.
        # Chunk only this SDPA-family clean-room path along query rows.
        query_chunk_size=256,
    )
    output = output.transpose(1, 2).contiguous().reshape(
        batch, query_length, geometry.num_query_heads * geometry.head_dim
    )
    return module.o_proj(output), weights


def resolve_decoder_layers(model: torch.nn.Module, family: str) -> tuple[torch.nn.Module, ...]:
    """Resolve the pinned decoder layer path without touching model weights."""

    if family not in FAMILIES:
        raise RuntimeIntegrationError(f"unsupported family: {family}")
    candidates = (
        ("model", "model", "layers"),
        ("model", "layers"),
        ("layers",),
    )
    for chain in candidates:
        current: Any = model
        for name in chain:
            current = getattr(current, name, None)
            if current is None:
                break
        if current is not None:
            try:
                layers = tuple(current)
            except TypeError:
                continue
            if len(layers) == EXPECTED_LAYERS[family] and all(
                hasattr(layer, "self_attn") for layer in layers
            ):
                return layers
    raise RuntimeIntegrationError(f"cannot resolve exact {family} decoder layer closure")


def _module_geometry(module: torch.nn.Module, family: str) -> AttentionGeometry:
    expected = MODEL_GEOMETRIES[family]
    q_out = int(getattr(module.q_proj, "out_features", -1))
    k_out = int(getattr(module.k_proj, "out_features", -1))
    v_out = int(getattr(module.v_proj, "out_features", -1))
    o_in = int(getattr(module.o_proj, "in_features", -1))
    if (
        q_out != expected.num_query_heads * expected.head_dim
        or k_out != expected.num_key_value_heads * expected.head_dim
        or v_out != expected.num_key_value_heads * expected.head_dim
        or o_in != expected.num_query_heads * expected.head_dim
        or derive_head_width(o_in, expected.num_query_heads) != 128
    ):
        raise RuntimeIntegrationError(f"{family} attention projection geometry drift")
    return expected


class EagerAttentionPatchContext:
    """Replace all decoder attention forwards and restore them exactly."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        family: str,
        provenance: ExpandedRoleTokenProvenance,
        system_layers: Sequence[int] = (),
        recipient_mode: str = "proportional_image_suffix",
        alpha: float = 0.0,
        variant: str = "redistribute",
        random_seed: int = 0,
        rotary_apply: Callable[..., Any] | None = None,
    ) -> None:
        if family not in FAMILIES or provenance.model_family != family:
            raise RuntimeIntegrationError("model family/provenance mismatch")
        self.model = model
        self.family = family
        self.provenance = provenance
        self.system_layers = tuple(int(value) for value in system_layers)
        if len(set(self.system_layers)) != len(self.system_layers) or any(
            not 0 <= layer < EXPECTED_LAYERS[family] for layer in self.system_layers
        ):
            raise RuntimeIntegrationError("system layer selection is duplicated/out of bounds")
        self.patch_kwargs = {
            "recipient_mode": recipient_mode,
            "alpha": alpha,
            "variant": variant,
            "random_seed": random_seed,
        }
        self.rotary_apply = rotary_apply
        self._restorers: list[Callable[[], None]] = []
        self.sessions: dict[int, LastFrozenPrefixAttentionSession] = {}

    def __enter__(self) -> "EagerAttentionPatchContext":
        layers = resolve_decoder_layers(self.model, self.family)
        try:
            for index, layer in enumerate(layers):
                attention = layer.self_attn
                geometry = _module_geometry(attention, self.family)
                original = attention.forward
                had_instance_forward = "forward" in attention.__dict__
                instance_forward = attention.__dict__.get("forward")
                if self.rotary_apply is not None:
                    function = getattr(original, "__func__", original)
                    globals_copy = dict(_forward_globals(original))
                    globals_copy["apply_rotary_pos_emb"] = self.rotary_apply
                    rebound = types.FunctionType(
                        function.__code__, globals_copy, function.__name__, function.__defaults__, function.__closure__
                    )
                    rebound.__kwdefaults__ = getattr(function, "__kwdefaults__", None)
                    original_for_helpers = types.MethodType(rebound, attention)
                else:
                    original_for_helpers = original
                session = None
                if index in self.system_layers:
                    session = LastFrozenPrefixAttentionSession(
                        PositionalPrefixAttentionPatch(
                            self.provenance.spans, **self.patch_kwargs
                        )
                    )
                    self.sessions[index] = session

                if self.family == "huatuo":
                    def replacement(module: torch.nn.Module, *args: Any, _original=original_for_helpers, _geometry=geometry, _session=session, **kwargs: Any):
                        if args:
                            if "hidden_states" in kwargs:
                                raise RuntimeIntegrationError("hidden_states supplied twice")
                            kwargs["hidden_states"] = args[0]
                            if len(args) > 1:
                                raise RuntimeIntegrationError("positional Qwen2 arguments beyond hidden_states are forbidden")
                        return _qwen2_forward(
                            module, original_forward=_original, geometry=_geometry,
                            session=_session, **kwargs,
                        )
                else:
                    def replacement(module: torch.nn.Module, *args: Any, _original=original_for_helpers, _geometry=geometry, _session=session, **kwargs: Any):
                        if args:
                            if "hidden_states" in kwargs:
                                raise RuntimeIntegrationError("hidden_states supplied twice")
                            kwargs["hidden_states"] = args[0]
                            if len(args) > 1:
                                raise RuntimeIntegrationError("positional Qwen3 arguments beyond hidden_states are forbidden")
                        return _qwen3_forward(
                            module, original_forward=_original, geometry=_geometry,
                            session=_session, **kwargs,
                        )
                attention.forward = types.MethodType(replacement, attention)

                def restore(
                    target=attention,
                    had=had_instance_forward,
                    prior=instance_forward,
                ) -> None:
                    if had:
                        target.forward = prior
                    else:
                        target.__dict__.pop("forward", None)

                self._restorers.append(restore)
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self) -> None:
        for restore in reversed(self._restorers):
            restore()
        self._restorers.clear()

    def __exit__(self, *_args: Any) -> None:
        self._restore()


class PIHPreOProjPatchContext:
    """Register per-layer batch-one PIH hooks for exactly one sample."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        family: str,
        provenance: ExpandedRoleTokenProvenance,
        selected_heads_by_layer: Mapping[int, Sequence[int]],
    ) -> None:
        if provenance.model_family != family:
            raise RuntimeIntegrationError("PIH model family/provenance mismatch")
        self.model = model
        self.family = family
        self.provenance = provenance
        self.selected = {
            int(layer): tuple(int(head) for head in heads)
            for layer, heads in selected_heads_by_layer.items()
        }
        self.hooks: dict[int, PIHMeanAblationHook] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> "PIHPreOProjPatchContext":
        layers = resolve_decoder_layers(self.model, self.family)
        geometry = MODEL_GEOMETRIES[self.family]
        if any(not 0 <= index < len(layers) for index in self.selected):
            raise RuntimeIntegrationError("PIH layer selection is out of bounds")
        for index, heads in self.selected.items():
            if (
                not heads
                or len(set(heads)) != len(heads)
                or any(not 0 <= head < geometry.num_query_heads for head in heads)
            ):
                raise RuntimeIntegrationError(
                    f"PIH query-head selection is empty, duplicated, or out of bounds at layer {index}"
                )
        try:
            for index, heads in sorted(self.selected.items()):
                attention = layers[index].self_attn
                _module_geometry(attention, self.family)
                if derive_head_width(attention.o_proj.in_features, geometry.num_query_heads) != 128:
                    raise RuntimeIntegrationError("PIH head width must equal 128")
                hook = PIHMeanAblationHook(
                    selected_heads=heads,
                    num_query_heads=geometry.num_query_heads,
                )
                hook.begin_sample(frozen_prefix_length=self.provenance.frozen_prefix_length)
                self.handles.append(attention.o_proj.register_forward_pre_hook(hook))
                self.hooks[index] = hook
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        for hook in self.hooks.values():
            hook.end_sample()
        self.hooks.clear()

    def __exit__(self, *_args: Any) -> None:
        self._restore()


class SystemPIHRuntimeContext:
    """Compose eager attention and PIH contexts with transactional rollback."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        family: str,
        provenance: ExpandedRoleTokenProvenance,
        system_layers: Sequence[int] = (),
        selected_heads_by_layer: Mapping[int, Sequence[int]] | None = None,
        recipient_mode: str = "proportional_image_suffix",
        alpha: float = 0.0,
        variant: str = "redistribute",
        random_seed: int = 0,
        rotary_apply: Callable[..., Any] | None = None,
    ) -> None:
        self.eager = EagerAttentionPatchContext(
            model,
            family=family,
            provenance=provenance,
            system_layers=system_layers,
            recipient_mode=recipient_mode,
            alpha=alpha,
            variant=variant,
            random_seed=random_seed,
            rotary_apply=rotary_apply,
        )
        self.pih = PIHPreOProjPatchContext(
            model,
            family=family,
            provenance=provenance,
            selected_heads_by_layer=selected_heads_by_layer or {},
        )
        self._stack: ExitStack | None = None

    def __enter__(self) -> "SystemPIHRuntimeContext":
        stack = ExitStack()
        try:
            stack.enter_context(self.eager)
            stack.enter_context(self.pih)
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return self

    def __exit__(self, *args: Any) -> None:
        if self._stack is not None:
            self._stack.__exit__(*args)
            self._stack = None


def _extract_first_token_logits(output: Any) -> torch.Tensor:
    logits = output if torch.is_tensor(output) else getattr(output, "logits", None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get("logits")
    if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[0] != 1:
        raise RuntimeIntegrationError("model forward must expose batch-one [B,T,V] logits")
    # Clone the final position so the returned view does not retain the full
    # [sequence, vocabulary] logits allocation across the two canary passes.
    return logits[:, -1, :].detach().clone()


def run_model_native_vs_eager_canary(
    *,
    model: torch.nn.Module,
    family: str,
    provenance: ExpandedRoleTokenProvenance,
    forward_kwargs: Mapping[str, Any],
    rotary_apply: Callable[..., Any] | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> CanaryResult:
    """Compare unmodified native and clean-room eager first-token logits."""

    if model.training:
        raise RuntimeIntegrationError("native/eager canary requires model.eval()")
    if forward_kwargs.get("past_key_values") is not None or forward_kwargs.get("past_key_value") is not None:
        raise RuntimeIntegrationError("native/eager canary requires a cache-free prefill")
    kwargs = dict(forward_kwargs)
    embeddings = kwargs.get("inputs_embeds")
    if (
        not torch.is_tensor(embeddings)
        or embeddings.ndim != 3
        or embeddings.shape[0] != 1
        or embeddings.shape[1] != len(provenance.role_provenance)
    ):
        raise RuntimeIntegrationError(
            "native/eager canary requires batch-one already-expanded inputs_embeds "
            "matching exact role provenance"
        )
    mask = kwargs.get("attention_mask")
    if mask is not None:
        if not torch.is_tensor(mask) or mask.shape[-1] != len(provenance.attention_mask):
            raise RuntimeIntegrationError("canary attention mask/provenance length mismatch")
        observed_mask = tuple(int(bool(value)) for value in mask[0].detach().cpu().tolist())
        if observed_mask != provenance.attention_mask:
            raise RuntimeIntegrationError("canary attention mask differs from frozen provenance")
    kwargs["use_cache"] = False
    with torch.inference_mode():
        native = _extract_first_token_logits(model(**kwargs))
        if embeddings.is_cuda:
            torch.cuda.empty_cache()
        with EagerAttentionPatchContext(
            model,
            family=family,
            provenance=provenance,
            system_layers=(),
            rotary_apply=rotary_apply,
        ):
            eager = _extract_first_token_logits(model(**kwargs))
    return compare_first_token_logits(native, eager, atol=atol, rtol=rtol)


def _load_callable(spec: str) -> tuple[Callable[..., Any], Path]:
    module_name, separator, name = spec.partition(":")
    if not separator or not module_name or not name:
        raise RuntimeIntegrationError("factory spec must be module.path:callable")
    module = importlib.import_module(module_name)
    value = getattr(module, name, None)
    source = inspect.getsourcefile(value) if callable(value) else None
    if not callable(value) or source is None:
        raise RuntimeIntegrationError(f"factory is not a source-backed callable: {spec}")
    return value, Path(source)


def runtime_description(family: str) -> dict[str, Any]:
    if family not in FAMILIES:
        raise RuntimeIntegrationError(f"unsupported family: {family}")
    geometry = MODEL_GEOMETRIES[family]
    payload = {
        "version": VERSION,
        "model_family": family,
        "expected_native_backend": EXPECTED_BACKENDS[family],
        "runtime_backend": "clean_room_eager",
        "backend_transition": (
            "native_eager_to_clean_room_eager"
            if family == "huatuo"
            else "native_sdpa_to_clean_room_eager"
        ),
        "decoder_layers": EXPECTED_LAYERS[family],
        "geometry": asdict(geometry),
        "head_width": geometry.head_dim,
        "primary_system_layers_zero_indexed": list(PRIMARY_SYSTEM_LAYERS[family]),
        "system_patch_boundary": "after_fp32_softmax_before_value_matmul_last_frozen_prefix_query_only",
        "pih_patch_boundary": "forward_pre_hook_on_attention_o_proj_batch_one",
        "context_restoration": "instance_forward_and_hook_restored_on_exit_or_exception",
        "paper_native_claimed": False,
        "official_code_copied": False,
        "model_loaded": False,
        "gpu_touched": False,
        "canary_run": False,
        "selected_heads_loaded": False,
        "source": file_record(Path(__file__)),
    }
    payload["fingerprint"] = canonical_sha256(payload)
    return payload


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeIntegrationError(f"write-once artifact collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    describe = subparsers.add_parser("describe")
    describe.add_argument("--family", choices=FAMILIES, required=True)
    describe.add_argument("--output", type=Path)
    canary = subparsers.add_parser("canary")
    canary.add_argument("--family", choices=FAMILIES, required=True)
    canary.add_argument("--model-factory", required=True)
    canary.add_argument("--input-factory", required=True)
    canary.add_argument("--output", type=Path, required=True)
    canary.add_argument("--atol", type=float, default=1e-4)
    canary.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    if args.command == "describe":
        result = runtime_description(args.family)
        if args.output:
            _write_once(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    model_factory, model_source = _load_callable(args.model_factory)
    input_factory, input_source = _load_callable(args.input_factory)
    model = model_factory()
    bundle = input_factory(model)
    required = {"provenance", "forward_kwargs", "input_identity"}
    if not isinstance(bundle, Mapping) or set(bundle) != required:
        raise RuntimeIntegrationError(f"input factory must return exactly {sorted(required)}")
    provenance = bundle["provenance"]
    if not isinstance(provenance, ExpandedRoleTokenProvenance):
        raise RuntimeIntegrationError("input factory provenance has wrong type")
    result = run_model_native_vs_eager_canary(
        model=model,
        family=args.family,
        provenance=provenance,
        forward_kwargs=bundle["forward_kwargs"],
        atol=args.atol,
        rtol=args.rtol,
    )
    artifact = {
        "schema_version": CANARY_ARTIFACT_SCHEMA,
        "status": "native_eager_canary_passed" if result.passed else "failed",
        "model_family": args.family,
        "result": result.as_dict(),
        "provenance_fingerprint": provenance.fingerprint,
        "input_identity": bundle["input_identity"],
        "model_factory": {"spec": args.model_factory, **file_record(model_source)},
        "input_factory": {"spec": args.input_factory, **file_record(input_source)},
        "integration_source": file_record(Path(__file__)),
        "paper_native_claimed": False,
        "selected_heads_consumed": False,
    }
    artifact["fingerprint"] = canonical_sha256(artifact)
    _write_once(args.output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    if not result.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
