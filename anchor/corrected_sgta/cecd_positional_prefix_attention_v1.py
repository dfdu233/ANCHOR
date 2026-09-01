#!/usr/bin/env python3
"""Post-softmax/pre-value positional-prefix attention controls.

This is a clean-room, architecture-neutral control, not a port of either
paper's implementation.  Its source is positional ``prefix_before_image``;
callers must never rename that source ``system`` without role proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .cecd_dynamic_span_builder_v1 import ExpandedPrefixSpans


SCHEMA_VERSION = "cecd-positional-prefix-attention-control-v1"


class AttentionControlError(RuntimeError):
    """Raised when the common-protocol attention contract is violated."""


@dataclass(frozen=True)
class AttentionGeometry:
    num_query_heads: int
    num_key_value_heads: int
    head_dim: int

    def validate(self) -> None:
        if min(self.num_query_heads, self.num_key_value_heads, self.head_dim) <= 0:
            raise AttentionControlError("attention geometry must be positive")
        if self.num_query_heads % self.num_key_value_heads:
            raise AttentionControlError("query heads must be divisible by KV heads")

    @property
    def kv_groups(self) -> int:
        self.validate()
        return self.num_query_heads // self.num_key_value_heads


MODEL_GEOMETRIES = {
    "huatuo": AttentionGeometry(28, 4, 128),
    "hulu": AttentionGeometry(32, 8, 128),
}


@dataclass(frozen=True)
class RedistributionDiagnostics:
    variant: str
    selected_rows: int
    max_mass_error: float
    mass_conserved: bool
    source_mass_before: tuple[float, ...]
    source_mass_after: tuple[float, ...]


def _indices(
    values: Sequence[int], *, key_length: int, label: str, allow_empty: bool = False
) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not allow_empty and not result:
        raise AttentionControlError(f"{label} span is empty")
    if len(set(result)) != len(result) or any(not 0 <= value < key_length for value in result):
        raise AttentionControlError(f"{label} span is duplicated or out of bounds")
    return result


def equal_width_random_span(
    *,
    key_length: int,
    width: int,
    seed: int,
    excluded: Iterable[int] = (),
) -> tuple[int, ...]:
    """Return a deterministic equal-width key control without magic bounds."""

    excluded_set = {int(value) for value in excluded}
    candidates = [index for index in range(key_length) if index not in excluded_set]
    if width <= 0 or width > len(candidates):
        raise AttentionControlError("random-span width exceeds eligible keys")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(len(candidates), generator=generator).tolist()
    return tuple(sorted(candidates[index] for index in order[:width]))


def redistribute_post_softmax_attention(
    attention_weights: torch.Tensor,
    *,
    source_keys: Sequence[int],
    recipient_groups: Sequence[Sequence[int]],
    query_index: int,
    selected_heads: Sequence[int] | None = None,
    alpha: float = 0.0,
    variant: str = "redistribute",
    epsilon: float = 1e-6,
    mass_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, RedistributionDiagnostics]:
    """Transform attention probabilities after softmax and before value matmul.

    Shape is ``[batch, query_heads, query, key]``.  For ``redistribute``, the
    removed source mass is split across recipient groups in proportion to each
    group's original mass and uniformly within each group.  ``source_zero`` is
    the explicitly non-conserving negative control and reports that fact.
    """

    if attention_weights.ndim != 4 or attention_weights.dtype != torch.float32:
        raise AttentionControlError("attention weights must be FP32 [B,H,Q,K] after softmax")
    batch, heads, queries, keys = attention_weights.shape
    if not 0 <= query_index < queries:
        raise AttentionControlError("query index is out of bounds")
    source = _indices(source_keys, key_length=keys, label="source")
    recipients = tuple(
        _indices(group, key_length=keys, label=f"recipient_{index}")
        for index, group in enumerate(recipient_groups)
    )
    all_recipients = set().union(*(set(group) for group in recipients))
    if set(source) & all_recipients:
        raise AttentionControlError("source and recipient keys overlap")
    if sum(len(group) for group in recipients) != len(all_recipients):
        raise AttentionControlError("recipient groups overlap")
    head_ids = (
        tuple(range(heads))
        if selected_heads is None
        else _indices(selected_heads, key_length=heads, label="selected_heads")
    )
    if not 0.0 <= float(alpha) <= 1.0:
        raise AttentionControlError("alpha must be in [0,1]")
    if variant not in {"redistribute", "source_zero"}:
        raise AttentionControlError("unknown attention-control variant")
    if variant == "source_zero" and alpha != 0.0:
        raise AttentionControlError("source_zero control requires alpha=0")

    result = attention_weights.clone()
    selected = result[:, list(head_ids), query_index, :]
    before = selected.sum(dim=-1).to(torch.float64)
    if not torch.isfinite(selected).all() or bool((selected < -mass_tolerance).any()):
        raise AttentionControlError("selected rows are not finite nonnegative probabilities")
    if not torch.allclose(
        before, torch.ones_like(before), atol=mass_tolerance, rtol=0.0
    ):
        raise AttentionControlError("selected attention rows do not sum to one")
    source_before = selected[..., list(source)].sum(dim=-1)
    selected[..., list(source)] *= float(alpha)
    removed = source_before * (1.0 - float(alpha))

    if variant == "redistribute" and alpha != 1.0:
        group_masses = torch.stack(
            [selected[..., list(group)].sum(dim=-1) for group in recipients], dim=-1
        )
        recipient_mass = group_masses.sum(dim=-1)
        if not torch.isfinite(recipient_mass).all() or bool((recipient_mass <= epsilon).any()):
            raise AttentionControlError("recipient mass is nonfinite or below eligibility floor")
        shares = removed.unsqueeze(-1) * group_masses / recipient_mass.unsqueeze(-1)
        for group_index, group in enumerate(recipients):
            selected[..., list(group)] += shares[..., group_index].unsqueeze(-1) / len(group)

    result[:, list(head_ids), query_index, :] = selected
    if not torch.isfinite(result).all():
        raise AttentionControlError("intervention produced nonfinite attention")
    after = selected.sum(dim=-1).to(torch.float64)
    max_error = float((after - before).abs().max().item())
    conserved = max_error <= mass_tolerance
    if variant == "redistribute" and not conserved:
        raise AttentionControlError(f"attention mass was not conserved: {max_error}")
    if variant == "source_zero":
        expected_after = before - source_before.to(torch.float64)
        deficit_error = float((after - expected_after).abs().max().item())
        if deficit_error > mass_tolerance:
            raise AttentionControlError("source-zero deficit differs from removed source mass")

    source_after = selected[..., list(source)].sum(dim=-1)
    return result, RedistributionDiagnostics(
        variant=variant,
        selected_rows=batch * len(head_ids),
        max_mass_error=max_error,
        mass_conserved=conserved,
        source_mass_before=tuple(float(value) for value in source_before.flatten().tolist()),
        source_mass_after=tuple(float(value) for value in source_after.flatten().tolist()),
    )


class PositionalPrefixAttentionPatch:
    """Callable source-patch payload for an eager attention implementation."""

    def __init__(
        self,
        spans: ExpandedPrefixSpans,
        *,
        recipient_mode: str = "proportional_image_suffix",
        alpha: float = 0.0,
        variant: str = "redistribute",
        random_seed: int = 0,
    ) -> None:
        self.spans = spans
        self.recipient_mode = recipient_mode
        self.alpha = alpha
        self.variant = variant
        self.random_seed = random_seed

    def _source_and_recipients(self, key_length: int) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        if key_length < self.spans.prefix_length:
            raise AttentionControlError("KV sequence is shorter than the frozen prefix")
        source = self.spans.prefix_before_image
        if self.recipient_mode == "proportional_image_suffix":
            recipients = (self.spans.image, self.spans.suffix_after_image)
        elif self.recipient_mode == "image_only":
            recipients = (self.spans.image,)
        elif self.recipient_mode == "text_only":
            recipients = (self.spans.suffix_after_image,)
        elif self.recipient_mode == "random_equal_width":
            source = equal_width_random_span(
                key_length=self.spans.prefix_length,
                width=len(source),
                seed=self.random_seed,
            )
            source_set = set(source)
            recipients = (
                tuple(index for index in self.spans.image if index not in source_set),
                tuple(
                    index
                    for index in self.spans.suffix_after_image
                    if index not in source_set
                ),
            )
        else:
            raise AttentionControlError("unknown recipient mode")
        return source, recipients

    def __call__(self, attention_weights: torch.Tensor) -> tuple[torch.Tensor, RedistributionDiagnostics]:
        source, recipients = self._source_and_recipients(attention_weights.shape[-1])
        return redistribute_post_softmax_attention(
            attention_weights,
            source_keys=source,
            recipient_groups=recipients,
            query_index=self.spans.prefix_length - 1,
            alpha=self.alpha,
            variant=self.variant,
        )


def repeat_kv_for_gqa(states: torch.Tensor, geometry: AttentionGeometry) -> torch.Tensor:
    """Repeat ``[B,KV,T,D]`` states to query heads with explicit GQA checks."""

    geometry.validate()
    if states.ndim != 4:
        raise AttentionControlError("KV states must have shape [B,KV,T,D]")
    if states.shape[1] != geometry.num_key_value_heads or states.shape[-1] != geometry.head_dim:
        raise AttentionControlError("KV tensor does not match declared geometry")
    if geometry.kv_groups == 1:
        return states
    batch, kv_heads, length, dim = states.shape
    return (
        states[:, :, None, :, :]
        .expand(batch, kv_heads, geometry.kv_groups, length, dim)
        .reshape(batch, geometry.num_query_heads, length, dim)
    )


def eager_gqa_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    geometry: AttentionGeometry,
    additive_mask: torch.Tensor | None = None,
    patch: PositionalPrefixAttentionPatch | None = None,
) -> tuple[torch.Tensor, torch.Tensor, RedistributionDiagnostics | None]:
    """Small eager reference locating the patch at the exact semantic boundary."""

    geometry.validate()
    if query.ndim != 4 or query.shape[1] != geometry.num_query_heads or query.shape[-1] != geometry.head_dim:
        raise AttentionControlError("query tensor does not match declared geometry")
    expanded_key = repeat_kv_for_gqa(key, geometry)
    expanded_value = repeat_kv_for_gqa(value, geometry)
    scores = torch.matmul(query.float(), expanded_key.float().transpose(-2, -1)) / (
        geometry.head_dim**0.5
    )
    if additive_mask is not None:
        scores = scores + additive_mask.float()
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
    diagnostics = None
    if patch is not None:
        weights, diagnostics = patch(weights)
    output = torch.matmul(weights, expanded_value.float()).to(query.dtype)
    return output, weights, diagnostics
