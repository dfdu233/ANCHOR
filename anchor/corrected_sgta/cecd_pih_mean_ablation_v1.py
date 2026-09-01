#!/usr/bin/env python3
"""Per-sample pre-o_proj query-head mean ablation.

The hook derives head width from the pre-o_proj width, never decoder hidden
size.  It requires batch one and stores only the current sample's prefix mean
for cached decoding, preventing cross-sample leakage.
"""

from __future__ import annotations

from typing import Iterable

import torch


SCHEMA_VERSION = "cecd-pih-inspired-mean-ablation-hook-v1"


class PIHHookError(RuntimeError):
    pass


def derive_head_width(o_proj_in_features: int, num_query_heads: int) -> int:
    if o_proj_in_features <= 0 or num_query_heads <= 0:
        raise PIHHookError("o_proj width and query-head count must be positive")
    if o_proj_in_features % num_query_heads:
        raise PIHHookError("o_proj width is not divisible by query-head count")
    return o_proj_in_features // num_query_heads


def mean_ablate_pre_o_proj(
    tensor: torch.Tensor,
    *,
    selected_heads: Iterable[int],
    num_query_heads: int,
    frozen_prefix_length: int,
    cached_means: dict[int, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Ablate selected heads and return detached per-sample prefix means."""

    if tensor.ndim != 3 or not tensor.is_floating_point():
        raise PIHHookError("pre-o_proj tensor must be floating [B,T,W]")
    if tensor.shape[0] != 1:
        raise PIHHookError("batch size must equal one; cross-sample means are forbidden")
    head_width = derive_head_width(tensor.shape[-1], num_query_heads)
    heads = tuple(int(value) for value in selected_heads)
    if not heads or len(set(heads)) != len(heads):
        raise PIHHookError("selected heads must be nonempty and unique")
    if any(not 0 <= head < num_query_heads for head in heads):
        raise PIHHookError("selected query head is out of bounds")
    if frozen_prefix_length <= 0:
        raise PIHHookError("frozen prefix length must be positive")

    output = tensor.clone()
    means: dict[int, torch.Tensor] = {}
    is_prefill = tensor.shape[1] >= frozen_prefix_length
    if not is_prefill and cached_means is None:
        raise PIHHookError("cached decode requires a current-sample prefix mean")
    for head in heads:
        start = head * head_width
        end = start + head_width
        if is_prefill:
            mean = tensor[:, :frozen_prefix_length, start:end].mean(dim=1, keepdim=True)
            means[head] = mean.detach().clone()
        else:
            if head not in cached_means:
                raise PIHHookError("cached mean missing for selected head")
            mean = cached_means[head].to(device=tensor.device, dtype=tensor.dtype)
            if tuple(mean.shape) != (1, 1, head_width):
                raise PIHHookError("cached mean shape drift")
            means[head] = mean.detach().clone()
        output[..., start:end] = mean.expand(1, tensor.shape[1], head_width)
    return output, means


class PIHMeanAblationHook:
    """Stateful pre-hook with explicit single-sample lifecycle."""

    def __init__(self, *, selected_heads: Iterable[int], num_query_heads: int) -> None:
        self.selected_heads = tuple(int(value) for value in selected_heads)
        self.num_query_heads = int(num_query_heads)
        self.frozen_prefix_length: int | None = None
        self._cached_means: dict[int, torch.Tensor] | None = None

    def begin_sample(self, *, frozen_prefix_length: int) -> None:
        if frozen_prefix_length <= 0:
            raise PIHHookError("frozen prefix length must be positive")
        self.frozen_prefix_length = int(frozen_prefix_length)
        self._cached_means = None

    def end_sample(self) -> None:
        self.frozen_prefix_length = None
        self._cached_means = None

    def __call__(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        del module
        if self.frozen_prefix_length is None:
            raise PIHHookError("begin_sample must be called before the hook")
        if len(inputs) != 1:
            raise PIHHookError("o_proj pre-hook expects one positional tensor")
        transformed, means = mean_ablate_pre_o_proj(
            inputs[0],
            selected_heads=self.selected_heads,
            num_query_heads=self.num_query_heads,
            frozen_prefix_length=self.frozen_prefix_length,
            cached_means=self._cached_means,
        )
        self._cached_means = means
        return (transformed,)

