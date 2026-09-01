#!/usr/bin/env python3
"""Native-vs-eager first-token numerical canary interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

import torch


SCHEMA_VERSION = "cecd-native-eager-first-token-canary-v1"


class NumericalCanaryError(RuntimeError):
    pass


@dataclass(frozen=True)
class CanaryResult:
    schema_version: str
    passed: bool
    shape: tuple[int, ...]
    max_absolute_error: float
    max_relative_error: float
    argmax_equal: bool
    atol: float
    rtol: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_first_token_logits(
    native_logits: torch.Tensor,
    eager_logits: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    require_argmax_equal: bool = True,
) -> CanaryResult:
    """Compare unmodified first generated-token logits, fail closed on drift."""

    if native_logits.shape != eager_logits.shape or native_logits.ndim not in (1, 2):
        raise NumericalCanaryError("first-token logit shapes must match as [V] or [B,V]")
    if not native_logits.is_floating_point() or not eager_logits.is_floating_point():
        raise NumericalCanaryError("logits must be floating tensors")
    native = native_logits.detach().float().cpu()
    eager = eager_logits.detach().float().cpu()
    if not torch.isfinite(native).all() or not torch.isfinite(eager).all():
        raise NumericalCanaryError("canary logits contain nonfinite values")
    absolute = (native - eager).abs()
    relative = absolute / torch.maximum(native.abs(), torch.full_like(native, 1e-12))
    argmax_equal = bool(torch.equal(native.argmax(dim=-1), eager.argmax(dim=-1)))
    close = bool(torch.allclose(native, eager, atol=atol, rtol=rtol))
    passed = close and (argmax_equal or not require_argmax_equal)
    return CanaryResult(
        schema_version=SCHEMA_VERSION,
        passed=passed,
        shape=tuple(native.shape),
        max_absolute_error=float(absolute.max().item()),
        max_relative_error=float(relative.max().item()),
        argmax_equal=argmax_equal,
        atol=float(atol),
        rtol=float(rtol),
    )


def run_first_token_canary(
    native_forward: Callable[..., torch.Tensor],
    eager_forward: Callable[..., torch.Tensor],
    *,
    forward_kwargs: Mapping[str, Any],
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> CanaryResult:
    """Run two caller-owned unmodified paths without selecting on outcomes."""

    with torch.inference_mode():
        native = native_forward(**dict(forward_kwargs))
        eager = eager_forward(**dict(forward_kwargs))
    result = compare_first_token_logits(native, eager, atol=atol, rtol=rtol)
    if not result.passed:
        raise NumericalCanaryError(
            f"native/eager first-token canary failed: abs={result.max_absolute_error} "
            f"rel={result.max_relative_error} argmax={result.argmax_equal}"
        )
    return result

