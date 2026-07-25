"""Source-only calibration of visual attention mass.

This module contains the model-independent mathematics for ANCHOR's attention
interface.  It deliberately does not patch a LLaVA/Transformers class.  A
model-specific hook should call :func:`anchor_attention_logits` immediately
after adding the causal/padding mask and immediately before softmax.

For attention logits ``s`` and visual-key set ``V``, define

    m(s) = sum_{j in V} softmax(s)_j.

Adding the same scalar ``delta`` to every visual-key logit gives

    logit(m(s + delta * 1_V)) = logit(m(s)) + delta.

Consequently, matching a source log-odds center is closed form and preserves
the normalized attention distribution *within* the image exactly.  Source
centers and all gate choices must be fitted with source domains only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


PROTOCOL_VERSION = "rule-attention-mass-anchor-v2"
SOURCE_SELECTION_SCOPE = "source_lodo"


class AttentionMassAnchorError(ValueError):
    """Raised when an attention/source-only contract is invalid."""


@dataclass(frozen=True)
class RobustAttentionCenter:
    """A robust center and source-domain stability mask."""

    center_log_odds: torch.Tensor
    domain_mad: torch.Tensor
    head_mask: torch.Tensor
    mad_threshold: torch.Tensor
    domain_names: tuple[str, ...]
    protocol_version: str = PROTOCOL_VERSION


@dataclass(frozen=True)
class SourceOnlyGateConfig:
    """Frozen criteria for selecting a hook configuration on source LODO.

    The gate is intentionally label-balanced.  In particular,
    ``max_abs_balanced_margin_shift`` prevents a candidate from passing by
    adding a nearly constant Yes/No prior.
    """

    min_net_rescues: int = 3
    max_harms: int = 1
    min_worst_fold_delta_pp: float = -0.5
    max_abs_balanced_margin_shift: float = 0.05
    require_both_labels_rescued: bool = True
    selection_scope: str = SOURCE_SELECTION_SCOPE

    def __post_init__(self) -> None:
        if self.selection_scope != SOURCE_SELECTION_SCOPE:
            raise AttentionMassAnchorError(
                "configuration selection_scope must be source_lodo"
            )
        if self.min_net_rescues < 0 or self.max_harms < 0:
            raise AttentionMassAnchorError("rescue/harm limits must be non-negative")
        if self.max_abs_balanced_margin_shift < 0:
            raise AttentionMassAnchorError(
                "max_abs_balanced_margin_shift must be non-negative"
            )


def _validate_visual_slice(
    attention_logits: torch.Tensor,
    image_start: int,
    image_end_exclusive: int,
) -> None:
    if not torch.is_floating_point(attention_logits):
        raise AttentionMassAnchorError("attention_logits must be floating point")
    if attention_logits.ndim < 1:
        raise AttentionMassAnchorError("attention_logits must have a key dimension")
    key_count = attention_logits.shape[-1]
    if not 0 <= image_start < image_end_exclusive <= key_count:
        raise AttentionMassAnchorError(
            "visual-key interval must be non-empty and inside the key dimension"
        )
    if image_start == 0 and image_end_exclusive == key_count:
        raise AttentionMassAnchorError(
            "at least one non-visual key is required to define attention odds"
        )


def visual_attention_log_odds(
    attention_logits: torch.Tensor,
    image_start: int,
    image_end_exclusive: int,
) -> torch.Tensor:
    """Return stable log odds of total attention assigned to visual keys.

    ``attention_logits`` may have any leading dimensions; the final dimension
    is interpreted as keys.  Causal/padding masks should already be included.
    """

    _validate_visual_slice(attention_logits, image_start, image_end_exclusive)
    visual = attention_logits[..., image_start:image_end_exclusive]
    nonvisual = torch.cat(
        (
            attention_logits[..., :image_start],
            attention_logits[..., image_end_exclusive:],
        ),
        dim=-1,
    )
    visual_lse = torch.logsumexp(visual.float(), dim=-1)
    nonvisual_lse = torch.logsumexp(nonvisual.float(), dim=-1)
    log_odds = visual_lse - nonvisual_lse
    if not torch.isfinite(log_odds).all():
        raise AttentionMassAnchorError(
            "visual and non-visual attention groups must each contain finite support"
        )
    return log_odds


def log_odds_to_mass(log_odds: torch.Tensor) -> torch.Tensor:
    """Convert attention log odds to mass without losing tail stability."""

    return torch.sigmoid(log_odds.float())


def _broadcast_head_vector(
    value: torch.Tensor | float,
    target: torch.Tensor,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=target.dtype, device=target.device)
    if result.ndim == 1 and target.ndim >= 2:
        head_count = target.shape[-2]
        if result.numel() != head_count:
            raise AttentionMassAnchorError(
                f"{name} has {result.numel()} heads, expected {head_count}"
            )
        result = result.reshape(
            (1,) * (target.ndim - 2) + (head_count, 1)
        )
    try:
        return torch.broadcast_to(result, target.shape)
    except RuntimeError as error:
        raise AttentionMassAnchorError(
            f"{name} is not broadcastable to attention queries {tuple(target.shape)}"
        ) from error


def closed_form_attention_delta(
    current_log_odds: torch.Tensor,
    center_log_odds: torch.Tensor | float,
    max_abs_delta: float,
    head_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the clipped source-center shift for every attention query."""

    if max_abs_delta < 0:
        raise AttentionMassAnchorError("max_abs_delta must be non-negative")
    center = _broadcast_head_vector(
        center_log_odds, current_log_odds, "center_log_odds"
    )
    delta = (center - current_log_odds).clamp(
        min=-float(max_abs_delta), max=float(max_abs_delta)
    )
    if head_mask is not None:
        mask = _broadcast_head_vector(head_mask, current_log_odds, "head_mask")
        delta = torch.where(mask.bool(), delta, torch.zeros_like(delta))
    return delta


def apply_visual_logit_shift(
    attention_logits: torch.Tensor,
    delta: torch.Tensor,
    image_start: int,
    image_end_exclusive: int,
) -> torch.Tensor:
    """Apply a query-wise scalar shift to every visual key."""

    _validate_visual_slice(attention_logits, image_start, image_end_exclusive)
    try:
        expanded_delta = torch.broadcast_to(
            delta.to(dtype=attention_logits.dtype, device=attention_logits.device),
            attention_logits.shape[:-1],
        )
    except RuntimeError as error:
        raise AttentionMassAnchorError(
            "delta is not broadcastable to attention queries"
        ) from error
    anchored = attention_logits.clone()
    anchored[..., image_start:image_end_exclusive] += expanded_delta.unsqueeze(-1)
    return anchored


def anchor_attention_logits(
    attention_logits: torch.Tensor,
    image_start: int,
    image_end_exclusive: int,
    center_log_odds: torch.Tensor | float,
    max_abs_delta: float,
    head_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply ANCHOR and return tensors required for an activation audit."""

    before = visual_attention_log_odds(
        attention_logits, image_start, image_end_exclusive
    )
    delta = closed_form_attention_delta(
        before,
        center_log_odds=center_log_odds,
        max_abs_delta=max_abs_delta,
        head_mask=head_mask,
    )
    anchored = apply_visual_logit_shift(
        attention_logits, delta, image_start, image_end_exclusive
    )
    after = visual_attention_log_odds(
        anchored, image_start, image_end_exclusive
    )
    return anchored, {
        "before_log_odds": before,
        "delta": delta,
        "after_log_odds": after,
        "before_mass": log_odds_to_mass(before),
        "after_mass": log_odds_to_mass(after),
    }


def robust_source_attention_center(
    domain_log_odds: Mapping[str, torch.Tensor],
    mad_multiplier: float = 2.5,
    minimum_domains: int = 2,
) -> RobustAttentionCenter:
    """Fit a label-free robust center and stable-head mask from source domains.

    Each mapping value has shape ``[samples, ..., heads]``.  A midpoint median
    is first taken over samples, followed by a midpoint median across domain
    centers.  A
    layer/head is enabled when its between-domain MAD is no larger than a
    robust, source-derived upper threshold over all layer/head MAD values.
    """

    if mad_multiplier < 0:
        raise AttentionMassAnchorError("mad_multiplier must be non-negative")
    names = tuple(sorted(domain_log_odds))
    if len(names) < minimum_domains:
        raise AttentionMassAnchorError(
            f"at least {minimum_domains} source domains are required"
        )
    domain_centers: list[torch.Tensor] = []
    feature_shape: torch.Size | None = None
    for name in names:
        values = domain_log_odds[name]
        if not torch.is_floating_point(values) or values.ndim < 2:
            raise AttentionMassAnchorError(
                f"domain {name!r} must be floating [samples, ..., heads]"
            )
        if values.shape[0] < 1 or not torch.isfinite(values).all():
            raise AttentionMassAnchorError(
                f"domain {name!r} must contain finite source samples"
            )
        if feature_shape is None:
            feature_shape = values.shape[1:]
        elif values.shape[1:] != feature_shape:
            raise AttentionMassAnchorError(
                "all source domains must share layer/head feature shape"
            )
        domain_centers.append(
            torch.quantile(
                values.float(), 0.5, dim=0, interpolation="midpoint"
            )
        )

    stacked = torch.stack(domain_centers, dim=0)
    center = torch.quantile(
        stacked, 0.5, dim=0, interpolation="midpoint"
    )
    domain_mad = torch.quantile(
        (stacked - center).abs(),
        0.5,
        dim=0,
        interpolation="midpoint",
    )
    flat_mad = domain_mad.flatten()
    global_median = torch.quantile(
        flat_mad, 0.5, interpolation="midpoint"
    )
    global_mad = torch.quantile(
        (flat_mad - global_median).abs(),
        0.5,
        interpolation="midpoint",
    )
    threshold = global_median + float(mad_multiplier) * global_mad
    tolerance = torch.finfo(domain_mad.dtype).eps * 16
    head_mask = domain_mad <= threshold + tolerance
    return RobustAttentionCenter(
        center_log_odds=center,
        domain_mad=domain_mad,
        head_mask=head_mask,
        mad_threshold=threshold,
        domain_names=names,
    )


def evaluate_source_only_gate(
    folds: Mapping[str, Mapping[str, Any]],
    config: SourceOnlyGateConfig,
    selection_scope: str,
) -> dict[str, Any]:
    """Fail closed unless a frozen source-LODO candidate passes every check.

    Required per-fold fields are ``rescues``, ``harms``, ``delta_pp``,
    ``balanced_margin_shift``, and ``rescues_by_label`` with ``yes``/``no``.
    Target labels are not an accepted input to this function.
    """

    if selection_scope != SOURCE_SELECTION_SCOPE:
        raise AttentionMassAnchorError(
            "gate metrics must come exclusively from source_lodo"
        )
    if not folds:
        raise AttentionMassAnchorError("at least one source LODO fold is required")

    required = {
        "rescues",
        "harms",
        "delta_pp",
        "balanced_margin_shift",
        "rescues_by_label",
    }
    total_rescues = 0
    total_harms = 0
    yes_rescues = 0
    no_rescues = 0
    worst_delta = float("inf")
    max_prior_shift = 0.0
    for fold_name in sorted(folds):
        metrics = folds[fold_name]
        missing = required - set(metrics)
        if missing:
            raise AttentionMassAnchorError(
                f"fold {fold_name!r} is missing {sorted(missing)}"
            )
        rescues = int(metrics["rescues"])
        harms = int(metrics["harms"])
        if rescues < 0 or harms < 0:
            raise AttentionMassAnchorError("rescue/harm counts must be non-negative")
        label_rescues = metrics["rescues_by_label"]
        if set(label_rescues) != {"yes", "no"}:
            raise AttentionMassAnchorError(
                "rescues_by_label must contain exactly yes and no"
            )
        total_rescues += rescues
        total_harms += harms
        yes_rescues += int(label_rescues["yes"])
        no_rescues += int(label_rescues["no"])
        worst_delta = min(worst_delta, float(metrics["delta_pp"]))
        max_prior_shift = max(
            max_prior_shift, abs(float(metrics["balanced_margin_shift"]))
        )

    checks = {
        "net_rescues": total_rescues - total_harms >= config.min_net_rescues,
        "harms": total_harms <= config.max_harms,
        "worst_fold_noninferiority": (
            worst_delta >= config.min_worst_fold_delta_pp
        ),
        "balanced_margin_shift": (
            max_prior_shift <= config.max_abs_balanced_margin_shift
        ),
        "both_labels_rescued": (
            not config.require_both_labels_rescued
            or (yes_rescues > 0 and no_rescues > 0)
        ),
    }
    passed = all(checks.values())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "selection_scope": SOURCE_SELECTION_SCOPE,
        "status": "passed" if passed else "failed",
        "target_falsification_allowed": passed,
        "checks": checks,
        "totals": {
            "rescues": total_rescues,
            "harms": total_harms,
            "net_rescues": total_rescues - total_harms,
            "yes_rescues": yes_rescues,
            "no_rescues": no_rescues,
            "worst_fold_delta_pp": worst_delta,
            "max_abs_balanced_margin_shift": max_prior_shift,
        },
    }
