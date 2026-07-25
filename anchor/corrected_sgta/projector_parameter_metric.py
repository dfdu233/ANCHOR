"""Pullback geometry of the projected token mean for a frozen LLaVA projector.

For raw CLIP tokens z_t and projector pi, a uniform raw-token shift changes
the projected mean g(delta)=mean_t pi(z_t+delta).  Its exact local Jacobian is
Jbar=mean_t J_pi(z_t).  This module applies M=Jbar^T Jbar without materializing
Jbar.  Source support means provide locations only; they are not claimed to be
the checkpoint's unidentified training-distribution mean.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch import nn


EPS = 1e-10


def gelu_derivative(values: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Derivative of PyTorch GELU for its two supported approximations."""

    x = values
    if approximate == "none":
        inv_sqrt_two = 2.0**-0.5
        inv_sqrt_two_pi = (2.0 * torch.pi) ** -0.5
        return 0.5 * (1.0 + torch.erf(x * inv_sqrt_two)) + (
            x * torch.exp(-0.5 * x.square()) * inv_sqrt_two_pi
        )
    if approximate == "tanh":
        coefficient = (2.0 / torch.pi) ** 0.5
        inner = coefficient * (x + 0.044715 * x.pow(3))
        tangent = torch.tanh(inner)
        inner_derivative = coefficient * (1.0 + 3.0 * 0.044715 * x.square())
        return 0.5 * (1.0 + tangent) + 0.5 * x * (
            1.0 - tangent.square()
        ) * inner_derivative
    raise ValueError(f"unsupported GELU approximation: {approximate!r}")


def projector_layers(projector: nn.Module) -> tuple[nn.Linear, nn.GELU, nn.Linear]:
    """Validate and return the exact LLaVA Linear-GELU-Linear projector."""

    if not isinstance(projector, nn.Sequential) or len(projector) != 3:
        raise TypeError("parameter metric requires a 3-module nn.Sequential projector")
    first, activation, second = projector
    if not isinstance(first, nn.Linear):
        raise TypeError("projector module 0 must be nn.Linear")
    if not isinstance(activation, nn.GELU):
        raise TypeError("projector module 1 must be nn.GELU")
    if not isinstance(second, nn.Linear):
        raise TypeError("projector module 2 must be nn.Linear")
    if first.out_features != second.in_features:
        raise ValueError("incompatible projector hidden dimensions")
    return first, activation, second


def mean_activation_derivative(
    projector: nn.Module, tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return W1, W2, and mean token-wise GELU derivative."""

    first, activation, second = projector_layers(projector)
    values = tokens.to(device=first.weight.device, dtype=torch.float32)
    if values.ndim != 2 or values.shape[1] != first.in_features:
        raise ValueError(
            f"raw tokens must have shape [T,{first.in_features}], got {tuple(values.shape)}"
        )
    weight_one = first.weight.float()
    bias_one = (
        first.bias.float()
        if first.bias is not None
        else torch.zeros(first.out_features, device=values.device)
    )
    preactivation = values @ weight_one.T + bias_one
    mean_derivative = gelu_derivative(
        preactivation, activation.approximate
    ).mean(dim=0)
    return weight_one, second.weight.float(), mean_derivative


def projected_mean_jvp(
    projector: nn.Module,
    tokens: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Return Jbar times vector for the exact local projected-mean Jacobian."""

    weight_one, weight_two, derivative = mean_activation_derivative(
        projector, tokens
    )
    value = vector.to(device=weight_one.device, dtype=torch.float32)
    return weight_two @ (derivative * (weight_one @ value))


def pullback_direction(
    projector: nn.Module,
    tokens: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Return Jbar^T Jbar residual for the projected-mean geometry."""

    weight_one, weight_two, derivative = mean_activation_derivative(
        projector, tokens
    )
    value = residual.to(device=weight_one.device, dtype=torch.float32)
    projected = weight_two @ (derivative * (weight_one @ value))
    return weight_one.T @ (derivative * (weight_two.T @ projected))


def metric_projection_from_factors(
    pooled: torch.Tensor,
    target: torch.Tensor,
    weight_one: torch.Tensor,
    weight_two: torch.Tensor,
    derivative: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Closed-form metric step using one already-computed Jbar factorization."""

    center = target.to(device=pooled.device, dtype=torch.float32)
    residual = center - pooled
    projected = weight_two @ (derivative * (weight_one @ residual))
    direction = weight_one.T @ (derivative * (weight_two.T @ projected))
    denominator = direction.square().sum()
    numerator = torch.dot(residual, direction)
    if not torch.isfinite(denominator) or float(denominator) <= EPS:
        delta = torch.zeros_like(residual)
        alpha = torch.zeros((), device=pooled.device)
    else:
        alpha = numerator / denominator
        delta = alpha * direction
    diagnostics = source_diagnostics(pooled, center, delta)
    diagnostics.update(
        {
            "alpha": float(alpha.detach().cpu()),
            "metric_direction_norm": float(direction.norm().detach().cpu()),
            "projection_identity_error": float(
                (torch.dot(residual, delta) - delta.square().sum())
                .abs()
                .detach()
                .cpu()
            ),
        }
    )
    return delta, diagnostics


def source_diagnostics(
    pooled: torch.Tensor,
    target: torch.Tensor,
    delta: torch.Tensor,
) -> dict[str, float]:
    """Measure actual raw-space source closure for an applied finite step."""

    center = target.to(device=pooled.device, dtype=torch.float32)
    residual = center - pooled.float()
    shift = delta.float()
    before = residual.norm()
    after = (residual - shift).norm()
    return {
        "raw_l2_dose": float(shift.norm().detach().cpu()),
        "source_l2_before": float(before.detach().cpu()),
        "source_l2_after": float(after.detach().cpu()),
        "source_l2_closure": float((before - after).detach().cpu()),
    }


def metric_projection_delta(
    projector: nn.Module,
    tokens: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the optimal one-dimensional step along Jbar^T Jbar R."""

    pooled = tokens.float().mean(dim=0)
    weight_one, weight_two, derivative = mean_activation_derivative(
        projector, tokens
    )
    return metric_projection_from_factors(
        pooled,
        target,
        weight_one,
        weight_two,
        derivative,
    )


def equal_dose(vector: torch.Tensor, dose: torch.Tensor | float) -> torch.Tensor:
    """Normalize vector to a fixed L2 dose, failing closed at zero norm."""

    norm = vector.float().norm()
    target_dose = torch.as_tensor(dose, device=vector.device, dtype=torch.float32)
    if float(norm) <= EPS or float(target_dose) <= EPS:
        return torch.zeros_like(vector, dtype=torch.float32)
    return vector.float() * (target_dose / norm)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.float().norm() * right.float().norm()
    if float(denominator) <= EPS:
        return 0.0
    return float(
        (torch.dot(left.float(), right.float()) / denominator).detach().cpu()
    )


@dataclass
class ShiftRecord:
    arm: str
    diagnostics: dict[str, float]


class ParameterMetricTransport:
    """Inject one uniform raw-token shift before the frozen projector."""

    def __init__(self, projector: nn.Module, centers: dict[str, np.ndarray]):
        projector_layers(projector)
        self.projector = projector
        first, _, _ = projector_layers(projector)
        self.centers = {
            name: torch.as_tensor(value, dtype=torch.float32, device=first.weight.device)
            for name, value in centers.items()
        }
        for name, value in self.centers.items():
            if value.ndim != 1 or value.numel() != first.in_features:
                raise ValueError(
                    f"center {name!r} has shape {tuple(value.shape)}; "
                    f"expected ({first.in_features},)"
                )
        if set(self.centers) != {"xray", "ct", "mri"}:
            raise ValueError("centers must contain exactly xray, ct, and mri")
        self.last_record: ShiftRecord | None = None
        self._frozen_tokens: torch.Tensor | None = None
        self._frozen_shifts: dict[str, tuple[torch.Tensor, dict]] | None = None

    def reset_sample(self) -> None:
        """Clear sample-level geometry before processing a new image/question."""

        self._frozen_tokens = None
        self._frozen_shifts = None
        self.last_record = None

    def token_drift(self, tokens: torch.Tensor) -> dict[str, float]:
        """Compare a repeated vision forward with the frozen sample tokens."""

        if self._frozen_tokens is None:
            return {"token_max_abs_drift": 0.0, "token_relative_l2_drift": 0.0}
        if self._frozen_tokens.shape != tokens.shape:
            raise RuntimeError(
                "raw token shape changed within one sample: "
                f"{tuple(self._frozen_tokens.shape)} -> {tuple(tokens.shape)}"
            )
        difference = tokens.float() - self._frozen_tokens.float()
        max_abs = float(difference.abs().max().detach().cpu())
        relative = float(
            (
                difference.norm()
                / self._frozen_tokens.float().norm().clamp_min(EPS)
            )
            .detach()
            .cpu()
        )
        if max_abs > 1e-3:
            raise RuntimeError(
                f"raw token drift exceeds preregistered 1e-3 threshold: {max_abs}"
            )
        return {
            "token_max_abs_drift": max_abs,
            "token_relative_l2_drift": relative,
        }

    def shifts(self, tokens: torch.Tensor) -> dict[str, tuple[torch.Tensor, dict]]:
        """Return source directions and controls at one common raw L2 dose."""

        if self._frozen_tokens is not None and self._frozen_shifts is not None:
            self.token_drift(tokens)
            return {
                arm: (delta, dict(diagnostics))
                for arm, (delta, diagnostics) in self._frozen_shifts.items()
            }
        pooled = tokens.float().mean(dim=0)
        weight_one, weight_two, derivative = mean_activation_derivative(
            self.projector, tokens
        )
        optimal = {}
        optimal_diagnostics = {}
        for modality in ("xray", "ct", "mri"):
            delta, diagnostics = metric_projection_from_factors(
                pooled,
                self.centers[modality],
                weight_one,
                weight_two,
                derivative,
            )
            optimal[modality] = delta
            optimal_diagnostics[modality] = diagnostics
        common_dose = min(delta.norm() for delta in optimal.values())
        matched = equal_dose(optimal["xray"], common_dose)
        ct = equal_dose(optimal["ct"], common_dose)
        mri = equal_dose(optimal["mri"], common_dose)
        euclidean = equal_dose(self.centers["xray"] - pooled, common_dose)
        applied = {
            "metric_matched": (matched, "xray"),
            "euclidean_matched": (euclidean, "xray"),
            "metric_wrong_ct": (ct, "ct"),
            "metric_wrong_mri": (mri, "mri"),
            "away": (-matched, "xray"),
        }
        output = {}
        for arm, (delta, target_name) in applied.items():
            diagnostics = source_diagnostics(
                pooled, self.centers[target_name], delta
            )
            diagnostics.update(
                {
                    "common_raw_l2_dose": float(common_dose.detach().cpu()),
                    "optimal_raw_l2_dose": float(
                        optimal[target_name].norm().detach().cpu()
                    ),
                    "optimal_projection_identity_error": optimal_diagnostics[
                        target_name
                    ]["projection_identity_error"],
                    "source_target": target_name,
                }
            )
            output[arm] = (delta, diagnostics)
        self._frozen_tokens = tokens.detach().clone()
        self._frozen_shifts = output
        return {
            arm: (delta, dict(diagnostics))
            for arm, (delta, diagnostics) in output.items()
        }

    @contextmanager
    def apply(self, arm: str) -> Iterator[None]:
        """Temporarily apply an arm by hooking only the projector input."""

        original_forward = self.projector.forward
        self.last_record = None

        def hooked(features: torch.Tensor, *args, **kwargs):
            if features.ndim != 3 or features.shape[0] != 1:
                raise RuntimeError(
                    "parameter-metric probe requires raw tokens shaped [1,T,D]"
                )
            tokens = features[0].float()
            shifts = self.shifts(tokens)
            drift = self.token_drift(tokens)
            if arm not in shifts:
                raise KeyError(f"unknown parameter-metric arm: {arm}")
            delta, diagnostics = shifts[arm]
            target_name = str(diagnostics["source_target"])
            actual_source = source_diagnostics(
                tokens.mean(dim=0),
                self.centers[target_name],
                delta,
            )
            actual_source["actual_raw_l2_dose"] = actual_source.pop(
                "raw_l2_dose"
            )
            actual_source["raw_l2_dose"] = diagnostics[
                "common_raw_l2_dose"
            ]
            diagnostics.update(actual_source)
            diagnostics.update(drift)
            baseline = original_forward(features, *args, **kwargs)
            shifted = features + delta.to(features.dtype).view(1, 1, -1)
            result = original_forward(shifted, *args, **kwargs)
            actual_delta = (
                result.float().mean(dim=1) - baseline.float().mean(dim=1)
            )[0]
            predicted_delta = projected_mean_jvp(
                self.projector, tokens, delta
            )
            error = actual_delta - predicted_delta
            predicted_norm = predicted_delta.norm()
            actual_norm = actual_delta.norm()
            diagnostics.update(
                {
                    "raw_token_count": int(features.shape[1]),
                    "actual_projected_mean_delta_norm": float(
                        actual_norm.detach().cpu()
                    ),
                    "first_order_jbar_delta_norm": float(
                        predicted_norm.detach().cpu()
                    ),
                    "first_order_absolute_error": float(
                        error.norm().detach().cpu()
                    ),
                    "first_order_relative_error": float(
                        (error.norm() / predicted_norm.clamp_min(EPS))
                        .detach()
                        .cpu()
                    ),
                    "first_order_cosine": cosine(
                        actual_delta, predicted_delta
                    ),
                    "actual_over_predicted_norm": float(
                        (actual_norm / predicted_norm.clamp_min(EPS))
                        .detach()
                        .cpu()
                    ),
                }
            )
            self.last_record = ShiftRecord(arm=arm, diagnostics=diagnostics)
            return result

        self.projector.forward = hooked
        try:
            yield
        finally:
            self.projector.forward = original_forward
