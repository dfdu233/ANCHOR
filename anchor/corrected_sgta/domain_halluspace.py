"""Deterministic mathematics for a Fisher-protected domain halluspace.

Given paired clean/shift activations, this module models domain-sensitive
directions with the weighted covariance of their displacement.  Clinical
gradients define a Fisher metric, so the generalized eigenproblem

    Sigma_domain v = lambda (F_clinical + ridge I) v

prefers directions with large harmful domain variation and small stable-domain
residual or first-order clinical sensitivity.  Marchenko--Pastur and
permutation parallel analysis are diagnostics, not theorem-level guarantees
for neural activations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch


DEFAULT_DTYPE: Final[torch.dtype] = torch.float64


class DomainHalluspaceError(ValueError):
    """Raised when a mathematical or numerical contract is violated."""


@dataclass(frozen=True)
class WeightedCovariance:
    mean: torch.Tensor
    covariance: torch.Tensor
    total_weight: float
    effective_sample_size: float
    unbiased_denominator: float


@dataclass(frozen=True)
class RankDiagnostics:
    mp_noise_scale: float
    mp_upper_edge: float
    mp_rank: int
    mp_identifiable: bool
    parallel_thresholds: torch.Tensor
    parallel_rank: int
    selected_rank: int
    parallel_quantile: float
    parallel_repetitions: int
    seed: int
    effective_sample_size: float
    harmful_effective_image_count: float
    stable_effective_image_count: float
    aspect_ratio: float


@dataclass(frozen=True)
class DomainHalluspace:
    displacement_mean: torch.Tensor
    span_basis: torch.Tensor
    reduced_domain_covariance: torch.Tensor
    reduced_protection_metric: torch.Tensor
    eigenvalues: torch.Tensor
    basis: torch.Tensor
    dual_basis: torch.Tensor
    rank: RankDiagnostics
    shrinkage: torch.Tensor
    total_weight: float
    effective_sample_size: float
    stable_total_weight: float
    stable_effective_sample_size: float
    fisher_weight: float
    fisher_ridge: float


def _matrix(value: torch.Tensor, name: str, *, minimum_rows: int = 1) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise DomainHalluspaceError(f"{name} must be a rank-two tensor")
    if value.shape[0] < minimum_rows or value.shape[1] < 1:
        raise DomainHalluspaceError(
            f"{name} must have at least {minimum_rows} rows and one feature"
        )
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise DomainHalluspaceError(f"{name} must be finite floating point")
    return value.detach().to(device="cpu", dtype=DEFAULT_DTYPE).contiguous()


def _weights(value: torch.Tensor, sample_count: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise DomainHalluspaceError("delta_nll_weights must be a vector")
    if value.numel() != sample_count:
        raise DomainHalluspaceError("delta_nll_weights must match the sample count")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise DomainHalluspaceError("delta_nll_weights must be finite floating point")
    result = value.detach().to(device="cpu", dtype=DEFAULT_DTYPE).contiguous()
    if bool((result < 0).any()) or float(result.sum()) <= 0:
        raise DomainHalluspaceError(
            "delta_nll_weights must be non-negative with positive total weight"
        )
    return result


def weighted_covariance(
    samples: torch.Tensor, weights: torch.Tensor
) -> WeightedCovariance:
    """Return the reliability-weighted, unbiased covariance.

    The denominator is ``sum(w) - sum(w^2)/sum(w)``.  This equals ``n-1`` for
    unit weights and fails closed when fewer than two effective observations
    remain.
    """

    values = _matrix(samples, "samples", minimum_rows=2)
    importance = _weights(weights, values.shape[0])
    total = importance.sum()
    sum_squared = importance.square().sum()
    denominator = total - sum_squared / total
    effective_n = total.square() / sum_squared
    tolerance = torch.finfo(DEFAULT_DTYPE).eps * max(1.0, float(total))
    if float(denominator) <= tolerance or float(effective_n) <= 1.0:
        raise DomainHalluspaceError(
            "weighted covariance requires more than one effective observation"
        )
    mean = (importance[:, None] * values).sum(dim=0) / total
    centered = values - mean
    covariance = (centered.T * importance) @ centered / denominator
    covariance = 0.5 * (covariance + covariance.T)
    if not torch.isfinite(covariance).all():
        raise DomainHalluspaceError("weighted covariance is non-finite")
    return WeightedCovariance(
        mean=mean,
        covariance=covariance,
        total_weight=float(total),
        effective_sample_size=float(effective_n),
        unbiased_denominator=float(denominator),
    )


def _weighted_centered_factor(
    samples: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Return ``mean`` and ``C`` such that ``C.T @ C`` is the covariance."""

    values = _matrix(samples, "samples", minimum_rows=2)
    importance = _weights(weights, values.shape[0])
    total = importance.sum()
    sum_squared = importance.square().sum()
    denominator = total - sum_squared / total
    effective_n = total.square() / sum_squared
    tolerance = torch.finfo(DEFAULT_DTYPE).eps * max(1.0, float(total))
    if float(denominator) <= tolerance or float(effective_n) <= 1.0:
        raise DomainHalluspaceError(
            "weighted covariance requires more than one effective observation"
        )
    mean = (importance[:, None] * values).sum(dim=0) / total
    factor = (values - mean) * (importance / denominator).sqrt()[:, None]
    return mean, factor, float(total), float(effective_n), float(denominator)


def _stable_factor(
    samples: torch.Tensor, stable_weights: torch.Tensor | None
) -> tuple[torch.Tensor | None, float, float]:
    if stable_weights is None:
        return None, 0.0, 0.0
    importance = _weights_allow_zero(stable_weights, samples.shape[0], "stable_weights")
    if float(importance.sum()) == 0.0:
        return None, 0.0, 0.0
    _, factor, total, effective_n, _ = _weighted_centered_factor(samples, importance)
    return factor, total, effective_n


def _weights_allow_zero(
    value: torch.Tensor, sample_count: int, name: str
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise DomainHalluspaceError(f"{name} must be a vector")
    if value.numel() != sample_count:
        raise DomainHalluspaceError(f"{name} must match the sample count")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise DomainHalluspaceError(f"{name} must be finite floating point")
    result = value.detach().to(device="cpu", dtype=DEFAULT_DTYPE).contiguous()
    if bool((result < 0).any()):
        raise DomainHalluspaceError(f"{name} must be non-negative")
    return result


def _orthonormal_row_span(rows: torch.Tensor) -> torch.Tensor:
    """Compute a thin orthonormal basis without an ambient-dimensional SVD."""

    gram = rows @ rows.T
    gram = 0.5 * (gram + gram.T)
    values, vectors = torch.linalg.eigh(gram)
    largest = float(values.max()) if values.numel() else 0.0
    tolerance = (
        torch.finfo(DEFAULT_DTYPE).eps
        * max(rows.shape)
        * max(1.0, largest)
        * 16.0
    )
    keep = values > tolerance
    if not bool(keep.any()):
        raise DomainHalluspaceError("compact span is numerically empty")
    values = values[keep]
    vectors = vectors[:, keep]
    order = torch.argsort(values, descending=True, stable=True)
    basis = rows.T @ vectors[:, order] @ torch.diag(values[order].rsqrt())
    gram_basis = basis.T @ basis
    if not torch.allclose(
        gram_basis,
        torch.eye(gram_basis.shape[0], dtype=DEFAULT_DTYPE),
        atol=2e-7,
        rtol=2e-7,
    ):
        raise DomainHalluspaceError("compact span basis failed orthonormalization")
    return basis


def _compact_eigensystem(
    displacements: torch.Tensor,
    weights: torch.Tensor,
    clinical_gradients: torch.Tensor,
    fisher_ridge: float,
    stable_weights: torch.Tensor | None = None,
    fisher_weight: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    float,
]:
    """Solve the generalized problem in ``span([weighted D_c; G])``."""

    mean, factor, total, effective_n, denominator = _weighted_centered_factor(
        displacements, weights
    )
    gradients = _matrix(clinical_gradients, "clinical_gradients")
    if gradients.shape[1] != factor.shape[1]:
        raise DomainHalluspaceError("clinical gradients must match displacement dimension")
    if not 0 < fisher_ridge < float("inf"):
        raise DomainHalluspaceError("fisher_ridge must be finite and positive")
    if not 0 <= fisher_weight < float("inf"):
        raise DomainHalluspaceError("fisher_weight must be finite and non-negative")
    stable, _, _ = _stable_factor(displacements, stable_weights)
    rows = [displacements - displacements.mean(dim=0, keepdim=True)]
    if fisher_weight > 0:
        rows.append(gradients * (fisher_weight / gradients.shape[0]) ** 0.5)
    span = _orthonormal_row_span(torch.cat(rows, dim=0))
    factor_reduced = factor @ span
    gradients_reduced = gradients @ span
    covariance_reduced = factor_reduced.T @ factor_reduced
    denominator_reduced = float(fisher_ridge) * torch.eye(
        span.shape[1], dtype=DEFAULT_DTYPE
    )
    if stable is not None:
        stable_reduced = stable @ span
        denominator_reduced += stable_reduced.T @ stable_reduced
    denominator_reduced += (
        float(fisher_weight)
        * gradients_reduced.T
        @ gradients_reduced
        / gradients.shape[0]
    )
    values, reduced_vectors = generalized_eigenspace(
        covariance_reduced, denominator_reduced
    )
    spectral_size = min(displacements.shape[0] - 1, span.shape[1])
    values = values[:spectral_size]
    basis = span @ reduced_vectors[:, :spectral_size]
    dual_basis = (
        float(fisher_ridge) * basis
        + float(fisher_weight)
        * gradients.T
        @ (gradients @ basis)
        / gradients.shape[0]
    )
    if stable is not None:
        dual_basis += stable.T @ (stable @ basis)
    fisher_gram = basis.T @ dual_basis
    if not torch.allclose(
        fisher_gram,
        torch.eye(spectral_size, dtype=DEFAULT_DTYPE),
        atol=3e-7,
        rtol=3e-7,
    ):
        raise DomainHalluspaceError("compact basis failed Fisher normalization")
    return (
        values,
        basis,
        dual_basis,
        span,
        covariance_reduced,
        denominator_reduced,
        mean,
        total,
        effective_n,
        denominator,
    )


def clinical_fisher_metric(
    clinical_gradients: torch.Tensor, ridge: float
) -> torch.Tensor:
    """Construct the empirical clinical Fisher metric plus isotropic ridge."""

    gradients = _matrix(clinical_gradients, "clinical_gradients")
    if not 0 < ridge < float("inf"):
        raise DomainHalluspaceError("fisher_ridge must be finite and positive")
    fisher = gradients.T @ gradients / gradients.shape[0]
    fisher += float(ridge) * torch.eye(gradients.shape[1], dtype=DEFAULT_DTYPE)
    return 0.5 * (fisher + fisher.T)


def generalized_eigenspace(
    domain_covariance: torch.Tensor, fisher_metric: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a symmetric generalized eigensystem with deterministic signs."""

    covariance = _matrix(domain_covariance, "domain_covariance")
    fisher = _matrix(fisher_metric, "fisher_metric")
    if covariance.shape[0] != covariance.shape[1] or fisher.shape != covariance.shape:
        raise DomainHalluspaceError("covariance and Fisher metric must be square and aligned")
    symmetry_tolerance = 1e-9
    if not torch.allclose(covariance, covariance.T, atol=symmetry_tolerance, rtol=1e-7):
        raise DomainHalluspaceError("domain_covariance must be symmetric")
    if not torch.allclose(fisher, fisher.T, atol=symmetry_tolerance, rtol=1e-7):
        raise DomainHalluspaceError("fisher_metric must be symmetric")
    fisher_values, fisher_vectors = torch.linalg.eigh(fisher)
    if float(fisher_values.min()) <= 0:
        raise DomainHalluspaceError("fisher_metric must be positive definite")
    inverse_root = (
        fisher_vectors
        @ torch.diag(fisher_values.rsqrt())
        @ fisher_vectors.T
    )
    whitened = inverse_root @ covariance @ inverse_root
    whitened = 0.5 * (whitened + whitened.T)
    values, vectors = torch.linalg.eigh(whitened)
    order = torch.argsort(values, descending=True, stable=True)
    values = values[order].clamp_min(0.0)
    vectors = inverse_root @ vectors[:, order]
    for column in range(vectors.shape[1]):
        pivot = int(torch.argmax(vectors[:, column].abs()))
        if float(vectors[pivot, column]) < 0:
            vectors[:, column].neg_()
    gram = vectors.T @ fisher @ vectors
    if not torch.allclose(
        gram, torch.eye(gram.shape[0], dtype=DEFAULT_DTYPE), atol=2e-7, rtol=2e-7
    ):
        raise DomainHalluspaceError("generalized eigenvectors failed Fisher normalization")
    return values, vectors


def _positive_tail_noise_scale(eigenvalues: torch.Tensor) -> float:
    relative_tolerance = (
        torch.finfo(DEFAULT_DTYPE).eps
        * max(1, eigenvalues.numel())
        * max(1.0, float(eigenvalues.max()))
        * 16.0
    )
    positive = eigenvalues[eigenvalues > relative_tolerance]
    if positive.numel() < 2:
        return 0.0
    tail = positive[positive.numel() // 2 :]
    return float(torch.median(tail))


def _mp_diagnostic(
    spectrum: torch.Tensor, aspect_ratio: float
) -> tuple[float, float, int, bool]:
    noise_scale = _positive_tail_noise_scale(spectrum)
    identifiable = noise_scale > 0.0
    if not identifiable:
        return 0.0, 0.0, int((spectrum > 0).sum()), False
    edge = noise_scale * (1.0 + aspect_ratio**0.5) ** 2
    return noise_scale, edge, _leading_exceedance_rank(spectrum, edge), True


def _leading_exceedance_rank(
    observed: torch.Tensor, threshold: torch.Tensor | float
) -> int:
    comparisons = observed > threshold
    rank = 0
    for passed in comparisons:
        if not bool(passed):
            break
        rank += 1
    return rank


def rank_diagnostics(
    displacements: torch.Tensor,
    weights: torch.Tensor,
    fisher_metric: torch.Tensor,
    observed_eigenvalues: torch.Tensor,
    *,
    parallel_repetitions: int = 32,
    parallel_quantile: float = 0.95,
    seed: int = 0,
) -> RankDiagnostics:
    """Estimate signal rank using MP edge and deterministic parallel analysis.

    The MP noise scale uses the lower spectral half and is therefore a robust
    heuristic, not a claim that neural activations are iid Gaussian.  Parallel
    analysis independently permutes every activation feature across samples.
    The conservative selected rank is the minimum of the two diagnostics.
    """

    values = _matrix(displacements, "displacements", minimum_rows=2)
    importance = _weights(weights, values.shape[0])
    fisher = _matrix(fisher_metric, "fisher_metric")
    spectrum = observed_eigenvalues.detach().to(device="cpu", dtype=DEFAULT_DTYPE)
    if spectrum.ndim != 1 or spectrum.numel() != values.shape[1]:
        raise DomainHalluspaceError("observed_eigenvalues must match feature dimension")
    if not torch.isfinite(spectrum).all() or bool((spectrum < 0).any()):
        raise DomainHalluspaceError("observed_eigenvalues must be finite and non-negative")
    if not isinstance(parallel_repetitions, int) or parallel_repetitions < 1:
        raise DomainHalluspaceError("parallel_repetitions must be a positive integer")
    if not 0 < parallel_quantile < 1:
        raise DomainHalluspaceError("parallel_quantile must lie strictly between zero and one")
    effective_n = float(importance.sum().square() / importance.square().sum())
    aspect_ratio = values.shape[1] / effective_n
    noise_scale, mp_edge, mp_rank, mp_identifiable = _mp_diagnostic(
        spectrum, aspect_ratio
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    null_spectra = []
    for _ in range(parallel_repetitions):
        permutation = torch.argsort(
            torch.rand(
                values.shape[1], values.shape[0], generator=generator
            ),
            dim=1,
            stable=True,
        ).T
        permuted = torch.gather(values, 0, permutation)
        covariance = weighted_covariance(permuted, importance).covariance
        null_values, _ = generalized_eigenspace(covariance, fisher)
        null_spectra.append(null_values)
    thresholds = torch.quantile(
        torch.stack(null_spectra),
        parallel_quantile,
        dim=0,
        interpolation="linear",
    )
    parallel_rank = _leading_exceedance_rank(spectrum, thresholds)
    return RankDiagnostics(
        mp_noise_scale=noise_scale,
        mp_upper_edge=mp_edge,
        mp_rank=mp_rank,
        parallel_thresholds=thresholds,
        parallel_rank=parallel_rank,
        selected_rank=min(mp_rank, parallel_rank) if mp_identifiable else parallel_rank,
        parallel_quantile=parallel_quantile,
        parallel_repetitions=parallel_repetitions,
        seed=seed,
        effective_sample_size=effective_n,
        harmful_effective_image_count=effective_n,
        stable_effective_image_count=0.0,
        aspect_ratio=aspect_ratio,
        mp_identifiable=mp_identifiable,
    )


def compact_rank_diagnostics(
    displacements: torch.Tensor,
    weights: torch.Tensor,
    clinical_gradients: torch.Tensor,
    fisher_ridge: float,
    observed_eigenvalues: torch.Tensor,
    *,
    stable_weights: torch.Tensor | None = None,
    fisher_weight: float = 1.0,
    parallel_repetitions: int = 32,
    parallel_quantile: float = 0.95,
    seed: int = 0,
) -> RankDiagnostics:
    """Compact MP/parallel diagnostics with no ambient square matrices."""

    values = _matrix(displacements, "displacements", minimum_rows=2)
    importance = _weights(weights, values.shape[0])
    gradients = _matrix(clinical_gradients, "clinical_gradients")
    spectrum = observed_eigenvalues.detach().to(device="cpu", dtype=DEFAULT_DTYPE)
    if spectrum.ndim != 1 or spectrum.numel() < 1:
        raise DomainHalluspaceError("observed compact spectrum must be non-empty")
    if not torch.isfinite(spectrum).all() or bool((spectrum < 0).any()):
        raise DomainHalluspaceError("observed_eigenvalues must be finite and non-negative")
    if not isinstance(parallel_repetitions, int) or parallel_repetitions < 1:
        raise DomainHalluspaceError("parallel_repetitions must be a positive integer")
    if not 0 < parallel_quantile < 1:
        raise DomainHalluspaceError("parallel_quantile must lie strictly between zero and one")
    effective_n = float(importance.sum().square() / importance.square().sum())
    _, _, stable_effective_n = _stable_factor(values, stable_weights)
    aspect_ratio = values.shape[1] / effective_n
    noise_scale, mp_edge, mp_rank, mp_identifiable = _mp_diagnostic(
        spectrum, aspect_ratio
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    null_spectra = []
    for _ in range(parallel_repetitions):
        permutation = torch.argsort(
            torch.rand(
                values.shape[1], values.shape[0], generator=generator
            ),
            dim=1,
            stable=True,
        ).T
        permuted = torch.gather(values, 0, permutation)
        null_values = _compact_eigensystem(
            permuted,
            importance,
            gradients,
            fisher_ridge,
            stable_weights=stable_weights,
            fisher_weight=fisher_weight,
        )[0]
        padded = torch.zeros_like(spectrum)
        count = min(padded.numel(), null_values.numel())
        padded[:count] = null_values[:count]
        null_spectra.append(padded)
    thresholds = torch.quantile(
        torch.stack(null_spectra),
        parallel_quantile,
        dim=0,
        interpolation="linear",
    )
    parallel_rank = _leading_exceedance_rank(spectrum, thresholds)
    return RankDiagnostics(
        mp_noise_scale=noise_scale,
        mp_upper_edge=mp_edge,
        mp_rank=mp_rank,
        parallel_thresholds=thresholds,
        parallel_rank=parallel_rank,
        selected_rank=min(mp_rank, parallel_rank) if mp_identifiable else parallel_rank,
        parallel_quantile=parallel_quantile,
        parallel_repetitions=parallel_repetitions,
        seed=seed,
        effective_sample_size=effective_n,
        harmful_effective_image_count=effective_n,
        stable_effective_image_count=stable_effective_n,
        aspect_ratio=aspect_ratio,
        mp_identifiable=mp_identifiable,
    )


def shrinkage_projection(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    fisher_metric: torch.Tensor,
    rank: RankDiagnostics,
    *,
    strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a Fisher-orthogonal, noise-aware soft-removal projection."""

    if not 0 <= strength <= 1 or not torch.isfinite(torch.tensor(strength)):
        raise DomainHalluspaceError("shrinkage_strength must be finite in [0, 1]")
    values = eigenvalues.detach().to(device="cpu", dtype=DEFAULT_DTYPE)
    vectors = _matrix(eigenvectors, "eigenvectors")
    fisher = _matrix(fisher_metric, "fisher_metric")
    dimension = vectors.shape[0]
    if vectors.shape != (dimension, dimension) or fisher.shape != (dimension, dimension):
        raise DomainHalluspaceError("eigenspace and Fisher metric must be aligned and square")
    if values.shape != (dimension,) or rank.parallel_thresholds.shape != (dimension,):
        raise DomainHalluspaceError("spectrum and rank thresholds must match eigenspace")
    shrinkage = torch.zeros(dimension, dtype=DEFAULT_DTYPE)
    retained = rank.selected_rank
    if retained:
        signal = values[:retained]
        noise = rank.parallel_thresholds[:retained].clamp_min(0.0)
        shrinkage[:retained] = float(strength) * (1.0 - noise / signal).clamp(0.0, 1.0)
    projection = torch.eye(dimension, dtype=DEFAULT_DTYPE)
    if bool((shrinkage > 0).any()):
        projection -= vectors @ torch.diag(shrinkage) @ vectors.T @ fisher
    if not torch.isfinite(projection).all():
        raise DomainHalluspaceError("shrinkage projection is non-finite")
    return projection, shrinkage


def fit_domain_halluspace(
    clean_activations: torch.Tensor,
    shifted_activations: torch.Tensor,
    delta_nll_weights: torch.Tensor,
    clinical_gradients: torch.Tensor,
    *,
    stable_weights: torch.Tensor | None = None,
    fisher_weight: float = 1.0,
    fisher_ridge: float = 1e-4,
    parallel_repetitions: int = 32,
    parallel_quantile: float = 0.95,
    shrinkage_strength: float = 1.0,
    seed: int = 0,
) -> DomainHalluspace:
    """Fit the complete deterministic ANCHOR-Null mathematical object."""

    clean = _matrix(clean_activations, "clean_activations", minimum_rows=2)
    shifted = _matrix(shifted_activations, "shifted_activations", minimum_rows=2)
    if clean.shape != shifted.shape:
        raise DomainHalluspaceError("clean and shifted activations must be paired and aligned")
    gradients = _matrix(clinical_gradients, "clinical_gradients")
    if gradients.shape[1] != clean.shape[1]:
        raise DomainHalluspaceError("clinical gradients must match activation dimension")
    importance = _weights(delta_nll_weights, clean.shape[0])
    stable_importance = (
        torch.zeros(clean.shape[0], dtype=DEFAULT_DTYPE)
        if stable_weights is None
        else _weights_allow_zero(stable_weights, clean.shape[0], "stable_weights")
    )
    displacement = shifted - clean
    (
        eigenvalues,
        basis,
        dual_basis,
        span_basis,
        reduced_covariance,
        reduced_protection_metric,
        displacement_mean,
        total_weight,
        effective_sample_size,
        _,
    ) = _compact_eigensystem(
        displacement,
        importance,
        gradients,
        fisher_ridge,
        stable_weights=stable_importance,
        fisher_weight=fisher_weight,
    )
    _, stable_total_weight, stable_effective_sample_size = _stable_factor(
        displacement, stable_importance
    )
    diagnostics = compact_rank_diagnostics(
        displacement,
        importance,
        gradients,
        fisher_ridge,
        eigenvalues,
        stable_weights=stable_importance,
        fisher_weight=fisher_weight,
        parallel_repetitions=parallel_repetitions,
        parallel_quantile=parallel_quantile,
        seed=seed,
    )
    if not 0 <= shrinkage_strength <= 1:
        raise DomainHalluspaceError("shrinkage_strength must lie in [0, 1]")
    shrinkage = torch.zeros_like(eigenvalues)
    retained = diagnostics.selected_rank
    if retained:
        signal = eigenvalues[:retained]
        noise = diagnostics.parallel_thresholds[:retained].clamp_min(0.0)
        shrinkage[:retained] = float(shrinkage_strength) * (
            1.0 - noise / signal
        ).clamp(0.0, 1.0)
    return DomainHalluspace(
        displacement_mean=displacement_mean,
        span_basis=span_basis,
        reduced_domain_covariance=reduced_covariance,
        reduced_protection_metric=reduced_protection_metric,
        eigenvalues=eigenvalues,
        basis=basis,
        dual_basis=dual_basis,
        rank=diagnostics,
        shrinkage=shrinkage,
        total_weight=total_weight,
        effective_sample_size=effective_sample_size,
        stable_total_weight=stable_total_weight,
        stable_effective_sample_size=stable_effective_sample_size,
        fisher_weight=fisher_weight,
        fisher_ridge=fisher_ridge,
    )


def apply_halluspace_projection(
    activations: torch.Tensor,
    fitted: DomainHalluspace,
    *,
    center: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the fitted projection around an explicit (default zero) center."""

    values = _matrix(activations, "activations")
    dimension = fitted.basis.shape[0]
    if values.shape[1] != dimension:
        raise DomainHalluspaceError("activation dimension does not match projection")
    if center is None:
        origin = torch.zeros(dimension, dtype=DEFAULT_DTYPE)
    else:
        if not isinstance(center, torch.Tensor) or center.shape != (dimension,):
            raise DomainHalluspaceError("center must be a feature vector")
        if not torch.is_floating_point(center) or not torch.isfinite(center).all():
            raise DomainHalluspaceError("center must be finite floating point")
        origin = center.detach().to(device="cpu", dtype=DEFAULT_DTYPE)
    centered = values - origin
    coefficients = centered @ fitted.dual_basis
    correction = (coefficients * fitted.shrinkage) @ fitted.basis.T
    return origin + centered - correction
