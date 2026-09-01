"""Architecture-neutral factorial controls for CECD's four-cell orbit.

These pure NumPy kernels define only the unambiguous 2 x 2 representation
arithmetic.  They do not install model hooks, average autoregressive strings or
claim that an activation edit is a paper-native baseline.  A real adapter must
separately establish token alignment, hook placement and output conformance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


VERSION = "cecd-dual-semantics-factorial-kernels-v1"
IMPLEMENTED_KERNEL_METHODS = (
    "unmitigated",
    "full_orbit",
    "render_only",
    "prompt_only",
    "random_norm",
    "sign_permuted",
    "main_effect_removal",
)


class FactorialKernelError(ValueError):
    """Raised when an orbit or control violates the frozen arithmetic."""


@dataclass(frozen=True)
class FactorialComponents:
    grand: np.ndarray
    render: np.ndarray
    prompt: np.ndarray
    interaction: np.ndarray


def _validated_orbit(orbit: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    expected = {"h00", "h10", "h01", "h11"}
    if not isinstance(orbit, Mapping) or set(orbit) != expected:
        raise FactorialKernelError("orbit must contain exactly h00, h10, h01 and h11")
    values = {key: np.asarray(orbit[key], dtype=np.float64) for key in expected}
    shapes = {value.shape for value in values.values()}
    if len(shapes) != 1 or next(iter(shapes)) == ():
        raise FactorialKernelError("four orbit cells must have one equal non-scalar shape")
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FactorialKernelError("orbit cells must be finite")
    return values


def factorial_components(orbit: Mapping[str, np.ndarray]) -> FactorialComponents:
    """Return the orthogonal Walsh/ANOVA components of a balanced 2 x 2 orbit."""

    h = _validated_orbit(orbit)
    return FactorialComponents(
        grand=(h["h00"] + h["h10"] + h["h01"] + h["h11"]) / 4.0,
        render=(-h["h00"] + h["h10"] - h["h01"] + h["h11"]) / 4.0,
        prompt=(-h["h00"] - h["h10"] + h["h01"] + h["h11"]) / 4.0,
        interaction=(h["h00"] - h["h10"] - h["h01"] + h["h11"]) / 4.0,
    )


def reconstruct_cells(components: FactorialComponents) -> dict[str, np.ndarray]:
    g, r, p, interaction = (
        components.grand,
        components.render,
        components.prompt,
        components.interaction,
    )
    return {
        "h00": g - r - p + interaction,
        "h10": g + r - p - interaction,
        "h01": g - r + p - interaction,
        "h11": g + r + p + interaction,
    }


def _orthogonal_random_like(values: np.ndarray, seed: int) -> np.ndarray:
    """Match interaction norm on the last axis with a deterministic random control."""

    if values.shape[-1] < 2:
        raise FactorialKernelError("random_norm requires hidden dimension >= 2")
    generator = np.random.default_rng(int(seed))
    random = generator.standard_normal(values.shape)
    target_norm = np.linalg.norm(values, axis=-1, keepdims=True)
    squared = np.sum(values * values, axis=-1, keepdims=True)
    projection = np.divide(
        np.sum(random * values, axis=-1, keepdims=True),
        squared,
        out=np.zeros_like(squared),
        where=squared > 0,
    )
    random = random - projection * values
    random_norm = np.linalg.norm(random, axis=-1, keepdims=True)
    # A zero target carries no interaction energy. A degenerate random draw is
    # a contract error rather than a reason to change the seed post hoc.
    if np.any((target_norm > 0) & (random_norm <= 1e-12)):
        raise FactorialKernelError("random_norm produced a degenerate orthogonal direction")
    return np.divide(
        random,
        random_norm,
        out=np.zeros_like(random),
        where=random_norm > 0,
    ) * target_norm


def _permuted_like(values: np.ndarray, seed: int) -> np.ndarray:
    """Permute signed coordinates independently on each last-axis vector."""

    generator = np.random.default_rng(int(seed))
    flat = values.reshape(-1, values.shape[-1])
    output = np.empty_like(flat)
    for index, row in enumerate(flat):
        output[index] = row[generator.permutation(row.size)]
    return output.reshape(values.shape)


def apply_factorial_control(
    orbit: Mapping[str, np.ndarray], method: str, *, seed: int = 42
) -> np.ndarray:
    """Apply one frozen control to the joint target cell ``h11``.

    ``render_only`` averages the render axis at the target prompt and therefore
    equals ``grand + prompt``. ``prompt_only`` averages the prompt axis at the
    target render and equals ``grand + render``. ``main_effect_removal`` keeps
    only grand and interaction components. Random controls replace, rather than
    add to, the interaction while preserving both main effects.
    """

    if method not in IMPLEMENTED_KERNEL_METHODS:
        raise FactorialKernelError(f"factorial kernel is not implemented: {method}")
    parts = factorial_components(orbit)
    additive_target = parts.grand + parts.render + parts.prompt
    if method == "unmitigated":
        return reconstruct_cells(parts)["h11"].copy()
    if method == "full_orbit":
        return parts.grand.copy()
    if method == "render_only":
        return (parts.grand + parts.prompt).copy()
    if method == "prompt_only":
        return (parts.grand + parts.render).copy()
    if method == "main_effect_removal":
        return (parts.grand + parts.interaction).copy()
    if method == "random_norm":
        return additive_target + _orthogonal_random_like(parts.interaction, seed)
    if method == "sign_permuted":
        return additive_target + _permuted_like(parts.interaction, seed)
    raise AssertionError(method)


def synthetic_adapter_conformance(seed: int = 42) -> dict[str, object]:
    """Exercise every kernel on an orbit with known nonzero components."""

    generator = np.random.default_rng(int(seed))
    shape = (3, 16)
    components = FactorialComponents(
        grand=generator.normal(size=shape),
        render=generator.normal(size=shape),
        prompt=generator.normal(size=shape),
        interaction=generator.normal(size=shape),
    )
    orbit = reconstruct_cells(components)
    recovered = factorial_components(orbit)
    reconstruction_error = max(
        float(np.max(np.abs(orbit[key] - reconstruct_cells(recovered)[key])))
        for key in orbit
    )
    random_residual = apply_factorial_control(orbit, "random_norm", seed=seed) - (
        recovered.grand + recovered.render + recovered.prompt
    )
    permuted_residual = apply_factorial_control(
        orbit, "sign_permuted", seed=seed
    ) - (recovered.grand + recovered.render + recovered.prompt)
    interaction_norm = np.linalg.norm(recovered.interaction, axis=-1)
    random_norm_error = float(
        np.max(np.abs(np.linalg.norm(random_residual, axis=-1) - interaction_norm))
    )
    permuted_norm_error = float(
        np.max(np.abs(np.linalg.norm(permuted_residual, axis=-1) - interaction_norm))
    )
    random_dot = np.sum(random_residual * recovered.interaction, axis=-1)
    determinism_error = float(
        np.max(
            np.abs(
                apply_factorial_control(orbit, "random_norm", seed=seed)
                - apply_factorial_control(orbit, "random_norm", seed=seed)
            )
        )
    )
    expected = {
        "unmitigated": orbit["h11"],
        "full_orbit": recovered.grand,
        "render_only": (orbit["h01"] + orbit["h11"]) / 2.0,
        "prompt_only": (orbit["h10"] + orbit["h11"]) / 2.0,
        "main_effect_removal": recovered.grand + recovered.interaction,
    }
    formula_error = max(
        float(np.max(np.abs(apply_factorial_control(orbit, method) - target)))
        for method, target in expected.items()
    )
    tolerance = 1e-10
    passed = bool(
        reconstruction_error <= tolerance
        and formula_error <= tolerance
        and random_norm_error <= tolerance
        and permuted_norm_error <= tolerance
        and float(np.max(np.abs(random_dot))) <= tolerance
        and determinism_error == 0.0
    )
    return {
        "version": VERSION,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "seed": int(seed),
        "shape": list(shape),
        "methods": list(IMPLEMENTED_KERNEL_METHODS),
        "reconstruction_max_abs_error": reconstruction_error,
        "closed_form_max_abs_error": formula_error,
        "random_norm_max_abs_error": random_norm_error,
        "sign_permuted_norm_max_abs_error": permuted_norm_error,
        "random_orthogonality_max_abs_dot": float(np.max(np.abs(random_dot))),
        "determinism_max_abs_error": determinism_error,
        "scientific_model_output": False,
        "gpu_used": False,
    }
