#!/usr/bin/env python3
"""Outcome-blind, non-authorizing CECD v4 Phase-2 Haar substrate.

This module freezes a deterministic 4,096-reference antithetic schedule in
the centered render-by-prompt interaction subspace.  Each independent Haar
draw produces ``H`` and ``-H``; paired model orbits receive the same left and
right orthogonal factors.  The construction preserves the interaction
singular spectrum but is only a stress reference.  It is not, and must not be
reported as, a randomization test.

The optional 8,192-reference path is a single pre-registered precision
doubling.  It is legal only when the antithetic-pair MC-SE from the complete
initial 4,096 references strictly exceeds its frozen fraction of a stable B0.
No clinical outcome is inspected while constructing the substrate, and no
artifact or audit returned here can authorize analysis or a scientific claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .analyze_clinical_equivalence_composition_defect_v1 import (
    ContractError as V1ContractError,
    build_orbits,
    sha256_file,
    validate_payload,
)
from .validate_cecd_v4_promotion_phase1_v1 import (
    EXPECTED_MODELS,
    Phase1Build,
    VERSION as PHASE1_VERSION,
    continuous_multiplier_trace,
    object_sha256,
)


VERSION = "cecd-v4-haar-reference-phase2-v1"
CONFIG_VERSION = "cecd-v4-haar-reference-phase2-contract-v1"
STATUS = "outcome_blind_non_authorizing_phase2_reference_only"
REFERENCE_FAMILY = "centered_subspace_spectral_haar_antithetic_stress_reference"
INFERENCE_BOUNDARY = "stress_reference_not_randomization_test"
MACRO_STATISTIC = "equal_16_stratum_model_brier_pael_integral_v1"


class Phase2ReferenceError(ValueError):
    """The frozen Phase-2 reference or precision contract is invalid."""


@dataclass(frozen=True)
class HaarReferencePlan:
    """Small auditable schedule; no real reference outcomes are materialized."""

    artifact: dict[str, Any]
    binding_sha256: str
    shared_orbit_keys: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class HaarGenerationLedger:
    """Replayable Phase-2 generation record, never an inference artifact."""

    artifact: dict[str, Any]
    records: tuple[dict[str, Any], ...]


@runtime_checkable
class ReferencePairEvaluator(Protocol):
    """Injectable evaluator whose declared contract and outputs are replayed.

    A formal analyzer should put its implementation/source hash, statistic
    definition, and calibrator semantics in ``contract``.  The ledger does not
    trust a caller-supplied aggregate: it invokes this evaluator once per
    generated model/draw/sign/orbit reference and invokes it again during
    validation.
    """

    @property
    def contract(self) -> Mapping[str, Any]: ...

    def evaluate_reference(
        self, *, reference_matrix: np.ndarray, model: str,
        shared_orbit: tuple[str, str], independent_draw: int, sign: str,
        calibrator_content: Mapping[str, Any],
        multiplier_trace: Mapping[str, Any],
    ) -> float: ...


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _module_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _exact_int(mapping: Mapping[str, Any], key: str, expected: int) -> None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise Phase2ReferenceError(f"{key} must equal frozen value {expected}")


def validate_phase2_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on every design choice that controls reference precision."""

    if config.get("schema_version") != CONFIG_VERSION:
        raise Phase2ReferenceError(f"schema_version must be {CONFIG_VERSION}")
    if config.get("scientific_status") != STATUS or config.get("authorized") is not False:
        raise Phase2ReferenceError("Phase 2 must remain explicitly non-authorizing")
    if config.get("frozen_before_outputs") is not True:
        raise Phase2ReferenceError("Phase 2 config was not frozen before outputs")
    if config.get("phase1_version") != PHASE1_VERSION:
        raise Phase2ReferenceError("Phase 1 version binding drift")
    if config.get("reference_family") != REFERENCE_FAMILY:
        raise Phase2ReferenceError("reference family drift")
    if config.get("inference_boundary") != INFERENCE_BOUNDARY:
        raise Phase2ReferenceError(
            "Haar must remain a stress reference, not a randomization test"
        )
    if config.get("shared_orbit_key") != ["image_id", "finding"]:
        raise Phase2ReferenceError("shared orbit key drift")
    if config.get("same_haar_factors_for_paired_models") is not True:
        raise Phase2ReferenceError("paired models must share Haar factors")
    for key, expected in (
        ("initial_independent_draws", 2048),
        ("antithetic_references_per_draw", 2),
        ("initial_reference_count", 4096),
        ("maximum_independent_draws", 4096),
        ("maximum_reference_count", 8192),
    ):
        _exact_int(config, key, expected)
    doubling = config.get("doubling")
    expected_doubling = {
        "allowed_once": True,
        "decision_uses_only_initial_4096": True,
        "trigger_metric": "antithetic_pair_mean_mcse_divided_by_B0",
        "trigger_if_strictly_greater_than": 0.005,
        "early_stopping_allowed": False,
        "further_doubling_allowed": False,
    }
    if doubling != expected_doubling:
        raise Phase2ReferenceError("single-doubling MC-SE contract drift")
    b0 = config.get("b0_stability")
    expected_b0 = {
        "absolute_floor": 0.0001,
        "bootstrap_lower_quantile": 0.01,
        "minimum_lower_quantile_over_point": 0.1,
        "minimum_bootstrap_denominators": 100,
    }
    if b0 != expected_b0:
        raise Phase2ReferenceError("near-zero B0 stability contract drift")
    numerics = config.get("numerics")
    expected_numerics = {
        "dtype": "float64",
        "center_basis": "fixed_helmert_v1",
        "centered_sum_absolute_tolerance": 1e-09,
        "ispectral_relative_tolerance": 1e-09,
    }
    if numerics != expected_numerics:
        raise Phase2ReferenceError("Haar numerical contract drift")
    return dict(config)


def _verify_phase1_build(phase1: Phase1Build) -> None:
    artifact = phase1.artifact
    if artifact.get("version") != PHASE1_VERSION:
        raise Phase2ReferenceError("wrong Phase 1 artifact version")
    if artifact.get("authorized") is not False or artifact.get("haar_implemented") is not False:
        raise Phase2ReferenceError("Phase 1 artifact crossed its non-authorizing scope")
    claimed = artifact.get("artifact_sha256")
    unsealed = dict(artifact)
    unsealed.pop("artifact_sha256", None)
    if not isinstance(claimed, str) or object_sha256(unsealed) != claimed:
        raise Phase2ReferenceError("Phase 1 artifact seal mismatch")
    if artifact.get("cluster_identity", {}).get("cluster_order_sha256") != object_sha256(
        list(phase1.cluster_order)
    ):
        raise Phase2ReferenceError("Phase 1 cluster order mismatch")
    matrix = np.asarray(phase1.multipliers)
    digest = hashlib.sha256(np.asarray(matrix, dtype="<f8").tobytes(order="C")).hexdigest()
    if artifact.get("shared_multiplier_plan", {}).get("multiplier_sha256") != digest:
        raise Phase2ReferenceError("Phase 1 shared multiplier matrix mismatch")
    expected_trace = continuous_multiplier_trace(matrix)
    if artifact.get("shared_multiplier_plan", {}).get(
        "continuous_multiplier_trace"
    ) != expected_trace:
        raise Phase2ReferenceError("Phase 1 continuous multiplier trace mismatch")
    if not np.isfinite(matrix).all() or np.any(matrix <= 0):
        raise Phase2ReferenceError("Phase 1 multipliers are not strictly positive")


def _confirmation_shared_orbits(
    payload: Mapping[str, Any], phase1: Phase1Build,
) -> tuple[tuple[str, str], ...]:
    try:
        contract = validate_payload(payload)
    except V1ContractError as error:
        raise Phase2ReferenceError(f"confirmation payload invalid: {error}") from error
    if contract["split"] != "confirmation_locked" or contract["source_manifest_split"] != "confirmation":
        raise Phase2ReferenceError("Phase 2 requires confirmation_locked/confirmation")
    by_model: dict[str, set[tuple[str, str, int]]] = {
        model: set() for model in EXPECTED_MODELS
    }
    for orbit in build_orbits(contract):
        model = str(orbit["model"])
        if model not in by_model:
            raise Phase2ReferenceError("confirmation contains an unexpected model")
        image = str(orbit["image_id"])
        if image not in phase1.image_to_cluster:
            raise Phase2ReferenceError(f"image {image} is absent from Phase 1 cluster map")
        by_model[model].add((image, str(orbit["finding"]), int(orbit["reader_votes"])))
    if by_model[EXPECTED_MODELS[0]] != by_model[EXPECTED_MODELS[1]]:
        raise Phase2ReferenceError("paired model shared-orbit sets differ")
    stratified_keys = tuple(sorted(by_model[EXPECTED_MODELS[0]]))
    if not stratified_keys:
        raise Phase2ReferenceError("confirmation shared-orbit set is empty")
    expected_hash = phase1.artifact.get("closure", {}).get("orbit_set_sha256", {}).get(
        "confirmation_locked"
    )
    phase2_hash = object_sha256(sorted(list(stratified_keys)))
    if phase2_hash != expected_hash:
        raise Phase2ReferenceError("confirmation orbit set differs from Phase 1 binding")
    # Reader votes remain in the exact pairing/stratum audit above, but are
    # deliberately absent from the RNG key.  Relabeling votes without changing
    # the pre-output image/finding design must not change finite Haar draws.
    return tuple(sorted({(image, finding) for image, finding, _ in stratified_keys}))


def build_reference_plan(
    phase1: Phase1Build, confirmation_payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> HaarReferencePlan:
    """Freeze the keyed schedule without reading losses or making references."""

    frozen = validate_phase2_config(config)
    _verify_phase1_build(phase1)
    shared = _confirmation_shared_orbits(confirmation_payload, phase1)
    config_sha256 = object_sha256(frozen)
    orbit_sha256 = object_sha256(sorted(list(shared)))
    cluster_map_sha256 = phase1.artifact["cluster_identity"]["image_to_cluster_sha256"]
    rng_design_identity = phase1.artifact["rng_design_identity_sha256"]
    basis_hashes = {
        "render_5": _array_sha256(_center_basis(5)),
        "prompt_3": _array_sha256(_center_basis(3)),
    }
    # Deliberately exclude Phase-1 payload/artifact hashes and reader-vote
    # labels.  Those bind observed scores/targets.  RNG is a pre-output design
    # choice, while stratified closure is audited separately above.
    binding = object_sha256({
        "version": VERSION,
        "phase2_config_sha256": config_sha256,
        "pre_output_rng_design_identity_sha256": rng_design_identity,
        "unlabeled_image_finding_set_sha256": orbit_sha256,
        "cluster_map_sha256": cluster_map_sha256,
        "fixed_helmert_basis_sha256": basis_hashes,
    })
    artifact = {
        "version": VERSION,
        "status": STATUS,
        "authorized": False,
        "authorizer_implemented": False,
        "real_results_inspected": False,
        "gpu_required": False,
        "source_sha256": _module_sha256(),
        "phase1_artifact_sha256": phase1.artifact["artifact_sha256"],
        "phase1_binding_sha256": phase1.artifact["binding_sha256"],
        "phase1_rng_design_identity_sha256": rng_design_identity,
        "phase1_continuous_multiplier_trace": phase1.artifact[
            "shared_multiplier_plan"
        ]["continuous_multiplier_trace"],
        "phase2_config_sha256": config_sha256,
        "binding_sha256": binding,
        "seed_binding_excludes_scores_logits_losses_and_reader_votes": True,
        "reference_family": REFERENCE_FAMILY,
        "inference_boundary": (
            "singular-spectrum-preserving centered-subspace stress reference; "
            "not a randomization test and no Haar p-value is permitted"
        ),
        "shared_orbits": {
            "key": list(frozen["shared_orbit_key"]),
            "count": len(shared),
            "set_sha256": orbit_sha256,
            "same_factors_for_both_models": True,
            "model_count": len(EXPECTED_MODELS),
            "reader_votes_excluded_from_seed_but_retained_in_pairing_audit": True,
            "stratified_pairing_audit_key": [
                "image_id", "finding", "reader_votes",
            ],
            "stratified_pairing_set_sha256": phase1.artifact["closure"][
                "orbit_set_sha256"
            ]["confirmation_locked"],
        },
        "fixed_center_basis": {
            "family": "fixed_helmert_v1",
            "basis_sha256": basis_hashes,
        },
        "schedule": {
            "initial_independent_draws": 2048,
            "antithetic_references_per_draw": 2,
            "initial_reference_count": 4096,
            "reserved_additional_independent_draws": 2048,
            "maximum_reference_count": 8192,
            "seed_key": "sha256(binding|shared_orbit|independent_draw)",
            "generator": "numpy.PCG64",
            "antithetic_order": ["H", "-H"],
            "no_early_stopping": True,
        },
        "precision_contract": frozen["doubling"],
        "b0_stability_contract": frozen["b0_stability"],
        "decision": "reference_substrate_ready_non_authorizing",
    }
    artifact["artifact_sha256"] = object_sha256(artifact)
    return HaarReferencePlan(artifact=artifact, binding_sha256=binding, shared_orbit_keys=shared)


def _center_basis(size: int) -> np.ndarray:
    """Return the sign/order-fixed Helmert basis, never an eigensolver basis."""

    if size < 2:
        raise Phase2ReferenceError("centered interaction dimension must be at least two")
    basis = np.zeros((size, size - 1), dtype=np.float64)
    for column in range(size - 1):
        count = column + 1
        scale = math.sqrt(count * (count + 1))
        basis[:count, column] = 1.0 / scale
        basis[count, column] = -count / scale
    return basis


def _haar_orthogonal(size: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(size, size)))
    sign = np.sign(np.diag(r))
    sign[sign == 0] = 1.0
    return q * sign


def _orbit_seed(binding: str, orbit: tuple[str, str], independent_draw: int) -> int:
    if independent_draw < 0 or independent_draw >= 4096:
        raise Phase2ReferenceError("independent Haar draw lies outside frozen [0,4096)")
    key = _json_bytes({
        "binding": binding, "shared_orbit": list(orbit), "draw": independent_draw,
    })
    return int.from_bytes(hashlib.sha256(key).digest()[:16], "big", signed=False)


def _verify_reference_plan(plan: HaarReferencePlan) -> None:
    artifact = plan.artifact
    if artifact.get("version") != VERSION or artifact.get("status") != STATUS:
        raise Phase2ReferenceError("wrong Phase-2 reference plan")
    if artifact.get("authorized") is not False or artifact.get("authorizer_implemented") is not False:
        raise Phase2ReferenceError("Phase-2 reference plan crossed its scope")
    claimed = artifact.get("artifact_sha256")
    unsealed = dict(artifact)
    unsealed.pop("artifact_sha256", None)
    if not isinstance(claimed, str) or object_sha256(unsealed) != claimed:
        raise Phase2ReferenceError("Phase-2 reference plan seal mismatch")
    if artifact.get("binding_sha256") != plan.binding_sha256:
        raise Phase2ReferenceError("Phase-2 in-memory binding differs from artifact")
    shared = artifact.get("shared_orbits", {})
    if shared.get("set_sha256") != object_sha256(sorted(list(plan.shared_orbit_keys))):
        raise Phase2ReferenceError("Phase-2 shared orbit set mismatch")
    if shared.get("count") != len(plan.shared_orbit_keys):
        raise Phase2ReferenceError("Phase-2 shared orbit count mismatch")
    expected_basis = {
        "render_5": _array_sha256(_center_basis(5)),
        "prompt_3": _array_sha256(_center_basis(3)),
    }
    if artifact.get("fixed_center_basis") != {
        "family": "fixed_helmert_v1", "basis_sha256": expected_basis,
    }:
        raise Phase2ReferenceError("fixed Helmert basis binding mismatch")
    trace = artifact.get("phase1_continuous_multiplier_trace")
    if not isinstance(trace, Mapping) or trace.get("algorithm") != (
        "sha256_chain_previous_digest_draw_index_float64_row_v1"
    ):
        raise Phase2ReferenceError("Phase-1 continuous multiplier trace is absent")


def spectral_haar_antithetic_pair(
    interaction: np.ndarray, *, plan: HaarReferencePlan,
    shared_orbit: tuple[str, str], independent_draw: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic ``(H, -H)`` ispectral centered stress references."""

    _verify_reference_plan(plan)
    if shared_orbit not in plan.shared_orbit_keys:
        raise Phase2ReferenceError("shared orbit is absent from the frozen Phase-2 plan")
    value = np.asarray(interaction, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 2 or not np.isfinite(value).all():
        raise Phase2ReferenceError("interaction must be a finite two-dimensional matrix")
    tolerance = 1e-9
    if np.max(np.abs(value.sum(axis=0))) > tolerance or np.max(np.abs(value.sum(axis=1))) > tolerance:
        raise Phase2ReferenceError("interaction is not in the frozen centered subspace")
    row_basis = _center_basis(value.shape[0])
    column_basis = _center_basis(value.shape[1])
    core = row_basis.T @ value @ column_basis
    rng = np.random.Generator(
        np.random.PCG64(_orbit_seed(plan.binding_sha256, shared_orbit, independent_draw))
    )
    left = _haar_orthogonal(core.shape[0], rng)
    right = _haar_orthogonal(core.shape[1], rng)
    candidate = row_basis @ (left @ core @ right.T) @ column_basis.T
    if np.max(np.abs(candidate.sum(axis=0))) > tolerance or np.max(np.abs(candidate.sum(axis=1))) > tolerance:
        raise RuntimeError("Haar reference escaped the centered subspace")
    observed_spectrum = np.linalg.svd(value, compute_uv=False)
    reference_spectrum = np.linalg.svd(candidate, compute_uv=False)
    if not np.allclose(observed_spectrum, reference_spectrum, rtol=1e-9, atol=1e-12):
        raise RuntimeError("Haar reference did not preserve the singular spectrum")
    return candidate, -candidate


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _b0_stability(
    b0_point: float, b0_bootstrap: Sequence[float], config: Mapping[str, Any],
) -> dict[str, Any]:
    rule = config["b0_stability"]
    point = float(b0_point)
    values = np.asarray(b0_bootstrap, dtype=np.float64)
    minimum = int(rule["minimum_bootstrap_denominators"])
    if not math.isfinite(point) or point <= float(rule["absolute_floor"]):
        raise Phase2ReferenceError("B0 point is at or below the frozen near-zero floor")
    if values.ndim != 1 or values.size < minimum or not np.isfinite(values).all():
        raise Phase2ReferenceError("B0 bootstrap stability audit is incomplete or non-finite")
    quantile_level = float(rule["bootstrap_lower_quantile"])
    lower = float(np.quantile(values, quantile_level))
    ratio = lower / point
    if lower <= float(rule["absolute_floor"]):
        raise Phase2ReferenceError("B0 bootstrap lower tail reaches the near-zero floor")
    if ratio < float(rule["minimum_lower_quantile_over_point"]):
        raise Phase2ReferenceError("B0 bootstrap lower tail is unstable relative to its point")
    return {
        "stable": True,
        "point": point,
        "absolute_floor": float(rule["absolute_floor"]),
        "bootstrap_n": int(values.size),
        "bootstrap_sha256": _array_sha256(values),
        "bootstrap_lower_quantile_level": quantile_level,
        "bootstrap_lower_quantile": lower,
        "lower_quantile_over_point": ratio,
    }


def bind_mcse_evaluation_trace(
    antithetic_values: np.ndarray, *, plan: HaarReferencePlan, model: str,
    orbit_order: Sequence[tuple[str, str]], macro_statistic: str,
    calibrator_sha256: str, b0_point: float, b0_bootstrap: Sequence[float],
) -> dict[str, Any]:
    """Seal aggregate reference values to their complete evaluation context.

    This is deliberately a non-authorizing transport seal.  It prevents an
    arbitrary array, reordered orbit set, different model/calibrator/B0, or a
    discontinuous Phase-1 multiplier plan from being passed to the MC-SE audit
    as if it came from the frozen Haar evaluation.
    """

    _verify_reference_plan(plan)
    values = np.asarray(antithetic_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise Phase2ReferenceError("reference values must be finite ordered antithetic pairs")
    if values.shape[0] not in (2048, 4096):
        raise Phase2ReferenceError("MC-SE trace requires exactly 4096 or 8192 references")
    if model not in EXPECTED_MODELS:
        raise Phase2ReferenceError("MC-SE trace model is outside the frozen model set")
    ordered = tuple(tuple(str(value) for value in key) for key in orbit_order)
    if ordered != plan.shared_orbit_keys:
        raise Phase2ReferenceError("MC-SE orbit order differs from the frozen Haar plan")
    if macro_statistic != MACRO_STATISTIC:
        raise Phase2ReferenceError("MC-SE macro statistic differs from the frozen target")
    if not (
        isinstance(calibrator_sha256, str) and len(calibrator_sha256) == 64
        and all(character in "0123456789abcdef" for character in calibrator_sha256)
    ):
        raise Phase2ReferenceError("MC-SE calibrator_sha256 is malformed")
    stability = _b0_stability(b0_point, b0_bootstrap, {
        "b0_stability": plan.artifact["b0_stability_contract"],
    })
    multiplier_trace = plan.artifact.get("phase1_continuous_multiplier_trace")
    if not isinstance(multiplier_trace, Mapping):
        raise Phase2ReferenceError("Haar plan lacks the Phase-1 continuous multiplier trace")
    context = {
        "haar_plan_artifact_sha256": plan.artifact["artifact_sha256"],
        "haar_plan_binding_sha256": plan.binding_sha256,
        "orbit_order_sha256": object_sha256(list(ordered)),
        "model": model,
        "macro_statistic": macro_statistic,
        "calibrator_sha256": calibrator_sha256,
        "b0_point": stability["point"],
        "b0_bootstrap_sha256": stability["bootstrap_sha256"],
        "phase1_continuous_multiplier_trace_sha256": object_sha256(multiplier_trace),
    }
    trace = {
        "schema_version": "cecd-v4-haar-mcse-evaluation-trace-v1",
        "status": STATUS,
        "authorized": False,
        "context": context,
        "context_sha256": object_sha256(context),
        "independent_antithetic_pairs": int(values.shape[0]),
        "reference_values_sha256": _array_sha256(values),
    }
    trace["trace_sha256"] = object_sha256(trace)
    return trace


def _verify_mcse_evaluation_trace(
    trace: Mapping[str, Any], *, values: np.ndarray, plan: HaarReferencePlan,
    model: str, orbit_order: Sequence[tuple[str, str]], macro_statistic: str,
    calibrator_sha256: str, stability: Mapping[str, Any],
) -> None:
    if trace.get("schema_version") != "cecd-v4-haar-mcse-evaluation-trace-v1":
        raise Phase2ReferenceError("MC-SE evaluation trace schema mismatch")
    claimed = trace.get("trace_sha256")
    unsealed = dict(trace)
    unsealed.pop("trace_sha256", None)
    if not isinstance(claimed, str) or object_sha256(unsealed) != claimed:
        raise Phase2ReferenceError("MC-SE evaluation trace seal mismatch")
    ordered = tuple(tuple(str(value) for value in key) for key in orbit_order)
    expected_context = {
        "haar_plan_artifact_sha256": plan.artifact["artifact_sha256"],
        "haar_plan_binding_sha256": plan.binding_sha256,
        "orbit_order_sha256": object_sha256(list(ordered)),
        "model": model,
        "macro_statistic": macro_statistic,
        "calibrator_sha256": calibrator_sha256,
        "b0_point": stability["point"],
        "b0_bootstrap_sha256": stability["bootstrap_sha256"],
        "phase1_continuous_multiplier_trace_sha256": object_sha256(
            plan.artifact["phase1_continuous_multiplier_trace"]
        ),
    }
    if trace.get("context") != expected_context:
        raise Phase2ReferenceError("MC-SE evaluation context binding drift")
    if trace.get("context_sha256") != object_sha256(expected_context):
        raise Phase2ReferenceError("MC-SE evaluation context seal mismatch")
    if trace.get("independent_antithetic_pairs") != int(values.shape[0]):
        raise Phase2ReferenceError("MC-SE evaluation trace draw count mismatch")
    if trace.get("reference_values_sha256") != _array_sha256(values):
        raise Phase2ReferenceError("arbitrary or modified reference array rejected by MC-SE trace")


def audit_reference_mcse(
    antithetic_values: np.ndarray, *, plan: HaarReferencePlan, model: str,
    orbit_order: Sequence[tuple[str, str]], macro_statistic: str,
    calibrator_sha256: str, evaluation_trace: Mapping[str, Any],
    b0_point: float, b0_bootstrap: Sequence[float], config: Mapping[str, Any],
    initial_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit precision after exactly 4,096 or conditionally 8,192 references.

    Rows are independent Haar draws and columns are the ordered ``(H, -H)``
    outcomes.  Pair means, not 4,096 pseudo-independent values, define MC-SE.
    This precision audit remains non-authorizing.
    """

    frozen = validate_phase2_config(config)
    _verify_reference_plan(plan)
    values = np.asarray(antithetic_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise Phase2ReferenceError("reference values must be finite ordered antithetic pairs")
    if values.shape[0] not in (2048, 4096):
        raise Phase2ReferenceError("MC-SE audit requires exactly 4096 or 8192 references")
    stability = _b0_stability(b0_point, b0_bootstrap, frozen)
    _verify_mcse_evaluation_trace(
        evaluation_trace, values=values, plan=plan, model=model,
        orbit_order=orbit_order, macro_statistic=macro_statistic,
        calibrator_sha256=calibrator_sha256, stability=stability,
    )
    pair_mean = values.mean(axis=1)
    mcse = float(np.std(pair_mean, ddof=1) / math.sqrt(pair_mean.size))
    scaled = mcse / stability["point"]
    threshold = float(frozen["doubling"]["trigger_if_strictly_greater_than"])
    initial_prefix_sha256 = _array_sha256(values[:2048])
    if values.shape[0] == 2048:
        if initial_audit is not None:
            raise Phase2ReferenceError("initial 4096-reference audit cannot consume a prior audit")
        trigger = scaled > threshold
        decision = "double_once_to_8192" if trigger else "precision_complete_at_4096"
    else:
        if not isinstance(initial_audit, Mapping):
            raise Phase2ReferenceError("8192-reference audit requires its frozen initial audit")
        if initial_audit.get("status") != STATUS or initial_audit.get("authorized") is not False:
            raise Phase2ReferenceError("initial MC-SE audit is not a non-authorizing Phase-2 audit")
        claimed = initial_audit.get("audit_sha256")
        unsealed = dict(initial_audit)
        unsealed.pop("audit_sha256", None)
        if not isinstance(claimed, str) or object_sha256(unsealed) != claimed:
            raise Phase2ReferenceError("initial MC-SE audit seal mismatch")
        if initial_audit.get("source_sha256") != _module_sha256():
            raise Phase2ReferenceError("initial MC-SE audit source binding drift")
        if initial_audit.get("config_sha256") != object_sha256(frozen):
            raise Phase2ReferenceError("initial MC-SE audit config binding drift")
        if initial_audit.get("mcse_context_sha256") != evaluation_trace.get("context_sha256"):
            raise Phase2ReferenceError("MC-SE model/orbit/statistic/calibrator/B0 context changed")
        if initial_audit.get("decision") != "double_once_to_8192":
            raise Phase2ReferenceError("8192 references were not triggered by the frozen rule")
        if initial_audit.get("initial_values_sha256") != initial_prefix_sha256:
            raise Phase2ReferenceError("8192 reference prefix differs from the frozen initial 4096")
        if initial_audit.get("b0_stability") != stability:
            raise Phase2ReferenceError("B0 changed between initial and doubled precision audits")
        decision = "precision_complete_at_8192_cap_no_further_doubling"
        trigger = False
    audit = {
        "version": VERSION,
        "status": STATUS,
        "authorized": False,
        "source_sha256": _module_sha256(),
        "config_sha256": object_sha256(frozen),
        "haar_plan_artifact_sha256": plan.artifact["artifact_sha256"],
        "haar_plan_binding_sha256": plan.binding_sha256,
        "mcse_evaluation_trace_sha256": evaluation_trace["trace_sha256"],
        "mcse_context_sha256": evaluation_trace["context_sha256"],
        "model": model,
        "orbit_order_sha256": evaluation_trace["context"]["orbit_order_sha256"],
        "macro_statistic": macro_statistic,
        "calibrator_sha256": calibrator_sha256,
        "phase1_continuous_multiplier_trace_sha256": evaluation_trace["context"][
            "phase1_continuous_multiplier_trace_sha256"
        ],
        "inference_boundary": (
            "MC-SE precision audit for a Haar stress-reference mean; not a "
            "randomization test, p-value, efficacy decision, or authorizer"
        ),
        "reference_count": int(values.size),
        "independent_antithetic_pairs": int(values.shape[0]),
        "pair_mean_estimate": float(np.mean(pair_mean)),
        "pair_mean_mcse": mcse,
        "mcse_over_B0": scaled,
        "doubling_threshold_strict": threshold,
        "doubling_triggered": trigger,
        "decision": decision,
        "initial_values_sha256": initial_prefix_sha256,
        "all_values_sha256": _array_sha256(values),
        "b0_stability": stability,
        "maximum_reference_count": 8192,
        "further_doubling_allowed": False,
    }
    audit["audit_sha256"] = object_sha256(audit)
    return audit


__all__ = [
    "HaarReferencePlan",
    "Phase2ReferenceError",
    "audit_reference_mcse",
    "bind_mcse_evaluation_trace",
    "build_reference_plan",
    "spectral_haar_antithetic_pair",
    "validate_phase2_config",
]
