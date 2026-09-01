#!/usr/bin/env python3
"""Non-authorizing CECD v4 promotion Phase-1 structural contract.

This CPU-only module closes only P0-1 (exact scientific grid/strata/pairing)
and P0-3 (global cluster identity and shared positive multipliers).  It does
not fit calibration, construct Haar references, inspect outcomes, authorize a
run, or modify the frozen v3 pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .analyze_clinical_equivalence_composition_defect_v1 import (
    ContractError as V1ContractError,
    sha256_file,
    validate_payload,
)


VERSION = "cecd-v4-promotion-phase1-structural-v1"
CONFIG_VERSION = "cecd-v4-promotion-phase1-contract-v1"
STATUS = "outcome_blind_non_authorizing_phase1_only"
EXPECTED_MODELS = ("huatuo:HuatuoGPT-Vision-7B", "hulu:Hulu-Med-4B")
EXPECTED_FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "pleural_effusion",
    "pulmonary_fibrosis",
)
EXPECTED_VOTES = (0, 1, 2, 3)
EXPECTED_RENDERS = (
    "baseline_percentile",
    "native_linear",
    "center_minus_0p05w",
    "center_plus_0p05w",
    "width_x1p25",
)
EXPECTED_PROMPTS = ("existential", "radiograph_subject", "visibility")
EXPECTED_CONTROLS = {
    "baseline_render": "baseline_percentile",
    "baseline_prompt": "existential",
    "identity_render": "identity_lossless_duplicate",
    "duplicate_prompt": "existential_exact_duplicate",
}
PATIENT_MANIFEST_VERSION = "cecd-external-patient-mapping-manifest-v1"


class Phase1ContractError(ValueError):
    """Phase-1 structure cannot support the frozen estimand."""


@dataclass(frozen=True)
class Phase1Build:
    """Auditable artifact plus the one shared in-memory multiplier plan."""

    artifact: dict[str, Any]
    cluster_order: tuple[str, ...]
    multipliers: np.ndarray
    image_to_cluster: dict[str, str]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _module_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _exact_sequence(value: Any, expected: Sequence[Any], label: str) -> None:
    if not isinstance(value, list) or tuple(value) != tuple(expected):
        raise Phase1ContractError(f"{label} must equal the frozen ordered set {list(expected)}")


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != CONFIG_VERSION:
        raise Phase1ContractError(f"config schema_version must be {CONFIG_VERSION}")
    if config.get("scientific_status") != STATUS or config.get("authorized") is not False:
        raise Phase1ContractError("Phase 1 config must remain explicitly non-authorizing")
    if config.get("frozen_before_outputs") is not True:
        raise Phase1ContractError("Phase 1 config was not frozen before outputs")
    _exact_sequence(config.get("models"), EXPECTED_MODELS, "models")
    _exact_sequence(config.get("findings"), EXPECTED_FINDINGS, "findings")
    _exact_sequence(config.get("reader_vote_bins"), EXPECTED_VOTES, "reader_vote_bins")
    grid = config.get("science_grid")
    if not isinstance(grid, Mapping):
        raise Phase1ContractError("science_grid is missing")
    _exact_sequence(grid.get("renders"), EXPECTED_RENDERS, "science_grid.renders")
    _exact_sequence(grid.get("prompts"), EXPECTED_PROMPTS, "science_grid.prompts")
    for key, expected in EXPECTED_CONTROLS.items():
        if grid.get(key) != expected:
            raise Phase1ContractError(f"science_grid.{key} must equal {expected}")
    stages = config.get("stage_contract")
    if not isinstance(stages, Mapping) or set(stages) != {"dev_fit", "confirmation_locked"}:
        raise Phase1ContractError("stage_contract must contain exact dev_fit and confirmation_locked")
    for stage, source in (("dev_fit", "dev"), ("confirmation_locked", "confirmation")):
        item = stages[stage]
        if not isinstance(item, Mapping) or item.get("source_manifest_split") != source:
            raise Phase1ContractError(f"{stage} source_manifest_split must be {source}")
        quota = item.get("exact_orbits_per_model_finding_vote")
        if isinstance(quota, bool) or not isinstance(quota, int) or quota < 1:
            raise Phase1ContractError(f"{stage} exact quota must be a positive integer")
    pairing = config.get("pairing")
    required_pairing = {
        "key": ["image_id", "finding", "reader_votes"],
        "model_orbit_sets_must_match_exactly": True,
        "same_image_may_not_cross_stage": True,
        "same_patient_may_not_cross_stage_when_declared": True,
    }
    if pairing != required_pairing:
        raise Phase1ContractError("pairing contract differs from the frozen exact policy")
    cluster = config.get("cluster_contract")
    required_cluster = {
        "mode": "patient_only_if_mapping_globally_complete_and_consistent_else_image",
        "conflicting_nonempty_patient_ids_for_one_image": "fail",
        "partial_but_nonconflicting_patient_mapping": "global_image_fallback",
        "mixed_per_orbit_fallback": False,
        "verifiable_patient_provenance_required_for_patient_inference": True,
        "unverifiable_patient_provenance_status": (
            "diagnostic_image_cluster_only_non_authorizing"
        ),
    }
    if not isinstance(cluster, Mapping):
        raise Phase1ContractError("cluster contract is missing")
    minimums = cluster.get("minimum_unique_patient_clusters_per_model_finding_vote")
    if not isinstance(minimums, Mapping) or set(minimums) != {
        "dev_fit", "confirmation_locked",
    }:
        raise Phase1ContractError("patient-cluster minima must be frozen for both stages")
    for stage in ("dev_fit", "confirmation_locked"):
        value = minimums[stage]
        quota = config["stage_contract"][stage]["exact_orbits_per_model_finding_vote"]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= quota:
            raise Phase1ContractError(
                f"patient-cluster minimum for {stage} must lie in [1, quota]"
            )
    cluster_without_minima = dict(cluster)
    cluster_without_minima.pop("minimum_unique_patient_clusters_per_model_finding_vote", None)
    anchor = cluster_without_minima.pop("patient_provenance_anchor", None)
    if not isinstance(anchor, Mapping) or anchor.get("policy") != (
        "stage_manifest_content_sha256_must_be_frozen_in_config_before_outputs"
    ):
        raise Phase1ContractError("patient provenance requires a frozen config anchor")
    if anchor.get("null_anchor_policy") != (
        "patient_mode_forbidden_use_image_cluster_diagnostic_only"
    ):
        raise Phase1ContractError("null patient-provenance anchor policy changed")
    stage_hashes = anchor.get("stage_manifest_sha256")
    if not isinstance(stage_hashes, Mapping) or set(stage_hashes) != {
        "dev_fit", "confirmation_locked",
    }:
        raise Phase1ContractError("patient provenance stage anchors are incomplete")
    for stage, value in stage_hashes.items():
        if value is not None and not _valid_sha256(value):
            raise Phase1ContractError(f"{stage} patient provenance anchor is malformed")
    if cluster_without_minima != required_cluster:
        raise Phase1ContractError("cluster contract differs from the frozen global-mode policy")
    bootstrap = config.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise Phase1ContractError("bootstrap contract is missing")
    if bootstrap.get("family") != "keyed_exponential_strictly_positive_multiplier":
        raise Phase1ContractError("bootstrap family is not the frozen strictly-positive family")
    if bootstrap.get("normalization") != "per_draw_mean_one":
        raise Phase1ContractError("bootstrap normalization drift")
    if bootstrap.get("conditional_rejection_allowed") is not False:
        raise Phase1ContractError("conditional bootstrap rejection must be disabled")
    for key in ("master_seed", "draws"):
        value = bootstrap.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise Phase1ContractError(f"bootstrap.{key} must be a positive integer")
    for key in (
        "minimum_ess_fraction", "maximum_single_cluster_weight_fraction",
        "maximum_single_cluster_orbit_contribution",
    ):
        try:
            value = float(bootstrap[key])
        except (KeyError, TypeError, ValueError) as error:
            raise Phase1ContractError(f"bootstrap.{key} must be finite") from error
        if not math.isfinite(value) or not 0 < value <= 1:
            raise Phase1ContractError(f"bootstrap.{key} must lie in (0,1]")
    return dict(config)


def _expected_cells() -> set[tuple[str, str]]:
    return {
        (render, prompt) for render in EXPECTED_RENDERS for prompt in EXPECTED_PROMPTS
    } | {
        (EXPECTED_CONTROLS["identity_render"], prompt) for prompt in EXPECTED_PROMPTS
    } | {
        (EXPECTED_CONTROLS["baseline_render"], EXPECTED_CONTROLS["duplicate_prompt"])
    }


def _precheck_raw(payload: Mapping[str, Any], label: str) -> None:
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise Phase1ContractError(f"{label} records must be non-empty")
    allowed_cells = _expected_cells()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Phase1ContractError(f"{label} row {index} is not an object")
        if str(row.get("model")) not in EXPECTED_MODELS:
            raise Phase1ContractError(f"{label} contains an unexpected model")
        if str(row.get("finding")) not in EXPECTED_FINDINGS:
            raise Phase1ContractError(f"{label} contains an unexpected finding")
        cell = (str(row.get("render_id")), str(row.get("prompt_id")))
        if cell not in allowed_cells:
            raise Phase1ContractError(f"{label} contains a non-frozen cell {cell}")


def _validate_stage(
    payload: Mapping[str, Any], config: Mapping[str, Any], stage: str,
) -> dict[str, Any]:
    _precheck_raw(payload, stage)
    try:
        contract = validate_payload(payload)
    except V1ContractError as error:
        raise Phase1ContractError(f"{stage} v1 payload invalid: {error}") from error
    expected_source = config["stage_contract"][stage]["source_manifest_split"]
    if contract["split"] != stage or contract["source_manifest_split"] != expected_source:
        raise Phase1ContractError(f"{stage} has wrong split/source_manifest_split")
    geometry = {
        "primary_renders": tuple(contract["primary_renders"]),
        "primary_prompts": tuple(contract["primary_prompts"]),
        "baseline_render": contract["baseline_render"],
        "baseline_prompt": contract["baseline_prompt"],
        "identity_render": contract["identity_render"],
        "duplicate_prompt": contract["duplicate_prompt"],
    }
    expected_geometry = {
        "primary_renders": EXPECTED_RENDERS,
        "primary_prompts": EXPECTED_PROMPTS,
        **EXPECTED_CONTROLS,
    }
    if geometry != expected_geometry:
        raise Phase1ContractError(f"{stage} geometry is not the exact frozen 5x3 grid")
    expected_cells = _expected_cells()
    for key, rows in contract["by_orbit"].items():
        cells = {(str(row["render_id"]), str(row["prompt_id"])) for row in rows}
        if cells != expected_cells or len(rows) != len(expected_cells):
            raise Phase1ContractError(f"{stage} orbit {key} does not have exact 15+3+1 cells")
    models = {key[0] for key in contract["by_orbit"]}
    findings = {key[2] for key in contract["by_orbit"]}
    if models != set(EXPECTED_MODELS):
        raise Phase1ContractError(f"{stage} model closure is not exact")
    if findings != set(EXPECTED_FINDINGS):
        raise Phase1ContractError(f"{stage} finding closure is not exact")
    quota = int(config["stage_contract"][stage]["exact_orbits_per_model_finding_vote"])
    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    keys_by_model: dict[str, set[tuple[str, str, int]]] = {
        model: set() for model in EXPECTED_MODELS
    }
    for (model, image, finding), rows in contract["by_orbit"].items():
        vote = int(rows[0]["reader_votes"])
        counts[(model, finding, vote)] += 1
        keys_by_model[model].add((image, finding, vote))
    expected_strata = {
        (model, finding, vote)
        for model in EXPECTED_MODELS for finding in EXPECTED_FINDINGS for vote in EXPECTED_VOTES
    }
    if set(counts) != expected_strata:
        raise Phase1ContractError(f"{stage} does not contain exact 2x4x4 strata")
    wrong = {key: value for key, value in counts.items() if value != quota}
    if wrong:
        example = next(iter(sorted(wrong.items())))
        raise Phase1ContractError(
            f"{stage} exact per-stratum quota failed; example {example[0]}={example[1]} expected {quota}"
        )
    if keys_by_model[EXPECTED_MODELS[0]] != keys_by_model[EXPECTED_MODELS[1]]:
        left_only = keys_by_model[EXPECTED_MODELS[0]] - keys_by_model[EXPECTED_MODELS[1]]
        right_only = keys_by_model[EXPECTED_MODELS[1]] - keys_by_model[EXPECTED_MODELS[0]]
        raise Phase1ContractError(
            "two models do not have exact orbit pairing; "
            f"left_only={len(left_only)} right_only={len(right_only)}"
        )
    return {
        "contract": contract,
        "quota": quota,
        "counts": counts,
        "paired_keys": keys_by_model[EXPECTED_MODELS[0]],
        "excluded_orbits": contract["excluded_orbits"],
    }


def _patient_value(row: Mapping[str, Any]) -> str | None:
    value = row.get("patient_id")
    if value in (None, ""):
        return None
    if not isinstance(value, str) or not value.strip():
        raise Phase1ContractError("patient_id must be a non-empty string when declared")
    return value.strip()


def _global_cluster_map(
    stages: Mapping[str, Mapping[str, Any]], *, patient_provenance_verified: bool,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    image_stage: dict[str, str] = {}
    image_patients: dict[str, set[str]] = defaultdict(set)
    image_missing: dict[str, bool] = defaultdict(bool)
    patient_stages: dict[str, set[str]] = defaultdict(set)
    for stage, entry in stages.items():
        for row in entry["contract"]["rows"]:
            image = str(row["image_id"])
            previous = image_stage.setdefault(image, stage)
            if previous != stage:
                raise Phase1ContractError(f"image {image} crosses dev/confirmation")
            patient = _patient_value(row)
            if patient is None:
                image_missing[image] = True
            else:
                image_patients[image].add(patient)
                patient_stages[patient].add(stage)
    conflict = {image: values for image, values in image_patients.items() if len(values) > 1}
    if conflict:
        image = next(iter(sorted(conflict)))
        raise Phase1ContractError(f"image {image} has conflicting patient mapping")
    crossing_patients = sorted(patient for patient, values in patient_stages.items() if len(values) > 1)
    if crossing_patients:
        raise Phase1ContractError(f"patient {crossing_patients[0]} crosses dev/confirmation")
    images = set(image_stage)
    complete = all(not image_missing[image] and len(image_patients[image]) == 1 for image in images)
    mode = "patient" if complete and patient_provenance_verified else "image"
    mapping = {
        image: (
            f"patient:{next(iter(image_patients[image]))}"
            if mode == "patient" else f"image:{image}"
        )
        for image in sorted(images)
    }
    # One image has exactly one global mapping by construction; this explicit
    # audit catches later accidental per-model/finding fallback.
    if len(mapping) != len(images) or any(not value.startswith(f"{mode}:") for value in mapping.values()):
        raise RuntimeError("global cluster mode was not applied uniformly")
    return mode, mapping, {
        "patient_mapping_complete": complete,
        "patient_provenance_verified_before_cluster_choice": patient_provenance_verified,
        "images_with_missing_patient_rows": sum(bool(image_missing[image]) for image in images),
        "declared_unique_patients": len(patient_stages),
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _patient_provenance_audit(
    payloads: Mapping[str, Mapping[str, Any]],
    external_manifests: Mapping[str, Mapping[str, Any]] | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute patient provenance from separately supplied mapping manifests.

    A payload assertion or 64-hex string is never evidence by itself.  Missing
    external manifests deliberately demote the analysis to image clustering;
    malformed or mismatched supplied manifests fail closed.
    """

    stage_sources: dict[str, str] = {}
    stage_mapping_hashes: dict[str, str] = {}
    reasons: list[str] = []
    frozen_anchors = config["cluster_contract"]["patient_provenance_anchor"][
        "stage_manifest_sha256"
    ]
    for stage, payload in payloads.items():
        manifest = None if external_manifests is None else external_manifests.get(stage)
        frozen_anchor = frozen_anchors[stage]
        if frozen_anchor is None:
            if manifest is not None:
                raise Phase1ContractError(
                    f"{stage} external patient manifest lacks a pre-output config anchor"
                )
            reasons.append(f"{stage}:no_trusted_patient_manifest_anchor_in_config")
            continue
        if manifest is None:
            reasons.append(f"{stage}:missing_external_patient_manifest")
            continue
        if manifest.get("schema_version") != PATIENT_MANIFEST_VERSION:
            raise Phase1ContractError(f"{stage} external patient manifest schema mismatch")
        if manifest.get("stage") != stage or manifest.get("frozen_before_model_outputs") is not True:
            raise Phase1ContractError(f"{stage} external patient manifest timing/stage mismatch")
        records = manifest.get("records")
        if not isinstance(records, list) or not records:
            raise Phase1ContractError(f"{stage} external patient manifest records are missing")
        manifest_map: dict[str, str] = {}
        for index, record in enumerate(records):
            if not isinstance(record, Mapping) or set(record) != {"image_id", "patient_id"}:
                raise Phase1ContractError(
                    f"{stage} external patient manifest record {index} has wrong schema"
                )
            image = record["image_id"]
            patient = record["patient_id"]
            if not isinstance(image, str) or not image.strip():
                raise Phase1ContractError(f"{stage} external patient manifest has empty image_id")
            if not isinstance(patient, str) or not patient.strip():
                raise Phase1ContractError(f"{stage} external patient manifest has empty patient_id")
            if image in manifest_map:
                raise Phase1ContractError(
                    f"{stage} external patient manifest repeats image {image}"
                )
            manifest_map[image] = patient
        payload_map: dict[str, str] = {}
        for row in payload.get("records", []):
            image = str(row.get("image_id", ""))
            patient = _patient_value(row)
            if patient is None:
                raise Phase1ContractError(
                    f"{stage} claims external provenance but row {image} lacks patient_id"
                )
            previous = payload_map.setdefault(image, patient)
            if previous != patient:
                raise Phase1ContractError(f"{stage} payload has conflicting patient mapping")
        if manifest_map != payload_map:
            raise Phase1ContractError(
                f"{stage} external patient manifest does not exactly reproduce payload mapping"
            )
        manifest_sha256 = object_sha256(manifest)
        if manifest_sha256 != frozen_anchor:
            raise Phase1ContractError(
                f"{stage} external patient manifest differs from frozen config anchor"
            )
        value = payload.get("patient_provenance")
        if not isinstance(value, Mapping):
            raise Phase1ContractError(f"{stage} supplied an external manifest without provenance")
        if value.get("schema_version") != "verified-external-patient-mapping-v1":
            raise Phase1ContractError(f"{stage} patient provenance schema mismatch")
        if value.get("verified_before_model_outputs") is not True:
            raise Phase1ContractError(f"{stage} patient provenance timing mismatch")
        source = value.get("source_manifest_sha256")
        if not _valid_sha256(source) or source != manifest_sha256:
            raise Phase1ContractError(f"{stage} patient provenance content seal mismatch")
        stage_sources[stage] = str(source)
        stage_mapping_hashes[stage] = object_sha256(manifest_map)
    verified = not reasons and set(stage_sources) == set(payloads)
    return {
        "verified": verified,
        "schema_version": "verified-external-patient-mapping-v1",
        "stage_source_manifest_sha256": stage_sources,
        "stage_image_to_patient_sha256": stage_mapping_hashes,
        "frozen_config_stage_manifest_sha256": dict(frozen_anchors),
        "external_manifest_content_recomputed": verified,
        "failure_reasons": reasons,
    }


def _patient_cluster_gate(
    stages: Mapping[str, Mapping[str, Any]], image_to_cluster: Mapping[str, str],
    *, mode: str, provenance: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    minima = config["cluster_contract"][
        "minimum_unique_patient_clusters_per_model_finding_vote"
    ]
    if mode == "patient":
        for stage, entry in stages.items():
            clusters: dict[tuple[str, str, int], set[str]] = defaultdict(set)
            for (model, image, finding), rows in entry["contract"]["by_orbit"].items():
                vote = int(rows[0]["reader_votes"])
                clusters[(model, finding, vote)].add(image_to_cluster[image])
            for key, values in sorted(clusters.items()):
                label = "|".join((stage, key[0], key[1], str(key[2])))
                counts[label] = len(values)
                if provenance["verified"] and len(values) < int(minima[stage]):
                    raise Phase1ContractError(
                        "verified patient-cluster minimum failed; "
                        f"{label}={len(values)} expected_at_least={minima[stage]}"
                    )
    eligible = bool(mode == "patient" and provenance["verified"])
    return {
        "patient_cluster_inference_eligible": eligible,
        "explicitly_non_authorizing_without_verified_provenance": True,
        "status": (
            "verified_patient_cluster_minima_passed"
            if eligible else
            "diagnostic_image_cluster_only_non_authorizing"
        ),
        "minimum_unique_patient_clusters_per_model_finding_vote": dict(minima),
        "observed_unique_patient_clusters_per_model_finding_vote": counts,
        "provenance": dict(provenance),
    }


def _keyed_exponential(binding: str, seed: int, draw: int, cluster: str) -> float:
    key = f"{binding}|{seed}|{draw}|{cluster}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(key).digest()[:8], "big", signed=False)
    uniform = (integer + 0.5) / float(2**64)
    return -math.log(uniform)


def generate_strictly_positive_multipliers(
    cluster_order: Sequence[str], *, binding_sha256: str, master_seed: int,
    draws: int,
) -> np.ndarray:
    if not cluster_order or len(set(cluster_order)) != len(cluster_order):
        raise Phase1ContractError("cluster_order must be non-empty and unique")
    if len(binding_sha256) != 64:
        raise Phase1ContractError("multiplier binding_sha256 is malformed")
    weights = np.empty((draws, len(cluster_order)), dtype=np.float64)
    for draw in range(draws):
        for column, cluster in enumerate(cluster_order):
            weights[draw, column] = _keyed_exponential(
                binding_sha256, int(master_seed), draw, str(cluster)
            )
        weights[draw] /= weights[draw].mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("strictly-positive multiplier construction failed")
    return weights


def continuous_multiplier_trace(multipliers: np.ndarray) -> dict[str, Any]:
    """Hash every ordered draw into one continuous, prefix-sensitive trace."""

    matrix = np.asarray(multipliers, dtype="<f8")
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise Phase1ContractError("multiplier trace requires one finite non-empty matrix")
    chain = bytes(32)
    trace = hashlib.sha256()
    checkpoints: dict[str, str] = {}
    for index, row in enumerate(matrix):
        digest = hashlib.sha256()
        digest.update(chain)
        digest.update(index.to_bytes(8, "big", signed=False))
        digest.update(row.tobytes(order="C"))
        chain = digest.digest()
        trace.update(chain)
        if index == 0 or (index + 1) % 1024 == 0 or index + 1 == matrix.shape[0]:
            checkpoints[str(index + 1)] = chain.hex()
    return {
        "algorithm": "sha256_chain_previous_digest_draw_index_float64_row_v1",
        "draws": int(matrix.shape[0]),
        "clusters": int(matrix.shape[1]),
        "final_chain_sha256": chain.hex(),
        "all_prefix_chain_sha256": trace.hexdigest(),
        "checkpoints": checkpoints,
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, q))
        for name, q in (
            ("min", 0.0), ("p01", 0.01), ("p05", 0.05), ("median", 0.5),
            ("p95", 0.95), ("p99", 0.99), ("max", 1.0),
        )
    }


def _bootstrap_audit(
    multipliers: np.ndarray, cluster_order: Sequence[str],
    confirmation: Mapping[str, Any], image_to_cluster: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    n_clusters = len(cluster_order)
    total = multipliers.sum(axis=1)
    ess = total**2 / np.sum(multipliers**2, axis=1)
    ess_fraction = ess / n_clusters
    weight_fraction = np.max(multipliers / total[:, None], axis=1)
    orbit_counts = defaultdict(int)
    unique_orbits = set()
    unique_clusters_by_stratum: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for (model, image, finding), rows in confirmation["contract"]["by_orbit"].items():
        vote = int(rows[0]["reader_votes"])
        key = (model, image, finding)
        if key in unique_orbits:
            raise RuntimeError("confirmation orbit counted twice")
        unique_orbits.add(key)
        cluster = image_to_cluster[image]
        orbit_counts[cluster] += 1
        unique_clusters_by_stratum[(model, finding, vote)].add(cluster)
    count = np.asarray([orbit_counts[cluster] for cluster in cluster_order], dtype=float)
    weighted_orbits = multipliers * count[None, :]
    contribution = np.max(weighted_orbits / weighted_orbits.sum(axis=1, keepdims=True), axis=1)
    bootstrap = config["bootstrap"]
    if float(np.min(ess_fraction)) < float(bootstrap["minimum_ess_fraction"]):
        raise Phase1ContractError("strictly-positive multiplier ESS fell below frozen threshold")
    if float(np.max(weight_fraction)) > float(bootstrap["maximum_single_cluster_weight_fraction"]):
        raise Phase1ContractError("single-cluster multiplier fraction exceeded frozen threshold")
    if float(np.max(contribution)) > float(bootstrap["maximum_single_cluster_orbit_contribution"]):
        raise Phase1ContractError("single-cluster orbit contribution exceeded frozen threshold")
    return {
        "attempted_draws": int(multipliers.shape[0]),
        "accepted_draws": int(multipliers.shape[0]),
        "rejected_draws": 0,
        "rejection_rate": 0.0,
        "strictly_positive": bool(np.all(multipliers > 0)),
        "minimum_raw_multiplier": float(np.min(multipliers)),
        "per_draw_mean_max_abs_error": float(np.max(np.abs(multipliers.mean(axis=1) - 1.0))),
        "effective_cluster_count": _quantiles(ess),
        "effective_cluster_fraction": _quantiles(ess_fraction),
        "maximum_single_cluster_weight_fraction": _quantiles(weight_fraction),
        "maximum_single_cluster_orbit_contribution": _quantiles(contribution),
        "static_orbits_per_cluster": _quantiles(count),
        "per_model_finding_vote_unique_cluster_count": {
            "|".join((model, finding, str(vote))): len(values)
            for (model, finding, vote), values in sorted(unique_clusters_by_stratum.items())
        },
    }


def build_phase1_contract(
    dev_payload: Mapping[str, Any], confirmation_payload: Mapping[str, Any],
    config: Mapping[str, Any], *,
    external_patient_manifests: Mapping[str, Mapping[str, Any]] | None = None,
) -> Phase1Build:
    """Validate exact closure and freeze one global positive multiplier plan."""

    frozen = validate_config(config)
    dev = _validate_stage(dev_payload, frozen, "dev_fit")
    confirmation = _validate_stage(
        confirmation_payload, frozen, "confirmation_locked"
    )
    stages = {"dev_fit": dev, "confirmation_locked": confirmation}
    patient_provenance = _patient_provenance_audit({
        "dev_fit": dev_payload,
        "confirmation_locked": confirmation_payload,
    }, external_patient_manifests, frozen)
    mode, image_to_cluster, mapping_audit = _global_cluster_map(
        stages, patient_provenance_verified=bool(patient_provenance["verified"]),
    )
    patient_gate = _patient_cluster_gate(
        stages, image_to_cluster, mode=mode, provenance=patient_provenance,
        config=frozen,
    )
    dev_hash = object_sha256(dev_payload)
    confirmation_hash = object_sha256(confirmation_payload)
    config_hash = object_sha256(frozen)
    cluster_map_hash = object_sha256(image_to_cluster)
    orbit_hashes = {
        stage: object_sha256(sorted(list(entry["paired_keys"])))
        for stage, entry in stages.items()
    }
    source_hash = _module_sha256()
    # This seal is deliberately outcome-bearing and is retained for provenance.
    # It must never be used as an RNG seed because payload hashes include scores
    # and reader votes.
    binding = object_sha256({
        "version": VERSION, "source_sha256": source_hash,
        "config_sha256": config_hash, "dev_payload_sha256": dev_hash,
        "confirmation_payload_sha256": confirmation_hash,
        "cluster_map_sha256": cluster_map_hash,
        "orbit_set_sha256": orbit_hashes,
    })
    confirmation_images = {
        image for _, image, _ in confirmation["contract"]["by_orbit"]
    }
    cluster_order = tuple(sorted({image_to_cluster[image] for image in confirmation_images}))
    bootstrap = frozen["bootstrap"]
    # RNG identity contains only pre-output design/provenance.  In particular,
    # no signed score, logit, entropy, commitment, loss, or observed reader vote
    # enters this object.  Stratified vote closure remains separately audited.
    unlabeled_orbit_sets = {
        stage: object_sha256(sorted({
            (str(model), str(image), str(finding))
            for model, image, finding in entry["contract"]["by_orbit"]
        }))
        for stage, entry in stages.items()
    }
    rng_design_identity = {
        "schema_version": "cecd-v4-pre-output-rng-design-v1",
        "models": list(EXPECTED_MODELS),
        "findings": list(EXPECTED_FINDINGS),
        "science_grid": frozen["science_grid"],
        "stage_exact_orbit_quotas": {
            stage: entry["quota"] for stage, entry in stages.items()
        },
        "unlabeled_model_image_finding_set_sha256": unlabeled_orbit_sets,
        "cluster_map_sha256": cluster_map_hash,
        "confirmation_cluster_order_sha256": object_sha256(list(cluster_order)),
        "bootstrap_family": bootstrap["family"],
        "bootstrap_draws": int(bootstrap["draws"]),
    }
    rng_design_binding = object_sha256(rng_design_identity)
    multipliers = generate_strictly_positive_multipliers(
        cluster_order, binding_sha256=rng_design_binding,
        master_seed=int(bootstrap["master_seed"]), draws=int(bootstrap["draws"]),
    )
    audit = _bootstrap_audit(
        multipliers, cluster_order, confirmation, image_to_cluster, frozen
    )
    multiplier_digest = hashlib.sha256(
        np.asarray(multipliers, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    multiplier_trace = continuous_multiplier_trace(multipliers)
    artifact = {
        "version": VERSION,
        "status": STATUS,
        "authorized": False,
        "authorizer_implemented": False,
        "haar_implemented": False,
        "calibration_implemented": False,
        "scope": "P0-1_and_P0-3_structure_only",
        "source_sha256": source_hash,
        "config_sha256": config_hash,
        "dev_payload_sha256": dev_hash,
        "confirmation_payload_sha256": confirmation_hash,
        "binding_sha256": binding,
        "rng_design_identity_sha256": rng_design_binding,
        "rng_design_identity_excludes_scores_logits_losses_and_reader_votes": True,
        "closure": {
            "models": list(EXPECTED_MODELS),
            "findings": list(EXPECTED_FINDINGS),
            "reader_vote_bins": list(EXPECTED_VOTES),
            "science_grid_shape": [len(EXPECTED_RENDERS), len(EXPECTED_PROMPTS)],
            "science_cells_per_orbit": len(EXPECTED_RENDERS) * len(EXPECTED_PROMPTS),
            "control_cells_per_orbit": 4,
            "total_cells_per_orbit": len(_expected_cells()),
            "stage_quota": {
                stage: entry["quota"] for stage, entry in stages.items()
            },
            "exact_strata_per_model": len(EXPECTED_FINDINGS) * len(EXPECTED_VOTES),
            "exact_model_orbit_pairing": True,
            "orbit_set_sha256": orbit_hashes,
            "whole_orbit_exclusions": {
                stage: len(entry["excluded_orbits"]) for stage, entry in stages.items()
            },
        },
        "cluster_identity": {
            "global_mode": mode,
            "no_mixed_fallback": True,
            "image_to_cluster_sha256": cluster_map_hash,
            "n_images_all_stages": len(image_to_cluster),
            "n_confirmation_clusters": len(cluster_order),
            "cluster_order_sha256": object_sha256(list(cluster_order)),
            "cross_split_image_overlap": 0,
            "cross_split_declared_patient_overlap": 0,
            **mapping_audit,
            "patient_cluster_gate": patient_gate,
        },
        "shared_multiplier_plan": {
            "family": bootstrap["family"],
            "master_seed": int(bootstrap["master_seed"]),
            "draws": int(bootstrap["draws"]),
            "key": "sha256(pre_output_design_binding|seed|draw|global_cluster_id)",
            "rng_design_identity_sha256": rng_design_binding,
            "rng_seed_excludes_scores_logits_losses_and_reader_votes": True,
            "normalization": bootstrap["normalization"],
            "shape": list(multipliers.shape),
            "dtype": "float64_little_endian_for_digest",
            "multiplier_sha256": multiplier_digest,
            "continuous_multiplier_trace": multiplier_trace,
            "same_matrix_for_all_models_findings_and_orbits": True,
            "audit": audit,
        },
        "decision": "structural_phase1_ready_non_authorizing",
        "next_blocked_phase": (
            "Haar/calibration/clinical authorizer remain absent; this artifact cannot "
            "authorize scoring, analysis promotion, or a scientific claim"
        ),
    }
    artifact["artifact_sha256"] = object_sha256(artifact)
    return Phase1Build(
        artifact=artifact, cluster_order=cluster_order,
        multipliers=multipliers, image_to_cluster=image_to_cluster,
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Phase1ContractError(f"{path} must contain one JSON object")
    return dict(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-payload", type=Path, required=True)
    parser.add_argument("--confirmation-payload", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dev-patient-manifest", type=Path)
    parser.add_argument("--confirmation-patient-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multiplier-output", type=Path, required=True)
    args = parser.parse_args()
    if args.multiplier_output.suffix != ".npz":
        raise Phase1ContractError("--multiplier-output must end in .npz")
    if bool(args.dev_patient_manifest) != bool(args.confirmation_patient_manifest):
        raise Phase1ContractError(
            "patient provenance requires both dev and confirmation external manifests"
        )
    manifests = None
    if args.dev_patient_manifest is not None:
        manifests = {
            "dev_fit": _load(args.dev_patient_manifest),
            "confirmation_locked": _load(args.confirmation_patient_manifest),
        }
    build = build_phase1_contract(
        _load(args.dev_payload), _load(args.confirmation_payload), _load(args.config),
        external_patient_manifests=manifests,
    )
    args.multiplier_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_multiplier = args.multiplier_output.with_name(
        args.multiplier_output.stem + ".tmp.npz"
    )
    np.savez_compressed(
        temporary_multiplier,
        cluster_order=np.asarray(build.cluster_order, dtype=str),
        multipliers=build.multipliers,
    )
    temporary_multiplier.replace(args.multiplier_output)
    _atomic_json(args.output, build.artifact)
    print(json.dumps({
        "status": STATUS, "authorized": False,
        "output": str(args.output.resolve()),
        "artifact_sha256": build.artifact["artifact_sha256"],
        "multiplier_output": str(args.multiplier_output.resolve()),
        "multiplier_file_sha256": sha256_file(args.multiplier_output),
    }, indent=2))


if __name__ == "__main__":
    main()
