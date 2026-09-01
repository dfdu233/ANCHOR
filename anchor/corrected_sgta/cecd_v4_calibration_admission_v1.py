#!/usr/bin/env python3
"""Fail-closed calibration admission substrate for CECD v4.

This module is outcome-blind with respect to confirmation results: every gate,
seed, and threshold is supplied by a hash-bound configuration.  It is also
deliberately non-authorizing.  Aggregate 0/1/2/3 reader-vote counts can define
the soft calibration target, but are explicitly never represented as a
named-reader leave-one-reader-out (LORO) analysis.

The substrate closes three numerical failure modes before a future v4
authorizer may be written:

* raw canonical scores must pass a four-bin directionality check before an
  increasing isotonic map is fit;
* confirmation score families are checked against each dev calibration
  support, with neither clipping nor extrapolation;
* the additive-loss denominator B0 must remain away from zero under a
  strictly-positive whole-cluster multiplier bootstrap.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

from .analyze_clinical_equivalence_composition_defect_v1 import (
    ContractError,
    build_orbits,
    sha256_file,
    validate_payload,
)


VERSION = "cecd-v4-calibration-admission-v1.0"
BUNDLE_VERSION = "cecd-v4-dev-calibration-admission-bundle-v1"
REPORT_VERSION = "cecd-v4-confirmation-calibration-admission-report-v1"
NON_AUTHORIZING = (
    "outcome_blind_non_authorizing_substrate_only; cannot authorize CECD v4, "
    "mitigation, a mechanism claim, or named-reader inference"
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    output = dict(payload)
    output[field] = object_sha256(output)
    return output


def _verify_seal(payload: Mapping[str, Any], field: str) -> None:
    claimed = payload.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ContractError(f"missing or invalid {field}")
    unsealed = dict(payload)
    del unsealed[field]
    if object_sha256(unsealed) != claimed:
        raise ContractError(f"{field} mismatch: artifact changed after freezing")


def _module_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be an integer >= {minimum}")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{name} must be an integer >= {minimum}") from error
    if result < minimum or float(result) != float(value):
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return result


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all frozen decisions; no scientific threshold has a code default."""

    if config.get("schema_version") != "cecd-v4-calibration-admission-contract-v1":
        raise ContractError("wrong calibration-admission config schema")
    if config.get("authorized") is not False or config.get("frozen_before_confirmation_outputs") is not True:
        raise ContractError("calibration-admission config must be frozen and non-authorizing")
    if config.get("promotion_effect") != "none; this artifact cannot authorize v4 or replace named-reader LORO":
        raise ContractError("config must forbid promotion")

    reader = config.get("reader_evidence")
    if not isinstance(reader, Mapping):
        raise ContractError("reader_evidence config is required")
    if reader.get("source") != "aggregate_vote_count_only":
        raise ContractError("this substrate accepts aggregate vote counts only")
    if list(reader.get("vote_bins", ())) != [0, 1, 2, 3] or int(reader.get("reader_count", -1)) != 3:
        raise ContractError("reader evidence must be the fixed 0/1/2/3 three-reader count")
    if reader.get("named_reader_loro_computed") is not False:
        raise ContractError("aggregate-vote config cannot claim named-reader LORO")
    if reader.get("aggregate_vote_may_substitute_for_named_reader_loro") is not False:
        raise ContractError("aggregate votes may not substitute for named-reader LORO")

    directional = config.get("directional_admission")
    calibration = config.get("calibration")
    denominator = config.get("denominator_regularity")
    if not all(isinstance(value, Mapping) for value in (directional, calibration, denominator)):
        raise ContractError("directional, calibration, and denominator config sections are required")
    if directional.get("location_statistic") != "arithmetic_mean_of_raw_canonical_score":
        raise ContractError("raw directionality location statistic is not frozen")
    if directional.get("require_strict_raw_bin_order") is not True:
        raise ContractError("strict raw vote-bin ordering must be required")
    _integer(directional.get("minimum_per_vote_bin"), "minimum_per_vote_bin", 2)
    _integer(directional.get("bootstrap_draws"), "directional bootstrap_draws", 19)
    confidence = _number(directional.get("bootstrap_confidence"), "bootstrap_confidence")
    if not 0.5 < confidence < 1.0:
        raise ContractError("bootstrap_confidence must lie in (0.5, 1)")
    _integer(directional.get("bootstrap_seed"), "directional bootstrap_seed", 0)
    for key in (
        "minimum_raw_adjacent_contrast", "minimum_spearman_rank_correlation",
        "minimum_bootstrap_lower_ordinal_slope", "minimum_valid_bootstrap_fraction",
    ):
        _number(directional.get(key), f"directional_admission.{key}")
    swap = directional.get("same_support_swap")
    if not isinstance(swap, Mapping) or swap.get("pairing") != "sha256_ordered_disjoint_pairs_within_model_finding_vote_bin":
        raise ContractError("same-support swap rule is not frozen")
    quantile = _number(swap.get("absolute_drift_quantile"), "absolute_drift_quantile")
    if not 0.5 <= quantile < 1.0:
        raise ContractError("absolute_drift_quantile must lie in [0.5, 1)")
    for key in (
        "maximum_drift_quantile_over_full_support_span",
        "minimum_full_support_span_minus_drift_quantile",
    ):
        _number(swap.get(key), f"same_support_swap.{key}")

    if calibration.get("family") != "isotonic_piecewise_linear_no_extrapolation":
        raise ContractError("only frozen no-extrapolation isotonic calibration is supported")
    if calibration.get("fit_split") != "dev" or calibration.get("apply_split") != "confirmation":
        raise ContractError("calibration must be dev-fit and confirmation-apply-only")
    _integer(calibration.get("folds"), "calibration.folds", 2)
    if calibration.get("fold_assignment") != "sha256_whole_cluster_modulo_k":
        raise ContractError("whole-cluster fold assignment is required")
    if calibration.get("tie_handling") != "sklearn_isotonic_secondary_method_average":
        raise ContractError("isotonic tie handling is not frozen")
    _number(calibration.get("probability_floor_for_nll_only"), "probability_floor_for_nll_only")
    families = calibration.get("required_confirmation_score_families")
    if not isinstance(families, list) or not families or len(set(families)) != len(families):
        raise ContractError("required confirmation score families must be unique and nonempty")
    overlap = calibration.get("support_overlap")
    if not isinstance(overlap, Mapping):
        raise ContractError("support_overlap config is required")
    if overlap.get("clipping_permitted") is not False or overlap.get("extrapolation_permitted") is not False:
        raise ContractError("clipping and extrapolation must both be forbidden")
    for key in ("maximum_outside_fraction", "maximum_normalized_tail_distance"):
        value = _number(overlap.get(key), f"support_overlap.{key}")
        if value < 0:
            raise ContractError(f"support_overlap.{key} must be nonnegative")
    _integer(
        overlap.get("minimum_in_support_count_per_model_finding_family"),
        "minimum_in_support_count_per_model_finding_family", 1,
    )

    if denominator.get("estimand") != "ratio_of_model_specific_16_stratum_macro_means":
        raise ContractError("B0 estimand is not frozen")
    if denominator.get("multiplier") != "shared_strictly_positive_keyed_exponential":
        raise ContractError("B0 bootstrap must use strictly-positive multipliers")
    _integer(denominator.get("bootstrap_draws"), "denominator bootstrap_draws", 19)
    _integer(denominator.get("bootstrap_seed"), "denominator bootstrap_seed", 0)
    for key in (
        "minimum_macro_b0", "minimum_stratum_b0", "minimum_bootstrap_b0_quantile",
        "bootstrap_guard_quantile", "maximum_bootstrap_coefficient_of_variation",
        "maximum_near_zero_draw_fraction",
    ):
        value = _number(denominator.get(key), f"denominator_regularity.{key}")
        if value < 0:
            raise ContractError(f"denominator_regularity.{key} must be nonnegative")
    guard_q = float(denominator["bootstrap_guard_quantile"])
    if not 0.0 < guard_q < 0.5:
        raise ContractError("bootstrap_guard_quantile must lie in (0, 0.5)")
    epsilon = _number(config.get("numeric_epsilon"), "numeric_epsilon")
    if epsilon <= 0:
        raise ContractError("numeric_epsilon must be positive")
    return json.loads(json.dumps(config))


def _global_clusters(contract: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    by_image: dict[str, set[str]] = defaultdict(set)
    missing = False
    for rows in contract["by_orbit"].values():
        for row in rows:
            patient = row.get("patient_id")
            if patient in (None, ""):
                missing = True
            else:
                by_image[str(row["image_id"])].add(str(patient))
    conflicts = {image: values for image, values in by_image.items() if len(values) > 1}
    if conflicts:
        raise ContractError(f"conflicting patient IDs for image(s): {sorted(conflicts)[:3]}")
    images = {str(row["image_id"]) for rows in contract["by_orbit"].values() for row in rows}
    complete = not missing and images == set(by_image)
    declared_patients = sorted({patient for values in by_image.values() for patient in values})
    if complete:
        mapping = {
            image: f"patient:{next(iter(by_image[image]))}" for image in sorted(images)
        }
        mode = "patient"
    else:
        mapping = {image: f"image:{image}" for image in sorted(images)}
        mode = "image"
    return mapping, {
        "mode": mode,
        "declared_patient_ids": declared_patients,
        "image_ids": sorted(images),
    }


def _orbits(
    payload: Mapping[str, Any], expected_stage: str, expected_source: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    contract = validate_payload(payload)
    if contract["split"] != expected_stage or contract["source_manifest_split"] != expected_source:
        raise ContractError(f"expected {expected_stage}/{expected_source} payload")
    cluster_map, cluster_audit = _global_clusters(contract)
    output = []
    for raw in build_orbits(contract):
        row = dict(raw)
        row["cluster_id"] = cluster_map[str(row["image_id"])]
        output.append(row)
    return contract, output, cluster_audit


def _clean_score(orbit: Mapping[str, Any], contract: Mapping[str, Any]) -> float:
    r = list(contract["primary_renders"]).index(contract["baseline_render"])
    p = list(contract["primary_prompts"]).index(contract["baseline_prompt"])
    return float(np.asarray(orbit["score"], dtype=float)[r, p])


def _spearman(votes: np.ndarray, scores: np.ndarray) -> float:
    rv = rankdata(np.asarray(votes, dtype=float), method="average")
    rs = rankdata(np.asarray(scores, dtype=float), method="average")
    if np.std(rv) == 0 or np.std(rs) == 0:
        return math.nan
    return float(np.corrcoef(rv, rs)[0, 1])


def _raw_direction_statistics(votes: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    means = np.asarray([np.mean(scores[votes == vote]) for vote in range(4)], dtype=float)
    contrasts = np.diff(means)
    slope = float(np.linalg.lstsq(
        np.column_stack((np.ones(scores.size), votes.astype(float))), scores, rcond=None,
    )[0][1])
    return {
        "raw_bin_means": {str(vote): float(means[vote]) for vote in range(4)},
        "raw_adjacent_contrasts": [float(value) for value in contrasts],
        "minimum_raw_adjacent_contrast": float(np.min(contrasts)),
        "full_support_span": float(means[3] - means[0]),
        "ordinal_slope": slope,
        "spearman_rank_correlation": _spearman(votes, scores),
        "strict_raw_bin_order": bool(np.all(contrasts > 0.0)),
    }


def _same_support_swap(votes: np.ndarray, scores: np.ndarray, cluster_ids: np.ndarray, model: str, finding: str, config: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(config["directional_admission"]["bootstrap_seed"])
    drift: list[float] = []
    counts: dict[str, int] = {}
    for vote in range(4):
        selected = np.flatnonzero(votes == vote).tolist()
        selected.sort(key=lambda index: hashlib.sha256(
            f"{seed}|{model}|{finding}|{vote}|{cluster_ids[index]}".encode()
        ).hexdigest())
        half = len(selected) // 2
        pairs = list(zip(selected[:half], selected[half:half * 2]))
        counts[str(vote)] = len(pairs)
        drift.extend(abs(float(scores[left] - scores[right])) for left, right in pairs)
    if not drift:
        raise ContractError(f"no deterministic same-support swaps for {model}/{finding}")
    q = float(config["directional_admission"]["same_support_swap"]["absolute_drift_quantile"])
    return {
        "pairing": "sha256_ordered_disjoint_pairs_within_model_finding_vote_bin",
        "pairs_by_vote_bin": counts,
        "n_pairs": len(drift),
        "median_absolute_drift": float(np.median(drift)),
        "absolute_drift_quantile_level": q,
        "absolute_drift_quantile": float(np.quantile(drift, q)),
    }


def _bootstrap_direction(votes: np.ndarray, scores: np.ndarray, clusters: np.ndarray, config: Mapping[str, Any], key: str) -> dict[str, Any]:
    section = config["directional_admission"]
    draws = int(section["bootstrap_draws"])
    confidence = float(section["bootstrap_confidence"])
    seed = int(section["bootstrap_seed"])
    digest = int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(digest)
    unique = np.unique(clusters)
    by_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    slopes: list[float] = []
    min_adjacent: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([by_cluster[value] for value in sampled])
        sampled_votes = votes[indices]
        if set(sampled_votes.tolist()) != {0, 1, 2, 3}:
            continue
        stats = _raw_direction_statistics(sampled_votes, scores[indices])
        slopes.append(float(stats["ordinal_slope"]))
        min_adjacent.append(float(stats["minimum_raw_adjacent_contrast"]))
    alpha = (1.0 - confidence) / 2.0
    if not slopes:
        return {"draws": draws, "valid_draws": 0, "valid_fraction": 0.0, "ordinal_slope_ci": None, "minimum_adjacent_contrast_ci": None}
    return {
        "draws": draws,
        "valid_draws": len(slopes),
        "valid_fraction": len(slopes) / draws,
        "ordinal_slope_ci": [float(np.quantile(slopes, alpha)), float(np.quantile(slopes, 1.0 - alpha))],
        "minimum_adjacent_contrast_ci": [float(np.quantile(min_adjacent, alpha)), float(np.quantile(min_adjacent, 1.0 - alpha))],
        "cluster_unit": "globally_patient_when_complete_else_globally_image",
    }


def _fit_isotonic(scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if scores.size < 8 or scores.size != targets.size or not np.isfinite(scores).all() or not np.isfinite(targets).all():
        raise ContractError("isotonic fit requires at least eight finite score-target pairs")
    if np.unique(scores).size < 2:
        raise ContractError("isotonic fit requires at least two unique raw scores")
    fitted = IsotonicRegression(increasing=True, out_of_bounds="raise", y_min=0.0, y_max=1.0)
    fitted.fit(scores, targets)
    x = np.asarray(fitted.X_thresholds_, dtype=float)
    y = np.asarray(fitted.y_thresholds_, dtype=float)
    if x.size < 2 or np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) < 0.0):
        raise ContractError("invalid isotonic fit")
    return {
        "kind": "isotonic_piecewise_linear_no_extrapolation",
        "x_thresholds": x.tolist(),
        "y_thresholds": y.tolist(),
        "support_min": float(x[0]),
        "support_max": float(x[-1]),
        "n_fit": int(scores.size),
        "clipping_permitted": False,
        "extrapolation_permitted": False,
    }


def strict_calibrated_probabilities(calibrator: Mapping[str, Any], scores: Sequence[float]) -> np.ndarray:
    """Apply only inside dev support; fail rather than silently clipping."""

    x = np.asarray(calibrator.get("x_thresholds"), dtype=float)
    y = np.asarray(calibrator.get("y_thresholds"), dtype=float)
    values = np.asarray(scores, dtype=float)
    if calibrator.get("kind") != "isotonic_piecewise_linear_no_extrapolation" or x.ndim != 1 or x.size < 2 or y.shape != x.shape:
        raise ContractError("malformed no-extrapolation calibrator")
    if not np.isfinite(values).all() or np.any(values < x[0]) or np.any(values > x[-1]):
        raise ContractError("score lies outside dev calibration support; clipping is forbidden")
    return np.interp(values, x, y)


def _oof_diagnostic(scores: np.ndarray, votes: np.ndarray, clusters: np.ndarray, config: Mapping[str, Any], key: str) -> dict[str, Any]:
    folds = int(config["calibration"]["folds"])
    floor = float(config["calibration"]["probability_floor_for_nll_only"])
    seed = int(config["directional_admission"]["bootstrap_seed"])
    assignment = np.asarray([
        int(hashlib.sha256(f"{seed}|{key}|{cluster}".encode()).hexdigest()[:16], 16) % folds
        for cluster in clusters
    ])
    prediction = np.full(scores.size, np.nan)
    outside = np.zeros(scores.size, dtype=bool)
    fold_support: dict[str, Any] = {}
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        counts = {str(vote): int(np.sum(votes[train] == vote)) for vote in range(4)}
        if not np.any(test) or min(counts.values()) == 0:
            fold_support[str(fold)] = {"train_vote_bin_counts": counts, "test_n": int(np.sum(test)), "complete": False}
            continue
        fitted = _fit_isotonic(scores[train], votes[train] / 3.0)
        supported = (scores[test] >= fitted["support_min"]) & (scores[test] <= fitted["support_max"])
        target_indices = np.flatnonzero(test)
        outside[target_indices[~supported]] = True
        if np.any(supported):
            prediction[target_indices[supported]] = strict_calibrated_probabilities(fitted, scores[target_indices[supported]])
        fold_support[str(fold)] = {
            "train_vote_bin_counts": counts,
            "test_n": int(np.sum(test)),
            "in_support_n": int(np.sum(supported)),
            "outside_n": int(np.sum(~supported)),
            "complete": True,
        }
    keep = np.isfinite(prediction)
    if np.any(keep):
        target = votes[keep] / 3.0
        clipped_for_log = np.clip(prediction[keep], floor, 1.0 - floor)
        design = np.column_stack((np.ones(np.sum(keep)), prediction[keep]))
        intercept, slope = np.linalg.lstsq(design, target, rcond=None)[0]
        brier = float(np.mean((prediction[keep] - target) ** 2))
        nll = float(np.mean(-(target * np.log(clipped_for_log) + (1.0 - target) * np.log(1.0 - clipped_for_log))))
    else:
        intercept = slope = brier = nll = math.nan
    return {
        "folds": folds,
        "fold_support": fold_support,
        "prediction_coverage": float(np.mean(keep)),
        "outside_support_fraction": float(np.mean(outside)),
        "brier_on_in_support_predictions": brier,
        "soft_bernoulli_nll_on_in_support_predictions": nll,
        "linear_reliability_intercept": float(intercept),
        "linear_reliability_slope": float(slope),
        "complete_without_clipping": bool(np.all(keep)),
    }


def fit_dev_admission(payload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Fit a sealed dev-only calibration bundle and raw directional diagnostics."""

    frozen = validate_config(config)
    contract, orbits, cluster_audit = _orbits(payload, "dev_fit", "dev")
    direction_config = frozen["directional_admission"]
    minimum = int(direction_config["minimum_per_vote_bin"])
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in orbits:
        groups[(str(row["model"]), str(row["finding"]))].append(row)
    if not groups:
        raise ContractError("dev payload has no model/finding groups")

    calibrators: dict[str, dict[str, Any]] = defaultdict(dict)
    admissions: dict[str, dict[str, Any]] = defaultdict(dict)
    for (model, finding), rows in sorted(groups.items()):
        votes = np.asarray([int(row["reader_votes"]) for row in rows], dtype=int)
        scores = np.asarray([_clean_score(row, contract) for row in rows], dtype=float)
        clusters = np.asarray([str(row["cluster_id"]) for row in rows], dtype=object)
        counts = {str(vote): int(np.sum(votes == vote)) for vote in range(4)}
        if min(counts.values()) < minimum:
            raise ContractError(f"dev vote-bin quota fails for {model}/{finding}: {counts}")
        raw = _raw_direction_statistics(votes, scores)
        swap = _same_support_swap(votes, scores, clusters, model, finding, frozen)
        bootstrap = _bootstrap_direction(votes, scores, clusters, frozen, f"{model}|{finding}")
        full_span = float(raw["full_support_span"])
        drift = float(swap["absolute_drift_quantile"])
        ratio = math.inf if full_span <= 0.0 else drift / full_span
        gates = {
            "vote_bin_quota": min(counts.values()) >= minimum,
            "strict_raw_bin_order": bool(raw["strict_raw_bin_order"]),
            "minimum_raw_adjacent_contrast": float(raw["minimum_raw_adjacent_contrast"]) > float(direction_config["minimum_raw_adjacent_contrast"]),
            "minimum_spearman_rank_correlation": math.isfinite(float(raw["spearman_rank_correlation"])) and float(raw["spearman_rank_correlation"]) > float(direction_config["minimum_spearman_rank_correlation"]),
            "bootstrap_valid_fraction": float(bootstrap["valid_fraction"]) >= float(direction_config["minimum_valid_bootstrap_fraction"]),
            "bootstrap_lower_ordinal_slope": bootstrap["ordinal_slope_ci"] is not None and float(bootstrap["ordinal_slope_ci"][0]) > float(direction_config["minimum_bootstrap_lower_ordinal_slope"]),
            "same_support_drift_ratio": ratio <= float(direction_config["same_support_swap"]["maximum_drift_quantile_over_full_support_span"]),
            "full_span_exceeds_same_support_drift": full_span - drift > float(direction_config["same_support_swap"]["minimum_full_support_span_minus_drift_quantile"]),
        }
        gates["passed"] = all(gates.values())
        admissions[model][finding] = {
            "reader_vote_bin_counts": counts,
            "raw_directionality": raw,
            "same_support_image_swap": {
                **swap,
                "drift_quantile_over_full_support_span": ratio,
                "full_support_span_minus_drift_quantile": full_span - drift,
            },
            "cluster_bootstrap": bootstrap,
            "gates": gates,
        }
        fitted = _fit_isotonic(scores, votes / 3.0)
        fitted["fit_split"] = "dev"
        fitted["reader_target"] = "aggregate_reader_votes_divided_by_three"
        fitted["raw_directional_admission_passed"] = bool(gates["passed"])
        fitted["oof_diagnostic"] = _oof_diagnostic(scores, votes, clusters, frozen, f"{model}|{finding}")
        calibrators[model][finding] = fitted

    geometry = {
        name: list(contract[name]) if isinstance(contract[name], tuple) else contract[name]
        for name in (
            "primary_renders", "primary_prompts", "baseline_render", "baseline_prompt",
            "identity_render", "duplicate_prompt",
        )
    }
    artifact = {
        "version": BUNDLE_VERSION,
        "status": NON_AUTHORIZING,
        "authorized": False,
        "source_sha256": _module_sha256(),
        "config_sha256": object_sha256(frozen),
        "dev_input_sha256": object_sha256(payload),
        "fit_split": "dev",
        "apply_split": "confirmation",
        "geometry": geometry,
        "dev_cluster_ids": sorted({str(row["cluster_id"]) for row in orbits}),
        "dev_cluster_audit": cluster_audit,
        "directional_admission": {model: dict(values) for model, values in admissions.items()},
        "all_model_finding_directional_gates_passed": all(
            record["gates"]["passed"]
            for model_records in admissions.values() for record in model_records.values()
        ),
        "calibrators": {model: dict(values) for model, values in calibrators.items()},
        "reader_inference_boundary": {
            "input_used": "aggregate_vote_count_0_1_2_3_only",
            "named_reader_ids_used": False,
            "named_reader_loro_computed": False,
            "aggregate_vote_substitutes_for_named_reader_loro": False,
            "promotion_requires_separate_hash_bound_named_reader_loro": True,
        },
    }
    return _seal(artifact, "bundle_sha256")


def _validate_bundle(bundle: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = validate_config(config)
    if bundle.get("version") != BUNDLE_VERSION or bundle.get("status") != NON_AUTHORIZING or bundle.get("authorized") is not False:
        raise ContractError("wrong or authorizing calibration-admission bundle")
    _verify_seal(bundle, "bundle_sha256")
    if bundle.get("source_sha256") != _module_sha256():
        raise ContractError("bundle source differs from current calibration-admission module")
    if bundle.get("config_sha256") != object_sha256(frozen):
        raise ContractError("confirmation config differs from dev-frozen config")
    boundary = bundle.get("reader_inference_boundary", {})
    if boundary.get("named_reader_loro_computed") is not False or boundary.get("aggregate_vote_substitutes_for_named_reader_loro") is not False:
        raise ContractError("bundle falsely represents aggregate votes as named-reader LORO")
    return frozen


def _score_families(orbits: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[float]]:
    output: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in orbits:
        model, finding = str(row["model"]), str(row["finding"])
        actual = np.asarray(row["score"], dtype=float)
        additive = actual - np.asarray(row["interaction"], dtype=float)
        grand = float(actual.mean())
        render_axis = actual.mean(axis=1, keepdims=True) - grand + grand
        render_axis = np.repeat(render_axis, actual.shape[1], axis=1)
        prompt_axis = actual.mean(axis=0, keepdims=True) - grand + grand
        prompt_axis = np.repeat(prompt_axis, actual.shape[0], axis=0)
        for family, values in (
            ("actual", actual), ("additive", additive),
            ("render_axis", render_axis), ("prompt_axis", prompt_axis),
        ):
            output[(model, finding, family)].extend(float(value) for value in values.ravel())
    return output


def _support_record(calibrator: Mapping[str, Any], values: Sequence[float], config: Mapping[str, Any]) -> dict[str, Any]:
    scores = np.asarray(values, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ContractError("confirmation score family must be nonempty and finite")
    low, high = float(calibrator["support_min"]), float(calibrator["support_max"])
    span = high - low
    epsilon = float(config["numeric_epsilon"])
    if span <= epsilon:
        raise ContractError("dev calibration support has near-zero width")
    below = np.maximum(low - scores, 0.0)
    above = np.maximum(scores - high, 0.0)
    tail = np.maximum(below, above)
    outside = tail > 0.0
    overlap = config["calibration"]["support_overlap"]
    record = {
        "n": int(scores.size),
        "in_support_n": int(np.sum(~outside)),
        "outside_n": int(np.sum(outside)),
        "outside_fraction": float(np.mean(outside)),
        "maximum_tail_distance": float(np.max(tail)),
        "maximum_normalized_tail_distance": float(np.max(tail) / span),
        "dev_support": [low, high],
        "clipped_values": 0,
        "extrapolated_values": 0,
    }
    record["gates"] = {
        "outside_fraction": record["outside_fraction"] <= float(overlap["maximum_outside_fraction"]),
        "normalized_tail_distance": record["maximum_normalized_tail_distance"] <= float(overlap["maximum_normalized_tail_distance"]),
        "minimum_in_support_count": record["in_support_n"] >= int(overlap["minimum_in_support_count_per_model_finding_family"]),
        "no_clipping": True,
        "no_extrapolation": True,
    }
    record["gates"]["passed"] = all(record["gates"].values())
    return record


def assess_b0_regularity(
    orbit_b0: Sequence[float], strata: Sequence[str], clusters: Sequence[str],
    config: Mapping[str, Any], *, stream_key: str,
) -> dict[str, Any]:
    """Assess a model-specific macro B0 without ever stabilizing it post hoc."""

    frozen = validate_config(config)
    section = frozen["denominator_regularity"]
    values = np.asarray(orbit_b0, dtype=float)
    stratum = np.asarray(strata, dtype=object)
    cluster = np.asarray(clusters, dtype=object)
    if values.ndim != 1 or values.size < 2 or stratum.shape != values.shape or cluster.shape != values.shape:
        raise ContractError("B0 regularity requires aligned one-dimensional orbit arrays")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ContractError("B0 orbit losses must be finite and nonnegative")
    unique_strata = sorted(set(str(value) for value in stratum))
    unique_clusters = sorted(set(str(value) for value in cluster))
    if not unique_strata or len(unique_clusters) < 2:
        raise ContractError("B0 regularity needs strata and at least two clusters")

    stratum_points = {
        key: float(np.mean(values[stratum == key])) for key in unique_strata
    }
    point = float(np.mean(list(stratum_points.values())))
    seed = int(section["bootstrap_seed"])
    digest = int(hashlib.sha256(f"{seed}|{stream_key}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(digest)
    draws = int(section["bootstrap_draws"])
    trace = np.empty(draws, dtype=float)
    for draw in range(draws):
        weights_by_cluster = {
            key: float(rng.exponential(1.0)) for key in unique_clusters
        }
        weights = np.asarray([weights_by_cluster[str(value)] for value in cluster], dtype=float)
        means = []
        for key in unique_strata:
            keep = stratum == key
            means.append(float(np.sum(weights[keep] * values[keep]) / np.sum(weights[keep])))
        trace[draw] = float(np.mean(means))
    guard_quantile = float(section["bootstrap_guard_quantile"])
    low = float(np.quantile(trace, guard_quantile))
    mean = float(np.mean(trace))
    sd = float(np.std(trace, ddof=1))
    cv = math.inf if mean <= float(frozen["numeric_epsilon"]) else sd / mean
    minimum_macro = float(section["minimum_macro_b0"])
    gates = {
        "macro_b0": point >= minimum_macro,
        "minimum_stratum_b0": min(stratum_points.values()) >= float(section["minimum_stratum_b0"]),
        "bootstrap_low_quantile": low >= float(section["minimum_bootstrap_b0_quantile"]),
        "bootstrap_coefficient_of_variation": cv <= float(section["maximum_bootstrap_coefficient_of_variation"]),
        "near_zero_draw_fraction": float(np.mean(trace < minimum_macro)) <= float(section["maximum_near_zero_draw_fraction"]),
    }
    gates["ratio_authorizable_from_denominator_only"] = all(gates.values())
    return {
        "macro_b0": point,
        "stratum_b0": stratum_points,
        "bootstrap": {
            "draws": draws,
            "guard_quantile_level": guard_quantile,
            "guard_quantile": low,
            "mean": mean,
            "standard_deviation": sd,
            "coefficient_of_variation": cv,
            "near_zero_draw_fraction": float(np.mean(trace < minimum_macro)),
            "minimum": float(np.min(trace)),
            "maximum": float(np.max(trace)),
            "multiplier": "shared_strictly_positive_keyed_exponential",
        },
        "gates": gates,
        "failure_policy": "ratio_non_authorizing_keep_absolute_theta_only" if not gates["ratio_authorizable_from_denominator_only"] else "denominator_only_pass_does_not_authorize_v4",
    }


def assess_confirmation(
    payload: Mapping[str, Any], bundle: Mapping[str, Any], config: Mapping[str, Any],
    *, additional_score_families: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Apply the frozen support and B0 guards to confirmation, without refit."""

    frozen = _validate_bundle(bundle, config)
    contract, orbits, confirmation_cluster_audit = _orbits(
        payload, "confirmation_locked", "confirmation"
    )
    geometry = {
        name: list(contract[name]) if isinstance(contract[name], tuple) else contract[name]
        for name in (
            "primary_renders", "primary_prompts", "baseline_render", "baseline_prompt",
            "identity_render", "duplicate_prompt",
        )
    }
    if geometry != bundle.get("geometry"):
        raise ContractError("confirmation geometry differs from dev")
    overlap = set(bundle.get("dev_cluster_ids", ())) & {str(row["cluster_id"]) for row in orbits}
    if overlap:
        raise ContractError(f"dev/confirmation cluster overlap: {sorted(overlap)[:3]}")
    dev_cluster_audit = bundle.get("dev_cluster_audit")
    if not isinstance(dev_cluster_audit, Mapping):
        raise ContractError("dev cluster audit is absent from the frozen bundle")
    if dev_cluster_audit.get("mode") != confirmation_cluster_audit.get("mode"):
        raise ContractError("dev/confirmation global cluster mode differs")
    patient_overlap = set(dev_cluster_audit.get("declared_patient_ids", ())) & set(
        confirmation_cluster_audit.get("declared_patient_ids", ())
    )
    if patient_overlap:
        raise ContractError(
            f"dev/confirmation declared patient overlap: {sorted(patient_overlap)[:3]}"
        )
    image_overlap = set(dev_cluster_audit.get("image_ids", ())) & set(
        confirmation_cluster_audit.get("image_ids", ())
    )
    if image_overlap:
        raise ContractError(f"dev/confirmation image overlap: {sorted(image_overlap)[:3]}")

    families = _score_families(orbits)
    for raw in additional_score_families:
        if not isinstance(raw, Mapping):
            raise ContractError("additional score family rows must be objects")
        model, finding, family = (str(raw.get(key, "")) for key in ("model", "finding", "family"))
        values = raw.get("scores")
        if not model or not finding or not family or not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ContractError("malformed additional score family")
        key = (model, finding, family)
        if key in families:
            raise ContractError(f"duplicate confirmation score family {key}")
        families[key] = [_number(value, f"{key} score") for value in values]

    required = set(str(value) for value in frozen["calibration"]["required_confirmation_score_families"])
    support: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    calibrators = bundle.get("calibrators", {})
    for model, finding_records in calibrators.items():
        for finding, calibrator in finding_records.items():
            present = {family for candidate_model, candidate_finding, family in families if candidate_model == model and candidate_finding == finding}
            missing = required - present
            if missing:
                raise ContractError(f"missing required confirmation score families for {model}/{finding}: {sorted(missing)}")
            for family in sorted(present):
                support[model][finding][family] = _support_record(
                    calibrator, families[(model, finding, family)], frozen,
                )
    expected_pairs = {(str(row["model"]), str(row["finding"])) for row in orbits}
    calibration_pairs = {(str(model), str(finding)) for model, records in calibrators.items() for finding in records}
    if expected_pairs != calibration_pairs:
        raise ContractError("confirmation model/finding set differs from dev calibrators")

    all_required_support = all(
        support[model][finding][family]["gates"]["passed"]
        for model, finding in sorted(expected_pairs) for family in required
    )

    b0: dict[str, Any] = {}
    if all_required_support:
        by_model: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        for row in orbits:
            model, finding = str(row["model"]), str(row["finding"])
            additive = np.asarray(row["score"], dtype=float) - np.asarray(row["interaction"], dtype=float)
            probability = strict_calibrated_probabilities(calibrators[model][finding], additive.ravel())
            target = float(row["reader_votes"]) / 3.0
            by_model[model]["value"].append(float(np.mean((probability - target) ** 2)))
            by_model[model]["stratum"].append(f"{finding}|{int(row['reader_votes'])}")
            by_model[model]["cluster"].append(str(row["cluster_id"]))
        for model, values in sorted(by_model.items()):
            b0[model] = assess_b0_regularity(
                values["value"], values["stratum"], values["cluster"], frozen,
                stream_key="confirmation|shared_across_models",
            )
    else:
        for model in sorted(calibrators):
            b0[model] = {
                "status": "not_computed_due_to_calibration_support_failure",
                "gates": {"ratio_authorizable_from_denominator_only": False},
                "failure_policy": "ratio_non_authorizing_keep_absolute_theta_only",
            }

    report = {
        "version": REPORT_VERSION,
        "status": NON_AUTHORIZING,
        "authorized": False,
        "source_sha256": _module_sha256(),
        "config_sha256": object_sha256(frozen),
        "bundle_sha256": bundle["bundle_sha256"],
        "confirmation_input_sha256": object_sha256(payload),
        "apply_only_no_refit": True,
        "calibration_support": {model: {finding: dict(values) for finding, values in records.items()} for model, records in support.items()},
        "all_required_score_families_inside_frozen_dev_support": all_required_support,
        "b0_regularity": b0,
        "b0_multiplier_pairing": {
            "same_stream_key_for_every_model": True,
            "stream_key": "confirmation|shared_across_models",
        },
        "reader_inference_boundary": {
            "input_used": "aggregate_vote_count_0_1_2_3_only",
            "named_reader_loro_computed": False,
            "aggregate_vote_substitutes_for_named_reader_loro": False,
            "named_reader_loro_gate": "absent_fail_closed",
        },
        "promotion_effect": "none",
    }
    return _seal(report, "report_sha256")
