#!/usr/bin/env python3
"""Outcome-blind CECD product-attributable clinical-risk prototype.

This module is deliberately separate from the frozen v3 gate.  It fits a
dev-only monotone score-to-reader map and evaluates, apply-only, the change in
proper loss after removing the two-way-centered product component.  Every
output is diagnostic and non-authorizing.

The confirmatory diagnostic is Brier PAEL against centered-subspace spectral
Haar stress rotations, aggregated orbit-first and then equally across
finding-by-reader-vote strata.  Matched-orbit exchange, cell-coordinate
permutation, whole-orbit signs, and NLL are sensitivities only.  No stress null
is presented as an exact semantic randomization test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .analyze_clinical_equivalence_composition_defect_v1 import (
    ContractError,
    build_orbits,
    sha256_file,
    two_way_centered,
    validate_payload,
)


VERSION = "cecd-product-attributable-risk-v4-prototype.1"
BUNDLE_VERSION = "cecd-product-risk-dev-calibration-v4-prototype.1"
RESULT_VERSION = "cecd-product-risk-confirmation-diagnostic-v4-prototype.1"
NON_AUTHORIZING = (
    "diagnostic_only_non_authorizing; cannot authorize model scoring, hidden-state "
    "work, mitigation, a positive CECD claim, or replacement of the frozen v3 gate"
)
EPS = 1e-7


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = object_sha256(result)
    return result


def _verify_seal(payload: Mapping[str, Any], field: str) -> None:
    claimed = payload.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ContractError(f"missing or invalid {field}")
    unsealed = dict(payload)
    del unsealed[field]
    if object_sha256(unsealed) != claimed:
        raise ContractError(f"{field} mismatch: artifact was modified after freezing")


def _module_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _orbit_cluster_map(contract: Mapping[str, Any]) -> dict[tuple[str, str, str], str]:
    output: dict[tuple[str, str, str], str] = {}
    for key, rows in contract["by_orbit"].items():
        patients = {
            str(row["patient_id"]) for row in rows
            if row.get("patient_id") not in (None, "")
        }
        if len(patients) > 1:
            raise ContractError(f"patient_id changes inside orbit {key}")
        output[key] = next(iter(patients)) if patients else str(key[1])
    return output


def _decorate_orbits(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    cluster = _orbit_cluster_map(contract)
    output = []
    for raw in build_orbits(contract):
        row = dict(raw)
        row["cluster_id"] = cluster[(row["model"], row["image_id"], row["finding"])]
        output.append(row)
    return output


def _clean_score(orbit: Mapping[str, Any], contract: Mapping[str, Any]) -> float:
    renders = list(contract["primary_renders"])
    prompts = list(contract["primary_prompts"])
    return float(
        orbit["score"][renders.index(contract["baseline_render"]), prompts.index(contract["baseline_prompt"])]
    )


def _clean_additive_score(orbit: Mapping[str, Any], contract: Mapping[str, Any]) -> float:
    renders = list(contract["primary_renders"])
    prompts = list(contract["primary_prompts"])
    r = renders.index(contract["baseline_render"])
    p = prompts.index(contract["baseline_prompt"])
    return float(orbit["score"][r, p] - orbit["interaction"][r, p])


def _fit_isotonic(scores: Sequence[float], targets: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(scores, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.size < 8 or x.size != y.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ContractError("monotone calibration needs at least eight finite pairs")
    if np.var(x) <= 1e-12 or float(np.cov(x, y, bias=True)[0, 1]) <= 1e-10:
        raise ContractError("non-positive dev score-to-reader relation")
    model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(x, y)
    xt = np.asarray(model.X_thresholds_, dtype=float)
    yt = np.asarray(model.y_thresholds_, dtype=float)
    if xt.size < 2 or np.any(np.diff(xt) <= 0) or np.any(np.diff(yt) < -1e-12):
        raise ContractError("invalid fitted monotone calibration")
    return {
        "kind": "isotonic_piecewise_linear_clip",
        "x_thresholds": xt.tolist(),
        "y_thresholds": yt.tolist(),
        "n_fit": int(x.size),
    }


def apply_calibrator(bundle: Mapping[str, Any], scores: np.ndarray) -> np.ndarray:
    if bundle.get("kind") != "isotonic_piecewise_linear_clip":
        raise ContractError("unknown monotone calibrator kind")
    x = np.asarray(bundle.get("x_thresholds"), dtype=float)
    y = np.asarray(bundle.get("y_thresholds"), dtype=float)
    if x.ndim != 1 or x.size < 2 or y.shape != x.shape:
        raise ContractError("malformed monotone calibrator")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ContractError("non-finite monotone calibrator")
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) < -1e-12):
        raise ContractError("non-monotone calibration artifact")
    return np.interp(np.asarray(scores, dtype=float), x, y, left=y[0], right=y[-1])


def _soft_nll(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    q = np.asarray(target, dtype=float)
    return -(q * np.log(p) + (1.0 - q) * np.log(1.0 - p))


def _calibration_diagnostics(probability: np.ndarray, target: np.ndarray, votes: np.ndarray) -> dict[str, Any]:
    p = np.asarray(probability, dtype=float)
    q = np.asarray(target, dtype=float)
    design = np.column_stack((np.ones(p.size), p))
    intercept, slope = np.linalg.lstsq(design, q, rcond=None)[0]
    return {
        "n": int(p.size),
        "brier": float(np.mean((p - q) ** 2)),
        "soft_bernoulli_nll": float(np.mean(_soft_nll(p, q))),
        "linear_reliability_intercept": float(intercept),
        "linear_reliability_slope": float(slope),
        "by_reader_vote": {
            str(v): {
                "n": int(np.sum(votes == v)),
                "mean_prediction": float(np.mean(p[votes == v])),
                "target": v / 3.0,
            }
            for v in range(4) if np.any(votes == v)
        },
    }


def _crossfit_diagnostic(orbits: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], folds: int) -> dict[str, Any]:
    if folds < 2:
        raise ContractError("cross-fitting requires at least two folds")
    scores = np.asarray([_clean_score(row, contract) for row in orbits], dtype=float)
    votes = np.asarray([int(row["reader_votes"]) for row in orbits], dtype=int)
    target = votes / 3.0
    assignments = np.asarray([
        int(hashlib.sha256(str(row["cluster_id"]).encode()).hexdigest()[:16], 16) % folds
        for row in orbits
    ])
    prediction = np.full(scores.size, np.nan, dtype=float)
    for fold in range(folds):
        test = assignments == fold
        if not np.any(test):
            continue
        train = ~test
        if set(votes[train].tolist()) != {0, 1, 2, 3}:
            raise ContractError("cross-fit training fold lacks a reader-vote bin")
        fitted = _fit_isotonic(scores[train], target[train])
        prediction[test] = apply_calibrator(fitted, scores[test])
    if not np.isfinite(prediction).all():
        raise ContractError("cross-fit prediction coverage is incomplete")
    return _calibration_diagnostics(prediction, target, votes)


def _quantile_edges(values: Sequence[float]) -> list[float]:
    x = np.asarray(values, dtype=float)
    if x.size < 8:
        return []
    edges = sorted(set(float(v) for v in np.quantile(x, [1 / 3, 2 / 3])))
    return edges if len(edges) == 2 else []


def _bin(value: float, edges: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=float), value, side="right"))


def _matching_plan(orbits: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    by_key: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in orbits:
        by_key[(str(row["model"]), str(row["finding"]), int(row["reader_votes"]))].append(row)
    output: dict[str, Any] = {}
    for key, rows in sorted(by_key.items()):
        margins = [_clean_additive_score(row, contract) for row in rows]
        energies = [
            0.0 if float(np.linalg.norm(row["interaction"])) <= 1e-12
            else float(np.sqrt(np.mean(np.asarray(row["interaction"]) ** 2)))
            for row in rows
        ]
        margin_edges = _quantile_edges(margins)
        energy_edges = _quantile_edges(energies)
        candidates = ("margin_energy", "margin", "energy", "base")
        selected = None
        for scheme in candidates:
            counts: dict[tuple[int, ...], int] = defaultdict(int)
            for margin, energy in zip(margins, energies):
                parts: list[int] = []
                if "margin" in scheme:
                    parts.append(_bin(margin, margin_edges))
                if "energy" in scheme:
                    parts.append(_bin(energy, energy_edges))
                counts[tuple(parts)] += 1
            if counts and min(counts.values()) >= 2:
                selected = scheme
                break
        if selected is None:
            raise ContractError(f"no complete matched-orbit donor rule for {key}")
        output["|".join(map(str, key))] = {
            "scheme": selected,
            "margin_edges": margin_edges,
            "energy_edges": energy_edges,
            "dev_n": len(rows),
            "fallback_order": list(candidates),
        }
    return output


def fit_dev_calibration(
    payload: Mapping[str, Any], *, folds: int = 4, null_draws: int = 499,
    bootstrap_draws: int = 2000, seed: int = 1729,
) -> dict[str, Any]:
    """Fit and seal the apply-only dev calibration/nuisance artifact."""

    if null_draws < 19 or bootstrap_draws < 19:
        raise ContractError("prototype requires at least 19 null and bootstrap draws")
    contract = validate_payload(payload)
    if contract["split"] != "dev_fit" or contract["source_manifest_split"] != "dev":
        raise ContractError("v4 calibration requires dev_fit on the truthful dev split")
    orbits = _decorate_orbits(contract)
    calibrators: dict[str, Any] = {}
    for model in sorted({str(row["model"]) for row in orbits}):
        calibrators[model] = {}
        for finding in sorted({str(row["finding"]) for row in orbits if row["model"] == model}):
            selected = [row for row in orbits if row["model"] == model and row["finding"] == finding]
            votes = np.asarray([int(row["reader_votes"]) for row in selected], dtype=int)
            if set(votes.tolist()) != {0, 1, 2, 3}:
                raise ContractError(f"all reader-vote bins required for {model}/{finding}")
            scores = np.asarray([_clean_score(row, contract) for row in selected], dtype=float)
            fitted = _fit_isotonic(scores, votes / 3.0)
            fitted["fit_split"] = "dev"
            fitted["crossfit_diagnostic"] = _crossfit_diagnostic(selected, contract, folds)
            calibrators[model][finding] = fitted
    artifact = {
        "version": BUNDLE_VERSION,
        "status": NON_AUTHORIZING,
        "fit_split": "dev",
        "apply_split": "confirmation",
        "dev_input_sha256": object_sha256(payload),
        "source_sha256": _module_sha256(),
        "source_contract": "v1 validate_payload + build_orbits; frozen v3 files unmodified",
        "geometry": {
            name: list(contract[name]) if isinstance(contract[name], tuple) else contract[name]
            for name in (
                "primary_renders", "primary_prompts", "baseline_render", "baseline_prompt",
                "identity_render", "duplicate_prompt",
            )
        },
        "dev_cluster_ids": sorted({str(row["cluster_id"]) for row in orbits}),
        "calibrators": calibrators,
        "matching_plan": _matching_plan(orbits, contract),
        "randomization": {
            "seed": int(seed), "null_draws": int(null_draws),
            "bootstrap_draws": int(bootstrap_draws),
            "matched_exchangeability_assumption": (
                "complete interaction matrices are conditionally exchangeable within the "
                "frozen model/finding/vote/additive-margin/energy fallback stratum"
            ),
            "spectral_haar_is_stress_null_not_exchangeability_test": True,
        },
    }
    return _seal(artifact, "bundle_sha256")


def _validate_bundle(bundle: Mapping[str, Any]) -> None:
    if bundle.get("version") != BUNDLE_VERSION or bundle.get("status") != NON_AUTHORIZING:
        raise ContractError("wrong or authorizing v4 prototype artifact")
    _verify_seal(bundle, "bundle_sha256")
    if bundle.get("source_sha256") != _module_sha256():
        raise ContractError("dev bundle source_sha256 differs from the current v4 module")
    if bundle.get("fit_split") != "dev" or bundle.get("apply_split") != "confirmation":
        raise ContractError("calibration artifact is not dev-fit/apply-only")
    randomization = bundle.get("randomization")
    if not isinstance(randomization, Mapping):
        raise ContractError("missing frozen randomization contract")
    if int(randomization.get("null_draws", 0)) < 19 or int(randomization.get("bootstrap_draws", 0)) < 19:
        raise ContractError("randomization contract is incomplete")


def _check_geometry(contract: Mapping[str, Any], bundle: Mapping[str, Any]) -> None:
    current = {
        name: list(contract[name]) if isinstance(contract[name], tuple) else contract[name]
        for name in (
            "primary_renders", "primary_prompts", "baseline_render", "baseline_prompt",
            "identity_render", "duplicate_prompt",
        )
    }
    if current != bundle.get("geometry"):
        raise ContractError("confirmation factorial geometry differs from dev")


def _center_basis(size: int) -> np.ndarray:
    if size < 2:
        raise ContractError("interaction subspace requires at least two levels")
    projector = np.eye(size) - np.ones((size, size)) / size
    values, vectors = np.linalg.eigh(projector)
    return vectors[:, values > 0.5]


def _haar_orthogonal(size: int, rng: np.random.Generator) -> np.ndarray:
    if size == 1:
        return np.asarray([[rng.choice((-1.0, 1.0))]])
    q, r = np.linalg.qr(rng.normal(size=(size, size)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def spectral_haar_interaction(interaction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotate inside centered row/column subspaces, preserving singular values."""

    value = two_way_centered(np.asarray(interaction, dtype=float))
    ur = _center_basis(value.shape[0])
    up = _center_basis(value.shape[1])
    core = ur.T @ value @ up
    rotated = ur @ (_haar_orthogonal(core.shape[0], rng) @ core @ _haar_orthogonal(core.shape[1], rng).T) @ up.T
    if not np.allclose(rotated.sum(axis=0), 0.0, atol=1e-9) or not np.allclose(rotated.sum(axis=1), 0.0, atol=1e-9):
        raise RuntimeError("spectral stress null left the centered subspace")
    return rotated


def cell_coordinate_interaction(interaction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    value = np.asarray(interaction, dtype=float)
    norm = float(np.linalg.norm(value))
    candidate = two_way_centered(rng.permutation(value.ravel()).reshape(value.shape))
    candidate_norm = float(np.linalg.norm(candidate))
    if norm <= 1e-14:
        return np.zeros_like(value)
    if candidate_norm <= 1e-14:
        return spectral_haar_interaction(value, rng)
    return candidate * (norm / candidate_norm)


def _stratum(row: Mapping[str, Any], contract: Mapping[str, Any], plan: Mapping[str, Any]) -> tuple[Any, ...]:
    key = f"{row['model']}|{row['finding']}|{int(row['reader_votes'])}"
    rule = plan.get(key)
    if not isinstance(rule, Mapping):
        raise ContractError(f"confirmation has no frozen matching rule for {key}")
    scheme = str(rule.get("scheme"))
    parts: list[Any] = [row["model"], row["finding"], int(row["reader_votes"])]
    if "margin" in scheme:
        parts.append(_bin(_clean_additive_score(row, contract), rule.get("margin_edges", ())))
    if "energy" in scheme:
        energy = float(np.sqrt(np.mean(np.asarray(row["interaction"]) ** 2)))
        if float(np.linalg.norm(row["interaction"])) <= 1e-12:
            energy = 0.0
        parts.append(_bin(energy, rule.get("energy_edges", ())))
    return tuple(parts)


def _matched_donors(
    orbits: Sequence[Mapping[str, Any]], contract: Mapping[str, Any],
    plan: Mapping[str, Any], rng: np.random.Generator,
) -> list[np.ndarray]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(orbits):
        groups[_stratum(row, contract, plan)].append(index)
    donors: list[np.ndarray | None] = [None] * len(orbits)
    for stratum, indices in groups.items():
        if len(indices) < 2:
            raise ContractError(f"incomplete confirmation matched-orbit stratum: {stratum}")
        offset = int(rng.integers(1, len(indices)))
        permuted = indices[offset:] + indices[:offset]
        for recipient, donor in zip(indices, permuted):
            source = np.asarray(orbits[donor]["interaction"], dtype=float)
            target = np.asarray(orbits[recipient]["interaction"], dtype=float)
            source_norm = float(np.linalg.norm(source))
            target_norm = float(np.linalg.norm(target))
            if source_norm <= 1e-14:
                if target_norm > 1e-14:
                    raise ContractError("matched donor has zero energy for a nonzero recipient")
                candidate = np.zeros_like(target)
            else:
                candidate = source * (target_norm / source_norm)
            donors[recipient] = candidate
    if any(value is None for value in donors):
        raise RuntimeError("matched donor assignment incomplete")
    return [np.asarray(value, dtype=float) for value in donors]


def _cell_table(
    orbits: Sequence[Mapping[str, Any]], contract: Mapping[str, Any],
    calibrators: Mapping[str, Any], interactions: Sequence[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    output: dict[str, list[Any]] = defaultdict(list)
    for index, row in enumerate(orbits):
        calibration = calibrators.get(str(row["model"]), {}).get(str(row["finding"]))
        if not isinstance(calibration, Mapping):
            raise ContractError(f"missing calibrator for {row['model']}/{row['finding']}")
        actual_score = np.asarray(row["score"], dtype=float)
        additive_score = actual_score - np.asarray(row["interaction"], dtype=float)
        candidate_score = actual_score if interactions is None else additive_score + np.asarray(interactions[index], dtype=float)
        q = float(row["reader_votes"]) / 3.0
        p = apply_calibrator(calibration, candidate_score.ravel())
        p0 = apply_calibrator(calibration, additive_score.ravel())
        brier_delta = (p - q) ** 2 - (p0 - q) ** 2
        additive_brier = (p0 - q) ** 2
        nll_delta = _soft_nll(p, np.full(p.size, q)) - _soft_nll(p0, np.full(p.size, q))
        clear = int(row["reader_votes"]) in (0, 3)
        truth = 1.0 if int(row["reader_votes"]) == 3 else -1.0
        actual_wrong = truth * candidate_score.ravel() <= 0.0
        additive_wrong = truth * additive_score.ravel() <= 0.0
        for cell in range(p.size):
            output["cluster"].append(str(row["cluster_id"]))
            output["orbit"].append(f"{row['model']}|{row['image_id']}|{row['finding']}")
            output["model"].append(str(row["model"]))
            output["finding"].append(str(row["finding"]))
            output["reader_vote"].append(float(row["reader_votes"]))
            output["brier"].append(float(brier_delta[cell]))
            output["additive_brier"].append(float(additive_brier[cell]))
            output["nll"].append(float(nll_delta[cell]))
            output["harmful"].append(float(brier_delta[cell] > 0.0))
            output["clear"].append(float(clear))
            output["introduced"].append(float(clear and actual_wrong[cell] and not additive_wrong[cell]))
            output["repaired"].append(float(clear and not actual_wrong[cell] and additive_wrong[cell]))
            output["flip"].append(float(clear and actual_wrong[cell] != additive_wrong[cell]))
            output["oriented_margin_loss"].append(float(truth * (additive_score.ravel()[cell] - candidate_score.ravel()[cell])) if clear else 0.0)
    categorical = {"cluster", "orbit", "model", "finding"}
    return {key: np.asarray(value, dtype=object if key in categorical else float) for key, value in output.items()}


def _summarize(table: Mapping[str, np.ndarray]) -> dict[str, float]:
    clear = np.asarray(table["clear"], dtype=bool)
    return {
        "brier_delta": float(np.mean(table["brier"])),
        "soft_bernoulli_nll_delta": float(np.mean(table["nll"])),
        "fraction_harmful_cells": float(np.mean(table["harmful"])),
        "clear_introduced_error": float(np.mean(table["introduced"][clear])) if np.any(clear) else math.nan,
        "clear_repaired_error": float(np.mean(table["repaired"][clear])) if np.any(clear) else math.nan,
        "clear_net_introduced_error": float(np.mean((table["introduced"] - table["repaired"])[clear])) if np.any(clear) else math.nan,
        "clear_polarity_flip_rate": float(np.mean(table["flip"][clear])) if np.any(clear) else math.nan,
        "clear_reader_oriented_margin_loss": float(np.mean(table["oriented_margin_loss"][clear])) if np.any(clear) else math.nan,
    }


def _cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, draws: int, rng: np.random.Generator) -> dict[str, Any]:
    unique = np.unique(clusters)
    if unique.size < 2:
        raise ContractError("paired whole-image bootstrap needs at least two clusters")
    by_cluster = [np.flatnonzero(clusters == key) for key in unique]
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.integers(0, len(by_cluster), size=len(by_cluster))
        selected = np.concatenate([by_cluster[index] for index in sampled])
        estimates[draw] = float(np.mean(values[selected]))
    return {
        "point": float(np.mean(values)),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "draws": int(draws), "cluster_unit": "patient_when_available_else_whole_image",
        "n_clusters": int(unique.size),
    }


def _orbit_aggregate(values: np.ndarray, table: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Average cells first so grid size cannot reweight a clinical orbit."""

    by_orbit: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(table["orbit"]):
        by_orbit[str(key)].append(index)
    output: dict[str, list[Any]] = defaultdict(list)
    for key, indices in sorted(by_orbit.items()):
        for name in ("model", "finding", "cluster"):
            unique = {str(table[name][index]) for index in indices}
            if len(unique) != 1:
                raise RuntimeError(f"{name} changes inside flattened orbit {key}")
            output[name].append(next(iter(unique)))
        vote_values = {int(float(table["reader_vote"][index])) for index in indices}
        if len(vote_values) != 1:
            raise RuntimeError(f"reader_vote changes inside flattened orbit {key}")
        output["reader_vote"].append(float(next(iter(vote_values))))
        output["value"].append(float(np.mean(values[indices])))
        output["orbit"].append(key)
    return {
        key: np.asarray(value, dtype=float if key in {"value", "reader_vote"} else object)
        for key, value in output.items()
    }


def _macro_point(
    values: np.ndarray, findings: np.ndarray, votes: np.ndarray,
    required_strata: Sequence[tuple[str, int]] | None = None,
) -> tuple[float, list[str]]:
    present = {(str(f), int(float(v))) for f, v in zip(findings, votes)}
    strata = sorted(present if required_strata is None else required_strata)
    if not strata:
        raise ContractError("finding-by-vote macro has no strata")
    if not set(strata).issubset(present):
        raise ContractError("finding-by-vote macro draw omitted a required stratum")
    means = [
        float(np.mean(values[(findings == finding) & (votes.astype(float) == vote)]))
        for finding, vote in strata
    ]
    # Object votes were serialized through str in _orbit_aggregate.
    if any(not math.isfinite(value) for value in means):
        raise ContractError("finding-by-vote macro has an empty stratum")
    return float(np.mean(means)), [f"{finding}|{vote}" for finding, vote in strata]


def _macro_cluster_bootstrap(
    excess_cells: np.ndarray, baseline_cells: np.ndarray,
    table: Mapping[str, np.ndarray], model: str,
    shared_cluster_order: Sequence[str], shared_cluster_weights: np.ndarray,
) -> dict[str, Any]:
    excess = _orbit_aggregate(excess_cells, table)
    baseline = _orbit_aggregate(baseline_cells, table)
    if not np.array_equal(excess["orbit"], baseline["orbit"]):
        raise RuntimeError("paired orbit aggregation lost alignment")
    keep = excess["model"] == model
    if not np.any(keep):
        raise ContractError(f"no confirmation orbits for model {model}")
    x = np.asarray(excess["value"][keep], dtype=float)
    b0 = np.asarray(baseline["value"][keep], dtype=float)
    finding = np.asarray(excess["finding"][keep], dtype=object)
    vote = np.asarray(excess["reader_vote"][keep], dtype=object)
    cluster = np.asarray(excess["cluster"][keep], dtype=object)
    point, strata = _macro_point(x, finding, vote)
    baseline_point, baseline_strata = _macro_point(b0, finding, vote)
    if strata != baseline_strata or baseline_point <= 0:
        raise ContractError("PAEL and additive Brier macro strata are not aligned")
    unique = np.unique(cluster)
    if unique.size < 2:
        raise ContractError("shared whole-image macro bootstrap needs at least two clusters")
    by_cluster = {str(key): np.flatnonzero(cluster == key) for key in unique}
    required = [(item.rsplit("|", 1)[0], int(item.rsplit("|", 1)[1])) for item in strata]
    weights = np.asarray(shared_cluster_weights, dtype=int)
    if weights.ndim != 2 or weights.shape[1] != len(shared_cluster_order):
        raise ContractError("shared cluster multiplier plan shape mismatch")
    draws = int(weights.shape[0])
    boot = np.empty(draws, dtype=float)
    ratio = np.empty(draws, dtype=float)
    for draw in range(draws):
        pieces = []
        for column, key in enumerate(shared_cluster_order):
            if weights[draw, column] <= 0 or key not in by_cluster:
                continue
            pieces.extend([by_cluster[key]] * int(weights[draw, column]))
        if not pieces:
            raise ContractError("shared cluster multiplier draw has no model observations")
        selected = np.concatenate(pieces)
        estimate, _ = _macro_point(
            x[selected], finding[selected], vote[selected], required
        )
        denominator, _ = _macro_point(
            b0[selected], finding[selected], vote[selected], required
        )
        boot[draw] = estimate
        ratio[draw] = estimate / denominator if denominator > 0 else math.nan
    if not np.isfinite(ratio).all():
        raise ContractError("bootstrap additive Brier denominator reached zero")
    return {
        "estimand": "orbit_first_then_equal_finding_by_reader_vote_macro",
        "point": point,
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "additive_brier_B0": baseline_point,
        "relative_R_pael_over_B0": point / baseline_point,
        "relative_R_ci95": [float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))],
        "operational_mcid_R": 0.05,
        "n_equal_weight_strata": len(strata),
        "strata": strata,
        "complete_expected_16_strata": len(strata) == 16,
        "draws": draws,
        "n_shared_clusters": int(unique.size),
        "cluster_unit": "patient_when_available_else_whole_image_shared_across_findings",
        "same_cluster_multiplier_draws_shared_across_models": True,
        "decision": "not_computed_non_authorizing_prototype",
    }


def _shared_cluster_multiplier_plan(
    table: Mapping[str, np.ndarray], models: Sequence[str], draws: int,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray, int, str]:
    """Freeze one cluster-bootstrap weight matrix used by every model."""

    orbit = _orbit_aggregate(np.zeros(len(table["cluster"]), dtype=float), table)
    cluster_order = sorted({str(value) for value in orbit["cluster"]})
    if len(cluster_order) < 2:
        raise ContractError("shared bootstrap requires at least two global clusters")
    required_by_model: dict[str, set[tuple[str, int]]] = {}
    for model in models:
        take = orbit["model"] == model
        required_by_model[model] = {
            (str(finding), int(vote))
            for finding, vote in zip(orbit["finding"][take], orbit["reader_vote"][take])
        }
    accepted = []
    attempts = 0
    while len(accepted) < draws:
        attempts += 1
        if attempts > max(1000, draws * 100):
            raise ContractError("shared cluster bootstrap cannot preserve every model stratum")
        weight = rng.multinomial(len(cluster_order), np.full(len(cluster_order), 1 / len(cluster_order)))
        included = {cluster_order[index] for index in np.flatnonzero(weight)}
        valid = True
        for model in models:
            take = (orbit["model"] == model) & np.isin(orbit["cluster"], list(included))
            present = {
                (str(finding), int(vote))
                for finding, vote in zip(orbit["finding"][take], orbit["reader_vote"][take])
            }
            if not required_by_model[model].issubset(present):
                valid = False
                break
        if valid:
            accepted.append(weight)
    matrix = np.asarray(accepted, dtype=np.int32)
    digest = hashlib.sha256()
    digest.update("\n".join(cluster_order).encode("utf-8"))
    digest.update(matrix.tobytes(order="C"))
    return cluster_order, matrix, attempts, digest.hexdigest()


def _null_interactions(
    kind: str, orbits: Sequence[Mapping[str, Any]], contract: Mapping[str, Any],
    plan: Mapping[str, Any], rng: np.random.Generator,
) -> list[np.ndarray]:
    if kind == "matched_orbit":
        return _matched_donors(orbits, contract, plan, rng)
    output = []
    for row in orbits:
        interaction = np.asarray(row["interaction"], dtype=float)
        if kind == "spectral_haar":
            output.append(spectral_haar_interaction(interaction, rng))
        elif kind == "cell_coordinate":
            output.append(cell_coordinate_interaction(interaction, rng))
        elif kind == "whole_orbit_sign":
            output.append(interaction * rng.choice((-1.0, 1.0)))
        else:
            raise ValueError(kind)
    return output


def apply_confirmation(
    payload: Mapping[str, Any], bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply frozen calibration/null contracts to a disjoint confirmation payload."""

    _validate_bundle(bundle)
    contract = validate_payload(payload)
    if contract["split"] != "confirmation_locked" or contract["source_manifest_split"] != "confirmation":
        raise ContractError("v4 evaluation is apply-only on confirmation_locked")
    _check_geometry(contract, bundle)
    orbits = _decorate_orbits(contract)
    overlap = set(bundle["dev_cluster_ids"]) & {str(row["cluster_id"]) for row in orbits}
    if overlap:
        raise ContractError(f"dev/confirmation whole-cluster overlap: {sorted(overlap)[:3]}")

    frozen = bundle["randomization"]
    seed = int(frozen["seed"])
    null_draws = int(frozen["null_draws"])
    bootstrap_draws = int(frozen["bootstrap_draws"])
    actual = _cell_table(orbits, contract, bundle["calibrators"])
    observed = _summarize(actual)
    null_names = ("matched_orbit", "spectral_haar", "cell_coordinate", "whole_orbit_sign")
    null_metrics = {name: defaultdict(list) for name in null_names}
    null_unavailable: dict[str, str] = {}
    haar_cell_brier = []
    for draw in range(null_draws):
        for kind_index, kind in enumerate(null_names):
            if kind in null_unavailable:
                continue
            rng = np.random.default_rng(seed + 1000003 * draw + 7919 * kind_index)
            try:
                interaction = _null_interactions(kind, orbits, contract, bundle["matching_plan"], rng)
            except ContractError as error:
                if kind != "matched_orbit":
                    raise
                null_unavailable[kind] = str(error)
                null_metrics[kind].clear()
                continue
            table = _cell_table(orbits, contract, bundle["calibrators"], interaction)
            summary = _summarize(table)
            for metric, value in summary.items():
                null_metrics[kind][metric].append(float(value))
            if kind == "spectral_haar":
                haar_cell_brier.append(np.asarray(table["brier"], dtype=float))
    if len(haar_cell_brier) != null_draws:
        raise RuntimeError("spectral Haar primary null coverage is incomplete")
    haar_mean_brier = np.mean(np.stack(haar_cell_brier), axis=0)
    pael_brier_haar = np.asarray(actual["brier"], dtype=float) - haar_mean_brier
    models = sorted({str(row["model"]) for row in orbits})
    cluster_order, cluster_weights, bootstrap_attempts, bootstrap_plan_sha256 = (
        _shared_cluster_multiplier_plan(
            actual, models, bootstrap_draws, np.random.default_rng(seed + 909091)
        )
    )
    primary = {
        "name": "Brier_PAEL_Haar",
        "definition": (
            "observed additive-counterfactual Brier delta minus mean centered-subspace "
            "spectral-Haar stress-null delta"
        ),
        "aggregation": "orbit first; then finding x reader-vote strata with equal weight",
        "inference_boundary": (
            "Haar rotations preserve centered-subspace singular values but semantic render/"
            "wording levels are not a proven exchangeability group; bootstrap PAEL, not a "
            "Haar randomization p-value, is the confirmatory diagnostic"
        ),
        "models": {
            model: _macro_cluster_bootstrap(
                pael_brier_haar, np.asarray(actual["additive_brier"], dtype=float),
                actual, model, cluster_order, cluster_weights,
            )
            for model in models
        },
        "shared_cluster_multiplier_plan": {
            "seed": seed + 909091,
            "draws": bootstrap_draws,
            "global_cluster_count": len(cluster_order),
            "attempts_including_rejected_missing_strata": bootstrap_attempts,
            "plan_sha256": bootstrap_plan_sha256,
            "same_weights_used_for_every_model": True,
        },
        "authorized": False,
    }
    null_output: dict[str, Any] = {}
    for kind in null_names:
        if kind in null_unavailable:
            null_output[kind] = {
                "available": False,
                "reason": null_unavailable[kind],
                "impact": "sensitivity unavailable; does not block the spectral-Haar Brier primary",
            }
            continue
        null_output[kind] = {}
        for metric, values in null_metrics[kind].items():
            distribution = np.asarray(values, dtype=float)
            obs = float(observed[metric])
            item = {
                "observed": obs,
                "null_mean": float(np.mean(distribution)),
                "excess": float(obs - np.mean(distribution)),
                "null_percentile": float(np.mean(distribution < obs)),
            }
            if kind == "matched_orbit":
                item["approximate_reference_tail_fraction_greater_equal"] = float(
                    (1 + np.sum(distribution >= obs)) / (distribution.size + 1)
                )
                item["inference_boundary"] = (
                    "descriptive sensitivity only; no whole-image conditional "
                    "exchangeability diagnosis and not a randomization p-value"
                )
            else:
                item["inference_boundary"] = "stress null only; no exchangeability p-value"
            null_output[kind][metric] = item

    # Apply-only calibration diagnostic uses the actual cell scores; no refit.
    all_prob = []
    all_q = []
    all_votes = []
    for row in orbits:
        calibration = bundle["calibrators"][row["model"]][row["finding"]]
        flat = np.asarray(row["score"], dtype=float).ravel()
        all_prob.extend(apply_calibrator(calibration, flat).tolist())
        all_q.extend([int(row["reader_votes"]) / 3.0] * flat.size)
        all_votes.extend([int(row["reader_votes"])] * flat.size)
    result = {
        "version": RESULT_VERSION,
        "status": NON_AUTHORIZING,
        "authorized": False,
        "fit_or_refit_on_confirmation": False,
        "confirmation_input_sha256": object_sha256(payload),
        "dev_bundle_sha256": bundle["bundle_sha256"],
        "source_sha256": _module_sha256(),
        "n_orbits": len(orbits),
        "n_clusters": len(set(actual["cluster"].tolist())),
        "observed_product_risk": observed,
        "primary_pael": primary,
        "null_diagnostics": null_output,
        "confirmation_calibration_apply_only": _calibration_diagnostics(
            np.asarray(all_prob), np.asarray(all_q), np.asarray(all_votes)
        ),
        "interpretation": (
            "Positive Brier PAEL_Haar means the reader-grounded loss of the localized product "
            "exceeds its singular-spectrum-preserving centered-subspace stress control. "
            "Matched/cell/sign and NLL outputs are sensitivities. This prototype makes no "
            "scientific or runtime decision."
        ),
    }
    return _seal(result, "result_sha256")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ContractError(f"{path} must contain one JSON object")
    return dict(value)


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-dev")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--folds", type=int, default=4)
    fit.add_argument("--null-draws", type=int, default=499)
    fit.add_argument("--bootstrap-draws", type=int, default=2000)
    fit.add_argument("--seed", type=int, default=1729)
    apply = sub.add_parser("apply-confirmation")
    apply.add_argument("--input", type=Path, required=True)
    apply.add_argument("--dev-bundle", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit-dev":
        result = fit_dev_calibration(
            _load(args.input), folds=args.folds, null_draws=args.null_draws,
            bootstrap_draws=args.bootstrap_draws, seed=args.seed,
        )
    else:
        result = apply_confirmation(_load(args.input), _load(args.dev_bundle))
    _write(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": result.get("bundle_sha256", result.get("result_sha256")), "status": NON_AUTHORIZING}, indent=2))


if __name__ == "__main__":
    main()
