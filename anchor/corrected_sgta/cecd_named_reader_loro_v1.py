#!/usr/bin/env python3
"""Named-reader leave-one-reader-out sensitivity for CECD v4.

This module is a deliberately non-authorizing substrate.  A held-out VinDr
``rad_ID`` is excluded from the development calibration target and used only
as a binary confirmation outcome.  Aggregate vote counts and anonymous vote
positions are therefore not valid inputs.

The primary descriptive statistic is the confirmation Brier loss of the
observed product-bearing score minus the loss of its additive counterfactual.
Positive values mean that the product component hurts prediction of a named
held-out reader.  Inference is paired at the patient level and macro-averages
findings within readers and readers within models.  It is a sensitivity
analysis only and cannot authorize CECD, mitigation, or model selection.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .analyze_clinical_equivalence_composition_defect_v1 import (
    ContractError as CECDContractError,
    build_orbits,
    sha256_file,
    validate_payload,
)


VERSION = "cecd-named-reader-loro-v1"
BUNDLE_VERSION = "cecd-named-reader-loro-dev-bundle-v1"
RESULT_VERSION = "cecd-named-reader-loro-confirmation-result-v1"
CONFIG_VERSION = "cecd-named-reader-loro-contract-v1"
MANIFEST_VERSION = "vindr-cecd-named-reader-manifest-v1"
STATUS = "outcome_blind_design_non_authorizing_named_reader_sensitivity"
IDENTITY_CONTRACT = {
    "source_field": "rad_ID",
    "semantics": "stable_pseudonymous_individual_radiologist",
    "named_identity_preserved": True,
    "aggregate_or_anonymous_position_substitution_allowed": False,
}
ANNOTATION_PROTOCOL = {
    "source": "official_vindr_cxr_1.0.0_training_annotations",
    "independent_reader_judgments": True,
}
EPS = 1e-7


class NamedReaderLOROError(ValueError):
    """The input cannot identify the frozen named-reader LORO estimand."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
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
        raise NamedReaderLOROError(f"missing or malformed {field}")
    unsealed = dict(payload)
    del unsealed[field]
    if object_sha256(unsealed) != claimed:
        raise NamedReaderLOROError(f"{field} mismatch; frozen artifact was modified")


def _module_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _string_sequence(value: Any, label: str, minimum: int = 1) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < minimum:
        raise NamedReaderLOROError(f"{label} must be a frozen non-empty list")
    output = tuple(str(item).strip() for item in value)
    if any(not item for item in output) or len(set(output)) != len(output):
        raise NamedReaderLOROError(f"{label} must contain unique non-empty strings")
    return output


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the pre-result, non-authorizing LORO design contract."""

    if config.get("schema_version") != CONFIG_VERSION:
        raise NamedReaderLOROError(f"config schema_version must be {CONFIG_VERSION}")
    if config.get("scientific_status") != STATUS or config.get("authorized") is not False:
        raise NamedReaderLOROError("named-reader LORO must remain explicitly non-authorizing")
    if config.get("frozen_before_real_outputs") is not True:
        raise NamedReaderLOROError("LORO design must be frozen before real outputs")
    models = _string_sequence(config.get("models"), "models", 2)
    findings = _string_sequence(config.get("findings"), "findings")
    readers = _string_sequence(config.get("named_readers"), "named_readers", 3)
    if len(readers) != 3:
        raise NamedReaderLOROError("VinDr LORO requires exactly three frozen named readers")
    if config.get("panel_policy") != "exact_same_three_named_rad_IDs_for_every_image":
        raise NamedReaderLOROError("variable or anonymous reader panels cannot identify this LORO")
    if config.get("patient_identity_policy") != "required_complete_and_cross_split_disjoint":
        raise NamedReaderLOROError("patient identity must be complete for leakage checks")
    if config.get("dev_target") != "mean_binary_vote_of_two_nonheldout_named_readers":
        raise NamedReaderLOROError("dev target must exclude the named held-out reader")
    if config.get("confirmation_target") != "binary_vote_of_named_heldout_reader":
        raise NamedReaderLOROError("confirmation target must be the held-out named reader")
    if config.get("primary_statistic") != "actual_minus_additive_brier_equal_reader_macro":
        raise NamedReaderLOROError("primary statistic differs from the frozen LORO estimand")
    if config.get("sample_selection_conditioning") != (
        "aggregate_reader_vote_balanced_upstream; not_a_population_prevalence_estimand"
    ):
        raise NamedReaderLOROError("upstream aggregate-vote selection conditioning must be explicit")
    for key in ("minimum_dev_orbits_per_model_finding", "minimum_confirmation_orbits_per_model_finding"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 8:
            raise NamedReaderLOROError(f"{key} must be an integer at least 8")
    bootstrap = config.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise NamedReaderLOROError("bootstrap contract is missing")
    if bootstrap.get("unit") != "patient" or bootstrap.get("family") != "shared_positive_exponential_multiplier":
        raise NamedReaderLOROError("bootstrap must use shared positive patient multipliers")
    for key in ("seed", "draws"):
        value = bootstrap.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise NamedReaderLOROError(f"bootstrap.{key} must be a positive integer")
    return {
        **dict(config),
        "models": list(models),
        "findings": list(findings),
        "named_readers": list(readers),
    }


def _patient(row: Mapping[str, Any], where: str) -> str:
    value = row.get("patient_id")
    if not isinstance(value, str) or not value.strip():
        raise NamedReaderLOROError(
            f"{where}: patient_id unavailable; patient-level leakage and LORO are not identifiable"
        )
    return value.strip()


def _validate_named_manifest(
    manifest: Mapping[str, Any], *, split: str, readers: Sequence[str], findings: Sequence[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise NamedReaderLOROError(f"named manifest schema_version must be {MANIFEST_VERSION}")
    if manifest.get("split") != split:
        raise NamedReaderLOROError(f"named manifest split must be {split}")
    if manifest.get("reader_identity") != IDENTITY_CONTRACT:
        raise NamedReaderLOROError(
            "reader identity provenance is absent or permits aggregate/anonymous vote positions"
        )
    if manifest.get("annotation_protocol") != ANNOTATION_PROTOCOL:
        raise NamedReaderLOROError(
            "three distinct rad_ID values do not by themselves establish independent judgments"
        )
    rows = manifest.get("records")
    if not isinstance(rows, list) or not rows:
        raise NamedReaderLOROError("named manifest records must be non-empty")
    required_panel = set(readers)
    required_findings = set(findings)
    output: dict[tuple[str, str], dict[str, Any]] = {}
    image_panel: dict[str, frozenset[str]] = {}
    image_patient: dict[str, str] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise NamedReaderLOROError(f"named manifest row {index} is not an object")
        image = str(raw.get("image_id", "")).strip()
        finding = str(raw.get("finding", "")).strip()
        if not image or finding not in required_findings:
            raise NamedReaderLOROError(f"named manifest row {index} has an invalid image/finding")
        patient = _patient(raw, f"named manifest row {index}")
        if image in image_patient and image_patient[image] != patient:
            raise NamedReaderLOROError(f"image {image} has conflicting patient identities")
        image_patient[image] = patient
        votes = raw.get("reader_votes")
        if not isinstance(votes, list) or len(votes) != 3:
            raise NamedReaderLOROError(
                f"{image}/{finding}: requires three reader-level named votes; aggregate counts are invalid"
            )
        named: dict[str, int] = {}
        for position, item in enumerate(votes):
            if not isinstance(item, Mapping) or "rad_id" not in item:
                raise NamedReaderLOROError(
                    f"{image}/{finding}: vote {position} lacks official named rad_ID; anonymous position is invalid"
                )
            reader = str(item.get("rad_id", "")).strip()
            vote = item.get("vote")
            if not reader or reader in named:
                raise NamedReaderLOROError(f"{image}/{finding}: duplicate or empty rad_ID")
            if isinstance(vote, bool) or vote not in (0, 1):
                raise NamedReaderLOROError(f"{image}/{finding}: named vote must be binary integer 0/1")
            named[reader] = int(vote)
        if set(named) != required_panel:
            raise NamedReaderLOROError(
                f"{image}/{finding}: reader-panel composition differs from frozen named panel"
            )
        declared_ids = raw.get("reader_ids")
        if not isinstance(declared_ids, list) or set(map(str, declared_ids)) != set(named) or len(declared_ids) != 3:
            raise NamedReaderLOROError(f"{image}/{finding}: reader_ids do not preserve the named vote identities")
        if raw.get("reader_count") != 3 or raw.get("positive_votes") != sum(named.values()):
            raise NamedReaderLOROError(f"{image}/{finding}: aggregate fields disagree with named votes")
        panel = frozenset(named)
        if image in image_panel and image_panel[image] != panel:
            raise NamedReaderLOROError(f"image {image} changes reader panel across findings")
        image_panel[image] = panel
        key = (image, finding)
        if key in output:
            raise NamedReaderLOROError(f"duplicate named claim record {image}/{finding}")
        output[key] = {"patient_id": patient, "votes": named}
    return output


def _validate_stage(
    payload: Mapping[str, Any], manifest: Mapping[str, Any], config: Mapping[str, Any], stage: str,
) -> dict[str, Any]:
    source_split = "dev" if stage == "dev_fit" else "confirmation"
    try:
        contract = validate_payload(payload)
    except CECDContractError as error:
        raise NamedReaderLOROError(f"{stage} CECD payload invalid: {error}") from error
    if contract["split"] != stage or contract["source_manifest_split"] != source_split:
        raise NamedReaderLOROError(f"{stage} must use truthful {source_split} source split")
    models = set(config["models"])
    findings = set(config["findings"])
    if {str(key[0]) for key in contract["by_orbit"]} != models:
        raise NamedReaderLOROError(f"{stage} model set differs from the frozen set")
    if {str(key[2]) for key in contract["by_orbit"]} != findings:
        raise NamedReaderLOROError(f"{stage} finding set differs from the frozen set")
    named = _validate_named_manifest(
        manifest, split=source_split, readers=config["named_readers"], findings=config["findings"]
    )
    claim_keys = {(str(key[1]), str(key[2])) for key in contract["by_orbit"]}
    if set(named) != claim_keys:
        missing = len(claim_keys - set(named))
        extra = len(set(named) - claim_keys)
        raise NamedReaderLOROError(
            f"{stage} named claim join is not exact (missing={missing}, extra={extra})"
        )
    image_patient: dict[str, str] = {}
    for (model, image, finding), rows in contract["by_orbit"].items():
        key = (str(image), str(finding))
        named_row = named[key]
        vote_count = sum(named_row["votes"].values())
        patient_values = {_patient(row, f"{stage} CECD row") for row in rows}
        if patient_values != {named_row["patient_id"]}:
            raise NamedReaderLOROError(f"{stage} patient join disagrees for {image}/{finding}")
        if any(isinstance(row.get("reader_votes"), (list, tuple, dict)) for row in rows):
            raise NamedReaderLOROError("CECD cell reader_votes must be the cross-checked aggregate count")
        if any(int(row["reader_votes"]) != vote_count for row in rows):
            raise NamedReaderLOROError(f"{stage} aggregate vote mismatch for {image}/{finding}")
        patient = named_row["patient_id"]
        previous = image_patient.setdefault(str(image), patient)
        if previous != patient:
            raise NamedReaderLOROError(f"{stage} image {image} maps to multiple patients")
    orbits = build_orbits(contract)
    minimum = int(config[
        "minimum_dev_orbits_per_model_finding"
        if stage == "dev_fit" else "minimum_confirmation_orbits_per_model_finding"
    ])
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in orbits:
        counts[(str(row["model"]), str(row["finding"]))] += 1
    expected = {(model, finding) for model in config["models"] for finding in config["findings"]}
    if set(counts) != expected or any(counts[key] < minimum for key in expected):
        raise NamedReaderLOROError(
            f"{stage} lacks the frozen minimum orbit count in at least one model/finding stratum"
        )
    return {
        "contract": contract,
        "orbits": orbits,
        "named": named,
        "images": set(image_patient),
        "patients": set(image_patient.values()),
    }


def _clean_scores(orbit: Mapping[str, Any], contract: Mapping[str, Any]) -> tuple[float, float]:
    renders = list(contract["primary_renders"])
    prompts = list(contract["primary_prompts"])
    r = renders.index(contract["baseline_render"])
    p = prompts.index(contract["baseline_prompt"])
    actual = float(np.asarray(orbit["score"], dtype=float)[r, p])
    additive = actual - float(np.asarray(orbit["interaction"], dtype=float)[r, p])
    return actual, additive


def _fit_isotonic(scores: Sequence[float], targets: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(scores, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.size < 8 or x.size != y.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise NamedReaderLOROError("LORO calibration needs at least eight finite dev pairs")
    if np.var(x) <= 1e-12 or np.var(y) <= 1e-12:
        raise NamedReaderLOROError("LORO calibration is not identifiable without score and target variation")
    if float(np.cov(x, y, bias=True)[0, 1]) <= 1e-10:
        raise NamedReaderLOROError("LORO dev score moves opposite to the non-heldout reader target")
    model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(x, y)
    xt = np.asarray(model.X_thresholds_, dtype=float)
    yt = np.asarray(model.y_thresholds_, dtype=float)
    if xt.size < 2 or np.any(np.diff(xt) <= 0) or np.any(np.diff(yt) < -1e-12):
        raise NamedReaderLOROError("invalid named-reader isotonic calibration")
    return {
        "kind": "isotonic_piecewise_linear_clip",
        "x_thresholds": xt.tolist(),
        "y_thresholds": yt.tolist(),
        "n_fit": int(x.size),
    }


def _apply_calibrator(bundle: Mapping[str, Any], scores: np.ndarray) -> np.ndarray:
    if bundle.get("kind") != "isotonic_piecewise_linear_clip":
        raise NamedReaderLOROError("unknown named-reader calibrator")
    x = np.asarray(bundle.get("x_thresholds"), dtype=float)
    y = np.asarray(bundle.get("y_thresholds"), dtype=float)
    if x.ndim != 1 or x.size < 2 or y.shape != x.shape:
        raise NamedReaderLOROError("malformed named-reader calibrator")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0):
        raise NamedReaderLOROError("non-finite or non-monotone named-reader calibrator")
    return np.interp(np.asarray(scores, dtype=float), x, y, left=y[0], right=y[-1])


def fit_named_reader_loro_dev(
    payload: Mapping[str, Any], manifest: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit calibrators on dev without ever using each held-out reader's vote."""

    frozen = validate_config(config)
    stage = _validate_stage(payload, manifest, frozen, "dev_fit")
    calibrators: dict[str, Any] = {}
    exclusion_audit: dict[str, Any] = {}
    for model in frozen["models"]:
        calibrators[model] = {}
        for finding in frozen["findings"]:
            rows = [
                row for row in stage["orbits"]
                if str(row["model"]) == model and str(row["finding"]) == finding
            ]
            scores = [_clean_scores(row, stage["contract"])[0] for row in rows]
            calibrators[model][finding] = {}
            for heldout in frozen["named_readers"]:
                targets = [
                    sum(
                        vote for reader, vote in stage["named"][(str(row["image_id"]), finding)]["votes"].items()
                        if reader != heldout
                    ) / 2.0
                    for row in rows
                ]
                fitted = _fit_isotonic(scores, targets)
                fitted.update({
                    "fit_split": "dev",
                    "heldout_reader": heldout,
                    "target_readers": [reader for reader in frozen["named_readers"] if reader != heldout],
                    "heldout_reader_outcomes_used_as_calibration_target": False,
                })
                calibrators[model][finding][heldout] = fitted
                exclusion_audit[f"{model}|{finding}|{heldout}"] = {
                    "n_fit": len(rows),
                    "heldout_outcomes_used_as_calibration_target": 0,
                    "nonheldout_named_votes_used": 2 * len(rows),
                }
    geometry = {
        key: list(stage["contract"][key]) if isinstance(stage["contract"][key], tuple) else stage["contract"][key]
        for key in (
            "primary_renders", "primary_prompts", "baseline_render", "baseline_prompt",
            "identity_render", "duplicate_prompt",
        )
    }
    artifact = {
        "version": BUNDLE_VERSION,
        "scientific_status": STATUS,
        "authorized": False,
        "fit_split": "dev",
        "apply_split": "confirmation",
        "source_sha256": _module_sha256(),
        "config_sha256": object_sha256(frozen),
        "dev_cecd_input_sha256": object_sha256(payload),
        "dev_named_manifest_sha256": object_sha256(manifest),
        "named_reader_identity_contract": IDENTITY_CONTRACT,
        "annotation_protocol": ANNOTATION_PROTOCOL,
        "named_readers": list(frozen["named_readers"]),
        "models": list(frozen["models"]),
        "findings": list(frozen["findings"]),
        "geometry": geometry,
        "dev_image_ids": sorted(stage["images"]),
        "dev_patient_ids": sorted(stage["patients"]),
        "calibrators": calibrators,
        "exclusion_audit": exclusion_audit,
        "selection_conditioning": (
            "the frozen sample may have been balanced using the aggregate vote, which includes "
            "the held-out reader; only the calibration target, not upstream inclusion, is LORO"
        ),
        "claim_boundary": (
            "named-reader sensitivity only; does not authorize CECD, mitigation, "
            "reader consensus, or clinical efficacy"
        ),
    }
    return _seal(artifact, "bundle_sha256")


def _validate_bundle(bundle: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if bundle.get("version") != BUNDLE_VERSION or bundle.get("scientific_status") != STATUS:
        raise NamedReaderLOROError("wrong named-reader LORO bundle")
    if bundle.get("authorized") is not False:
        raise NamedReaderLOROError("authorizing named-reader bundle is prohibited")
    _verify_seal(bundle, "bundle_sha256")
    if bundle.get("source_sha256") != _module_sha256():
        raise NamedReaderLOROError("bundle source differs from current LORO module")
    if bundle.get("config_sha256") != object_sha256(config):
        raise NamedReaderLOROError("confirmation config differs from frozen dev config")
    if bundle.get("fit_split") != "dev" or bundle.get("apply_split") != "confirmation":
        raise NamedReaderLOROError("bundle is not dev-fit/confirmation-apply-only")


def _soft_nll(probability: np.ndarray, target: float) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    return -(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))


def _macro(values: np.ndarray, rows: Sequence[Mapping[str, Any]], model: str, weights: np.ndarray | None = None) -> float:
    reader_means = []
    for reader in sorted({str(row["reader"]) for row in rows}):
        finding_means = []
        for finding in sorted({str(row["finding"]) for row in rows}):
            indices = [
                index for index, row in enumerate(rows)
                if row["model"] == model and row["reader"] == reader and row["finding"] == finding
            ]
            if not indices:
                raise NamedReaderLOROError(f"missing LORO stratum {model}/{reader}/{finding}")
            x = values[indices]
            if weights is None:
                finding_means.append(float(np.mean(x)))
            else:
                w = weights[indices]
                if np.sum(w) <= 0:
                    raise NamedReaderLOROError("bootstrap LORO stratum has zero weight")
                finding_means.append(float(np.sum(w * x) / np.sum(w)))
        reader_means.append(float(np.mean(finding_means)))
    if not reader_means:
        raise NamedReaderLOROError(f"model {model} has no LORO strata")
    return float(np.mean(reader_means))


def _positive_multiplier(seed: int, draw: int, patient: str) -> float:
    digest = hashlib.sha256(f"cecd-loro|{seed}|{draw}|{patient}".encode()).digest()
    uniform = (int.from_bytes(digest[:8], "big") + 0.5) / float(2**64)
    return -math.log(uniform)


def apply_named_reader_loro_confirmation(
    payload: Mapping[str, Any], manifest: Mapping[str, Any], bundle: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply frozen dev calibrators to named confirmation outcomes only."""

    frozen = validate_config(config)
    _validate_bundle(bundle, frozen)
    stage = _validate_stage(payload, manifest, frozen, "confirmation_locked")
    image_overlap = set(bundle["dev_image_ids"]) & stage["images"]
    patient_overlap = set(bundle["dev_patient_ids"]) & stage["patients"]
    if image_overlap:
        raise NamedReaderLOROError(f"image crosses dev/confirmation: {sorted(image_overlap)[0]}")
    if patient_overlap:
        raise NamedReaderLOROError(f"patient crosses dev/confirmation: {sorted(patient_overlap)[0]}")
    geometry = {
        key: list(stage["contract"][key]) if isinstance(stage["contract"][key], tuple) else stage["contract"][key]
        for key in bundle["geometry"]
    }
    if geometry != bundle["geometry"]:
        raise NamedReaderLOROError("confirmation CECD geometry differs from dev")

    observations: list[dict[str, Any]] = []
    for orbit in stage["orbits"]:
        model = str(orbit["model"])
        finding = str(orbit["finding"])
        image = str(orbit["image_id"])
        named = stage["named"][(image, finding)]
        actual = np.asarray(orbit["score"], dtype=float).ravel()
        additive = (np.asarray(orbit["score"], dtype=float) - np.asarray(orbit["interaction"], dtype=float)).ravel()
        for heldout in frozen["named_readers"]:
            calibrator = bundle["calibrators"].get(model, {}).get(finding, {}).get(heldout)
            if not isinstance(calibrator, Mapping) or calibrator.get(
                "heldout_reader_outcomes_used_as_calibration_target"
            ) is not False:
                raise NamedReaderLOROError(f"missing safe calibrator for {model}/{finding}/{heldout}")
            y = float(named["votes"][heldout])
            p1 = _apply_calibrator(calibrator, actual)
            p0 = _apply_calibrator(calibrator, additive)
            actual_brier = float(np.mean((p1 - y) ** 2))
            additive_brier = float(np.mean((p0 - y) ** 2))
            actual_nll = float(np.mean(_soft_nll(p1, y)))
            additive_nll = float(np.mean(_soft_nll(p0, y)))
            observations.append({
                "model": model,
                "finding": finding,
                "reader": heldout,
                "image_id": image,
                "patient_id": named["patient_id"],
                "target": int(y),
                "actual_brier": actual_brier,
                "additive_brier": additive_brier,
                "brier_excess": actual_brier - additive_brier,
                "actual_nll": actual_nll,
                "additive_nll": additive_nll,
                "nll_excess": actual_nll - additive_nll,
            })
    expected = {
        (model, finding, reader)
        for model in frozen["models"] for finding in frozen["findings"] for reader in frozen["named_readers"]
    }
    present = {(row["model"], row["finding"], row["reader"]) for row in observations}
    if present != expected:
        raise NamedReaderLOROError("one or more named-reader confirmation strata are unidentifiable")

    by_stratum: dict[str, Any] = {}
    for key in sorted(expected):
        selected = [row for row in observations if (row["model"], row["finding"], row["reader"]) == key]
        if len(selected) < int(frozen["minimum_confirmation_orbits_per_model_finding"]):
            raise NamedReaderLOROError(f"insufficient confirmation observations for {'/'.join(key)}")
        by_stratum["|".join(key)] = {
            "n_orbits": len(selected),
            "n_patients": len({row["patient_id"] for row in selected}),
            "positive_outcomes": sum(row["target"] for row in selected),
            "actual_brier": float(np.mean([row["actual_brier"] for row in selected])),
            "additive_brier": float(np.mean([row["additive_brier"] for row in selected])),
            "actual_minus_additive_brier": float(np.mean([row["brier_excess"] for row in selected])),
            "actual_minus_additive_nll": float(np.mean([row["nll_excess"] for row in selected])),
        }

    brier = np.asarray([row["brier_excess"] for row in observations], dtype=float)
    nll = np.asarray([row["nll_excess"] for row in observations], dtype=float)
    patients = sorted({str(row["patient_id"]) for row in observations})
    patient_index = {patient: index for index, patient in enumerate(patients)}
    draws = int(frozen["bootstrap"]["draws"])
    seed = int(frozen["bootstrap"]["seed"])
    models: dict[str, Any] = {}
    for model in frozen["models"]:
        brier_point = _macro(brier, observations, model)
        nll_point = _macro(nll, observations, model)
        brier_boot = np.empty(draws, dtype=float)
        nll_boot = np.empty(draws, dtype=float)
        for draw in range(draws):
            patient_weights = np.asarray(
                [_positive_multiplier(seed, draw, patient) for patient in patients], dtype=float
            )
            row_weights = np.asarray(
                [patient_weights[patient_index[str(row["patient_id"])]] for row in observations], dtype=float
            )
            brier_boot[draw] = _macro(brier, observations, model, row_weights)
            nll_boot[draw] = _macro(nll, observations, model, row_weights)
        models[model] = {
            "actual_minus_additive_brier": {
                "point": brier_point,
                "ci95": [float(np.quantile(brier_boot, 0.025)), float(np.quantile(brier_boot, 0.975))],
            },
            "actual_minus_additive_nll": {
                "point": nll_point,
                "ci95": [float(np.quantile(nll_boot, 0.025)), float(np.quantile(nll_boot, 0.975))],
            },
        }
    result = {
        "version": RESULT_VERSION,
        "scientific_status": STATUS,
        "authorized": False,
        "fit_split": "dev",
        "apply_split": "confirmation",
        "bundle_sha256": bundle["bundle_sha256"],
        "confirmation_cecd_input_sha256": object_sha256(payload),
        "confirmation_named_manifest_sha256": object_sha256(manifest),
        "source_sha256": _module_sha256(),
        "identifiability": {
            "named_rad_ID_preserved": True,
            "exact_three_distinct_named_reader_fixed_panel": True,
            "official_protocol_declares_independent_reader_judgments": True,
            "aggregate_or_anonymous_position_substitution": False,
            "heldout_reader_vote_excluded_from_calibration_target": True,
            "upstream_selection_may_use_aggregate_vote_including_heldout": True,
            "dev_confirmation_image_disjoint": True,
            "dev_confirmation_patient_disjoint": True,
            "all_reader_model_finding_strata_present": True,
        },
        "primary": {
            "name": "actual_minus_additive_brier_equal_reader_macro",
            "interpretation": "positive means the product component increases loss against a named held-out reader",
            "aggregation": "cells_to_orbit_then_finding_within_reader_then_equal_readers_within_model",
            "inference": "shared strictly-positive patient multiplier bootstrap",
            "bootstrap_draws": draws,
            "models": models,
        },
        "by_model_finding_heldout_reader": by_stratum,
        "n_observations": len(observations),
        "n_patients": len(patients),
        "claim_boundary": (
            "non-authorizing reader-identity sensitivity; not consensus truth, not a CECD GO gate, "
            "and not evidence of clinical mitigation efficacy"
        ),
    }
    return _seal(result, "result_sha256")


def fail_closed_record(error: Exception, stage: str) -> dict[str, Any]:
    """Create an explicit non-authorizing record for an unidentifiable design."""

    if stage not in {"dev_fit", "confirmation_locked"}:
        raise ValueError("stage must be dev_fit or confirmation_locked")
    return _seal({
        "version": RESULT_VERSION,
        "scientific_status": "failed_closed_non_authorizing",
        "authorized": False,
        "identifiable": False,
        "failed_stage": stage,
        "reason": str(error),
        "statistics_emitted": False,
        "source_sha256": _module_sha256(),
    }, "result_sha256")
