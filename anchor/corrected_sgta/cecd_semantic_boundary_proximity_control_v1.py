#!/usr/bin/env python3
"""Reader-free semantic-boundary proximity control for CECD product grids.

This is a synthetic-ready, diagnostic substrate.  A development split freezes a
text-proxy semantic boundary and fixed, model-by-finding logistic models of
*generic* joint boundary crossing.  Every predictor covariate is computed before
the joint cell: from h00, h10 and h01 only.  The h11 representation is an
endpoint and may appear only in the target and an explicitly post-hoc descriptor.
A disjoint confirmation split is apply-only.  Reader votes, clinical labels,
PAEL and outcome-selected thresholds are deliberately outside the input
contract.

The module is not an implementation of Semantic Robustness Certification and
does not certify invariance.  It supplies the narrower collision control needed
by CECD: can ordinary proximity to a frozen semantic boundary predict failures
under a render x wording composition?
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .analyze_clinical_equivalence_composition_defect_v1 import ContractError, sha256_file


SCHEMA_VERSION = "cecd-semantic-boundary-proximity-input-v2"
BUNDLE_VERSION = "cecd-semantic-boundary-proximity-dev-bundle-v2"
RESULT_VERSION = "cecd-semantic-boundary-proximity-confirmation-v2"
NON_AUTHORIZING = (
    "reader_free_collision_control_only_non_authorizing; cannot authorize PAEL, "
    "a fusion mechanism, mitigation, model selection, or a clinical claim"
)
BASE_FEATURES = (
    "clean_abs_distance",
    "render_only_abs_distance",
    "wording_only_abs_distance",
    "minimum_single_axis_abs_distance",
    "render_boundary_approach",
    "wording_boundary_approach",
    "additive_abs_distance",
    "additive_boundary_approach",
    "additive_crosses_clean",
    "render_displacement_norm",
    "wording_displacement_norm",
    "additive_displacement_norm",
    "marginal_displacement_cosine",
    "marginal_displacement_norm_gap",
)
POSTHOC_DESCRIPTOR_FIELDS = (
    "joint_abs_distance",
    "joint_boundary_approach",
    "signed_product_residual_abs",
    "embedding_product_residual_norm",
    "joint_displacement_norm",
    "product_residual_off_axis_norm",
)
PROHIBITED_KEYS = {
    "reader_votes", "reader_vote", "reader_id", "reader_ids", "label",
    "labels", "target", "targets", "outcome", "outcomes", "ground_truth",
    "correct", "accuracy", "brier", "pael", "clinical_loss", "authorized",
    "threshold", "decision_threshold",
}
TOP_LEVEL_KEYS = {
    "schema_version", "split", "source_manifest_split", "source_manifest_sha256",
    "frozen_before_reader_outcomes", "baseline_render", "baseline_prompt",
    "primary_renders", "primary_prompts", "model_provenance",
    "representation_spec", "transform_spec_sha256", "prompt_spec_sha256",
    "proxies", "records",
}
MODEL_KEYS = {
    "checkpoint_sha256", "tokenizer_sha256", "processor_sha256", "code_revision_sha256",
}
REPRESENTATION_KEYS = {
    "layer_id", "token_selector", "pooling", "normalization", "dtype",
    "extraction_code_sha256",
}
PROXY_KEYS = {
    "model", "finding", "proxy_source_sha256", "present_embeddings",
    "refuted_embeddings",
}
RECORD_KEYS = {
    "model", "image_id", "patient_id", "finding", "render_id", "prompt_id",
    "source_image_sha256", "transformed_image_sha256", "exact_prompt_sha256",
    "proposition_sha256", "embedding",
}
DEV_FOLDS = 4
MIN_CLASS_COUNT = 8
LOGISTIC_C = 1.0
EPS = 1e-12


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = object_sha256(result)
    return result


def _verify_seal(value: Mapping[str, Any], field: str) -> None:
    claimed = value.get(field)
    if not _is_sha256(claimed):
        raise ContractError(f"missing or invalid {field}")
    unsealed = dict(value)
    del unsealed[field]
    if object_sha256(unsealed) != claimed:
        raise ContractError(f"{field} mismatch: artifact was modified after dev freeze")


def _module_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_sha256(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise ContractError(f"{name} must be a lowercase/uppercase hexadecimal sha256")
    return str(value).lower()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{name} schema drift: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _walk_prohibited(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in PROHIBITED_KEYS:
                raise ContractError(f"reader/outcome field is prohibited at {path}.{key}")
            _walk_prohibited(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_prohibited(child, f"{path}[{index}]")


def _vector(value: Any, name: str, dimension: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size < 2 or not np.isfinite(result).all():
        raise ContractError(f"{name} must be a finite vector of dimension >=2")
    if dimension is not None and result.size != dimension:
        raise ContractError(f"{name} dimension changed: expected {dimension}, got {result.size}")
    norm = float(np.linalg.norm(result))
    if norm <= EPS:
        raise ContractError(f"{name} has zero norm")
    return result / norm


def _proxy_geometry(proxy: Mapping[str, Any], dimension: int | None = None) -> dict[str, Any]:
    present_raw = proxy["present_embeddings"]
    refuted_raw = proxy["refuted_embeddings"]
    if not isinstance(present_raw, list) or not isinstance(refuted_raw, list):
        raise ContractError("proxy embeddings must be non-empty lists of vectors")
    if len(present_raw) < 2 or len(refuted_raw) < 2:
        raise ContractError("each semantic proxy pole needs at least two frozen text embeddings")
    present = [_vector(v, "present proxy", dimension) for v in present_raw]
    dim = present[0].size
    present = [_vector(v, "present proxy", dim) for v in present_raw]
    refuted = [_vector(v, "refuted proxy", dim) for v in refuted_raw]
    positive = np.mean(np.stack(present), axis=0)
    negative = np.mean(np.stack(refuted), axis=0)
    positive /= max(float(np.linalg.norm(positive)), EPS)
    negative /= max(float(np.linalg.norm(negative)), EPS)
    direction = positive - negative
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        raise ContractError("present and refuted text proxy poles are degenerate")
    direction /= norm
    midpoint = (positive + negative) / 2.0
    return {
        "dimension": int(dim),
        "direction": direction,
        "midpoint": midpoint,
        "content_sha256": object_sha256({
            "present_embeddings": present_raw,
            "refuted_embeddings": refuted_raw,
        }),
    }


def _cluster_digest(value: str) -> str:
    return hashlib.sha256(("cecd-boundary-cluster-v1:" + value).encode()).hexdigest()


def _validate_payload(payload: Mapping[str, Any], expected_split: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ContractError("input must be one JSON object")
    _require_exact_keys(payload, TOP_LEVEL_KEYS, "top-level")
    _walk_prohibited({k: v for k, v in payload.items() if k != "frozen_before_reader_outcomes"})
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"schema_version must be {SCHEMA_VERSION}")
    if payload["split"] != expected_split:
        raise ContractError(f"apply-only split violation: expected {expected_split}")
    expected_source_split = "dev" if expected_split == "dev_fit" else "confirmation"
    if payload["source_manifest_split"] != expected_source_split:
        raise ContractError(f"source_manifest_split must be {expected_source_split}")
    if payload["frozen_before_reader_outcomes"] is not True:
        raise ContractError("input must be frozen before reader outcomes")
    _require_sha256(payload["source_manifest_sha256"], "source_manifest_sha256")

    renders = payload["primary_renders"]
    prompts = payload["primary_prompts"]
    if not isinstance(renders, list) or len(renders) < 2 or len(set(renders)) != len(renders):
        raise ContractError("primary_renders must contain at least two unique strings")
    if not isinstance(prompts, list) or len(prompts) < 2 or len(set(prompts)) != len(prompts):
        raise ContractError("primary_prompts must contain at least two unique strings")
    if any(not isinstance(v, str) or not v for v in renders + prompts):
        raise ContractError("render and prompt identifiers must be non-empty strings")
    if payload["baseline_render"] not in renders or payload["baseline_prompt"] not in prompts:
        raise ContractError("baseline cell must belong to the primary product grid")

    transform_specs = payload["transform_spec_sha256"]
    prompt_specs = payload["prompt_spec_sha256"]
    if not isinstance(transform_specs, Mapping) or set(transform_specs) != set(renders):
        raise ContractError("transform_spec_sha256 keys must exactly equal primary_renders")
    if not isinstance(prompt_specs, Mapping) or set(prompt_specs) != set(prompts):
        raise ContractError("prompt_spec_sha256 keys must exactly equal primary_prompts")
    for key, value in transform_specs.items():
        _require_sha256(value, f"transform_spec_sha256[{key}]")
    for key, value in prompt_specs.items():
        _require_sha256(value, f"prompt_spec_sha256[{key}]")

    model_provenance = payload["model_provenance"]
    if not isinstance(model_provenance, Mapping) or not model_provenance:
        raise ContractError("model_provenance must be a non-empty mapping")
    for model, provenance in model_provenance.items():
        if not isinstance(model, str) or not model or not isinstance(provenance, Mapping):
            raise ContractError("malformed model_provenance")
        _require_exact_keys(provenance, MODEL_KEYS, f"model_provenance[{model}]")
        for key, value in provenance.items():
            _require_sha256(value, f"model_provenance[{model}].{key}")

    representations = payload["representation_spec"]
    if not isinstance(representations, Mapping) or set(representations) != set(model_provenance):
        raise ContractError(
            "representation_spec must map every frozen model, and only those models, to a spec"
        )
    for model, representation in representations.items():
        if not isinstance(representation, Mapping):
            raise ContractError(f"representation_spec[{model}] must be an object")
        _require_exact_keys(
            representation, REPRESENTATION_KEYS, f"representation_spec[{model}]"
        )
        if any(
            not isinstance(representation[k], str) or not representation[k]
            for k in REPRESENTATION_KEYS
        ):
            raise ContractError(
                f"representation_spec[{model}] values must be non-empty strings"
            )
        _require_sha256(
            representation["extraction_code_sha256"],
            f"representation_spec[{model}].extraction_code_sha256",
        )
        if representation["normalization"] != "l2_at_analysis":
            raise ContractError(
                f"representation_spec[{model}].normalization must be l2_at_analysis"
            )

    proxies = payload["proxies"]
    if not isinstance(proxies, list) or not proxies:
        raise ContractError("proxies must be a non-empty list")
    proxy_geometry: dict[tuple[str, str], dict[str, Any]] = {}
    proxy_source: dict[tuple[str, str], str] = {}
    dimension_by_model: dict[str, int] = {}
    for index, proxy in enumerate(proxies):
        if not isinstance(proxy, Mapping):
            raise ContractError(f"proxy[{index}] must be an object")
        _require_exact_keys(proxy, PROXY_KEYS, f"proxy[{index}]")
        key = (str(proxy["model"]), str(proxy["finding"]))
        if key in proxy_geometry:
            raise ContractError(f"duplicate proxy definition for {key}")
        if key[0] not in model_provenance or not key[1]:
            raise ContractError(f"proxy references unknown model or empty finding: {key}")
        proxy_source[key] = _require_sha256(proxy["proxy_source_sha256"], f"proxy[{index}].proxy_source_sha256")
        geometry = _proxy_geometry(proxy, dimension_by_model.get(key[0]))
        dimension_by_model.setdefault(key[0], geometry["dimension"])
        proxy_geometry[key] = geometry

    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise ContractError("records must be a non-empty list")
    cells: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    orbit_cells: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    image_metadata: dict[str, tuple[str, str]] = {}
    source_to_image: dict[str, str] = {}
    transformed_metadata: dict[tuple[str, str], str] = {}
    transformed_to_identity: dict[str, tuple[str, str]] = {}
    prompt_hashes: dict[tuple[str, str], str] = {}
    proposition_hashes: dict[str, str] = {}
    expected_cells = {(r, p) for r in renders for p in prompts}
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ContractError(f"record[{index}] must be an object")
        _require_exact_keys(row, RECORD_KEYS, f"record[{index}]")
        model = str(row["model"])
        image = str(row["image_id"])
        patient = str(row["patient_id"])
        finding = str(row["finding"])
        render = str(row["render_id"])
        prompt = str(row["prompt_id"])
        if not all((model, image, patient, finding, render, prompt)):
            raise ContractError(f"record[{index}] has an empty identity field")
        if model not in model_provenance or (model, finding) not in proxy_geometry:
            raise ContractError(f"record[{index}] lacks frozen model/proxy provenance")
        if render not in renders or prompt not in prompts:
            raise ContractError(f"record[{index}] is outside the frozen product grid")
        source_hash = _require_sha256(row["source_image_sha256"], f"record[{index}].source_image_sha256")
        transformed_hash = _require_sha256(
            row["transformed_image_sha256"], f"record[{index}].transformed_image_sha256"
        )
        prompt_hash = _require_sha256(row["exact_prompt_sha256"], f"record[{index}].exact_prompt_sha256")
        proposition_hash = _require_sha256(row["proposition_sha256"], f"record[{index}].proposition_sha256")
        key = (model, image, finding, render, prompt)
        if key in cells:
            raise ContractError(f"duplicate product cell {key}")
        embedding = _vector(
            row["embedding"], f"record[{index}].embedding", dimension_by_model[model]
        )
        cells[key] = {**dict(row), "embedding_array": embedding}
        orbit_cells[(model, image, finding)].add((render, prompt))

        previous = image_metadata.setdefault(image, (patient, source_hash))
        if previous != (patient, source_hash):
            raise ContractError(f"patient/source identity drift for image {image}")
        other_image = source_to_image.setdefault(source_hash, image)
        if other_image != image:
            raise ContractError("duplicate source image bytes occur under different image_id values")
        transform_key = (image, render)
        previous_transform = transformed_metadata.setdefault(transform_key, transformed_hash)
        if previous_transform != transformed_hash:
            raise ContractError(f"transformed image hash changes across prompts/models for {transform_key}")
        other_identity = transformed_to_identity.setdefault(transformed_hash, transform_key)
        if other_identity != transform_key:
            raise ContractError("duplicate transformed image bytes occur under different image/render identities")
        prompt_key = (finding, prompt)
        previous_prompt = prompt_hashes.setdefault(prompt_key, prompt_hash)
        if previous_prompt != prompt_hash:
            raise ContractError(f"exact prompt hash changes across images/models for {prompt_key}")
        previous_prop = proposition_hashes.setdefault(finding, proposition_hash)
        if previous_prop != proposition_hash:
            raise ContractError(f"proposition hash changes within finding {finding}")

    for key, value in orbit_cells.items():
        if value != expected_cells:
            raise ContractError(f"incomplete factorial orbit {key}")
    by_model: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for model, image, finding in orbit_cells:
        by_model[model].add((image, finding))
    expected_orbits = next(iter(by_model.values()))
    if set(by_model) != set(model_provenance):
        raise ContractError("records do not cover every frozen model")
    if any(value != expected_orbits for value in by_model.values()):
        raise ContractError("models do not share exact held-out orbit identities")

    return {
        "cells": cells,
        "proxy_geometry": proxy_geometry,
        "proxy_source": proxy_source,
        "dimension_by_model": dimension_by_model,
        "images": set(image_metadata),
        "patients": {value[0] for value in image_metadata.values()},
        "source_hashes": {value[1] for value in image_metadata.values()},
        "transformed_hashes": set(transformed_to_identity),
        "orbits": set(orbit_cells),
    }


def _signed_distance(embedding: np.ndarray, geometry: Mapping[str, Any]) -> float:
    return float(np.dot(embedding - geometry["midpoint"], geometry["direction"]))


def _prejoint_features(
    h00: np.ndarray,
    h10: np.ndarray,
    h01: np.ndarray,
    d00: float,
    d10: float,
    d01: float,
) -> dict[str, float]:
    """Return predictor covariates; the signature intentionally cannot receive h11."""
    render_displacement = h10 - h00
    wording_displacement = h01 - h00
    additive_embedding = h10 + h01 - h00
    additive_distance = d10 + d01 - d00
    displacement_denominator = (
        float(np.linalg.norm(render_displacement))
        * float(np.linalg.norm(wording_displacement))
    )
    marginal_cosine = (
        float(np.dot(render_displacement, wording_displacement))
        / displacement_denominator
        if displacement_denominator > EPS
        else 0.0
    )
    clean_positive = d00 >= 0.0
    return {
        "clean_abs_distance": abs(d00),
        "render_only_abs_distance": abs(d10),
        "wording_only_abs_distance": abs(d01),
        "minimum_single_axis_abs_distance": min(abs(d10), abs(d01)),
        "render_boundary_approach": abs(d00) - abs(d10),
        "wording_boundary_approach": abs(d00) - abs(d01),
        "additive_abs_distance": abs(additive_distance),
        "additive_boundary_approach": abs(d00) - abs(additive_distance),
        "additive_crosses_clean": float(
            (additive_distance >= 0.0) != clean_positive
        ),
        "render_displacement_norm": float(np.linalg.norm(render_displacement)),
        "wording_displacement_norm": float(np.linalg.norm(wording_displacement)),
        "additive_displacement_norm": float(np.linalg.norm(additive_embedding - h00)),
        "marginal_displacement_cosine": marginal_cosine,
        "marginal_displacement_norm_gap": abs(
            float(np.linalg.norm(render_displacement))
            - float(np.linalg.norm(wording_displacement))
        ),
    }


def _posthoc_joint_endpoint_descriptor(
    h00: np.ndarray,
    h10: np.ndarray,
    h01: np.ndarray,
    h11: np.ndarray,
    d00: float,
    d10: float,
    d01: float,
    d11: float,
    direction: np.ndarray,
) -> dict[str, float]:
    """Describe observed h11 after scoring; never consumed by `_matrix`."""
    residual = h11 - h10 - h01 + h00
    on_axis = np.dot(residual, direction) * direction
    return {
        "joint_abs_distance": abs(d11),
        "joint_boundary_approach": abs(d00) - abs(d11),
        "signed_product_residual_abs": abs(d11 - d10 - d01 + d00),
        "embedding_product_residual_norm": float(np.linalg.norm(residual)),
        "joint_displacement_norm": float(np.linalg.norm(h11 - h00)),
        "product_residual_off_axis_norm": float(np.linalg.norm(residual - on_axis)),
    }


def _comparison_rows(payload: Mapping[str, Any], validated: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = validated["cells"]
    r0 = str(payload["baseline_render"])
    p0 = str(payload["baseline_prompt"])
    rows = []
    for model, image, finding in sorted(validated["orbits"]):
        geometry = validated["proxy_geometry"][(model, finding)]
        for render in payload["primary_renders"]:
            if render == r0:
                continue
            for prompt in payload["primary_prompts"]:
                if prompt == p0:
                    continue
                h00 = cells[(model, image, finding, r0, p0)]["embedding_array"]
                h10 = cells[(model, image, finding, render, p0)]["embedding_array"]
                h01 = cells[(model, image, finding, r0, prompt)]["embedding_array"]
                h11 = cells[(model, image, finding, render, prompt)]["embedding_array"]
                d00, d10, d01, d11 = (
                    _signed_distance(h, geometry) for h in (h00, h10, h01, h11)
                )
                clean_positive = d00 >= 0.0
                features = _prejoint_features(h00, h10, h01, d00, d10, d01)
                posthoc_descriptor = _posthoc_joint_endpoint_descriptor(
                    h00, h10, h01, h11, d00, d10, d01, d11,
                    geometry["direction"],
                )
                render_positive = d10 >= 0.0
                wording_positive = d01 >= 0.0
                joint_positive = d11 >= 0.0
                rows.append({
                    "model": model,
                    "image_id": image,
                    "patient_digest": _cluster_digest(cells[(model, image, finding, r0, p0)]["patient_id"]),
                    "finding": finding,
                    "render_id": render,
                    "prompt_id": prompt,
                    "features": features,
                    "posthoc_joint_endpoint_descriptor": posthoc_descriptor,
                    "generic_joint_boundary_crossing_endpoint": bool(
                        joint_positive != clean_positive
                    ),
                    "single_axes_stable_joint_crossing": bool(
                        render_positive == clean_positive
                        and wording_positive == clean_positive
                        and joint_positive != clean_positive
                    ),
                })
    return rows


def _matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    result = np.asarray([[float(row["features"][key]) for key in BASE_FEATURES] for row in rows])
    if result.ndim != 2 or result.shape[1] != len(BASE_FEATURES) or not np.isfinite(result).all():
        raise ContractError("non-finite semantic-boundary feature matrix")
    return result


def _targets(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([
        int(row["generic_joint_boundary_crossing_endpoint"]) for row in rows
    ], dtype=int)


def _patient_class_counts(y: np.ndarray, patient_digests: np.ndarray) -> np.ndarray:
    return np.asarray([
        len(set(patient_digests[y == value].tolist())) for value in (0, 1)
    ], dtype=int)


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    patient_digests: np.ndarray,
    *,
    stratum: str,
    minimum_patient_clusters_per_class: int = MIN_CLASS_COUNT,
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    counts = _patient_class_counts(y, patient_digests)
    if np.any(counts < minimum_patient_clusters_per_class):
        raise ContractError(
            "dev generic boundary target needs >= "
            f"{minimum_patient_clusters_per_class} distinct patient clusters per class "
            f"within {stratum}; "
            f"got {counts.tolist()}"
        )
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale <= EPS] = 1.0
    model = LogisticRegression(
        C=LOGISTIC_C, solver="lbfgs", max_iter=2000,
        class_weight=None, fit_intercept=True, random_state=0,
    )
    model.fit((x - mean) / scale, y)
    if model.n_iter_[0] >= model.max_iter:
        raise ContractError("fixed dev logistic fit did not converge")
    return mean, scale, model


def _sigmoid(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value, dtype=float)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _apply_parameters(parameters: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    if parameters.get("feature_names") != list(BASE_FEATURES):
        raise ContractError("frozen feature order changed")
    mean = np.asarray(parameters.get("mean"), dtype=float)
    scale = np.asarray(parameters.get("scale"), dtype=float)
    coefficient = np.asarray(parameters.get("coefficient"), dtype=float)
    intercept = float(parameters.get("intercept"))
    expected = (len(BASE_FEATURES),)
    if mean.shape != expected or scale.shape != expected or coefficient.shape != expected:
        raise ContractError("malformed frozen boundary predictor")
    if not all(np.isfinite(v).all() for v in (mean, scale, coefficient)) or not math.isfinite(intercept):
        raise ContractError("non-finite frozen boundary predictor")
    if np.any(scale <= 0):
        raise ContractError("invalid frozen feature scale")
    return _sigmoid(((x - mean) / scale) @ coefficient + intercept)


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(y, minlength=2)
    result = {
        "n": int(y.size),
        "class_counts": counts.tolist(),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "mean_risk": float(np.mean(probability)),
        "generic_crossing_rate": float(np.mean(y)),
    }
    result["auroc"] = float(roc_auc_score(y, probability)) if np.all(counts > 0) else None
    return result


def _group_oof(
    rows: Sequence[Mapping[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    *,
    stratum: str,
) -> np.ndarray:
    patient_digests = np.asarray([str(row["patient_digest"]) for row in rows])
    ordered_patients = sorted(
        set(patient_digests.tolist()),
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    fold_by_patient = {
        patient: rank % DEV_FOLDS for rank, patient in enumerate(ordered_patients)
    }
    fold = np.asarray([fold_by_patient[value] for value in patient_digests])
    prediction = np.full(y.size, np.nan)
    for current in range(DEV_FOLDS):
        test = fold == current
        if not np.any(test):
            raise ContractError(f"dev patient-cluster fold {current} is empty")
        train = ~test
        mean, scale, model = _fit_logistic(
            x[train], y[train], patient_digests[train],
            stratum=f"{stratum}/fold-{current}",
            minimum_patient_clusters_per_class=1,
        )
        prediction[test] = model.predict_proba((x[test] - mean) / scale)[:, 1]
    if not np.isfinite(prediction).all():
        raise ContractError("dev OOF prediction coverage is incomplete")
    return prediction


def fit_dev_control(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fit the fixed reader-free predictor on dev only; no threshold is selected."""
    validated = _validate_payload(payload, "dev_fit")
    rows = _comparison_rows(payload, validated)
    x = _matrix(rows)
    y = _targets(rows)
    oof = np.full(y.size, np.nan)
    fixed_fits: dict[str, Any] = {}
    dev_by_model_finding: dict[str, Any] = {}
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(str(row["model"]), str(row["finding"]))].append(index)
    for (model_name, finding), indices_list in sorted(strata.items()):
        indices = np.asarray(indices_list, dtype=int)
        stratum = f"{model_name}\u001f{finding}"
        subrows = [rows[index] for index in indices_list]
        subx = x[indices]
        suby = y[indices]
        patient_digests = np.asarray([
            str(row["patient_digest"]) for row in subrows
        ])
        mean, scale, fitted = _fit_logistic(
            subx, suby, patient_digests, stratum=stratum
        )
        oof[indices] = _group_oof(subrows, subx, suby, stratum=stratum)
        fixed_fits[stratum] = {
            "family": "l2_logistic_regression",
            "C": LOGISTIC_C,
            "solver": "lbfgs",
            "dev_patient_cluster_folds": DEV_FOLDS,
            "fit_scope": "model_by_finding",
            "model": model_name,
            "finding": finding,
            "feature_names": list(BASE_FEATURES),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "coefficient": fitted.coef_[0].tolist(),
            "intercept": float(fitted.intercept_[0]),
            "patient_cluster_class_counts": _patient_class_counts(
                suby, patient_digests
            ).tolist(),
            "n_patient_clusters": int(len(set(patient_digests.tolist()))),
        }
        dev_by_model_finding[stratum] = _metrics(suby, oof[indices])
    if not np.isfinite(oof).all():
        raise ContractError("dev OOF model-by-finding prediction coverage is incomplete")
    proxy_content = {
        f"{model_name}\u001f{finding}": geometry["content_sha256"]
        for (model_name, finding), geometry in sorted(validated["proxy_geometry"].items())
    }
    result = {
        "version": BUNDLE_VERSION,
        "status": NON_AUTHORIZING,
        "authorized": False,
        "reader_or_clinical_outcomes_used": False,
        "binary_decision_or_threshold": None,
        "threshold_selection": "prohibited; continuous generic risk only",
        "source_sha256": _module_sha256(),
        "dev_input_sha256": object_sha256(payload),
        "source_manifest_sha256": str(payload["source_manifest_sha256"]).lower(),
        "frozen_contract": {
            "schema_version": payload["schema_version"],
            "baseline_render": payload["baseline_render"],
            "baseline_prompt": payload["baseline_prompt"],
            "primary_renders": payload["primary_renders"],
            "primary_prompts": payload["primary_prompts"],
            "model_provenance": payload["model_provenance"],
            "representation_spec": payload["representation_spec"],
            "representation_dimension_by_model": validated["dimension_by_model"],
            "transform_spec_sha256": payload["transform_spec_sha256"],
            "prompt_spec_sha256": payload["prompt_spec_sha256"],
            "proxy_source_sha256": {
                f"{model_name}\u001f{finding}": value
                for (model_name, finding), value in sorted(validated["proxy_source"].items())
            },
            "proxy_content_sha256": proxy_content,
        },
        "dev_identity_digests": {
            "images": sorted(_cluster_digest(v) for v in validated["images"]),
            "patients": sorted(_cluster_digest(v) for v in validated["patients"]),
            "source_image_sha256": sorted(validated["source_hashes"]),
            "transformed_image_sha256": sorted(validated["transformed_hashes"]),
        },
        "predictor_covariate_contract": {
            "allowed_cells": ["h00", "h10", "h01"],
            "prohibited_predictor_cell": "h11",
            "h11_role": "endpoint_and_separate_posthoc_descriptor_only",
            "feature_names": list(BASE_FEATURES),
            "posthoc_descriptor_fields": list(POSTHOC_DESCRIPTOR_FIELDS),
        },
        "fixed_fits_by_model_finding": fixed_fits,
        "dev_oof_reader_free_diagnostic": {
            "overall": _metrics(y, oof),
            "by_model_finding": dev_by_model_finding,
        },
        "n_dev_product_comparisons": len(rows),
        "n_dev_orbits": len(validated["orbits"]),
    }
    return _seal(result, "bundle_sha256")


def _verify_contract(payload: Mapping[str, Any], validated: Mapping[str, Any], bundle: Mapping[str, Any]) -> None:
    frozen = bundle.get("frozen_contract")
    if not isinstance(frozen, Mapping):
        raise ContractError("missing frozen semantic-boundary contract")
    direct = (
        "schema_version", "baseline_render", "baseline_prompt", "primary_renders",
        "primary_prompts", "model_provenance", "representation_spec",
        "transform_spec_sha256", "prompt_spec_sha256",
    )
    for key in direct:
        if payload.get(key) != frozen.get(key):
            raise ContractError(f"confirmation provenance/geometry drift at {key}")
    if validated["dimension_by_model"] != frozen.get("representation_dimension_by_model"):
        raise ContractError("confirmation representation dimension drift")
    source = {
        f"{model}\u001f{finding}": value
        for (model, finding), value in sorted(validated["proxy_source"].items())
    }
    content = {
        f"{model}\u001f{finding}": geometry["content_sha256"]
        for (model, finding), geometry in sorted(validated["proxy_geometry"].items())
    }
    if source != frozen.get("proxy_source_sha256") or content != frozen.get("proxy_content_sha256"):
        raise ContractError("confirmation text-proxy provenance/content drift")

    if str(payload["source_manifest_sha256"]).lower() == bundle.get("source_manifest_sha256"):
        raise ContractError("dev and confirmation cannot reuse the same source manifest")
    dev = bundle.get("dev_identity_digests")
    if not isinstance(dev, Mapping):
        raise ContractError("missing dev identity digests")
    confirmation = {
        "images": {_cluster_digest(v) for v in validated["images"]},
        "patients": {_cluster_digest(v) for v in validated["patients"]},
        "source_image_sha256": set(validated["source_hashes"]),
        "transformed_image_sha256": set(validated["transformed_hashes"]),
    }
    for key, values in confirmation.items():
        overlap = values.intersection(set(dev.get(key, [])))
        if overlap:
            raise ContractError(f"dev/confirmation {key} overlap")


def apply_confirmation_control(payload: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the dev-frozen reader-free control to disjoint confirmation orbits."""
    _verify_seal(bundle, "bundle_sha256")
    if bundle.get("version") != BUNDLE_VERSION or bundle.get("source_sha256") != _module_sha256():
        raise ContractError("bundle is not bound to the current boundary-control source")
    if bundle.get("reader_or_clinical_outcomes_used") is not False:
        raise ContractError("bundle reader-free claim was altered")
    if bundle.get("binary_decision_or_threshold") is not None:
        raise ContractError("outcome-guided threshold is prohibited")
    validated = _validate_payload(payload, "confirmation_locked")
    _verify_contract(payload, validated, bundle)
    rows = _comparison_rows(payload, validated)
    x = _matrix(rows)
    y = _targets(rows)
    contract = bundle.get("predictor_covariate_contract")
    expected_contract = {
        "allowed_cells": ["h00", "h10", "h01"],
        "prohibited_predictor_cell": "h11",
        "h11_role": "endpoint_and_separate_posthoc_descriptor_only",
        "feature_names": list(BASE_FEATURES),
        "posthoc_descriptor_fields": list(POSTHOC_DESCRIPTOR_FIELDS),
    }
    if contract != expected_contract:
        raise ContractError("frozen pre-joint predictor covariate contract changed")
    fixed_fits = bundle.get("fixed_fits_by_model_finding")
    if not isinstance(fixed_fits, Mapping):
        raise ContractError("missing frozen model-by-finding boundary predictors")
    probability = np.full(y.size, np.nan)
    confirmation_by_model_finding: dict[str, Any] = {}
    for model_name, finding in sorted({
        (str(row["model"]), str(row["finding"])) for row in rows
    }):
        stratum = f"{model_name}\u001f{finding}"
        parameters = fixed_fits.get(stratum)
        if not isinstance(parameters, Mapping):
            raise ContractError(f"missing frozen boundary predictor for {stratum}")
        if parameters.get("fit_scope") != "model_by_finding":
            raise ContractError(f"boundary predictor scope drift for {stratum}")
        if parameters.get("model") != model_name or parameters.get("finding") != finding:
            raise ContractError(f"boundary predictor identity drift for {stratum}")
        indices = np.asarray([
            index for index, row in enumerate(rows)
            if row["model"] == model_name and row["finding"] == finding
        ])
        probability[indices] = _apply_parameters(parameters, x[indices])
        confirmation_by_model_finding[stratum] = _metrics(
            y[indices], probability[indices]
        )
    if set(fixed_fits) != set(confirmation_by_model_finding):
        raise ContractError("confirmation model-by-finding predictor strata drift")
    if not np.isfinite(probability).all():
        raise ContractError("confirmation model-by-finding prediction coverage is incomplete")
    by_pair = {}
    for render in payload["primary_renders"]:
        if render == payload["baseline_render"]:
            continue
        for prompt in payload["primary_prompts"]:
            if prompt == payload["baseline_prompt"]:
                continue
            indices = np.asarray([
                index for index, row in enumerate(rows)
                if row["render_id"] == render and row["prompt_id"] == prompt
            ])
            by_pair[f"{render}\u001f{prompt}"] = _metrics(y[indices], probability[indices])
    scored_rows = []
    for row, risk in zip(rows, probability):
        public = dict(row)
        public["boundary_proximity_risk"] = float(risk)
        scored_rows.append(public)
    result = {
        "version": RESULT_VERSION,
        "status": NON_AUTHORIZING,
        "authorized": False,
        "fit_or_refit_on_confirmation": False,
        "reader_or_clinical_outcomes_used": False,
        "binary_decision_or_threshold": None,
        "held_out_orbit_evaluation": True,
        "dev_bundle_sha256": bundle["bundle_sha256"],
        "confirmation_input_sha256": object_sha256(payload),
        "source_sha256": _module_sha256(),
        "n_confirmation_orbits": len(validated["orbits"]),
        "n_confirmation_product_comparisons": len(rows),
        "reader_free_metrics": {
            "overall": _metrics(y, probability),
            "by_model_finding": confirmation_by_model_finding,
            "by_transform_pair": by_pair,
        },
        "scored_product_comparisons": scored_rows,
        "interpretation": (
            "The model-by-finding continuous score predicts the held-out h11 reader-free "
            "crossing endpoint using h00/h10/h01 covariates only. The separately listed "
            "post-hoc h11 descriptor was not available to the predictor. This is a generic "
            "collision-control covariate, not a clinical outcome, an invariance certificate, "
            "an absorption test, or an authorization gate."
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
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit-dev")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    apply = subparsers.add_parser("apply-confirmation")
    apply.add_argument("--input", type=Path, required=True)
    apply.add_argument("--dev-bundle", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit-dev":
        result = fit_dev_control(_load(args.input))
    else:
        result = apply_confirmation_control(_load(args.input), _load(args.dev_bundle))
    _write(args.output, result)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": result.get("bundle_sha256", result.get("result_sha256")),
        "status": NON_AUTHORIZING,
    }, indent=2))


if __name__ == "__main__":
    main()
