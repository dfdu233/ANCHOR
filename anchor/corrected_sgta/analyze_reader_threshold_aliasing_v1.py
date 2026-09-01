#!/usr/bin/env python3
"""Locked alternative-explanation control for reader-threshold aliasing.

The development split is the only place where coefficients and the candidate
reader ordering are learned.  Confirmation applies the serialized development
models once.  The exact R8/R9/R10 pattern is tested only as an increment over
vote count, finding, model, and finding/model-specific clean-score sensitivity.

This is an observational exclusion control.  In particular, the identity
increment is identically zero for unanimous 000/111 patterns.  Consequently a
positive disagreement result cannot be promoted to a clear-case hallucination
mechanism unless an independently specified clear-case predictor also passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


VERSION = "reader-threshold-aliasing-control-v1"
FIT_SCHEMA = "reader-threshold-aliasing-dev-fit-v1"
CONFIRM_SCHEMA = "reader-threshold-aliasing-confirmation-v1"
READERS = ("R8", "R9", "R10")
MODELS = ("huatuo", "hulu")
PRIMARY_FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "other_lesion",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
PATTERNS = tuple(f"{value:03b}" for value in range(8))
DISAGREEMENT_CONTRASTS = ("010", "100", "101", "110")
ENDPOINTS = ("positive_commitment", "clinical_error")


class AliasingError(ValueError):
    """Raised when the locked reader-aliasing contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise AliasingError(f"write-once artifact collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _votes(source: Any) -> tuple[int, int, int]:
    if isinstance(source, Mapping):
        if set(source) != set(READERS):
            raise AliasingError("reader_votes must contain exactly R8/R9/R10")
        values = tuple(source[name] for name in READERS)
    elif isinstance(source, list) and len(source) == 3:
        if not all(isinstance(item, Mapping) for item in source):
            raise AliasingError(
                "positional reader vote lists are ambiguous; named rad_id records are required"
            )
        mapping = {str(item.get("rad_id")): item.get("vote") for item in source}
        if len(mapping) != 3 or set(mapping) != set(READERS):
            raise AliasingError("reader vote records must be the fixed R8/R9/R10 panel")
        values = tuple(mapping[name] for name in READERS)
    else:
        raise AliasingError(
            "reader_votes must be a fixed-panel mapping or three named rad_id records"
        )
    if any(isinstance(value, bool) or value not in (0, 1) for value in values):
        raise AliasingError("reader votes must be integer 0/1")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def load_rows(
    path: Path,
    expected_stage: str,
    *,
    findings: Sequence[str] = PRIMARY_FINDINGS,
    models: Sequence[str] = MODELS,
    require_complete_cells: bool = True,
) -> list[dict[str, Any]]:
    if expected_stage not in {"dev_fit", "confirmation_locked"}:
        raise AliasingError("stage must be dev_fit or confirmation_locked")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_findings, allowed_models = set(findings), set(models)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            source = json.loads(line)
        except json.JSONDecodeError as error:
            raise AliasingError(f"{path}:{line_number}: invalid JSON") from error
        required = {
            "record_key", "image_id", "finding", "model", "stage", "task",
            "condition", "reader_votes", "positive_votes", "clean_margin",
            "endpoint", "target",
        }
        missing = required - set(source)
        if missing:
            raise AliasingError(f"{path}:{line_number}: missing {sorted(missing)}")
        key = str(source["record_key"])
        finding, model = str(source["finding"]), str(source["model"])
        votes = _votes(source["reader_votes"])
        target = source["target"]
        margin = source["clean_margin"]
        if not key or key in seen:
            raise AliasingError(f"missing/duplicate record_key: {key}")
        if source["stage"] != expected_stage or source["task"] != "ce":
            raise AliasingError(f"{key}: wrong stage/task")
        if source["condition"] != "clean":
            raise AliasingError(f"{key}: only clean-condition scores are admissible")
        if finding not in allowed_findings or model not in allowed_models:
            raise AliasingError(f"{key}: finding/model outside frozen closure")
        if source["endpoint"] not in ENDPOINTS:
            raise AliasingError(f"{key}: unsupported endpoint")
        if isinstance(target, bool) or target not in (0, 1):
            raise AliasingError(f"{key}: target must be integer 0/1")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin):
            raise AliasingError(f"{key}: clean_margin must be finite")
        if int(source["positive_votes"]) != sum(votes):
            raise AliasingError(f"{key}: vote count disagrees with reader pattern")
        seen.add(key)
        rows.append(
            {
                "record_key": key,
                "image_id": str(source["image_id"]),
                "finding": finding,
                "model": model,
                "stage": expected_stage,
                "endpoint": str(source["endpoint"]),
                "votes": votes,
                "pattern": "".join(str(value) for value in votes),
                "positive_votes": sum(votes),
                "clean_margin": float(margin),
                "target": int(target),
            }
        )
    if not rows:
        raise AliasingError(f"empty input: {path}")
    if len({row["endpoint"] for row in rows}) != 1:
        raise AliasingError("one analysis file must contain exactly one endpoint")
    duplicate_units: set[tuple[str, str, str]] = set()
    units: set[tuple[str, str, str]] = set()
    for row in rows:
        unit = (row["image_id"], row["finding"], row["model"])
        if unit in units:
            duplicate_units.add(unit)
        units.add(unit)
    if duplicate_units:
        raise AliasingError(f"duplicate image/finding/model units: {len(duplicate_units)}")
    if require_complete_cells:
        observed_findings = {row["finding"] for row in rows}
        observed_models = {row["model"] for row in rows}
        if observed_findings != allowed_findings or observed_models != allowed_models:
            raise AliasingError("input does not cover the frozen finding/model closure")
        expected_quota = 20 if expected_stage == "dev_fit" else 60
        for model in models:
            for finding in findings:
                for vote_count in range(4):
                    observed = sum(
                        row["model"] == model
                        and row["finding"] == finding
                        and row["positive_votes"] == vote_count
                        for row in rows
                    )
                    if observed != expected_quota:
                        raise AliasingError(
                            f"frozen cell quota mismatch: {model}/{finding}/{vote_count}/3 "
                            f"expected={expected_quota} observed={observed}"
                        )
    return rows


def group_folds(rows: Sequence[Mapping[str, Any]], folds: int, seed: int) -> np.ndarray:
    images = sorted(
        {str(row["image_id"]) for row in rows},
        key=lambda value: hashlib.sha256(
            f"{seed}:reader-alias-fold:{value}".encode()
        ).hexdigest(),
    )
    if folds < 2 or len(images) < folds:
        raise AliasingError("group cross-fitting requires >=2 folds and images")
    assignment = {image: index % folds for index, image in enumerate(images)}
    return np.asarray([assignment[str(row["image_id"])] for row in rows], dtype=int)


def fit_standardization(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for model in sorted({str(row["model"]) for row in rows}):
        for finding in sorted({str(row["finding"]) for row in rows}):
            values = np.asarray(
                [
                    float(row["clean_margin"])
                    for row in rows
                    if row["model"] == model and row["finding"] == finding
                ]
            )
            if len(values) < 2 or not np.isfinite(values).all():
                raise AliasingError(f"cannot standardize {model}/{finding}")
            scale = float(values.std())
            if scale < 1e-8:
                raise AliasingError(f"zero clean-margin variance: {model}/{finding}")
            result[f"{model}|{finding}"] = {
                "mean": float(values.mean()), "scale": scale
            }
    return result


def feature_names(findings: Sequence[str], models: Sequence[str], augmented: bool) -> list[str]:
    cells = [(model, finding) for model in models for finding in findings]
    names = [f"vote={value}" for value in (1, 2, 3)]
    names += [f"cell={model}|{finding}" for model, finding in cells[1:]]
    names += ["clean_z"]
    names += [f"clean_z*cell={model}|{finding}" for model, finding in cells[1:]]
    if augmented:
        for pattern in DISAGREEMENT_CONTRASTS:
            names.append(f"pattern={pattern}")
        for model, finding in cells[1:]:
            for pattern in DISAGREEMENT_CONTRASTS:
                names.append(f"pattern={pattern}*cell={model}|{finding}")
    return names


def design_matrix(
    rows: Sequence[Mapping[str, Any]],
    standardization: Mapping[str, Mapping[str, float]],
    findings: Sequence[str],
    models: Sequence[str],
    augmented: bool,
) -> np.ndarray:
    vectors: list[list[float]] = []
    cells = [(candidate_model, candidate_finding) for candidate_model in models for candidate_finding in findings]
    for row in rows:
        finding, model = str(row["finding"]), str(row["model"])
        record = standardization.get(f"{model}|{finding}")
        if record is None:
            raise AliasingError(f"unseen model/finding at apply: {model}/{finding}")
        clean_z = (float(row["clean_margin"]) - float(record["mean"])) / float(record["scale"])
        vector = [float(row["positive_votes"] == value) for value in (1, 2, 3)]
        vector += [float((model, finding) == value) for value in cells[1:]]
        vector += [clean_z]
        vector += [clean_z * float((model, finding) == value) for value in cells[1:]]
        if augmented:
            indicators = [float(row["pattern"] == pattern) for pattern in DISAGREEMENT_CONTRASTS]
            vector += indicators
            for value in cells[1:]:
                vector += [indicator * float((model, finding) == value) for indicator in indicators]
        vectors.append(vector)
    output = np.asarray(vectors, dtype=float)
    expected = len(feature_names(findings, models, augmented))
    if output.shape != (len(rows), expected) or not np.isfinite(output).all():
        raise AliasingError("feature construction failed")
    return output


def fit_logistic(x: np.ndarray, y: np.ndarray, c_value: float) -> dict[str, Any]:
    if np.unique(y).size != 2:
        raise AliasingError("logistic fit requires both endpoint classes")
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000, random_state=0)
    model.fit(x, y)
    return {
        "coefficient": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
        "c_value": float(c_value),
    }


def predict_logistic(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    coefficient = np.asarray(model["coefficient"], dtype=float)
    if coefficient.shape != (x.shape[1],):
        raise AliasingError("serialized coefficient dimension mismatch")
    linear = x @ coefficient + float(model["intercept"])
    positive = np.empty(len(linear), dtype=float)
    mask = linear >= 0
    positive[mask] = 1.0 / (1.0 + np.exp(-linear[mask]))
    exponential = np.exp(linear[~mask])
    positive[~mask] = exponential / (1.0 + exponential)
    return positive


def linear_predictor(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    coefficient = np.asarray(model["coefficient"], dtype=float)
    if coefficient.shape != (x.shape[1],):
        raise AliasingError("serialized coefficient dimension mismatch")
    return x @ coefficient + float(model["intercept"])


def fit_pattern_increment(
    pattern_x: np.ndarray, y: np.ndarray, frozen_offset: np.ndarray, c_value: float
) -> dict[str, Any]:
    """Fit only the pattern residual while the saturated baseline stays frozen."""

    if pattern_x.shape[0] != len(y) or len(y) != len(frozen_offset):
        raise AliasingError("pattern-residual inputs have unequal lengths")
    penalty = 1.0 / (max(float(c_value), 1e-12) * len(y))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = frozen_offset + pattern_x @ beta
        probability = np.empty(len(eta), dtype=float)
        mask = eta >= 0
        probability[mask] = 1.0 / (1.0 + np.exp(-eta[mask]))
        exponential = np.exp(eta[~mask])
        probability[~mask] = exponential / (1.0 + exponential)
        loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
        loss += 0.5 * penalty * float(beta @ beta)
        gradient = pattern_x.T @ (probability - y) / len(y) + penalty * beta
        return loss, gradient

    initial = np.zeros(pattern_x.shape[1], dtype=float)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not result.success or not np.isfinite(result.x).all():
        raise AliasingError(f"pattern residual optimization failed: {result.message}")
    return {
        "coefficient": result.x.astype(float).tolist(),
        "baseline_frozen": True,
        "l2_penalty": penalty,
        "optimizer": "L-BFGS-B",
    }


def predict_with_increment(
    baseline_model: Mapping[str, Any], increment_model: Mapping[str, Any],
    baseline_x: np.ndarray, pattern_x: np.ndarray,
) -> np.ndarray:
    coefficient = np.asarray(increment_model["coefficient"], dtype=float)
    if coefficient.shape != (pattern_x.shape[1],):
        raise AliasingError("serialized pattern coefficient dimension mismatch")
    linear = linear_predictor(baseline_model, baseline_x) + pattern_x @ coefficient
    probability = np.empty(len(linear), dtype=float)
    mask = linear >= 0
    probability[mask] = 1.0 / (1.0 + np.exp(-linear[mask]))
    exponential = np.exp(linear[~mask])
    probability[~mask] = exponential / (1.0 + exponential)
    return probability


def auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positive, negative = y == 1, y == 0
    if not positive.any() or not negative.any():
        raise AliasingError("AUROC requires both classes")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def nll(y: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    target = np.asarray(y, dtype=float)
    return float(np.mean(-(target * np.log(p) + (1 - target) * np.log(1 - p))))


def metrics(y: np.ndarray, baseline: np.ndarray, augmented: np.ndarray) -> dict[str, float]:
    base_nll, aug_nll = nll(y, baseline), nll(y, augmented)
    return {
        "baseline_auroc": auc(y, baseline),
        "augmented_auroc": auc(y, augmented),
        "delta_auroc": auc(y, augmented) - auc(y, baseline),
        "baseline_nll": base_nll,
        "augmented_nll": aug_nll,
        "relative_nll_improvement": (base_nll - aug_nll) / max(base_nll, 1e-12),
    }


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], baseline: np.ndarray, augmented: np.ndarray,
    *, draws: int, seed: int,
) -> dict[str, Any]:
    if draws < 100:
        raise AliasingError("at least 100 bootstrap draws required")
    y = np.asarray([row["target"] for row in rows], dtype=int)
    groups = np.asarray([row["image_id"] for row in rows], dtype=object)
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    delta, relative = [], []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        chosen = np.concatenate([indices[group] for group in sampled])
        if np.unique(y[chosen]).size != 2:
            continue
        value = metrics(y[chosen], baseline[chosen], augmented[chosen])
        delta.append(value["delta_auroc"])
        relative.append(value["relative_nll_improvement"])
    if len(delta) < max(50, draws // 2):
        raise AliasingError("too many degenerate image-cluster bootstrap draws")
    point = metrics(y, baseline, augmented)
    return {
        "point": point,
        "delta_auroc": {
            "estimate": point["delta_auroc"],
            "ci_low": float(np.quantile(delta, .025)),
            "ci_high": float(np.quantile(delta, .975)),
        },
        "relative_nll_improvement": {
            "estimate": point["relative_nll_improvement"],
            "ci_low": float(np.quantile(relative, .025)),
            "ci_high": float(np.quantile(relative, .975)),
        },
        "draws_requested": draws,
        "valid_draws": len(delta),
        "cluster_unit": "whole_image_id",
        "n_clusters": len(unique),
    }


def crossfit(
    rows: Sequence[Mapping[str, Any]], *, folds: int, seed: int, c_value: float,
    findings: Sequence[str], models: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    fold_ids = group_folds(rows, folds, seed)
    baseline = np.full(len(rows), np.nan)
    augmented = np.full(len(rows), np.nan)
    audits: list[dict[str, Any]] = []
    y = np.asarray([row["target"] for row in rows], dtype=int)
    for fold in range(folds):
        train_idx, held_idx = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        train = [rows[index] for index in train_idx]
        held = [rows[index] for index in held_idx]
        standardization = fit_standardization(train)
        xb_train = design_matrix(train, standardization, findings, models, False)
        xa_train = design_matrix(train, standardization, findings, models, True)
        xb_held = design_matrix(held, standardization, findings, models, False)
        xa_held = design_matrix(held, standardization, findings, models, True)
        baseline_width = xb_train.shape[1]
        zi_train, zi_held = xa_train[:, baseline_width:], xa_held[:, baseline_width:]
        baseline_model = fit_logistic(xb_train, y[train_idx], c_value)
        augmented_model = fit_pattern_increment(
            zi_train, y[train_idx], linear_predictor(baseline_model, xb_train), c_value
        )
        baseline[held_idx] = predict_logistic(baseline_model, xb_held)
        augmented[held_idx] = predict_with_increment(
            baseline_model, augmented_model, xb_held, zi_held
        )
        audits.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "held_rows": len(held_idx),
            "train_images": len({row["image_id"] for row in train}),
            "held_images": len({row["image_id"] for row in held}),
        })
    if not np.isfinite(baseline).all() or not np.isfinite(augmented).all():
        raise AliasingError("cross-fit predictions incomplete")
    return baseline, augmented, audits


def reader_agreement_order(
    rows: Sequence[Mapping[str, Any]], *, baseline_probability: Sequence[float] | None = None,
    model: str | None = None, finding: str | None = None
) -> dict[str, Any]:
    if baseline_probability is None:
        baseline_probability = np.full(len(rows), .5)
    if len(baseline_probability) != len(rows):
        raise AliasingError("reader ordering probabilities have unequal length")
    eligible = [
        (row, float(probability)) for row, probability in zip(rows, baseline_probability)
        if row["positive_votes"] in (1, 2)
        and (model is None or row["model"] == model)
        and (finding is None or row["finding"] == finding)
    ]
    if not eligible:
        return {"status": "insufficient", "n": 0, "ordering": []}
    scores = {
        reader: float(np.mean([
            (int(row["target"]) - probability) * (2 * int(row["votes"][index]) - 1)
            for row, probability in eligible
        ]))
        for index, reader in enumerate(READERS)
    }
    ordering = sorted(READERS, key=lambda reader: (-scores[reader], reader))
    tied = any(abs(scores[ordering[i]] - scores[ordering[i + 1]]) < 1e-12 for i in range(2))
    return {
        "status": "tied" if tied else "ordered", "n": len(eligible),
        "baseline_adjusted_alignment": scores, "ordering": ordering,
    }


def fit_dev(
    rows: Sequence[Mapping[str, Any]], *, input_sha256: str, folds: int = 5,
    bootstrap_draws: int = 5000, seed: int = 20260803, c_value: float = 1.0,
    findings: Sequence[str] = PRIMARY_FINDINGS, models: Sequence[str] = MODELS,
) -> dict[str, Any]:
    if {row["stage"] for row in rows} != {"dev_fit"}:
        raise AliasingError("fit_dev accepts only dev_fit")
    baseline_oof, augmented_oof, fold_audit = crossfit(
        rows, folds=folds, seed=seed, c_value=c_value, findings=findings, models=models
    )
    standardization = fit_standardization(rows)
    y = np.asarray([row["target"] for row in rows], dtype=int)
    baseline_model = fit_logistic(
        design_matrix(rows, standardization, findings, models, False), y, c_value
    )
    augmented_design = design_matrix(rows, standardization, findings, models, True)
    baseline_design = design_matrix(rows, standardization, findings, models, False)
    baseline_width = baseline_design.shape[1]
    augmented_model = fit_pattern_increment(
        augmented_design[:, baseline_width:], y,
        linear_predictor(baseline_model, baseline_design), c_value,
    )
    ordering = (
        reader_agreement_order(rows, baseline_probability=baseline_oof)
        if rows[0]["endpoint"] == "positive_commitment"
        else {"status": "not_applicable_for_clinical_error", "n": 0, "ordering": []}
    )
    body: dict[str, Any] = {
        "schema_version": FIT_SCHEMA,
        "implementation_version": VERSION,
        "status": "dev_fit_complete_confirmation_not_read",
        "endpoint": rows[0]["endpoint"],
        "dev_input_sha256": input_sha256,
        "dev_image_id_sha256s": sorted({
            hashlib.sha256(str(row["image_id"]).encode()).hexdigest() for row in rows
        }),
        "source_sha256": sha256_file(Path(__file__)),
        "frozen_contract": {
            "reader_panel": list(READERS), "findings": list(findings), "models": list(models),
            "baseline": "vote_count+saturated_model_by_finding_intercepts+clean_z_by_model_by_finding_slopes",
            "increment": "frozen_baseline_logit_plus_exact_R8_R9_R10_pattern_residual_by_model_by_finding",
            "fold_unit": "whole_image_id", "folds": folds, "seed": seed,
            "regularization_C": c_value, "confirmation_refit": False,
            "thresholds": {"delta_auroc": .05, "relative_nll_improvement": .05, "ci_excludes_zero": True},
            "ordering_requirement": "dev_order_exactly_repeats_in_at_least_6_of_8_findings_for_each_model",
            "clear_case_identity_increment": "structurally_nonidentifiable_on_000_and_111",
            "future_clear_case_requirement": "separately_preregistered_independent_predictor_or_intervention_not_implemented_here",
        },
        "standardization": standardization,
        "feature_names": {
            "baseline": feature_names(findings, models, False),
            "augmented": feature_names(findings, models, True),
        },
        "models_fit": {"baseline": baseline_model, "pattern_increment": augmented_model},
        "dev_oof": cluster_bootstrap(
            rows, baseline_oof, augmented_oof, draws=bootstrap_draws, seed=seed + 1
        ),
        "dev_reader_order": ordering,
        "fold_audit": fold_audit,
        "confirmation_consumed": False,
        "paper_claim_authorized": False,
        "mitigation_authorized": False,
    }
    body["fingerprint"] = canonical_sha256(body)
    return body


def _validate_fit(fit: Mapping[str, Any]) -> None:
    fingerprint = fit.get("fingerprint")
    body = {key: value for key, value in fit.items() if key != "fingerprint"}
    if (
        fit.get("schema_version") != FIT_SCHEMA
        or fit.get("implementation_version") != VERSION
        or fit.get("status") != "dev_fit_complete_confirmation_not_read"
        or fit.get("confirmation_consumed") is not False
        or fit.get("paper_claim_authorized") is not False
        or fit.get("mitigation_authorized") is not False
        or fingerprint != canonical_sha256(body)
    ):
        raise AliasingError("invalid or drifted development fit")
    if fit.get("source_sha256") != sha256_file(Path(__file__)):
        raise AliasingError("analyzer source changed after development fit")


def confirm(
    fit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *,
    input_sha256: str, bootstrap_draws: int = 5000, seed: int = 20260803,
) -> dict[str, Any]:
    _validate_fit(fit)
    if {row["stage"] for row in rows} != {"confirmation_locked"}:
        raise AliasingError("confirm accepts only confirmation_locked")
    if {row["endpoint"] for row in rows} != {fit["endpoint"]}:
        raise AliasingError("confirmation endpoint differs from frozen development endpoint")
    dev_images = set(fit.get("dev_image_id_sha256s", []))
    confirmation_images = {
        hashlib.sha256(str(row["image_id"]).encode()).hexdigest() for row in rows
    }
    if dev_images & confirmation_images:
        raise AliasingError("development/confirmation whole-image overlap")
    contract = fit["frozen_contract"]
    findings, models = tuple(contract["findings"]), tuple(contract["models"])
    standardization = fit["standardization"]
    baseline_design = design_matrix(rows, standardization, findings, models, False)
    augmented_design = design_matrix(rows, standardization, findings, models, True)
    pattern_design = augmented_design[:, baseline_design.shape[1]:]
    baseline = predict_logistic(fit["models_fit"]["baseline"], baseline_design)
    augmented = predict_with_increment(
        fit["models_fit"]["baseline"], fit["models_fit"]["pattern_increment"],
        baseline_design, pattern_design,
    )
    comparison = cluster_bootstrap(
        rows, baseline, augmented, draws=bootstrap_draws, seed=seed + 2
    )
    pooled_identity_gate = bool(
        comparison["delta_auroc"]["estimate"] >= .05
        and comparison["delta_auroc"]["ci_low"] > 0
        and comparison["relative_nll_improvement"]["estimate"] >= .05
        and comparison["relative_nll_improvement"]["ci_low"] > 0
    )
    endpoint_supports_order = fit["endpoint"] == "positive_commitment"
    frozen_order = fit["dev_reader_order"].get("ordering", []) if endpoint_supports_order else []
    ordering_cells: dict[str, Any] = {}
    stable_counts: dict[str, int] = {}
    for model in models:
        stable = 0
        for finding in findings:
            cell_indices = np.asarray([
                index for index, row in enumerate(rows)
                if row["model"] == model and row["finding"] == finding
            ], dtype=int)
            cell_rows = [rows[index] for index in cell_indices]
            value = (
                reader_agreement_order(
                    cell_rows, baseline_probability=baseline[cell_indices],
                    model=model, finding=finding,
                )
                if endpoint_supports_order
                else {"status": "not_applicable_for_clinical_error", "n": 0, "ordering": []}
            )
            value["matches_frozen_dev_order"] = bool(
                value["status"] == "ordered" and value["ordering"] == frozen_order
            )
            stable += int(value["matches_frozen_dev_order"])
            ordering_cells[f"{model}|{finding}"] = value
        stable_counts[model] = stable
    ordering_gate = bool(
        endpoint_supports_order
        and
        len(frozen_order) == 3 and all(stable_counts[model] >= 6 for model in models)
    )
    clear_rows = [row for row in rows if row["positive_votes"] in (0, 3)]
    clear_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["positive_votes"] in (0, 3)], dtype=int
    )
    if len(clear_rows):
        if not np.all(pattern_design[clear_indices] == 0):
            raise AliasingError("unanimous patterns unexpectedly carry identity features")
        maximum_difference = float(np.max(np.abs(baseline[clear_indices] - augmented[clear_indices])))
        if maximum_difference > 1e-12:
            raise AliasingError("frozen-baseline identity increment changed a unanimous case")
    else:
        maximum_difference = 0.0
    clear_comparison = {
        "status": "structurally_nonidentifiable_from_exact_reader_pattern",
        "clear_case_identity_increment_defined": False,
        "reason": "R8/R9/R10 are identical on 000 and 111; all pattern-residual features are exactly zero",
        "n": len(clear_rows),
        "maximum_absolute_prediction_difference": maximum_difference,
        "delta_auroc": 0.0,
        "relative_nll_improvement": 0.0,
        "future_reopen_requires": "separately_preregistered_independent_clear_case_predictor_or_intervention_not_implemented_here",
    }
    clear_gate = False
    both_models = {}
    both_model_gates: dict[str, bool] = {}
    for model in models:
        selected = np.asarray([index for index, row in enumerate(rows) if row["model"] == model])
        model_rows = [rows[index] for index in selected]
        if len({row["target"] for row in model_rows}) == 2:
            both_models[model] = cluster_bootstrap(
                model_rows, baseline[selected], augmented[selected],
                draws=bootstrap_draws, seed=seed + 100 + len(both_models),
            )
            value = both_models[model]
            both_model_gates[model] = bool(
                value["delta_auroc"]["estimate"] >= .05
                and value["delta_auroc"]["ci_low"] > 0
                and value["relative_nll_improvement"]["estimate"] >= .05
                and value["relative_nll_improvement"]["ci_low"] > 0
            )
        else:
            both_models[model] = {"status": "insufficient_endpoint_classes"}
            both_model_gates[model] = False
    both_models_identity_gate = all(both_model_gates.get(model, False) for model in models)
    identity_gate = pooled_identity_gate and both_models_identity_gate
    disagreement_semantics = identity_gate and ordering_gate
    if fit["endpoint"] == "clinical_error":
        classification = (
            "clinical_error_pattern_predictive_control_only"
            if identity_gate else "reader_threshold_aliasing_not_supported"
        )
    else:
        classification = (
            "reader_disagreement_semantics_only"
            if disagreement_semantics else "reader_threshold_aliasing_not_supported"
        )
    output: dict[str, Any] = {
        "schema_version": CONFIRM_SCHEMA,
        "implementation_version": VERSION,
        "status": "confirmation_applied_once_no_refit",
        "fit_fingerprint": fit["fingerprint"],
        "confirmation_input_sha256": input_sha256,
        "endpoint": fit["endpoint"],
        "comparison": comparison,
        "by_model": both_models,
        "reader_order": {
            "frozen_dev": frozen_order, "confirmation_cells": ordering_cells,
            "stable_finding_count_by_model": stable_counts,
        },
        "clear_case_increment": clear_comparison,
        "gates": {
            "pooled_identity_increment": pooled_identity_gate,
            "both_models_identity_increment": both_models_identity_gate,
            "identity_increment": identity_gate,
            "ordering_6_of_8_each_model": ordering_gate,
            "clear_case_identity_increment_defined": False,
            "clear_case_beyond_finding_specific_sensitivity": clear_gate,
        },
        "classification": classification,
        "claim_boundary": (
            "This observational control cannot establish a causal hallucination mechanism, "
            "authorize mitigation, or alter the CECD primary gate."
        ),
        "confirmation_refit": False,
        "paper_claim_authorized": False,
        "mitigation_authorized": False,
        "power_caveat": "Failure is indeterminate when pattern cells or endpoint events are too sparse; thresholds may not be relaxed post hoc.",
    }
    output["fingerprint"] = canonical_sha256(output)
    return output


def _main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit-dev")
    fit_parser.add_argument("--input", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--folds", type=int, default=5)
    fit_parser.add_argument("--draws", type=int, default=5000)
    fit_parser.add_argument("--seed", type=int, default=20260803)
    confirm_parser = subparsers.add_parser("confirm")
    confirm_parser.add_argument("--fit", type=Path, required=True)
    confirm_parser.add_argument("--input", type=Path, required=True)
    confirm_parser.add_argument("--output", type=Path, required=True)
    confirm_parser.add_argument("--draws", type=int, default=5000)
    confirm_parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if args.command == "fit-dev":
        rows = load_rows(args.input, "dev_fit")
        result = fit_dev(
            rows, input_sha256=sha256_file(args.input), folds=args.folds,
            bootstrap_draws=args.draws, seed=args.seed,
        )
    else:
        fit = json.loads(args.fit.read_text(encoding="utf-8"))
        rows = load_rows(args.input, "confirmation_locked")
        result = confirm(
            fit, rows, input_sha256=sha256_file(args.input),
            bootstrap_draws=args.draws, seed=args.seed,
        )
    write_once_json(args.output, result)


if __name__ == "__main__":
    _main()
