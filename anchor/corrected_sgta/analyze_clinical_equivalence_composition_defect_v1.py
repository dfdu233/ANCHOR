#!/usr/bin/env python3
"""Fail-closed analysis of a Clinical-Equivalence Composition Defect (CECD).

The estimand is a two-way centered render-by-prompt interaction (a discrete
mixed derivative).  It is deliberately *not* called a commutator: the
factorial does not evaluate RP-PR or an algebraic order effect.

The analyzer is runner-independent.  Its compact JSON contract is documented
by ``validate_payload`` and emitted in every result.  Clear reader-vote cases
(0/3 and 3/3) define polarity error.  Disagreement cases (1/3 and 2/3) are
reported separately as reader-support-gap and commitment diagnostics; the
Maybe verbalizer is never treated as clinical truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


VERSION = "clinical-equivalence-composition-defect-analysis-v1.4-recomputable"
CONTRACT_VERSION = "clinical-equivalence-factorial-v1"
PRIMARY_GATE_NAME = "behavioral_dev_authorization_only"
DEV_FIT_VERSION = "clinical-equivalence-composition-defect-dev-fit-v2-recomputable"
CONFIRMATION_VERSION = (
    "clinical-equivalence-composition-defect-confirmation-locked-v2-recomputable"
)
TRISTATE_STATES = ("supported", "refuted", "undetermined")


class ContractError(ValueError):
    """The factorial cannot support the frozen analysis."""


def _one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def tristate_entropy_from_logits(logits: Mapping[str, Any]) -> float:
    values = np.asarray([finite(value, "tristate_logit") for value in logits.values()], dtype=float)
    if values.size != 3:
        raise ContractError("tristate logits must contain exactly three values")
    probability = np.exp(values - values.max())
    probability /= probability.sum()
    return float(-np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0))))


def two_way_centered(matrix: np.ndarray) -> np.ndarray:
    """Return M - rowmean - colmean + grandmean for a complete 2-D grid."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ContractError("two-way centering needs a finite >=2 by >=2 grid")
    return values - values.mean(axis=1, keepdims=True) - values.mean(
        axis=0, keepdims=True
    ) + values.mean()


def adjacent_reader_slope(votes: Sequence[int], scores: Sequence[float]) -> float:
    """Median of the three adjacent clean-score bin differences."""

    vote = np.asarray(votes, dtype=int)
    score = np.asarray(scores, dtype=np.float64)
    means = []
    for value in range(4):
        selected = score[vote == value]
        if selected.size == 0:
            raise ContractError("each reader-vote bin is required for a slope")
        means.append(float(selected.mean()))
    return float(np.median(np.diff(means)))


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["model"]),
        str(row["image_id"]),
        str(row["finding"]),
        str(row["render_id"]),
        str(row["prompt_id"]),
    )


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the runner-independent factorial contract."""

    if payload.get("schema_version") != CONTRACT_VERSION:
        raise ContractError(f"schema_version must be {CONTRACT_VERSION!r}")
    split = str(payload.get("split", ""))
    if split not in {"dev", "pilot_screen", "dev_fit", "confirmation_locked"}:
        raise ContractError("unknown CECD stage label")
    if payload.get("frozen_before_outputs") is not True:
        raise ContractError("factorial transforms/prompts must be frozen before outputs")
    if payload.get("score_definition") != "fp32_yes_minus_no_logit":
        raise ContractError("score_definition must be fp32_yes_minus_no_logit")

    primary_renders = tuple(str(value) for value in payload.get("primary_renders", ()))
    primary_prompts = tuple(str(value) for value in payload.get("primary_prompts", ()))
    baseline_render = str(payload.get("baseline_render", ""))
    baseline_prompt = str(payload.get("baseline_prompt", ""))
    identity_render = str(payload.get("identity_render", ""))
    duplicate_prompt = str(payload.get("duplicate_prompt", ""))
    if len(set(primary_renders)) < 2 or len(set(primary_prompts)) < 2:
        raise ContractError("at least two unique primary renders and prompts are required")
    if baseline_render not in primary_renders or baseline_prompt not in primary_prompts:
        raise ContractError("baseline render/prompt must be primary")
    if not identity_render or identity_render in primary_renders:
        raise ContractError("a non-primary identity_render control is required")
    if not duplicate_prompt or duplicate_prompt in primary_prompts:
        raise ContractError("a non-primary duplicate_prompt control is required")

    raw_rows = payload.get("records")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ContractError("records must be a non-empty list")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    external_folds: dict[tuple[str, str], str] = {}
    excluded_orbits: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ContractError("every record must be an object")
        missing = {
            "model", "image_id", "finding", "reader_votes", "render_id",
            "prompt_id", "signed_score", "commitment_score",
            "acquisition_view", "tristate_entropy", "tristate_logits",
            "input_prompt_length_tokens", "answer_length_tokens",
        } - set(raw)
        if missing:
            raise ContractError(f"record lacks required keys: {sorted(missing)}")
        row = dict(raw)
        row["model"] = str(row["model"])
        row["image_id"] = str(row["image_id"])
        row["finding"] = str(row["finding"])
        row["render_id"] = str(row["render_id"])
        row["prompt_id"] = str(row["prompt_id"])
        row["acquisition_view"] = str(row["acquisition_view"])
        if not row["acquisition_view"]:
            raise ContractError("acquisition_view must be a non-empty normalized category")
        row["reader_votes"] = int(row["reader_votes"])
        if row["reader_votes"] not in (0, 1, 2, 3):
            raise ContractError("reader_votes must lie in 0..3")
        orbit_key = (row["model"], row["image_id"], row["finding"])
        if row.get("valid", True) is not True:
            reasons = row.get(
                "exclusion_reasons",
                row.get("invalid_reasons", row.get("missing_reasons", ())),
            )
            if isinstance(reasons, str):
                reasons = [reasons]
            excluded_orbits[orbit_key].extend(str(value) for value in reasons)
            if not excluded_orbits[orbit_key]:
                excluded_orbits[orbit_key].append("invalid_required_factorial_cell")
            continue
        row["signed_score"] = finite(row["signed_score"], "signed_score")
        row["commitment_score"] = finite(row["commitment_score"], "commitment_score")
        row["tristate_entropy"] = finite(row["tristate_entropy"], "tristate_entropy")
        logits = row["tristate_logits"]
        if not isinstance(logits, Mapping) or set(logits) != set(TRISTATE_STATES):
            raise ContractError(
                "tristate_logits must contain exactly supported/refuted/undetermined"
            )
        row["tristate_logits"] = {
            state: finite(logits[state], f"tristate_logits.{state}")
            for state in TRISTATE_STATES
        }
        expected_signed = (
            row["tristate_logits"]["supported"]
            - row["tristate_logits"]["refuted"]
        )
        expected_commitment = (
            max(
                row["tristate_logits"]["supported"],
                row["tristate_logits"]["refuted"],
            )
            - row["tristate_logits"]["undetermined"]
        )
        expected_entropy = tristate_entropy_from_logits(row["tristate_logits"])
        if not math.isclose(row["signed_score"], expected_signed, abs_tol=1e-5):
            raise ContractError("signed_score disagrees with the bound tristate logits")
        if not math.isclose(
            row["commitment_score"], expected_commitment, abs_tol=1e-5
        ):
            raise ContractError("commitment_score disagrees with the bound tristate logits")
        if not math.isclose(row["tristate_entropy"], expected_entropy, abs_tol=1e-6):
            raise ContractError("tristate_entropy disagrees with the bound tristate logits")
        prompt_length = row["input_prompt_length_tokens"]
        answer_length = row["answer_length_tokens"]
        if (
            isinstance(prompt_length, bool)
            or not isinstance(prompt_length, int)
            or prompt_length <= 0
        ):
            raise ContractError("input_prompt_length_tokens must be a positive integer")
        if answer_length != 1 or isinstance(answer_length, bool):
            raise ContractError(
                "answer_length_tokens must equal one under the frozen next-token task"
            )
        if row.get("crossmodal_direct_effect_scalar_surrogate") is not None:
            row["crossmodal_direct_effect_scalar_surrogate"] = finite(
                row["crossmodal_direct_effect_scalar_surrogate"],
                "crossmodal_direct_effect_scalar_surrogate",
            )
        key = _row_key(row)
        if key in seen:
            raise ContractError(f"duplicate factorial cell: {key}")
        seen.add(key)
        if "fold_id" in row:
            group_key = (row["model"], row["image_id"])
            fold = str(row["fold_id"])
            previous = external_folds.setdefault(group_key, fold)
            if previous != fold:
                raise ContractError("external fold_id leaks one image across folds")
        rows.append(row)

    # A two-way centered interaction requires a complete rectangle.  One
    # invalid required cell excludes its entire image-claim orbit; no baseline
    # substitution and no cellwise deletion are permitted.
    rows = [
        row for row in rows
        if (row["model"], row["image_id"], row["finding"]) not in excluded_orbits
    ]

    # Runner contract: complete science grid, one identity-image row across all
    # primary prompts, and one duplicate baseline prompt at the baseline render.
    # Requiring the full cross of controls would add calls with no estimand.
    required = {
        (render, prompt) for render in primary_renders for prompt in primary_prompts
    } | {
        (identity_render, prompt) for prompt in primary_prompts
    } | {(baseline_render, duplicate_prompt)}
    by_orbit: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_orbit[(row["model"], row["image_id"], row["finding"])].append(row)
    for key, orbit_rows in by_orbit.items():
        vote_values = {row["reader_votes"] for row in orbit_rows}
        if len(vote_values) != 1:
            raise ContractError(f"reader vote changes inside orbit {key}")
        acquisition_views = {row["acquisition_view"] for row in orbit_rows}
        if len(acquisition_views) != 1:
            raise ContractError(f"acquisition_view changes inside orbit {key}")
        available = {(row["render_id"], row["prompt_id"]) for row in orbit_rows}
        absent = required - available
        if absent:
            raise ContractError(f"incomplete factorial orbit {key}: missing {sorted(absent)}")
    if not by_orbit:
        raise ContractError("no complete factorial orbits")
    return {
        "split": split,
        "source_manifest_split": str(payload.get("source_manifest_split", "unknown")),
        "rows": rows,
        "by_orbit": by_orbit,
        "primary_renders": primary_renders,
        "primary_prompts": primary_prompts,
        "baseline_render": baseline_render,
        "baseline_prompt": baseline_prompt,
        "identity_render": identity_render,
        "duplicate_prompt": duplicate_prompt,
        "excluded_orbits": [
            {
                "model": key[0], "image_id": key[1], "finding": key[2],
                "reasons": sorted(set(reasons)),
                "policy": "whole_orbit_excluded_no_imputation",
            }
            for key, reasons in sorted(excluded_orbits.items())
        ],
    }


def runner_rows_to_payload(raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Adapt one or more runner ``factorial_rows.jsonl`` files to the contract."""

    if not raw_rows:
        raise ContractError("runner JSONL is empty")
    if any(row.get("contract_version") != CONTRACT_VERSION for row in raw_rows):
        raise ContractError("runner row contract_version mismatch")
    science = [row for row in raw_rows if row.get("cell_role") == "science_factorial"]
    identity = [row for row in raw_rows if row.get("cell_role") == "identity_image_control"]
    duplicates = [
        row for row in raw_rows
        if row.get("cell_role") == "exact_duplicate_prompt_control"
    ]
    if not science or not identity or not duplicates:
        raise ContractError("runner JSONL lacks science or identity controls")
    primary_renders = list(dict.fromkeys(str(row["render_id"]) for row in science))
    primary_prompts = list(dict.fromkeys(str(row["prompt_id"]) for row in science))
    identity_renders = {str(row["render_id"]) for row in identity}
    duplicate_prompts = {str(row["prompt_id"]) for row in duplicates}
    baseline_renders = {str(row["render_id"]) for row in duplicates}
    if len(identity_renders) != 1 or len(duplicate_prompts) != 1 or len(baseline_renders) != 1:
        raise ContractError("runner controls do not identify unique baseline/duplicate names")
    baseline_render = next(iter(baseline_renders))
    reference_ids = {str(row.get("reference_cell_id")) for row in duplicates}
    referenced_prompts = {
        str(row["prompt_id"]) for row in science if str(row.get("cell_id")) in reference_ids
    }
    if len(referenced_prompts) != 1:
        raise ContractError("duplicate prompt does not resolve to one baseline science prompt")
    baseline_prompt = next(iter(referenced_prompts))

    stage_labels = {str(row.get("stage_label", "dev")) for row in raw_rows}
    source_splits = {
        str(row.get("source_manifest_split", "pilot")) for row in raw_rows
    }
    if len(stage_labels) != 1 or len(source_splits) != 1:
        raise ContractError("runner rows mix stage labels or manifest splits")
    stage_label = next(iter(stage_labels))
    source_split = next(iter(source_splits))
    normalized = []
    for row in raw_rows:
        entropy = row.get("tristate_entropy")
        if entropy is None and isinstance(row.get("tristate_logits"), Mapping):
            entropy = tristate_entropy_from_logits(row["tristate_logits"])
        normalized.append(
            {
                "model": str(row["model"]),
                "image_id": str(row["image_id"]),
                "finding": str(row["finding"]),
                "reader_votes": int(row["positive_votes"]),
                "render_id": str(row["render_id"]),
                "prompt_id": str(row["prompt_id"]),
                "signed_score": row.get("signed_score"),
                "commitment_score": row.get("commitment_score"),
                "acquisition_view": str(row.get("acquisition_view", "unknown")),
                "tristate_entropy": entropy,
                "tristate_logits": row.get("tristate_logits"),
                "input_prompt_length_tokens": row.get("raw_prompt_token_count"),
                "answer_length_tokens": row.get("answer_length_tokens"),
                "valid": row.get("status") == "ok",
                "exclusion_reasons": row.get("missing_reasons", ()),
            }
        )
    return {
        "schema_version": CONTRACT_VERSION,
        "split": stage_label,
        "source_manifest_split": source_split,
        "frozen_before_outputs": True,
        "score_definition": "fp32_yes_minus_no_logit",
        "primary_renders": primary_renders,
        "primary_prompts": primary_prompts,
        "baseline_render": baseline_render,
        "baseline_prompt": baseline_prompt,
        "identity_render": next(iter(identity_renders)),
        "duplicate_prompt": next(iter(duplicate_prompts)),
        "records": normalized,
    }


def load_inputs(paths: Sequence[Path]) -> dict[str, Any]:
    """Load either the native wrapper JSON or runner JSONL file(s)."""

    if not paths:
        raise ContractError("at least one input is required")
    if len(paths) == 1:
        text = paths[0].read_text(encoding="utf-8")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping) and parsed.get("schema_version") == CONTRACT_VERSION:
            return dict(parsed)
    rows: list[Mapping[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, Mapping):
                    raise ContractError("every runner JSONL line must be an object")
                rows.append(item)
    return runner_rows_to_payload(rows)


def _cell_map(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["render_id"]), str(row["prompt_id"])): row for row in rows}


def build_orbits(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    renders = contract["primary_renders"]
    prompts = contract["primary_prompts"]
    output = []
    for (model, image_id, finding), raw_rows in sorted(contract["by_orbit"].items()):
        cells = _cell_map(raw_rows)
        vote = int(raw_rows[0]["reader_votes"])
        score = np.asarray(
            [[cells[(render, prompt)]["signed_score"] for prompt in prompts] for render in renders],
            dtype=np.float64,
        )
        commitment = np.asarray(
            [[cells[(render, prompt)]["commitment_score"] for prompt in prompts] for render in renders],
            dtype=np.float64,
        )
        entropy = np.asarray(
            [
                [cells[(render, prompt)]["tristate_entropy"] for prompt in prompts]
                for render in renders
            ],
            dtype=np.float64,
        )
        logits = np.asarray(
            [
                [
                    [
                        cells[(render, prompt)]["tristate_logits"][state]
                        for state in TRISTATE_STATES
                    ]
                    for prompt in prompts
                ]
                for render in renders
            ],
            dtype=np.float64,
        )
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        surrogate_available = all(
            cells[(render, prompt)].get(
                "crossmodal_direct_effect_scalar_surrogate"
            ) is not None
            for render in renders for prompt in prompts
        )
        surrogate = None
        if surrogate_available:
            surrogate = np.asarray(
                [[cells[(render, prompt)]["crossmodal_direct_effect_scalar_surrogate"] for prompt in prompts] for render in renders],
                dtype=np.float64,
            )
        output.append(
            {
                "model": model,
                "image_id": image_id,
                "finding": finding,
                "reader_votes": vote,
                "cells": cells,
                "score": score,
                "interaction": two_way_centered(score),
                "commitment": commitment,
                "commitment_interaction": two_way_centered(commitment),
                "entropy": entropy,
                "probabilities": probabilities,
                "crossmodal_direct_effect_scalar_surrogate": surrogate,
            }
        )
    return output


def _kl_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """KL(left || right) over the last axis with deterministic clipping."""

    p = np.clip(np.asarray(left, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(right, dtype=np.float64), 1e-12, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def behavioral_pid_mmi(probabilities: np.ndarray) -> dict[str, Any]:
    """Behavior-level MMI PID-style control over the uniform product orbit.

    This is deliberately a *control*, not representation-level PID.  Render
    and prompt are uniform independent sources and the model's three-state
    probability vector defines the stochastic target.  MMI redundancy yields
    ``synergy = I(R,P;Y) - max(I(R;Y), I(P;Y))``.  Per-cell local excess is
    retained so the control is not merely an orbit-constant scalar.
    """

    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.ndim != 3 or probability.shape[-1] != 3:
        raise ContractError("behavioral PID requires a render x prompt x 3 tensor")
    if not np.isfinite(probability).all() or np.any(probability <= 0):
        raise ContractError("behavioral PID probabilities must be finite and positive")
    joint_mean = probability.mean(axis=(0, 1))
    render_mean = probability.mean(axis=1)
    prompt_mean = probability.mean(axis=0)
    joint_local = _kl_rows(probability, joint_mean)
    render_local = _kl_rows(render_mean, joint_mean)
    prompt_local = _kl_rows(prompt_mean, joint_mean)
    information_joint = float(joint_local.mean())
    information_render = float(render_local.mean())
    information_prompt = float(prompt_local.mean())
    synergy = max(
        0.0,
        information_joint - max(information_render, information_prompt),
    )
    local_excess = joint_local - np.maximum(
        render_local[:, None], prompt_local[None, :]
    )
    return {
        "information_render_prompt_target_nats": information_joint,
        "information_render_target_nats": information_render,
        "information_prompt_target_nats": information_prompt,
        "mmi_synergy_nats": float(synergy),
        "local_synergy_excess_nats": local_excess,
        "guardrail": (
            "behavioral output-distribution MMI PID-style control only; not hidden-state "
            "PID, causal synergy, or a substitute for the post-Stage-1 mechanism control"
        ),
    }


def grouped_folds(orbits: Sequence[Mapping[str, Any]], folds: int, seed: int) -> np.ndarray:
    if folds < 2:
        raise ContractError("grouped CV requires at least two folds")
    groups = np.asarray([row["image_id"] for row in orbits], dtype=object)
    strata = np.asarray(
        [f"{row['finding']}|{row['reader_votes']}" for row in orbits], dtype=object
    )
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignment = np.full(len(orbits), -1, dtype=int)
    for fold, (train, test) in enumerate(splitter.split(strata, strata, groups)):
        if set(groups[train]).intersection(groups[test]):
            raise RuntimeError("image group leaked across internally generated folds")
        assignment[test] = fold
    if (assignment < 0).any():
        raise RuntimeError("grouped fold assignment is incomplete")
    by_group: dict[str, set[int]] = defaultdict(set)
    for group, fold in zip(groups, assignment):
        by_group[str(group)].add(int(fold))
    if any(len(values) != 1 for values in by_group.values()):
        raise RuntimeError("one image was assigned to multiple folds")
    return assignment


def _clean_score(orbit: Mapping[str, Any], contract: Mapping[str, Any]) -> float:
    return float(
        orbit["cells"][(contract["baseline_render"], contract["baseline_prompt"])][
            "signed_score"
        ]
    )


def crossfit_reader_scale(
    orbits: Sequence[Mapping[str, Any]], fold_ids: np.ndarray, contract: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Cross-fit finding slopes and clean-score centers for every orbit."""

    scales = np.full(len(orbits), np.nan, dtype=np.float64)
    centers = np.full(len(orbits), np.nan, dtype=np.float64)
    audit: list[dict[str, Any]] = []
    for fold in sorted(np.unique(fold_ids)):
        train = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        strata = sorted({(str(orbits[index]["model"]), str(orbits[index]["finding"])) for index in test})
        for model, finding in strata:
            fit_indices = [
                index for index in train
                if orbits[index]["model"] == model and orbits[index]["finding"] == finding
            ]
            apply_indices = [
                index for index in test
                if orbits[index]["model"] == model and orbits[index]["finding"] == finding
            ]
            votes = [int(orbits[index]["reader_votes"]) for index in fit_indices]
            scores = [_clean_score(orbits[index], contract) for index in fit_indices]
            slope = adjacent_reader_slope(votes, scores)
            if slope <= 1e-8:
                raise ContractError(
                    f"non-positive directional reader scale for {model}/{finding}/fold{fold}"
                )
            center = float(np.mean(scores))
            for index in apply_indices:
                scales[index] = slope
                centers[index] = center
            audit.append(
                {"fold": int(fold), "model": model, "finding": finding, "slope": slope, "train_n": len(scores)}
            )
    if not np.isfinite(scales).all() or not np.isfinite(centers).all():
        raise RuntimeError("cross-fitted reader scales are incomplete")
    return scales, centers, audit


def slope_cluster_bootstrap(
    orbits: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], draws: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for model in sorted({str(row["model"]) for row in orbits}):
        result[model] = {}
        for finding in sorted({str(row["finding"]) for row in orbits if row["model"] == model}):
            selected = [row for row in orbits if row["model"] == model and row["finding"] == finding]
            point = adjacent_reader_slope(
                [row["reader_votes"] for row in selected],
                [_clean_score(row, contract) for row in selected],
            )
            groups = sorted({str(row["image_id"]) for row in selected})
            by_group = {group: [row for row in selected if row["image_id"] == group] for group in groups}
            values = []
            for _ in range(draws):
                sampled = rng.choice(groups, len(groups), replace=True)
                rows = [row for group in sampled for row in by_group[str(group)]]
                try:
                    values.append(
                        adjacent_reader_slope(
                            [row["reader_votes"] for row in rows],
                            [_clean_score(row, contract) for row in rows],
                        )
                    )
                except ContractError:
                    continue
            result[model][finding] = {
                "point": point,
                "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                if values else None,
                "valid_draws": len(values),
                "positive_directional_scale": bool(point > 0),
            }
    return result


def make_cell_table(
    orbits: Sequence[Mapping[str, Any]],
    scales: np.ndarray,
    centers: np.ndarray,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    renders = contract["primary_renders"]
    prompts = contract["primary_prompts"]
    render_index = {value: index for index, value in enumerate(renders)}
    prompt_index = {value: index for index, value in enumerate(prompts)}
    rows: list[dict[str, Any]] = []
    for orbit_index, orbit in enumerate(orbits):
        beta = float(scales[orbit_index])
        if beta <= 1e-8:
            continue
        score = orbit["score"]
        interaction = orbit["interaction"]
        entropy = orbit["entropy"]
        probabilities = orbit["probabilities"]
        pid = behavioral_pid_mmi(probabilities)
        row_main = score.mean(axis=1) - score.mean()
        col_main = score.mean(axis=0) - score.mean()
        clean = _clean_score(orbit, contract)
        render_rms = float(np.sqrt(np.mean(row_main**2)) / beta)
        prompt_rms = float(np.sqrt(np.mean(col_main**2)) / beta)
        interaction_rms = float(np.sqrt(np.mean(interaction**2)) / beta)
        reader_fraction = float(orbit["reader_votes"] / 3.0)
        for render in renders:
            ri = render_index[render]
            for prompt in prompts:
                pi = prompt_index[prompt]
                cell_score = float(score[ri, pi])
                cell_interaction = float(interaction[ri, pi])
                orbit_probability_mean = probabilities.mean(axis=(0, 1))
                render_slice = probabilities[:, pi, :]
                prompt_slice = probabilities[ri, :, :]
                render_probability_mean = render_slice.mean(axis=0)
                prompt_probability_mean = prompt_slice.mean(axis=0)
                additive_score = cell_score - cell_interaction
                # Training-only clean center plus the finding-specific adjacent
                # reader scale maps one score step to 1/3 reader support.
                support_actual = float(
                    np.clip(0.5 + (cell_score - centers[orbit_index]) / (3.0 * beta), 0, 1)
                )
                support_additive = float(
                    np.clip(0.5 + (additive_score - centers[orbit_index]) / (3.0 * beta), 0, 1)
                )
                clear = int(orbit["reader_votes"]) in (0, 3)
                truth_sign = 1.0 if int(orbit["reader_votes"]) == 3 else -1.0
                harmful_sign = -truth_sign
                surrogate_value = None
                if orbit["crossmodal_direct_effect_scalar_surrogate"] is not None:
                    surrogate_value = float(
                        orbit["crossmodal_direct_effect_scalar_surrogate"][ri, pi]
                    ) * harmful_sign / beta
                rows.append(
                    {
                        "model": orbit["model"],
                        "image_id": orbit["image_id"],
                        "finding": orbit["finding"],
                        "reader_votes": int(orbit["reader_votes"]),
                        "render_id": render,
                        "prompt_id": prompt,
                        "clear": clear,
                        "polarity_error": bool(clear and truth_sign * cell_score < 0),
                        "clean_correct_margin_re": truth_sign * clean / beta,
                        "render_main_harmful_re": harmful_sign * float(row_main[ri]) / beta,
                        "prompt_main_harmful_re": harmful_sign * float(col_main[pi]) / beta,
                        "render_marginal_rms_re": render_rms,
                        "prompt_marginal_rms_re": prompt_rms,
                        "full_orbit_harmful_re": harmful_sign * float(score.mean()) / beta,
                        "input_prompt_length_tokens": float(
                            orbit["cells"][(render, prompt)][
                                "input_prompt_length_tokens"
                            ]
                        ),
                        "answer_length_tokens": float(
                            orbit["cells"][(render, prompt)]["answer_length_tokens"]
                        ),
                        # Generic two-axis consistency/stability controls.  They
                        # intentionally do not use the centered mixed derivative.
                        "visual_axis_score_sd_re": float(np.std(score[:, pi]) / beta),
                        "language_axis_score_sd_re": float(np.std(score[ri, :]) / beta),
                        "visual_axis_entropy_mean": float(entropy[:, pi].mean()),
                        "language_axis_entropy_mean": float(entropy[ri, :].mean()),
                        "visual_axis_probability_dispersion": float(
                            np.sqrt(np.mean(np.sum((render_slice - render_probability_mean) ** 2, axis=-1)))
                        ),
                        "language_axis_probability_dispersion": float(
                            np.sqrt(np.mean(np.sum((prompt_slice - prompt_probability_mean) ** 2, axis=-1)))
                        ),
                        "full_orbit_probability_dispersion": float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        (probabilities - orbit_probability_mean) ** 2,
                                        axis=-1,
                                    )
                                )
                            )
                        ),
                        "cell_to_orbit_probability_kl": float(
                            _kl_rows(probabilities[ri, pi], orbit_probability_mean)
                        ),
                        "orbit_predictive_entropy": float(
                            -np.sum(
                                orbit_probability_mean
                                * np.log(np.clip(orbit_probability_mean, 1e-12, 1.0))
                            )
                        ),
                        # MMI is a transparent output-distribution PID-style
                        # proxy.  It does not close the hidden-PID collision.
                        "behavioral_pid_mmi_synergy_nats": float(
                            pid["mmi_synergy_nats"]
                        ),
                        "behavioral_local_synergy_excess_nats": float(
                            pid["local_synergy_excess_nats"][ri, pi]
                        ),
                        "behavioral_local_synergy_excess_abs_nats": abs(
                            float(pid["local_synergy_excess_nats"][ri, pi])
                        ),
                        "crossmodal_direct_effect_scalar_surrogate_harmful_re": surrogate_value,
                        "interaction_harmful_re": harmful_sign * cell_interaction / beta,
                        "interaction_abs_re": abs(cell_interaction) / beta,
                        "orbit_interaction_rms_re": interaction_rms,
                        "reader_support_gap": abs(support_actual - reader_fraction),
                        "additive_reader_support_gap": abs(support_additive - reader_fraction),
                        "gap_increase_due_to_interaction": abs(support_actual - reader_fraction)
                        - abs(support_additive - reader_fraction),
                        "commitment": float(orbit["commitment"][ri, pi]),
                        "commitment_interaction": float(orbit["commitment_interaction"][ri, pi]),
                        "acquisition_view": str(
                            orbit["cells"][(render, prompt)]["acquisition_view"]
                        ),
                        "tristate_entropy": float(
                            orbit["cells"][(render, prompt)]["tristate_entropy"]
                        ),
                    }
                )
    return rows


CLEAN_LENGTH_FEATURES = (
    "clean_correct_margin_re",
    "tristate_entropy",
    "input_prompt_length_tokens",
    "answer_length_tokens",
)
MARGINAL_FEATURES = CLEAN_LENGTH_FEATURES + (
    "render_main_harmful_re",
    "prompt_main_harmful_re",
    "render_marginal_rms_re",
    "prompt_marginal_rms_re",
)
CLOSEST_WORK_FEATURES = MARGINAL_FEATURES + (
    "full_orbit_harmful_re",
    "visual_axis_score_sd_re",
    "language_axis_score_sd_re",
    "visual_axis_entropy_mean",
    "language_axis_entropy_mean",
    "visual_axis_probability_dispersion",
    "language_axis_probability_dispersion",
    "full_orbit_probability_dispersion",
    "cell_to_orbit_probability_kl",
    "orbit_predictive_entropy",
)
BEHAVIORAL_PID_CONTROL_FEATURES = CLOSEST_WORK_FEATURES + (
    "behavioral_pid_mmi_synergy_nats",
    "behavioral_local_synergy_excess_nats",
    "behavioral_local_synergy_excess_abs_nats",
)
SURROGATE_FEATURES = BEHAVIORAL_PID_CONTROL_FEATURES + (
    "crossmodal_direct_effect_scalar_surrogate_harmful_re",
)
SURROGATE_CECD_FEATURES = SURROGATE_FEATURES + (
    "interaction_harmful_re",
    "interaction_abs_re",
)


def _fit_predict(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    seed: int,
) -> np.ndarray:
    x_train = np.asarray([[row[name] for name in features] for row in train_rows], dtype=float)
    x_test = np.asarray([[row[name] for name in features] for row in test_rows], dtype=float)
    y_train = np.asarray([row["polarity_error"] for row in train_rows], dtype=int)
    if np.unique(y_train).size < 2:
        return np.full(len(test_rows), float(y_train.mean()), dtype=float)
    scaler = StandardScaler().fit(x_train)
    categorical_train = np.asarray(
        [[row["finding"], row["acquisition_view"]] for row in train_rows], dtype=object
    )
    categorical_test = np.asarray(
        [[row["finding"], row["acquisition_view"]] for row in test_rows], dtype=object
    )
    encoder = _one_hot().fit(categorical_train)
    train = np.concatenate(
        (scaler.transform(x_train), encoder.transform(categorical_train)), axis=1
    )
    test = np.concatenate(
        (scaler.transform(x_test), encoder.transform(categorical_test)), axis=1
    )
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed)
    model.fit(train, y_train)
    return model.predict_proba(test)[:, 1]


def fit_serialized_predictor(
    rows: Sequence[Mapping[str, Any]], features: Sequence[str], seed: int
) -> dict[str, Any]:
    """Fit one dev-only predictor and serialize every apply-time parameter."""

    x = np.asarray([[row[name] for name in features] for row in rows], dtype=float)
    y = np.asarray([row["polarity_error"] for row in rows], dtype=int)
    if np.unique(y).size < 2:
        raise ContractError("dev_fit requires both polarity-error classes")
    scaler = StandardScaler().fit(x)
    categorical = np.asarray(
        [[row["finding"], row["acquisition_view"]] for row in rows], dtype=object
    )
    encoder = _one_hot().fit(categorical)
    design = np.concatenate(
        (scaler.transform(x), encoder.transform(categorical)), axis=1
    )
    model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=5000, random_state=seed
    ).fit(design, y)
    return {
        "features": list(features),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "categorical_columns": ["finding", "acquisition_view"],
        "categorical_categories": [
            [str(value) for value in category] for category in encoder.categories_
        ],
        "coefficient": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
        "class_order": [int(value) for value in model.classes_],
        "solver": "lbfgs",
        "C": 1.0,
        "seed": seed,
        "n_rows": len(rows),
    }


def apply_serialized_predictor(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]
) -> np.ndarray:
    features = [str(value) for value in bundle["features"]]
    mean = np.asarray(bundle["scaler_mean"], dtype=float)
    scale = np.asarray(bundle["scaler_scale"], dtype=float)
    x = np.asarray([[row[name] for name in features] for row in rows], dtype=float)
    numeric = (x - mean) / scale
    categories = [list(map(str, values)) for values in bundle["categorical_categories"]]
    categorical_parts = []
    for column_index, column in enumerate(("finding", "acquisition_view")):
        values = [str(row[column]) for row in rows]
        categorical_parts.append(
            np.asarray(
                [[float(value == category) for category in categories[column_index]] for value in values],
                dtype=float,
            )
        )
    design = np.concatenate((numeric, *categorical_parts), axis=1)
    coefficient = np.asarray(bundle["coefficient"], dtype=float)
    if design.shape[1] != coefficient.size:
        raise ContractError("frozen predictor design width mismatch")
    logit = design @ coefficient + float(bundle["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))


def _contract_geometry(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "primary_renders": list(contract["primary_renders"]),
        "primary_prompts": list(contract["primary_prompts"]),
        "baseline_render": contract["baseline_render"],
        "baseline_prompt": contract["baseline_prompt"],
        "identity_render": contract["identity_render"],
        "duplicate_prompt": contract["duplicate_prompt"],
    }


def _fixed_scales(
    orbits: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, dict[str, float]]]]:
    scales = np.full(len(orbits), np.nan, dtype=float)
    centers = np.full(len(orbits), np.nan, dtype=float)
    by_model: dict[str, dict[str, dict[str, float]]] = {}
    for model in sorted({str(row["model"]) for row in orbits}):
        by_model[model] = {}
        for finding in sorted({str(row["finding"]) for row in orbits if row["model"] == model}):
            take = [
                index for index, row in enumerate(orbits)
                if row["model"] == model and row["finding"] == finding
            ]
            beta = adjacent_reader_slope(
                [int(orbits[index]["reader_votes"]) for index in take],
                [_clean_score(orbits[index], contract) for index in take],
            )
            center = float(np.mean([_clean_score(orbits[index], contract) for index in take]))
            if beta <= 1e-8:
                raise ContractError(f"non-positive dev reader scale for {model}/{finding}")
            by_model[model][finding] = {"scale": beta, "center": center}
            scales[take] = beta
            centers[take] = center
    if not np.isfinite(scales).all() or not np.isfinite(centers).all():
        raise RuntimeError("dev scale assignment incomplete")
    return scales, centers, by_model


def fit_dev_stage(
    payload: Mapping[str, Any], *, folds: int, draws: int, seed: int
) -> dict[str, Any]:
    contract = validate_payload(payload)
    if contract["split"] != "dev_fit" or contract["source_manifest_split"] != "dev":
        raise ContractError("dev_fit mode requires the truthful dev manifest split")
    orbits = build_orbits(contract)
    scales, centers, fixed = _fixed_scales(orbits, contract)
    model_bundles: dict[str, Any] = {}
    for model in sorted(fixed):
        take = [index for index, row in enumerate(orbits) if row["model"] == model]
        selected = [orbits[index] for index in take]
        table = make_cell_table(selected, scales[take], centers[take], contract)
        clear = [row for row in table if row["clear"]]
        model_bundles[model] = {
            "reader_scale_and_center": fixed[model],
            "baseline_predictor": fit_serialized_predictor(
                clear, BEHAVIORAL_PID_CONTROL_FEATURES, seed + 101
            ),
            "candidate_predictor": fit_serialized_predictor(
                clear,
                BEHAVIORAL_PID_CONTROL_FEATURES
                + ("interaction_harmful_re", "interaction_abs_re"),
                seed + 103,
            ),
            "dev_clear_rows": len(clear),
        }
    # OOF analysis remains a development diagnostic only.  Scrub every legacy
    # authorization field even if a synthetic dev fixture happens to pass.
    diagnostic = analyze(payload, folds=folds, draws=draws, seed=seed)
    diagnostic["gate"]["authorized_for_method_level_treble_adapter_run"] = False
    diagnostic["gate"]["authorized_for_hidden_state_stage"] = False
    diagnostic["gate"]["behavioral_phenomenon_confirmed_on_locked_test"] = False
    return {
        "version": DEV_FIT_VERSION,
        "status": "dev_fit_complete_confirmation_not_opened",
        "stage_label": "dev_fit",
        "source_manifest_split": "dev",
        "contract_geometry": _contract_geometry(contract),
        "dev_image_ids": sorted({str(row["image_id"]) for row in orbits}),
        "models": model_bundles,
        "dev_oof_diagnostic": diagnostic,
        "gate": {
            "formal_null_decision_allowed": False,
            "formal_mechanism_confirmation": False,
            "authorized_for_method_level_treble_adapter_run": False,
            "authorized_for_hidden_state_stage": False,
        },
    }


def _confirmation_scales(
    selected: Sequence[Mapping[str, Any]], model_fit: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    scale, center = [], []
    frozen = model_fit["reader_scale_and_center"]
    for orbit in selected:
        finding = str(orbit["finding"])
        if finding not in frozen:
            raise ContractError(f"confirmation finding absent from dev fit: {finding}")
        scale.append(float(frozen[finding]["scale"]))
        center.append(float(frozen[finding]["center"]))
    return np.asarray(scale), np.asarray(center)


def _harmful_alignment(
    rows: Sequence[Mapping[str, Any]], target: np.ndarray, values: np.ndarray,
    draws: int, seed: int,
) -> dict[str, Any]:
    if not target.any() or not (~target.astype(bool)).any():
        return {"point": None, "ci95": None, "valid_draws": 0}
    point = float(values[target == 1].mean() - values[target == 0].mean())
    groups = sorted({str(row["image_id"]) for row in rows})
    by_group = {
        group: [index for index, row in enumerate(rows) if str(row["image_id"]) == group]
        for group in groups
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        take = np.asarray(
            [index for group in rng.choice(groups, len(groups), replace=True)
             for index in by_group[str(group)]], dtype=int
        )
        y = target[take]
        if y.any() and (~y.astype(bool)).any():
            x = values[take]
            samples.append(float(x[y == 1].mean() - x[y == 0].mean()))
    return {"point": point, "ci95": _quantile(samples), "valid_draws": len(samples)}


def apply_confirmation_stage(
    payload: Mapping[str, Any], dev_fit: Mapping[str, Any], *, draws: int, seed: int
) -> dict[str, Any]:
    contract = validate_payload(payload)
    if (
        contract["split"] != "confirmation_locked"
        or contract["source_manifest_split"] != "confirmation"
    ):
        raise ContractError("confirmation mode requires the truthful confirmation split")
    if (
        dev_fit.get("version") != DEV_FIT_VERSION
        or dev_fit.get("status") != "dev_fit_complete_confirmation_not_opened"
        or dev_fit.get("gate", {}).get("authorized_for_method_level_treble_adapter_run") is not False
        or dev_fit.get("contract_geometry") != _contract_geometry(contract)
    ):
        raise ContractError("invalid or incompatible dev-fit artifact")
    orbits = build_orbits(contract)
    confirmation_images = {str(row["image_id"]) for row in orbits}
    if confirmation_images & set(map(str, dev_fit.get("dev_image_ids", []))):
        raise ContractError("dev/confirmation whole-image leakage")
    model_names = sorted({str(row["model"]) for row in orbits})
    if set(model_names) != set(dev_fit.get("models", {})):
        raise ContractError("dev/confirmation model identity mismatch")
    slope_bootstrap = slope_cluster_bootstrap(orbits, contract, draws, seed + 701)
    models: dict[str, Any] = {}
    passing: list[str] = []
    for model_index, model in enumerate(model_names):
        selected = [row for row in orbits if row["model"] == model]
        model_fit = dev_fit["models"][model]
        scales, centers = _confirmation_scales(selected, model_fit)
        table = make_cell_table(selected, scales, centers, contract)
        clear = [row for row in table if row["clear"]]
        target = np.asarray([row["polarity_error"] for row in clear], dtype=int)
        baseline = apply_serialized_predictor(clear, model_fit["baseline_predictor"])
        candidate = apply_serialized_predictor(clear, model_fit["candidate_predictor"])
        pooled = prediction_metrics(target, baseline, candidate)
        pooled_boot = clustered_prediction_bootstrap(
            clear, target, baseline, candidate, draws, seed + 1000 * model_index + 11
        )
        interaction = np.asarray(
            [row["interaction_harmful_re"] for row in clear], dtype=float
        )
        harmful = _harmful_alignment(
            clear, target, interaction, draws, seed + 1000 * model_index + 13
        )
        per_finding: dict[str, Any] = {}
        for finding_index, finding in enumerate(sorted({row["finding"] for row in clear})):
            take = [index for index, row in enumerate(clear) if row["finding"] == finding]
            finding_rows = [clear[index] for index in take]
            metric = prediction_metrics(target[take], baseline[take], candidate[take])
            boot = clustered_prediction_bootstrap(
                finding_rows, target[take], baseline[take], candidate[take], draws,
                seed + 1000 * model_index + 100 + finding_index,
            )
            alignment = _harmful_alignment(
                finding_rows, target[take], interaction[take], draws,
                seed + 1000 * model_index + 200 + finding_index,
            )
            per_finding[finding] = {
                **metric,
                "image_cluster_bootstrap": boot,
                "harmful_alignment": alignment,
            }
        delta_positive = sum(
            int(row["delta_auc"] is not None and row["delta_auc"] > 0)
            for row in per_finding.values()
        )
        alignment_positive = sum(
            int(row["harmful_alignment"]["point"] is not None
                and row["harmful_alignment"]["point"] > 0)
            for row in per_finding.values()
        )
        no_opposite_mcid = all(
            row["delta_auc"] is not None and row["delta_auc"] > -0.03
            for row in per_finding.values()
        )
        no_significant_opposite = all(
            row["image_cluster_bootstrap"]["delta_auc_ci95"] is not None
            and row["image_cluster_bootstrap"]["delta_auc_ci95"][1] >= 0
            for row in per_finding.values()
        )
        orbit_ms = np.asarray(
            [float(np.mean((row["interaction"] / scale) ** 2))
             for row, scale in zip(selected, scales)], dtype=float
        )
        rms = clustered_rms_ci(
            orbit_ms, [str(row["image_id"]) for row in selected], draws,
            seed + 1000 * model_index + 17,
        )
        noise = identity_noise(selected, scales, contract)
        noise_pass = noise["maximum_rms_re"] <= 0.1 * rms["point"]
        reader_slopes = slope_bootstrap[model]
        reader_pass = len(reader_slopes) == 4 and all(
            row["ci95"] and row["ci95"][0] > 0 for row in reader_slopes.values()
        )
        pooled_pass = bool(
            pooled["delta_auc"] is not None
            and pooled["delta_auc"] >= 0.03
            and pooled_boot["delta_auc_ci95"]
            and pooled_boot["delta_auc_ci95"][0] > 0
        )
        harmful_pass = bool(harmful["ci95"] and harmful["ci95"][0] > 0)
        rms_pass = bool(rms["point"] >= 0.25 and rms["ci95"] and rms["ci95"][0] > 0)
        heterogeneity_pass = bool(
            delta_positive >= 3 and alignment_positive >= 3
            and no_opposite_mcid and no_significant_opposite
        )
        model_pass = bool(
            pooled_pass and harmful_pass and rms_pass and noise_pass
            and reader_pass and heterogeneity_pass
        )
        if model_pass:
            passing.append(model)
        models[model] = {
            "n_confirmation_orbits": len(selected),
            "n_clear_claims": len(clear),
            "dev_predictor_refit_on_confirmation": False,
            "pooled_four_finding_delta_auc": {
                **pooled, "image_cluster_bootstrap": pooled_boot,
            },
            "pooled_harmful_alignment": harmful,
            "interaction_rms_reader_equivalents": rms,
            "identity_controls": {**noise, "below_one_tenth": noise_pass},
            "reader_slope_cluster_bootstrap": reader_slopes,
            "per_finding": per_finding,
            "heterogeneity_guard": {
                "delta_positive_findings": delta_positive,
                "harmful_alignment_positive_findings": alignment_positive,
                "no_finding_delta_at_or_below_minus_0p03": no_opposite_mcid,
                "no_finding_ci_strictly_below_zero": no_significant_opposite,
                "passed": heterogeneity_pass,
            },
            "gate_components": {
                "pooled_delta_auc_point_at_least_0p03_and_ci_above_zero": pooled_pass,
                "pooled_harmful_alignment_ci_above_zero": harmful_pass,
                "interaction_rms_at_least_0p25_re_and_ci_above_zero": rms_pass,
                "identity_below_one_tenth": noise_pass,
                "all_reader_slopes_ci_above_zero": reader_pass,
                "heterogeneity_guard": heterogeneity_pass,
            },
            "model_confirmation_pass": model_pass,
        }
    authorized = len(passing) == len(model_names) == 2
    return {
        "version": CONFIRMATION_VERSION,
        "status": "complete",
        "stage_label": "confirmation_locked",
        "source_manifest_split": "confirmation",
        "dev_fit_version": DEV_FIT_VERSION,
        "models": models,
        "gate": {
            "name": "behavioral_confirmation_locked_v1",
            "confirmation_passing_models": passing,
            "both_models_pass": authorized,
            "authorized_for_method_level_treble_adapter_run": authorized,
            "authorized_for_hidden_state_stage": False,
            "behavioral_phenomenon_confirmed_on_locked_test": authorized,
        },
        "exact_treble_method_collision": {
            "status": "still_requires_separate_dual_semantics_envelope",
            "hidden_state_authorized": False,
        },
    }


def grouped_oof(
    rows: Sequence[Mapping[str, Any]],
    fold_by_image: Mapping[str, int],
    features: Sequence[str],
    seed: int,
) -> np.ndarray:
    prediction = np.full(len(rows), np.nan, dtype=float)
    folds = sorted(set(fold_by_image.values()))
    for fold in folds:
        train_index = [i for i, row in enumerate(rows) if fold_by_image[row["image_id"]] != fold]
        test_index = [i for i, row in enumerate(rows) if fold_by_image[row["image_id"]] == fold]
        train = [rows[i] for i in train_index]
        test = [rows[i] for i in test_index]
        values = _fit_predict(train, test, features, seed + fold)
        prediction[test_index] = values
    if not np.isfinite(prediction).all():
        raise RuntimeError("OOF predictions are incomplete")
    return prediction


def leakage_free_model_oof(
    orbits: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    contract: Mapping[str, Any],
    *,
    surrogate_complete: bool,
    seed: int,
) -> dict[str, Any]:
    """Create features and fit models using only each outer fold's training data.

    A single precomputed OOF scale is safe for a held-out row but unsafe for
    the *training* rows of another fold because it may have used that fold.
    Rebuilding all feature scales inside each predictor fold avoids this subtle
    normalization leakage.
    """

    held_rows: list[dict[str, Any]] = []
    baseline_feature_sets = {
        "clean_length": CLEAN_LENGTH_FEATURES,
        "marginal": MARGINAL_FEATURES,
        "generic_stability": CLOSEST_WORK_FEATURES,
        "behavioral_pid": BEHAVIORAL_PID_CONTROL_FEATURES,
    }
    baseline_values: dict[str, list[float]] = {
        name: [] for name in baseline_feature_sets
    }
    candidate_values: dict[str, list[float]] = {
        name: [] for name in baseline_feature_sets
    }
    surrogate_base_values: list[float] = []
    surrogate_candidate_values: list[float] = []
    orbit_fold = {
        (str(orbit["image_id"]), str(orbit["finding"])): int(fold)
        for orbit, fold in zip(orbits, fold_ids)
    }
    for fold in sorted(np.unique(fold_ids)):
        train_orbits = [index for index in range(len(orbits)) if fold_ids[index] != fold]
        scale = np.full(len(orbits), np.nan, dtype=float)
        center = np.full(len(orbits), np.nan, dtype=float)
        for finding in sorted({str(row["finding"]) for row in orbits}):
            fit = [index for index in train_orbits if orbits[index]["finding"] == finding]
            beta = adjacent_reader_slope(
                [orbits[index]["reader_votes"] for index in fit],
                [_clean_score(orbits[index], contract) for index in fit],
            )
            mean = float(np.mean([_clean_score(orbits[index], contract) for index in fit]))
            for index, orbit in enumerate(orbits):
                if orbit["finding"] == finding:
                    scale[index] = beta
                    center[index] = mean
        table = [row for row in make_cell_table(orbits, scale, center, contract) if row["clear"]]
        train = [
            row for row in table
            if orbit_fold[(str(row["image_id"]), str(row["finding"]))] != fold
        ]
        test = [
            row for row in table
            if orbit_fold[(str(row["image_id"]), str(row["finding"]))] == fold
        ]
        held_rows.extend(test)
        for feature_index, (name, features) in enumerate(
            baseline_feature_sets.items(), start=1
        ):
            baseline_values[name].extend(
                _fit_predict(
                    train,
                    test,
                    features,
                    seed + 100 * fold + 10 * feature_index + 1,
                )
            )
            candidate_values[name].extend(
                _fit_predict(
                    train,
                    test,
                    features + ("interaction_harmful_re", "interaction_abs_re"),
                    seed + 100 * fold + 10 * feature_index + 2,
                )
            )
        if surrogate_complete:
            surrogate_base_values.extend(
                _fit_predict(train, test, SURROGATE_FEATURES, seed + 100 * fold + 7)
            )
            surrogate_candidate_values.extend(
                _fit_predict(train, test, SURROGATE_CECD_FEATURES, seed + 100 * fold + 9)
            )
    return {
        "rows": held_rows,
        "baseline_predictions": {
            name: np.asarray(values, dtype=float)
            for name, values in baseline_values.items()
        },
        "candidate_predictions": {
            name: np.asarray(values, dtype=float)
            for name, values in candidate_values.items()
        },
        # Compatibility aliases.  The engineering screen is now explicitly
        # the strongest behavior-only baseline, including generic stability
        # and the transparent behavioral MMI PID-style control.
        "marginal": np.asarray(baseline_values["marginal"], dtype=float),
        "marginal_candidate": np.asarray(candidate_values["marginal"], dtype=float),
        "engineering_baseline": np.asarray(
            baseline_values["behavioral_pid"], dtype=float
        ),
        "engineering_candidate": np.asarray(
            candidate_values["behavioral_pid"], dtype=float
        ),
        "surrogate_baseline": np.asarray(surrogate_base_values, dtype=float)
        if surrogate_complete else None,
        "surrogate_candidate": np.asarray(surrogate_candidate_values, dtype=float)
        if surrogate_complete else None,
    }


def prediction_metrics(target: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float | None]:
    if np.unique(target).size < 2:
        return {
            "baseline_auc": None, "candidate_auc": None, "delta_auc": None,
            "baseline_brier": float(brier_score_loss(target, baseline)),
            "candidate_brier": float(brier_score_loss(target, candidate)),
            "brier_improvement": float(brier_score_loss(target, baseline) - brier_score_loss(target, candidate)),
        }
    base_auc = float(roc_auc_score(target, baseline))
    candidate_auc = float(roc_auc_score(target, candidate))
    base_brier = float(brier_score_loss(target, baseline))
    candidate_brier = float(brier_score_loss(target, candidate))
    return {
        "baseline_auc": base_auc,
        "candidate_auc": candidate_auc,
        "delta_auc": candidate_auc - base_auc,
        "baseline_brier": base_brier,
        "candidate_brier": candidate_brier,
        "brier_improvement": base_brier - candidate_brier,
    }


def _quantile(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def clustered_prediction_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    groups = sorted({str(row["image_id"]) for row in rows})
    indices = {group: np.flatnonzero(np.asarray([row["image_id"] for row in rows]) == group) for group in groups}
    rng = np.random.default_rng(seed)
    auc, brier = [], []
    for _ in range(draws):
        sampled = rng.choice(groups, len(groups), replace=True)
        take = np.concatenate([indices[str(group)] for group in sampled])
        metrics = prediction_metrics(target[take], baseline[take], candidate[take])
        if metrics["delta_auc"] is not None:
            auc.append(float(metrics["delta_auc"]))
        brier.append(float(metrics["brier_improvement"]))
    return {"delta_auc_ci95": _quantile(auc), "brier_improvement_ci95": _quantile(brier), "valid_auc_draws": len(auc)}


def clustered_mean_ci(values: np.ndarray, groups: Sequence[str], draws: int, seed: int) -> dict[str, Any]:
    unique = sorted(set(groups))
    group_array = np.asarray(groups, dtype=object)
    by_group = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        chosen = rng.choice(unique, len(unique), replace=True)
        take = np.concatenate([by_group[str(group)] for group in chosen])
        samples.append(float(np.mean(values[take])))
    return {"point": float(np.mean(values)), "ci95": _quantile(samples), "valid_draws": len(samples)}


def clustered_rms_ci(
    orbit_mean_squares: np.ndarray, groups: Sequence[str], draws: int, seed: int
) -> dict[str, Any]:
    """Pooled RMS with image-cluster resampling of orbit mean-square terms."""

    values = np.asarray(orbit_mean_squares, dtype=float)
    unique = sorted(set(groups))
    group_array = np.asarray(groups, dtype=object)
    by_group = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        chosen = rng.choice(unique, len(unique), replace=True)
        take = np.concatenate([by_group[str(group)] for group in chosen])
        samples.append(float(np.sqrt(np.mean(values[take]))))
    return {
        "point": float(np.sqrt(np.mean(values))),
        "ci95": _quantile(samples),
        "valid_draws": len(samples),
        "aggregation": "sqrt(mean over cells of squared reader-equivalent interaction)",
    }


def identity_noise(
    orbits: Sequence[Mapping[str, Any]], scales: np.ndarray, contract: Mapping[str, Any]
) -> dict[str, Any]:
    render_deltas, prompt_deltas = [], []
    for index, orbit in enumerate(orbits):
        beta = float(scales[index])
        if beta <= 1e-8:
            continue
        cells = orbit["cells"]
        for prompt in contract["primary_prompts"]:
            render_deltas.append(
                (float(cells[(contract["identity_render"], prompt)]["signed_score"])
                 - float(cells[(contract["baseline_render"], prompt)]["signed_score"])) / beta
            )
        render = contract["baseline_render"]
        prompt_deltas.append(
            (float(cells[(render, contract["duplicate_prompt"])]["signed_score"])
             - float(cells[(render, contract["baseline_prompt"])]["signed_score"])) / beta
        )
    render_rms = float(np.sqrt(np.mean(np.square(render_deltas)))) if render_deltas else math.inf
    prompt_rms = float(np.sqrt(np.mean(np.square(prompt_deltas)))) if prompt_deltas else math.inf
    return {"identity_render_rms_re": render_rms, "duplicate_prompt_rms_re": prompt_rms, "maximum_rms_re": max(render_rms, prompt_rms)}


def analyze_model(
    model: str,
    orbits: Sequence[Mapping[str, Any]],
    fold_ids: np.ndarray,
    scales: np.ndarray,
    centers: np.ndarray,
    contract: Mapping[str, Any],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    selected_indices = [index for index, row in enumerate(orbits) if row["model"] == model]
    selected = [orbits[index] for index in selected_indices]
    selected_scales = scales[selected_indices]
    selected_centers = centers[selected_indices]
    selected_folds = fold_ids[selected_indices]
    table = make_cell_table(selected, selected_scales, selected_centers, contract)
    disagreement = [row for row in table if not row["clear"]]
    surrogate_complete = bool(selected) and all(
        row["crossmodal_direct_effect_scalar_surrogate"] is not None
        for row in selected
    )
    fold_by_image: dict[str, int] = {}
    for orbit, fold in zip(selected, selected_folds):
        previous = fold_by_image.setdefault(str(orbit["image_id"]), int(fold))
        if previous != int(fold):
            raise RuntimeError("image group received two model-local folds")

    oof = leakage_free_model_oof(
        selected, selected_folds, contract, surrogate_complete=surrogate_complete,
        seed=seed + 11
    )
    clear = oof["rows"]
    target = np.asarray([row["polarity_error"] for row in clear], dtype=int)
    marginal_prediction = oof["marginal"]
    marginal_candidate_prediction = oof["marginal_candidate"]
    engineering_baseline_prediction = oof["engineering_baseline"]
    engineering_candidate_prediction = oof["engineering_candidate"]
    surrogate_baseline_prediction = oof["surrogate_baseline"]
    surrogate_candidate_prediction = oof["surrogate_candidate"]

    behavioral_control_comparisons: dict[str, Any] = {}
    comparison_seed_offset = {
        "clean_length": 91,
        "marginal": 101,
        "generic_stability": 111,
        "behavioral_pid": 121,
    }
    for name in comparison_seed_offset:
        baseline_prediction = oof["baseline_predictions"][name]
        candidate_prediction = oof["candidate_predictions"][name]
        behavioral_control_comparisons[name] = {
            **prediction_metrics(target, baseline_prediction, candidate_prediction),
            "image_cluster_bootstrap": clustered_prediction_bootstrap(
                clear,
                target,
                baseline_prediction,
                candidate_prediction,
                draws,
                seed + comparison_seed_offset[name],
            ),
        }
    marginal_metrics = behavioral_control_comparisons["marginal"]
    engineering_metrics = prediction_metrics(
        target, engineering_baseline_prediction, engineering_candidate_prediction
    )
    marginal_boot = marginal_metrics["image_cluster_bootstrap"]
    engineering_boot = clustered_prediction_bootstrap(
        clear, target, engineering_baseline_prediction, engineering_candidate_prediction,
        draws, seed + 123
    )
    if surrogate_complete:
        surrogate_metrics = prediction_metrics(
            target, surrogate_baseline_prediction, surrogate_candidate_prediction
        )
        surrogate_boot = clustered_prediction_bootstrap(
            clear, target, surrogate_baseline_prediction, surrogate_candidate_prediction,
            draws, seed + 105
        )
    else:
        surrogate_metrics = {
            "baseline_auc": None, "candidate_auc": None, "delta_auc": None,
            "baseline_brier": None, "candidate_brier": None, "brier_improvement": None,
        }
        surrogate_boot = {
            "delta_auc_ci95": None, "brier_improvement_ci95": None,
            "valid_auc_draws": 0,
        }

    interaction_values = np.asarray([row["interaction_harmful_re"] for row in clear], dtype=float)
    if target.any() and (~target.astype(bool)).any():
        harmful_difference = float(interaction_values[target == 1].mean() - interaction_values[target == 0].mean())
        groups = sorted({row["image_id"] for row in clear})
        by_group = {group: [i for i, row in enumerate(clear) if row["image_id"] == group] for group in groups}
        rng = np.random.default_rng(seed + 107)
        harmful_boot_values = []
        for _ in range(draws):
            take = np.asarray([i for group in rng.choice(groups, len(groups), replace=True) for i in by_group[str(group)]], dtype=int)
            y = target[take]
            if y.any() and (~y.astype(bool)).any():
                harmful_boot_values.append(float(interaction_values[take][y == 1].mean() - interaction_values[take][y == 0].mean()))
        harmful = {"error_minus_correct_mean_harmful_interaction_re": harmful_difference, "ci95": _quantile(harmful_boot_values), "valid_draws": len(harmful_boot_values)}
    else:
        harmful = {"error_minus_correct_mean_harmful_interaction_re": None, "ci95": None, "valid_draws": 0}

    orbit_mean_squares = np.asarray(
        [float(np.mean((row["interaction"] / scale) ** 2)) for row, scale in zip(selected, selected_scales) if scale > 1e-8],
        dtype=float,
    )
    orbit_groups = [str(row["image_id"]) for row, scale in zip(selected, selected_scales) if scale > 1e-8]
    rms_summary = clustered_rms_ci(
        orbit_mean_squares, orbit_groups, draws, seed + 109
    )
    noise = identity_noise(selected, selected_scales, contract)
    noise["below_one_tenth_of_clinical_interaction"] = bool(
        noise["maximum_rms_re"] <= 0.1 * rms_summary["point"]
    )

    per_finding = {}
    for finding in sorted({row["finding"] for row in clear}):
        take = [i for i, row in enumerate(clear) if row["finding"] == finding]
        metric = prediction_metrics(
            target[take], engineering_baseline_prediction[take],
            engineering_candidate_prediction[take]
        )
        finding_target = target[take]
        finding_interaction = interaction_values[take]
        alignment = None
        alignment_boot_values: list[float] = []
        if finding_target.any() and (~finding_target.astype(bool)).any():
            alignment = float(finding_interaction[finding_target == 1].mean() - finding_interaction[finding_target == 0].mean())
            finding_rows = [clear[index] for index in take]
            finding_groups = sorted({str(row["image_id"]) for row in finding_rows})
            by_group = {
                group: [
                    index
                    for index, row in enumerate(finding_rows)
                    if str(row["image_id"]) == group
                ]
                for group in finding_groups
            }
            rng = np.random.default_rng(seed + 211 + len(per_finding))
            for _ in range(draws):
                sampled = rng.choice(finding_groups, len(finding_groups), replace=True)
                selected_take = np.asarray(
                    [index for group in sampled for index in by_group[str(group)]],
                    dtype=int,
                )
                sampled_target = finding_target[selected_take]
                if sampled_target.any() and (~sampled_target.astype(bool)).any():
                    sampled_interaction = finding_interaction[selected_take]
                    alignment_boot_values.append(
                        float(
                            sampled_interaction[sampled_target == 1].mean()
                            - sampled_interaction[sampled_target == 0].mean()
                        )
                    )
        finding_boot = clustered_prediction_bootstrap(
            [clear[index] for index in take],
            target[take],
            engineering_baseline_prediction[take],
            engineering_candidate_prediction[take],
            draws,
            seed + 311 + len(per_finding),
        )
        delta_ci = finding_boot["delta_auc_ci95"]
        alignment_ci = _quantile(alignment_boot_values)
        same_direction = bool(
            metric["delta_auc"] is not None
            and metric["delta_auc"] >= 0.03
            and delta_ci
            and delta_ci[0] > 0
            and alignment is not None
            and alignment > 0
            and alignment_ci
            and alignment_ci[0] > 0
            and finding_boot["valid_auc_draws"] >= int(0.95 * draws)
        )
        per_finding[finding] = {
            "n_cells": len(take), "delta_auc_over_strongest": metric["delta_auc"],
            "brier_improvement_over_strongest": metric["brier_improvement"],
            "image_cluster_bootstrap": finding_boot,
            "harmful_alignment": alignment,
            "harmful_alignment_cluster_ci95": alignment_ci,
            "harmful_alignment_valid_draws": len(alignment_boot_values),
            "same_direction": same_direction,
        }
    same_direction = sum(int(value["same_direction"]) for value in per_finding.values())

    if disagreement:
        gap_values = np.asarray([row["gap_increase_due_to_interaction"] for row in disagreement], dtype=float)
        commitment_values = np.asarray(
            [abs(row["commitment_interaction"]) for row in disagreement], dtype=float
        )
        disagreement_groups = [row["image_id"] for row in disagreement]
        disagreement_report = {
            "n_cells": len(disagreement),
            "reader_grounded_ordinal_support_gap_increase": clustered_mean_ci(
                gap_values, disagreement_groups, draws, seed + 113
            ),
            "commitment_interaction_mean_absolute": clustered_mean_ci(
                commitment_values, disagreement_groups, draws, seed + 127
            ),
            "gate_role": "diagnostic_only",
            "truth_guardrail": "Maybe/uncertain verbalizer is a model coordinate, never reader truth",
        }
    else:
        disagreement_report = {"n_cells": 0, "gate_role": "diagnostic_only"}

    rms_pass = bool(rms_summary["point"] >= 0.25 and rms_summary["ci95"] and rms_summary["ci95"][0] > 0)
    engineering_auc_pass = bool(
        engineering_metrics["delta_auc"] is not None
        and engineering_metrics["delta_auc"] >= 0.03
        and engineering_boot["delta_auc_ci95"]
        and engineering_boot["delta_auc_ci95"][0] > 0
        and engineering_boot["valid_auc_draws"] >= int(0.95 * draws)
    )
    surrogate_increment_diagnostic_pass = bool(
        surrogate_complete
        and surrogate_metrics["delta_auc"] is not None
        and surrogate_metrics["delta_auc"] >= 0.03
        and surrogate_boot["delta_auc_ci95"]
        and surrogate_boot["delta_auc_ci95"][0] > 0
    )
    harmful_pass = bool(harmful["ci95"] and harmful["ci95"][0] > 0)
    finding_pass = bool(len(per_finding) >= 4 and same_direction >= 3)
    engineering_pass = bool(
        rms_pass and engineering_auc_pass and harmful_pass
        and noise["below_one_tenth_of_clinical_interaction"] and finding_pass
    )
    return {
        "model": model,
        "n_orbits": len(selected),
        "n_clear_cells": len(clear),
        "clear_reader_polarity_error": {
            "definition": "vote0: signed_score>0; vote3: signed_score<0; exact zero is not forced into error",
            "error_rate": float(target.mean()) if len(target) else None,
            "cecd_vs_clean_render_prompt_marginals": {**marginal_metrics, "image_cluster_bootstrap": marginal_boot},
            "behavioral_incremental_collision_ladder": {
                "clean_margin_entropy_prompt_and_fixed_answer_length": behavioral_control_comparisons[
                    "clean_length"
                ],
                "render_prompt_marginals": behavioral_control_comparisons["marginal"],
                "generic_two_axis_stability_and_full_grid": behavioral_control_comparisons[
                    "generic_stability"
                ],
                "behavioral_mmi_pid_style_synergy": behavioral_control_comparisons[
                    "behavioral_pid"
                ],
                "gate_uses": "behavioral_mmi_pid_style_synergy",
                "guardrail": (
                    "the last rung is an output-distribution MMI proxy, not representation-level "
                    "PID and not closure of the post-Stage-1 mechanism collision"
                ),
            },
            "cecd_vs_marginals_plus_full_orbit": {
                **engineering_metrics, "image_cluster_bootstrap": engineering_boot,
                "available": True,
                "stage": "dev_group_oof_behavioral_screen",
                "baseline_now_includes": (
                    "clean margin, entropy, prompt/fixed-answer length, render/prompt marginals, "
                    "full-grid and generic two-axis stability, behavioral MMI PID-style synergy"
                ),
            },
            "cecd_vs_marginals_full_orbit_and_crossmodal_direct_effect_scalar_surrogate": {
                **surrogate_metrics, "image_cluster_bootstrap": surrogate_boot,
                "available": surrogate_complete,
                "gate_role": "diagnostic_only_never_exact_treble_never_hidden_state_authorization",
                "semantic_guardrail": (
                    "a per-cell scalar surrogate is not Treble's global PCA steering "
                    "intervention and cannot substitute for a method-level reproduction"
                ),
            },
            "clinically_harmful_direction": harmful,
        },
        "interaction_rms_reader_equivalents": rms_summary,
        "identity_controls": noise,
        "reader_disagreement_separate_axis": disagreement_report,
        "per_finding": per_finding,
        "same_direction_findings": same_direction,
        "model_engineering_screen_pass": engineering_pass,
        "model_scalar_surrogate_diagnostic_pass": surrogate_increment_diagnostic_pass,
        "model_screen_pass": engineering_pass,
        "gate_components": {
            "interaction_rms_at_least_0_25_re_and_ci_above_zero": rms_pass,
            "incremental_auc_at_least_0_03_ci_above_zero_over_complete_behavioral_collision_ladder": engineering_auc_pass,
            "scalar_surrogate_increment_diagnostic_only": surrogate_increment_diagnostic_pass,
            "clinically_harmful_direction_ci_above_zero": harmful_pass,
            "identity_below_one_tenth": noise["below_one_tenth_of_clinical_interaction"],
            "at_least_three_of_four_findings_same_direction": finding_pass,
            "crossmodal_direct_effect_scalar_surrogate_complete": surrogate_complete,
        },
    }


def analyze(payload: Mapping[str, Any], *, folds: int = 5, draws: int = 2000, seed: int = 42) -> dict[str, Any]:
    contract = validate_payload(payload)
    orbits = build_orbits(contract)
    fold_ids = grouped_folds(orbits, folds, seed)
    scales, centers, crossfit_audit = crossfit_reader_scale(orbits, fold_ids, contract)
    model_names = sorted({str(row["model"]) for row in orbits})
    models = {
        model: analyze_model(model, orbits, fold_ids, scales, centers, contract, draws, seed + 1000 * index)
        for index, model in enumerate(model_names)
    }
    reader_slope_bootstrap = slope_cluster_bootstrap(
        orbits, contract, draws, seed + 701
    )
    for model, result in models.items():
        slopes = reader_slope_bootstrap[model]
        scale_pass = bool(
            len(slopes) >= 4
            and all(value["ci95"] and value["ci95"][0] > 0 for value in slopes.values())
        )
        result["gate_components"][
            "all_finding_reader_scale_cluster_ci_above_zero"
        ] = scale_pass
        result["model_engineering_screen_pass"] = bool(
            result["model_engineering_screen_pass"] and scale_pass
        )
        result["model_screen_pass"] = result["model_engineering_screen_pass"]
    engineering_passing = [
        name for name, result in models.items()
        if result["model_engineering_screen_pass"]
    ]
    two_model_engineering = len(engineering_passing) >= 2
    return {
        "version": VERSION,
        "status": "complete",
        "estimand": {
            "name": "Clinical-Equivalence Composition Defect (CECD)",
            "formula": "m_rp - mean_p(m_rp) - mean_r(m_rp) + mean_rp(m_rp)",
            "type": "two-way centered interaction / discrete mixed derivative",
            "guardrail": "not an algebraic commutator, RP-PR contrast, or order effect",
        },
        "contract": {
            "schema_version": CONTRACT_VERSION,
            "split": "dev-only",
            "score_definition": "FP32 Yes-minus-No logit",
            "clear_truth": "independent reader polarity at vote 0/3",
            "disagreement_truth": "ordinal reader support only; Maybe is never truth",
            "image_grouped_cv": True,
            "screen_partition": "dev_only",
            "held_out_semantics": (
                "outer image-group OOF folds within dev; never describe as locked-test confirmation"
            ),
            "locked_test_behavioral_confirmation_required": True,
            "finding_specific_adjacent_reader_scale_crossfit": True,
            "fold_local_controls": [
                "finding",
                "normalized_acquisition_view",
                "tristate_entropy",
                "input_prompt_length_tokens",
                "fixed_one_token_answer_length",
                "render_prompt_marginals",
                "generic_two_axis_stability",
                "full_grid_stability",
                "behavioral_output_distribution_mmi_pid_style_synergy",
            ],
            "control_feature_sets": {
                "clean_length": list(CLEAN_LENGTH_FEATURES),
                "marginal": list(MARGINAL_FEATURES),
                "generic_stability": list(CLOSEST_WORK_FEATURES),
                "behavioral_pid": list(BEHAVIORAL_PID_CONTROL_FEATURES),
                "candidate_increment": [
                    "interaction_harmful_re",
                    "interaction_abs_re",
                ],
            },
            "behavioral_pid_guardrail": (
                "MMI decomposition of the three-state output distribution is a generic behavior-level "
                "synergy control only; actual hidden-state PID remains a downstream mechanism requirement"
            ),
            "excluded_orbits": contract["excluded_orbits"],
            "excluded_orbit_count": len(contract["excluded_orbits"]),
            "invalid_cell_policy": "exclude complete image-claim orbit; never impute",
        },
        "reader_slope_cluster_bootstrap": reader_slope_bootstrap,
        "crossfit_reader_scale_audit": crossfit_audit,
        "models": models,
        "gate": {
            "name": PRIMARY_GATE_NAME,
            "engineering_passing_models": engineering_passing,
            "two_model_three_of_four_finding_engineering_screen": two_model_engineering,
            "authorized_for_method_level_treble_adapter_run": two_model_engineering,
            "authorized_for_hidden_state_stage": False,
            "behavioral_phenomenon_confirmed_on_locked_test": False,
            "oral_baseline_closure_established": False,
            "dynamic_activation_baseline_closure_established": False,
            "interpretation": (
                "a two-model pass is a dev-only outcome-blind screen that may authorize only the "
                "separately locked comparison; it neither confirms the phenomenon on locked test nor "
                "authorizes hidden-state compute or claims oral-level baseline closure"
            ),
            "fail_closed": True,
        },
        "stage_boundary": {
            "phenomenon_gate_status": (
                "dev_group_oof_screen_passed" if two_model_engineering else "dev_screen_no_go"
            ),
            "method_gate_status": "not_run",
            "locked_test_status": "not_opened_by_this_analyzer",
            "method_controls_still_required": [
                "treble_proceedings_common_protocol",
                "treble_released_source_common_protocol",
                "full_orbit",
                "official_compatible_static_activation_baseline",
                "official_compatible_dynamic_or_multimodal_activation_baseline",
                "representation_level_pid_or_explicitly_bounded_nonclosure",
            ],
            "prohibited_inference": (
                "the ten-method Treble envelope contains factorial controls but no official-compatible "
                "dynamic activation baseline; it cannot be labelled full method or ICLR-oral closure"
            ),
        },
        "exact_treble_method_collision": {
            "required_after_two_model_stage1": two_model_engineering,
            "artifact_schema": "cecd-treble-method-collision-v1",
            "status": "blocked" if two_model_engineering else "not_authorized",
            "blocking_reason": (
                "official paper/code intervention semantics unresolved; per-cell scalar "
                "surrogates are prohibited substitutes"
            ) if two_model_engineering else None,
            "accepted_external_fields": [
                "source_repo_commit", "reproduction_fidelity", "model_fingerprint",
                "calibration_split", "evaluation_split", "record_keys_sha256",
                "compute_ledger", "paired_method_metrics",
                "paired_cluster_bootstrap", "collision_verdict",
            ],
            "hidden_state_authorized": False,
            "authority": "independent external validator only",
        },
        "closest_work_guardrail": {
            "paper": "Treble Counterfactual VLMs (arXiv:2503.06169)",
            "warning": "cross-modal interaction alone is not novel",
            "surviving_claim": "clinical-equivalence product orbit + independent reader-vote units + incremental prediction of medical error",
            "required_pruning_rule": (
                "terminate unless a separate faithful method-level reproduction shows "
                "CECD beats official Treble intervention and full-orbit averaging on "
                "the preregistered paired clinical metric"
            ),
            "prohibited_shortcut": (
                "never use a per-cell direct-effect scalar surrogate as exact Treble "
                "or as hidden-state authorization"
            ),
        },
        "threshold_policy": "A wide or non-positive CI fails/inconclusive; thresholds are never relaxed for the 160-case pilot.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, action="append", required=True,
        help="wrapper JSON or repeat for Huatuo/Hulu runner factorial_rows.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("legacy_screen", "pilot_screen", "dev_fit", "confirmation_locked"),
        default="legacy_screen",
    )
    parser.add_argument(
        "--frozen-dev-fit", type=Path,
        help="required apply-only predictor artifact for confirmation_locked",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_inputs(args.input)
    if args.mode == "dev_fit":
        if args.frozen_dev_fit is not None:
            raise ContractError("dev_fit cannot consume a prior fit")
        result = fit_dev_stage(
            payload, folds=args.folds, draws=args.bootstrap_draws, seed=args.seed
        )
    elif args.mode == "confirmation_locked":
        if args.frozen_dev_fit is None or not args.frozen_dev_fit.is_file():
            raise ContractError("confirmation_locked requires --frozen-dev-fit")
        frozen_fit = json.loads(args.frozen_dev_fit.read_text(encoding="utf-8"))
        result = apply_confirmation_stage(
            payload, frozen_fit, draws=args.bootstrap_draws, seed=args.seed
        )
        result["dev_fit_binding"] = {
            "path": str(args.frozen_dev_fit.resolve()),
            "sha256": sha256_file(args.frozen_dev_fit),
        }
    elif args.mode == "pilot_screen":
        contract = validate_payload(payload)
        if contract["split"] != "pilot_screen" or contract["source_manifest_split"] != "pilot":
            raise ContractError("pilot_screen mode requires truthful pilot source")
        orbits = build_orbits(contract)
        result = {
            "version": VERSION,
            "status": "engineering_pilot_screen_complete",
            "stage_label": "pilot_screen",
            "source_manifest_split": "pilot",
            "models": sorted({str(row["model"]) for row in orbits}),
            "claim_orbits": len(orbits),
            "gate": {
                "formal_mechanism_no_go": False,
                "formal_mechanism_confirmation": False,
                "authorized_for_method_level_treble_adapter_run": False,
                "authorized_for_hidden_state_stage": False,
            },
        }
    else:
        # Compatibility only.  A legacy pilot-as-dev result can be inspected,
        # but no longer authorizes any downstream method stage.
        result = analyze(payload, folds=args.folds, draws=args.bootstrap_draws, seed=args.seed)
        result["legacy_authorization_revoked_by_three_stage_contract"] = True
        result["gate"]["authorized_for_method_level_treble_adapter_run"] = False
        result["gate"]["authorized_for_hidden_state_stage"] = False
    result["provenance"] = {
        "input_sha256": {str(path): sha256_file(path) for path in args.input},
        "code_sha256": sha256_file(Path(__file__)),
        "seed": args.seed,
        "folds": args.folds,
        "bootstrap_draws": args.bootstrap_draws,
        "mode": args.mode,
    }
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
