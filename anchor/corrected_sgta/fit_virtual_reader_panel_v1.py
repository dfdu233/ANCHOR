#!/usr/bin/env python3
"""Fit and confirm a virtual fixed-reader panel from final-layer VLM logits.

The development split is the only fitting split.  Each reader vote is modeled
as a Bernoulli observation with a shared image-score slope, finding fixed
effects, and (optionally) centered reader fixed effects.  The three reader
probabilities are then convolved exactly into the preregistered evidence states
0/3, 1--2/3, and 3/3.  Confirmation data never changes a parameter or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file


VERSION = "virtual-fixed-reader-panel-v1"
STATE_NAMES = ("zero_of_three", "one_or_two_of_three", "three_of_three")


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    array = np.asarray(values, dtype=np.float64) / float(temperature)
    array -= np.max(array)
    probabilities = np.exp(array)
    return probabilities / probabilities.sum()


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def poisson_binomial_three_state(probabilities: Sequence[float]) -> np.ndarray:
    """Return P(0 positives), P(1 or 2 positives), P(3 positives)."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("the fixed panel requires exactly three finite probabilities")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("reader probabilities must lie in [0, 1]")
    p0 = float(np.prod(1.0 - values))
    p3 = float(np.prod(values))
    return np.asarray([p0, max(0.0, 1.0 - p0 - p3), p3], dtype=np.float64)


def maybe_margin(logits: Sequence[float]) -> float:
    """Shift-invariant uncertainty coordinate U - logsumexp(Y, N)."""

    no, uncertain, yes = (float(value) for value in logits)
    maximum = max(yes, no)
    return uncertain - (maximum + math.log(math.exp(yes - maximum) + math.exp(no - maximum)))


def validate_reader_votes(
    row: dict[str, Any], reader_panel: Sequence[str]
) -> np.ndarray:
    raw = row.get("reader_votes")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{row.get('record_key')}: exactly three reader_votes required")
    mapping: dict[str, int] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("reader_votes entries must be objects")
        reader = str(item.get("rad_id", ""))
        vote = int(item.get("vote", -1))
        if reader in mapping or reader not in reader_panel or vote not in {0, 1}:
            raise ValueError(f"invalid fixed-panel vote: {item}")
        mapping[reader] = vote
    if set(mapping) != set(reader_panel):
        raise ValueError(f"reader panel mismatch: found {sorted(mapping)}")
    votes = np.asarray([mapping[reader] for reader in reader_panel], dtype=np.int64)
    if int(votes.sum()) != int(row.get("positive_votes", -1)):
        raise ValueError("individual votes disagree with positive_votes")
    return votes


def final_state_logits(row: dict[str, Any]) -> np.ndarray:
    raw = row.get("diagnostic_plain_logit_lens")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{row.get('record_key')}: final diagnostic logits absent")
    try:
        final_layer = max(raw, key=lambda value: int(value))
        values = raw[final_layer]
        return np.asarray(
            [values["refuted"], values["undetermined"], values["supported"]],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{row.get('record_key')}: malformed diagnostic logits") from error


def load_feature_records(
    directory: Path, expected_split: str, reader_panel: Sequence[str]
) -> list[dict[str, Any]]:
    path = directory / "metadata.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty metadata: {path}")
    output = []
    seen: set[str] = set()
    for source in rows:
        if str(source.get("experiment_split")) != expected_split:
            raise ValueError(f"{path}: row outside expected {expected_split} split")
        key = str(source.get("record_key", ""))
        if not key or key in seen:
            raise ValueError(f"missing or duplicate record_key: {key}")
        seen.add(key)
        votes = validate_reader_votes(source, reader_panel)
        logits = final_state_logits(source)
        if not np.isfinite(logits).all():
            raise ValueError(f"{key}: non-finite final logits")
        output.append(
            {
                "record_key": key,
                "image_id": str(source["image_id"]),
                "finding": str(source["finding"]),
                "positive_votes": int(votes.sum()),
                "reader_votes": votes,
                "logits": logits,
                "signed_score": float(logits[2] - logits[0]),
                "maybe_margin": maybe_margin(logits),
            }
        )
    return output


def attach_population_weights(
    rows: Sequence[dict[str, Any]], sampling_summary: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    """Inverse sampling-probability weights for the balanced vote-bin design."""

    availability = sampling_summary.get("availability_before_sampling")
    if not isinstance(availability, dict):
        raise ValueError("sampling summary lacks availability_before_sampling")
    selected: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        selected[(str(row["finding"]), int(row["positive_votes"]))] += 1
    expected_quota = int(
        sampling_summary.get("split_contract", {})
        .get("quotas_per_finding_vote_bin", {})
        .get(split, -1)
    )
    observed_findings = sorted({str(row["finding"]) for row in rows})
    incomplete = {
        f"{finding}:{vote}/3": selected[(finding, vote)]
        for finding in observed_findings
        for vote in range(4)
        if selected[(finding, vote)] != expected_quota
    }
    if expected_quota <= 0 or incomplete:
        raise ValueError(
            f"{split} input is not the complete frozen balanced design; "
            f"expected quota={expected_quota}, mismatches={incomplete}"
        )
    output = []
    for source in rows:
        finding = str(source["finding"])
        vote = int(source["positive_votes"])
        try:
            population = int(availability[finding][f"{vote}/3"][split])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"sampling summary lacks {finding} {vote}/3 {split} availability"
            ) from error
        sampled = selected[(finding, vote)]
        if population < sampled or sampled <= 0:
            raise ValueError(
                f"invalid sampling counts for {finding} {vote}/3 {split}: "
                f"population={population}, sampled={sampled}"
            )
        row = dict(source)
        row["population_weight"] = float(population / sampled)
        output.append(row)
    # Normalize for numerically stable optimization.  Weighted means are
    # unchanged; retain the raw factor separately for auditability.
    mean_weight = float(np.mean([row["population_weight"] for row in output]))
    for row in output:
        row["population_weight_raw"] = row["population_weight"]
        row["population_weight"] /= mean_weight
    return output


def weight_audit(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    weights = _weights(rows)
    raw = np.asarray(
        [float(row.get("population_weight_raw", weight)) for row, weight in zip(rows, weights)]
    )
    return {
        "normalized_min": float(weights.min()),
        "normalized_max": float(weights.max()),
        "raw_inverse_sampling_probability_min": float(raw.min()),
        "raw_inverse_sampling_probability_max": float(raw.max()),
        "kish_effective_sample_size": float(weights.sum() ** 2 / np.sum(weights**2)),
        "population_rows_represented": float(raw.sum()),
    }


def _weights(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    values = np.asarray([float(row.get("population_weight", 1.0)) for row in rows])
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("population weights must be finite and positive")
    return values


def _standardization(
    values: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float]:
    weights = np.ones(len(values), dtype=np.float64) if weights is None else weights
    center = float(np.average(values, weights=weights))
    scale = float(np.sqrt(np.average((values - center) ** 2, weights=weights)))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("development feature has zero variance")
    return center, scale


def _weighted_quantile(values: np.ndarray, quantiles: Sequence[float], weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ordered = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= ordered_weights.sum()
    return np.interp(np.asarray(quantiles), cumulative, ordered)


def fit_reader_logistic(
    rows: Sequence[dict[str, Any]],
    reader_panel: Sequence[str],
    *,
    include_reader_effects: bool,
    include_maybe_margin: bool,
    flexible_score: bool = False,
    l2: float = 1e-4,
    max_steps: int = 300,
) -> dict[str, Any]:
    """Fit one convex shared-slope logistic model on development votes."""

    if not rows or l2 < 0 or max_steps <= 0:
        raise ValueError("reader fitting requires rows, nonnegative l2, and positive steps")
    findings = sorted({str(row["finding"]) for row in rows})
    finding_index = {value: index for index, value in enumerate(findings)}
    item_weights = _weights(rows)
    score = np.asarray([row["signed_score"] for row in rows], dtype=np.float64)
    maybe = np.asarray([row["maybe_margin"] for row in rows], dtype=np.float64)
    row_findings = np.asarray([str(row["finding"]) for row in rows])
    score_center: dict[str, float] = {}
    score_scale: dict[str, float] = {}
    maybe_center: dict[str, float] = {}
    maybe_scale: dict[str, float] = {}
    knots: dict[str, list[float]] = {}
    score_z = np.empty(len(rows), dtype=np.float64)
    maybe_z = np.zeros(len(rows), dtype=np.float64)
    for finding in findings:
        mask = row_findings == finding
        center, scale = _standardization(score[mask], item_weights[mask])
        score_center[finding], score_scale[finding] = center, scale
        score_z[mask] = (score[mask] - center) / scale
        if include_maybe_margin:
            m_center, m_scale = _standardization(maybe[mask], item_weights[mask])
            maybe_center[finding], maybe_scale[finding] = m_center, m_scale
            maybe_z[mask] = (maybe[mask] - m_center) / m_scale
        else:
            maybe_center[finding], maybe_scale[finding] = 0.0, 1.0
        # Retain all three predeclared knots.  Duplicate quantiles are allowed:
        # the weak L2 penalty gives a unique prediction without data-driven
        # basis deletion or a post-hoc change of model capacity.
        knots[finding] = (
            [
                float(value)
                for value in _weighted_quantile(
                    score_z[mask], [0.25, 0.5, 0.75], item_weights[mask]
                )
            ]
            if flexible_score
            else []
        )
    score_dimensions = 4 if flexible_score else 1
    score_basis = np.empty((len(rows), score_dimensions), dtype=np.float64)
    for index, finding in enumerate(row_findings):
        score_basis[index] = [
            score_z[index],
            *[max(score_z[index] - knot, 0.0) for knot in knots[finding]],
        ]

    item_ids = np.repeat(np.arange(len(rows)), 3)
    finding_ids = np.repeat(
        np.asarray([finding_index[str(row["finding"])] for row in rows]), 3
    )
    reader_ids = np.tile(np.arange(3), len(rows))
    targets = np.concatenate([np.asarray(row["reader_votes"]) for row in rows])
    score_tensor = torch.tensor(score_basis[item_ids], dtype=torch.float64)
    maybe_tensor = torch.tensor(maybe_z[item_ids], dtype=torch.float64)
    finding_tensor = torch.tensor(finding_ids, dtype=torch.long)
    reader_tensor = torch.tensor(reader_ids, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.float64)
    weight_tensor = torch.tensor(item_weights[item_ids], dtype=torch.float64)

    size = len(findings) * score_dimensions + 2 * len(findings) * int(include_maybe_margin) + len(findings) + (
        len(reader_panel) if include_reader_effects else 0
    )
    parameters = torch.zeros(size, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [parameters], lr=1.0, max_iter=max_steps, tolerance_grad=1e-10,
        tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def unpack() -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor
    ]:
        cursor = 0
        slope = parameters[cursor : cursor + len(findings) * score_dimensions].reshape(
            len(findings), score_dimensions
        )
        cursor += len(findings) * score_dimensions
        maybe_slope = (
            parameters[cursor : cursor + len(findings)]
            if include_maybe_margin else None
        )
        cursor += len(findings) * int(include_maybe_margin)
        interaction = (
            parameters[cursor : cursor + len(findings)]
            if include_maybe_margin else None
        )
        cursor += len(findings) * int(include_maybe_margin)
        finding_effect = parameters[cursor : cursor + len(findings)]
        cursor += len(findings)
        if include_reader_effects:
            raw_reader = parameters[cursor : cursor + len(reader_panel)]
            reader_effect = raw_reader - raw_reader.mean()
        else:
            reader_effect = torch.zeros(len(reader_panel), dtype=torch.float64)
        return slope, maybe_slope, interaction, finding_effect, reader_effect

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        slope, maybe_slope, interaction, finding_effect, reader_effect = unpack()
        logits = (score_tensor * slope[finding_tensor]).sum(dim=1) + finding_effect[finding_tensor]
        if maybe_slope is not None:
            logits = (
                logits + maybe_slope[finding_tensor] * maybe_tensor
                + interaction[finding_tensor] * score_tensor[:, 0] * maybe_tensor
            )
        logits = logits + reader_effect[reader_tensor]
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target_tensor, reduction="none"
        )
        likelihood = (losses * weight_tensor).sum() / weight_tensor.sum()
        # Reader effects are centered, so penalizing the raw vector also fixes
        # its otherwise irrelevant common offset.
        penalty = l2 * parameters.square().mean()
        loss = likelihood + penalty
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        slope, maybe_slope, interaction, finding_effect, reader_effect = unpack()
        likelihood_logits = (
            score_tensor * slope[finding_tensor]
        ).sum(dim=1) + finding_effect[finding_tensor]
        if maybe_slope is not None:
            likelihood_logits = (
                likelihood_logits + maybe_slope[finding_tensor] * maybe_tensor
                + interaction[finding_tensor] * score_tensor[:, 0] * maybe_tensor
            )
        likelihood_logits = likelihood_logits + reader_effect[reader_tensor]
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            likelihood_logits, target_tensor, reduction="none"
        )
        binary_nll = (losses * weight_tensor).sum() / weight_tensor.sum()
        fitted = {
            "score_slope_standardized_by_finding": {
                finding: float(slope[index, 0])
                for finding, index in finding_index.items()
            },
            "score_spline_coefficients_by_finding": {
                finding: [float(value) for value in slope[index]]
                for finding, index in finding_index.items()
            },
            "score_slope_raw_by_finding": {
                finding: float(slope[index, 0]) / score_scale[finding]
                for finding, index in finding_index.items()
            },
            "score_spline_knots_standardized_by_finding": knots,
            "flexible_score": flexible_score,
            "maybe_margin_slope_standardized_by_finding": (
                {
                    finding: float(maybe_slope[index])
                    for finding, index in finding_index.items()
                }
                if maybe_slope is not None else None
            ),
            "score_by_maybe_interaction_by_finding": (
                {
                    finding: float(interaction[index])
                    for finding, index in finding_index.items()
                }
                if interaction is not None else None
            ),
            "finding_effects": {
                finding: float(finding_effect[index])
                for finding, index in finding_index.items()
            },
            "reader_effects": {
                reader: float(reader_effect[index])
                for index, reader in enumerate(reader_panel)
            },
            "score_center_by_finding": score_center,
            "score_scale_by_finding": score_scale,
            "maybe_margin_center_by_finding": maybe_center,
            "maybe_margin_scale_by_finding": maybe_scale,
            "include_reader_effects": include_reader_effects,
            "include_maybe_margin": include_maybe_margin,
            "l2": l2,
            "fit_weighting": "inverse_sampling_probability_population",
            "development_binary_nll": float(binary_nll),
        }
        if not flexible_score and not include_maybe_margin:
            fitted["raw_score_thresholds_at_reader_p_half"] = {
                finding: {
                    reader: (
                        score_center[finding]
                        - score_scale[finding]
                    * (
                        fitted["finding_effects"][finding]
                        + fitted["reader_effects"][reader]
                    )
                        / float(slope[finding_index[finding], 0])
                        if abs(float(slope[finding_index[finding], 0])) > 1e-10
                        else None
                    )
                    for reader in reader_panel
                }
                for finding in findings
            }
    return fitted


def predict_reader_model(
    model: dict[str, Any], rows: Sequence[dict[str, Any]], reader_panel: Sequence[str]
) -> np.ndarray:
    findings = set(model["finding_effects"])
    unseen = sorted({str(row["finding"]) for row in rows} - findings)
    if unseen:
        raise ValueError(f"confirmation contains findings unseen on dev: {unseen}")
    output = []
    for row in rows:
        finding = str(row["finding"])
        score_z = (
            float(row["signed_score"]) - model["score_center_by_finding"][finding]
        ) / model["score_scale_by_finding"][finding]
        score_basis = np.asarray(
            [
                score_z,
                *[
                    max(score_z - knot, 0.0)
                    for knot in model["score_spline_knots_standardized_by_finding"][finding]
                ],
            ],
            dtype=np.float64,
        )
        maybe_z = (
            float(row["maybe_margin"])
            - model["maybe_margin_center_by_finding"][finding]
        ) / model["maybe_margin_scale_by_finding"][finding]
        reader_probabilities = []
        for reader in reader_panel:
            value = (
                float(
                    np.dot(
                        model["score_spline_coefficients_by_finding"][finding],
                        score_basis,
                    )
                )
                + model["finding_effects"][finding]
                + model["reader_effects"][reader]
            )
            if model["include_maybe_margin"]:
                value += (
                    model["maybe_margin_slope_standardized_by_finding"][finding]
                    * maybe_z
                    + model["score_by_maybe_interaction_by_finding"][finding]
                    * score_z
                    * maybe_z
                )
            reader_probabilities.append(sigmoid(float(value)))
        output.append(poisson_binomial_three_state(reader_probabilities))
    return np.asarray(output)


def fit_multinomial_e_only(
    rows: Sequence[dict[str, Any]], *, l2: float = 1e-4, max_steps: int = 300
) -> dict[str, Any]:
    """M3: unconstrained three-state model with the M1 evidence spline."""

    findings = sorted({str(row["finding"]) for row in rows})
    finding_index = {value: index for index, value in enumerate(findings)}
    weights = _weights(rows)
    score = np.asarray([row["signed_score"] for row in rows], dtype=np.float64)
    row_findings = np.asarray([str(row["finding"]) for row in rows])
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    knots: dict[str, list[float]] = {}
    design = np.zeros((len(rows), len(findings) * 5), dtype=np.float64)
    for finding in findings:
        mask = row_findings == finding
        center, scale = _standardization(score[mask], weights[mask])
        centers[finding], scales[finding] = center, scale
        standardized = (score[mask] - center) / scale
        finding_knots = [
            float(value)
            for value in _weighted_quantile(
                standardized, [0.25, 0.5, 0.75], weights[mask]
            )
        ]
        knots[finding] = finding_knots
        block = np.column_stack(
            [
                standardized,
                *[
                    np.maximum(standardized - knot, 0.0)
                    for knot in finding_knots
                ],
                np.ones(mask.sum()),
            ]
        )
        start = finding_index[finding] * 5
        design[np.flatnonzero(mask), start : start + 5] = block
    x = torch.tensor(design, dtype=torch.float64)
    y = torch.tensor(targets(rows), dtype=torch.long)
    w = torch.tensor(weights, dtype=torch.float64)
    parameters = torch.zeros((design.shape[1], 3), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [parameters], lr=1.0, max_iter=max_steps, tolerance_grad=1e-10,
        tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x @ parameters
        losses = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        likelihood = (losses * w).sum() / w.sum()
        loss = likelihood + l2 * parameters.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        coefficients = parameters - parameters.mean(dim=1, keepdim=True)
        probabilities = torch.softmax(x @ coefficients, dim=1)
        losses = torch.nn.functional.nll_loss(
            torch.log(probabilities.clamp_min(1e-12)), y, reduction="none"
        )
        development_nll = float((losses * w).sum() / w.sum())
    return {
        "score_center_by_finding": centers,
        "score_scale_by_finding": scales,
        "score_spline_knots_standardized_by_finding": knots,
        "findings": findings,
        "coefficients": coefficients.cpu().numpy().tolist(),
        "development_multiclass_nll": development_nll,
        "fit_weighting": "inverse_sampling_probability_population",
        "l2": l2,
    }


def predict_multinomial_e_only(
    model: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> np.ndarray:
    findings = list(model["findings"])
    finding_index = {value: index for index, value in enumerate(findings)}
    output = []
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    for row in rows:
        finding = str(row["finding"])
        if finding not in finding_index:
            raise ValueError(f"confirmation finding unseen on dev: {finding}")
        score = (
            float(row["signed_score"])
            - model["score_center_by_finding"][finding]
        ) / model["score_scale_by_finding"][finding]
        spline = [
            score,
            *[
                max(score - knot, 0.0)
                for knot in model["score_spline_knots_standardized_by_finding"][finding]
            ],
            1.0,
        ]
        design = np.zeros(len(findings) * 5, dtype=np.float64)
        start = finding_index[finding] * 5
        design[start : start + 5] = spline
        output.append(softmax(design @ coefficients))
    return np.asarray(output)


def fit_finding_prior(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Population-weighted dev empirical state distribution for each finding."""

    target = targets(rows)
    weights = _weights(rows)
    probabilities = {}
    for finding in sorted({str(row["finding"]) for row in rows}):
        mask = np.asarray([str(row["finding"]) == finding for row in rows])
        counts = np.asarray(
            [weights[mask & (target == state)].sum() for state in range(3)],
            dtype=np.float64,
        )
        if np.any(counts <= 0):
            raise ValueError(f"finding prior has an empty state for {finding}")
        probabilities[finding] = (counts / counts.sum()).tolist()
    return {
        "probabilities_by_finding": probabilities,
        "fit_weighting": "inverse_sampling_probability_population",
        "smoothing": "none; complete frozen dev contains every vote state",
    }


def predict_finding_prior(
    model: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> np.ndarray:
    return np.asarray(
        [model["probabilities_by_finding"][str(row["finding"])] for row in rows],
        dtype=np.float64,
    )


def fit_multinomial_em_finding(
    rows: Sequence[dict[str, Any]], *, l2: float = 1e-4, max_steps: int = 300
) -> dict[str, Any]:
    """Strong unconstrained calibration on shift-invariant (e, m) by finding.

    The e basis matches M1/M2 (linear plus three dev-quantile hinges); m and
    e*m are added before an unconstrained three-state softmax.  Thus a VRP gain
    cannot be attributed merely to comparing against scalar temperature.
    """

    findings = sorted({str(row["finding"]) for row in rows})
    finding_index = {value: index for index, value in enumerate(findings)}
    weights = _weights(rows)
    score = np.asarray([row["signed_score"] for row in rows], dtype=np.float64)
    maybe = np.asarray([row["maybe_margin"] for row in rows], dtype=np.float64)
    row_findings = np.asarray([str(row["finding"]) for row in rows])
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    maybe_centers: dict[str, float] = {}
    maybe_scales: dict[str, float] = {}
    knots: dict[str, list[float]] = {}
    # Per-finding block: e, three hinges, m, e*m, intercept.
    design = np.zeros((len(rows), len(findings) * 7), dtype=np.float64)
    for finding in findings:
        mask = row_findings == finding
        e_center, e_scale = _standardization(score[mask], weights[mask])
        m_center, m_scale = _standardization(maybe[mask], weights[mask])
        centers[finding], scales[finding] = e_center, e_scale
        maybe_centers[finding], maybe_scales[finding] = m_center, m_scale
        e = (score[mask] - e_center) / e_scale
        m = (maybe[mask] - m_center) / m_scale
        finding_knots = [
            float(value)
            for value in _weighted_quantile(e, [0.25, 0.5, 0.75], weights[mask])
        ]
        knots[finding] = finding_knots
        block = np.column_stack(
            [
                e,
                *[np.maximum(e - knot, 0.0) for knot in finding_knots],
                m,
                e * m,
                np.ones(mask.sum()),
            ]
        )
        start = finding_index[finding] * 7
        design[np.flatnonzero(mask), start : start + 7] = block
    x = torch.tensor(design, dtype=torch.float64)
    y = torch.tensor(targets(rows), dtype=torch.long)
    w = torch.tensor(weights, dtype=torch.float64)
    parameters = torch.zeros((design.shape[1], 3), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [parameters], lr=1.0, max_iter=max_steps, tolerance_grad=1e-10,
        tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        losses = torch.nn.functional.cross_entropy(x @ parameters, y, reduction="none")
        loss = (losses * w).sum() / w.sum() + l2 * parameters.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        coefficients = parameters - parameters.mean(dim=1, keepdim=True)
    return {
        "findings": findings,
        "score_center_by_finding": centers,
        "score_scale_by_finding": scales,
        "maybe_margin_center_by_finding": maybe_centers,
        "maybe_margin_scale_by_finding": maybe_scales,
        "score_spline_knots_standardized_by_finding": knots,
        "coefficients": coefficients.cpu().numpy().tolist(),
        "fit_weighting": "inverse_sampling_probability_population",
        "l2": l2,
    }


def predict_multinomial_em_finding(
    model: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> np.ndarray:
    findings = list(model["findings"])
    finding_index = {value: index for index, value in enumerate(findings)}
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    output = []
    for row in rows:
        finding = str(row["finding"])
        if finding not in finding_index:
            raise ValueError(f"confirmation finding unseen on dev: {finding}")
        e = (
            float(row["signed_score"]) - model["score_center_by_finding"][finding]
        ) / model["score_scale_by_finding"][finding]
        m = (
            float(row["maybe_margin"])
            - model["maybe_margin_center_by_finding"][finding]
        ) / model["maybe_margin_scale_by_finding"][finding]
        block = [
            e,
            *[
                max(e - knot, 0.0)
                for knot in model["score_spline_knots_standardized_by_finding"][finding]
            ],
            m,
            e * m,
            1.0,
        ]
        design = np.zeros(len(findings) * 7, dtype=np.float64)
        start = finding_index[finding] * 7
        design[start : start + 7] = block
        output.append(softmax(design @ coefficients))
    return np.asarray(output)


def direct_probabilities(rows: Sequence[dict[str, Any]], temperature: float = 1.0) -> np.ndarray:
    return np.asarray([softmax(row["logits"], temperature) for row in rows])


def targets(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [0 if row["positive_votes"] == 0 else 2 if row["positive_votes"] == 3 else 1 for row in rows],
        dtype=np.int64,
    )


def brier_per_row(probabilities: np.ndarray, target: np.ndarray) -> np.ndarray:
    one_hot = np.eye(3, dtype=np.float64)[target]
    return np.sum((probabilities - one_hot) ** 2, axis=1)


def nll_per_row(probabilities: np.ndarray, target: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probabilities[np.arange(len(target)), target], 1e-12, 1.0))


def _golden_section(function, low: float, high: float, steps: int = 100) -> float:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    f_left, f_right = function(left), function(right)
    for _ in range(steps):
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - ratio * (high - low)
            f_left = function(left)
        else:
            low, left, f_left = left, right, f_right
            right = low + ratio * (high - low)
            f_right = function(right)
    return float((low + high) / 2.0)


def fit_temperature(rows: Sequence[dict[str, Any]]) -> float:
    target = targets(rows)
    weights = _weights(rows)

    def objective(log_temperature: float) -> float:
        probability = direct_probabilities(rows, math.exp(log_temperature))
        return float(np.average(nll_per_row(probability, target), weights=weights))

    return math.exp(_golden_section(objective, math.log(0.05), math.log(20.0)))


def metric_summary(
    probabilities: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    return {
        "brier": float(np.average(brier_per_row(probabilities, target), weights=weights)),
        "nll": float(np.average(nll_per_row(probabilities, target), weights=weights)),
        "accuracy": float(np.average(probabilities.argmax(axis=1) == target, weights=weights)),
    }


def _group_metric_bootstrap(
    values: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    unique = np.unique(groups)
    masks = [groups == value for value in unique]
    sums = np.asarray([(values[mask] * weights[mask]).sum() for mask in masks], dtype=np.float64)
    counts = np.asarray([weights[mask].sum() for mask in masks], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        chosen = rng.integers(0, len(unique), size=len(unique))
        estimates.append(float(sums[chosen].sum() / counts[chosen].sum()))
    return {
        "estimate": float(np.average(values, weights=weights)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "valid_draws": draws,
    }


def _paired_relative_brier_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    unique = np.unique(groups)
    masks = [groups == value for value in unique]
    base_sum = np.asarray([(baseline[mask] * weights[mask]).sum() for mask in masks])
    candidate_sum = np.asarray([(candidate[mask] * weights[mask]).sum() for mask in masks])
    counts = np.asarray([weights[mask].sum() for mask in masks])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        chosen = rng.integers(0, len(unique), size=len(unique))
        base_mean = base_sum[chosen].sum() / counts[chosen].sum()
        candidate_mean = candidate_sum[chosen].sum() / counts[chosen].sum()
        estimates.append(float((base_mean - candidate_mean) / max(base_mean, 1e-12)))
    baseline_mean = float(np.average(baseline, weights=weights))
    candidate_mean = float(np.average(candidate, weights=weights))
    estimate = (baseline_mean - candidate_mean) / max(
        baseline_mean, 1e-12
    )
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "valid_draws": draws,
    }


def _paired_difference_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    return _group_metric_bootstrap(left - right, groups, weights, draws, seed)


def _directional_slope(rows: Sequence[dict[str, Any]]) -> float:
    x = np.asarray([row["signed_score"] for row in rows], dtype=np.float64)
    y = np.asarray([row["positive_votes"] / 3.0 for row in rows], dtype=np.float64)
    weights = _weights(rows)
    x_center = float(np.average(x, weights=weights))
    y_center = float(np.average(y, weights=weights))
    denominator = float(np.sum(weights * (x - x_center) ** 2))
    return float(
        np.sum(weights * (x - x_center) * (y - y_center))
        / max(denominator, 1e-12)
    )


def directional_cluster_bootstrap(
    rows: Sequence[dict[str, Any]], draws: int, seed: int
) -> dict[str, float | int]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[str(row["image_id"])].append(row)
    images = sorted(by_image)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        sampled = rng.choice(images, size=len(images), replace=True)
        batch = [row for image in sampled for row in by_image[str(image)]]
        if np.std([row["signed_score"] for row in batch]) < 1e-10:
            continue
        estimates.append(_directional_slope(batch))
    if not estimates:
        raise RuntimeError("no valid directional bootstrap draws")
    return {
        "estimate": _directional_slope(rows),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "valid_draws": len(estimates),
    }


def analyze(
    dev: Sequence[dict[str, Any]],
    confirmation: Sequence[dict[str, Any]],
    reader_panel: Sequence[str],
    *,
    bootstrap_draws: int,
    seed: int,
    relative_brier_target: float,
    maybe_equivalence_relative_bound: float,
    maybe_nll_equivalence_bound: float,
    poisson_binomial_excess_brier_bound: float,
    direct_maybe_effect_bound: float,
    l2: float,
) -> dict[str, Any]:
    if set(row["image_id"] for row in dev) & set(row["image_id"] for row in confirmation):
        raise ValueError("development and confirmation images overlap")
    if {row["finding"] for row in dev} != {row["finding"] for row in confirmation}:
        raise ValueError("development and confirmation finding sets differ")
    if bootstrap_draws <= 0 or relative_brier_target <= 0:
        raise ValueError("bootstrap draws and Brier target must be positive")
    if maybe_equivalence_relative_bound <= 0:
        raise ValueError("Maybe equivalence bound must be positive")

    no_reader = fit_reader_logistic(
        dev, reader_panel, include_reader_effects=False,
        include_maybe_margin=False, l2=l2,
    )
    m0 = fit_reader_logistic(
        dev, reader_panel, include_reader_effects=True,
        include_maybe_margin=False, l2=l2,
    )
    m1 = fit_reader_logistic(
        dev, reader_panel, include_reader_effects=True,
        include_maybe_margin=False, flexible_score=True, l2=l2,
    )
    m2 = fit_reader_logistic(
        dev, reader_panel, include_reader_effects=True,
        include_maybe_margin=True, flexible_score=True, l2=l2,
    )
    m3 = fit_multinomial_e_only(dev, l2=l2)
    finding_prior = fit_finding_prior(dev)
    strong_calibrator = fit_multinomial_em_finding(dev, l2=l2)
    temperature = fit_temperature(dev)
    methods = {
        "dev_finding_only_empirical_prior": lambda rows: predict_finding_prior(finding_prior, rows),
        "direct_yes_maybe_no": lambda rows: direct_probabilities(rows),
        "dev_temperature_scaling": lambda rows: direct_probabilities(rows, temperature),
        "no_reader_fe_linear_threshold": lambda rows: predict_reader_model(no_reader, rows, reader_panel),
        "M0_linear_e_reader_finding_threshold": lambda rows: predict_reader_model(m0, rows, reader_panel),
        "M1_flexible_e_virtual_reader_panel": lambda rows: predict_reader_model(m1, rows, reader_panel),
        "M2_flexible_e_maybe_interaction_panel": lambda rows: predict_reader_model(m2, rows, reader_panel),
        "M3_unconstrained_e_only_multinomial": lambda rows: predict_multinomial_e_only(m3, rows),
        "strong_em_finding_multinomial_calibration": lambda rows: predict_multinomial_em_finding(strong_calibrator, rows),
    }
    split_predictions = {
        split: {name: function(rows) for name, function in methods.items()}
        for split, rows in (("dev", dev), ("confirmation", confirmation))
    }
    split_rows = {"dev": dev, "confirmation": confirmation}
    metrics: dict[str, Any] = {}
    for split, predictions in split_predictions.items():
        target = targets(split_rows[split])
        groups = np.asarray([row["image_id"] for row in split_rows[split]])
        population_weights = _weights(split_rows[split])
        balanced_weights = np.ones(len(target), dtype=np.float64)
        metrics[split] = {}
        for index, (name, probability) in enumerate(predictions.items()):
            point = {
                "population_weighted": metric_summary(
                    probability, target, population_weights
                ),
                "balanced_design_mechanism_only": metric_summary(
                    probability, target, balanced_weights
                ),
            }
            if split == "confirmation":
                point["population_weighted"]["brier_image_cluster_bootstrap"] = _group_metric_bootstrap(
                    brier_per_row(probability, target), groups, population_weights, bootstrap_draws,
                    seed + 10 * index,
                )
                point["population_weighted"]["nll_image_cluster_bootstrap"] = _group_metric_bootstrap(
                    nll_per_row(probability, target), groups, population_weights, bootstrap_draws,
                    seed + 10 * index + 1,
                )
            metrics[split][name] = point

    confirmation_target = targets(confirmation)
    groups = np.asarray([row["image_id"] for row in confirmation])
    weights = _weights(confirmation)
    confirmation_predictions = split_predictions["confirmation"]
    temperature_brier = brier_per_row(
        confirmation_predictions["dev_temperature_scaling"], confirmation_target
    )
    m1_brier = brier_per_row(
        confirmation_predictions["M1_flexible_e_virtual_reader_panel"], confirmation_target
    )
    m2_brier = brier_per_row(
        confirmation_predictions["M2_flexible_e_maybe_interaction_panel"],
        confirmation_target,
    )
    m3_brier = brier_per_row(
        confirmation_predictions["M3_unconstrained_e_only_multinomial"],
        confirmation_target,
    )
    prior_brier = brier_per_row(
        confirmation_predictions["dev_finding_only_empirical_prior"], confirmation_target
    )
    m0_brier = brier_per_row(
        confirmation_predictions["M0_linear_e_reader_finding_threshold"], confirmation_target
    )
    strong_brier = brier_per_row(
        confirmation_predictions["strong_em_finding_multinomial_calibration"],
        confirmation_target,
    )
    panel_vs_temperature = _paired_relative_brier_bootstrap(
        temperature_brier, m1_brier, groups, weights, bootstrap_draws, seed + 1000
    )
    m0_vs_prior = _paired_relative_brier_bootstrap(
        prior_brier, m0_brier, groups, weights, bootstrap_draws, seed + 1050
    )
    raw_m0_negative_excess = _paired_relative_brier_bootstrap(
        m1_brier, m0_brier, groups, weights, bootstrap_draws, seed + 1075
    )
    m0_excess_over_m1 = {
        "estimate": -float(raw_m0_negative_excess["estimate"]),
        "ci_low": -float(raw_m0_negative_excess["ci_high"]),
        "ci_high": -float(raw_m0_negative_excess["ci_low"]),
        "valid_draws": raw_m0_negative_excess["valid_draws"],
    }
    m1_vs_strong_calibration = _paired_relative_brier_bootstrap(
        strong_brier, m1_brier, groups, weights, bootstrap_draws, seed + 1090
    )
    # Positive is M2 improvement over M1.  Redundancy requires its full CI to
    # remain inside a predeclared practical-equivalence interval.
    maybe_relative_brier_improvement = _paired_relative_brier_bootstrap(
        m1_brier, m2_brier, groups, weights, bootstrap_draws, seed + 2000
    )
    m1_nll = nll_per_row(
        confirmation_predictions["M1_flexible_e_virtual_reader_panel"], confirmation_target
    )
    m2_nll = nll_per_row(
        confirmation_predictions["M2_flexible_e_maybe_interaction_panel"], confirmation_target
    )
    maybe_nll_improvement = _paired_difference_bootstrap(
        m1_nll, m2_nll, groups, weights, bootstrap_draws, seed + 2100
    )
    raw_negative_excess = _paired_relative_brier_bootstrap(
        m3_brier, m1_brier, groups, weights, bootstrap_draws, seed + 2200
    )
    m1_excess_over_m3 = {
        "estimate": -float(raw_negative_excess["estimate"]),
        "ci_low": -float(raw_negative_excess["ci_high"]),
        "ci_high": -float(raw_negative_excess["ci_low"]),
        "valid_draws": raw_negative_excess["valid_draws"],
    }
    maybe_z = np.asarray(
        [
            (
                row["maybe_margin"]
                - m2["maybe_margin_center_by_finding"][str(row["finding"])]
            )
            / m2["maybe_margin_scale_by_finding"][str(row["finding"])]
            for row in confirmation
        ]
    )
    observed_middle = (confirmation_target == 1).astype(np.float64)
    direct_maybe_values = maybe_z * (
        observed_middle
        - confirmation_predictions["M1_flexible_e_virtual_reader_panel"][:, 1]
    )
    direct_maybe_effect = _group_metric_bootstrap(
        direct_maybe_values, groups, weights, bootstrap_draws, seed + 2300
    )
    direction = directional_cluster_bootstrap(confirmation, bootstrap_draws, seed + 3000)
    direction_by_finding = {
        finding: directional_cluster_bootstrap(
            [row for row in confirmation if str(row["finding"]) == finding],
            bootstrap_draws,
            seed + 3100 + index,
        )
        for index, finding in enumerate(sorted({str(row["finding"]) for row in confirmation}))
    }
    score_direction_correct = (
        all(
            float(value) > 0
            for value in m0["score_slope_standardized_by_finding"].values()
        )
        and float(direction["ci_low"]) > 0
        and all(float(value["ci_low"]) > 0 for value in direction_by_finding.values())
    )
    direction_count = {
        "positive_point_estimate": int(
            sum(float(value["estimate"]) > 0 for value in direction_by_finding.values())
        ),
        "ci_strictly_above_zero": int(
            sum(float(value["ci_low"]) > 0 for value in direction_by_finding.values())
        ),
        "total_findings": len(direction_by_finding),
    }
    maybe_brier_equivalent = (
        float(maybe_relative_brier_improvement["ci_low"])
        > -maybe_equivalence_relative_bound
        and float(maybe_relative_brier_improvement["ci_high"])
        < maybe_equivalence_relative_bound
    )
    maybe_nll_equivalent = (
        float(maybe_nll_improvement["ci_low"]) > -maybe_nll_equivalence_bound
        and float(maybe_nll_improvement["ci_high"]) < maybe_nll_equivalence_bound
    )
    direct_maybe_negligible = (
        float(direct_maybe_effect["ci_low"]) > -direct_maybe_effect_bound
        and float(direct_maybe_effect["ci_high"]) < direct_maybe_effect_bound
    )
    gate = {
        "score_direction_correct": score_direction_correct,
        "score_direction_definition": (
            "all unconstrained dev M0 finding-specific slopes are positive and "
            "every confirmation finding has positive support-on-(Yes-No score) "
            "direction (pooled and per-finding image-cluster bootstrap)"
        ),
        "M1_relative_temperature_brier_improvement_at_least_target": (
            float(panel_vs_temperature["estimate"]) >= relative_brier_target
            and float(panel_vs_temperature["ci_low"]) > 0
        ),
        "relative_brier_target": relative_brier_target,
        "M0_relative_finding_prior_brier_improvement_at_least_target": (
            float(m0_vs_prior["estimate"]) >= relative_brier_target
            and float(m0_vs_prior["ci_low"]) > 0
        ),
        "M0_excess_brier_over_M1_below_one_percent": (
            float(m0_excess_over_m1["ci_high"]) < 0.01
        ),
        "M1_beats_strong_em_finding_calibration": (
            float(m1_vs_strong_calibration["estimate"]) > 0
            and float(m1_vs_strong_calibration["ci_low"]) > 0
        ),
        "M2_vs_M1_brier_equivalent": maybe_brier_equivalent,
        "M2_vs_M1_nll_equivalent": maybe_nll_equivalent,
        "maybe_equivalence_relative_bound": maybe_equivalence_relative_bound,
        "maybe_nll_equivalence_bound": maybe_nll_equivalence_bound,
        "M1_poisson_binomial_excess_brier_below_bound": (
            float(m1_excess_over_m3["ci_high"]) < poisson_binomial_excess_brier_bound
        ),
        "poisson_binomial_excess_brier_bound": poisson_binomial_excess_brier_bound,
        "direct_maybe_residual_effect_equivalent": direct_maybe_negligible,
        "direct_maybe_effect_bound": direct_maybe_effect_bound,
    }
    gate["virtual_reader_panel_confirmation_passed"] = bool(
        gate["score_direction_correct"]
        and gate["M0_relative_finding_prior_brier_improvement_at_least_target"]
        and gate["M0_excess_brier_over_M1_below_one_percent"]
        and gate["M1_beats_strong_em_finding_calibration"]
        and gate["M1_relative_temperature_brier_improvement_at_least_target"]
        and gate["M2_vs_M1_brier_equivalent"]
        and gate["M2_vs_M1_nll_equivalent"]
        and gate["M1_poisson_binomial_excess_brier_below_bound"]
        and gate["direct_maybe_residual_effect_equivalent"]
    )
    return {
        "version": VERSION,
        "status": "complete",
        "split_contract": {
            "development_role": "all fitting, temperature selection, and feature scaling",
            "confirmation_role": "locked one-shot evaluation only",
            "image_disjoint": True,
            "reader_panel": list(reader_panel),
            "population_weighting": (
                "inverse selection probability from summary_v2 availability; "
                "balanced-design metrics are discrimination diagnostics only"
            ),
        },
        "n": {"dev": len(dev), "confirmation": len(confirmation)},
        "unique_images": {
            "dev": len({row["image_id"] for row in dev}),
            "confirmation": len({row["image_id"] for row in confirmation}),
        },
        "population_weight_audit": {
            "dev": weight_audit(dev),
            "confirmation": weight_audit(confirmation),
        },
        "development_fits": {
            "temperature": temperature,
            "no_reader_fe_linear_threshold": no_reader,
            "M0_linear_e_reader_finding_threshold": m0,
            "M1_flexible_e_virtual_reader_panel": m1,
            "M2_flexible_e_maybe_interaction_panel": m2,
            "M3_unconstrained_e_only_multinomial": m3,
            "dev_finding_only_empirical_prior": finding_prior,
            "strong_em_finding_multinomial_calibration": strong_calibrator,
        },
        "metrics": metrics,
        "confirmation_comparisons": {
            "M1_relative_brier_improvement_over_temperature": panel_vs_temperature,
            "M0_relative_brier_improvement_over_finding_prior": m0_vs_prior,
            "M0_excess_brier_over_M1": m0_excess_over_m1,
            "M1_relative_brier_improvement_over_strong_em_calibration": m1_vs_strong_calibration,
            "M2_relative_brier_improvement_over_M1": maybe_relative_brier_improvement,
            "M2_nll_improvement_over_M1": maybe_nll_improvement,
            "M1_poisson_binomial_excess_brier_over_M3": m1_excess_over_m3,
            "direct_maybe_residual_effect": {
                **direct_maybe_effect,
                "definition": (
                    "population-weighted mean standardized-m times "
                    "(observed-disagreement - M1 predicted-disagreement)"
                ),
            },
            "score_direction": direction,
            "score_direction_by_finding": direction_by_finding,
            "score_direction_finding_count": direction_count,
        },
        "gate": gate,
        "selected_model": (
            "M1_flexible_e_virtual_reader_panel"
            if gate["virtual_reader_panel_confirmation_passed"]
            else "none"
        ),
        "construct_scope": (
            "A calibrated representation of this exact fixed R8/R9/R10 panel; "
            "not a reader population, clinical truth, or a novel annotator model."
        ),
        "novelty_scope": (
            "Only conditional redundancy of the native Maybe coordinate and a "
            "downstream certainty-only projection may support a contribution."
        ),
        "required_verbalizer_prior_controls_before_claim": [
            "answer option order permutations",
            "matched-token A/B/C labels",
            "Maybe/Uncertain/Cannot-determine paraphrases",
            "content-free prior subtraction",
        ],
        "verbalizer_prior_controls_status": "not_run_by_this_analyzer",
        "paper_claim_authorized": False,
        "authorization": (
            "confirmation gate evaluates the construct; it does not authorize "
            "hallucination mitigation or replace OE/report validation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-features-dir", type=Path, required=True)
    parser.add_argument("--confirmation-features-dir", type=Path, required=True)
    parser.add_argument("--sampling-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reader-panel", nargs=3, default=("R8", "R9", "R10"))
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relative-brier-target", type=float, default=0.05)
    parser.add_argument("--maybe-equivalence-relative-bound", type=float, default=0.01)
    parser.add_argument("--maybe-nll-equivalence-bound", type=float, default=0.005)
    parser.add_argument("--poisson-binomial-excess-brier-bound", type=float, default=0.01)
    parser.add_argument("--direct-maybe-effect-bound", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    args = parser.parse_args()

    panel = tuple(str(value) for value in args.reader_panel)
    if len(set(panel)) != 3:
        raise ValueError("reader panel must contain three distinct IDs")
    sampling_summary = json.loads(args.sampling_summary.read_text(encoding="utf-8"))
    if set(str(value) for value in sampling_summary.get("reader_panel", [])) != set(panel):
        raise ValueError("sampling summary reader panel disagrees with --reader-panel")
    dev = attach_population_weights(
        load_feature_records(args.dev_features_dir, "dev", panel),
        sampling_summary,
        "dev",
    )
    confirmation = attach_population_weights(
        load_feature_records(args.confirmation_features_dir, "confirmation", panel),
        sampling_summary,
        "confirmation",
    )
    result = analyze(
        dev, confirmation, panel,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        relative_brier_target=args.relative_brier_target,
        maybe_equivalence_relative_bound=args.maybe_equivalence_relative_bound,
        maybe_nll_equivalence_bound=args.maybe_nll_equivalence_bound,
        poisson_binomial_excess_brier_bound=args.poisson_binomial_excess_brier_bound,
        direct_maybe_effect_bound=args.direct_maybe_effect_bound,
        l2=args.l2,
    )
    result["provenance"] = {
        "dev_features_dir": str(args.dev_features_dir.resolve()),
        "confirmation_features_dir": str(args.confirmation_features_dir.resolve()),
        "dev_metadata_sha256": sha256_file(args.dev_features_dir / "metadata.jsonl"),
        "confirmation_metadata_sha256": sha256_file(
            args.confirmation_features_dir / "metadata.jsonl"
        ),
        "sampling_summary": str(args.sampling_summary.resolve()),
        "sampling_summary_sha256": sha256_file(args.sampling_summary),
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "command": " ".join(sys.argv),
    }
    result["fingerprint"] = hashlib.sha256(
        json.dumps(result["provenance"], sort_keys=True).encode()
    ).hexdigest()
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "n": result["n"],
                "selected_model": result["selected_model"],
                "gate": result["gate"],
                "confirmation_comparisons": result["confirmation_comparisons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
