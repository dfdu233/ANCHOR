#!/usr/bin/env python3
"""Development-only grouped cross-fit screen for the virtual reader panel.

This screen is deliberately separate from the locked confirmation analyzer.
Every reported development prediction is out-of-fold at image level; its only
decision is whether collecting the 1,920-case confirmation set is justified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from corrected_sgta.fit_virtual_reader_panel_v1 import (
    VERSION as PANEL_VERSION,
    _group_metric_bootstrap,
    _paired_difference_bootstrap,
    _paired_relative_brier_bootstrap,
    _weights,
    attach_population_weights,
    brier_per_row,
    direct_probabilities,
    directional_cluster_bootstrap,
    fit_multinomial_e_only,
    fit_multinomial_em_finding,
    fit_finding_prior,
    fit_reader_logistic,
    fit_temperature,
    load_feature_records,
    metric_summary,
    nll_per_row,
    predict_multinomial_e_only,
    predict_multinomial_em_finding,
    predict_finding_prior,
    predict_reader_model,
    targets,
    weight_audit,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file


VERSION = "virtual-fixed-reader-panel-dev-crossfit-v1"


def image_group_folds(
    rows: Sequence[dict[str, Any]], folds: int, seed: int
) -> np.ndarray:
    images = sorted(
        {str(row["image_id"]) for row in rows},
        key=lambda image: hashlib.sha256(f"{seed}:vrp-dev-fold:{image}".encode()).hexdigest(),
    )
    if folds < 2 or len(images) < folds:
        raise ValueError("grouped cross-fitting requires at least two folds and images per fold")
    assignment = {image: index % folds for index, image in enumerate(images)}
    output = np.asarray([assignment[str(row["image_id"])] for row in rows], dtype=np.int64)
    for image in images:
        if len(set(output[[str(row["image_id"]) == image for row in rows]])) != 1:
            raise AssertionError("one image crossed development folds")
    return output


def crossfit_predictions(
    rows: Sequence[dict[str, Any]],
    reader_panel: Sequence[str],
    folds: int,
    seed: int,
    l2: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], np.ndarray]:
    fold_ids = image_group_folds(rows, folds, seed)
    names = (
        "dev_finding_only_empirical_prior",
        "dev_temperature_scaling",
        "M0_linear_e_reader_finding_threshold",
        "M1_flexible_e_virtual_reader_panel",
        "M2_flexible_e_maybe_interaction_panel",
        "M3_unconstrained_e_only_multinomial",
        "strong_em_finding_multinomial_calibration",
    )
    predictions = {name: np.full((len(rows), 3), np.nan) for name in names}
    audits = []
    all_findings = sorted({str(row["finding"]) for row in rows})
    for fold in range(folds):
        train_indices = np.flatnonzero(fold_ids != fold)
        held_indices = np.flatnonzero(fold_ids == fold)
        train = [rows[index] for index in train_indices]
        held = [rows[index] for index in held_indices]
        if sorted({str(row["finding"]) for row in train}) != all_findings:
            raise ValueError(f"fold {fold} training set omits a finding")
        m0 = fit_reader_logistic(
            train, reader_panel, include_reader_effects=True,
            include_maybe_margin=False, flexible_score=False, l2=l2,
        )
        m1 = fit_reader_logistic(
            train, reader_panel, include_reader_effects=True,
            include_maybe_margin=False, flexible_score=True, l2=l2,
        )
        m2 = fit_reader_logistic(
            train, reader_panel, include_reader_effects=True,
            include_maybe_margin=True, flexible_score=True, l2=l2,
        )
        m3 = fit_multinomial_e_only(train, l2=l2)
        finding_prior = fit_finding_prior(train)
        strong_calibrator = fit_multinomial_em_finding(train, l2=l2)
        temperature = fit_temperature(train)
        fold_predictions = {
            "dev_finding_only_empirical_prior": predict_finding_prior(finding_prior, held),
            "dev_temperature_scaling": direct_probabilities(held, temperature),
            "M0_linear_e_reader_finding_threshold": predict_reader_model(m0, held, reader_panel),
            "M1_flexible_e_virtual_reader_panel": predict_reader_model(m1, held, reader_panel),
            "M2_flexible_e_maybe_interaction_panel": predict_reader_model(m2, held, reader_panel),
            "M3_unconstrained_e_only_multinomial": predict_multinomial_e_only(m3, held),
            "strong_em_finding_multinomial_calibration": predict_multinomial_em_finding(strong_calibrator, held),
        }
        for name, values in fold_predictions.items():
            predictions[name][held_indices] = values
        audits.append(
            {
                "fold": fold,
                "train_n": len(train),
                "held_out_n": len(held),
                "train_unique_images": len({row["image_id"] for row in train}),
                "held_out_unique_images": len({row["image_id"] for row in held}),
                "temperature": temperature,
                "M0_score_slopes_by_finding": m0["score_slope_standardized_by_finding"],
                "M1_first_segment_slopes_by_finding": m1["score_slope_standardized_by_finding"],
            }
        )
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("development cross-fitting left non-finite predictions")
    return predictions, audits, fold_ids


def screen(
    rows: Sequence[dict[str, Any]],
    reader_panel: Sequence[str],
    *,
    folds: int,
    bootstrap_draws: int,
    seed: int,
    l2: float,
) -> dict[str, Any]:
    predictions, fold_audits, fold_ids = crossfit_predictions(
        rows, reader_panel, folds, seed, l2
    )
    target = targets(rows)
    groups = np.asarray([row["image_id"] for row in rows])
    weights = _weights(rows)
    balanced = np.ones(len(rows), dtype=np.float64)
    metrics = {
        name: {
            "population_weighted": metric_summary(probability, target, weights),
            "balanced_design_mechanism_only": metric_summary(
                probability, target, balanced
            ),
        }
        for name, probability in predictions.items()
    }
    losses = {
        name: {
            "brier": brier_per_row(probability, target),
            "nll": nll_per_row(probability, target),
        }
        for name, probability in predictions.items()
    }
    m1_vs_temperature = _paired_relative_brier_bootstrap(
        losses["dev_temperature_scaling"]["brier"],
        losses["M1_flexible_e_virtual_reader_panel"]["brier"],
        groups, weights, bootstrap_draws, seed + 1000,
    )
    m0_vs_prior = _paired_relative_brier_bootstrap(
        losses["dev_finding_only_empirical_prior"]["brier"],
        losses["M0_linear_e_reader_finding_threshold"]["brier"],
        groups, weights, bootstrap_draws, seed + 1050,
    )
    m1_vs_m0 = _paired_relative_brier_bootstrap(
        losses["M0_linear_e_reader_finding_threshold"]["brier"],
        losses["M1_flexible_e_virtual_reader_panel"]["brier"],
        groups, weights, bootstrap_draws, seed + 1100,
    )
    raw_m0_negative_excess = _paired_relative_brier_bootstrap(
        losses["M1_flexible_e_virtual_reader_panel"]["brier"],
        losses["M0_linear_e_reader_finding_threshold"]["brier"],
        groups, weights, bootstrap_draws, seed + 1150,
    )
    m0_excess_m1 = {
        "estimate": -float(raw_m0_negative_excess["estimate"]),
        "ci_low": -float(raw_m0_negative_excess["ci_high"]),
        "ci_high": -float(raw_m0_negative_excess["ci_low"]),
        "valid_draws": raw_m0_negative_excess["valid_draws"],
    }
    m1_vs_strong = _paired_relative_brier_bootstrap(
        losses["strong_em_finding_multinomial_calibration"]["brier"],
        losses["M1_flexible_e_virtual_reader_panel"]["brier"],
        groups, weights, bootstrap_draws, seed + 1175,
    )
    m2_vs_m1 = _paired_relative_brier_bootstrap(
        losses["M1_flexible_e_virtual_reader_panel"]["brier"],
        losses["M2_flexible_e_maybe_interaction_panel"]["brier"],
        groups, weights, bootstrap_draws, seed + 1200,
    )
    m2_vs_m1_nll = _paired_difference_bootstrap(
        losses["M1_flexible_e_virtual_reader_panel"]["nll"],
        losses["M2_flexible_e_maybe_interaction_panel"]["nll"],
        groups, weights, bootstrap_draws, seed + 1300,
    )
    raw_negative_excess = _paired_relative_brier_bootstrap(
        losses["M3_unconstrained_e_only_multinomial"]["brier"],
        losses["M1_flexible_e_virtual_reader_panel"]["brier"],
        groups, weights, bootstrap_draws, seed + 1400,
    )
    m1_excess_m3 = {
        "estimate": -float(raw_negative_excess["estimate"]),
        "ci_low": -float(raw_negative_excess["ci_high"]),
        "ci_high": -float(raw_negative_excess["ci_low"]),
        "valid_draws": raw_negative_excess["valid_draws"],
    }
    direction = directional_cluster_bootstrap(rows, bootstrap_draws, seed + 1500)
    by_finding = {
        finding: directional_cluster_bootstrap(
            [row for row in rows if str(row["finding"]) == finding],
            bootstrap_draws,
            seed + 1600 + index,
        )
        for index, finding in enumerate(sorted({str(row["finding"]) for row in rows}))
    }
    fold_slopes = {
        finding: [
            float(audit["M0_score_slopes_by_finding"][finding])
            for audit in fold_audits
        ]
        for finding in sorted(by_finding)
    }
    direction_ready = (
        float(direction["ci_low"]) > 0
        and all(float(value["estimate"]) > 0 for value in by_finding.values())
        and all(np.median(values) > 0 for values in fold_slopes.values())
    )
    direction_count = {
        "positive_point_estimate": int(
            sum(float(value["estimate"]) > 0 for value in by_finding.values())
        ),
        "ci_strictly_above_zero": int(
            sum(float(value["ci_low"]) > 0 for value in by_finding.values())
        ),
        "total_findings": len(by_finding),
    }
    panel_promising = (
        direction_ready
        and float(m0_vs_prior["estimate"]) >= 0.05
        and float(m0_vs_prior["ci_low"]) > 0
        and float(m0_excess_m1["ci_high"]) < 0.02
        and float(m1_vs_strong["estimate"]) > 0
        and float(m1_vs_temperature["estimate"]) >= 0.05
        and float(m1_vs_temperature["ci_low"]) > 0
        and float(m1_excess_m3["ci_high"]) < 0.02
    )
    maybe_redundancy_promising = (
        float(m2_vs_m1["ci_high"]) < 0.05
        and float(m2_vs_m1_nll["ci_high"]) < 0.02
    )
    return {
        "version": VERSION,
        "panel_analyzer_version": PANEL_VERSION,
        "status": "complete",
        "role": "development-only cost gate; no confirmation data are read",
        "n": len(rows),
        "unique_images": len(set(groups)),
        "folds": folds,
        "fold_assignment_sha256": hashlib.sha256(fold_ids.tobytes()).hexdigest(),
        "fold_audits": fold_audits,
        "population_weight_audit": weight_audit(rows),
        "metrics": metrics,
        "comparisons": {
            "M1_relative_brier_improvement_over_temperature": m1_vs_temperature,
            "M0_relative_brier_improvement_over_finding_prior": m0_vs_prior,
            "M1_relative_brier_improvement_over_M0": m1_vs_m0,
            "M0_excess_brier_over_M1": m0_excess_m1,
            "M1_relative_brier_improvement_over_strong_em_calibration": m1_vs_strong,
            "M2_relative_brier_improvement_over_M1": m2_vs_m1,
            "M2_nll_improvement_over_M1": m2_vs_m1_nll,
            "M1_poisson_binomial_excess_brier_over_M3": m1_excess_m3,
            "score_direction": direction,
            "score_direction_by_finding": by_finding,
            "score_direction_finding_count": direction_count,
            "M0_crossfit_train_slopes_by_finding": fold_slopes,
        },
        "dev_cost_gate": {
            "score_direction_promising": direction_ready,
            "M1_panel_structure_promising": panel_promising,
            "conditional_maybe_redundancy_promising": maybe_redundancy_promising,
            "spend_confirmation_compute": panel_promising and maybe_redundancy_promising,
            "confirmation_thresholds_remain_frozen": {
                "M1_vs_temperature_relative_brier": 0.05,
                "M2_vs_M1_relative_brier_equivalence": 0.01,
                "M2_vs_M1_nll_equivalence": 0.005,
                "M1_vs_M3_excess_brier": 0.01,
            },
        },
        "scope": (
            "Fixed-panel calibration screen only. It is not evidence for a reader "
            "population, clinical truth, hallucination mitigation, or paper novelty."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-features-dir", type=Path, required=True)
    parser.add_argument("--sampling-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reader-panel", nargs=3, default=("R8", "R9", "R10"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l2", type=float, default=1e-4)
    args = parser.parse_args()
    panel = tuple(str(value) for value in args.reader_panel)
    if len(set(panel)) != 3:
        raise ValueError("reader panel must contain three distinct IDs")
    summary = json.loads(args.sampling_summary.read_text(encoding="utf-8"))
    if set(str(value) for value in summary.get("reader_panel", [])) != set(panel):
        raise ValueError("sampling summary reader panel disagrees with --reader-panel")
    rows = attach_population_weights(
        load_feature_records(args.dev_features_dir, "dev", panel), summary, "dev"
    )
    result = screen(
        rows, panel, folds=args.folds, bootstrap_draws=args.bootstrap_draws,
        seed=args.seed, l2=args.l2,
    )
    result["provenance"] = {
        "dev_features_dir": str(args.dev_features_dir.resolve()),
        "dev_metadata_sha256": sha256_file(args.dev_features_dir / "metadata.jsonl"),
        "sampling_summary": str(args.sampling_summary.resolve()),
        "sampling_summary_sha256": sha256_file(args.sampling_summary),
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "bootstrap_draws": args.bootstrap_draws,
        "folds": args.folds,
        "seed": args.seed,
        "command": " ".join(sys.argv),
    }
    atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"], "n": result["n"],
        "dev_cost_gate": result["dev_cost_gate"],
        "comparisons": result["comparisons"],
    }, indent=2))


if __name__ == "__main__":
    main()
