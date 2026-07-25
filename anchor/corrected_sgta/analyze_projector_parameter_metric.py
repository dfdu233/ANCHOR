"""Paired analysis for the LLaVA projector parameter-metric pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from corrected_sgta.cache import iter_successes


VERSION = "sgta-projector-parameter-metric-analysis-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def accuracy(prediction: np.ndarray, truth: np.ndarray) -> dict:
    correct = int(np.sum(prediction == truth))
    return {
        "n": int(len(truth)),
        "correct": correct,
        "accuracy": correct / len(truth) if len(truth) else None,
    }


def paired_flips(
    original: np.ndarray, candidate: np.ndarray, truth: np.ndarray
) -> dict:
    original_correct = original == truth
    candidate_correct = candidate == truth
    return {
        "changed": int(np.sum(original != candidate)),
        "rescues": int(np.sum(~original_correct & candidate_correct)),
        "harmful": int(np.sum(original_correct & ~candidate_correct)),
        "net": int(np.sum(candidate_correct) - np.sum(original_correct)),
    }


def main() -> None:
    args = parse_args()
    metadata = json.loads(
        args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text()
    )
    rows = list(iter_successes(args.cache, metadata["fingerprint"]))
    if not rows:
        raise RuntimeError("no successful records")
    qids = [str(row["qid"]) for row in rows]
    if len(qids) != len(set(qids)):
        raise RuntimeError("duplicate qid in successful cache rows")
    maximum = int(metadata["config"]["max_samples"])
    if maximum and len(rows) > maximum:
        raise RuntimeError(
            f"cache has {len(rows)} rows, exceeding configured max_samples={maximum}"
        )
    expected_arms = metadata["config"]["arms"]
    if any(row["arms"] != expected_arms for row in rows):
        raise RuntimeError("arm order mismatch")
    truth = np.asarray([row["gt_index"] for row in rows], dtype=np.int64)
    nll = np.asarray(
        [row["exact_next_token_nll"] for row in rows], dtype=np.float64
    )
    surface = np.asarray([row["surface_logits"] for row in rows], dtype=np.float64)
    if nll.shape != (len(rows), len(expected_arms), 2):
        raise RuntimeError(f"unexpected exact NLL shape: {nll.shape}")
    if not np.isfinite(nll).all() or not np.isfinite(surface).all():
        raise RuntimeError("non-finite score in cache")
    exact_predictions = np.argmin(nll, axis=-1)
    surface_predictions = np.argmax(surface, axis=-1)
    original = exact_predictions[:, 0]
    point = {}
    for index, arm in enumerate(expected_arms):
        point[arm] = {
            "exact_next_token": accuracy(exact_predictions[:, index], truth),
            "surface": accuracy(surface_predictions[:, index], truth),
            "exact_flips_vs_original": paired_flips(
                original, exact_predictions[:, index], truth
            ),
            "surface_flips_vs_original": paired_flips(
                surface_predictions[:, 0],
                surface_predictions[:, index],
                truth,
            ),
        }
    matched_oracle = np.any(
        exact_predictions[:, [0, expected_arms.index("metric_matched")]]
        == truth[:, None],
        axis=1,
    )
    baseline_accuracy = float(np.mean(original == truth))
    headroom = float(np.mean(matched_oracle) - baseline_accuracy)
    matched_flips = point["metric_matched"]["exact_flips_vs_original"]
    matched_net = matched_flips["net"]
    transformed_arms = expected_arms[1:]
    diagnostics_by_arm = {
        arm: [
            row["transform_diagnostics"][expected_arms.index(arm)]
            for row in rows
        ]
        for arm in transformed_arms
    }
    diagnostics = diagnostics_by_arm["metric_matched"]
    dose = np.asarray([item["raw_l2_dose"] for item in diagnostics])
    closure = np.asarray([item["source_l2_closure"] for item in diagnostics])
    fidelity_cosine = np.asarray(
        [
            item["first_order_cosine"]
            for arm in transformed_arms
            for item in diagnostics_by_arm[arm]
        ]
    )
    first_order_relative_error = np.asarray(
        [
            item["first_order_relative_error"]
            for arm in transformed_arms
            for item in diagnostics_by_arm[arm]
        ]
    )
    actual_projected_norm = np.asarray(
        [
            item["actual_projected_mean_delta_norm"]
            for item in diagnostics
        ]
    )
    common_dose_spread = []
    for row_index in range(len(rows)):
        row_doses = np.asarray(
            [
                diagnostics_by_arm[arm][row_index]["raw_l2_dose"]
                for arm in transformed_arms
            ]
        )
        common_dose_spread.append(float(row_doses.max() - row_doses.min()))
    common_dose_spread = np.asarray(common_dose_spread)
    score_agreement = exact_predictions == surface_predictions
    token_max_abs_drift = np.asarray(
        [
            item["token_max_abs_drift"]
            for arm in transformed_arms
            for item in diagnostics_by_arm[arm]
        ]
    )
    token_relative_l2_drift = np.asarray(
        [
            item["token_relative_l2_drift"]
            for arm in transformed_arms
            for item in diagnostics_by_arm[arm]
        ]
    )
    toward_arms = (
        "metric_matched",
        "euclidean_matched",
        "metric_wrong_ct",
        "metric_wrong_mri",
    )
    all_toward_closure = np.asarray(
        [
            item["source_l2_closure"]
            for arm in toward_arms
            for item in diagnostics_by_arm[arm]
        ]
    )
    away_closure = np.asarray(
        [
            item["source_l2_closure"]
            for item in diagnostics_by_arm["away"]
        ]
    )
    optimal_dose = np.asarray(
        [
            item["optimal_raw_l2_dose"]
            for arm in transformed_arms
            for item in diagnostics_by_arm[arm]
        ]
    )
    common_dose = np.asarray(
        [
            item["common_raw_l2_dose"]
            for arm in transformed_arms
            for item in diagnostics_by_arm[arm]
        ]
    )
    projection_identity_error = np.asarray(
        [
            item["optimal_projection_identity_error"]
            for arm in transformed_arms
            for item in diagnostics_by_arm[arm]
        ]
    )
    norm_ratio = np.asarray(
        [
            item["actual_over_predicted_norm"]
            for arm in transformed_arms
            for item in diagnostics_by_arm[arm]
        ]
    )
    arm_mechanism = {}
    for arm in transformed_arms:
        values = diagnostics_by_arm[arm]
        arm_mechanism[arm] = {
            "mean_source_l2_closure": float(
                np.mean([item["source_l2_closure"] for item in values])
            ),
            "positive_source_l2_closure_fraction": float(
                np.mean([item["source_l2_closure"] > 0 for item in values])
            ),
            "mean_actual_projected_delta_norm": float(
                np.mean(
                    [
                        item["actual_projected_mean_delta_norm"]
                        for item in values
                    ]
                )
            ),
            "minimum_first_order_cosine": float(
                np.min([item["first_order_cosine"] for item in values])
            ),
            "mean_first_order_relative_error": float(
                np.mean([item["first_order_relative_error"] for item in values])
            ),
            "mean_actual_over_predicted_norm": float(
                np.mean(
                    [
                        item["actual_over_predicted_norm"]
                        for item in values
                    ]
                )
            ),
        }
    pilot_gate = {
        "at_least_one_rescue": matched_flips["rescues"] >= 1,
        "zero_harmful": matched_flips["harmful"] == 0,
        "oracle_headroom_strictly_gt_3pp": headroom > 0.03,
        "matched_net_strictly_beats_wrong_and_away": all(
            matched_net > point[arm]["exact_flips_vs_original"]["net"]
            for arm in ("metric_wrong_ct", "metric_wrong_mri", "away")
        ),
        "metric_gain_not_explained_by_euclidean_same_dose": (
            matched_net
            > point["euclidean_matched"]["exact_flips_vs_original"]["net"]
        ),
        "matched_actual_source_closure_strictly_positive": bool(
            np.all(closure > 0)
        ),
        "matched_actual_projected_change_nonzero": bool(
            np.all(actual_projected_norm > 1e-10)
        ),
        "finite_step_fidelity_cosine_gt_0_99_all_arms": bool(
            np.all(fidelity_cosine > 0.99)
        ),
        "all_transforms_use_common_dose": bool(
            np.all(common_dose_spread < 1e-6)
        ),
        "raw_token_drift_within_1e_3": bool(
            np.all(token_max_abs_drift <= 1e-3)
        ),
        "raw_token_drift_exactly_zero": bool(
            np.all(token_max_abs_drift == 0)
            and np.all(token_relative_l2_drift == 0)
        ),
        "surface_and_literal_token_predictions_agree": bool(
            np.all(score_agreement)
        ),
        "all_toward_arms_actual_source_closure_positive": bool(
            np.all(all_toward_closure > 0)
        ),
        "away_actual_source_closure_negative": bool(
            np.all(away_closure < 0)
        ),
        "all_optimal_doses_at_least_common_dose": bool(
            np.all(optimal_dose + 1e-6 >= common_dose)
        ),
        "optimal_projection_identity_error_lt_1e_5": bool(
            np.all(projection_identity_error < 1e-5)
        ),
        "first_order_relative_error_lt_0_1_all_arms": bool(
            np.all(first_order_relative_error < 0.1)
        ),
        "actual_over_predicted_norm_in_0_9_1_1_all_arms": bool(
            np.all((norm_ratio >= 0.9) & (norm_ratio <= 1.1))
        ),
    }
    effect_gate_names = {
        "at_least_one_rescue",
        "zero_harmful",
        "oracle_headroom_strictly_gt_3pp",
        "matched_net_strictly_beats_wrong_and_away",
        "metric_gain_not_explained_by_euclidean_same_dose",
    }
    mechanism_gate_names = [
        name for name in pilot_gate if name not in effect_gate_names
    ]
    pilot_gate["overall_mechanism_pass"] = all(
        pilot_gate[name] for name in mechanism_gate_names
    )
    report = {
        "version": VERSION,
        "cache": str(args.cache.resolve()),
        "fingerprint": metadata["fingerprint"],
        "n": len(rows),
        "primary_interface": (
            "argmin exact next-token conditional NLL; this is complete label "
            "NLL here only because literal Yes and No are verified one-token labels"
        ),
        "point": point,
        "metric_matched_oracle": {
            "accuracy": float(np.mean(matched_oracle)),
            "headroom": headroom,
        },
        "mechanism": {
            "positive_source_l2_closure_fraction": float(np.mean(closure > 0)),
            "zero_dose_fraction": float(np.mean(dose <= 1e-10)),
            "mean_raw_l2_dose": float(np.mean(dose)),
            "mean_source_l2_closure": float(np.mean(closure)),
            "nonzero_actual_projected_fraction": float(
                np.mean(actual_projected_norm > 1e-10)
            ),
            "minimum_first_order_cosine_all_arms": float(
                np.min(fidelity_cosine)
            ),
            "mean_first_order_cosine_all_arms": float(
                np.mean(fidelity_cosine)
            ),
            "mean_first_order_relative_error": float(
                np.mean(first_order_relative_error)
            ),
            "max_first_order_relative_error": float(
                np.max(first_order_relative_error)
            ),
            "max_common_dose_spread": float(np.max(common_dose_spread)),
            "surface_literal_prediction_agreement": float(
                np.mean(score_agreement)
            ),
            "max_raw_token_abs_drift": float(np.max(token_max_abs_drift)),
            "max_raw_token_relative_l2_drift": float(
                np.max(token_relative_l2_drift)
            ),
            "maximum_optimal_projection_identity_error": float(
                np.max(projection_identity_error)
            ),
            "maximum_first_order_relative_error_all_arms": float(
                np.max(first_order_relative_error)
            ),
            "actual_over_predicted_norm_range": [
                float(np.min(norm_ratio)),
                float(np.max(norm_ratio)),
            ],
            "by_arm": arm_mechanism,
        },
        "pilot_gate": pilot_gate,
        "rows": [
            {
                "qid": row["qid"],
                "gt_index": int(row["gt_index"]),
                "exact_predictions": {
                    arm: int(exact_predictions[row_index, arm_index])
                    for arm_index, arm in enumerate(expected_arms)
                },
            }
            for row_index, row in enumerate(rows)
        ],
        "caveat": (
            "n=16 is an identifiability pilot. Even one flip is 6.25pp and "
            "cannot establish paired statistical significance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "point": report["point"],
                "metric_matched_oracle": report["metric_matched_oracle"],
                "mechanism": report["mechanism"],
                "pilot_gate": report["pilot_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
