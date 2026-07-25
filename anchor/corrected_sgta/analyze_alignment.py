"""Analyze DG closure, prediction flips, and minimal consensus from alignment caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from corrected_sgta.cache import decode_array, iter_successes
from corrected_sgta.methods import softmax_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--laplacian-lambda-grid",
        type=float,
        nargs="+",
        default=(0.1, 0.3, 1.0, 3.0),
    )
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12, None)


def anchored_laplacian(
    probabilities: np.ndarray, features: np.ndarray, strength: float
) -> np.ndarray:
    """Smooth view posteriors and return the refined original-image anchor."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(probabilities) <= 1:
        return probabilities[0]
    features = normalize_rows(features)
    squared = np.maximum(
        0.0,
        2.0 - 2.0 * np.clip(features @ features.T, -1.0, 1.0),
    )
    off_diagonal = squared[np.triu_indices(len(features), k=1)]
    positive = off_diagonal[off_diagonal > 1e-12]
    sigma2 = float(np.median(positive)) if len(positive) else 1.0
    weights = np.exp(-squared / max(sigma2, 1e-12))
    np.fill_diagonal(weights, 0.0)
    laplacian = np.diag(weights.sum(axis=1)) - weights
    smoothed = np.linalg.solve(
        np.eye(len(features)) + float(strength) * laplacian,
        probabilities,
    )
    anchor = np.clip(smoothed[0], 1e-12, None)
    return anchor / anchor.sum()


def prediction_summary(predictions: list[int], rows: list[dict]) -> dict:
    labels = [int(row["gt_index"]) for row in rows]
    correct = np.asarray(predictions) == np.asarray(labels)
    return {"n": len(rows), "accuracy": float(correct.mean())}


def flip_summary(predictions: list[int], baseline: list[int], rows: list[dict]) -> dict:
    gt = np.asarray([int(row["gt_index"]) for row in rows])
    pred = np.asarray(predictions)
    base = np.asarray(baseline)
    rescues = int(np.sum((base != gt) & (pred == gt)))
    harmful = int(np.sum((base == gt) & (pred != gt)))
    return {
        "rescues": rescues,
        "harmful": harmful,
        "net": rescues - harmful,
        "changed": int(np.sum(pred != base)),
    }


def main() -> None:
    args = parse_args()
    metadata = json.loads(
        args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text()
    )
    fingerprint = metadata["fingerprint"]
    rows = list(iter_successes(args.cache, fingerprint))
    if not rows:
        raise RuntimeError("no successful alignment rows")

    methods: dict[str, list[int]] = {
        "original": [],
        "single_best_closure": [],
        "uniform_mean": [],
    }
    for value in args.laplacian_lambda_grid:
        methods[f"laplacian_anchor_l{value:g}"] = []
    any_disagreement = []
    oracle_correct = []
    selected_matched_closure = []
    selected_shuffled_closure = []
    all_selected_structure = []
    audits = []

    for row in rows:
        logits = np.asarray(row["style_logits"], dtype=np.float64)
        probabilities = softmax_np(logits)
        visual = decode_array(row["style_visual_features"]).astype(np.float32)
        predictions = np.argmax(logits, axis=-1)
        baseline = int(predictions[0])
        methods["original"].append(baseline)
        methods["single_best_closure"].append(
            int(predictions[1]) if len(predictions) > 1 else baseline
        )
        methods["uniform_mean"].append(int(np.argmax(probabilities.mean(axis=0))))
        for value in args.laplacian_lambda_grid:
            methods[f"laplacian_anchor_l{value:g}"].append(
                int(np.argmax(anchored_laplacian(probabilities, visual, value)))
            )
        any_disagreement.append(bool(np.any(predictions[1:] != baseline)))
        oracle_correct.append(bool(np.any(predictions == int(row["gt_index"]))))

        selected = [
            item for item in row.get("alignment_candidates", []) if item.get("selected")
        ]
        for item in selected:
            selected_matched_closure.append(float(item["relative_closure"]))
            if item.get("shuffled_relative_closure") is not None:
                selected_shuffled_closure.append(
                    float(item["shuffled_relative_closure"])
                )
            all_selected_structure.append(item.get("structure") or {})
        audits.append(
            {
                "qid": row["qid"],
                "gt_index": int(row["gt_index"]),
                "baseline_prediction": baseline,
                "style_predictions": predictions.tolist(),
                "style_names": row["style_names"],
                "fallback_to_original": row.get("fallback_to_original", False),
            }
        )

    point = {name: prediction_summary(values, rows) for name, values in methods.items()}
    flips = {
        name: flip_summary(values, methods["original"], rows)
        for name, values in methods.items()
        if name != "original"
    }
    baseline_accuracy = point["original"]["accuracy"]
    best_method = max(point, key=lambda name: point[name]["accuracy"])
    matched_median = (
        float(np.median(selected_matched_closure))
        if selected_matched_closure
        else None
    )
    shuffled_median = (
        float(np.median(selected_shuffled_closure))
        if selected_shuffled_closure
        else None
    )
    disagreement_rate = float(np.mean(any_disagreement))
    oracle_accuracy = float(np.mean(oracle_correct))
    best_flips = flips.get(best_method, {"rescues": 0, "harmful": 0})
    checks = {
        "matched_median_relative_closure_ge_20pct": matched_median is not None
        and matched_median >= 0.20,
        "disagreement_between_3_and_20pct": 0.03 <= disagreement_rate <= 0.20,
        "style_oracle_headroom_ge_2pp": oracle_accuracy - baseline_accuracy >= 0.02,
        "best_consensus_rescues_ge_harmful": best_flips["rescues"]
        >= best_flips["harmful"],
        "matched_closure_gt_shuffled": matched_median is not None
        and shuffled_median is not None
        and matched_median > shuffled_median,
    }
    report = {
        "version": "sgta-alignment-analysis-v1",
        "source_cache": str(args.cache),
        "fingerprint": fingerprint,
        "n": len(rows),
        "point_accuracy": point,
        "flips_vs_original": flips,
        "best_method_diagnostic_only": best_method,
        "style_oracle_accuracy_diagnostic_only": oracle_accuracy,
        "style_oracle_headroom_diagnostic_only": oracle_accuracy - baseline_accuracy,
        "domain_diagnostics": {
            "cross_view_prediction_disagreement_rate": disagreement_rate,
            "selected_view_count": len(selected_matched_closure),
            "fallback_rate": float(
                np.mean([row.get("fallback_to_original", False) for row in rows])
            ),
            "matched_relative_closure_median": matched_median,
            "matched_relative_closure_p25": float(np.percentile(selected_matched_closure, 25))
            if selected_matched_closure
            else None,
            "shuffled_relative_closure_median": shuffled_median,
            "selected_psnr_median": float(
                np.median(
                    [item["psnr"] for item in all_selected_structure if item.get("psnr") is not None]
                )
            )
            if all_selected_structure
            else None,
            "selected_edge_correlation_median": float(
                np.median(
                    [
                        item["edge_correlation"]
                        for item in all_selected_structure
                        if item.get("edge_correlation") is not None
                    ]
                )
            )
            if all_selected_structure
            else None,
        },
        "gate": {
            "stage": "formal" if len(rows) >= 128 else "smoke_diagnostic_only",
            "checks": checks,
            "pass": len(rows) >= 128 and all(checks.values()),
        },
        "prediction_audit": audits,
        "method_note": (
            "Laplacian reports the refined original anchor. Averaging all nodes after "
            "symmetric Laplacian smoothing is mathematically mean-preserving and is not "
            "treated as an independent method."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"point": point, "diagnostics": report["domain_diagnostics"], "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
