"""Analyze matched-source SGTA against an actual wrong-center intervention."""

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
    parser.add_argument("--laplacian-lambda-grid", type=float, nargs="+", default=(0.1, 0.3, 1.0, 3.0))
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12, None)


def anchored_laplacian(probabilities: np.ndarray, features: np.ndarray, strength: float) -> np.ndarray:
    """Smooth graph posteriors and return the refined original-image anchor."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(probabilities) <= 1:
        return probabilities[0]
    features = normalize_rows(features)
    squared = np.maximum(0.0, 2.0 - 2.0 * np.clip(features @ features.T, -1.0, 1.0))
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
    labels = np.asarray([int(row["gt_index"]) for row in rows])
    correct = np.asarray(predictions) == labels
    return {"n": len(rows), "correct": int(correct.sum()), "accuracy": float(correct.mean())}


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


def subset_indices(roles: list[str], role: str) -> list[int]:
    return [0] + [index for index, value in enumerate(roles) if value == role]


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text())
    fingerprint = metadata["fingerprint"]
    rows = list(iter_successes(args.cache, fingerprint))
    if not rows:
        raise RuntimeError("no successful alignment rows")

    methods: dict[str, list[int]] = {
        "original": [],
        "matched_single_best_closure": [],
        "matched_uniform_mean": [],
        "wrong_control_uniform_mean": [],
    }
    for value in args.laplacian_lambda_grid:
        methods[f"matched_laplacian_anchor_l{value:g}"] = []
        methods[f"wrong_control_laplacian_anchor_l{value:g}"] = []
    disagreement = []
    matched_oracle_correct = []
    matched_closure = []
    wrong_closure = []
    matched_structure = []
    wrong_structure = []
    audits = []

    for row in rows:
        logits = np.asarray(row["style_logits"], dtype=np.float64)
        probabilities = softmax_np(logits)
        visual = decode_array(row["style_visual_features"]).astype(np.float32)
        predictions = np.argmax(logits, axis=-1)
        roles = list(row["style_roles"])
        matched_indices = subset_indices(roles, "matched")
        wrong_indices = subset_indices(roles, "wrong_control")
        baseline = int(predictions[0])
        methods["original"].append(baseline)
        methods["matched_single_best_closure"].append(
            int(predictions[matched_indices[1]]) if len(matched_indices) > 1 else baseline
        )
        methods["matched_uniform_mean"].append(
            int(np.argmax(probabilities[matched_indices].mean(axis=0)))
        )
        methods["wrong_control_uniform_mean"].append(
            int(np.argmax(probabilities[wrong_indices].mean(axis=0)))
        )
        for value in args.laplacian_lambda_grid:
            methods[f"matched_laplacian_anchor_l{value:g}"].append(
                int(np.argmax(anchored_laplacian(probabilities[matched_indices], visual[matched_indices], value)))
            )
            methods[f"wrong_control_laplacian_anchor_l{value:g}"].append(
                int(np.argmax(anchored_laplacian(probabilities[wrong_indices], visual[wrong_indices], value)))
            )
        matched_predictions = predictions[matched_indices]
        disagreement.append(bool(np.any(matched_predictions[1:] != baseline)))
        matched_oracle_correct.append(bool(np.any(matched_predictions == int(row["gt_index"]))))
        selected = [item for item in row.get("alignment_candidates", []) if item.get("selected")]
        for item in selected:
            matched_closure.append(float(item["relative_closure"]))
            matched_structure.append(item.get("structure") or {})
            if item.get("wrong_relative_closure") is not None and item.get("wrong_safe"):
                wrong_closure.append(float(item["wrong_relative_closure"]))
                wrong_structure.append(item.get("wrong_structure") or {})
        audits.append(
            {
                "qid": row["qid"],
                "gt_index": int(row["gt_index"]),
                "baseline_prediction": baseline,
                "style_predictions": predictions.tolist(),
                "style_names": row["style_names"],
                "style_roles": roles,
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
    matched_methods = [name for name in point if name.startswith("matched_")]
    best_matched = max(matched_methods, key=lambda name: point[name]["accuracy"])
    matched_median = float(np.median(matched_closure)) if matched_closure else None
    wrong_median = float(np.median(wrong_closure)) if wrong_closure else None
    disagreement_rate = float(np.mean(disagreement))
    oracle_accuracy = float(np.mean(matched_oracle_correct))
    best_flips = flips[best_matched]
    matched_control_delta = (
        point["matched_uniform_mean"]["accuracy"]
        - point["wrong_control_uniform_mean"]["accuracy"]
    )
    checks = {
        "matched_median_relative_closure_ge_20pct": matched_median is not None and matched_median >= 0.20,
        "matched_closure_gt_wrong_control": matched_median is not None and wrong_median is not None and matched_median > wrong_median,
        "disagreement_between_3_and_20pct": 0.03 <= disagreement_rate <= 0.20,
        "style_oracle_headroom_ge_2pp": oracle_accuracy - baseline_accuracy >= 0.02,
        "best_matched_rescues_ge_harmful": best_flips["rescues"] >= best_flips["harmful"],
    }

    def median_metric(values: list[dict], key: str) -> float | None:
        numbers = [float(item[key]) for item in values if item.get(key) is not None]
        return float(np.median(numbers)) if numbers else None

    report = {
        "version": "sgta-alignment-analysis-v2",
        "source_cache": str(args.cache),
        "fingerprint": fingerprint,
        "n": len(rows),
        "point_accuracy": point,
        "flips_vs_original": flips,
        "best_matched_method_diagnostic_only": best_matched,
        "matched_style_oracle_accuracy_diagnostic_only": oracle_accuracy,
        "matched_style_oracle_headroom_diagnostic_only": oracle_accuracy - baseline_accuracy,
        "matched_minus_wrong_control_uniform_accuracy": matched_control_delta,
        "domain_diagnostics": {
            "matched_cross_view_prediction_disagreement_rate": disagreement_rate,
            "selected_matched_view_count": len(matched_closure),
            "selected_wrong_control_view_count": len(wrong_closure),
            "fallback_rate": float(np.mean([row.get("fallback_to_original", False) for row in rows])),
            "matched_relative_closure_median": matched_median,
            "matched_relative_closure_p25": float(np.percentile(matched_closure, 25)) if matched_closure else None,
            "wrong_control_relative_closure_median": wrong_median,
            "matched_psnr_median": median_metric(matched_structure, "psnr"),
            "matched_edge_correlation_median": median_metric(matched_structure, "edge_correlation"),
            "wrong_control_psnr_median": median_metric(wrong_structure, "psnr"),
            "wrong_control_edge_correlation_median": median_metric(wrong_structure, "edge_correlation"),
        },
        "gate": {
            "stage": "formal" if len(rows) >= 128 else "smoke_diagnostic_only",
            "checks": checks,
            "pass": len(rows) >= 128 and all(checks.values()),
        },
        "prediction_audit": audits,
        "method_note": (
            "Matched and wrong-control graphs are analyzed separately. The control is an "
            "actual cross-modality amplitude intervention evaluated against the unchanged "
            "target source feature center. Laplacian output is the refined original anchor; "
            "averaging all graph nodes is mean-preserving and is not a separate method."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.output)
    print(json.dumps({"point": point, "diagnostics": report["domain_diagnostics"], "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
