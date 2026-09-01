#!/usr/bin/env python3
"""Outcome-blind static VinDr fallback audit for the longitudinal route.

The analysis uses a fixed nearest-centroid readout so that a small development
set cannot hide a hyperparameter search.  Confirmation labels are never used
for layer selection, score normalization, or thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


VERSION = "static-vindr-fallback-analysis-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def clear_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = np.asarray([float(row["reader_support"]) for row in rows])
    clear = np.isin(support, (0.0, 1.0))
    labels = support.astype(np.int64)
    findings = np.asarray([str(row["finding"]) for row in rows])
    return clear, labels, findings


def unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def fit_cosine_directions(
    features: np.ndarray, rows: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    clear, labels, findings = clear_arrays(rows)
    normalized = unit_rows(features.astype(np.float64))
    directions = {}
    for finding in sorted(set(findings.tolist())):
        subset = clear & (findings == finding)
        positive = normalized[subset & (labels == 1)]
        negative = normalized[subset & (labels == 0)]
        if not len(positive) or not len(negative):
            raise RuntimeError(f"missing clear polarity for {finding}")
        direction = positive.mean(axis=0) - negative.mean(axis=0)
        directions[finding] = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return directions


def apply_cosine_directions(
    features: np.ndarray, rows: list[dict[str, Any]], directions: dict[str, np.ndarray]
) -> np.ndarray:
    normalized = unit_rows(features.astype(np.float64))
    return np.asarray(
        [normalized[index] @ directions[str(row["finding"])] for index, row in enumerate(rows)]
    )


def macro_auroc(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    clear, labels, findings = clear_arrays(rows)
    by_finding = {}
    for finding in sorted(set(findings.tolist())):
        subset = clear & (findings == finding)
        by_finding[finding] = float(roc_auc_score(labels[subset], scores[subset]))
    return {
        "macro": float(np.mean(list(by_finding.values()))),
        "pooled": float(roc_auc_score(labels[clear], scores[clear])),
        "by_finding": by_finding,
        "n_clear": int(clear.sum()),
    }


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(scores)
    if len(candidates) == 1:
        return float(candidates[0])
    mids = (candidates[:-1] + candidates[1:]) / 2.0
    candidates = np.concatenate(([candidates[0] - 1e-12], mids, [candidates[-1] + 1e-12]))
    values = [balanced_accuracy_score(labels, scores >= value) for value in candidates]
    best = max(range(len(candidates)), key=lambda index: (values[index], -abs(candidates[index])))
    return float(candidates[best])


def fit_location_scale(scores: np.ndarray) -> tuple[float, float]:
    location = float(np.median(scores))
    q25, q75 = np.quantile(scores, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= 1e-12:
        scale = float(np.std(scores))
    return location, max(scale, 1e-12)


def fit_finding_rules(
    rows: list[dict[str, Any]], scores: np.ndarray
) -> dict[str, dict[str, float]]:
    clear, labels, findings = clear_arrays(rows)
    rules = {}
    for finding in sorted(set(findings.tolist())):
        subset = clear & (findings == finding)
        location, scale = fit_location_scale(scores[subset])
        standardized = (scores[subset] - location) / scale
        rules[finding] = {
            "location": location,
            "scale": scale,
            "threshold": best_threshold(labels[subset], standardized),
        }
    return rules


def apply_rules(
    rows: list[dict[str, Any]], scores: np.ndarray, rules: dict[str, dict[str, float]]
) -> tuple[np.ndarray, np.ndarray]:
    standardized, prediction = [], []
    for row, score in zip(rows, scores):
        rule = rules[str(row["finding"])]
        value = (float(score) - rule["location"]) / rule["scale"]
        standardized.append(value)
        prediction.append(int(value >= rule["threshold"]))
    return np.asarray(standardized), np.asarray(prediction)


def macro_balanced_accuracy(rows: list[dict[str, Any]], predictions: np.ndarray) -> dict[str, Any]:
    clear, labels, findings = clear_arrays(rows)
    by_finding = {}
    for finding in sorted(set(findings.tolist())):
        subset = clear & (findings == finding)
        by_finding[finding] = float(balanced_accuracy_score(labels[subset], predictions[subset]))
    return {"macro": float(np.mean(list(by_finding.values()))), "by_finding": by_finding}


def clustered_ba_gain_bootstrap(
    rows: list[dict[str, Any]],
    native_predictions: np.ndarray,
    fused_predictions: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Resample whole images and compare both methods on identical draws."""
    image_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if float(row["reader_support"]) in (0.0, 1.0):
            image_to_indices[str(row["image_id"])].append(index)
    image_ids = np.asarray(sorted(image_to_indices))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        sampled_images = rng.choice(image_ids, size=len(image_ids), replace=True)
        indices = [index for image_id in sampled_images for index in image_to_indices[image_id]]
        sampled_rows = [rows[index] for index in indices]
        try:
            native = macro_balanced_accuracy(sampled_rows, native_predictions[indices])["macro"]
            fused = macro_balanced_accuracy(sampled_rows, fused_predictions[indices])["macro"]
        except ValueError:
            continue
        deltas.append(fused - native)
    return {
        "unit": "image cluster",
        "unique_images": len(image_ids),
        "draws_used": len(deltas),
        "gain_95_ci": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
        "probability_gain_at_least_3pp": float(np.mean(np.asarray(deltas) >= 0.03)),
    }


def final_margin(rows: list[dict[str, Any]], layer: int) -> np.ndarray:
    return np.asarray(
        [
            float(row["diagnostic_plain_logit_lens"][str(layer)]["supported"])
            - float(row["diagnostic_plain_logit_lens"][str(layer)]["refuted"])
            for row in rows
        ]
    )


def roi_control(raw_path: Path, draws: int, seed: int) -> dict[str, Any]:
    rows = [row for row in load_jsonl(raw_path) if row.get("status") == "ok"]
    pairs = []
    for row in rows:
        baseline = float(row["baseline_coordinates"]["polarity"])
        grouped: dict[float, dict[str, float]] = defaultdict(dict)
        for item in row["interventions"]:
            drop = baseline - float(item["coordinates"]["polarity"])
            grouped[float(item["dose"])][str(item["region"])] = drop
        for dose, values in grouped.items():
            if set(values) != {"roi", "background"}:
                raise RuntimeError(f"incomplete ROI-control dose: {row['record_key']} {dose}")
            pairs.append(
                {
                    "image_id": row["image_id"],
                    "record_key": row["record_key"],
                    "dose": dose,
                    "difference": values["roi"] - values["background"],
                }
            )
    differences = np.asarray([item["difference"] for item in pairs])

    def concordance(values: np.ndarray) -> float:
        return float(np.mean((values > 0).astype(float) + 0.5 * (values == 0)))

    by_image: dict[str, list[float]] = defaultdict(list)
    for item in pairs:
        by_image[item["image_id"]].append(float(item["difference"]))
    image_values = {key: float(np.mean(value)) for key, value in by_image.items()}
    estimate = concordance(np.asarray(list(image_values.values())))
    rng = np.random.default_rng(seed)
    image_ids = np.asarray(sorted(image_values))
    bootstrap = []
    for _ in range(draws):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        bootstrap.append(concordance(np.asarray([image_values[key] for key in sampled])))
    return {
        "definition": "P(mean-dose ROI aligned drop > matched background drop) + 0.5*ties",
        "n_records": len(rows),
        "n_unique_images": len(image_values),
        "n_dose_pairs": len(pairs),
        "paired_auc": estimate,
        "bootstrap_95_ci": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "mean_roi_minus_background_drop": float(np.mean(differences)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-dir", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--roi-raw", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    dev_rows = load_jsonl(args.dev_dir / "metadata.jsonl")
    test_rows = load_jsonl(args.confirmation_dir / "metadata.jsonl")
    dev_hidden = np.load(args.dev_dir / "hidden_states.npz")
    test_hidden = np.load(args.confirmation_dir / "hidden_states.npz")
    dev_layers = [int(value) for value in dev_hidden["layers"]]
    test_layers = [int(value) for value in test_hidden["layers"]]
    if dev_layers != test_layers:
        raise RuntimeError("development/confirmation layer mismatch")
    if len(dev_rows) != len(dev_hidden["visual_mean"]) or len(test_rows) != len(test_hidden["visual_mean"]):
        raise RuntimeError("metadata/hidden-state row mismatch")

    visual = {}
    dev_visual_scores = {}
    test_visual_scores = {}
    for layer_index, layer in enumerate(dev_layers):
        directions = fit_cosine_directions(dev_hidden["visual_mean"][:, layer_index], dev_rows)
        dev_score = apply_cosine_directions(
            dev_hidden["visual_mean"][:, layer_index], dev_rows, directions
        )
        test_score = apply_cosine_directions(
            test_hidden["visual_mean"][:, layer_index], test_rows, directions
        )
        dev_visual_scores[layer] = dev_score
        test_visual_scores[layer] = test_score
        visual[str(layer)] = {
            "development": macro_auroc(dev_rows, dev_score),
            "confirmation": macro_auroc(test_rows, test_score),
        }

    final_layer = max(dev_layers)
    candidate_layers = [layer for layer in dev_layers if layer != final_layer]
    selected_layer = max(
        candidate_layers,
        key=lambda layer: (visual[str(layer)]["development"]["macro"], -layer),
    )
    dev_final = final_margin(dev_rows, final_layer)
    test_final = final_margin(test_rows, final_layer)
    native_rules = fit_finding_rules(dev_rows, dev_final)
    _, native_prediction = apply_rules(test_rows, test_final, native_rules)
    native_ba = macro_balanced_accuracy(test_rows, native_prediction)

    visual_rules = fit_finding_rules(dev_rows, dev_visual_scores[selected_layer])
    dev_visual_z, _ = apply_rules(dev_rows, dev_visual_scores[selected_layer], visual_rules)
    test_visual_z, _ = apply_rules(test_rows, test_visual_scores[selected_layer], visual_rules)
    final_scale_rules = fit_finding_rules(dev_rows, dev_final)
    dev_final_z, _ = apply_rules(dev_rows, dev_final, final_scale_rules)
    test_final_z, _ = apply_rules(test_rows, test_final, final_scale_rules)
    dev_fused = 0.5 * (dev_visual_z + dev_final_z)
    test_fused = 0.5 * (test_visual_z + test_final_z)
    fused_rules = fit_finding_rules(dev_rows, dev_fused)
    _, fused_prediction = apply_rules(test_rows, test_fused, fused_rules)
    fused_ba = macro_balanced_accuracy(test_rows, fused_prediction)
    fusion_gain = fused_ba["macro"] - native_ba["macro"]
    fusion_bootstrap = clustered_ba_gain_bootstrap(
        test_rows, native_prediction, fused_prediction, args.draws, args.seed
    )

    roi = roi_control(args.roi_raw, args.draws, args.seed)
    gate = json.loads(args.gate.read_text())
    thresholds = gate["gates"]
    checks = {
        "best_nonfinal_visual_macro_auroc_at_least": visual[str(selected_layer)][
            "confirmation"
        ]["macro"]
        >= float(thresholds["best_nonfinal_visual_macro_auroc_at_least"]),
        "roi_control_paired_auc_at_least": roi["paired_auc"]
        >= float(thresholds["roi_control_paired_auc_at_least"]),
        "fusion_macro_balanced_accuracy_gain_at_least": fusion_gain
        >= float(thresholds["fusion_macro_balanced_accuracy_gain_at_least"]),
    }
    passed = all(checks.values())
    result = {
        "version": VERSION,
        "status": "go" if passed else "no_go",
        "scope": "static VinDr fallback only; not evidence for longitudinal transfer",
        "measurement_boundary": (
            "visual_mean contains prompt-conditioned visual-token states inside the LLM; "
            "it is not a raw vision-encoder or projector measurement. Passing this screen "
            "therefore shows decodability, not localized causal evidence."
        ),
        "n_dev": len(dev_rows),
        "n_confirmation": len(test_rows),
        "layers": dev_layers,
        "selected_nonfinal_layer_from_dev": selected_layer,
        "visual_polarity": visual,
        "roi_control": roi,
        "minimal_fusion": {
            "native_final_macro_balanced_accuracy": native_ba,
            "fused_macro_balanced_accuracy": fused_ba,
            "macro_balanced_accuracy_gain": fusion_gain,
            "paired_image_bootstrap": fusion_bootstrap,
        },
        "gate_checks": checks,
        "decision": {
            "train_static_evidence_projector": passed,
            "reason": "All fixed static fallback gates must pass before any adapter training.",
        },
        "provenance": {
            "dev_metadata_sha256": sha256(args.dev_dir / "metadata.jsonl"),
            "dev_hidden_sha256": sha256(args.dev_dir / "hidden_states.npz"),
            "confirmation_metadata_sha256": sha256(args.confirmation_dir / "metadata.jsonl"),
            "confirmation_hidden_sha256": sha256(args.confirmation_dir / "hidden_states.npz"),
            "roi_raw_sha256": sha256(args.roi_raw),
            "gate_sha256": sha256(args.gate),
            "analysis_code_sha256": sha256(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "visual_polarity"}, indent=2))


if __name__ == "__main__":
    main()
