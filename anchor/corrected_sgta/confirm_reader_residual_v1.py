#!/usr/bin/env python3
"""Evaluate frozen early/final reader-unanimity probes on confirmation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file
from corrected_sgta.screen_reader_residual_v1 import (
    STRATUM_NAMES,
    fit_predict_models,
    metric_delta,
    reader_targets,
)


VERSION = "vindr-reader-residual-confirmation-v2"


def load_features(directory: Path, split: str) -> dict[str, Any]:
    config = json.loads((directory / "config.json").read_text())
    if config.get("split") != split:
        raise ValueError(f"expected {split} collector input")
    metadata = [json.loads(x) for x in (directory / "metadata.jsonl").read_text().splitlines() if x]
    arrays = np.load(directory / "hidden_states.npz", allow_pickle=False)
    layers = np.asarray(arrays["layers"], dtype=int)
    families = {
        "claim": np.asarray(arrays["claim"], dtype=np.float32),
        "visual_mean": np.asarray(arrays["visual_mean"], dtype=np.float32),
        "visual_std": np.asarray(arrays["visual_std"], dtype=np.float32),
        "routing": np.asarray(arrays["routing_statistics"], dtype=np.float32),
    }
    final = str(int(layers[-1]))
    logits = np.asarray([
        [row["diagnostic_plain_logit_lens"][final][key] for key in ("supported", "refuted", "undetermined")]
        for row in metadata
    ], dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True)
    target, stratum = reader_targets([row["positive_votes"] for row in metadata])
    return {
        "config": config, "metadata": metadata, "layers": layers, "families": families,
        "target": target, "stratum": stratum,
        "evidence": logits[:, 0] - logits[:, 1],
        "maybe": logits[:, 2] - np.logaddexp(logits[:, 0], logits[:, 1]),
        "confidence": probabilities.max(axis=1),
        "entropy": -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1),
        "findings": np.asarray([row["finding"] for row in metadata], dtype=object),
        "groups": np.asarray([row["image_id"] for row in metadata], dtype=object),
    }


def bootstrap_pair(y, groups, left, right, draws, seed):
    y = np.asarray(y)
    groups = np.asarray(groups)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if not (len(y) == len(groups) == len(left) == len(right)):
        raise ValueError("paired bootstrap inputs have unequal lengths")
    if np.unique(y).size != 2:
        raise ValueError("paired comparison requires both target classes")
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    unique = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    aucs, briers = [], []
    for _ in range(draws):
        idx = np.concatenate([by_group[g] for g in rng.choice(unique, len(unique), replace=True)])
        if np.unique(y[idx]).size < 2:
            continue
        aucs.append(float(roc_auc_score(y[idx], right[idx]) - roc_auc_score(y[idx], left[idx])))
        lb = float(brier_score_loss(y[idx], left[idx])); rb = float(brier_score_loss(y[idx], right[idx]))
        briers.append((lb - rb) / max(lb, 1e-12))
    if not aucs:
        raise RuntimeError("all clustered bootstrap draws were degenerate")
    left_auc = float(roc_auc_score(y, left)); right_auc = float(roc_auc_score(y, right))
    left_brier = float(brier_score_loss(y, left)); right_brier = float(brier_score_loss(y, right))
    def record(point, values):
        return {"estimate": float(point), "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975))}
    return {
        "delta_auc": record(right_auc - left_auc, aucs),
        "relative_brier_improvement": record(
            (left_brier - right_brier) / max(left_brier, 1e-12), briers
        ),
        "valid_draws": len(aucs),
        "n_clusters": len(unique),
    }


def passes_positive_control(comparison, auc_margin=0.0, brier_margin=0.0):
    """Require an augmented model to beat a matched alternative on both scores."""

    auc = comparison["delta_auc"]
    brier = comparison["relative_brier_improvement"]
    return bool(
        auc["estimate"] > auc_margin
        and auc["ci_low"] > 0
        and brier["estimate"] > brier_margin
        and brier["ci_low"] > 0
    )


def classify_boundary(
    early_evidence,
    final_evidence,
    early_final,
    margin=.05,
    decodability_controls=(),
):
    early_usable = early_evidence["delta_auc"]["estimate"] >= margin and early_evidence["delta_auc"]["ci_low"] > 0 and early_evidence["relative_brier_improvement"]["estimate"] >= .05 and early_evidence["relative_brier_improvement"]["ci_low"] > 0
    final_usable = final_evidence["delta_auc"]["estimate"] >= margin and final_evidence["delta_auc"]["ci_low"] > 0 and final_evidence["relative_brier_improvement"]["estimate"] >= .05 and final_evidence["relative_brier_improvement"]["ci_low"] > 0
    ef = early_final["delta_auc"]
    if early_usable and ef["estimate"] >= margin and ef["ci_low"] > 0:
        return "Early erasure"
    if final_usable and ef["estimate"] <= -margin and ef["ci_high"] < 0:
        return "Late emergence"
    if ef["ci_low"] > -margin and ef["ci_high"] < margin:
        return "Layer-stable"
    no_decodability = all(
        comparison["delta_auc"]["ci_high"] < margin
        and comparison["relative_brier_improvement"]["ci_high"] < .05
        for comparison in (early_evidence, final_evidence, *decodability_controls)
    )
    if no_decodability:
        return "Not decodable"
    return "Indeterminate"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dev", type=Path, required=True); p.add_argument("--confirmation", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=5000); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    dev, confirmation = load_features(args.dev, "dev"), load_features(args.confirmation, "confirmation")
    lock = json.loads(args.lock.read_text())
    if lock.get("status") != "frozen_before_confirmation_features": raise ValueError("invalid direction lock")
    if lock["provenance"]["dev_hidden_states_sha256"] != sha256_file(args.dev / "hidden_states.npz"): raise ValueError("lock/dev hash mismatch")
    results = {}
    for direction_index, (stratum_value, name) in enumerate(STRATUM_NAMES.items()):
        dm, cm = dev["stratum"] == stratum_value, confirmation["stratum"] == stratum_value
        spec = lock["directions"][name]; family = spec["family"]; k = int(spec["pca_k"])
        early_layer, final_layer = int(spec["layer"]), int(spec["final_comparator_layer"])
        di, fi = list(dev["layers"]).index(early_layer), list(dev["layers"]).index(final_layer)
        cdi, cfi = list(confirmation["layers"]).index(early_layer), list(confirmation["layers"]).index(final_layer)
        y = np.concatenate((dev["target"][dm], confirmation["target"][cm]))
        evidence = np.concatenate((dev["evidence"][dm], confirmation["evidence"][cm])); strata = np.concatenate((dev["stratum"][dm], confirmation["stratum"][cm]))
        findings = np.concatenate((dev["findings"][dm], confirmation["findings"][cm])); maybe = np.concatenate((dev["maybe"][dm], confirmation["maybe"][cm]))
        confidence = np.concatenate((dev["confidence"][dm], confirmation["confidence"][cm]))
        entropy = np.concatenate((dev["entropy"][dm], confirmation["entropy"][cm]))
        early_x = np.concatenate((dev["families"][family][dm, di], confirmation["families"][family][cm, cdi]))
        final_x = np.concatenate((dev["families"][family][dm, fi], confirmation["families"][family][cm, cfi]))
        ndev = int(dm.sum()); train = np.arange(ndev); test = np.arange(ndev, len(y)); seed = args.seed + direction_index * 100000
        baseline, early = fit_predict_models(train=train, test=test, target=y, evidence=evidence, stratum=strata, findings=findings, features=early_x, k=k, mode="pca", seed=seed, residual_alpha=10.0)
        _, final = fit_predict_models(train=train, test=test, target=y, evidence=evidence, stratum=strata, findings=findings, features=final_x, k=k, mode="pca", seed=seed, residual_alpha=10.0)
        _, random = fit_predict_models(train=train, test=test, target=y, evidence=evidence, stratum=strata, findings=findings, features=early_x, k=k, mode="random", seed=seed + 104729, residual_alpha=10.0)
        _, direct = fit_predict_models(train=train, test=test, target=y, evidence=evidence, stratum=strata, findings=findings, features=None, k=None, mode="pca", seed=seed + 15485863, residual_alpha=10.0, maybe_margin=maybe)
        _, confidence_control = fit_predict_models(train=train, test=test, target=y, evidence=evidence, stratum=strata, findings=findings, features=None, k=None, mode="pca", seed=seed + 32452843, residual_alpha=10.0, maybe_margin=confidence)
        _, entropy_control = fit_predict_models(train=train, test=test, target=y, evidence=evidence, stratum=strata, findings=findings, features=None, k=None, mode="pca", seed=seed + 49979687, residual_alpha=10.0, maybe_margin=entropy)
        yc = y[test]; groups = confirmation["groups"][cm]
        comparisons = {
            "early_vs_evidence": bootstrap_pair(yc, groups, baseline, early, args.draws, seed + 1),
            "final_vs_evidence": bootstrap_pair(yc, groups, baseline, final, args.draws, seed + 2),
            "early_vs_final": bootstrap_pair(yc, groups, final, early, args.draws, seed + 3),
            "early_vs_random": bootstrap_pair(yc, groups, random, early, args.draws, seed + 4),
            "early_vs_direct_maybe": bootstrap_pair(yc, groups, direct, early, args.draws, seed + 5),
            "random_vs_evidence": bootstrap_pair(yc, groups, baseline, random, args.draws, seed + 6),
            "direct_maybe_vs_evidence": bootstrap_pair(yc, groups, baseline, direct, args.draws, seed + 7),
            "confidence_vs_evidence": bootstrap_pair(yc, groups, baseline, confidence_control, args.draws, seed + 8),
            "entropy_vs_evidence": bootstrap_pair(yc, groups, baseline, entropy_control, args.draws, seed + 9),
            "early_vs_confidence": bootstrap_pair(yc, groups, confidence_control, early, args.draws, seed + 10),
            "early_vs_entropy": bootstrap_pair(yc, groups, entropy_control, early, args.draws, seed + 11),
        }
        finding_results = {}
        for finding_index, finding in enumerate(sorted(set(confirmation["findings"][cm]))):
            fm = confirmation["findings"][cm] == finding
            fseed = seed + 1000 + finding_index * 100
            finding_comparisons = {
                "early_vs_evidence": bootstrap_pair(yc[fm], groups[fm], baseline[fm], early[fm], args.draws, fseed + 1),
                "final_vs_evidence": bootstrap_pair(yc[fm], groups[fm], baseline[fm], final[fm], args.draws, fseed + 2),
                "early_vs_final": bootstrap_pair(yc[fm], groups[fm], final[fm], early[fm], args.draws, fseed + 3),
                "early_vs_random": bootstrap_pair(yc[fm], groups[fm], random[fm], early[fm], args.draws, fseed + 4),
                "early_vs_direct_maybe": bootstrap_pair(yc[fm], groups[fm], direct[fm], early[fm], args.draws, fseed + 5),
                "random_vs_evidence": bootstrap_pair(yc[fm], groups[fm], baseline[fm], random[fm], args.draws, fseed + 6),
                "direct_maybe_vs_evidence": bootstrap_pair(yc[fm], groups[fm], baseline[fm], direct[fm], args.draws, fseed + 7),
                "confidence_vs_evidence": bootstrap_pair(yc[fm], groups[fm], baseline[fm], confidence_control[fm], args.draws, fseed + 8),
                "entropy_vs_evidence": bootstrap_pair(yc[fm], groups[fm], baseline[fm], entropy_control[fm], args.draws, fseed + 9),
                "early_vs_confidence": bootstrap_pair(yc[fm], groups[fm], confidence_control[fm], early[fm], args.draws, fseed + 10),
                "early_vs_entropy": bootstrap_pair(yc[fm], groups[fm], entropy_control[fm], early[fm], args.draws, fseed + 11),
            }
            finding_results[str(finding)] = {
                "n": int(fm.sum()),
                "comparisons": finding_comparisons,
                "boundary": classify_boundary(
                    finding_comparisons["early_vs_evidence"],
                    finding_comparisons["final_vs_evidence"],
                    finding_comparisons["early_vs_final"],
                    decodability_controls=(
                        finding_comparisons["random_vs_evidence"],
                        finding_comparisons["direct_maybe_vs_evidence"],
                        finding_comparisons["confidence_vs_evidence"],
                        finding_comparisons["entropy_vs_evidence"],
                    ),
                ),
            }
        boundary = classify_boundary(
            comparisons["early_vs_evidence"],
            comparisons["final_vs_evidence"],
            comparisons["early_vs_final"],
            decodability_controls=(
                comparisons["random_vs_evidence"],
                comparisons["direct_maybe_vs_evidence"],
                comparisons["confidence_vs_evidence"],
                comparisons["entropy_vs_evidence"],
            ),
        )
        controls_pass = (
            passes_positive_control(comparisons["early_vs_random"])
            and passes_positive_control(comparisons["early_vs_direct_maybe"])
            and passes_positive_control(comparisons["early_vs_confidence"])
            and passes_positive_control(comparisons["early_vs_entropy"])
        )
        results[name] = {
            "spec": spec,
            "n": len(yc),
            "point_metrics": {
                "evidence": metric_delta(yc, baseline, baseline),
                "early": metric_delta(yc, baseline, early),
                "final": metric_delta(yc, baseline, final),
                "random": metric_delta(yc, baseline, random),
                "direct_maybe": metric_delta(yc, baseline, direct),
                "confidence": metric_delta(yc, baseline, confidence_control),
                "entropy": metric_delta(yc, baseline, entropy_control),
            },
            "comparisons": comparisons,
            "boundary": boundary,
            "representation_controls_pass": controls_pass,
            "finding_wise": finding_results,
        }
    observational_gate = all(
        value["boundary"] == "Early erasure"
        and value["representation_controls_pass"]
        for value in results.values()
    )
    output = {
        "version": VERSION,
        "status": "complete",
        "model_id": lock["model_id"],
        "results": results,
        "early_erasure_all_directions": all(
            x["boundary"] == "Early erasure" for x in results.values()
        ),
        "observational_gate_passed": observational_gate,
        "method_authorized": False,
        "method_authorization_reason": "observational confirmation cannot replace the preregistered causal patch",
        "provenance": {
            "dev_sha256": sha256_file(args.dev / "hidden_states.npz"),
            "confirmation_sha256": sha256_file(args.confirmation / "hidden_states.npz"),
            "lock_sha256": sha256_file(args.lock),
            "code_sha256": sha256_file(Path(__file__)),
            "draws": args.draws,
            "seed": args.seed,
            "cluster_unit": "image_id",
            "confidence_interval": "image-cluster percentile bootstrap",
            "fitting": "development only; confirmation predictions are never refit",
            "calibration_control": "flexible spline evidence calibration with finding fixed effects; scalar final-logit Maybe, confidence, and entropy additions",
        },
    }
    atomic_json(args.output, output); print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
