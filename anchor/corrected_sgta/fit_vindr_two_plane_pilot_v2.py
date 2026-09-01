#!/usr/bin/env python3
"""Fail-closed pilot analysis for unified raw VinDr hidden states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file


VERSION = "vindr-two-plane-pilot-analysis-v2"


def classifier(family: str, dimensions: int, seed: int) -> Pipeline:
    steps = [
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=dimensions, whiten=True, random_state=seed)),
    ]
    if family == "linear":
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    elif family == "rbf":
        model = SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=False, random_state=seed)
    elif family == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(16,), alpha=1.0, max_iter=2000, early_stopping=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown classifier family {family}")
    return Pipeline([*steps, ("model", model)])


def low_dim_classifier(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)),
        ]
    )


def grouped_oof(
    x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    estimator_factory: Callable[[int], object],
    seed: int,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.full(len(labels), np.nan, dtype=float)
    for fold, (train, test) in enumerate(splitter.split(x, labels, groups)):
        estimator = estimator_factory(seed + fold)
        estimator.fit(x[train], labels[train])
        if hasattr(estimator, "decision_function"):
            values = estimator.decision_function(x[test])
        else:
            values = estimator.predict_proba(x[test])[:, 1]
        scores[test] = np.asarray(values, dtype=float)
    if not np.isfinite(scores).all():
        raise RuntimeError("grouped OOF probe left non-finite predictions")
    return scores


def calibrated_lens_oof(
    x: np.ndarray,
    final_logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    dimensions: int,
    seed: int,
) -> np.ndarray:
    """Fit label-free hidden->final-logit transport inside every outer fold."""

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.full(len(labels), np.nan, dtype=float)
    for fold, (train, test) in enumerate(splitter.split(x, labels, groups)):
        scale = StandardScaler()
        train_scaled = scale.fit_transform(x[train])
        test_scaled = scale.transform(x[test])
        pca = PCA(n_components=dimensions, whiten=True, random_state=seed + fold)
        train_reduced = pca.fit_transform(train_scaled)
        test_reduced = pca.transform(test_scaled)
        lens = Ridge(alpha=10.0).fit(train_reduced, final_logits[train])
        mapped_train = lens.predict(train_reduced)
        mapped_test = lens.predict(test_reduced)
        reader = low_dim_classifier(seed + fold).fit(mapped_train, labels[train])
        scores[test] = reader.decision_function(mapped_test)
    if not np.isfinite(scores).all():
        raise RuntimeError("calibrated lens left non-finite predictions")
    return scores


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def paired_group_bootstrap(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    unique = np.unique(groups)
    by_group = {value: np.flatnonzero(groups == value) for value in unique}
    estimate = auc(labels, left) - auc(labels, right)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[value] for value in sampled])
        if np.unique(labels[indices]).size != 2:
            continue
        values.append(auc(labels[indices], left[indices]) - auc(labels[indices], right[indices]))
    if not values:
        raise RuntimeError("no valid grouped bootstrap draws")
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "valid_draws": len(values),
    }


def effective_rank(x: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    power = singular**2
    probability = power / max(power.sum(), 1e-12)
    return float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-12)))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-dimensions", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = [json.loads(line) for line in (args.features_dir / "metadata.jsonl").read_text().splitlines() if line.strip()]
    arrays = np.load(args.features_dir / "hidden_states.npz", allow_pickle=False)
    claim = arrays["claim"].astype(np.float32)
    visual = arrays["visual_mean"].astype(np.float32)
    visual_std = arrays["visual_std"].astype(np.float32) if "visual_std" in arrays else None
    routing = arrays["routing_statistics"].astype(np.float32) if "routing_statistics" in arrays else None
    layers = [int(value) for value in arrays["layers"]]
    if claim.shape[:2] != (len(metadata), len(layers)) or visual.shape != claim.shape:
        raise ValueError("hidden array and metadata shapes disagree")
    labels = np.asarray([int(int(row["positive_votes"]) in {0, 3}) for row in metadata])
    groups = np.asarray([str(row["image_id"]) for row in metadata])
    if np.unique(labels).size != 2:
        raise ValueError("clarity pilot requires unanimous and disagreement rows")
    minimum_train = int(len(labels) * 0.75)
    dimensions = min(args.pca_dimensions, minimum_train - 2, claim.shape[-1])
    if dimensions < 2:
        raise ValueError("insufficient rows for the frozen PCA probe")

    plain_logits = np.asarray(
        [
            [
                [row["diagnostic_plain_logit_lens"][str(layer)][state] for state in ("supported", "refuted", "undetermined")]
                for layer in layers
            ]
            for row in metadata
        ],
        dtype=float,
    )
    final_logits = plain_logits[:, -1]
    families = ("linear", "rbf", "mlp")
    results: dict[str, dict[str, object]] = {}
    predictions: dict[tuple[str, int], np.ndarray] = {}
    for layer_index, layer in enumerate(layers):
        record: dict[str, object] = {
            "claim_norm_mean": float(np.linalg.norm(claim[:, layer_index], axis=1).mean()),
            "claim_effective_rank": effective_rank(claim[:, layer_index]),
            "visual_mean_effective_rank": effective_rank(visual[:, layer_index]),
        }
        for family in families:
            scores = grouped_oof(
                claim[:, layer_index], labels, groups,
                lambda local_seed, family=family: classifier(family, dimensions, local_seed),
                args.seed,
            )
            predictions[(family, layer)] = scores
            record[f"raw_hidden_{family}_auroc"] = auc(labels, scores)
        visual_scores = grouped_oof(
            visual[:, layer_index], labels, groups,
            lambda local_seed: classifier("linear", dimensions, local_seed), args.seed,
        )
        record["visual_mean_linear_auroc"] = auc(labels, visual_scores)
        if visual_std is not None:
            visual_std_scores = grouped_oof(
                visual_std[:, layer_index], labels, groups,
                lambda local_seed: classifier("linear", dimensions, local_seed), args.seed,
            )
            predictions[("visual_dispersion_linear", layer)] = visual_std_scores
            record["visual_dispersion_linear_auroc"] = auc(labels, visual_std_scores)
        if routing is not None:
            routing_scores = grouped_oof(
                routing[:, layer_index], labels, groups, low_dim_classifier, args.seed,
            )
            record["claim_visual_routing_geometry_auroc"] = auc(labels, routing_scores)
        plain_scores = grouped_oof(
            plain_logits[:, layer_index], labels, groups,
            low_dim_classifier, args.seed,
        )
        record["plain_logit_lens_auroc"] = auc(labels, plain_scores)
        polarity = (plain_logits[:, layer_index, 0] - plain_logits[:, layer_index, 1]).reshape(-1, 1)
        polarity_scores = grouped_oof(polarity, labels, groups, low_dim_classifier, args.seed)
        record["polarity_only_auroc"] = auc(labels, polarity_scores)
        calibrated = calibrated_lens_oof(
            claim[:, layer_index], final_logits, labels, groups, dimensions, args.seed
        )
        record["calibrated_lens_auroc"] = auc(labels, calibrated)
        results[str(layer)] = record

    final_layer = layers[-1]
    early_candidates = layers[:-1]
    selected_early = max(
        early_candidates,
        key=lambda layer: float(results[str(layer)]["raw_hidden_linear_auroc"]),
    )
    linear_delta = paired_group_bootstrap(
        labels,
        predictions[("linear", selected_early)],
        predictions[("linear", final_layer)],
        groups,
        args.bootstrap_draws,
        args.seed,
    )
    strongest_control = max(
        float(results[str(selected_early)][name])
        for name in ("plain_logit_lens_auroc", "polarity_only_auroc", "calibrated_lens_auroc")
    )
    increment = float(results[str(selected_early)]["raw_hidden_linear_auroc"]) - strongest_control
    family_deltas = {
        family: float(results[str(selected_early)][f"raw_hidden_{family}_auroc"])
        - float(results[str(final_layer)][f"raw_hidden_{family}_auroc"])
        for family in families
    }
    evidence_routing_gaps = {}
    finding_routing_gaps = {}
    if visual_std is not None:
        for layer in layers:
            evidence_routing_gaps[str(layer)] = paired_group_bootstrap(
                labels,
                predictions[("visual_dispersion_linear", layer)],
                predictions[("linear", layer)],
                groups,
                args.bootstrap_draws,
                args.seed + layer,
            )
        findings = np.asarray([str(row["finding"]) for row in metadata])
        for finding in sorted(np.unique(findings)):
            mask = findings == finding
            finding_routing_gaps[finding] = {
                str(layer): (
                    auc(labels[mask], predictions[("visual_dispersion_linear", layer)][mask])
                    - auc(labels[mask], predictions[("linear", layer)][mask])
                )
                for layer in layers
            }
    final_routing = evidence_routing_gaps.get(str(final_layer))
    gate = {
        "pilot_only_not_formal_confirmation": True,
        "selected_early_layer": selected_early,
        "raw_linear_early_minus_final": linear_delta,
        "raw_early_increment_over_strongest_readout_control": increment,
        "all_probe_families_same_positive_direction": all(value > 0 for value in family_deltas.values()),
        "mechanism_kill_probe_passed": (
            float(linear_delta["estimate"]) >= 0.05
            and float(linear_delta["ci_low"]) > 0
            and increment > 0
            and all(value > 0 for value in family_deltas.values())
        ),
        "alternative_evidence_routing_gap_passed": bool(
            final_routing
            and float(final_routing["estimate"]) >= 0.05
            and float(final_routing["ci_low"]) > 0
            and all(
                finding_routing_gaps[finding][str(final_layer)] > 0
                for finding in finding_routing_gaps
            )
        ),
        "authorization": "none; natural bidirectional patch and dev/confirmation remain required",
    }
    output = {
        "version": VERSION,
        "n": len(metadata),
        "unique_images": len(np.unique(groups)),
        "label_counts": {str(value): int((labels == value).sum()) for value in (0, 1)},
        "layers": results,
        "probe_families": list(families),
        "pca_dimensions": dimensions,
        "family_early_minus_final": family_deltas,
        "visual_dispersion_minus_claim_token": evidence_routing_gaps,
        "visual_dispersion_minus_claim_token_by_finding": finding_routing_gaps,
        "gate": gate,
        "provenance": {
            "features_dir": str(args.features_dir.resolve()),
            "hidden_states_sha256": sha256_file(args.features_dir / "hidden_states.npz"),
            "metadata_sha256": sha256_file(args.features_dir / "metadata.jsonl"),
        },
    }
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
