#!/usr/bin/env python3
"""Development-only screen for reader-unanimity residuals in VinDr states.

The screen deliberately compares adjacent reader-vote bins inside a fixed
polarity stratum (0/3 vs 1/3 and 2/3 vs 3/3).  It asks whether a hidden-state
family adds cross-fitted information beyond a flexible scalar evidence model;
it is not a confirmation test and must not be reported as paper evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file


VERSION = "vindr-reader-residual-screen-v1"
STRATUM_NAMES = {0: "negative_0v1", 1: "positive_2v3"}
DEFAULT_K = (1, 2, 4, 8, 16)


def _one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float64
        )
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float64)


def reader_targets(votes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (reader-unanimity target, polarity stratum) for adjacent bins."""

    values = np.asarray(votes, dtype=int)
    if not np.isin(values, (0, 1, 2, 3)).all():
        raise ValueError("positive_votes must lie in 0..3")
    stratum = (values >= 2).astype(np.int8)
    target = np.isin(values, (0, 3)).astype(np.int8)
    return target, stratum


def joint_stratification(
    target: np.ndarray, stratum: np.ndarray, findings: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [f"{int(y)}|{int(s)}|{finding}" for y, s, finding in zip(target, stratum, findings)],
        dtype=object,
    )


def admissible_pca_dimensions(
    requested: Iterable[int], train_sizes: Iterable[int], feature_dim: int
) -> list[int]:
    smallest_train = min(int(value) for value in train_sizes)
    maximum = min(int(feature_dim), smallest_train // 20)
    return sorted({int(k) for k in requested if int(k) >= 1 and int(k) <= maximum})


def grouped_splits(
    target: np.ndarray,
    stratum: np.ndarray,
    findings: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds < 2:
        raise ValueError("grouped CV requires at least two folds")
    labels = joint_stratification(target, stratum, findings)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = [(train, test) for train, test in splitter.split(labels, labels, groups)]
    for train, test in splits:
        if set(groups[train]).intersection(groups[test]):
            raise RuntimeError("image group leaked across a CV fold")
        if np.unique(target[train]).size != 2 or np.unique(target[test]).size != 2:
            raise ValueError("a grouped fold lacks one reader-unanimity class")
    return splits


@dataclass
class NuisanceTransform:
    spline: SplineTransformer
    finding_encoder: OneHotEncoder

    @classmethod
    def fit(
        cls, evidence: np.ndarray, stratum: np.ndarray, findings: np.ndarray
    ) -> "NuisanceTransform":
        spline = SplineTransformer(
            degree=3, n_knots=4, knots="quantile", include_bias=False
        ).fit(np.asarray(evidence, dtype=float).reshape(-1, 1))
        finding_encoder = _one_hot().fit(np.asarray(findings, dtype=object).reshape(-1, 1))
        return cls(spline=spline, finding_encoder=finding_encoder)

    def transform(
        self, evidence: np.ndarray, stratum: np.ndarray, findings: np.ndarray
    ) -> np.ndarray:
        spline = self.spline.transform(np.asarray(evidence, dtype=float).reshape(-1, 1))
        polarity = np.asarray(stratum, dtype=float).reshape(-1, 1)
        # Full spline(e) by stratum interaction, explicit stratum main effect,
        # and finding fixed effects. The estimator supplies the intercept.
        interacted = np.concatenate((spline * (1.0 - polarity), spline * polarity), axis=1)
        finding = self.finding_encoder.transform(
            np.asarray(findings, dtype=object).reshape(-1, 1)
        )
        return np.concatenate((polarity, interacted, finding), axis=1)


@dataclass
class ResidualTransform:
    nuisance_ridge: Ridge
    scaler: StandardScaler
    projector: Any

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        nuisance: np.ndarray,
        k: int,
        mode: str,
        seed: int,
        residual_alpha: float,
    ) -> "ResidualTransform":
        features = np.asarray(features, dtype=np.float64)
        nuisance_ridge = Ridge(alpha=residual_alpha).fit(nuisance, features)
        residual = features - nuisance_ridge.predict(nuisance)
        scaler = StandardScaler().fit(residual)
        scaled = scaler.transform(residual)
        if mode == "pca":
            solver = "randomized" if k < min(scaled.shape) else "full"
            projector: Any = PCA(
                n_components=k, whiten=False, svd_solver=solver, random_state=seed
            ).fit(scaled)
        elif mode == "random":
            rng = np.random.default_rng(seed)
            matrix = rng.normal(size=(scaled.shape[1], k))
            matrix, _ = np.linalg.qr(matrix, mode="reduced")
            projector = np.asarray(matrix[:, :k], dtype=np.float64)
        else:
            raise ValueError(f"unknown projection mode: {mode}")
        return cls(nuisance_ridge=nuisance_ridge, scaler=scaler, projector=projector)

    def transform(self, features: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
        residual = np.asarray(features, dtype=np.float64) - self.nuisance_ridge.predict(nuisance)
        scaled = self.scaler.transform(residual)
        if hasattr(self.projector, "transform"):
            return self.projector.transform(scaled)
        return scaled @ self.projector


def ridge_logistic(seed: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, random_state=seed
    )


def fit_predict_models(
    *,
    train: np.ndarray,
    test: np.ndarray,
    target: np.ndarray,
    evidence: np.ndarray,
    stratum: np.ndarray,
    findings: np.ndarray,
    features: np.ndarray | None,
    k: int | None,
    mode: str,
    seed: int,
    residual_alpha: float,
    maybe_margin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    nuisance_transform = NuisanceTransform.fit(
        evidence[train], stratum[train], findings[train]
    )
    nuisance_train = nuisance_transform.transform(
        evidence[train], stratum[train], findings[train]
    )
    nuisance_test = nuisance_transform.transform(
        evidence[test], stratum[test], findings[test]
    )
    baseline = ridge_logistic(seed).fit(nuisance_train, target[train])
    baseline_probability = baseline.predict_proba(nuisance_test)[:, 1]

    if maybe_margin is not None:
        maybe_scale = StandardScaler().fit(maybe_margin[train].reshape(-1, 1))
        maybe_train = maybe_scale.transform(maybe_margin[train].reshape(-1, 1))
        maybe_test = maybe_scale.transform(maybe_margin[test].reshape(-1, 1))
        train_polarity = stratum[train].astype(float).reshape(-1, 1)
        test_polarity = stratum[test].astype(float).reshape(-1, 1)
        # The common Maybe term is the preregistered direct-unanimity control;
        # its interaction with polarity is retained only as an a priori
        # heterogeneity term. This parameterization does not force directions.
        extra_train = np.concatenate(
            (maybe_train, maybe_train * train_polarity),
            axis=1,
        )
        extra_test = np.concatenate(
            (maybe_test, maybe_test * test_polarity), axis=1
        )
    else:
        if features is None or k is None:
            raise ValueError("features and k are required for a representation model")
        residual_transform = ResidualTransform.fit(
            features[train], nuisance_train, k, mode, seed, residual_alpha
        )
        extra_train = residual_transform.transform(features[train], nuisance_train)
        extra_test = residual_transform.transform(features[test], nuisance_test)
    augmented = ridge_logistic(seed + 997).fit(
        np.concatenate((nuisance_train, extra_train), axis=1), target[train]
    )
    augmented_probability = augmented.predict_proba(
        np.concatenate((nuisance_test, extra_test), axis=1)
    )[:, 1]
    return baseline_probability, augmented_probability


def select_k_inner(
    *,
    features: np.ndarray,
    target: np.ndarray,
    evidence: np.ndarray,
    stratum: np.ndarray,
    findings: np.ndarray,
    groups: np.ndarray,
    requested_k: Sequence[int],
    folds: int,
    seed: int,
    residual_alpha: float,
) -> tuple[int, dict[str, float]]:
    splits = grouped_splits(target, stratum, findings, groups, folds, seed)
    candidates = admissible_pca_dimensions(
        requested_k, (len(train) for train, _ in splits), features.shape[1]
    )
    if not candidates:
        raise ValueError("no PCA k satisfies k <= n_train/20 in every inner fold")
    losses: dict[int, list[float]] = {k: [] for k in candidates}
    for fold, (train, test) in enumerate(splits):
        nuisance_transform = NuisanceTransform.fit(
            evidence[train], stratum[train], findings[train]
        )
        nuisance_train = nuisance_transform.transform(
            evidence[train], stratum[train], findings[train]
        )
        nuisance_test = nuisance_transform.transform(
            evidence[test], stratum[test], findings[test]
        )
        nuisance_ridge = Ridge(alpha=residual_alpha).fit(
            nuisance_train, np.asarray(features[train], dtype=np.float64)
        )
        residual_train = np.asarray(features[train], dtype=np.float64) - nuisance_ridge.predict(
            nuisance_train
        )
        residual_test = np.asarray(features[test], dtype=np.float64) - nuisance_ridge.predict(
            nuisance_test
        )
        scaler = StandardScaler().fit(residual_train)
        scaled_train = scaler.transform(residual_train)
        scaled_test = scaler.transform(residual_test)
        maximum_k = max(candidates)
        solver = "randomized" if maximum_k < min(scaled_train.shape) else "full"
        pca = PCA(
            n_components=maximum_k,
            whiten=False,
            svd_solver=solver,
            random_state=seed + 101 * fold,
        ).fit(scaled_train)
        projected_train = pca.transform(scaled_train)
        projected_test = pca.transform(scaled_test)
        for k in candidates:
            model = ridge_logistic(seed + 101 * fold + k).fit(
                np.concatenate((nuisance_train, projected_train[:, :k]), axis=1),
                target[train],
            )
            probability = model.predict_proba(
                np.concatenate((nuisance_test, projected_test[:, :k]), axis=1)
            )[:, 1]
            losses[k].append(float(brier_score_loss(target[test], probability)))
    mean_losses = {k: float(np.mean(values)) for k, values in losses.items()}
    selected = min(candidates, key=lambda value: (mean_losses[value], value))
    return selected, {str(k): mean_losses[k] for k in candidates}


def nested_oof(
    *,
    features: np.ndarray,
    target: np.ndarray,
    evidence: np.ndarray,
    maybe_margin: np.ndarray,
    stratum: np.ndarray,
    findings: np.ndarray,
    groups: np.ndarray,
    requested_k: Sequence[int],
    outer_folds: int,
    inner_folds: int,
    seed: int,
    residual_alpha: float,
) -> dict[str, Any]:
    outer = grouped_splits(
        target, stratum, findings, groups, outer_folds, seed
    )
    baseline = np.full(len(target), np.nan)
    candidate = np.full(len(target), np.nan)
    random_control = np.full(len(target), np.nan)
    direct_maybe = np.full(len(target), np.nan)
    selected_k = []
    inner_scores = []
    for fold, (train, test) in enumerate(outer):
        k, scores = select_k_inner(
            features=features[train],
            target=target[train],
            evidence=evidence[train],
            stratum=stratum[train],
            findings=findings[train],
            groups=groups[train],
            requested_k=requested_k,
            folds=inner_folds,
            seed=seed + 1009 * (fold + 1),
            residual_alpha=residual_alpha,
        )
        selected_k.append(k)
        inner_scores.append(scores)
        baseline_fold, candidate_fold = fit_predict_models(
            train=train,
            test=test,
            target=target,
            evidence=evidence,
            stratum=stratum,
            findings=findings,
            features=features,
            k=k,
            mode="pca",
            seed=seed + 7919 * (fold + 1),
            residual_alpha=residual_alpha,
        )
        _, random_fold = fit_predict_models(
            train=train,
            test=test,
            target=target,
            evidence=evidence,
            stratum=stratum,
            findings=findings,
            features=features,
            k=k,
            mode="random",
            seed=seed + 104729 * (fold + 1),
            residual_alpha=residual_alpha,
        )
        _, maybe_fold = fit_predict_models(
            train=train,
            test=test,
            target=target,
            evidence=evidence,
            stratum=stratum,
            findings=findings,
            features=None,
            k=None,
            mode="pca",
            seed=seed + 15485863 * (fold + 1),
            residual_alpha=residual_alpha,
            maybe_margin=maybe_margin,
        )
        baseline[test] = baseline_fold
        candidate[test] = candidate_fold
        random_control[test] = random_fold
        direct_maybe[test] = maybe_fold
    for name, values in {
        "baseline": baseline,
        "candidate": candidate,
        "random_control": random_control,
        "direct_maybe": direct_maybe,
    }.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"nested CV left non-finite {name} predictions")
    return {
        "baseline": baseline,
        "candidate": candidate,
        "random_control": random_control,
        "direct_maybe": direct_maybe,
        "outer_selected_k": selected_k,
        "inner_mean_brier": inner_scores,
    }


def _auc(target: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(target, probability))


def metric_delta(
    target: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    baseline_brier = float(brier_score_loss(target, baseline))
    candidate_brier = float(brier_score_loss(target, candidate))
    return {
        "baseline_auc": _auc(target, baseline),
        "candidate_auc": _auc(target, candidate),
        "delta_auc": _auc(target, candidate) - _auc(target, baseline),
        "baseline_brier": baseline_brier,
        "candidate_brier": candidate_brier,
        "relative_brier_improvement": (baseline_brier - candidate_brier)
        / max(baseline_brier, 1e-12),
    }


def scoped_metrics(
    target: np.ndarray,
    stratum: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    scopes: dict[str, Any] = {}
    for value, name in STRATUM_NAMES.items():
        mask = stratum == value
        scopes[name] = metric_delta(target[mask], baseline[mask], candidate[mask])
        scopes[name]["n"] = int(mask.sum())
    scopes["macro_delta_auc"] = float(
        np.mean([scopes[name]["delta_auc"] for name in STRATUM_NAMES.values()])
    )
    scopes["macro_relative_brier_improvement"] = float(
        np.mean(
            [
                scopes[name]["relative_brier_improvement"]
                for name in STRATUM_NAMES.values()
            ]
        )
    )
    scopes["pooled_diagnostic"] = metric_delta(target, baseline, candidate)
    return scopes


def cluster_bootstrap(
    *,
    target: np.ndarray,
    stratum: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    delta_auc: list[float] = []
    relative_brier: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        try:
            metrics = scoped_metrics(
                target[indices], stratum[indices], baseline[indices], candidate[indices]
            )
        except ValueError:
            continue
        delta_auc.append(float(metrics["macro_delta_auc"]))
        relative_brier.append(float(metrics["macro_relative_brier_improvement"]))
    if not delta_auc:
        raise RuntimeError("no valid image-cluster bootstrap samples")

    def interval(values: Sequence[float]) -> dict[str, float]:
        return {
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        }

    return {
        "valid_draws": len(delta_auc),
        "macro_delta_auc": interval(delta_auc),
        "macro_relative_brier_improvement": interval(relative_brier),
    }


def per_finding_metrics(
    *,
    target: np.ndarray,
    stratum: np.ndarray,
    findings: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    output = {}
    for finding in sorted(np.unique(findings)):
        mask = findings == finding
        try:
            metrics = scoped_metrics(
                target[mask], stratum[mask], baseline[mask], candidate[mask]
            )
            output[str(finding)] = {
                "n": int(mask.sum()),
                "macro_delta_auc": metrics["macro_delta_auc"],
                "macro_relative_brier_improvement": metrics[
                    "macro_relative_brier_improvement"
                ],
                "auc_sign": int(np.sign(metrics["macro_delta_auc"])),
                "brier_sign": int(
                    np.sign(metrics["macro_relative_brier_improvement"])
                ),
            }
        except ValueError:
            output[str(finding)] = {"n": int(mask.sum()), "status": "insufficient_class_support"}
    return output


def equivalence_diagnostic(
    *,
    target: np.ndarray,
    stratum: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    direct_maybe: np.ndarray,
    draws: int,
    seed: int,
    auc_margin: float,
    brier_margin: float,
) -> dict[str, Any]:
    candidate_metrics = scoped_metrics(target, stratum, baseline, candidate)
    maybe_metrics = scoped_metrics(target, stratum, baseline, direct_maybe)
    point_auc = float(candidate_metrics["macro_delta_auc"] - maybe_metrics["macro_delta_auc"])
    point_brier = float(
        candidate_metrics["macro_relative_brier_improvement"]
        - maybe_metrics["macro_relative_brier_improvement"]
    )
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    brier_values: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        try:
            left = scoped_metrics(
                target[indices], stratum[indices], baseline[indices], candidate[indices]
            )
            right = scoped_metrics(
                target[indices], stratum[indices], baseline[indices], direct_maybe[indices]
            )
        except ValueError:
            continue
        auc_values.append(float(left["macro_delta_auc"] - right["macro_delta_auc"]))
        brier_values.append(
            float(
                left["macro_relative_brier_improvement"]
                - right["macro_relative_brier_improvement"]
            )
        )
    auc_ci = (float(np.quantile(auc_values, 0.025)), float(np.quantile(auc_values, 0.975)))
    brier_ci = (
        float(np.quantile(brier_values, 0.025)),
        float(np.quantile(brier_values, 0.975)),
    )
    return {
        "definition": "candidate incremental performance minus final direct-Maybe-margin incremental performance",
        "direct_maybe_margin": "undetermined_logit-logaddexp(supported_logit,refuted_logit)",
        "direct_maybe_parameterization": "common Maybe main effect plus a polarity-stratum heterogeneity term",
        "delta_auc_difference": point_auc,
        "delta_auc_difference_ci": list(auc_ci),
        "relative_brier_difference": point_brier,
        "relative_brier_difference_ci": list(brier_ci),
        "equivalence_margins": {"auc": auc_margin, "relative_brier": brier_margin},
        "auc_equivalent": bool(auc_ci[0] > -auc_margin and auc_ci[1] < auc_margin),
        "brier_equivalent": bool(
            brier_ci[0] > -brier_margin and brier_ci[1] < brier_margin
        ),
        "interpretation": "diagnostic_only; equivalence means the representation adds no more than the explicit Maybe readout within the declared margin",
    }


def validate_inputs(features_dir: Path, sampling_summary: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads((features_dir / "config.json").read_text(encoding="utf-8"))
    run_summary = json.loads((features_dir / "summary.json").read_text(encoding="utf-8"))
    sampling = json.loads(sampling_summary.read_text(encoding="utf-8"))
    if config.get("split") != "dev":
        raise ValueError("reader residual screening is dev-only; collector config split must be dev")
    if run_summary.get("status") != "complete":
        raise ValueError("feature collection summary is not complete")
    if sampling.get("split_contract", {}).get("image_disjoint") is not True:
        raise ValueError("sampling summary does not certify image-disjoint splits")
    if int(sampling.get("split_contract", {}).get("claim_rows", {}).get("dev", 0)) <= 0:
        raise ValueError("sampling summary contains no declared dev claims")
    return config, sampling


def load_inputs(features_dir: Path) -> dict[str, Any]:
    metadata = [
        json.loads(line)
        for line in (features_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    arrays = np.load(features_dir / "hidden_states.npz", allow_pickle=False)
    layers = np.asarray(arrays["layers"], dtype=int)
    families = {
        "claim": np.asarray(arrays["claim"], dtype=np.float32),
        "visual_mean": np.asarray(arrays["visual_mean"], dtype=np.float32),
        "visual_std": np.asarray(arrays["visual_std"], dtype=np.float32),
        "routing": np.asarray(arrays["routing_statistics"], dtype=np.float32),
    }
    for name, values in families.items():
        if values.shape[:2] != (len(metadata), len(layers)):
            raise ValueError(f"{name} shape disagrees with metadata/layers")
    final_layer = str(int(layers[-1]))
    logits = np.asarray(
        [
            [
                float(row["diagnostic_plain_logit_lens"][final_layer][state])
                for state in ("supported", "refuted", "undetermined")
            ]
            for row in metadata
        ],
        dtype=np.float64,
    )
    evidence = logits[:, 0] - logits[:, 1]
    maybe_margin = logits[:, 2] - np.logaddexp(logits[:, 0], logits[:, 1])
    votes = np.asarray([int(row["positive_votes"]) for row in metadata])
    if any(str(row.get("experiment_split")) != "dev" for row in metadata):
        raise ValueError("metadata contains a non-dev row")
    target, stratum = reader_targets(votes)
    return {
        "metadata": metadata,
        "layers": layers,
        "families": families,
        "target": target,
        "stratum": stratum,
        "evidence": evidence,
        "maybe_margin": maybe_margin,
        "findings": np.asarray([str(row["finding"]) for row in metadata], dtype=object),
        "groups": np.asarray([str(row["image_id"]) for row in metadata], dtype=object),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--sampling-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-k", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--residual-alpha", type=float, default=10.0)
    parser.add_argument("--equivalence-auc-margin", type=float, default=0.02)
    parser.add_argument("--equivalence-brier-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config, sampling = validate_inputs(args.features_dir, args.sampling_summary)
    data = load_inputs(args.features_dir)
    target = data["target"]
    stratum = data["stratum"]
    findings = data["findings"]
    groups = data["groups"]
    counts = {
        STRATUM_NAMES[value]: {
            "non_unanimous": int(((stratum == value) & (target == 0)).sum()),
            "unanimous": int(((stratum == value) & (target == 1)).sum()),
        }
        for value in STRATUM_NAMES
    }
    results: dict[str, Any] = {}
    prediction_cache: dict[tuple[int, str], dict[str, Any]] = {}
    for layer_index, layer in enumerate(data["layers"]):
        for family, array in data["families"].items():
            key = f"layer_{int(layer)}:{family}"
            predictions = nested_oof(
                features=array[:, layer_index],
                target=target,
                evidence=data["evidence"],
                maybe_margin=data["maybe_margin"],
                stratum=stratum,
                findings=findings,
                groups=groups,
                requested_k=args.pca_k,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                seed=args.seed + int(layer) * 1000 + list(data["families"]).index(family) * 100,
                residual_alpha=args.residual_alpha,
            )
            prediction_cache[(int(layer), family)] = predictions
            candidate_metrics = scoped_metrics(
                target, stratum, predictions["baseline"], predictions["candidate"]
            )
            random_metrics = scoped_metrics(
                target, stratum, predictions["baseline"], predictions["random_control"]
            )
            results[key] = {
                "layer": int(layer),
                "family": family,
                "feature_dimension": int(array.shape[-1]),
                "outer_selected_k": predictions["outer_selected_k"],
                "outer_inner_mean_brier": predictions["inner_mean_brier"],
                "crossfit": candidate_metrics,
                "random_projection_control": random_metrics,
                "candidate_minus_random_projection": scoped_metrics(
                    target,
                    stratum,
                    predictions["random_control"],
                    predictions["candidate"],
                ),
                "candidate_minus_random_projection_ci": cluster_bootstrap(
                    target=target,
                    stratum=stratum,
                    groups=groups,
                    baseline=predictions["random_control"],
                    candidate=predictions["candidate"],
                    draws=args.bootstrap_draws,
                    seed=args.seed + int(layer) + 1709,
                ),
                "image_cluster_ci": cluster_bootstrap(
                    target=target,
                    stratum=stratum,
                    groups=groups,
                    baseline=predictions["baseline"],
                    candidate=predictions["candidate"],
                    draws=args.bootstrap_draws,
                    seed=args.seed + int(layer),
                ),
                "per_finding_sign": per_finding_metrics(
                    target=target,
                    stratum=stratum,
                    findings=findings,
                    baseline=predictions["baseline"],
                    candidate=predictions["candidate"],
                ),
                "direct_maybe_increment_equivalence": equivalence_diagnostic(
                    target=target,
                    stratum=stratum,
                    groups=groups,
                    baseline=predictions["baseline"],
                    candidate=predictions["candidate"],
                    direct_maybe=predictions["direct_maybe"],
                    draws=args.bootstrap_draws,
                    seed=args.seed + int(layer) + 31337,
                    auc_margin=args.equivalence_auc_margin,
                    brier_margin=args.equivalence_brier_margin,
                ),
            }

    # Selection is deliberately based on one frozen scalar objective. Its OOF
    # estimate is selection-biased and is never treated as confirmation.
    selected_key = min(
        results,
        key=lambda key: (
            -float(results[key]["crossfit"]["macro_relative_brier_improvement"]),
            -float(results[key]["crossfit"]["macro_delta_auc"]),
            key,
        ),
    )
    selected = results[selected_key]
    selected_array = data["families"][selected["family"]][
        :, list(data["layers"]).index(selected["layer"])
    ]
    locked_k, locked_inner_scores = select_k_inner(
        features=selected_array,
        target=target,
        evidence=data["evidence"],
        stratum=stratum,
        findings=findings,
        groups=groups,
        requested_k=args.pca_k,
        folds=args.inner_folds,
        seed=args.seed + 999983,
        residual_alpha=args.residual_alpha,
    )
    locked_spec = {
        "development_only": True,
        "paper_claim_authorized": False,
        "selection_objective": "maximum crossfit macro relative Brier improvement; delta AUC tie-break",
        "layer": selected["layer"],
        "family": selected["family"],
        "pca_k": locked_k,
        "pca_candidates": list(args.pca_k),
        "full_dev_inner_mean_brier": locked_inner_scores,
        "baseline": "cubic quantile spline(final supported-refuted evidence) by polarity stratum + finding fixed effects",
        "target": "reader unanimity within polarity strata: 0/3 over 1/3 and 3/3 over 2/3",
        "residualization": f"multioutput Ridge(alpha={args.residual_alpha}) of feature on nuisance design",
        "classifier": "L2 logistic regression C=1.0",
        "random_projection_seed_rule": "seed + 104729 * (outer_fold + 1)",
        "confirmation_policy": "apply this exact layer/family/k without reselection on the untouched confirmation split",
    }
    output = {
        "version": VERSION,
        "status": "complete",
        "scope": "development_screen_only_not_paper_evidence",
        "n_claims": len(target),
        "unique_images": len(np.unique(groups)),
        "stratum_counts": counts,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "results": results,
        "locked_spec": locked_spec,
        "provenance": {
            "features_dir": str(args.features_dir.resolve()),
            "collector_fingerprint": config.get("fingerprint"),
            "hidden_states_sha256": sha256_file(args.features_dir / "hidden_states.npz"),
            "metadata_sha256": sha256_file(args.features_dir / "metadata.jsonl"),
            "sampling_summary": str(args.sampling_summary.resolve()),
            "sampling_summary_sha256": sha256_file(args.sampling_summary),
            "sampling_version": sampling.get("version"),
            "code_sha256": sha256_file(Path(__file__)),
            "seed": args.seed,
        },
    }
    output["fingerprint"] = hashlib.sha256(
        json.dumps(output["locked_spec"], sort_keys=True).encode()
    ).hexdigest()
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
