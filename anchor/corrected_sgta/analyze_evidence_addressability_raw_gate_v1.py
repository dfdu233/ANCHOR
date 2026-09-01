#!/usr/bin/env python3
"""Formal Stage-2 incremental-information gate for raw visual features.

All representation/probe choices are selected on development by image-grouped
cross-validation.  The image-disjoint confirmation set is evaluated once with
frozen transforms.  A PASS means incremental decodability only; it is not a
localization, causal-control, or hallucination-mitigation result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from corrected_sgta.analyze_evidence_addressability_gate_v1 import (
    brier,
    fit_logistic,
    metric_summary,
    reader_nll,
)


PROTOCOL = "evidence-addressability-raw-increment-gate-v1"
FOLDS = 5
PCS = (8, 16, 32)
REGULARIZATION = (0.01, 0.1, 1.0, 10.0)
REPRESENTATIONS = (
    "pre_mean",
    "pre_mean_std",
    "post_mean",
    "post_mean_std",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hidden_rows(directory: Path) -> dict[str, Any]:
    metadata_path = directory / "metadata.jsonl"
    hidden_path = directory / "hidden_states.npz"
    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    with np.load(hidden_path, allow_pickle=False) as archive:
        layers = np.asarray(archive["layers"], dtype=np.int64)
    final = str(int(layers[-1]))
    margins = []
    for row in rows:
        logits = row["diagnostic_plain_logit_lens"][final]
        margins.append(float(logits["supported"]) - float(logits["refuted"]))
    return {
        "directory": str(directory.resolve()),
        "metadata_sha256": sha256(metadata_path),
        "hidden_sha256": sha256(hidden_path),
        "finding": np.asarray([str(row["finding"]) for row in rows]),
        "votes": np.asarray([int(row["positive_votes"]) for row in rows], dtype=np.int64),
        "image": np.asarray([str(row["image_id"]) for row in rows]),
        "margin": np.asarray(margins, dtype=np.float64),
        "rows": rows,
    }


def load_raw(directory: Path) -> dict[str, Any]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"raw visual artifact is incomplete: {directory}")
    metadata_path = directory / "metadata.jsonl"
    features_path = directory / "features.npz"
    metadata = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    with np.load(features_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}
    if any(value.shape[0] != len(metadata) for value in arrays.values()):
        raise ValueError("raw metadata/array length mismatch")
    image_ids = [str(row["image_id"]) for row in metadata]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("raw visual artifact has duplicate images")
    return {
        "directory": str(directory.resolve()),
        "metadata_sha256": sha256(metadata_path),
        "features_sha256": sha256(features_path),
        "metadata": metadata,
        "arrays": arrays,
        "index": {image: index for index, image in enumerate(image_ids)},
    }


def nuisance_row(metadata: dict[str, Any]) -> tuple[np.ndarray, tuple[str, str, str]]:
    values = metadata["simple_image_metadata"]
    numeric = np.asarray(
        [
            np.log1p(float(values["image_width"])),
            np.log1p(float(values["image_height"])),
            np.log1p(float(values["dicom_rows"])),
            np.log1p(float(values["dicom_columns"])),
            float(values["brightness_mean"]),
            float(values["brightness_std"]),
            float(values["brightness_p05"]),
            float(values["brightness_p95"]),
        ],
        dtype=np.float64,
    )
    categorical = (
        str(values["view_position"]),
        str(values["patient_position"]),
        str(values["photometric_interpretation"]),
    )
    return numeric, categorical


def join_raw(data: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    try:
        indices = np.asarray([raw["index"][image] for image in data["image"]], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"raw artifact lacks hidden-set image: {error}") from error
    arrays = raw["arrays"]
    data = dict(data)
    data["representations"] = {
        "pre_mean": arrays["pre_mean"][indices],
        "pre_mean_std": np.concatenate(
            [arrays["pre_mean"][indices], arrays["pre_std"][indices]], axis=1
        ),
        "post_mean": arrays["post_mean"][indices],
        "post_mean_std": np.concatenate(
            [arrays["post_mean"][indices], arrays["post_std"][indices]], axis=1
        ),
    }
    nuisance = [nuisance_row(raw["metadata"][index]) for index in indices]
    data["nuisance_numeric"] = np.stack([row[0] for row in nuisance])
    data["nuisance_categorical"] = np.asarray([row[1] for row in nuisance], dtype=object)
    return data


def grouped_folds(data: dict[str, Any]) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    strata = np.asarray(
        [f"{finding}:{vote}" for finding, vote in zip(data["finding"], data["votes"])]
    )
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    return splitter.split(np.zeros(len(strata)), strata, groups=data["image"])


@dataclass
class BaseTransform:
    findings: tuple[str, ...]
    margin_mean: np.ndarray
    margin_std: np.ndarray

    def one_hot(self, finding: np.ndarray) -> np.ndarray:
        lookup = {name: index for index, name in enumerate(self.findings)}
        output = np.zeros((len(finding), len(self.findings)), dtype=np.float64)
        for row, name in enumerate(finding):
            output[row, lookup[str(name)]] = 1.0
        return output

    def base_design(self, finding: np.ndarray, margin: np.ndarray) -> np.ndarray:
        one_hot = self.one_hot(finding)
        index = np.argmax(one_hot, axis=1)
        scaled = (margin - self.margin_mean[index]) / self.margin_std[index]
        return np.concatenate([one_hot, one_hot * scaled[:, None]], axis=1)


def fit_base_transform(
    finding: np.ndarray, margin: np.ndarray, findings: tuple[str, ...]
) -> BaseTransform:
    means = np.zeros(len(findings), dtype=np.float64)
    stds = np.ones(len(findings), dtype=np.float64)
    for index, name in enumerate(findings):
        values = margin[finding == name]
        means[index] = values.mean()
        stds[index] = max(values.std(), 1e-6)
    return BaseTransform(findings, means, stds)


@dataclass
class CentroidTransform:
    base: BaseTransform
    visual_mean: np.ndarray
    visual_std: np.ndarray
    directions: np.ndarray
    score_mean: np.ndarray
    score_std: np.ndarray

    def raw_score(self, finding: np.ndarray, visual: np.ndarray) -> np.ndarray:
        standardized = (visual.astype(np.float64) - self.visual_mean) / self.visual_std
        index = np.argmax(self.base.one_hot(finding), axis=1)
        return np.einsum("nd,nd->n", standardized, self.directions[index])

    def design(
        self, finding: np.ndarray, margin: np.ndarray, visual: np.ndarray
    ) -> np.ndarray:
        one_hot = self.base.one_hot(finding)
        index = np.argmax(one_hot, axis=1)
        score = self.raw_score(finding, visual)
        scaled = (score - self.score_mean[index]) / self.score_std[index]
        return np.concatenate(
            [self.base.base_design(finding, margin), one_hot * scaled[:, None]], axis=1
        )


def fit_centroid(
    finding: np.ndarray,
    votes: np.ndarray,
    margin: np.ndarray,
    visual: np.ndarray,
    findings: tuple[str, ...],
) -> CentroidTransform:
    base = fit_base_transform(finding, margin, findings)
    x = visual.astype(np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0
    standardized = (x - mean) / std
    directions = np.zeros((len(findings), x.shape[1]), dtype=np.float64)
    for index, name in enumerate(findings):
        positive = standardized[(finding == name) & (votes == 3)]
        negative = standardized[(finding == name) & (votes == 0)]
        if len(positive) == 0 or len(negative) == 0:
            raise ValueError(f"centroid direction lacks clear classes: {name}")
        direction = positive.mean(axis=0) - negative.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm <= 1e-10:
            raise ValueError(f"zero centroid direction: {name}")
        directions[index] = direction / norm
    finding_index = np.argmax(base.one_hot(finding), axis=1)
    scores = np.einsum("nd,nd->n", standardized, directions[finding_index])
    score_mean = np.zeros(len(findings), dtype=np.float64)
    score_std = np.ones(len(findings), dtype=np.float64)
    for index, name in enumerate(findings):
        values = scores[finding == name]
        score_mean[index] = values.mean()
        score_std[index] = max(values.std(), 1e-6)
    return CentroidTransform(base, mean, std, directions, score_mean, score_std)


@dataclass
class PCATransform:
    base: BaseTransform
    visual_mean: np.ndarray
    visual_std: np.ndarray
    pca: PCA

    def design(
        self,
        finding: np.ndarray,
        margin: np.ndarray,
        visual: np.ndarray,
        components: int,
    ) -> np.ndarray:
        one_hot = self.base.one_hot(finding)
        standardized = (visual.astype(np.float64) - self.visual_mean) / self.visual_std
        projected = self.pca.transform(standardized)[:, :components]
        interaction = (one_hot[:, :, None] * projected[:, None, :]).reshape(
            len(finding), -1
        )
        return np.concatenate([self.base.base_design(finding, margin), interaction], axis=1)


def fit_pca(
    finding: np.ndarray,
    margin: np.ndarray,
    visual: np.ndarray,
    findings: tuple[str, ...],
) -> PCATransform:
    base = fit_base_transform(finding, margin, findings)
    x = visual.astype(np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0
    standardized = (x - mean) / std
    components = min(max(PCS), len(x) - 1, x.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=42)
    pca.fit(standardized)
    return PCATransform(base, mean, std, pca)


@dataclass
class NuisanceTransform:
    base: BaseTransform
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    categories: tuple[tuple[str, ...], ...]

    def vector(self, numeric: np.ndarray, categorical: np.ndarray) -> np.ndarray:
        blocks = [(numeric - self.numeric_mean) / self.numeric_std]
        for column, categories in enumerate(self.categories):
            lookup = {name: index for index, name in enumerate(categories)}
            encoded = np.zeros((len(numeric), len(categories)), dtype=np.float64)
            for row, value in enumerate(categorical[:, column]):
                if str(value) in lookup:
                    encoded[row, lookup[str(value)]] = 1.0
            blocks.append(encoded)
        return np.concatenate(blocks, axis=1)

    def design(
        self,
        finding: np.ndarray,
        margin: np.ndarray,
        numeric: np.ndarray,
        categorical: np.ndarray,
    ) -> np.ndarray:
        one_hot = self.base.one_hot(finding)
        values = self.vector(numeric, categorical)
        interaction = (one_hot[:, :, None] * values[:, None, :]).reshape(len(finding), -1)
        return np.concatenate([self.base.base_design(finding, margin), interaction], axis=1)


def fit_nuisance(data: dict[str, Any], rows: np.ndarray, findings: tuple[str, ...]) -> NuisanceTransform:
    numeric = data["nuisance_numeric"][rows].astype(np.float64)
    mean, std = numeric.mean(axis=0), numeric.std(axis=0)
    std[std < 1e-6] = 1.0
    categorical = data["nuisance_categorical"][rows]
    categories = tuple(
        tuple(sorted(set(str(value) for value in categorical[:, column])))
        for column in range(categorical.shape[1])
    )
    base = fit_base_transform(data["finding"][rows], data["margin"][rows], findings)
    return NuisanceTransform(base, mean, std, categories)


def select_models(data: dict[str, Any], findings: tuple[str, ...]) -> dict[str, Any]:
    baseline_scores = {
        c: np.full(len(data["votes"]), np.nan, dtype=np.float64) for c in REGULARIZATION
    }
    nuisance_scores = {
        c: np.full(len(data["votes"]), np.nan, dtype=np.float64) for c in REGULARIZATION
    }
    configurations: list[tuple[str, str, int, float]] = []
    for representation in REPRESENTATIONS:
        configurations.extend(
            (representation, "centroid_scalar", 1, c) for c in REGULARIZATION
        )
        configurations.extend(
            (representation, "pca_interaction", pcs, c)
            for pcs in PCS
            for c in REGULARIZATION
        )
    enhanced_scores = {
        config: np.full(len(data["votes"]), np.nan, dtype=np.float64)
        for config in configurations
    }
    for train, valid in grouped_folds(data):
        base = fit_base_transform(data["finding"][train], data["margin"][train], findings)
        x_train = base.base_design(data["finding"][train], data["margin"][train])
        x_valid = base.base_design(data["finding"][valid], data["margin"][valid])
        nuisance = fit_nuisance(data, train, findings)
        n_train = nuisance.design(
            data["finding"][train], data["margin"][train],
            data["nuisance_numeric"][train], data["nuisance_categorical"][train]
        )
        n_valid = nuisance.design(
            data["finding"][valid], data["margin"][valid],
            data["nuisance_numeric"][valid], data["nuisance_categorical"][valid]
        )
        for c_value in REGULARIZATION:
            baseline_scores[c_value][valid] = fit_logistic(
                x_train, data["votes"][train], c_value
            ).predict_proba(x_valid)[:, 1]
            nuisance_scores[c_value][valid] = fit_logistic(
                n_train, data["votes"][train], c_value
            ).predict_proba(n_valid)[:, 1]
        for representation in REPRESENTATIONS:
            visual = data["representations"][representation]
            centroid = fit_centroid(
                data["finding"][train], data["votes"][train], data["margin"][train],
                visual[train], findings
            )
            c_train = centroid.design(data["finding"][train], data["margin"][train], visual[train])
            c_valid = centroid.design(data["finding"][valid], data["margin"][valid], visual[valid])
            pca = fit_pca(data["finding"][train], data["margin"][train], visual[train], findings)
            for c_value in REGULARIZATION:
                config = (representation, "centroid_scalar", 1, c_value)
                enhanced_scores[config][valid] = fit_logistic(
                    c_train, data["votes"][train], c_value
                ).predict_proba(c_valid)[:, 1]
            for pcs in PCS:
                p_train = pca.design(data["finding"][train], data["margin"][train], visual[train], pcs)
                p_valid = pca.design(data["finding"][valid], data["margin"][valid], visual[valid], pcs)
                for c_value in REGULARIZATION:
                    config = (representation, "pca_interaction", pcs, c_value)
                    enhanced_scores[config][valid] = fit_logistic(
                        p_train, data["votes"][train], c_value
                    ).predict_proba(p_valid)[:, 1]
    base_nll = {str(c): float(reader_nll(data["votes"], p).mean()) for c, p in baseline_scores.items()}
    nuisance_nll = {str(c): float(reader_nll(data["votes"], p).mean()) for c, p in nuisance_scores.items()}
    enhanced_nll = {
        f"representation={r};family={f};size={s};C={c}": float(
            reader_nll(data["votes"], enhanced_scores[(r, f, s, c)]).mean()
        )
        for r, f, s, c in configurations
    }
    selected = min(
        configurations,
        key=lambda value: enhanced_nll[
            f"representation={value[0]};family={value[1]};size={value[2]};C={value[3]}"
        ],
    )
    return {
        "baseline_C": min(REGULARIZATION, key=lambda c: base_nll[str(c)]),
        "baseline_cv_nll": base_nll,
        "nuisance_C": min(REGULARIZATION, key=lambda c: nuisance_nll[str(c)]),
        "nuisance_cv_nll": nuisance_nll,
        "enhanced": {
            "representation": selected[0],
            "family": selected[1],
            "size": selected[2],
            "C": selected[3],
        },
        "enhanced_cv_nll": enhanced_nll,
    }


def fit_confirmation(
    dev: dict[str, Any], test: dict[str, Any], findings: tuple[str, ...], selection: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, LogisticRegression]:
    base = fit_base_transform(dev["finding"], dev["margin"], findings)
    x_dev = base.base_design(dev["finding"], dev["margin"])
    x_test = base.base_design(test["finding"], test["margin"])
    baseline = fit_logistic(x_dev, dev["votes"], selection["baseline_C"]).predict_proba(x_test)[:, 1]
    nuisance_transform = fit_nuisance(dev, np.arange(len(dev["votes"])), findings)
    n_dev = nuisance_transform.design(
        dev["finding"], dev["margin"], dev["nuisance_numeric"], dev["nuisance_categorical"]
    )
    n_test = nuisance_transform.design(
        test["finding"], test["margin"], test["nuisance_numeric"], test["nuisance_categorical"]
    )
    nuisance = fit_logistic(n_dev, dev["votes"], selection["nuisance_C"]).predict_proba(n_test)[:, 1]
    chosen = selection["enhanced"]
    visual_dev = dev["representations"][chosen["representation"]]
    visual_test = test["representations"][chosen["representation"]]
    if chosen["family"] == "centroid_scalar":
        transform = fit_centroid(
            dev["finding"], dev["votes"], dev["margin"], visual_dev, findings
        )
        e_dev = transform.design(dev["finding"], dev["margin"], visual_dev)
        e_test = transform.design(test["finding"], test["margin"], visual_test)
    else:
        transform = fit_pca(dev["finding"], dev["margin"], visual_dev, findings)
        e_dev = transform.design(dev["finding"], dev["margin"], visual_dev, chosen["size"])
        e_test = transform.design(test["finding"], test["margin"], visual_test, chosen["size"])
    model = fit_logistic(e_dev, dev["votes"], chosen["C"])
    enhanced = model.predict_proba(e_test)[:, 1]
    return baseline, nuisance, enhanced, transform, model


def paired_bootstrap(
    test: dict[str, Any], left: np.ndarray, right: np.ndarray, draws: int
) -> dict[str, Any]:
    images = np.unique(test["image"])
    groups = {image: np.flatnonzero(test["image"] == image) for image in images}
    left_nll, right_nll = reader_nll(test["votes"], left), reader_nll(test["votes"], right)
    left_brier, right_brier = brier(test["votes"], left), brier(test["votes"], right)
    rng = np.random.default_rng(20260811)
    nll_delta, brier_delta = [], []
    for _ in range(draws):
        selected = rng.choice(images, len(images), replace=True)
        rows = np.concatenate([groups[image] for image in selected])
        nll_delta.append(float(left_nll[rows].mean() - right_nll[rows].mean()))
        brier_delta.append(float(left_brier[rows].mean() - right_brier[rows].mean()))
    return {
        "draws": draws,
        "left_minus_right_nll_ci95": np.quantile(nll_delta, [0.025, 0.975]).tolist(),
        "left_minus_right_brier_ci95": np.quantile(brier_delta, [0.025, 0.975]).tolist(),
    }


def swap_control(
    test: dict[str, Any], selection: dict[str, Any], transform: Any,
    model: LogisticRegression, aligned: np.ndarray, permutations: int
) -> dict[str, Any]:
    chosen = selection["enhanced"]
    visual = test["representations"][chosen["representation"]]
    aligned_nll = float(reader_nll(test["votes"], aligned).mean())
    rng = np.random.default_rng(20260811)
    swap_nll = []
    for _ in range(permutations):
        swapped = visual.copy()
        for finding in sorted(set(test["finding"])):
            rows = np.flatnonzero(test["finding"] == finding)
            swapped[rows] = visual[rng.permutation(rows)]
        if chosen["family"] == "centroid_scalar":
            design = transform.design(test["finding"], test["margin"], swapped)
        else:
            design = transform.design(
                test["finding"], test["margin"], swapped, chosen["size"]
            )
        probability = model.predict_proba(design)[:, 1]
        swap_nll.append(float(reader_nll(test["votes"], probability).mean()))
    advantage = np.asarray(swap_nll) - aligned_nll
    return {
        "permutations": permutations,
        "aligned_nll": aligned_nll,
        "swap_nll_mean": float(np.mean(swap_nll)),
        "aligned_advantage_ci95": np.quantile(advantage, [0.025, 0.975]).tolist(),
        "fraction_swaps_beaten": float(np.mean(advantage > 0)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw = load_raw(args.raw)
    dev = join_raw(hidden_rows(args.dev), raw)
    findings = tuple(sorted(set(dev["finding"])))
    selection = select_models(dev, findings)
    test = join_raw(hidden_rows(args.confirmation), raw)
    if set(findings) != set(test["finding"]):
        raise ValueError("development/confirmation finding mismatch")
    if set(dev["image"]) & set(test["image"]):
        raise ValueError("development/confirmation image overlap")
    baseline, nuisance, enhanced, transform, model = fit_confirmation(
        dev, test, findings, selection
    )
    metrics = {
        "baseline": metric_summary(test, baseline),
        "nuisance_only": metric_summary(test, nuisance),
        "enhanced": metric_summary(test, enhanced),
    }
    base_nll = metrics["baseline"]["reader_nll"]
    base_brier = metrics["baseline"]["reader_support_brier"]
    nll_delta = base_nll - metrics["enhanced"]["reader_nll"]
    brier_delta = base_brier - metrics["enhanced"]["reader_support_brier"]
    positive_findings = sum(
        metrics["enhanced"]["by_finding"][finding]["nll"]
        < metrics["baseline"]["by_finding"][finding]["nll"]
        for finding in findings
    )
    base_bootstrap = paired_bootstrap(test, baseline, enhanced, args.bootstrap_draws)
    nuisance_bootstrap = paired_bootstrap(test, nuisance, enhanced, args.bootstrap_draws)
    swap = swap_control(test, selection, transform, model, enhanced, args.permutations)
    model_pass = bool(
        nll_delta / base_nll >= 0.05
        and base_bootstrap["left_minus_right_nll_ci95"][0] > 0
        and brier_delta / base_brier >= 0.05
        and base_bootstrap["left_minus_right_brier_ci95"][0] > 0
        and positive_findings >= 5
        and swap["fraction_swaps_beaten"] >= 0.95
        and metrics["nuisance_only"]["reader_nll"] > metrics["enhanced"]["reader_nll"]
        and nuisance_bootstrap["left_minus_right_nll_ci95"][0] > 0
    )
    return {
        "protocol": PROTOCOL,
        "status": "complete",
        "claim_boundary": (
            "PASS is incremental reader-vote decodability beyond final margin and simple "
            "image metadata. It is not localization, causality, controllability, "
            "correctability, hallucination mitigation, or an ICLR-ready contribution."
        ),
        "data": {
            "development": {
                "directory": dev["directory"], "n": len(dev["votes"]),
                "unique_images": int(len(np.unique(dev["image"]))),
                "metadata_sha256": dev["metadata_sha256"],
                "hidden_sha256": dev["hidden_sha256"],
            },
            "confirmation": {
                "directory": test["directory"], "n": len(test["votes"]),
                "unique_images": int(len(np.unique(test["image"]))),
                "metadata_sha256": test["metadata_sha256"],
                "hidden_sha256": test["hidden_sha256"],
            },
            "raw": {
                "directory": raw["directory"],
                "metadata_sha256": raw["metadata_sha256"],
                "features_sha256": raw["features_sha256"],
            },
            "findings": list(findings),
            "image_overlap": 0,
        },
        "development_selection": selection,
        "confirmation": {
            **metrics,
            "reader_nll_delta": nll_delta,
            "reader_nll_relative_improvement": nll_delta / base_nll,
            "reader_support_brier_delta": brier_delta,
            "reader_support_brier_relative_improvement": brier_delta / base_brier,
            "positive_nll_findings": int(positive_findings),
            "baseline_vs_enhanced_bootstrap": base_bootstrap,
            "nuisance_vs_enhanced_bootstrap": nuisance_bootstrap,
            "same_finding_image_swap": swap,
        },
        "raw_stage_model_gate": {
            "model_pass": model_pass,
            "requires_both_models": True,
            "pass_thresholds": {
                "relative_nll_improvement": 0.05,
                "relative_brier_improvement": 0.05,
                "positive_findings": 5,
                "bootstrap_lower_bound": 0.0,
                "fraction_swaps_beaten": 0.95,
                "beats_nuisance_nll_with_positive_bootstrap_lower_bound": True,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    compact = {
        "status": result["status"],
        "development_selection": result["development_selection"]["enhanced"],
        "confirmation": {
            key: value
            for key, value in result["confirmation"].items()
            if key not in {"baseline", "nuisance_only", "enhanced"}
        },
        "raw_stage_model_gate": result["raw_stage_model_gate"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
