#!/usr/bin/env python3
"""Test whether cached visual-token states add case-aligned reader information.

This is the CPU-only Stage-1 pre-screen in the Evidence Addressability
preregistration.  It fits exclusively on the development artifact, selects all
hyperparameters by image-grouped CV, and opens the image-disjoint confirmation
artifact once.  A positive result is only incremental decodability; it is not
localization, causality, controllability, or mitigation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


PROTOCOL = "evidence-addressability-cached-visual-increment-v1"
PCS = (8, 16, 32)
REGULARIZATION = (0.01, 0.1, 1.0, 10.0)
FOLDS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact(directory: Path) -> dict[str, Any]:
    metadata_path = directory / "metadata.jsonl"
    hidden_path = directory / "hidden_states.npz"
    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with np.load(hidden_path, allow_pickle=False) as archive:
        layers = np.asarray(archive["layers"], dtype=np.int64)
        visual = np.asarray(archive["visual_mean"], dtype=np.float32)
    if visual.shape[:2] != (len(rows), len(layers)):
        raise ValueError(f"{directory}: hidden-state/metadata shape mismatch")
    findings = np.asarray([str(row["finding"]) for row in rows])
    votes = np.asarray([int(row["positive_votes"]) for row in rows], dtype=np.int64)
    images = np.asarray([str(row["image_id"]) for row in rows])
    final_layer = int(layers[-1])
    margins = []
    for row in rows:
        logits = row["diagnostic_plain_logit_lens"][str(final_layer)]
        margins.append(float(logits["supported"]) - float(logits["refuted"]))
    return {
        "directory": str(directory),
        "metadata_sha256": sha256(metadata_path),
        "hidden_sha256": sha256(hidden_path),
        "rows": rows,
        "layers": layers,
        "visual": visual,
        "finding": findings,
        "votes": votes,
        "image": images,
        "margin": np.asarray(margins, dtype=np.float64),
    }


def reader_nll(votes: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability.astype(np.float64), 1e-7, 1.0 - 1e-7)
    support = votes.astype(np.float64) / 3.0
    return -(support * np.log(p) + (1.0 - support) * np.log(1.0 - p))


def brier(votes: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return (probability.astype(np.float64) - votes.astype(np.float64) / 3.0) ** 2


def expand_reader_trials(
    design: np.ndarray, votes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([design, design], axis=0)
    y = np.concatenate(
        [np.ones(len(design), dtype=np.int64), np.zeros(len(design), dtype=np.int64)]
    )
    weights = np.concatenate([votes, 3 - votes]).astype(np.float64)
    keep = weights > 0
    return x[keep], y[keep], weights[keep]


def fit_logistic(design: np.ndarray, votes: np.ndarray, c_value: float) -> LogisticRegression:
    x, y, weights = expand_reader_trials(design, votes)
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        fit_intercept=False,
        solver="liblinear",
        max_iter=4000,
        random_state=42,
    )
    model.fit(x, y, sample_weight=weights)
    return model


@dataclass
class Transform:
    findings: tuple[str, ...]
    margin_mean: np.ndarray
    margin_std: np.ndarray
    visual_mean: np.ndarray | None = None
    visual_std: np.ndarray | None = None
    pca: PCA | None = None

    def _one_hot(self, finding: np.ndarray) -> np.ndarray:
        index = {name: position for position, name in enumerate(self.findings)}
        output = np.zeros((len(finding), len(self.findings)), dtype=np.float64)
        for row, name in enumerate(finding):
            output[row, index[str(name)]] = 1.0
        return output

    def design(
        self,
        finding: np.ndarray,
        margin: np.ndarray,
        visual: np.ndarray | None = None,
        components: int | None = None,
    ) -> np.ndarray:
        one_hot = self._one_hot(finding)
        finding_index = np.argmax(one_hot, axis=1)
        scaled_margin = (
            margin - self.margin_mean[finding_index]
        ) / self.margin_std[finding_index]
        blocks = [one_hot, one_hot * scaled_margin[:, None]]
        if visual is not None:
            if self.visual_mean is None or self.visual_std is None or self.pca is None:
                raise ValueError("visual transform was not fitted")
            if components is None:
                raise ValueError("components required for visual design")
            standardized = (visual.astype(np.float64) - self.visual_mean) / self.visual_std
            projected = self.pca.transform(standardized)[:, :components]
            interaction = (one_hot[:, :, None] * projected[:, None, :]).reshape(
                len(one_hot), -1
            )
            blocks.append(interaction)
        return np.concatenate(blocks, axis=1)


def fit_transform(
    finding: np.ndarray,
    margin: np.ndarray,
    visual: np.ndarray | None,
    findings: tuple[str, ...],
    max_components: int,
) -> Transform:
    means = np.zeros(len(findings), dtype=np.float64)
    stds = np.ones(len(findings), dtype=np.float64)
    for index, name in enumerate(findings):
        values = margin[finding == name]
        means[index] = values.mean()
        stds[index] = max(values.std(), 1e-6)
    transform = Transform(findings=findings, margin_mean=means, margin_std=stds)
    if visual is not None:
        x = visual.astype(np.float64)
        transform.visual_mean = x.mean(axis=0)
        transform.visual_std = x.std(axis=0)
        transform.visual_std[transform.visual_std < 1e-6] = 1.0
        standardized = (x - transform.visual_mean) / transform.visual_std
        count = min(max_components, standardized.shape[0] - 1, standardized.shape[1])
        transform.pca = PCA(
            n_components=count,
            svd_solver="randomized",
            random_state=42,
        ).fit(standardized)
    return transform


def metric_summary(data: dict[str, Any], probability: np.ndarray) -> dict[str, Any]:
    nll_values = reader_nll(data["votes"], probability)
    brier_values = brier(data["votes"], probability)
    by_finding: dict[str, Any] = {}
    aucs = []
    for finding in sorted(set(data["finding"])):
        mask = data["finding"] == finding
        clear = mask & np.isin(data["votes"], (0, 3))
        auc = float(roc_auc_score((data["votes"][clear] == 3).astype(int), probability[clear]))
        aucs.append(auc)
        by_finding[finding] = {
            "n": int(mask.sum()),
            "nll": float(nll_values[mask].mean()),
            "brier": float(brier_values[mask].mean()),
            "clear_auroc": auc,
        }
    return {
        "n": len(probability),
        "reader_nll": float(nll_values.mean()),
        "reader_support_brier": float(brier_values.mean()),
        "clear_macro_auroc": float(np.mean(aucs)),
        "by_finding": by_finding,
    }


def grouped_folds(data: dict[str, Any]) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    strata = np.asarray(
        [f"{finding}:{vote}" for finding, vote in zip(data["finding"], data["votes"])]
    )
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    return splitter.split(np.zeros(len(strata)), strata, groups=data["image"])


def select_baseline(data: dict[str, Any], findings: tuple[str, ...]) -> tuple[float, dict[str, float]]:
    scores: dict[float, list[np.ndarray]] = {
        value: np.full(len(data["votes"]), np.nan, dtype=np.float64)
        for value in REGULARIZATION
    }
    for train, valid in grouped_folds(data):
        transform = fit_transform(
            data["finding"][train], data["margin"][train], None, findings, max(PCS)
        )
        x_train = transform.design(data["finding"][train], data["margin"][train])
        x_valid = transform.design(data["finding"][valid], data["margin"][valid])
        for c_value in REGULARIZATION:
            model = fit_logistic(x_train, data["votes"][train], c_value)
            scores[c_value][valid] = model.predict_proba(x_valid)[:, 1]
    nlls = {str(value): float(reader_nll(data["votes"], pred).mean()) for value, pred in scores.items()}
    selected = min(REGULARIZATION, key=lambda value: nlls[str(value)])
    return float(selected), nlls


def select_enhanced(
    data: dict[str, Any], findings: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, float]]:
    configurations = [
        (layer_index, int(layer), pcs, c_value)
        for layer_index, layer in enumerate(data["layers"])
        for pcs in PCS
        for c_value in REGULARIZATION
    ]
    scores = {
        config: np.full(len(data["votes"]), np.nan, dtype=np.float64)
        for config in configurations
    }
    for train, valid in grouped_folds(data):
        for layer_index, layer in enumerate(data["layers"]):
            visual_train = data["visual"][train, layer_index]
            visual_valid = data["visual"][valid, layer_index]
            transform = fit_transform(
                data["finding"][train],
                data["margin"][train],
                visual_train,
                findings,
                max(PCS),
            )
            for pcs in PCS:
                x_train = transform.design(
                    data["finding"][train], data["margin"][train], visual_train, pcs
                )
                x_valid = transform.design(
                    data["finding"][valid], data["margin"][valid], visual_valid, pcs
                )
                for c_value in REGULARIZATION:
                    config = (layer_index, int(layer), pcs, c_value)
                    model = fit_logistic(x_train, data["votes"][train], c_value)
                    scores[config][valid] = model.predict_proba(x_valid)[:, 1]
    nlls = {
        f"layer={layer};pcs={pcs};C={c_value}": float(
            reader_nll(data["votes"], scores[(layer_index, layer, pcs, c_value)]).mean()
        )
        for layer_index, layer, pcs, c_value in configurations
    }
    selected = min(configurations, key=lambda config: nlls[f"layer={config[1]};pcs={config[2]};C={config[3]}"])
    return {
        "layer_index": int(selected[0]),
        "layer": int(selected[1]),
        "pcs": int(selected[2]),
        "C": float(selected[3]),
    }, nlls


def fit_and_predict(
    dev: dict[str, Any],
    test: dict[str, Any],
    findings: tuple[str, ...],
    baseline_c: float,
    enhanced: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Transform, LogisticRegression]:
    baseline_transform = fit_transform(
        dev["finding"], dev["margin"], None, findings, max(PCS)
    )
    baseline_train = baseline_transform.design(dev["finding"], dev["margin"])
    baseline_test = baseline_transform.design(test["finding"], test["margin"])
    baseline_model = fit_logistic(baseline_train, dev["votes"], baseline_c)
    baseline_probability = baseline_model.predict_proba(baseline_test)[:, 1]

    layer_index = int(enhanced["layer_index"])
    visual_train = dev["visual"][:, layer_index]
    visual_test = test["visual"][:, layer_index]
    visual_transform = fit_transform(
        dev["finding"], dev["margin"], visual_train, findings, max(PCS)
    )
    enhanced_train = visual_transform.design(
        dev["finding"], dev["margin"], visual_train, int(enhanced["pcs"])
    )
    enhanced_test = visual_transform.design(
        test["finding"], test["margin"], visual_test, int(enhanced["pcs"])
    )
    enhanced_model = fit_logistic(enhanced_train, dev["votes"], float(enhanced["C"]))
    enhanced_probability = enhanced_model.predict_proba(enhanced_test)[:, 1]
    return baseline_probability, enhanced_probability, visual_transform, enhanced_model


def bootstrap_delta(
    test: dict[str, Any], baseline: np.ndarray, enhanced: np.ndarray, draws: int
) -> dict[str, Any]:
    unique = np.unique(test["image"])
    indices = {image: np.flatnonzero(test["image"] == image) for image in unique}
    base_nll = reader_nll(test["votes"], baseline)
    enhanced_nll = reader_nll(test["votes"], enhanced)
    base_brier = brier(test["votes"], baseline)
    enhanced_brier = brier(test["votes"], enhanced)
    rng = np.random.default_rng(20260811)
    nll_delta = []
    brier_delta = []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        take = np.concatenate([indices[image] for image in sampled])
        nll_delta.append(float(base_nll[take].mean() - enhanced_nll[take].mean()))
        brier_delta.append(float(base_brier[take].mean() - enhanced_brier[take].mean()))
    return {
        "draws": draws,
        "reader_nll_delta_ci95": np.quantile(nll_delta, [0.025, 0.975]).tolist(),
        "reader_support_brier_delta_ci95": np.quantile(brier_delta, [0.025, 0.975]).tolist(),
    }


def aligned_swap_control(
    test: dict[str, Any],
    transform: Transform,
    model: LogisticRegression,
    enhanced: dict[str, Any],
    aligned_probability: np.ndarray,
    permutations: int,
) -> dict[str, Any]:
    layer_index = int(enhanced["layer_index"])
    visual = test["visual"][:, layer_index]
    aligned_nll = float(reader_nll(test["votes"], aligned_probability).mean())
    rng = np.random.default_rng(20260811)
    swap_nll = []
    for _ in range(permutations):
        swapped = visual.copy()
        for finding in sorted(set(test["finding"])):
            rows = np.flatnonzero(test["finding"] == finding)
            swapped[rows] = visual[rng.permutation(rows)]
        design = transform.design(
            test["finding"], test["margin"], swapped, int(enhanced["pcs"])
        )
        probability = model.predict_proba(design)[:, 1]
        swap_nll.append(float(reader_nll(test["votes"], probability).mean()))
    aligned_advantage = np.asarray(swap_nll) - aligned_nll
    return {
        "permutations": permutations,
        "aligned_reader_nll": aligned_nll,
        "swap_reader_nll_mean": float(np.mean(swap_nll)),
        "aligned_advantage_over_swap_ci95": np.quantile(
            aligned_advantage, [0.025, 0.975]
        ).tolist(),
        "fraction_swaps_beaten_by_aligned": float(np.mean(aligned_advantage > 0)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dev = load_artifact(args.dev)
    test = load_artifact(args.confirmation)
    if list(dev["layers"]) != list(test["layers"]):
        raise ValueError("development and confirmation layers differ")
    if set(dev["finding"]) != set(test["finding"]):
        raise ValueError("development and confirmation findings differ")
    overlap = set(dev["image"]) & set(test["image"])
    if overlap:
        raise ValueError(f"development/confirmation image overlap: {len(overlap)}")
    findings = tuple(sorted(set(dev["finding"])))
    baseline_c, baseline_cv = select_baseline(dev, findings)
    enhanced_config, enhanced_cv = select_enhanced(dev, findings)
    baseline, enhanced, transform, model = fit_and_predict(
        dev, test, findings, baseline_c, enhanced_config
    )
    baseline_metrics = metric_summary(test, baseline)
    enhanced_metrics = metric_summary(test, enhanced)
    nll_delta = baseline_metrics["reader_nll"] - enhanced_metrics["reader_nll"]
    brier_delta = (
        baseline_metrics["reader_support_brier"] - enhanced_metrics["reader_support_brier"]
    )
    positive_findings = sum(
        enhanced_metrics["by_finding"][finding]["nll"]
        < baseline_metrics["by_finding"][finding]["nll"]
        for finding in findings
    )
    bootstrap = bootstrap_delta(test, baseline, enhanced, args.bootstrap_draws)
    swap = aligned_swap_control(
        test, transform, model, enhanced_config, enhanced, args.permutations
    )
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "claim_boundary": (
            "CPU-only cached decoder visual-token pre-screen. PASS means only "
            "incremental aligned decodability and authorizes raw vision/projector "
            "collection; it does not establish localization, causality, control, "
            "correctability, mitigation, or an ICLR contribution."
        ),
        "data": {
            "development": {
                "directory": dev["directory"],
                "n": len(dev["rows"]),
                "unique_images": int(len(np.unique(dev["image"]))),
                "metadata_sha256": dev["metadata_sha256"],
                "hidden_sha256": dev["hidden_sha256"],
            },
            "confirmation": {
                "directory": test["directory"],
                "n": len(test["rows"]),
                "unique_images": int(len(np.unique(test["image"]))),
                "metadata_sha256": test["metadata_sha256"],
                "hidden_sha256": test["hidden_sha256"],
            },
            "findings": list(findings),
            "layers": dev["layers"].tolist(),
            "image_overlap": 0,
        },
        "development_selection": {
            "grouped_folds": FOLDS,
            "baseline_C": baseline_c,
            "baseline_cv_reader_nll": baseline_cv,
            "enhanced": enhanced_config,
            "enhanced_cv_reader_nll": enhanced_cv,
        },
        "confirmation": {
            "baseline": baseline_metrics,
            "enhanced": enhanced_metrics,
            "reader_nll_delta": nll_delta,
            "reader_nll_relative_improvement": nll_delta / baseline_metrics["reader_nll"],
            "reader_support_brier_delta": brier_delta,
            "reader_support_brier_relative_improvement": brier_delta
            / baseline_metrics["reader_support_brier"],
            "positive_nll_findings": int(positive_findings),
            "bootstrap": bootstrap,
            "same_finding_image_swap": swap,
        },
        "cached_stage_gate": {
            "requires_both_models": True,
            "model_pass": bool(
                nll_delta / baseline_metrics["reader_nll"] >= 0.05
                and bootstrap["reader_nll_delta_ci95"][0] > 0
                and brier_delta / baseline_metrics["reader_support_brier"] >= 0.05
                and bootstrap["reader_support_brier_delta_ci95"][0] > 0
                and positive_findings >= 5
                and swap["fraction_swaps_beaten_by_aligned"] >= 0.95
            ),
            "failure_action": (
                "Do not tune cached decoder features. Run one minimal raw "
                "vision/projector confirmation before closing the internal route."
            ),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    compact = {
        "status": result["status"],
        "confirmation": {
            key: value
            for key, value in result["confirmation"].items()
            if key not in {"baseline", "enhanced"}
        },
        "cached_stage_gate": result["cached_stage_gate"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
