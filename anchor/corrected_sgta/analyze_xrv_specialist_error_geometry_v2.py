#!/usr/bin/env python3
"""Explain when a frozen CXR specialist adds information to medical VLMs.

This is a CPU-only, post-hoc analysis of frozen VinDr manifests and the
already-computed 18-dimensional TorchXRayVision logits.  It does not touch the
baseline queue and does not claim a mitigation result.  The analysis answers
four narrow questions:

1. Which findings and VLM-confidence strata account for the XRV gain?
2. Does fusion repair false positives, false negatives, or merely move the
   operating point?
3. What happens when the VLM and specialist disagree?
4. Does the previously observed local-neighbour boundary still add information
   after exposing a linear or nonlinear model to all 18 XRV logits?

All models are fit on the frozen development split and evaluated once on the
image-disjoint confirmation split.  Bootstrap resampling is clustered by
image, because one image can contribute more than one claim.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.svm import SVC

PROTOCOL = "xrv-specialist-error-geometry-v2"
FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
XRV_LABELS = (
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
)
FINDING_TARGETS = {
    "aortic_enlargement": ("Enlarged Cardiomediastinum",),
    "cardiomegaly": ("Cardiomegaly",),
    "lung_opacity": ("Lung Opacity",),
    "nodule_mass": ("Nodule", "Mass", "Lung Lesion"),
    "pleural_effusion": ("Effusion",),
    "pleural_thickening": ("Pleural_Thickening",),
    "pulmonary_fibrosis": ("Fibrosis",),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_logits(path: Path) -> dict[str, np.ndarray]:
    pack = np.load(path, allow_pickle=False)
    labels = tuple(str(value) for value in pack["labels"])
    if labels != XRV_LABELS:
        raise ValueError(f"XRV label-order drift: {labels}")
    return {
        str(image_id): np.asarray(vector, dtype=np.float64)
        for image_id, vector in zip(pack["image_ids"], pack["logits"])
    }


def final_margin(row: dict[str, Any]) -> float:
    lens = row["diagnostic_plain_logit_lens"]
    final_layer = max(lens, key=lambda value: int(value))
    return float(lens[final_layer]["supported"] - lens[final_layer]["refuted"])


def load_claims(path: Path, split: str, model: str) -> list[dict[str, Any]]:
    rows = []
    for raw in read_jsonl(path):
        if raw["finding"] not in FINDINGS or int(raw["positive_votes"]) not in (0, 3):
            continue
        rows.append(
            {
                "image_id": raw["image_id"],
                "finding": raw["finding"],
                "label": int(raw["positive_votes"] == 3),
                "margin": final_margin(raw),
                "split": split,
                "model": model,
            }
        )
    return rows


def _standardized_logits(
    development: list[dict[str, Any]], logits: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    image_ids = sorted({row["image_id"] for row in development})
    matrix = np.stack([logits[image_id] for image_id in image_ids])
    mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {image_id: (value - mean) / scale for image_id, value in logits.items()}


def _neighbor_score(
    query: np.ndarray,
    banks: dict[int, tuple[np.ndarray, np.ndarray]],
    query_id: str,
    k: int,
) -> float:
    distances = {}
    for label in (0, 1):
        matrix, image_ids = banks[label]
        eligible = image_ids != query_id
        matrix = matrix[eligible]
        if not len(matrix):
            raise ValueError("Leave-one-image-out removed an entire neighbour class")
        values = np.linalg.norm(matrix - query[None, :], axis=1)
        count = min(k, len(values))
        distances[label] = float(np.partition(values, count - 1)[:count].mean())
    return distances[0] - distances[1]


def attach_geometry(
    development: list[dict[str, Any]],
    target: list[dict[str, Any]],
    logits: dict[str, np.ndarray],
    k: int,
    seed: int,
) -> None:
    standardized = _standardized_logits(development, logits)
    rng = np.random.default_rng(seed)
    for finding in FINDINGS:
        dev = [row for row in development if row["finding"] == finding]
        banks = {
            label: (
                np.stack([standardized[row["image_id"]] for row in dev if row["label"] == label]),
                np.asarray([row["image_id"] for row in dev if row["label"] == label]),
            )
            for label in (0, 1)
        }
        # Retain the shuffled bank calculation as a deterministic construction
        # audit, even though this analysis focuses on all-18D controls.
        shuffled = np.asarray([row["label"] for row in dev])
        rng.shuffle(shuffled)
        for row in target:
            if row["finding"] == finding:
                row["neighbor_boundary"] = _neighbor_score(
                    standardized[row["image_id"]], banks, row["image_id"], k
                )


def attach_xrv(rows: list[dict[str, Any]], logits: dict[str, np.ndarray]) -> None:
    indices = {name: index for index, name in enumerate(XRV_LABELS)}
    for row in rows:
        vector = logits[row["image_id"]]
        row["xrv_logits"] = vector
        row["xrv_scalar"] = float(
            max(vector[indices[target]] for target in FINDING_TARGETS[row["finding"]])
        )


def dev_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    output: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for finding in FINDINGS:
        selected = [row for row in rows if row["finding"] == finding]
        output[finding] = {}
        for key in ("margin", "xrv_scalar", "xrv_logits", "neighbor_boundary"):
            values = np.asarray([row[key] for row in selected], dtype=np.float64)
            mean = values.mean(axis=0)
            scale = values.std(axis=0)
            scale = np.where(scale < 1e-8, 1.0, scale)
            output[finding][key] = (mean, scale)
    return output


def z(row: dict[str, Any], stats, key: str) -> np.ndarray:
    mean, scale = stats[row["finding"]][key]
    return np.atleast_1d((np.asarray(row[key], dtype=np.float64) - mean) / scale)


def finding_dummies(row: dict[str, Any]) -> list[float]:
    return [float(row["finding"] == finding) for finding in FINDINGS[:-1]]


def design(rows: list[dict[str, Any]], stats, specification: str) -> np.ndarray:
    matrix: list[list[float]] = []
    for row in rows:
        onehot = finding_dummies(row)
        margin = z(row, stats, "margin").tolist()
        scalar = z(row, stats, "xrv_scalar").tolist()
        logits = z(row, stats, "xrv_logits").tolist()
        boundary = z(row, stats, "neighbor_boundary").tolist()
        if specification == "vlm_only":
            values = onehot + margin
        elif specification == "xrv_only":
            values = onehot + logits
        elif specification == "vlm_scalar":
            values = onehot + margin + scalar
        elif specification == "vlm_all18":
            values = onehot + margin + logits
        elif specification in ("vlm_all18_interactions", "vlm_all18_interactions_boundary"):
            full_onehot = [float(row["finding"] == finding) for finding in FINDINGS]
            interactions = [flag * value for flag in full_onehot for value in logits]
            values = onehot + margin + logits + interactions
            if specification.endswith("boundary"):
                values += boundary
        elif specification == "vlm_scalar_boundary":
            values = onehot + margin + scalar + boundary
        elif specification in ("vlm_all18_boundary", "vlm_rbf_all18", "vlm_rbf_all18_boundary"):
            values = onehot + margin + logits
            if specification.endswith("boundary"):
                values += boundary
        else:
            raise KeyError(specification)
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64)


def fit_predict(
    development: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    stats,
    specification: str,
    seed: int,
) -> np.ndarray:
    x_dev = design(development, stats, specification)
    x_test = design(confirmation, stats, specification)
    labels = np.asarray([row["label"] for row in development], dtype=np.int64)
    if specification.startswith("vlm_rbf"):
        # RBF is deliberately fixed before confirmation.  It is a strong
        # nonlinear control, not a test-set tuned candidate method.
        model = SVC(
            C=1.0,
            gamma="scale",
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        model = LogisticRegression(C=1.0, max_iter=5000, random_state=seed)
    model.fit(x_dev, labels)
    return model.predict_proba(x_test)[:, 1]


def macro_auc(rows: list[dict[str, Any]], scores: np.ndarray) -> float:
    values = []
    for finding in FINDINGS:
        indices = [index for index, row in enumerate(rows) if row["finding"] == finding]
        labels = np.asarray([rows[index]["label"] for index in indices])
        values.append(roc_auc_score(labels, scores[indices]))
    return float(np.mean(values))


def classification_counts(rows: list[dict[str, Any]], probability: np.ndarray) -> dict[str, int]:
    label = np.asarray([row["label"] for row in rows], dtype=np.int64)
    pred = probability >= 0.5
    return {
        "tp": int(np.sum(pred & (label == 1))),
        "tn": int(np.sum(~pred & (label == 0))),
        "fp": int(np.sum(pred & (label == 0))),
        "fn": int(np.sum(~pred & (label == 1))),
    }


def error_transitions(
    rows: list[dict[str, Any]], baseline: np.ndarray, enhanced: np.ndarray
) -> dict[str, int]:
    label = np.asarray([row["label"] for row in rows], dtype=np.int64)
    before, after = baseline >= 0.5, enhanced >= 0.5
    return {
        "fp_corrected": int(np.sum(before & (label == 0) & ~after)),
        "fp_created": int(np.sum(~before & (label == 0) & after)),
        "fn_corrected": int(np.sum(~before & (label == 1) & after)),
        "fn_created": int(np.sum(before & (label == 1) & ~after)),
        "all_errors_corrected": int(np.sum((before != label) & (after == label))),
        "new_errors_created": int(np.sum((before == label) & (after != label))),
    }


def metrics(rows: list[dict[str, Any]], probability: np.ndarray) -> dict[str, Any]:
    label = np.asarray([row["label"] for row in rows], dtype=np.int64)
    by_finding = {}
    for finding in FINDINGS:
        indices = np.asarray([i for i, row in enumerate(rows) if row["finding"] == finding])
        y, p = label[indices], probability[indices]
        by_finding[finding] = {
            "n": int(len(indices)),
            "auroc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "classification": classification_counts([rows[i] for i in indices], p),
        }
    return {
        "macro_auroc": macro_auc(rows, probability),
        "nll": float(log_loss(label, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(label, probability)),
        "classification": classification_counts(rows, probability),
        "by_finding": by_finding,
    }


def bootstrap_deltas(
    rows: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    references: dict[str, str],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["image_id"]].append(index)
    image_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in references}
    for _ in range(draws):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        indices = np.asarray([index for image_id in sampled for index in groups[image_id]])
        sampled_rows = [rows[index] for index in indices]
        for name, reference in references.items():
            try:
                delta = macro_auc(sampled_rows, predictions[name][indices]) - macro_auc(
                    sampled_rows, predictions[reference][indices]
                )
            except ValueError:
                continue
            values[name].append(delta)
    return {
        name: {
            "reference": references[name],
            "mean": float(np.mean(current)),
            "ci95": [float(np.quantile(current, 0.025)), float(np.quantile(current, 0.975))],
            "valid_draws": len(current),
        }
        for name, current in values.items()
    }


def margin_bins(
    development: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    stats,
    base: np.ndarray,
    specialist: np.ndarray,
) -> dict[str, Any]:
    dev_abs = np.asarray([abs(float(z(row, stats, "margin")[0])) for row in development])
    edges = np.quantile(dev_abs, [0.25, 0.5, 0.75]).tolist()
    test_abs = np.asarray([abs(float(z(row, stats, "margin")[0])) for row in confirmation])
    bins = np.digitize(test_abs, edges, right=True)
    label = np.asarray([row["label"] for row in confirmation], dtype=np.int64)
    output = {"dev_abs_margin_quartile_edges": edges, "bins": {}}
    for bin_index in range(4):
        indices = np.flatnonzero(bins == bin_index)
        y, p0, p1 = label[indices], base[indices], specialist[indices]
        output["bins"][str(bin_index + 1)] = {
            "n": int(len(indices)),
            "mean_abs_margin_z": float(test_abs[indices].mean()),
            "base_accuracy": float(np.mean((p0 >= 0.5) == y)),
            "specialist_accuracy": float(np.mean((p1 >= 0.5) == y)),
            "accuracy_delta": float(np.mean((p1 >= 0.5) == y) - np.mean((p0 >= 0.5) == y)),
            "nll_improvement": float(
                log_loss(y, p0, labels=[0, 1]) - log_loss(y, p1, labels=[0, 1])
            ),
            "error_transitions": error_transitions(
                [confirmation[index] for index in indices], p0, p1
            ),
        }
    return output


def disagreement_panel(
    rows: list[dict[str, Any]],
    vlm_probability: np.ndarray,
    xrv_probability: np.ndarray,
    fused_probability: np.ndarray,
) -> dict[str, Any]:
    label = np.asarray([row["label"] for row in rows], dtype=np.int64)
    vlm = vlm_probability >= 0.5
    xrv = xrv_probability >= 0.5
    groups = {
        "both_negative": ~vlm & ~xrv,
        "vlm_positive_xrv_negative": vlm & ~xrv,
        "vlm_negative_xrv_positive": ~vlm & xrv,
        "both_positive": vlm & xrv,
    }
    output = {}
    for name, mask in groups.items():
        indices = np.flatnonzero(mask)
        y = label[indices]
        output[name] = {
            "n": int(len(indices)),
            "positive_prevalence": float(y.mean()) if len(y) else None,
            "vlm_accuracy": float(np.mean(vlm[indices] == y)) if len(y) else None,
            "xrv_accuracy": float(np.mean(xrv[indices] == y)) if len(y) else None,
            "fused_accuracy": float(np.mean((fused_probability[indices] >= 0.5) == y))
            if len(y)
            else None,
        }
    return output


def reader_vote_panel(
    development_path: Path,
    confirmation_path: Path,
    logits: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    def load(path: Path) -> list[dict[str, Any]]:
        rows = []
        for raw in read_jsonl(path):
            if raw["finding"] not in FINDINGS or raw["image_id"] not in logits:
                continue
            rows.append(
                {
                    "image_id": raw["image_id"],
                    "finding": raw["finding"],
                    "reader_support": float(raw["positive_votes"]) / 3.0,
                    "positive_votes": int(raw["positive_votes"]),
                    "margin": final_margin(raw),
                }
            )
        attach_xrv(rows, logits)
        return rows

    dev, test = load(development_path), load(confirmation_path)
    # Finding-specific z-scoring removes easy between-finding offsets before
    # asking whether a score tracks the 0/1/2/3 reader continuum.
    stats = {}
    for finding in FINDINGS:
        selected = [row for row in dev if row["finding"] == finding]
        stats[finding] = {}
        for key in ("margin", "xrv_scalar"):
            values = np.asarray([row[key] for row in selected])
            stats[finding][key] = (float(values.mean()), float(max(values.std(), 1e-8)))

    def matrix(rows, enhanced):
        result = []
        for row in rows:
            onehot = finding_dummies(row)
            m_mean, m_scale = stats[row["finding"]]["margin"]
            x_mean, x_scale = stats[row["finding"]]["xrv_scalar"]
            values = onehot + [(row["margin"] - m_mean) / m_scale]
            if enhanced:
                values.append((row["xrv_scalar"] - x_mean) / x_scale)
            result.append(values)
        return np.asarray(result)

    y_dev = np.asarray([row["reader_support"] for row in dev])
    y_test = np.asarray([row["reader_support"] for row in test])
    base_model, enhanced_model = Ridge(alpha=1.0), Ridge(alpha=1.0)
    base_model.fit(matrix(dev, False), y_dev)
    enhanced_model.fit(matrix(dev, True), y_dev)
    base = np.clip(base_model.predict(matrix(test, False)), 0, 1)
    enhanced = np.clip(enhanced_model.predict(matrix(test, True)), 0, 1)

    margin_z = np.asarray(
        [
            (row["margin"] - stats[row["finding"]]["margin"][0])
            / stats[row["finding"]]["margin"][1]
            for row in test
        ]
    )
    xrv_z = np.asarray(
        [
            (row["xrv_scalar"] - stats[row["finding"]]["xrv_scalar"][0])
            / stats[row["finding"]]["xrv_scalar"][1]
            for row in test
        ]
    )
    by_vote = {}
    for votes in range(4):
        indices = [index for index, row in enumerate(test) if row["positive_votes"] == votes]
        by_vote[str(votes)] = {
            "n": len(indices),
            "mean_vlm_margin_z": float(margin_z[indices].mean()),
            "mean_xrv_scalar_z": float(xrv_z[indices].mean()),
        }
    return {
        "scope_warning": (
            "Exploratory only: XRV cache images were selected through unanimous claims of at least "
            "one finding, so intermediate-vote claims are not an independently sampled reader panel."
        ),
        "development_n": len(dev),
        "confirmation_n": len(test),
        "confirmation_vote_counts": dict(sorted(Counter(row["positive_votes"] for row in test).items())),
        "by_vote": by_vote,
        "spearman_reader_support": {
            "vlm_margin_z": float(spearmanr(y_test, margin_z).statistic),
            "xrv_scalar_z": float(spearmanr(y_test, xrv_z).statistic),
        },
        "reader_support_mse": {
            "vlm_only": float(np.mean((base - y_test) ** 2)),
            "vlm_plus_xrv": float(np.mean((enhanced - y_test) ** 2)),
            "improvement": float(np.mean((base - y_test) ** 2) - np.mean((enhanced - y_test) ** 2)),
        },
    }


def analyze_model(
    development_path: Path,
    confirmation_path: Path,
    model_name: str,
    logits: dict[str, np.ndarray],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    development = load_claims(development_path, "development", model_name)
    confirmation = load_claims(confirmation_path, "confirmation", model_name)
    attach_xrv(development + confirmation, logits)
    # Same development-only local label banks as the earlier upper-bound test.
    attach_geometry(development, development + confirmation, logits, k=3, seed=seed)
    stats = dev_stats(development)

    specifications = (
        "vlm_only",
        "xrv_only",
        "vlm_scalar",
        "vlm_all18",
        "vlm_all18_interactions",
        "vlm_all18_interactions_boundary",
        "vlm_scalar_boundary",
        "vlm_all18_boundary",
        "vlm_rbf_all18",
        "vlm_rbf_all18_boundary",
    )
    predictions = {
        name: fit_predict(development, confirmation, stats, name, seed) for name in specifications
    }
    points = {name: metrics(confirmation, values) for name, values in predictions.items()}
    by_finding_deltas = {}
    for finding in FINDINGS:
        by_finding_deltas[finding] = {
            name: points[name]["by_finding"][finding]["auroc"]
            - points["vlm_only"]["by_finding"][finding]["auroc"]
            for name in specifications
            if name != "vlm_only"
        }

    references = {
        "vlm_scalar": "vlm_only",
        "vlm_all18": "vlm_scalar",
        "vlm_all18_interactions": "vlm_scalar",
        "vlm_all18_interactions_boundary": "vlm_all18_interactions",
        "vlm_scalar_boundary": "vlm_scalar",
        "vlm_all18_boundary": "vlm_all18",
        "vlm_rbf_all18": "vlm_scalar",
        "vlm_rbf_all18_boundary": "vlm_rbf_all18",
    }
    return {
        "development_n": len(development),
        "confirmation_n": len(confirmation),
        "points": points,
        "by_finding_auroc_deltas_over_vlm": by_finding_deltas,
        "scalar_fusion_error_transitions": error_transitions(
            confirmation, predictions["vlm_only"], predictions["vlm_scalar"]
        ),
        "margin_confidence_bins": margin_bins(
            development,
            confirmation,
            stats,
            predictions["vlm_only"],
            predictions["vlm_scalar"],
        ),
        "specialist_disagreement": disagreement_panel(
            confirmation,
            predictions["vlm_only"],
            predictions["xrv_only"],
            predictions["vlm_scalar"],
        ),
        "image_cluster_bootstrap_auroc_deltas": bootstrap_deltas(
            confirmation, predictions, references, draws, seed
        ),
        "reader_votes_exploratory": reader_vote_panel(
            development_path, confirmation_path, logits, seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-dev", type=Path, required=True)
    parser.add_argument("--huatuo-confirmation", type=Path, required=True)
    parser.add_argument("--hulu-dev", type=Path, required=True)
    parser.add_argument("--hulu-confirmation", type=Path, required=True)
    parser.add_argument("--xrv-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise RuntimeError("CPU-only analysis; set CUDA_VISIBLE_DEVICES=''")
    if args.output.exists():
        raise FileExistsError(args.output)

    logits = load_logits(args.xrv_logits)
    analyses = {
        "huatuo": analyze_model(
            args.huatuo_dev,
            args.huatuo_confirmation,
            "huatuo",
            logits,
            args.draws,
            args.seed,
        ),
        "hulu": analyze_model(
            args.hulu_dev,
            args.hulu_confirmation,
            "hulu",
            logits,
            args.draws,
            args.seed,
        ),
    }
    result = {
        "status": "complete_cpu_posthoc",
        "protocol": PROTOCOL,
        "scope": (
            "Post-hoc mechanism analysis of frozen specialist logits. This does not establish a "
            "new mitigation method or generalize beyond chest radiography."
        ),
        "question": "Why does XRV add much more to Huatuo than to Hulu?",
        "analyses": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
