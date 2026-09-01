#!/usr/bin/env python3
"""CPU-only competition fast paths with anti-shortcut controls.

The three analyses deliberately separate a useful score upper bound from a
scientific mechanism claim:

1. CE operating-point specificity: does a method change the *right* examples,
   beyond a label-blind placebo with identical 0->1 / 1->0 flip counts?
2. VinDr model complementarity: is two-model gain global, finding-specific, or
   concentrated in small lesions?  Bounding-box-dependent results are labelled
   oracle diagnostics because boxes are not available at inference.
3. Visual-MIMIC fixed-K claim proxy: compare reports at equal extracted clinical
   claim count, while separately reporting insufficient-claim coverage.

No source generations are modified and no GPU is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from corrected_sgta.analyze_sparse_lesion_boundary_v1 import (
    load_box_summary,
    load_dimensions,
)


VERSION = "competition-fastpaths-v3"
SEED = 20260812
BOOTSTRAP = 5000
ROOT = Path("/home/dbw/ANCHOR")
MATRIX = ROOT / "corrected_runs/paper_baselines_v1/full_matrix_v1"
VINDR = ROOT / "corrected_runs/vindr_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y, pred = y.astype(int), pred.astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return {
        "n": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(np.mean(y == pred)),
        "yes_rate": float(np.mean(pred)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def grouped_bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    sampled = rng.choice(unique, len(unique), replace=True)
    return np.concatenate([by_group[group] for group in sampled])


def ci(values: Iterable[float]) -> list[float]:
    return np.quantile(np.asarray(list(values), dtype=float), [0.025, 0.975]).tolist()


def expected_direction_matched_placebo(
    y: np.ndarray, base: np.ndarray, target: np.ndarray
) -> float:
    """Expected BAcc after label-blind flips with target's directional counts."""

    up = int(np.sum((base == 0) & (target == 1)))
    down = int(np.sum((base == 1) & (target == 0)))
    b0, b1 = base == 0, base == 1
    y0, y1 = y == 0, y == 1
    # Expected confusion after choosing `up` uniformly among base negatives and
    # `down` uniformly among base positives.  This preserves target yes-rate and
    # both directional flip counts without using labels.
    p_up = up / max(1, int(np.sum(b0)))
    p_down = down / max(1, int(np.sum(b1)))
    expected_tp = np.sum(y1 & b1) * (1 - p_down) + np.sum(y1 & b0) * p_up
    expected_tn = np.sum(y0 & b0) * (1 - p_up) + np.sum(y0 & b1) * p_down
    tpr = expected_tp / max(1, int(np.sum(y1)))
    tnr = expected_tn / max(1, int(np.sum(y0)))
    return float(0.5 * (tpr + tnr))


def analyze_ce_operating_point() -> dict[str, Any]:
    methods = ["AvisC", "DoLa", "OPERA", "PAI", "VCD", "VISTA"]
    eval_paths = {
        method: MATRIX / "derived_scores/llava" / method / "cxr_vishal/evaluation_ce_v7.json"
        for method in methods
    }
    details = {method: json.loads(path.read_text())["details"] for method, path in eval_paths.items()}
    qids = [[row["question_id"] for row in rows] for rows in details.values()]
    if any(order != qids[0] for order in qids[1:]):
        raise ValueError("CXR-VisHal method evaluation orders differ")

    keep = []
    for index in range(len(qids[0])):
        rows = [details[method][index] for method in methods]
        gt = rows[0]["ground_truth"]
        # The benchmark's strict leading-decision parser marks complete clinical
        # sentences invalid (e.g. "The image does not show ...").  For this
        # operating-point diagnostic only, use the already-frozen legacy
        # semantic binary field and require it to be parseable for every method.
        predictions = [row["legacy_semantic_prediction"] for row in rows]
        keep.append(
            rows[0]["answer_type"] == "binary"
            and gt in (["yes"], ["no"])
            and all(prediction in ("yes", "no") for prediction in predictions)
        )
    indices = np.flatnonzero(keep)
    y = np.asarray([
        int(details[methods[0]][index]["ground_truth"] == ["yes"])
        for index in indices
    ])
    groups = np.asarray([details[methods[0]][index]["cluster_id"] for index in indices])
    pred = {
        method: np.asarray([int(details[method][index]["legacy_semantic_prediction"] == "yes") for index in indices])
        for method in methods
    }
    base_metrics = {method: metrics(y, value) for method, value in pred.items()}
    best = max(methods, key=lambda name: base_metrics[name]["balanced_accuracy"])

    pairwise = {}
    rng = np.random.default_rng(SEED)
    for target in methods:
        if target == best:
            continue
        base, other = pred[best], pred[target]
        observed_delta = metrics(y, other)["balanced_accuracy"] - metrics(y, base)["balanced_accuracy"]
        placebo_delta = expected_direction_matched_placebo(y, base, other) - metrics(y, base)["balanced_accuracy"]
        boot_observed, boot_excess = [], []
        for _ in range(BOOTSTRAP):
            sample = grouped_bootstrap_indices(groups, rng)
            observed = metrics(y[sample], other[sample])["balanced_accuracy"] - metrics(y[sample], base[sample])["balanced_accuracy"]
            placebo = expected_direction_matched_placebo(y[sample], base[sample], other[sample]) - metrics(y[sample], base[sample])["balanced_accuracy"]
            boot_observed.append(observed)
            boot_excess.append(observed - placebo)
        pairwise[target] = {
            "base": best,
            "directional_changes": {
                "no_to_yes": int(np.sum((base == 0) & (other == 1))),
                "yes_to_no": int(np.sum((base == 1) & (other == 0))),
            },
            "corrections": int(np.sum((base != y) & (other == y))),
            "harms": int(np.sum((base == y) & (other != y))),
            "observed_bacc_delta": observed_delta,
            "observed_bacc_delta_cluster_ci95": ci(boot_observed),
            "same_directional_flip_placebo_bacc_delta": placebo_delta,
            "selection_specificity_excess": observed_delta - placebo_delta,
            "selection_specificity_excess_cluster_ci95": ci(boot_excess),
        }

    correct_matrix = np.column_stack([pred[method] == y for method in methods])
    oracle = np.any(correct_matrix, axis=1)
    base_correct = pred[best] == y
    oracle_gain = np.mean(oracle) - np.mean(base_correct)
    oracle_boot = []
    for _ in range(BOOTSTRAP):
        sample = grouped_bootstrap_indices(groups, rng)
        oracle_boot.append(float(np.mean(oracle[sample]) - np.mean(base_correct[sample])))
    return {
        "status": "complete",
        "dataset": "MedHEval CXR-VisHal",
        "model": "LLaVA-Med-v1.5-Mistral-7B",
        "methods": methods,
        "scope": "jointly legacy-semantic-parseable binary intersection; invalid/choice rows excluded; diagnostic only",
        "n": int(len(y)),
        "unique_image_clusters": int(len(np.unique(groups))),
        "joint_retention_rate_all_questions": float(len(y) / len(qids[0])),
        "base_metrics": base_metrics,
        "best_method": best,
        "pairwise_vs_best": pairwise,
        "oracle_any_method_correct": {
            "accuracy": float(np.mean(oracle)),
            "accuracy_gain_vs_best": float(oracle_gain),
            "image_cluster_ci95": ci(oracle_boot),
            "warning": "label-dependent unattainable upper bound, not a method",
        },
        "anti_shortcut_control": "label-blind placebo preserves target no->yes and yes->no counts exactly in expectation; therefore any excess tests which examples changed, not Yes-rate",
        "length_control": "binary jointly parseable decisions; answer length cannot change the parsed endpoint",
        "inputs": {method: {"path": str(path), "sha256": sha256(path)} for method, path in eval_paths.items()},
    }


def final_margin(row: dict[str, Any]) -> float:
    layer = max(int(value) for value in row["margins"])
    return float(row["margins"][str(layer)])


def threshold_grid(probability: np.ndarray, y: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        value = float(balanced_accuracy_score(y, probability >= threshold))
        candidate = (value, -abs(float(threshold) - 0.5))
        incumbent = (best[0], -abs(best[1] - 0.5))
        if candidate > incumbent:
            best = (value, float(threshold))
    return best[1]


def fit_margin_calibrator(rows: list[dict[str, Any]], names: list[str], interactions: bool):
    findings = sorted({row["finding"] for row in rows})
    finding_to_index = {name: i for i, name in enumerate(findings)}
    y = np.asarray([int(row["positive_votes"] == 3) for row in rows])
    margins = np.asarray([[row[f"margin_{name}"] for name in names] for row in rows])
    one_hot = np.zeros((len(rows), len(findings)), dtype=float)
    one_hot[np.arange(len(rows)), [finding_to_index[row["finding"]] for row in rows]] = 1.0
    columns = [margins, one_hot[:, 1:]]
    if interactions:
        columns.extend([margins[:, [j]] * one_hot[:, 1:] for j in range(len(names))])
    x = np.column_stack(columns)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, random_state=SEED))
    model.fit(x, y)
    probability = model.predict_proba(x)[:, 1]
    return model, threshold_grid(probability, y), findings


def transform_margin_rows(rows: list[dict[str, Any]], names: list[str], findings: list[str], interactions: bool) -> np.ndarray:
    finding_to_index = {name: i for i, name in enumerate(findings)}
    margins = np.asarray([[row[f"margin_{name}"] for name in names] for row in rows])
    one_hot = np.zeros((len(rows), len(findings)), dtype=float)
    for i, row in enumerate(rows):
        if row["finding"] in finding_to_index:
            one_hot[i, finding_to_index[row["finding"]]] = 1.0
    columns = [margins, one_hot[:, 1:]]
    if interactions:
        columns.extend([margins[:, [j]] * one_hot[:, 1:] for j in range(len(names))])
    return np.column_stack(columns)


def load_paired_recoverability(huatuo: Path, hulu: Path) -> list[dict[str, Any]]:
    left = json.loads(huatuo.read_text())["records"]
    right = {row["record_key"]: row for row in json.loads(hulu.read_text())["records"]}
    rows = []
    for row in left:
        if row["positive_votes"] not in (0, 3):
            continue
        peer = right.get(row["record_key"])
        if peer is None or peer["positive_votes"] != row["positive_votes"]:
            raise ValueError("Huatuo/Hulu recoverability records do not align")
        rows.append({
            "record_key": row["record_key"],
            "image_id": row["image_id"],
            "finding": row["finding"],
            "positive_votes": row["positive_votes"],
            "margin_huatuo": final_margin(row),
            "margin_hulu": final_margin(peer),
        })
    return rows


def analyze_vindr_complementarity(bbox_csv: Path, dicom_root: Path) -> dict[str, Any]:
    dev_paths = {
        "huatuo": VINDR / "evidence_recoverability_dev_selfcheck_huatuo_v2/analysis.json",
        "hulu": VINDR / "evidence_recoverability_dev_selfcheck_hulu_v2/analysis.json",
    }
    test_paths = {
        "huatuo": VINDR / "evidence_recoverability_confirmation_huatuo_v2/analysis.json",
        "hulu": VINDR / "evidence_recoverability_confirmation_hulu_v2/analysis.json",
    }
    dev = load_paired_recoverability(dev_paths["huatuo"], dev_paths["hulu"])
    test = load_paired_recoverability(test_paths["huatuo"], test_paths["hulu"])
    y_dev = np.asarray([int(row["positive_votes"] == 3) for row in dev])
    y = np.asarray([int(row["positive_votes"] == 3) for row in test])
    groups = np.asarray([row["image_id"] for row in test])

    raw = {
        name: np.asarray([int(row[f"margin_{name}"] > 0) for row in test])
        for name in ("huatuo", "hulu")
    }
    dev_raw = {
        name: np.asarray([int(row[f"margin_{name}"] > 0) for row in dev])
        for name in ("huatuo", "hulu")
    }
    best = max(raw, key=lambda name: metrics(y_dev, dev_raw[name])["balanced_accuracy"])

    pooled_model, pooled_threshold, pooled_findings = fit_margin_calibrator(dev, ["huatuo", "hulu"], False)
    type_model, type_threshold, type_findings = fit_margin_calibrator(dev, ["huatuo", "hulu"], True)
    pooled_probability = pooled_model.predict_proba(transform_margin_rows(test, ["huatuo", "hulu"], pooled_findings, False))[:, 1]
    type_probability = type_model.predict_proba(transform_margin_rows(test, ["huatuo", "hulu"], type_findings, True))[:, 1]
    predictions = {
        "huatuo_raw": raw["huatuo"],
        "hulu_raw": raw["hulu"],
        "best_single_dev_selected": raw[best],
        "pooled_two_margin_calibration": (pooled_probability >= pooled_threshold).astype(int),
        "finding_interaction_two_margin": (type_probability >= type_threshold).astype(int),
    }
    all_metrics = {name: metrics(y, value) for name, value in predictions.items()}
    base = predictions["best_single_dev_selected"]
    rng = np.random.default_rng(SEED + 1)
    gains = {}
    for name in ("pooled_two_margin_calibration", "finding_interaction_two_margin"):
        boot = []
        value = predictions[name]
        for _ in range(BOOTSTRAP):
            sample = grouped_bootstrap_indices(groups, rng)
            boot.append(metrics(y[sample], value[sample])["balanced_accuracy"] - metrics(y[sample], base[sample])["balanced_accuracy"])
        gains[name] = {
            "bacc_delta_vs_best_single": all_metrics[name]["balanced_accuracy"] - all_metrics["best_single_dev_selected"]["balanced_accuracy"],
            "image_bootstrap_ci95": ci(boot),
        }

    by_finding = {}
    for finding in sorted({row["finding"] for row in test}):
        mask = np.asarray([row["finding"] == finding for row in test])
        h_correct = raw["huatuo"][mask] == y[mask]
        u_correct = raw["hulu"][mask] == y[mask]
        by_finding[finding] = {
            "n": int(mask.sum()),
            "huatuo_accuracy": float(np.mean(h_correct)),
            "hulu_accuracy": float(np.mean(u_correct)),
            "either_correct_oracle_accuracy": float(np.mean(h_correct | u_correct)),
            "huatuo_only_correct": int(np.sum(h_correct & ~u_correct)),
            "hulu_only_correct": int(np.sum(u_correct & ~h_correct)),
        }

    # Size audit is intentionally restricted to reader-positive boxed claims.
    positive = [row for row in test if row["positive_votes"] == 3]
    image_ids = {row["image_id"] for row in positive}
    dimensions = load_dimensions(dicom_root, image_ids)
    boxes = load_box_summary(bbox_csv, dimensions, {row["finding"] for row in positive})
    boxed = [row for row in positive if (row["image_id"], row["finding"]) in boxes]
    log_area = np.asarray([boxes[(row["image_id"], row["finding"])]["log_union_area_fraction"] for row in boxed])
    quantiles = np.quantile(log_area, [0.25, 0.5, 0.75]) if len(log_area) else np.asarray([])
    size_bins = np.digitize(log_area, quantiles)
    by_size = {}
    for bin_id, label in enumerate(("smallest_q", "small_mid_q", "large_mid_q", "largest_q")):
        mask = size_bins == bin_id
        if not np.any(mask):
            continue
        h = np.asarray([row["margin_huatuo"] > 0 for row in boxed])[mask]
        u = np.asarray([row["margin_hulu"] > 0 for row in boxed])[mask]
        by_size[label] = {
            "n": int(mask.sum()),
            "area_fraction_range": [float(10 ** np.min(log_area[mask])), float(10 ** np.max(log_area[mask]))],
            "huatuo_recall": float(np.mean(h)),
            "hulu_recall": float(np.mean(u)),
            "either_model_oracle_recall": float(np.mean(h | u)),
            "both_miss_rate": float(np.mean(~h & ~u)),
        }

    return {
        "status": "complete",
        "dataset": "VinDr-CXR reader-unanimous 0/3 vs 3/3",
        "models": ["HuatuoGPT-Vision-7B", "Hulu-Med-4B"],
        "method": "development-frozen two-margin calibration with and without finding interactions",
        "seed": SEED,
        "n_development": len(dev),
        "n_confirmation": len(test),
        "unique_confirmation_images": int(len(np.unique(groups))),
        "best_single_selected_on_development": best,
        "thresholds_selected_on_development": {"pooled": pooled_threshold, "finding_interaction": type_threshold},
        "confirmation_metrics": all_metrics,
        "confirmation_gains": gains,
        "by_finding_error_complementarity": by_finding,
        "positive_boxed_size_oracle": {
            "n": len(boxed),
            "bins": by_size,
            "warning": "uses released reader boxes and true-positive membership; diagnostic upper bound only, not an inference-time method or hallucination metric",
        },
        "anti_shortcut_controls": {
            "yes_rate": {name: value["yes_rate"] for name, value in all_metrics.items()},
            "fp_fn": {name: {"fp": value["fp"], "fn": value["fn"]} for name, value in all_metrics.items()},
            "test_label_isolation": "all calibration parameters and thresholds frozen on 640 development claims; 1920 confirmation claims used once",
        },
        "inputs": {
            **{f"dev_{name}": {"path": str(path), "sha256": sha256(path)} for name, path in dev_paths.items()},
            **{f"test_{name}": {"path": str(path), "sha256": sha256(path)} for name, path in test_paths.items()},
            "bbox_csv": {"path": str(bbox_csv), "sha256": sha256(bbox_csv)},
            "dicom_root": str(dicom_root),
        },
    }


CONCEPT_ALIASES = {
    "atelectasis": [r"atelecta"],
    "cardiomegaly": [r"cardiomegal", r"enlarged heart", r"heart (?:is )?enlarged"],
    "consolidation": [r"consolidat"],
    "edema": [r"pulmonary edema", r"interstitial edema"],
    "effusion": [r"pleural effusion", r"pleural fluid"],
    "emphysema": [r"emphysema", r"hyperinflation"],
    "fibrosis": [r"fibrosis", r"fibrotic"],
    "fracture": [r"fracture"],
    "infiltrate": [r"infiltrat"],
    "mass": [r"\bmass(?:es)?\b", r"tumou?r"],
    "nodule": [r"\bnodule(?:s)?\b", r"nodular"],
    "opacity": [r"\bopacit"],
    "pneumonia": [r"pneumonia"],
    "pneumothorax": [r"pneumothorax"],
}
NEGATION = re.compile(r"\b(no|not|without|absent|absence of|negative for|free of|neither)\b", re.I)


def clinical_claims(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|[\n;]+", text.lower())
    claims = []
    for sentence in sentences:
        for concept, aliases in CONCEPT_ALIASES.items():
            for alias in aliases:
                match = re.search(alias, sentence)
                if not match:
                    continue
                prefix = sentence[max(0, match.start() - 45):match.start()]
                polarity = "negative" if NEGATION.search(prefix) else "positive"
                claim = f"{concept}:{polarity}"
                if claim not in claims:
                    claims.append(claim)
                break
    return claims


def claim_scores(reference: list[str], prediction: list[str]) -> tuple[float, float, float]:
    ref, pred = set(reference), set(prediction)
    if not pred:
        precision = 1.0 if not ref else 0.0
    else:
        precision = len(ref & pred) / len(pred)
    recall = len(ref & pred) / len(ref) if ref else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def mean_claim_metrics(references: list[list[str]], predictions: list[list[str]]) -> dict[str, float]:
    values = np.asarray([claim_scores(ref, pred) for ref, pred in zip(references, predictions)])
    return {
        "precision": float(values[:, 0].mean()),
        "recall": float(values[:, 1].mean()),
        "f1": float(values[:, 2].mean()),
        "mean_claim_count": float(np.mean([len(value) for value in predictions])),
        "empty_claim_rate": float(np.mean([len(value) == 0 for value in predictions])),
    }


def analyze_fixed_k_oe(manifest: Path) -> dict[str, Any]:
    methods = ["AvisC", "DoLa", "OPERA", "PAI", "VCD", "VISTA"]
    paths = {
        method: MATRIX / "derived_scores/llava" / method / "visual_mimic_oe/answers.jsonl"
        for method in methods
    }
    rows = {method: read_jsonl(path) for method, path in paths.items()}
    order = [[str(row["question_id"]) for row in values] for values in rows.values()]
    if any(value != order[0] for value in order[1:]):
        raise ValueError("Visual-MIMIC output orders differ")
    manifest_rows = json.loads(manifest.read_text())
    manifest_by_id = {str(row.get("qid", row.get("id"))): row for row in manifest_rows}
    references = [clinical_claims(rows[methods[0]][i]["gt_ans"]) for i in range(len(order[0]))]
    predictions = {
        method: [clinical_claims(row.get("text", "")) for row in values]
        for method, values in rows.items()
    }
    groups = np.asarray([
        manifest_by_id[qid]["img_name"].split("/")[1]
        if "/" in manifest_by_id[qid]["img_name"] else manifest_by_id[qid]["img_name"]
        for qid in order[0]
    ])
    native = {method: mean_claim_metrics(references, values) for method, values in predictions.items()}
    rng = np.random.default_rng(SEED + 2)
    fixed_k = {}
    for k in (1, 2, 3):
        eligible = np.asarray([
            len(references[i]) >= k and all(len(predictions[method][i]) >= k for method in methods)
            for i in range(len(references))
        ])
        index = np.flatnonzero(eligible)
        section: dict[str, Any] = {
            "k": k,
            "jointly_eligible_n": int(len(index)),
            "joint_coverage_rate": float(len(index) / len(references)),
            "eligibility_rule": "reference and every method contain at least K extracted claims",
            "methods": {},
        }
        if not len(index):
            fixed_k[str(k)] = section
            continue
        section_references = [references[i] for i in index]
        for method in methods:
            section_predictions = [predictions[method][i][:k] for i in index]
            score = mean_claim_metrics(section_references, section_predictions)
            boot_precision, boot_recall, boot_f1 = [], [], []
            section_groups = groups[index]
            for _ in range(BOOTSTRAP):
                sample = grouped_bootstrap_indices(section_groups, rng)
                sampled = mean_claim_metrics(
                    [section_references[i] for i in sample],
                    [section_predictions[i] for i in sample],
                )
                boot_precision.append(sampled["precision"])
                boot_recall.append(sampled["recall"])
                boot_f1.append(sampled["f1"])
            section["methods"][method] = {
                **score,
                "patient_bootstrap_ci95": {
                    "precision": ci(boot_precision),
                    "recall": ci(boot_recall),
                    "f1": ci(boot_f1),
                },
            }
        fixed_k[str(k)] = section

    # The all-method intersection above is intentionally strict and can collapse
    # when one method is silent.  Pairwise fixed-K deltas retain the identical-K
    # control without allowing a low-coverage method to erase every comparison.
    pairwise_fixed_k: dict[str, Any] = {}
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1:]:
            pair = f"{left}__minus__{right}"
            pairwise_fixed_k[pair] = {}
            for k in (1, 2, 3):
                eligible = np.asarray([
                    len(references[i]) >= k
                    and len(predictions[left][i]) >= k
                    and len(predictions[right][i]) >= k
                    for i in range(len(references))
                ])
                index = np.flatnonzero(eligible)
                section: dict[str, Any] = {
                    "k": k,
                    "jointly_eligible_n": int(len(index)),
                    "joint_coverage_rate": float(len(index) / len(references)),
                }
                if not len(index):
                    pairwise_fixed_k[pair][str(k)] = section
                    continue
                left_scores = np.asarray([
                    claim_scores(references[i], predictions[left][i][:k]) for i in index
                ])
                right_scores = np.asarray([
                    claim_scores(references[i], predictions[right][i][:k]) for i in index
                ])
                delta = left_scores - right_scores
                pair_groups = groups[index]
                boot = [[], [], []]
                for _ in range(BOOTSTRAP):
                    sample = grouped_bootstrap_indices(pair_groups, rng)
                    value = delta[sample].mean(axis=0)
                    for metric_index in range(3):
                        boot[metric_index].append(float(value[metric_index]))
                section.update({
                    "left": mean_claim_metrics(
                        [references[i] for i in index],
                        [predictions[left][i][:k] for i in index],
                    ),
                    "right": mean_claim_metrics(
                        [references[i] for i in index],
                        [predictions[right][i][:k] for i in index],
                    ),
                    "left_minus_right": {
                        "precision": float(delta[:, 0].mean()),
                        "recall": float(delta[:, 1].mean()),
                        "f1": float(delta[:, 2].mean()),
                    },
                    "patient_bootstrap_delta_ci95": {
                        "precision": ci(boot[0]),
                        "recall": ci(boot[1]),
                        "f1": ci(boot[2]),
                    },
                    "interpretation_gate": "descriptive_only" if len(index) < 30 else "adequate_pairwise_fixed_k_sample",
                })
                pairwise_fixed_k[pair][str(k)] = section

    per_method_coverage = {
        method: {str(k): float(np.mean([len(value) >= k for value in predictions[method]])) for k in (1, 2, 3)}
        for method in methods
    }
    return {
        "status": "complete",
        "dataset": "MedHEval Visual-MIMIC OE",
        "model": "LLaVA-Med-v1.5-Mistral-7B",
        "methods": methods,
        "n": len(references),
        "unique_patients": int(len(np.unique(groups))),
        "metric_scope": "deterministic 14-concept signed lexical claim proxy; not clinician-verified hallucination truth",
        "native_claim_metrics": native,
        "fixed_k": fixed_k,
        "pairwise_fixed_k": pairwise_fixed_k,
        "per_method_claim_coverage": per_method_coverage,
        "anti_shortcut_controls": {
            "fixed_k": "on the jointly eligible subset every method contributes exactly K ordered unique signed claims",
            "silence": "joint coverage and each method's >=K coverage are reported; empty/short outputs cannot appear as precision gains",
            "length": "fixed-K precision/recall/F1 compare identical extracted claim counts, independent of prose length",
        },
        "inputs": {
            **{method: {"path": str(path), "sha256": sha256(path)} for method, path in paths.items()},
            "manifest": {"path": str(manifest), "sha256": sha256(manifest)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bbox-csv", type=Path, default=Path("/workspace/vinbigdata/train.csv"))
    parser.add_argument("--dicom-root", type=Path, default=Path("/workspace/vinbigdata/train"))
    parser.add_argument("--visual-mimic-manifest", type=Path, default=ROOT / "corrected_runs/unified_eval/inputs/baseline_matrix_v1/visual_mimic_oe.json")
    args = parser.parse_args()
    outputs = {
        "ce_operating_point": args.output_dir / "competition_ce_operating_point_specificity_v3.json",
        "vindr_complementarity": args.output_dir / "competition_vindr_model_complementarity_v3.json",
        "fixed_k_oe": args.output_dir / "competition_visual_mimic_fixed_k_v3.json",
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(path)
    common = {
        "version": VERSION,
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP,
        "command": " ".join(sys.argv),
        "source": str(Path(__file__).resolve()),
        "source_sha256": sha256(Path(__file__)),
        "execution": "CPU-only; no generation files modified",
    }
    results = {
        "ce_operating_point": analyze_ce_operating_point(),
        "vindr_complementarity": analyze_vindr_complementarity(args.bbox_csv, args.dicom_root),
        "fixed_k_oe": analyze_fixed_k_oe(args.visual_mimic_manifest),
    }
    for name, result in results.items():
        atomic_json(outputs[name], {"provenance": common, **result})
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
