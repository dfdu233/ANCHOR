#!/usr/bin/env python3
"""Cache-only gate for a specialist-as-observer update.

The frozen specialist emits an 18-dimensional radiographic state ``s`` and
the medical VLM emits projected visual tokens with pooled state ``h``.  An
unlabelled image panel fits a linear observation map ``H: h -> s``.  At test
time the only message from the specialist is the innovation ``s - Hh``.

This script is deliberately an exploratory L0 screen on previously opened
VinDr labels.  It does not claim that changing the pooled state changes real
generation; it tests that premise before spending GPU time on a causal update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

from anchor.corrected_sgta.screen_external_visual_increment_v1 import FINDINGS, load_claims
from anchor.corrected_sgta.screen_xrv_visual_increment_v1 import FINDING_TARGETS, XRV_LABELS


VERSION = "xrv-observer-innovation-screen-v1"


def stable_bucket(value: str, seed: int, modulus: int = 12) -> int:
    raw = f"{seed}:{value}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % modulus


def load_raw(path: Path) -> tuple[dict[str, np.ndarray], int]:
    metadata = [json.loads(line) for line in (path / "metadata.jsonl").read_text().splitlines() if line.strip()]
    pack = np.load(path / "features.npz", allow_pickle=False)
    values = np.asarray(pack["post_mean"], dtype=np.float64)
    if len(metadata) != len(values):
        raise ValueError(f"raw feature/metadata length mismatch at {path}")
    image_ids = [str(row["image_id"]) for row in metadata]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"raw feature metadata repeats image IDs at {path}")
    return dict(zip(image_ids, values)), values.shape[1]


def load_xrv(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    pack = np.load(path, allow_pickle=False)
    labels = [str(value) for value in pack["labels"]]
    if labels != list(XRV_LABELS):
        raise ValueError("XRV label order drift")
    values = np.asarray(pack["logits"], dtype=np.float64)
    return dict(zip(map(str, pack["image_ids"]), values)), labels


def final_rows(path: Path, model: str) -> list[dict[str, Any]]:
    rows = load_claims(path, "opened_confirmation", model)
    return [dict(row) for row in rows]


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return tuple((array - mean) / scale for array in (train,) + others)


def target_xrv(row: dict[str, Any], score: np.ndarray, label_index: dict[str, int]) -> float:
    return float(max(score[label_index[name]] for name in FINDING_TARGETS[row["finding"]]))


def macro_auc(rows: list[dict[str, Any]], probability: np.ndarray) -> float:
    values = []
    for finding in FINDINGS:
        index = np.asarray([i for i, row in enumerate(rows) if row["finding"] == finding])
        if not len(index):
            continue
        y = np.asarray([rows[i]["label"] for i in index])
        if len(np.unique(y)) == 2:
            values.append(roc_auc_score(y, probability[index]))
    if not values:
        raise ValueError("macro AUROC has no valid finding")
    return float(np.mean(values))


def per_finding_predict(
    development: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    fields: tuple[str, ...],
    seed: int,
) -> np.ndarray:
    output = np.zeros(len(confirmation), dtype=np.float64)
    for finding in FINDINGS:
        dev_index = [i for i, row in enumerate(development) if row["finding"] == finding]
        test_index = [i for i, row in enumerate(confirmation) if row["finding"] == finding]
        if not dev_index or not test_index:
            continue
        x_dev = np.stack([np.concatenate([np.atleast_1d(development[i][field]) for field in fields]) for i in dev_index])
        x_test = np.stack([np.concatenate([np.atleast_1d(confirmation[i][field]) for field in fields]) for i in test_index])
        x_dev, x_test = standardize(x_dev, x_test)
        y_dev = np.asarray([development[i]["label"] for i in dev_index])
        if len(np.unique(y_dev)) != 2:
            raise ValueError(f"calibration split lacks both labels for {finding}")
        model = LogisticRegression(C=0.1, max_iter=5000, random_state=seed).fit(x_dev, y_dev)
        output[test_index] = model.predict_proba(x_test)[:, 1]
    return output


def tune_gamma(development: list[dict[str, Any]], candidates: tuple[float, ...]) -> float:
    best = (float("-inf"), 0.0)
    for gamma in candidates:
        score = np.asarray([row["margin"] + gamma * row["observer_delta_margin"] for row in development])
        value = macro_auc(development, score)
        if value > best[0] + 1e-12:
            best = (value, gamma)
    return float(best[1])


def tune_thresholds(rows: list[dict[str, Any]], probability: np.ndarray) -> dict[str, float]:
    output = {}
    for finding in FINDINGS:
        index = np.asarray([i for i, row in enumerate(rows) if row["finding"] == finding])
        y = np.asarray([rows[i]["label"] for i in index])
        p = probability[index]
        candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
        values = [balanced_accuracy_score(y, p >= threshold) for threshold in candidates]
        output[finding] = float(candidates[int(np.argmax(values))])
    return output


def metrics(
    rows: list[dict[str, Any]], probability: np.ndarray, thresholds: dict[str, float]
) -> dict[str, float | int]:
    y = np.asarray([row["label"] for row in rows])
    pred = np.asarray([probability[i] >= thresholds[row["finding"]] for i, row in enumerate(rows)])
    return {
        "n": len(rows),
        "images": len({row["image_id"] for row in rows}),
        "macro_auroc": macro_auc(rows, probability),
        "nll": float(log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier": float(np.mean((y - probability) ** 2)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "fp": int(np.sum((pred == 1) & (y == 0))),
        "fn": int(np.sum((pred == 0) & (y == 1))),
        "tp": int(np.sum((pred == 1) & (y == 1))),
        "tn": int(np.sum((pred == 0) & (y == 0))),
    }


def cluster_bootstrap(
    rows: list[dict[str, Any]], predictions: dict[str, np.ndarray], draws: int, seed: int
) -> dict[str, dict[str, list[float] | float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["image_id"]].append(i)
    image_ids = np.asarray(sorted(groups))
    rng = np.random.default_rng(seed)
    pairs = (
        ("raw18", "target_scalar"),
        ("innovation18", "target_scalar"),
        ("innovation18", "raw18"),
        ("observer_proxy", "vlm"),
        ("observer_placebo", "vlm"),
    )
    values = {f"{left}_minus_{right}": [] for left, right in pairs}
    for _ in range(draws):
        sampled = rng.choice(image_ids, len(image_ids), replace=True)
        index = np.asarray([i for image_id in sampled for i in groups[str(image_id)]])
        sampled_rows = [rows[i] for i in index]
        for left, right in pairs:
            key = f"{left}_minus_{right}"
            try:
                values[key].append(
                    macro_auc(sampled_rows, predictions[left][index])
                    - macro_auc(sampled_rows, predictions[right][index])
                )
            except ValueError:
                continue
    return {
        key: {
            "mean": float(np.mean(current)),
            "ci95": [float(np.quantile(current, 0.025)), float(np.quantile(current, 0.975))],
        }
        for key, current in values.items()
    }


def analyze_model(
    model_name: str,
    hidden_path: Path,
    raw_path: Path,
    xrv: dict[str, np.ndarray],
    label_index: dict[str, int],
    seed: int,
    draws: int,
) -> dict[str, Any]:
    raw, dimension = load_raw(raw_path)
    rows = [row for row in final_rows(hidden_path, model_name) if row["image_id"] in raw and row["image_id"] in xrv]
    image_ids = sorted({row["image_id"] for row in rows})
    split = {}
    for image_id in image_ids:
        bucket = stable_bucket(image_id, seed)
        split[image_id] = "observer" if bucket < 6 else ("calibration" if bucket < 9 else "confirmation")

    observer_ids = [image_id for image_id in image_ids if split[image_id] == "observer"]
    calibration_ids = [image_id for image_id in image_ids if split[image_id] == "calibration"]
    confirmation_ids = [image_id for image_id in image_ids if split[image_id] == "confirmation"]
    h_observer = np.stack([raw[image_id] for image_id in observer_ids])
    h_calibration = np.stack([raw[image_id] for image_id in calibration_ids])
    h_confirmation = np.stack([raw[image_id] for image_id in confirmation_ids])
    s_observer = np.stack([xrv[image_id] for image_id in observer_ids])
    s_calibration = np.stack([xrv[image_id] for image_id in calibration_ids])
    s_confirmation = np.stack([xrv[image_id] for image_id in confirmation_ids])
    h_observer, h_calibration, h_confirmation = standardize(h_observer, h_calibration, h_confirmation)
    s_observer, s_calibration, s_confirmation = standardize(s_observer, s_calibration, s_confirmation)

    # H is fitted without clinical labels. Ridge is used only to stabilize the
    # high-dimensional pooled representation; the causal canary will use the
    # corresponding minimum-norm innovation direction.
    observation = Ridge(alpha=10.0).fit(h_observer, s_observer)
    h_by_id = dict(zip(observer_ids + calibration_ids + confirmation_ids, np.concatenate([h_observer, h_calibration, h_confirmation])))
    s_by_id = dict(zip(observer_ids + calibration_ids + confirmation_ids, np.concatenate([s_observer, s_calibration, s_confirmation])))
    predicted_by_id = {image_id: observation.predict(h_by_id[image_id][None])[0] for image_id in image_ids}
    residual_by_id = {image_id: s_by_id[image_id] - predicted_by_id[image_id] for image_id in image_ids}

    coefficient = np.asarray(observation.coef_, dtype=np.float64)  # [18, d]
    gram = coefficient @ coefficient.T
    inverse = np.linalg.pinv(gram, rcond=1e-6)
    delta_by_id = {
        image_id: coefficient.T @ inverse @ residual_by_id[image_id]
        for image_id in image_ids
    }
    rng = np.random.default_rng(seed + 101)
    shuffled = confirmation_ids.copy()
    rng.shuffle(shuffled)
    placebo_delta = {
        image_id: coefficient.T @ inverse @ (s_by_id[donor] - predicted_by_id[image_id])
        for image_id, donor in zip(confirmation_ids, shuffled)
    }

    observer_rows = [row for row in rows if split[row["image_id"]] == "observer"]
    calibration = [dict(row) for row in rows if split[row["image_id"]] == "calibration"]
    confirmation = [dict(row) for row in rows if split[row["image_id"]] == "confirmation"]
    if min(len(observer_rows), len(calibration), len(confirmation)) == 0:
        raise ValueError("deterministic split produced an empty claim partition")

    # VLM-margin readout is fitted without labels and serves only as a cheap
    # approximation to what the real decoder may do after the hidden update.
    readouts: dict[str, Ridge] = {}
    for finding in FINDINGS:
        current = [row for row in observer_rows if row["finding"] == finding]
        if len(current) < 8:
            raise ValueError(f"too few observer claims for margin readout: {finding}")
        readouts[finding] = Ridge(alpha=100.0).fit(
            np.stack([h_by_id[row["image_id"]] for row in current]),
            np.asarray([row["margin"] for row in current]),
        )

    for collection in (calibration, confirmation):
        for row in collection:
            image_id = row["image_id"]
            row["target_scalar"] = np.asarray([target_xrv(row, xrv[image_id], label_index)])
            row["raw18"] = s_by_id[image_id]
            row["predicted18"] = predicted_by_id[image_id]
            row["innovation18"] = residual_by_id[image_id]
            weight = np.asarray(readouts[row["finding"]].coef_)
            row["observer_delta_margin"] = float(weight @ delta_by_id[image_id])
            row["observer_placebo_delta_margin"] = float(
                weight @ placebo_delta.get(image_id, delta_by_id[image_id])
            )

    probability_calibration = {
        "vlm": per_finding_predict(calibration, calibration, ("margin",), seed),
        "target_scalar": per_finding_predict(calibration, calibration, ("margin", "target_scalar"), seed),
        "raw18": per_finding_predict(calibration, calibration, ("margin", "raw18"), seed),
        "innovation18": per_finding_predict(calibration, calibration, ("margin", "innovation18"), seed),
    }
    probability_confirmation = {
        "vlm": per_finding_predict(calibration, confirmation, ("margin",), seed),
        "target_scalar": per_finding_predict(calibration, confirmation, ("margin", "target_scalar"), seed),
        "raw18": per_finding_predict(calibration, confirmation, ("margin", "raw18"), seed),
        "innovation18": per_finding_predict(calibration, confirmation, ("margin", "innovation18"), seed),
    }
    gamma = tune_gamma(calibration, (0.0, 0.1, 0.25, 0.5, 1.0, 2.0))
    placebo_gamma = tune_gamma(
        [dict(row, observer_delta_margin=row["observer_placebo_delta_margin"]) for row in calibration],
        (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
    )
    observer_score_cal = np.asarray([row["margin"] + gamma * row["observer_delta_margin"] for row in calibration])
    observer_score_test = np.asarray([row["margin"] + gamma * row["observer_delta_margin"] for row in confirmation])
    placebo_score_cal = np.asarray([row["margin"] + placebo_gamma * row["observer_placebo_delta_margin"] for row in calibration])
    placebo_score_test = np.asarray([row["margin"] + placebo_gamma * row["observer_placebo_delta_margin"] for row in confirmation])
    # Convert the raw proxy scores to calibrated probabilities without changing
    # their ranking; this keeps NLL/Brier and operating-point audits meaningful.
    for name, cal_score, test_score in (
        ("observer_proxy", observer_score_cal, observer_score_test),
        ("observer_placebo", placebo_score_cal, placebo_score_test),
    ):
        for finding in FINDINGS:
            dev_i = [i for i, row in enumerate(calibration) if row["finding"] == finding]
            test_i = [i for i, row in enumerate(confirmation) if row["finding"] == finding]
            model = LogisticRegression(C=0.1, max_iter=3000, random_state=seed).fit(
                cal_score[dev_i, None], [calibration[i]["label"] for i in dev_i]
            )
            probability_calibration.setdefault(name, np.zeros(len(calibration)))[dev_i] = model.predict_proba(cal_score[dev_i, None])[:, 1]
            probability_confirmation.setdefault(name, np.zeros(len(confirmation)))[test_i] = model.predict_proba(test_score[test_i, None])[:, 1]

    points = {}
    for name, test_probability in probability_confirmation.items():
        thresholds = tune_thresholds(calibration, probability_calibration[name])
        points[name] = metrics(confirmation, test_probability, thresholds)

    return {
        "raw_dimension": dimension,
        "split_images": {
            "observer_unlabelled_fit": len(observer_ids),
            "labelled_calibration": len(calibration_ids),
            "opened_confirmation": len(confirmation_ids),
        },
        "split_claims": {
            "observer_margin_readout": len(observer_rows),
            "labelled_calibration": len(calibration),
            "opened_confirmation": len(confirmation),
        },
        "observation_fit": {
            "ridge_alpha": 10.0,
            "mean_test_residual_l2": float(np.mean(np.linalg.norm(s_confirmation - observation.predict(h_confirmation), axis=1))),
            "mean_test_s_l2": float(np.mean(np.linalg.norm(s_confirmation, axis=1))),
            "effective_rank": int(np.linalg.matrix_rank(coefficient)),
        },
        "observer_proxy_gamma_selected_on_calibration": gamma,
        "placebo_proxy_gamma_selected_on_calibration": placebo_gamma,
        "points": points,
        "bootstrap_macro_auroc_deltas": cluster_bootstrap(
            confirmation, probability_confirmation, draws, seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-hidden", type=Path, required=True)
    parser.add_argument("--hulu-hidden", type=Path, required=True)
    parser.add_argument("--huatuo-raw", type=Path, required=True)
    parser.add_argument("--hulu-raw", type=Path, required=True)
    parser.add_argument("--xrv-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    xrv, labels = load_xrv(args.xrv_logits)
    label_index = {label: i for i, label in enumerate(labels)}
    analyses = {
        "huatuo": analyze_model(
            "huatuo", args.huatuo_hidden, args.huatuo_raw, xrv, label_index, args.seed, args.draws
        ),
        "hulu": analyze_model(
            "hulu", args.hulu_hidden, args.hulu_raw, xrv, label_index, args.seed, args.draws
        ),
    }
    gates = []
    for analysis in analyses.values():
        delta = analysis["bootstrap_macro_auroc_deltas"]["observer_proxy_minus_vlm"]
        placebo = analysis["bootstrap_macro_auroc_deltas"]["observer_placebo_minus_vlm"]
        point = analysis["points"]["observer_proxy"]["macro_auroc"] - analysis["points"]["vlm"]["macro_auroc"]
        gates.append(point >= 0.01 and delta["ci95"][0] > 0 and point > placebo["mean"])
    result = {
        "version": VERSION,
        "status": "complete_cache_only_opened_label_screen",
        "decision": "PASS_TO_CAUSAL_CANARY" if all(gates) else "NO_GO_OBSERVER_UPDATE",
        "decision_rule": (
            "Both VLMs: unsupervised observer hidden-update proxy adds >=.01 macro AUROC over "
            "the VLM, image-cluster CI lower bound >0, and exceeds shuffled-specialist placebo."
        ),
        "claim_boundary": (
            "The raw/residual classifiers are information screens. Only the observer_proxy tests "
            "the rank prediction of a minimum-norm hidden update, and even a pass requires a real "
            "decoder intervention before any mitigation claim."
        ),
        "analyses": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
