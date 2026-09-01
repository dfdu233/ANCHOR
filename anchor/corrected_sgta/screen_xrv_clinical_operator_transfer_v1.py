#!/usr/bin/env python3
"""Test whether a specialist full-state transform transfers across medical VLMs.

The frozen input is the same 18-dimensional TorchXRayVision (XRV) diagnostic
state for both VLMs.  A source VLM is allowed to learn a finding-conditioned
direction through that state on the frozen development split.  On a target
VLM, the 18-D direction is frozen: target development data may only calibrate
the VLM margin and one scalar specialist score (globally or per finding).

This is a CPU-only novelty gate.  Strong transfer would support a reusable
"clinical differential operator"; failure means that the earlier full-state
gain is ordinary model-specific stacking.  Confirmation images are never used
for fitting, normalisation, or model selection.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PROTOCOL = "xrv-clinical-operator-transfer-v1"
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
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum",
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


def final_margin(row: dict[str, Any]) -> float:
    lens = row["diagnostic_plain_logit_lens"]
    layer = max(lens, key=lambda value: int(value))
    return float(lens[layer]["supported"] - lens[layer]["refuted"])


def load_claims(path: Path, model: str, split: str) -> list[dict[str, Any]]:
    rows = []
    for raw in read_jsonl(path):
        if raw["finding"] not in FINDINGS or int(raw["positive_votes"]) not in (0, 3):
            continue
        rows.append({
            "image_id": str(raw["image_id"]),
            "finding": str(raw["finding"]),
            "label": int(raw["positive_votes"] == 3),
            "margin": final_margin(raw),
            "model": model,
            "split": split,
        })
    rows.sort(key=lambda row: (row["image_id"], row["finding"]))
    return rows


def load_xrv(path: Path) -> dict[str, np.ndarray]:
    pack = np.load(path, allow_pickle=False)
    labels = tuple(str(value) for value in pack["labels"])
    if labels != XRV_LABELS:
        raise ValueError(f"XRV label order drift: {labels}")
    return {
        str(image_id): np.asarray(value, dtype=np.float64)
        for image_id, value in zip(pack["image_ids"], pack["logits"])
    }


def assert_paired(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> None:
    a = [(row["image_id"], row["finding"], row["label"]) for row in left]
    b = [(row["image_id"], row["finding"], row["label"]) for row in right]
    if a != b:
        raise ValueError("The two VLM caches do not contain identical frozen claims")


def attach_xrv(rows: list[dict[str, Any]], xrv: dict[str, np.ndarray]) -> None:
    label_to_index = {label: index for index, label in enumerate(XRV_LABELS)}
    for row in rows:
        vector = xrv[row["image_id"]]
        row["xrv"] = vector
        row["scalar"] = max(
            float(vector[label_to_index[label]]) for label in FINDING_TARGETS[row["finding"]]
        )


def fit_normalisation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # XRV normalisation is model-independent and fitted once on unique dev images.
    unique = {row["image_id"]: row["xrv"] for row in rows}
    xrv = np.stack([unique[key] for key in sorted(unique)])
    x_mean, x_scale = xrv.mean(axis=0), xrv.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    finding = {}
    for name in FINDINGS:
        selected = [row for row in rows if row["finding"] == name]
        finding[name] = {}
        for key in ("margin", "scalar"):
            values = np.asarray([row[key] for row in selected], dtype=np.float64)
            scale = float(values.std())
            finding[name][key] = (float(values.mean()), scale if scale >= 1e-8 else 1.0)
    return {"xrv": (x_mean, x_scale), "finding": finding}


def z_value(row: dict[str, Any], normalisation: dict[str, Any], key: str) -> np.ndarray:
    if key == "xrv":
        mean, scale = normalisation["xrv"]
        return (row["xrv"] - mean) / scale
    mean, scale = normalisation["finding"][row["finding"]][key]
    return np.atleast_1d((float(row[key]) - mean) / scale)


def finding_features(row: dict[str, Any], drop_last: bool = True) -> list[float]:
    names = FINDINGS[:-1] if drop_last else FINDINGS
    return [float(row["finding"] == name) for name in names]


def base_design(rows, norm) -> np.ndarray:
    return np.asarray([
        finding_features(row) + z_value(row, norm, "margin").tolist() for row in rows
    ])


def scalar_design(rows, norm) -> np.ndarray:
    return np.asarray([
        finding_features(row)
        + z_value(row, norm, "margin").tolist()
        + z_value(row, norm, "scalar").tolist()
        for row in rows
    ])


def full_design(rows, norm) -> np.ndarray:
    matrix = []
    for row in rows:
        state = z_value(row, norm, "xrv").tolist()
        interactions = [flag * value for flag in finding_features(row, False) for value in state]
        matrix.append(
            finding_features(row) + z_value(row, norm, "margin").tolist() + state + interactions
        )
    return np.asarray(matrix)


def specialist_directions(model: LogisticRegression) -> np.ndarray:
    # Layout: 6 finding intercepts, one margin, 18 shared XRV terms,
    # followed by 7 x 18 finding-specific interaction terms.
    coef = model.coef_[0]
    shared_start = 6 + 1
    shared = coef[shared_start: shared_start + 18]
    interaction = coef[shared_start + 18:].reshape(len(FINDINGS), 18)
    return shared[None, :] + interaction


def transfer_scores(rows, xrv_norm, directions) -> np.ndarray:
    return np.asarray([
        float(directions[FINDINGS.index(row["finding"])] @ z_value(row, xrv_norm, "xrv"))
        for row in rows
    ])


def transfer_design(rows, target_norm, scores, per_finding: bool) -> np.ndarray:
    matrix = []
    for row, score in zip(rows, scores):
        values = finding_features(row) + z_value(row, target_norm, "margin").tolist() + [score]
        if per_finding:
            values += [flag * score for flag in finding_features(row, False)]
        matrix.append(values)
    return np.asarray(matrix)


def fit_lr(x: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    model = LogisticRegression(C=1.0, max_iter=5000, random_state=seed)
    model.fit(x, y)
    return model


def macro_auc(rows, scores) -> float:
    result = []
    for finding in FINDINGS:
        indices = [i for i, row in enumerate(rows) if row["finding"] == finding]
        y = np.asarray([rows[i]["label"] for i in indices])
        result.append(roc_auc_score(y, scores[indices]))
    return float(np.mean(result))


def classification(rows, scores) -> dict[str, int]:
    y = np.asarray([row["label"] for row in rows])
    pred = scores >= 0.5
    return {
        "tp": int(np.sum(pred & (y == 1))), "tn": int(np.sum(~pred & (y == 0))),
        "fp": int(np.sum(pred & (y == 0))), "fn": int(np.sum(~pred & (y == 1))),
    }


def metrics(rows, scores) -> dict[str, Any]:
    y = np.asarray([row["label"] for row in rows])
    return {
        "macro_auroc": macro_auc(rows, scores),
        "nll": float(log_loss(y, scores, labels=[0, 1])),
        "brier": float(brier_score_loss(y, scores)),
        "classification": classification(rows, scores),
        "by_finding_auroc": {
            finding: float(roc_auc_score(
                [row["label"] for row in rows if row["finding"] == finding],
                [score for row, score in zip(rows, scores) if row["finding"] == finding],
            ))
            for finding in FINDINGS
        },
    }


def bootstrap(rows, predictions, native_name, transfer_names, draws, seed) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["image_id"]].append(index)
    image_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    deltas = {name: [] for name in predictions if name != "base"}
    deltas_over_scalar = {
        name: [] for name in predictions if name not in ("base", "target_scalar")
    }
    retention = {name: [] for name in transfer_names}
    incremental_retention = {name: [] for name in transfer_names}
    for _ in range(draws):
        sampled = rng.choice(image_ids, len(image_ids), replace=True)
        indices = np.asarray([i for image_id in sampled for i in groups[image_id]])
        subset = [rows[i] for i in indices]
        try:
            auc = {
                name: macro_auc(subset, values[indices])
                for name, values in predictions.items()
            }
            base = auc["base"]
            scalar = auc["target_scalar"]
            native = auc[native_name]
        except ValueError:
            continue
        native_gain = native - base
        for name in predictions:
            if name != "base":
                deltas[name].append(auc[name] - base)
            if name not in ("base", "target_scalar"):
                deltas_over_scalar[name].append(auc[name] - scalar)
        if native_gain > 1e-6:
            for name in transfer_names:
                gain = auc[name] - base
                retention[name].append(gain / native_gain)
        native_increment = native - scalar
        if native_increment > 1e-6:
            for name in transfer_names:
                gain = auc[name] - scalar
                incremental_retention[name].append(gain / native_increment)
    output = {}
    for name, values in deltas.items():
        output[name] = {
            "delta_over_base_mean": float(np.mean(values)),
            "delta_over_base_ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
            "valid_draws": len(values),
        }
    for name, values in retention.items():
        output[name]["retained_native_gain_ratio_bootstrap"] = {
            "median": float(np.median(values)),
            "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
            "valid_draws_native_gain_positive": len(values),
            "warning": "Ratio CI is unstable when the resampled native gain is near zero.",
        }
    for name, values in deltas_over_scalar.items():
        output[name]["delta_over_target_scalar_mean"] = float(np.mean(values))
        output[name]["delta_over_target_scalar_ci95"] = [
            float(np.quantile(values, .025)), float(np.quantile(values, .975))
        ]
    for name, values in incremental_retention.items():
        output[name]["retained_native_increment_over_scalar_bootstrap"] = {
            "median": float(np.median(values)),
            "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
            "valid_draws_native_increment_positive": len(values),
            "warning": "Ratio CI is unstable when the resampled native increment is near zero.",
        }
    return output


def run_target(source, target, xrv_norm, draws, seed) -> dict[str, Any]:
    y_source = np.asarray([row["label"] for row in source["dev"]])
    y_target = np.asarray([row["label"] for row in target["dev"]])

    source_full = fit_lr(full_design(source["dev"], source["norm"]), y_source, seed)
    directions = specialist_directions(source_full)
    target_native = fit_lr(full_design(target["dev"], target["norm"]), y_target, seed)
    native_directions = specialist_directions(target_native)

    models = {
        "base": fit_lr(base_design(target["dev"], target["norm"]), y_target, seed),
        "target_scalar": fit_lr(scalar_design(target["dev"], target["norm"]), y_target, seed),
        "within_model_full18": target_native,
    }
    prediction = {
        "base": models["base"].predict_proba(base_design(target["test"], target["norm"]))[:, 1],
        "target_scalar": models["target_scalar"].predict_proba(scalar_design(target["test"], target["norm"]))[:, 1],
        "within_model_full18": target_native.predict_proba(full_design(target["test"], target["norm"]))[:, 1],
    }
    dev_score = transfer_scores(target["dev"], xrv_norm, directions)
    test_score = transfer_scores(target["test"], xrv_norm, directions)
    for per_finding, name in ((False, "transferred_global_calibration"), (True, "transferred_per_finding_calibration")):
        model = fit_lr(transfer_design(target["dev"], target["norm"], dev_score, per_finding), y_target, seed)
        prediction[name] = model.predict_proba(
            transfer_design(target["test"], target["norm"], test_score, per_finding)
        )[:, 1]

    point = {name: metrics(target["test"], values) for name, values in prediction.items()}
    base_auc = point["base"]["macro_auroc"]
    scalar_auc = point["target_scalar"]["macro_auroc"]
    native_gain = point["within_model_full18"]["macro_auroc"] - base_auc
    native_increment = point["within_model_full18"]["macro_auroc"] - scalar_auc
    transfer_retention = {}
    for name in ("transferred_global_calibration", "transferred_per_finding_calibration"):
        gain = point[name]["macro_auroc"] - base_auc
        transfer_retention[name] = {
            "gain_over_base": gain,
            "native_gain": native_gain,
            "retained_native_gain_ratio": gain / native_gain if abs(native_gain) > 1e-12 else None,
            "incremental_gain_over_target_scalar": point[name]["macro_auroc"] - scalar_auc,
            "native_incremental_gain_over_target_scalar": native_increment,
            "retained_native_increment_over_scalar_ratio": (
                (point[name]["macro_auroc"] - scalar_auc) / native_increment
                if abs(native_increment) > 1e-12 else None
            ),
        }
    cosine = {}
    for index, finding in enumerate(FINDINGS):
        a, b = directions[index], native_directions[index]
        cosine[finding] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    return {
        "source_model": source["name"], "target_model": target["name"],
        "development_n": len(target["dev"]), "confirmation_n": len(target["test"]),
        "target_fit_restriction": (
            "The transferred 18-D direction is frozen. Target dev fits finding intercepts, one "
            "VLM-margin coefficient, and either one global or seven finding-specific scalar weights."
        ),
        "point_metrics": point,
        "transfer_retention": transfer_retention,
        "source_native_direction_cosine": cosine,
        "mean_direction_cosine": float(np.mean(list(cosine.values()))),
        "image_cluster_bootstrap": bootstrap(
            target["test"], prediction, "within_model_full18",
            ("transferred_global_calibration", "transferred_per_finding_calibration"),
            draws, seed,
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
        raise RuntimeError("CPU-only gate; set CUDA_VISIBLE_DEVICES=''")
    if args.output.exists():
        raise FileExistsError(args.output)

    xrv = load_xrv(args.xrv_logits)
    models = {}
    for name, dev_path, test_path in (
        ("huatuo", args.huatuo_dev, args.huatuo_confirmation),
        ("hulu", args.hulu_dev, args.hulu_confirmation),
    ):
        dev = load_claims(dev_path, name, "development")
        test = load_claims(test_path, name, "confirmation")
        attach_xrv(dev + test, xrv)
        models[name] = {"name": name, "dev": dev, "test": test, "norm": fit_normalisation(dev)}
    assert_paired(models["huatuo"]["dev"], models["hulu"]["dev"])
    assert_paired(models["huatuo"]["test"], models["hulu"]["test"])

    # The XRV state and its normalisation are common across VLMs; the paired
    # development claims make either cache yield exactly the same statistics.
    xrv_norm = models["huatuo"]["norm"]
    result = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "bootstrap_draws": args.draws,
        "inputs": {
            "huatuo_development": str(args.huatuo_dev.resolve()),
            "huatuo_confirmation": str(args.huatuo_confirmation.resolve()),
            "hulu_development": str(args.hulu_dev.resolve()),
            "hulu_confirmation": str(args.hulu_confirmation.resolve()),
            "xrv_logits": str(args.xrv_logits.resolve()),
        },
        "scope": "CPU-only, seven unanimous VinDr findings, frozen dev/confirmation split",
        "comparisons": {
            "huatuo_to_hulu": run_target(models["huatuo"], models["hulu"], xrv_norm, args.draws, args.seed),
            "hulu_to_huatuo": run_target(models["hulu"], models["huatuo"], xrv_norm, args.draws, args.seed),
        },
    }
    passes = []
    for comparison in result["comparisons"].values():
        transfer = comparison["transfer_retention"]["transferred_per_finding_calibration"]
        ci = comparison["image_cluster_bootstrap"]["transferred_per_finding_calibration"]["delta_over_target_scalar_ci95"]
        passes.append(
            transfer["retained_native_increment_over_scalar_ratio"] >= 0.70 and ci[0] > 0
        )
    result["preregistered_gate"] = {
        "criterion": (
            "In both transfer directions, the frozen source 18-D direction must retain >=70% of "
            "the within-model full18 AUROC increment beyond target-scalar fusion, and its paired "
            "image-bootstrap increment over target-scalar fusion must exclude zero."
        ),
        "passed_each_direction": passes,
        "decision": "GO_VLM_AGNOSTIC_OPERATOR" if all(passes) else "NO_GO_MODEL_SPECIFIC_STACKING",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
