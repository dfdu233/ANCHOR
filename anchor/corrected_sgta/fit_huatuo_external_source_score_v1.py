#!/usr/bin/env python3
"""Fit an external MIMIC-vs-IU source probe and score the frozen 128 cohort.

All target patients are removed from probe training before any fit.  The target
error labels are never read by this script.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "huatuo-external-source-score-v1"


def patient_from_path(path: str) -> str | None:
    for part in Path(path).parts:
        if re.fullmatch(r"p\d{8}", part):
            return part
    return None


def pipeline(seed: int):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=8, whiten=True, random_state=seed),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-raw", type=Path, nargs="+", required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = []
    for source_path in args.source_raw:
        source.extend(
            json.loads(line)
            for line in source_path.read_text().splitlines()
            if line.strip()
        )
    target = json.loads(args.target_manifest.read_text())["records"]
    target_patients = {str(row["patient_id"]) for row in target}
    eligible = []
    excluded = []
    seen_images: set[str] = set()
    for row in source:
        if row.get("domain") not in {"mimic", "iuxray"} or row.get("status") != "ok":
            continue
        patient = patient_from_path(row["image"])
        if patient is not None and patient in target_patients:
            excluded.append({"record_key": row["record_key"], "patient_id": patient})
            continue
        image_key = str(Path(row["image"]).resolve())
        if image_key in seen_images:
            continue
        seen_images.add(image_key)
        eligible.append(row)
    x = np.stack([np.load(row["feature_file"])["visual_pre"] for row in eligible])
    y = np.asarray([int(row["domain"] == "mimic") for row in eligible])
    groups = np.asarray(
        [
            patient_from_path(row["image"])
            or f"image:{Path(row['image']).resolve()}"
            for row in eligible
        ]
    )
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20260806)
    oof = cross_val_predict(
        pipeline(20260806), x, y, cv=cv, groups=groups, method="predict_proba"
    )[:, 1]
    model = pipeline(20260806).fit(x, y)
    target_x = np.stack([np.load(row["feature_file"])["visual_pre"] for row in target])
    target_p = model.predict_proba(target_x)[:, 1]
    target_logit = model.decision_function(target_x)
    payload = {
        "version": VERSION,
        "target_outcome_labels_read": False,
        "training": {
            "n": len(eligible),
            "n_groups": int(len(set(groups))),
            "mimic": int(y.sum()),
            "iuxray": int((1 - y).sum()),
            "excluded_target_patient_rows": excluded,
            "feature": "global mean Huatuo visual pre-projector vector",
            "model": "StandardScaler + whitened PCA(8) + class-balanced logistic regression",
            "cross_validation": "four-fold stratified group CV; MIMIC grouped by patient, IU-Xray by image",
            "four_fold_oof_auroc": float(roc_auc_score(y, oof)),
            "four_fold_oof_auprc_mimic": float(average_precision_score(y, oof)),
            "four_fold_oof_accuracy_at_0.5": float(accuracy_score(y, oof >= 0.5)),
        },
        "score_semantics": "positive means more MIMIC-like relative to the external IU-Xray contrast",
        "target_scores": [
            {
                "question_id": row["question_id"],
                "patient_id": row["patient_id"],
                "mimic_probability": float(target_p[i]),
                "source_logit": float(target_logit[i]),
            }
            for i, row in enumerate(target)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "training": payload["training"], "target_probability_summary": {"min": float(target_p.min()), "median": float(np.median(target_p)), "max": float(target_p.max())}}, indent=2))


if __name__ == "__main__":
    main()
