#!/usr/bin/env python3
"""Confirm whether lesion area predicts 3/3-positive VLM misses across splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from corrected_sgta.analyze_sparse_lesion_boundary_v1 import (
    load_box_summary,
    load_dimensions,
    load_rows,
)


VERSION = "lesion-area-miss-prediction-v1"
SEED = 20260812
BOOTSTRAP_DRAWS = 5000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def joined(hidden: Path, boxes: dict) -> list[dict]:
    output = []
    for row in load_rows(hidden):
        summary = boxes.get((row["image_id"], row["finding"]))
        if summary is not None:
            output.append({**row, **summary, "miss": int(row["margin"] <= 0)})
    return output


def design(rows: list[dict], findings: list[str], mean: float, std: float, area: bool) -> np.ndarray:
    fixed = np.column_stack(
        [np.asarray([row["finding"] == name for row in rows], dtype=float) for name in findings[:-1]]
    )
    if not area:
        return fixed
    values = np.asarray([row["log_union_area_fraction"] for row in rows])
    return np.column_stack([fixed, (values - mean) / std])


def audit_model(model: str, dev_hidden: Path, test_hidden: Path, boxes: dict) -> dict:
    dev_rows, test_rows = joined(dev_hidden, boxes), joined(test_hidden, boxes)
    findings = sorted({row["finding"] for row in dev_rows} & {row["finding"] for row in test_rows})
    dev_rows = [row for row in dev_rows if row["finding"] in findings]
    test_rows = [row for row in test_rows if row["finding"] in findings]
    area_dev = np.asarray([row["log_union_area_fraction"] for row in dev_rows])
    mean, std = float(area_dev.mean()), float(area_dev.std())
    y_dev = np.asarray([row["miss"] for row in dev_rows])
    y_test = np.asarray([row["miss"] for row in test_rows])
    base = LogisticRegression(C=0.1, max_iter=10000, random_state=SEED).fit(
        design(dev_rows, findings, mean, std, False), y_dev
    )
    enhanced = LogisticRegression(C=0.1, max_iter=10000, random_state=SEED).fit(
        design(dev_rows, findings, mean, std, True), y_dev
    )
    p0 = base.predict_proba(design(test_rows, findings, mean, std, False))[:, 1]
    p1 = enhanced.predict_proba(design(test_rows, findings, mean, std, True))[:, 1]
    auc0, auc1 = roc_auc_score(y_test, p0), roc_auc_score(y_test, p1)
    nll0, nll1 = log_loss(y_test, p0), log_loss(y_test, p1)
    cells = [np.flatnonzero(np.asarray([row["finding"] for row in test_rows]) == name) for name in findings]
    rng = np.random.default_rng(SEED)
    auc_delta, nll_delta = [], []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = np.concatenate([rng.choice(cell, len(cell), replace=True) for cell in cells])
        if len(set(y_test[indices].tolist())) < 2:
            continue
        auc_delta.append(roc_auc_score(y_test[indices], p1[indices]) - roc_auc_score(y_test[indices], p0[indices]))
        nll_delta.append(log_loss(y_test[indices], p0[indices]) - log_loss(y_test[indices], p1[indices]))
    ci_auc = np.quantile(auc_delta, [0.025, 0.975]).tolist()
    ci_nll = np.quantile(nll_delta, [0.025, 0.975]).tolist()
    pass_gate = bool(
        auc1 - auc0 >= 0.05
        and ci_auc[0] > 0
        and nll0 - nll1 > 0
        and ci_nll[0] > 0
        and enhanced.coef_[0, -1] < 0
    )
    return {
        "model": model,
        "development_n": len(dev_rows),
        "confirmation_n": len(test_rows),
        "development_miss_rate": float(y_dev.mean()),
        "confirmation_miss_rate": float(y_test.mean()),
        "base_finding_only_auroc": float(auc0),
        "enhanced_finding_plus_log_area_auroc": float(auc1),
        "auroc_delta": float(auc1 - auc0),
        "auroc_delta_ci95": ci_auc,
        "base_nll": float(nll0),
        "enhanced_nll": float(nll1),
        "nll_improvement": float(nll0 - nll1),
        "nll_improvement_ci95": ci_nll,
        "standardized_log_area_coefficient": float(enhanced.coef_[0, -1]),
        "gate": {
            "rule": "AUROC delta>=0.05, AUROC/NLL CI lower bounds>0, area coefficient negative",
            "pass": pass_gate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-development", type=Path, required=True)
    parser.add_argument("--huatuo-confirmation", type=Path, required=True)
    parser.add_argument("--hulu-development", type=Path, required=True)
    parser.add_argument("--hulu-confirmation", type=Path, required=True)
    parser.add_argument("--bbox-csv", type=Path, required=True)
    parser.add_argument("--dicom-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    all_rows = (
        load_rows(args.huatuo_development)
        + load_rows(args.huatuo_confirmation)
        + load_rows(args.hulu_development)
        + load_rows(args.hulu_confirmation)
    )
    image_ids = {row["image_id"] for row in all_rows}
    findings = {row["finding"] for row in all_rows}
    boxes = load_box_summary(
        args.bbox_csv, load_dimensions(args.dicom_root, image_ids), findings
    )
    models = {
        "huatuo": audit_model("huatuo", args.huatuo_development, args.huatuo_confirmation, boxes),
        "hulu": audit_model("hulu", args.hulu_development, args.hulu_confirmation, boxes),
    }
    result = {
        "version": VERSION,
        "status": "complete",
        "scope": "development-fitted prediction of margin<=0 among 3/3 reader-positive boxed claims",
        "models": models,
        "joint_gate": {
            "both_models_pass": all(value["gate"]["pass"] for value in models.values()),
        },
        "configuration": {
            "seed": SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "command": " ".join(sys.argv),
            "source_sha256": sha256(Path(__file__)),
            "inputs": {
                "huatuo_development_metadata_sha256": sha256(args.huatuo_development / "metadata.jsonl"),
                "huatuo_confirmation_metadata_sha256": sha256(args.huatuo_confirmation / "metadata.jsonl"),
                "hulu_development_metadata_sha256": sha256(args.hulu_development / "metadata.jsonl"),
                "hulu_confirmation_metadata_sha256": sha256(args.hulu_confirmation / "metadata.jsonl"),
                "bbox_csv_sha256": sha256(args.bbox_csv),
                "dicom_root": str(args.dicom_root.resolve()),
            },
        },
        "boundary": "A miss is defined from the diagnostic claim margin, not generated OE text. Area is a released-box extent proxy, not lesion conspicuity.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result["joint_gate"], indent=2))


if __name__ == "__main__":
    main()
