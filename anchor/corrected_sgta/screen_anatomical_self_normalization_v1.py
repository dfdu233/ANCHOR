#!/usr/bin/env python3
"""CPU fatal screen for contralateral self-normalized visual evidence.

For approximately bilateral anatomy, a patient's mirrored tissue can serve as
a case-specific control.  The proposed statistic cancels a shared symmetric
component before claim scoring.  This screen asks two necessary questions:

1. do consensus lesion boxes score higher than their mirrored counterparts;
2. does the strongest left-right residual classify unilateral findings better
   than the strongest raw patch response?

No generation or GPU inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


VERSION = "anatomical-self-normalization-screen-v1"
FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
CSV_NAMES = {
    "Aortic enlargement": "aortic_enlargement",
    "Cardiomegaly": "cardiomegaly",
    "Lung Opacity": "lung_opacity",
    "Nodule/Mass": "nodule_mass",
    "Pleural effusion": "pleural_effusion",
    "Pleural thickening": "pleural_thickening",
    "Pulmonary fibrosis": "pulmonary_fibrosis",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_annotations(path: Path):
    votes: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    boxes: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            finding = CSV_NAMES.get(row["class_name"])
            if finding is None:
                continue
            image_id = row["image_id"]
            votes[image_id][finding].add(row["rad_id"])
            if row["x_min"]:
                boxes[(image_id, finding)].append(
                    tuple(float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
                )
    return votes, boxes


def load_metadata(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dimensions(raw_metadata: Path) -> dict[str, tuple[int, int]]:
    result = {}
    for row in load_metadata(raw_metadata):
        meta = row["simple_image_metadata"]
        result[row["image_id"]] = (int(meta["image_width"]), int(meta["image_height"]))
    return result


def standardize(
    scores: np.ndarray, labels: np.ndarray, metadata: list[dict[str, Any]]
) -> np.ndarray:
    output = np.empty_like(scores, dtype=np.float64)
    for finding in range(len(FINDINGS)):
        index = np.asarray(
            [
                row["split"] == "development" and labels[i, finding] == 0
                for i, row in enumerate(metadata)
            ]
        )
        values = scores[index, :, :, finding]
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        positive = std[std > 0]
        floor = float(np.quantile(positive, 0.1))
        output[:, :, :, finding] = (scores[:, :, :, finding] - mean) / np.maximum(std, floor)
    return output


def statistics(zscores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = zscores.max(axis=(1, 2))
    left = zscores[:, :, : zscores.shape[2] // 2, :]
    right = zscores[:, :, zscores.shape[2] // 2 :, :][:, :, ::-1, :]
    residual = np.abs(left - right) / np.sqrt(2.0)
    asymmetry = residual.max(axis=(1, 2))
    return raw, asymmetry


def macro_auc(labels: np.ndarray, scores: np.ndarray, indices: np.ndarray) -> tuple[float, dict]:
    values = []
    by_finding = {}
    for finding, name in enumerate(FINDINGS):
        valid = indices & np.isin(labels[:, finding], (0, 3))
        y = (labels[valid, finding] == 3).astype(int)
        if len(np.unique(y)) < 2:
            continue
        auc = float(roc_auc_score(y, scores[valid, finding]))
        values.append(auc)
        by_finding[name] = {
            "n": int(valid.sum()),
            "positive": int(y.sum()),
            "negative": int((1 - y).sum()),
            "auroc": auc,
        }
    return float(np.mean(values)), by_finding


def bootstrap_auc_delta(
    labels: np.ndarray,
    raw: np.ndarray,
    asymmetry: np.ndarray,
    metadata: list[dict[str, Any]],
    draws: int,
    seed: int,
) -> dict:
    confirmation = np.asarray([row["split"] == "confirmation" for row in metadata])
    image_indices = np.flatnonzero(confirmation)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        sampled = rng.choice(image_indices, len(image_indices), replace=True)
        mask = np.zeros(len(metadata), dtype=bool)
        # Duplicate draws must retain multiplicity, so compute per-finding directly.
        raw_auc = []
        asym_auc = []
        for finding in range(len(FINDINGS)):
            valid = np.isin(labels[sampled, finding], (0, 3))
            current = sampled[valid]
            y = (labels[current, finding] == 3).astype(int)
            if len(np.unique(y)) < 2:
                continue
            raw_auc.append(roc_auc_score(y, raw[current, finding]))
            asym_auc.append(roc_auc_score(y, asymmetry[current, finding]))
        if raw_auc:
            deltas.append(float(np.mean(asym_auc) - np.mean(raw_auc)))
    values = np.asarray(deltas)
    return {
        "draws": int(values.size),
        "unit": "image",
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
    }


def union_box_mask(
    boxes: list[tuple[float, float, float, float]], width: int, height: int, side: int
) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        gx0 = max(0, min(side - 1, int(np.floor(x0 / width * side))))
        gx1 = max(0, min(side, int(np.ceil(x1 / width * side))))
        gy0 = max(0, min(side - 1, int(np.floor(y0 / height * side))))
        gy1 = max(0, min(side, int(np.ceil(y1 / height * side))))
        mask[gy0:gy1, gx0:gx1] = True
    return mask


def bbox_analysis(
    zscores: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    boxes: dict,
    dims: dict[str, tuple[int, int]],
    draws: int,
    seed: int,
) -> dict:
    rows = []
    side = zscores.shape[1]
    for index, row in enumerate(metadata):
        if row["split"] != "confirmation":
            continue
        image_id = row["image_id"]
        if image_id not in dims:
            continue
        width, height = dims[image_id]
        for finding, name in enumerate(FINDINGS):
            current_boxes = boxes.get((image_id, name), [])
            if labels[index, finding] != 3 or not current_boxes:
                continue
            mask = union_box_mask(current_boxes, width, height, side)
            mirrored = mask[:, ::-1]
            overlap = (mask & mirrored).sum() / max((mask | mirrored).sum(), 1)
            # Exclude midline/bilateral boxes: they do not admit a contralateral null.
            if overlap > 0.10 or not mask.any() or not mirrored.any():
                continue
            score = zscores[index, :, :, finding]
            rows.append(
                {
                    "image_id": image_id,
                    "finding": name,
                    "delta": float(score[mask].mean() - score[mirrored].mean()),
                }
            )
    values = np.asarray([row["delta"] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    rates = np.empty(draws)
    for draw in range(draws):
        sample = rng.choice(values, len(values), replace=True)
        means[draw] = sample.mean()
        rates[draw] = (sample > 0).mean()
    by_finding = {}
    for name in FINDINGS:
        current = np.asarray([row["delta"] for row in rows if row["finding"] == name])
        if current.size:
            by_finding[name] = {
                "n": int(current.size),
                "mean_delta": float(current.mean()),
                "positive_rate": float((current > 0).mean()),
            }
    return {
        "n": len(rows),
        "mean_delta": float(values.mean()) if len(values) else None,
        "positive_rate": float((values > 0).mean()) if len(values) else None,
        "mean_delta_ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])]
        if len(values)
        else None,
        "positive_rate_ci95": [float(x) for x in np.quantile(rates, [0.025, 0.975])]
        if len(values)
        else None,
        "by_finding": by_finding,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--raw-metadata", type=Path, required=True)
    parser.add_argument("--vindr-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    metadata_path = args.patch_dir / "metadata.jsonl"
    scores_path = args.patch_dir / "patch_scores.npz"
    metadata = load_metadata(metadata_path)
    scores = np.load(scores_path)["patch_scores"].astype(np.float64)
    scores = scores.reshape(len(metadata), 24, 24, len(FINDINGS))
    votes, boxes = load_annotations(args.vindr_csv)
    labels = np.asarray(
        [[len(votes[row["image_id"]][finding]) for finding in FINDINGS] for row in metadata],
        dtype=int,
    )
    zscores = standardize(scores, labels, metadata)
    raw, asymmetry = statistics(zscores)
    results = {}
    for split in ("development", "confirmation"):
        mask = np.asarray([row["split"] == split for row in metadata])
        raw_macro, raw_by = macro_auc(labels, raw, mask)
        asym_macro, asym_by = macro_auc(labels, asymmetry, mask)
        results[split] = {
            "images": int(mask.sum()),
            "raw_max_macro_auroc": raw_macro,
            "self_normalized_macro_auroc": asym_macro,
            "macro_auroc_delta": asym_macro - raw_macro,
            "raw_by_finding": raw_by,
            "self_normalized_by_finding": asym_by,
        }
    boot = bootstrap_auc_delta(
        labels, raw, asymmetry, metadata, args.bootstrap_draws, args.seed
    )
    bbox = bbox_analysis(
        zscores,
        labels,
        metadata,
        boxes,
        dimensions(args.raw_metadata),
        args.bootstrap_draws,
        args.seed + 1,
    )
    qualified_bbox_findings = sum(
        value["n"] >= 20 and value["positive_rate"] >= 0.60
        for value in bbox["by_finding"].values()
    )
    passed = bool(
        results["confirmation"]["macro_auroc_delta"] >= 0.02
        and boot["ci95"][0] > 0
        and bbox["n"] >= 60
        and bbox["mean_delta_ci95"][0] > 0
        and bbox["positive_rate"] >= 0.60
        and qualified_bbox_findings >= 3
    )
    payload = {
        "version": VERSION,
        "hypothesis": (
            "for unilateral abnormalities, mirrored patient anatomy is a case-specific null "
            "that cancels symmetric nuisance and isolates clinical evidence"
        ),
        "method": (
            "max absolute left-right mirrored patch-score residual divided by sqrt(2), "
            "compared with raw max patch response"
        ),
        "inputs": {
            "patch_scores": str(scores_path.resolve()),
            "patch_scores_sha256": sha256(scores_path),
            "patch_metadata_sha256": sha256(metadata_path),
            "raw_metadata_sha256": sha256(args.raw_metadata),
            "vindr_csv_sha256": sha256(args.vindr_csv),
        },
        "results": results,
        "confirmation_bootstrap": boot,
        "bbox_directional_check": bbox,
        "gate": {
            "rule": (
                "confirmation macro AUROC gain>=.02 with image bootstrap lower>0; at least "
                "60 unilateral consensus-box cases; box-minus-mirror mean CI lower>0 and "
                "positive rate>=.60; at least three findings each n>=20/rate>=.60"
            ),
            "qualified_bbox_findings": qualified_bbox_findings,
            "pass": passed,
        },
        "decision": "GO_TO_COLLISION_AND_DECODER" if passed else "NO_GO",
        "claim_boundary": (
            "A pass would support only unilateral, approximately symmetric anatomy; bilateral "
            "disease, rotation and patient laterality remain outside the claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["gate"] | {"decision": payload["decision"], "confirmation": results["confirmation"], "bbox": bbox}, indent=2))


if __name__ == "__main__":
    main()
