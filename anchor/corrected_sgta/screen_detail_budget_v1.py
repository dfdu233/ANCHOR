#!/usr/bin/env python3
"""CPU-only fatal gate for label-blind detail-adaptive pixel allocation.

The gate asks whether a simple image-derived sampling density allocates more
of a fixed pixel/token budget to reader boxes than to matched sham boxes.  It
does not run a VLM and does not use finding labels to construct the density.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pydicom


FOCAL = {
    "Atelectasis", "Calcification", "Clavicle fracture", "Consolidation",
    "Infiltration", "Lung Opacity", "Lung cavity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Rib fracture",
}


def load_image(path: Path, side: int = 512) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    x = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(x, (0.5, 99.5))
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        x = 1 - x
    return cv2.resize(x, (side, side), interpolation=cv2.INTER_AREA)


def density(x: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 4.0)
    # A floor preserves every part of the image; only the remaining budget is
    # redistributed by label-blind local detail.
    rho = 0.25 * float(g.mean()) + g
    return rho / max(float(rho.sum()), 1e-12)


def box_mass(rho: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    return float(rho[y0:y1, x0:x1].sum())


def bootstrap(values: np.ndarray, seed: int, draws: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = rng.choice(values, len(values), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("/workspace/vinbigdata/train.csv"))
    p.add_argument("--image-root", type=Path, default=Path("/workspace/vinbigdata/train"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-images", type=int, default=800)
    p.add_argument("--seed", type=int, default=20260813)
    a = p.parse_args()

    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    with a.csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["class_name"] in FOCAL and row["x_min"]:
                by_image[row["image_id"]].append(row)
    ids = sorted(by_image)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(ids)
    ids = ids[: a.max_images]

    records = []
    for image_id in ids:
        path = a.image_root / f"{image_id}.dicom"
        if not path.is_file():
            continue
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        width, height = int(ds.Columns), int(ds.Rows)
        x = load_image(path)
        rho = density(x)
        side = rho.shape[0]
        for row in by_image[image_id]:
            x0 = max(0, min(side - 1, int(float(row["x_min"]) / width * side)))
            x1 = max(x0 + 1, min(side, int(np.ceil(float(row["x_max"]) / width * side))))
            y0 = max(0, min(side - 1, int(float(row["y_min"]) / height * side)))
            y1 = max(y0 + 1, min(side, int(np.ceil(float(row["y_max"]) / height * side))))
            box = (x0, y0, x1, y1)
            area = (x1 - x0) * (y1 - y0) / (side * side)
            true_enrichment = box_mass(rho, box) / max(area, 1e-12)
            sham = []
            bw, bh = x1 - x0, y1 - y0
            for _ in range(16):
                sx = int(rng.integers(0, max(side - bw + 1, 1)))
                sy = int(rng.integers(0, max(side - bh + 1, 1)))
                sham.append(box_mass(rho, (sx, sy, sx + bw, sy + bh)) / max(area, 1e-12))
            records.append({
                "image_id": image_id,
                "finding": row["class_name"],
                "area_fraction": area,
                "true_enrichment": true_enrichment,
                "sham_enrichment": float(np.mean(sham)),
                "true_minus_sham": true_enrichment - float(np.mean(sham)),
            })

    values = np.asarray([r["true_minus_sham"] for r in records])
    per_finding = {}
    for finding in sorted({r["finding"] for r in records}):
        v = np.asarray([r["true_minus_sham"] for r in records if r["finding"] == finding])
        per_finding[finding] = {
            "n": len(v), "mean_delta": float(v.mean()),
            "ci95": bootstrap(v, a.seed + sum(map(ord, finding))) if len(v) >= 10 else None,
            "positive_rate": float((v > 0).mean()),
        }
    qualified = [v for v in per_finding.values() if v["n"] >= 30 and v["ci95"][0] > 0]
    result = {
        "version": "detail-budget-screen-v1",
        "n_images": len({r["image_id"] for r in records}),
        "n_boxes": len(records),
        "overall_mean_delta": float(values.mean()),
        "overall_ci95": bootstrap(values, a.seed),
        "overall_positive_rate": float((values > 0).mean()),
        "per_finding": per_finding,
        "decision": "GO" if len(qualified) >= 5 else "NO_GO",
        "gate": "at least 5 findings with n>=30 and image-bootstrap-style box mean CI lower>0",
        "boundary": "GO only admits a VLM probe; it does not establish hallucination mitigation or novelty",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
