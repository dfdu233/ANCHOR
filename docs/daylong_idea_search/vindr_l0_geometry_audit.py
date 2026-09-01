#!/usr/bin/env python3
"""CPU-only VinDr geometry/contrast audit against frozen VLM margins.

This script is deliberately descriptive.  It joins only the already-frozen
Evidence Addressability confirmation cohort to the public VinDr annotations;
it does not change labels or run a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy.stats import rankdata, spearmanr


ROOT = Path("/home/dbw/ANCHOR")
VINDR = Path("/workspace/vinbigdata")
OUT = ROOT / "docs/daylong_idea_search/vindr_l0_geometry_results.json"

META = {
    "huatuo": ROOT
    / "corrected_runs/evidence_addressability_gate_v2/hidden_fresh_huatuo_v2/metadata.jsonl",
    "hulu": ROOT
    / "corrected_runs/evidence_addressability_gate_v2/hidden_fresh_hulu_v3/metadata.jsonl",
}

NAME = {
    "aortic_enlargement": "Aortic enlargement",
    "cardiomegaly": "Cardiomegaly",
    "lung_opacity": "Lung Opacity",
    "nodule_mass": "Nodule/Mass",
    "pleural_effusion": "Pleural effusion",
    "pleural_thickening": "Pleural thickening",
    "pulmonary_fibrosis": "Pulmonary fibrosis",
}


def load_margin(path: Path, model: str) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            layer = str(max(map(int, d["diagnostic_plain_logit_lens"])))
            z = d["diagnostic_plain_logit_lens"][layer]
            rows.append(
                {
                    "model": model,
                    "finding": d["finding"],
                    "image_id": d["image_id"],
                    "votes": d["positive_votes"],
                    "margin": z["supported"] - z["refuted"],
                }
            )
    return pd.DataFrame(rows)


def raster_mask(boxes: pd.DataFrame, h: int, w: int, side: int = 128) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    for r in boxes.itertuples():
        x0 = int(np.floor(r.x_min / w * side))
        x1 = int(np.ceil(r.x_max / w * side))
        y0 = int(np.floor(r.y_min / h * side))
        y1 = int(np.ceil(r.y_max / h * side))
        mask[max(0, y0) : min(side, y1), max(0, x0) : min(side, x1)] = True
    return mask


def image_features(rows: pd.DataFrame, image_id: str) -> dict[str, float]:
    path = VINDR / "train" / f"{image_id}.dicom"
    ds = pydicom.dcmread(path)
    h, w = int(ds.Rows), int(ds.Columns)
    area = float(h * w)
    per_reader = []
    masks = []
    centroids = []
    for _, boxes in rows.groupby("rad_id"):
        a = (boxes.x_max - boxes.x_min) * (boxes.y_max - boxes.y_min)
        a_sum = float(a.sum())
        cx = ((boxes.x_min + boxes.x_max) / 2).to_numpy()
        cy = ((boxes.y_min + boxes.y_max) / 2).to_numpy()
        weights = a.to_numpy() / max(a_sum, 1e-9)
        centroids.append((float((cx * weights).sum() / w), float((cy * weights).sum() / h)))
        side = np.where(cx < w / 2, -1.0, 1.0)
        side_balance = 1.0 - abs(float((side * weights).sum()))
        radial = np.sqrt(((cx / w - 0.5) * 2) ** 2 + ((cy / h - 0.5) * 2) ** 2)
        widths = (boxes.x_max - boxes.x_min).to_numpy()
        heights = (boxes.y_max - boxes.y_min).to_numpy()
        per_reader.append(
            {
                "area_fraction": a_sum / area,
                "largest_fraction": float(a.max()) / area,
                "box_count": float(len(boxes)),
                "fragmentation": 1.0 - float(a.max()) / max(a_sum, 1e-9),
                "aspect_log_abs": float(
                    (np.abs(np.log(np.maximum(widths, 1) / np.maximum(heights, 1))) * weights).sum()
                ),
                "radial_position": float((radial * weights).sum()),
                "bilateral_balance": side_balance,
            }
        )
        masks.append(raster_mask(boxes, h, w))

    out = {k: float(np.mean([r[k] for r in per_reader])) for k in per_reader[0]}
    ious = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            union = np.logical_or(masks[i], masks[j]).sum()
            ious.append(float(np.logical_and(masks[i], masks[j]).sum() / max(union, 1)))
    out["reader_mask_iou"] = float(np.mean(ious)) if ious else np.nan
    c = np.asarray(centroids)
    out["reader_centroid_dispersion"] = float(
        np.sqrt(((c - c.mean(axis=0, keepdims=True)) ** 2).sum(axis=1)).mean()
    )

    # A label-free visibility proxy: absolute mean intensity difference between
    # the reader box union and an equal-width local ring, standardized by the
    # image robust scale.  It is not a clinical contrast ground truth.
    px = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    px = px * slope + intercept
    sample = px[:: max(h // 256, 1), :: max(w // 256, 1)]
    q25, q75 = np.percentile(sample, [25, 75])
    robust_scale = max(float((q75 - q25) / 1.349), 1e-6)
    contrasts = []
    texture_ratios = []
    for _, boxes in rows.groupby("rad_id"):
        inner = np.zeros((h, w), dtype=bool)
        outer = np.zeros((h, w), dtype=bool)
        for r in boxes.itertuples():
            x0, x1 = max(0, int(r.x_min)), min(w, int(np.ceil(r.x_max)))
            y0, y1 = max(0, int(r.y_min)), min(h, int(np.ceil(r.y_max)))
            inner[y0:y1, x0:x1] = True
            pad_x = max(8, int(0.25 * (x1 - x0)))
            pad_y = max(8, int(0.25 * (y1 - y0)))
            outer[max(0, y0 - pad_y) : min(h, y1 + pad_y), max(0, x0 - pad_x) : min(w, x1 + pad_x)] = True
        ring = outer & ~inner
        if inner.sum() and ring.sum():
            contrasts.append(abs(float(px[inner].mean() - px[ring].mean())) / robust_scale)
            texture_ratios.append(float(px[inner].std()) / max(float(px[ring].std()), 1e-6))
    out["local_abs_contrast"] = float(np.mean(contrasts)) if contrasts else np.nan
    out["local_texture_ratio"] = float(np.mean(texture_ratios)) if texture_ratios else np.nan
    out["height"] = h
    out["width"] = w
    out["reader_count"] = int(rows.rad_id.nunique())
    out["annotation_rows"] = int(len(rows))
    return out


def within_finding_rank(df: pd.DataFrame, col: str) -> np.ndarray:
    out = np.empty(len(df), dtype=float)
    for _, idx in df.groupby("finding").groups.items():
        pos = df.index.get_indexer(idx)
        out[pos] = rankdata(df.loc[idx, col], method="average") / len(idx)
    return out


def bootstrap_rho(df: pd.DataFrame, x: str, y: str, draws: int = 5000) -> dict:
    rng = np.random.default_rng(42)
    vals = []
    groups = [g.reset_index(drop=True) for _, g in df.groupby("finding")]
    for _ in range(draws):
        b = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups], ignore_index=True)
        vals.append(float(spearmanr(within_finding_rank(b, x), within_finding_rank(b, y)).statistic))
    observed = float(spearmanr(within_finding_rank(df, x), within_finding_rank(df, y)).statistic)
    return {"rho": observed, "ci95": np.quantile(vals, [0.025, 0.975]).tolist()}


def main() -> None:
    annotations = pd.read_csv(VINDR / "train.csv")
    margins = pd.concat([load_margin(path, model) for model, path in META.items()], ignore_index=True)
    positives = margins[(margins.model == "huatuo") & (margins.votes == 3)].copy()
    feature_rows = []
    missing = []
    for r in positives.itertuples():
        boxes = annotations[
            (annotations.image_id == r.image_id) & (annotations.class_name == NAME[r.finding])
        ].dropna(subset=["x_min", "y_min", "x_max", "y_max"])
        if boxes.empty:
            missing.append(f"{r.finding}:{r.image_id}")
            continue
        feature_rows.append({"finding": r.finding, "image_id": r.image_id, **image_features(boxes, r.image_id)})
    features = pd.DataFrame(feature_rows)
    joined = margins[margins.votes == 3].merge(features, on=["finding", "image_id"], how="inner")

    feature_cols = [
        "area_fraction",
        "largest_fraction",
        "box_count",
        "fragmentation",
        "aspect_log_abs",
        "radial_position",
        "bilateral_balance",
        "reader_mask_iou",
        "reader_centroid_dispersion",
        "local_abs_contrast",
        "local_texture_ratio",
    ]
    summary = {
        "cohort": {
            "expected_clear_positive_images": int(len(positives)),
            "bbox_joined_images": int(len(features)),
            "missing": missing,
            "findings": features.finding.value_counts().sort_index().to_dict(),
        },
        "feature_summary": {},
        "model_associations": {},
    }
    for c in feature_cols:
        s = features[c].dropna()
        summary["feature_summary"][c] = {
            "n": int(len(s)),
            "p10": float(s.quantile(0.1)),
            "median": float(s.median()),
            "p90": float(s.quantile(0.9)),
        }
    for model, q in joined.groupby("model"):
        q = q.reset_index(drop=True)
        model_out = {
            "n": int(len(q)),
            "false_negative_n": int((q.margin <= 0).sum()),
            "false_negative_rate": float((q.margin <= 0).mean()),
            "associations_with_margin": {},
            "fn_vs_tp_medians": {},
        }
        for c in feature_cols:
            clean = q.dropna(subset=[c, "margin"]).reset_index(drop=True)
            model_out["associations_with_margin"][c] = bootstrap_rho(clean, c, "margin")
            model_out["fn_vs_tp_medians"][c] = {
                "fn": float(clean.loc[clean.margin <= 0, c].median()),
                "tp": float(clean.loc[clean.margin > 0, c].median()),
            }
        summary["model_associations"][model] = model_out

    # Per-finding substrate summaries make clear which future tests are powered.
    summary["by_finding"] = {}
    for finding, q in features.groupby("finding"):
        summary["by_finding"][finding] = {
            "n": int(len(q)),
            **{f"median_{c}": float(q[c].median()) for c in feature_cols},
        }
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
