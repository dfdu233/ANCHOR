#!/usr/bin/env python3
"""CPU-only fatal gate for anatomy-conditioned visual nulls on VinDr-CXR.

This script does not score a VLM.  It asks whether a non-generative patch
replacement is sufficiently close to a conditional randomization draw to
justify spending GPU time.  It is deliberately fail-closed: matching on a
pixel anatomy descriptor is an approximation, never a proof of exchangeability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "anatomy-conditional-null-fastgate-v1"
PANEL = {"R8", "R9", "R10"}
FINDING_KEY = "nodule_mass"
FINDING_CSV = "Nodule/Mass"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable(seed: int, *parts: str) -> str:
    return hashlib.sha256((str(seed) + ":" + ":".join(parts)).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def render_dicom(path: Path, size: int) -> tuple[np.ndarray, dict]:
    import pydicom

    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array.astype(np.float32)
    pixels = pixels * float(getattr(ds, "RescaleSlope", 1.0)) + float(
        getattr(ds, "RescaleIntercept", 0.0)
    )
    finite = pixels[np.isfinite(pixels)]
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    pixels = np.clip((pixels - lo) / max(hi - lo, 1e-6), 0, 1)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = 1.0 - pixels
    image = Image.fromarray(np.round(pixels * 255).astype(np.uint8), mode="L")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    spacing = getattr(ds, "PixelSpacing", None)
    metadata = {
        "rows": int(getattr(ds, "Rows", pixels.shape[0])),
        "columns": int(getattr(ds, "Columns", pixels.shape[1])),
        "bits_stored": int(getattr(ds, "BitsStored", 0) or 0),
        "photometric": str(getattr(ds, "PhotometricInterpretation", "")),
        "pixel_spacing": [float(x) for x in spacing] if spacing is not None else None,
        "view_position": str(getattr(ds, "ViewPosition", "") or ""),
        "patient_orientation": [str(x) for x in getattr(ds, "PatientOrientation", [])],
        "image_laterality": str(getattr(ds, "ImageLaterality", "") or ""),
        "laterality": str(getattr(ds, "Laterality", "") or ""),
        "image_orientation_patient": [
            float(x) for x in getattr(ds, "ImageOrientationPatient", [])
        ],
    }
    return array, metadata


def normalized_box(rows: list[dict], original_width: int, original_height: int) -> list[float]:
    x0 = min(float(r["x_min"]) for r in rows) / original_width
    y0 = min(float(r["y_min"]) for r in rows) / original_height
    x1 = max(float(r["x_max"]) for r in rows) / original_width
    y1 = max(float(r["y_max"]) for r in rows) / original_height
    return [x0, y0, x1, y1]


def pixel_box(box: list[float], size: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    pad_x = 0.10 * (x1 - x0)
    pad_y = 0.10 * (y1 - y0)
    xa = max(1, int(np.floor((x0 - pad_x) * size)))
    ya = max(1, int(np.floor((y0 - pad_y) * size)))
    xb = min(size - 1, int(np.ceil((x1 + pad_x) * size)))
    yb = min(size - 1, int(np.ceil((y1 + pad_y) * size)))
    if xb - xa < 4 or yb - ya < 4:
        raise ValueError("degenerate resized box")
    return xa, ya, xb, yb


def descriptor(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Low-resolution anatomy/context descriptor, excluding the target ROI."""
    size = image.shape[0]
    small = np.asarray(
        Image.fromarray(np.round(image * 255).astype(np.uint8), mode="L").resize(
            (24, 24), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    xa, ya, xb, yb = box
    sx0, sy0 = int(xa * 24 / size), int(ya * 24 / size)
    sx1, sy1 = max(sx0 + 1, int(np.ceil(xb * 24 / size))), max(
        sy0 + 1, int(np.ceil(yb * 24 / size))
    )
    mask = np.ones((24, 24), dtype=bool)
    mask[max(0, sy0 - 1) : min(24, sy1 + 1), max(0, sx0 - 1) : min(24, sx1 + 1)] = False
    values = small[mask]
    mean, std = float(values.mean()), float(values.std() + 1e-6)
    normalized = (small - mean) / std
    normalized[~mask] = 0.0
    gradients = np.hypot(ndimage.sobel(small, axis=0), ndimage.sobel(small, axis=1))
    gradients[~mask] = 0.0
    return np.concatenate(
        [normalized.ravel(), gradients.ravel(), np.asarray([mean, std], dtype=np.float32)]
    )


def cosine_alpha(height: int, width: int) -> np.ndarray:
    transition = max(2, min(height, width) // 6)
    yy = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(float)
    xx = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(float)
    distance = np.minimum(yy[:, None], xx[None, :])
    t = np.clip(distance / transition, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def harmonize_patch(patch: np.ndarray, recipient: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    xa, ya, xb, yb = box
    pad = max(2, min(xb - xa, yb - ya) // 4)
    y0, y1 = max(0, ya - pad), min(recipient.shape[0], yb + pad)
    x0, x1 = max(0, xa - pad), min(recipient.shape[1], xb + pad)
    context = recipient[y0:y1, x0:x1].copy()
    context[max(0, ya - y0) : yb - y0, max(0, xa - x0) : xb - x0] = np.nan
    target = context[np.isfinite(context)]
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    transformed = (patch - border.mean()) / (border.std() + 1e-5)
    transformed = transformed * (target.std() + 1e-5) + target.mean()
    return np.clip(transformed, 0, 1)


def replace_patch(
    recipient: np.ndarray,
    donor: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    xa, ya, xb, yb = box
    patch = harmonize_patch(donor[ya:yb, xa:xb], recipient, box)
    alpha = cosine_alpha(yb - ya, xb - xa)
    output = recipient.copy()
    output[ya:yb, xa:xb] = (
        alpha * patch + (1.0 - alpha) * recipient[ya:yb, xa:xb]
    )
    return output


def candidate_features(image: np.ndarray, box: tuple[int, int, int, int]) -> list[float]:
    xa, ya, xb, yb = box
    roi = image[ya:yb, xa:xb]
    pad = max(2, min(xb - xa, yb - ya) // 5)
    outer = image[max(0, ya - pad) : min(image.shape[0], yb + pad), max(0, xa - pad) : min(image.shape[1], xb + pad)].copy()
    inner_y0, inner_x0 = ya - max(0, ya - pad), xa - max(0, xa - pad)
    ring_mask = np.ones_like(outer, dtype=bool)
    ring_mask[inner_y0 : inner_y0 + (yb - ya), inner_x0 : inner_x0 + (xb - xa)] = False
    ring = outer[ring_mask]
    jumps = np.concatenate(
        [
            np.abs(image[ya, xa:xb] - image[ya - 1, xa:xb]),
            np.abs(image[yb - 1, xa:xb] - image[yb, xa:xb]),
            np.abs(image[ya:yb, xa] - image[ya:yb, xa - 1]),
            np.abs(image[ya:yb, xb - 1] - image[ya:yb, xb]),
        ]
    )
    lap_roi = np.abs(ndimage.laplace(roi)).mean()
    lap_ring = np.abs(ndimage.laplace(outer))[ring_mask].mean()
    return [
        float(jumps.mean()),
        float(np.quantile(jumps, 0.95)),
        float(roi.mean() - ring.mean()),
        float(np.log((roi.std() + 1e-5) / (ring.std() + 1e-5))),
        float(np.log((lap_roi + 1e-5) / (lap_ring + 1e-5))),
        float(np.mean(np.abs(roi - ndimage.gaussian_filter(roi, sigma=1.0)))),
    ]


def bootstrap_detectability(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    values = []
    for _ in range(draws):
        chosen = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        if len(np.unique(labels[idx])) < 2:
            continue
        auc = roc_auc_score(labels[idx], scores[idx])
        values.append(max(float(auc), 1.0 - float(auc)))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("/home/dbw/datasets/physionet/vindr-cxr/1.0.0/manifests_v2/reader_vote_manifest_v2.jsonl"))
    parser.add_argument("--bbox-csv", type=Path, default=Path("/workspace/vinbigdata/train.csv"))
    parser.add_argument("--dicom-root", type=Path, default=Path("/workspace/vinbigdata/train"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="confirmation")
    parser.add_argument("--per-state", type=int, default=16)
    parser.add_argument("--matched-donors", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir()

    vote_records = {}
    for line in args.manifest.read_text().splitlines():
        row = json.loads(line)
        if row["finding"] == FINDING_KEY and row["experiment_split"] == args.split and row["positive_votes"] in {0, 3}:
            vote_records[(row["image_id"], row["positive_votes"])] = row

    bbox_rows: dict[str, list[dict]] = defaultdict(list)
    with args.bbox_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["class_name"] == FINDING_CSV and row["rad_id"] in PANEL and row["x_min"]:
                bbox_rows[row["image_id"]].append(row)

    positive_ids = sorted(
        [iid for (iid, votes) in vote_records if votes == 3 and iid in bbox_rows],
        key=lambda iid: stable(args.seed, "positive", iid),
    )[: args.per_state]
    negative_ids = sorted(
        [iid for (iid, votes) in vote_records if votes == 0],
        key=lambda iid: stable(args.seed, "negative", iid),
    )
    if len(positive_ids) < args.per_state or len(negative_ids) < args.per_state + args.matched_donors:
        raise RuntimeError("insufficient positive/negative fixed-panel cases")
    recipient_negative_ids = negative_ids[: args.per_state]
    donor_ids = negative_ids[args.per_state :]

    cache: dict[str, tuple[np.ndarray, dict]] = {}
    for iid in sorted(set(positive_ids + recipient_negative_ids + donor_ids)):
        cache[iid] = render_dicom(args.dicom_root / f"{iid}.dicom", args.size)

    templates = []
    for iid in positive_ids:
        meta = cache[iid][1]
        templates.append(normalized_box(bbox_rows[iid], meta["columns"], meta["rows"]))

    recipients = [(iid, 3, templates[index]) for index, iid in enumerate(positive_ids)]
    recipients += [(iid, 0, templates[index]) for index, iid in enumerate(recipient_negative_ids)]
    records = []
    donor_distances, random_distances = [], []
    rng = np.random.default_rng(args.seed)
    for recipient_index, (iid, votes, normalized) in enumerate(recipients):
        image, metadata = cache[iid]
        box = pixel_box(normalized, args.size)
        recipient_descriptor = descriptor(image, box)
        distances = []
        for donor_id in donor_ids:
            donor_image, _ = cache[donor_id]
            distances.append((float(np.mean((recipient_descriptor - descriptor(donor_image, box)) ** 2)), donor_id))
        distances.sort()
        matched = [donor_id for _, donor_id in distances[: args.matched_donors]]
        random_pool = [donor_id for _, donor_id in distances[args.matched_donors :]]
        random_ids = list(rng.choice(random_pool, args.matched_donors, replace=False))
        donor_distances.extend([distance for distance, _ in distances[: args.matched_donors]])
        lookup = dict((donor_id, distance) for distance, donor_id in distances)
        random_distances.extend([lookup[str(donor_id)] for donor_id in random_ids])

        candidates = [("self", iid)] + [("matched", x) for x in matched] + [("random", str(x)) for x in random_ids]
        for candidate_index, (kind, donor_id) in enumerate(candidates):
            donor_image = image if kind == "self" else cache[donor_id][0]
            candidate = replace_patch(image, donor_image, box)
            path = image_dir / f"{recipient_index:02d}_{votes}_{kind}_{candidate_index}_{donor_id}.png"
            Image.fromarray(np.round(candidate * 255).astype(np.uint8), mode="L").save(path)
            outside = np.ones_like(image, dtype=bool)
            xa, ya, xb, yb = box
            outside[ya:yb, xa:xb] = False
            records.append(
                {
                    "recipient_id": iid,
                    "recipient_votes": votes,
                    "candidate_kind": kind,
                    "donor_id": donor_id,
                    "donor_votes": 0 if kind != "self" else votes,
                    "normalized_box": normalized,
                    "pixel_box": list(box),
                    "features": candidate_features(candidate, box),
                    "context_distance": 0.0 if kind == "self" else lookup[donor_id],
                    "outside_max_abs_change": float(np.max(np.abs(candidate[outside] - image[outside]))),
                    "inside_mean_abs_change": float(np.mean(np.abs(candidate[~outside] - image[~outside]))),
                    "image": str(path.resolve()),
                    "recipient_metadata": metadata,
                }
            )

    # Low-level edit detector: can an observer identify self-reconstruction
    # versus matched negative replacements on claim-negative recipients?
    audit = [r for r in records if r["recipient_votes"] == 0 and r["candidate_kind"] in {"self", "matched"}]
    x = np.asarray([r["features"] for r in audit], dtype=float)
    y = np.asarray([r["candidate_kind"] == "matched" for r in audit], dtype=int)
    groups = np.asarray([r["recipient_id"] for r in audit])
    predictions = np.zeros(len(audit), dtype=float)
    folds = min(8, len(np.unique(groups)))
    for train, test in GroupKFold(folds).split(x, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
        model.fit(x[train], y[train])
        predictions[test] = model.predict_proba(x[test])[:, 1]
    auc = float(roc_auc_score(y, predictions))
    detectability = max(auc, 1.0 - auc)
    detect_ci = bootstrap_detectability(y, predictions, groups, args.bootstrap_draws, args.seed)

    matched_distances = np.asarray(donor_distances)
    random_distances_array = np.asarray(random_distances)
    positive_changes = [r["inside_mean_abs_change"] for r in records if r["recipient_votes"] == 3 and r["candidate_kind"] == "matched"]
    negative_changes = [r["inside_mean_abs_change"] for r in records if r["recipient_votes"] == 0 and r["candidate_kind"] == "matched"]
    metadata_presence = {}
    for key in ["view_position", "patient_orientation", "image_laterality", "laterality", "image_orientation_patient"]:
        metadata_presence[key] = int(sum(bool(cache[iid][1][key]) for iid in cache))
    structural_anatomy_observed = bool(metadata_presence["view_position"] and (metadata_presence["patient_orientation"] or metadata_presence["image_orientation_patient"]))
    edit_indistinguishable = bool(detect_ci[1] < 0.65)
    matching_improves = bool(np.median(matched_distances) < np.median(random_distances_array))
    outside_invariant = bool(max(r["outside_max_abs_change"] for r in records) <= 1e-7)
    gate_pass = structural_anatomy_observed and edit_indistinguishable and matching_improves and outside_invariant

    result = {
        "version": VERSION,
        "status": "complete",
        "created_at": now(),
        "configuration": vars(args) | {"manifest": str(args.manifest.resolve()), "bbox_csv": str(args.bbox_csv.resolve()), "dicom_root": str(args.dicom_root.resolve()), "output_dir": str(args.output_dir.resolve())},
        "provenance": {
            "command": " ".join(sys.argv),
            "source_sha256": sha256_file(Path(__file__)),
            "manifest_sha256": sha256_file(args.manifest),
            "bbox_csv_sha256": sha256_file(args.bbox_csv),
            "dataset": "VinDr-CXR 1.0.0 fixed reader panel R8/R9/R10",
            "model": "none (CPU image-manipulation audit)",
            "method": "same-coordinate negative-donor patch, context-nearest-neighbor match, border moment harmonization, cosine feather",
            "seed": args.seed,
        },
        "n_recipients": len(recipients),
        "n_negative": len(recipient_negative_ids),
        "n_positive": len(positive_ids),
        "n_candidates": len(records),
        "metadata_presence_among_loaded_images": {**metadata_presence, "denominator": len(cache)},
        "matching": {
            "matched_context_distance_median": float(np.median(matched_distances)),
            "random_context_distance_median": float(np.median(random_distances_array)),
            "median_ratio": float(np.median(matched_distances) / np.median(random_distances_array)),
            "improves_over_random": matching_improves,
        },
        "exchangeability_diagnostics": {
            "estimand": "low-level detectability of matched replacement versus self-reconstruction on 0/3 recipients",
            "group_oof_auc": auc,
            "direction_free_detectability": detectability,
            "recipient_cluster_bootstrap_95_ci": detect_ci,
            "frozen_pass_rule": "upper 95% CI of direction-free detectability < 0.65",
            "pass": edit_indistinguishable,
            "caveat": "Failure to distinguish cannot prove conditional exchangeability; this detector only tries to falsify it.",
        },
        "manipulation_diagnostics": {
            "outside_roi_max_abs_change": max(r["outside_max_abs_change"] for r in records),
            "outside_exactly_invariant": outside_invariant,
            "positive_inside_mean_abs_change": float(np.mean(positive_changes)),
            "negative_inside_mean_abs_change": float(np.mean(negative_changes)),
            "all_nonself_donors_are_fixed_panel_0_of_3": True,
            "boundary": "Pixel change and donor label verify operator execution, not clinical lesion removal.",
        },
        "decision": {
            "structural_anatomy_condition_observed": structural_anatomy_observed,
            "edit_indistinguishable": edit_indistinguishable,
            "matching_improves_over_random": matching_improves,
            "outside_invariant": outside_invariant,
            "pass_cpu_gate": gate_pass,
            "schedule_gpu_32": gate_pass,
            "reason": "All four fail-closed conditions passed." if gate_pass else "At least one structural or manipulation/exchangeability condition failed; do not schedule GPU scoring.",
        },
        "records_file": str((args.output_dir / "records.jsonl").resolve()),
        "interpretation_boundary": "This is an approximate anatomy-matched imputation audit, not an exact CRT and not a mitigation result.",
    }
    with (args.output_dir / "records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    atomic_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
