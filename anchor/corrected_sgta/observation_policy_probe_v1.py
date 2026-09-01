#!/usr/bin/env python3
"""Build and analyze the Observation-Policy Pragmatics probe.

The build stage creates five deterministic renderings of one frozen ROI and
asks three provenance-framed versions of the same claim.  Crucially, all three
prompts for a render point to the *same PNG*, so provenance is changed without
changing a pixel.  The analysis stage tests whether random-provenance wording
has a crop-specific effect beyond its ordinary effect on a full radiograph.

This is a mechanism probe, not a mitigation benchmark.  It deliberately uses
the already-confirmed 62-image global-negative panel and a finding-matched
62-image, reader-unanimous-positive panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter

from corrected_sgta.run_huatuo_vindr_commitment_probe import dicom_to_pil, sha256_file


VERSION = "observation-policy-pragmatics-v1"
SEED = 20260812
RENDERS = (
    "full",
    "native_context_removed",
    "native_sham_panel",
    "zoom_sham_panel",
    "zoom_true_context_panel",
)
PROMPTS = ("neutral", "random_provenance", "suspicious_provenance")
PRIMARY_CROP = "zoom_sham_panel"
FINDING_TO_CLASS = {
    "aortic_enlargement": "Aortic enlargement",
    "cardiomegaly": "Cardiomegaly",
    "lung_opacity": "Lung Opacity",
    "nodule_mass": "Nodule/Mass",
    "pleural_effusion": "Pleural effusion",
    "pleural_thickening": "Pleural thickening",
    "pulmonary_fibrosis": "Pulmonary fibrosis",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def stable_seed(*values: object) -> int:
    payload = "|".join(map(str, values)).encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def question(finding: str, prompt: str) -> str:
    label = finding.replace("_", " ")
    base = f"Does this chest X-ray show {label}?"
    if prompt == "neutral":
        return base
    if prompt == "random_provenance":
        return (
            f"An automated procedure selected this view uniformly at random before checking "
            f"for {label}; selection of the view is not clinical evidence. {base}"
        )
    if prompt == "suspicious_provenance":
        return (
            f"A clinical observer selected this view after identifying a region suspicious "
            f"for {label}; selection of the view may reflect clinical concern. {base}"
        )
    raise ValueError(f"unknown prompt: {prompt}")


def grid_crop_box(
    image_size: tuple[int, int], row: int, col: int, side: int, window: int
) -> tuple[int, int, int, int]:
    """Reproduce the 50%-FOV k1/r16 placebo ROI from the search-reuse probe."""
    width, height = image_size
    x0, x1 = int(round(col / side * width)), int(round((col + window) / side * width))
    y0, y1 = int(round(row / side * height)), int(round((row + window) / side * height))
    pad_x, pad_y = max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)
    target_width = min(width, (x1 - x0) + 2 * pad_x)
    target_height = min(height, (y1 - y0) + 2 * pad_y)
    left = max(0, min(x0 - pad_x, width - target_width))
    top = max(0, min(y0 - pad_y, height - target_height))
    return left, top, left + target_width, top + target_height


def centered_roi_box(
    image_size: tuple[int, int], center: tuple[float, float], fraction: float
) -> tuple[int, int, int, int]:
    """Return a fixed-FOV ROI that contains the supplied lesion center."""
    if not 0 < fraction <= 1:
        raise ValueError("ROI fraction must lie in (0, 1]")
    width, height = image_size
    roi_width = max(1, int(round(width * fraction)))
    roi_height = max(1, int(round(height * fraction)))
    cx, cy = center
    left = int(round(cx - roi_width / 2))
    top = int(round(cy - roi_height / 2))
    left = max(0, min(left, width - roi_width))
    top = max(0, min(top, height - roi_height))
    return left, top, left + roi_width, top + roi_height


def phase_scramble(image: Image.Image, seed: int) -> Image.Image:
    """Destroy anatomy while approximately preserving spectrum and histogram."""
    array = np.asarray(image.convert("L"), dtype=np.float64)
    spectrum = np.fft.rfft2(array)
    phase = np.random.default_rng(seed).uniform(-np.pi, np.pi, spectrum.shape)
    phase[0, 0] = float(np.angle(spectrum[0, 0]))
    scrambled = np.fft.irfft2(np.abs(spectrum) * np.exp(1j * phase), s=array.shape)
    # Exact rank-based histogram matching removes brightness as an easy cue.
    source_sorted = np.sort(array.ravel())
    ordering = np.argsort(scrambled.ravel(), kind="mergesort")
    matched = np.empty(scrambled.size, dtype=np.float64)
    matched[ordering] = source_sorted
    output = np.clip(matched.reshape(array.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="L").convert("RGB")


def fit_inside(image: Image.Image, maximum: tuple[int, int], enlarge: bool = True) -> Image.Image:
    max_width, max_height = maximum
    scale = min(max_width / image.width, max_height / image.height)
    if not enlarge:
        scale = min(scale, 1.0)
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    if size == image.size:
        return image.copy()
    return image.resize(size, Image.Resampling.BICUBIC)


def panel_render(
    full: Image.Image,
    roi: Image.Image,
    context: Image.Image,
    *,
    zoom: bool,
) -> Image.Image:
    """Place ROI and context on a fixed full-image-sized canvas."""
    full = full.convert("RGB")
    width, height = full.size
    gap = max(2, int(round(width * 0.015)))
    context_width = max(1, int(round(width * 0.24)))
    main_width = width - context_width - 3 * gap
    usable_height = height - 2 * gap
    background = tuple(int(v) for v in np.median(np.asarray(full), axis=(0, 1)))
    canvas = Image.new("RGB", full.size, background)
    roi_fit = fit_inside(roi, (main_width, usable_height), enlarge=zoom)
    roi_left = gap + (main_width - roi_fit.width) // 2
    roi_top = gap + (usable_height - roi_fit.height) // 2
    context_fit = fit_inside(context, (context_width, usable_height), enlarge=True)
    context_left = width - gap - context_width + (context_width - context_fit.width) // 2
    context_top = gap + (usable_height - context_fit.height) // 2
    canvas.paste(roi_fit, (roi_left, roi_top))
    canvas.paste(context_fit, (context_left, context_top))
    return canvas


def render_variants(
    full: Image.Image,
    roi_box: tuple[int, int, int, int],
    *,
    seed: int,
    blur_radius_fraction: float,
) -> dict[str, Image.Image]:
    full = full.convert("RGB")
    roi = full.crop(roi_box)
    radius = max(2.0, min(full.size) * blur_radius_fraction)
    blurred = full.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred.paste(roi, (roi_box[0], roi_box[1]))
    sham = phase_scramble(full, seed)
    renders = {
        "full": full.copy(),
        "native_context_removed": blurred,
        "native_sham_panel": panel_render(full, roi, sham, zoom=False),
        "zoom_sham_panel": panel_render(full, roi, sham, zoom=True),
        "zoom_true_context_panel": panel_render(full, roi, full, zoom=True),
    }
    if tuple(renders) != RENDERS or any(value.size != full.size for value in renders.values()):
        raise AssertionError("render contract violated")
    return renders


def load_negative_selections(search_reuse_dir: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(search_reuse_dir / "selections.jsonl")
    selected = [
        row for row in rows
        if int(row["claim_count"]) == 1
        and int(row["region_count"]) == 16
        and row["variant"] == "random"
    ]
    if len(selected) != 62 or len({row["image_id"] for row in selected}) != 62:
        raise ValueError(f"expected the frozen 62-image k1/r16 random panel, found {len(selected)}")
    return sorted(selected, key=lambda row: row["image_id"])


def load_positive_pool(fresh_hidden: Path) -> dict[str, list[dict[str, Any]]]:
    by_finding: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(fresh_hidden / "metadata.jsonl"):
        if int(row["positive_votes"]) == 3 and row["finding"] in FINDING_TO_CLASS:
            by_finding[row["finding"]].append(row)
    for finding in by_finding:
        by_finding[finding].sort(key=lambda row: stable_seed(SEED, "positive", row["record_key"]))
    return by_finding


def load_bbox_centers(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    centers: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    class_to_finding = {value: key for key, value in FINDING_TO_CLASS.items()}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            finding = class_to_finding.get(row["class_name"])
            if finding is None or not row.get("x_min"):
                continue
            centers[(row["image_id"], finding)].append((
                (float(row["x_min"]) + float(row["x_max"])) / 2,
                (float(row["y_min"]) + float(row["y_max"])) / 2,
            ))
    return {
        key: (float(np.median([p[0] for p in values])), float(np.median([p[1] for p in values])))
        for key, values in centers.items()
    }


def select_positive_rows(
    negative_rows: Iterable[dict[str, Any]],
    positive_pool: dict[str, list[dict[str, Any]]],
    bbox_centers: dict[tuple[str, str], tuple[float, float]],
) -> list[dict[str, Any]]:
    requested = Counter(row["finding"] for row in negative_rows)
    output = []
    for finding in FINDING_TO_CLASS:
        eligible = [
            row for row in positive_pool.get(finding, [])
            if (row["image_id"], finding) in bbox_centers
        ]
        if len(eligible) < requested[finding]:
            raise ValueError(f"positive bbox shortage for {finding}: {len(eligible)} < {requested[finding]}")
        output.extend(eligible[: requested[finding]])
    if len(output) != sum(requested.values()) or Counter(row["finding"] for row in output) != requested:
        raise AssertionError("positive matching contract violated")
    return sorted(output, key=lambda row: row["image_id"])


def build(args: argparse.Namespace) -> None:
    if not 0 < args.roi_fraction <= 1:
        raise ValueError("--roi-fraction must lie in (0, 1]")
    receipt_path = args.output_dir / "provenance_receipt.json"
    inputs = {
        "search_selections_sha256": sha256_file(args.search_reuse_dir / "selections.jsonl"),
        "search_receipt_sha256": sha256_file(args.search_reuse_dir / "receipt.json"),
        "fresh_metadata_sha256": sha256_file(args.fresh_hidden / "metadata.jsonl"),
        "bbox_csv_sha256": sha256_file(args.bbox_csv),
    }
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        expected = {
            "version": VERSION,
            "inputs": inputs,
            "roi_fraction": args.roi_fraction,
            "blur_radius_fraction": args.blur_radius_fraction,
            "seed": args.seed,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("completed build configuration drift")
        print(json.dumps({"status": "already_complete", "manifest_n": receipt["manifest_n"]}))
        return

    negatives = load_negative_selections(args.search_reuse_dir)
    bbox_centers = load_bbox_centers(args.bbox_csv)
    positives = select_positive_rows(negatives, load_positive_pool(args.fresh_hidden), bbox_centers)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    render_hashes: dict[str, str] = {}
    qid = 1

    base_rows: list[dict[str, Any]] = []
    for row in negatives:
        base_rows.append({
            "sample_id": f"negative-{row['image_id']}",
            "image_id": row["image_id"],
            "finding": row["finding"],
            "label": 0,
            "source": "search_reuse_huatuo_v1:k1_r16_random",
            "grid_window": row["random_window"],
        })
    for row in positives:
        base_rows.append({
            "sample_id": f"positive-{row['image_id']}",
            "image_id": row["image_id"],
            "finding": row["finding"],
            "label": 1,
            "source": "fresh_hidden:3_of_3_with_bbox",
            "bbox_center": bbox_centers[(row["image_id"], row["finding"])],
        })

    for base in base_rows:
        dicom_path = args.dicom_root / f"{base['image_id']}.dicom"
        full = dicom_to_pil(dicom_path).convert("RGB")
        if base["label"] == 0:
            row, col, window = map(int, base["grid_window"])
            roi_box = grid_crop_box(full.size, row, col, side=24, window=window)
        else:
            roi_box = centered_roi_box(full.size, tuple(base["bbox_center"]), args.roi_fraction)
        renders = render_variants(
            full,
            roi_box,
            seed=stable_seed(args.seed, base["image_id"], "sham"),
            blur_radius_fraction=args.blur_radius_fraction,
        )
        for render_name, rendered in renders.items():
            relative = f"{base['sample_id']}-{render_name}.png"
            target = image_dir / relative
            atomic_png(target, rendered)
            digest = sha256_file(target)
            render_hashes[f"{base['sample_id']}:{render_name}"] = digest
            for prompt in PROMPTS:
                record = {
                    "qid": qid,
                    "img_name": relative,
                    "question": question(base["finding"], prompt),
                    "answer": "yes" if base["label"] else "no",
                }
                manifest.append(record)
                selections.append({
                    **base,
                    "qid": qid,
                    "render": render_name,
                    "prompt": prompt,
                    "roi_box": list(roi_box),
                    "source_image_size": list(full.size),
                    "render_sha256": digest,
                    "img_name": relative,
                })
                qid += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "manifest.json", manifest)
    selections_path = args.output_dir / "selections.jsonl"
    selections_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selections))
    # A direct receipt makes the same-render/same-pixels contract auditable
    # without trusting filenames or the builder implementation.
    repeated_hash_ok = all(
        len({row["render_sha256"] for row in selections if row["sample_id"] == sample and row["render"] == render}) == 1
        for sample in {row["sample_id"] for row in selections}
        for render in RENDERS
    )
    finding_match = {
        "negative": dict(sorted(Counter(row["finding"] for row in base_rows if row["label"] == 0).items())),
        "positive": dict(sorted(Counter(row["finding"] for row in base_rows if row["label"] == 1).items())),
    }
    receipt = {
        "version": VERSION,
        "status": "complete",
        "seed": args.seed,
        "roi_fraction": args.roi_fraction,
        "blur_radius_fraction": args.blur_radius_fraction,
        "negative_images": len(negatives),
        "positive_images": len(positives),
        "finding_match": finding_match,
        "renders": list(RENDERS),
        "prompts": list(PROMPTS),
        "manifest_n": len(manifest),
        "expected_manifest_n": (len(negatives) + len(positives)) * len(RENDERS) * len(PROMPTS),
        "same_render_pixel_identity_across_prompts": repeated_hash_ok,
        "unique_render_files": len(render_hashes),
        "render_hashes_sha256": hashlib.sha256(
            json.dumps(render_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "inputs": inputs,
        "manifest_sha256": sha256_file(args.output_dir / "manifest.json"),
        "selections_sha256": sha256_file(selections_path),
        "source_sha256": sha256_file(Path(__file__)),
        "command": " ".join(sys.argv),
        "truth_boundary": "negative: all seven searched findings 0/3; positive: matched finding 3/3 with released bbox",
    }
    if finding_match["negative"] != finding_match["positive"] or not repeated_hash_ok:
        raise AssertionError("build provenance contract violated")
    atomic_json(receipt_path, receipt)
    print(json.dumps({"status": "complete", "manifest_n": len(manifest), "images": len(base_rows)}))


def mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> list[float]:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("paired bootstrap expects a non-empty one-dimensional vector")
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def metric(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, Any]:
    return {"mean": float(np.mean(values)), "ci95": mean_ci(values, rng, draws), "n": len(values)}


def paired_vector(
    table: dict[tuple[str, str, str], float],
    sample_ids: list[str],
    render: str,
    left_prompt: str,
    right_prompt: str,
) -> np.ndarray:
    return np.asarray([
        table[(sample, render, left_prompt)] - table[(sample, render, right_prompt)]
        for sample in sample_ids
    ], dtype=float)


def analyze(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    selections = {int(row["qid"]): row for row in read_jsonl(args.selections)}
    raw_by_qid: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(args.raw):
        if row.get("status") == "ok":
            qid = int(row.get("question_id", row.get("qid")))
            raw_by_qid[qid] = row
    if set(selections) != set(raw_by_qid):
        raise ValueError(f"score coverage mismatch: selections={len(selections)} raw={len(raw_by_qid)}")

    table: dict[tuple[str, str, str], float] = {}
    base: dict[str, dict[str, Any]] = {}
    pixel_identity_ok = True
    for qid, selection in selections.items():
        raw = raw_by_qid[qid]
        if raw.get("image_sha256") and raw["image_sha256"] != selection["render_sha256"]:
            raise ValueError(f"scorer/build image hash mismatch at qid={qid}")
        key = (selection["sample_id"], selection["render"], selection["prompt"])
        if key in table:
            raise ValueError(f"duplicate score key: {key}")
        table[key] = float(raw["scores"]["original_margin"])
        base.setdefault(selection["sample_id"], selection)
    for sample in base:
        for render in RENDERS:
            hashes = {
                selections[qid]["render_sha256"] for qid in selections
                if selections[qid]["sample_id"] == sample and selections[qid]["render"] == render
            }
            pixel_identity_ok &= len(hashes) == 1
            for prompt in PROMPTS:
                if (sample, render, prompt) not in table:
                    raise ValueError(f"missing factorial cell: {(sample, render, prompt)}")
    if not pixel_identity_ok:
        raise ValueError("same-render pixel identity failed")

    negative = sorted(sample for sample, row in base.items() if int(row["label"]) == 0)
    positive = sorted(sample for sample, row in base.items() if int(row["label"]) == 1)
    if not negative or not positive:
        raise ValueError("analysis requires negative and positive panels")
    rng = np.random.default_rng(args.seed)

    neg_crop_margin_drop = paired_vector(table, negative, PRIMARY_CROP, "neutral", "random_provenance")
    neg_full_margin_drop = paired_vector(table, negative, "full", "neutral", "random_provenance")
    gamma = neg_crop_margin_drop - neg_full_margin_drop
    neg_crop_fp_drop = np.asarray([
        float(table[(sample, PRIMARY_CROP, "neutral")] > 0)
        - float(table[(sample, PRIMARY_CROP, "random_provenance")] > 0)
        for sample in negative
    ])
    neg_full_fp_drop = np.asarray([
        float(table[(sample, "full", "neutral")] > 0)
        - float(table[(sample, "full", "random_provenance")] > 0)
        for sample in negative
    ])
    fp_drop_interaction = neg_crop_fp_drop - neg_full_fp_drop
    pos_crop_recall_loss = np.asarray([
        float(table[(sample, PRIMARY_CROP, "neutral")] > 0)
        - float(table[(sample, PRIMARY_CROP, "random_provenance")] > 0)
        for sample in positive
    ])

    summaries: dict[str, dict[str, Any]] = {}
    for label, ids in (("negative", negative), ("positive", positive)):
        for render in RENDERS:
            for prompt in PROMPTS:
                margins = np.asarray([table[(sample, render, prompt)] for sample in ids])
                summaries[f"{label}:{render}:{prompt}"] = {
                    "n": len(ids),
                    "mean_margin": float(margins.mean()),
                    "positive_rate": float(np.mean(margins > 0)),
                }

    def fp_contrast(left: str, right: str, prompt: str = "neutral") -> np.ndarray:
        return np.asarray([
            float(table[(sample, left, prompt)] > 0) - float(table[(sample, right, prompt)] > 0)
            for sample in negative
        ])

    def margin_contrast(left: str, right: str, prompt: str = "neutral") -> np.ndarray:
        return np.asarray([
            table[(sample, left, prompt)] - table[(sample, right, prompt)]
            for sample in negative
        ])

    mechanism_vectors = {
        "context_removed_minus_full": (
            margin_contrast("native_context_removed", "full"),
            fp_contrast("native_context_removed", "full"),
        ),
        "zoom_minus_native_with_sham": (
            margin_contrast("zoom_sham_panel", "native_sham_panel"),
            fp_contrast("zoom_sham_panel", "native_sham_panel"),
        ),
        "true_context_rescue_from_zoom_sham": (
            margin_contrast("zoom_sham_panel", "zoom_true_context_panel"),
            fp_contrast("zoom_sham_panel", "zoom_true_context_panel"),
        ),
    }
    mechanisms = {
        name: {
            "margin_inflation": metric(vectors[0], rng, args.bootstrap_draws),
            "fp_inflation": metric(vectors[1], rng, args.bootstrap_draws),
        }
        for name, vectors in mechanism_vectors.items()
    }
    suspicious_crop = paired_vector(table, negative, PRIMARY_CROP, "suspicious_provenance", "neutral")
    suspicious_full = paired_vector(table, negative, "full", "suspicious_provenance", "neutral")

    finding_primary: dict[str, dict[str, Any]] = {}
    for finding in FINDING_TO_CLASS:
        neg_ids = [sample for sample in negative if base[sample]["finding"] == finding]
        pos_ids = [sample for sample in positive if base[sample]["finding"] == finding]
        if not neg_ids or not pos_ids:
            continue
        crop_drop = paired_vector(table, neg_ids, PRIMARY_CROP, "neutral", "random_provenance")
        full_drop = paired_vector(table, neg_ids, "full", "neutral", "random_provenance")
        crop_fp = np.asarray([
            float(table[(sample, PRIMARY_CROP, "neutral")] > 0)
            - float(table[(sample, PRIMARY_CROP, "random_provenance")] > 0)
            for sample in neg_ids
        ])
        recall_loss = np.asarray([
            float(table[(sample, PRIMARY_CROP, "neutral")] > 0)
            - float(table[(sample, PRIMARY_CROP, "random_provenance")] > 0)
            for sample in pos_ids
        ])
        finding_primary[finding] = {
            "negative_n": len(neg_ids),
            "positive_n": len(pos_ids),
            "gamma_random_margin": metric(crop_drop - full_drop, rng, args.bootstrap_draws),
            "negative_crop_fp_drop": metric(crop_fp, rng, args.bootstrap_draws),
            "positive_crop_recall_loss": metric(recall_loss, rng, args.bootstrap_draws),
        }

    gamma_result = metric(gamma, rng, args.bootstrap_draws)
    crop_fp_result = metric(neg_crop_fp_drop, rng, args.bootstrap_draws)
    full_fp_result = metric(neg_full_fp_drop, rng, args.bootstrap_draws)
    fp_interaction_result = metric(fp_drop_interaction, rng, args.bootstrap_draws)
    recall_loss_result = metric(pos_crop_recall_loss, rng, args.bootstrap_draws)
    huatuo_gate = bool(
        gamma_result["mean"] > 0.25
        and gamma_result["ci95"][0] > 0
        and crop_fp_result["mean"] >= 0.10
        and abs(full_fp_result["mean"]) <= 0.03
        and fp_interaction_result["ci95"][0] > 0
        and recall_loss_result["mean"] <= 0.01
    )
    context_route = mechanisms["true_context_rescue_from_zoom_sham"]["fp_inflation"]["mean"] >= 0.10
    resize_route = mechanisms["zoom_minus_native_with_sham"]["fp_inflation"]["mean"] >= 0.10
    ordinary_criterion_shift = bool(
        abs(crop_fp_result["mean"] - full_fp_result["mean"]) < 0.03
        and fp_interaction_result["ci95"][0] <= 0 <= fp_interaction_result["ci95"][1]
    )
    result = {
        "version": VERSION,
        "status": "complete",
        "model": args.model,
        "sample_counts": {"negative": len(negative), "positive": len(positive)},
        "pixel_identity_across_prompt_counterfactuals": pixel_identity_ok,
        "primary": {
            "crop_render": PRIMARY_CROP,
            "gamma_random_margin": gamma_result,
            "negative_crop_fp_drop_neutral_to_random": crop_fp_result,
            "negative_full_fp_drop_neutral_to_random": full_fp_result,
            "negative_crop_specific_fp_drop": fp_interaction_result,
            "positive_crop_recall_loss_neutral_to_random": recall_loss_result,
            "single_model_gate": huatuo_gate,
            "formal_cross_model_go": False,
            "formal_cross_model_go_reason": "requires an independently scored Hulu replication and majority-finding agreement",
            "gate_rule": (
                "Gamma>0.25 with CI lower>0; crop FP drop>=10pp; |full FP change|<=3pp; "
                "crop-vs-full FP-drop CI lower>0; positive crop recall loss<=1pp"
            ),
        },
        "render_competition": mechanisms,
        "prompt_controls": {
            "suspicious_minus_neutral_crop_margin": metric(suspicious_crop, rng, args.bootstrap_draws),
            "suspicious_minus_neutral_full_margin": metric(suspicious_full, rng, args.bootstrap_draws),
        },
        "primary_by_finding": finding_primary,
        "routing": {
            "context_loss_if_provenance_fails": bool(context_route and not huatuo_gate),
            "resize_scale_ood_present": resize_route,
            "ordinary_criterion_shift": ordinary_criterion_shift,
            "observation_policy_candidate_survives_single_model": huatuo_gate,
        },
        "cell_summaries": summaries,
        "configuration": {
            "seed": args.seed,
            "bootstrap_draws": args.bootstrap_draws,
            "raw_sha256": sha256_file(args.raw),
            "selections_sha256": sha256_file(args.selections),
            "source_sha256": sha256_file(Path(__file__)),
            "command": " ".join(sys.argv),
        },
        "boundary": (
            "A single-model pass identifies a crop-specific provenance interaction; it is not a "
            "mitigation result and is not the preregistered cross-model GO."
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps(result["primary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    builder = subparsers.add_parser("build")
    builder.add_argument("--search-reuse-dir", type=Path, required=True)
    builder.add_argument("--fresh-hidden", type=Path, required=True)
    builder.add_argument("--bbox-csv", type=Path, required=True)
    builder.add_argument("--dicom-root", type=Path, required=True)
    builder.add_argument("--output-dir", type=Path, required=True)
    builder.add_argument("--roi-fraction", type=float, default=0.5)
    builder.add_argument("--blur-radius-fraction", type=float, default=0.025)
    builder.add_argument("--seed", type=int, default=SEED)
    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--selections", type=Path, required=True)
    analysis.add_argument("--raw", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--model", default="huatuo")
    analysis.add_argument("--bootstrap-draws", type=int, default=5000)
    analysis.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
