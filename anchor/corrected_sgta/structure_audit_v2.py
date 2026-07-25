"""Reproducible SSIM and CXR structure-proxy audit for selected Wave-A views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from corrected_sgta.cache import iter_successes
from corrected_sgta.frequency_alignment_v2 import feddg_frequency_interpolation_v2
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.protocol_v2 import resolve_image
from corrected_sgta.source_bank_v2 import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ssim-min", type=float, default=0.90)
    parser.add_argument("--local-contrast-correlation-min", type=float, default=0.85)
    parser.add_argument("--gradient-ratio-min", type=float, default=0.75)
    parser.add_argument("--gradient-ratio-max", type=float, default=1.25)
    return parser.parse_args()


def gray(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float64)


def ssim(left: np.ndarray, right: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mu_right = cv2.GaussianBlur(right, (11, 11), 1.5)
    sigma_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mu_left * mu_left
    sigma_right = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mu_right * mu_right
    sigma_cross = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mu_left * mu_right
    score = ((2 * mu_left * mu_right + c1) * (2 * sigma_cross + c2)) / (
        (mu_left * mu_left + mu_right * mu_right + c1)
        * (sigma_left + sigma_right + c2)
    )
    return float(np.mean(score))


def central_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    mask[int(0.08 * height) : int(0.92 * height), int(0.08 * width) : int(0.92 * width)] = True
    return mask


def structure_proxy(left: np.ndarray, right: np.ndarray) -> dict:
    mask = central_mask(left.shape)
    left_local = left - cv2.GaussianBlur(left, (0, 0), 3.0)
    right_local = right - cv2.GaussianBlur(right, (0, 0), 3.0)
    a, b = left_local[mask], right_local[mask]
    correlation = 0.0 if a.std() < 1e-8 or b.std() < 1e-8 else float(np.corrcoef(a, b)[0, 1])
    left_gx = cv2.Sobel(left, cv2.CV_64F, 1, 0, ksize=3)
    left_gy = cv2.Sobel(left, cv2.CV_64F, 0, 1, ksize=3)
    right_gx = cv2.Sobel(right, cv2.CV_64F, 1, 0, ksize=3)
    right_gy = cv2.Sobel(right, cv2.CV_64F, 0, 1, ksize=3)
    left_gradient = np.hypot(left_gx, left_gy)[mask].mean()
    right_gradient = np.hypot(right_gx, right_gy)[mask].mean()
    return {
        "central_local_contrast_correlation": correlation,
        "central_gradient_magnitude_ratio": float(right_gradient / max(left_gradient, 1e-12)),
        "scope": "deterministic CXR structure proxy; not a validated lesion segmenter",
    }


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text())
    rows = list(iter_successes(args.cache, metadata["fingerprint"]))
    manifest = load_manifest(args.source_bank)
    entries = {item["source_id"]: item for item in manifest.get("entries", [])}
    records = []
    for row in rows:
        image_path = resolve_image(row.get("img_name", ""))
        if image_path is None:
            raise RuntimeError(f"cannot resolve cached image: {row.get('img_name')}")
        with Image.open(image_path) as source:
            original = resize_image(source, metadata["config"]["max_image_side"])
        original_gray = gray(original)
        selected = [item for item in row.get("alignment_candidates", []) if item.get("selected")]
        for item in selected:
            for role, source_id in (
                ("matched", item["source_id"]),
                ("wrong_control", item["wrong_source_id"]),
            ):
                entry = entries[source_id]
                transformed = feddg_frequency_interpolation_v2(
                    original,
                    np.load(entry["amplitude_file"], allow_pickle=False),
                    low_frequency_ratio=float(item["low_frequency_ratio"]),
                    source_ratio=float(item["source_ratio"]),
                )
                transformed_gray = gray(transformed)
                proxy = structure_proxy(original_gray, transformed_gray)
                record = {
                    "qid": row["qid"],
                    "role": role,
                    "target_source_id": item["source_id"],
                    "amplitude_source_id": source_id,
                    "low_frequency_ratio": item["low_frequency_ratio"],
                    "ssim": ssim(original_gray, transformed_gray),
                    **proxy,
                }
                record["pass"] = (
                    record["ssim"] >= args.ssim_min
                    and record["central_local_contrast_correlation"]
                    >= args.local_contrast_correlation_min
                    and args.gradient_ratio_min
                    <= record["central_gradient_magnitude_ratio"]
                    <= args.gradient_ratio_max
                )
                records.append(record)
    matched = [item for item in records if item["role"] == "matched"]
    wrong = [item for item in records if item["role"] == "wrong_control"]

    def summarize(values: list[dict]) -> dict:
        return {
            "n": len(values),
            "pass_rate": None if not values else float(np.mean([item["pass"] for item in values])),
            "ssim_median": None if not values else float(np.median([item["ssim"] for item in values])),
            "local_contrast_correlation_median": None
            if not values
            else float(np.median([item["central_local_contrast_correlation"] for item in values])),
            "gradient_ratio_median": None
            if not values
            else float(np.median([item["central_gradient_magnitude_ratio"] for item in values])),
        }

    report = {
        "version": "sgta-structure-audit-v2",
        "source_cache": str(args.cache),
        "fingerprint": metadata["fingerprint"],
        "thresholds": {
            "ssim_min": args.ssim_min,
            "local_contrast_correlation_min": args.local_contrast_correlation_min,
            "gradient_ratio": [args.gradient_ratio_min, args.gradient_ratio_max],
        },
        "matched": summarize(matched),
        "wrong_control": summarize(wrong),
        "formal_matched_structure_pass": bool(matched) and all(item["pass"] for item in matched),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.output)
    print(json.dumps({key: report[key] for key in ("matched", "wrong_control", "formal_matched_structure_pass")}, indent=2))


if __name__ == "__main__":
    main()
