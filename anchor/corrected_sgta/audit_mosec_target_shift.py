#!/usr/bin/env python3
"""Audit target-image distance from a Huatuo source-frequency envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corrected_sgta.mosec import (
    gamma_style_shift,
    load_bank,
    model_visible_image,
    radial_log_amplitude,
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    payload = json.loads(text) if text.lstrip().startswith("[") else None
    if payload is not None:
        return list(payload)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def identifier(row: dict[str, Any], index: int) -> str:
    return str(row.get("question_id", row.get("qid", row.get("id", index))))


def image_name(row: dict[str, Any]) -> str:
    value = row.get("image", row.get("img_name"))
    if isinstance(value, list):
        value = value[0]
    if not value:
        raise ValueError("row has no image/img_name")
    return str(value)


def select_rows(
    rows: list[dict[str, Any]], maximum: int, seed: int
) -> list[dict[str, Any]]:
    if maximum <= 0 or maximum >= len(rows):
        return rows
    scored = []
    for index, row in enumerate(rows):
        key = f"{seed}:{identifier(row, index)}".encode()
        scored.append((hashlib.sha256(key).hexdigest(), row))
    return [row for _, row in sorted(scored)[:maximum]]


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-gamma", type=float, default=1.0)
    parser.add_argument("--allow-truncated-images", action="store_true")
    args = parser.parse_args()
    if args.input_gamma <= 0.0:
        parser.error("--input-gamma must be positive")

    ImageFile.LOAD_TRUNCATED_IMAGES = args.allow_truncated_images
    rows = select_rows(load_rows(args.input), args.max_samples, args.seed)
    bank = load_bank(args.bank)
    scale = np.maximum(bank["scale"].astype(np.float64), 1e-6)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for index, row in enumerate(rows):
        qid = identifier(row, index)
        path = args.image_root / image_name(row)
        try:
            with Image.open(path) as source:
                visible = model_visible_image(source.convert("RGB"))
                visible = gamma_style_shift(visible, args.input_gamma)
            descriptor = radial_log_amplitude(visible).astype(np.float64)
            below = np.maximum(bank["lower"] - descriptor, 0.0)
            above = np.maximum(descriptor - bank["upper"], 0.0)
            exceedance = below + above
            normalized = exceedance / scale
            center_distance = np.abs(descriptor - bank["median"]) / scale
            records.append(
                {
                    "question_id": qid,
                    "image": str(path),
                    "identity": bool(np.all(exceedance <= 1e-8)),
                    "outside_band_count": int(np.count_nonzero(exceedance > 1e-8)),
                    "mean_normalized_exceedance": float(normalized.mean()),
                    "max_normalized_exceedance": float(normalized.max()),
                    "mean_center_distance": float(center_distance.mean()),
                    "max_center_distance": float(center_distance.max()),
                }
            )
        except Exception as exc:
            errors.append(
                {"question_id": qid, "image": str(path), "error": repr(exc)}
            )

    if not records:
        raise SystemExit("no readable images")
    payload = {
        "version": "mosec-target-shift-audit-v1",
        "dataset": args.dataset,
        "input": str(args.input),
        "image_root": str(args.image_root),
        "bank": str(args.bank),
        "input_style": {"type": "gamma", "gamma": args.input_gamma},
        "n_requested": len(rows),
        "n_readable": len(records),
        "n_errors": len(errors),
        "identity_rate": sum(row["identity"] for row in records) / len(records),
        "outside_band_count": percentile_summary(
            [row["outside_band_count"] for row in records]
        ),
        "mean_normalized_exceedance": percentile_summary(
            [row["mean_normalized_exceedance"] for row in records]
        ),
        "max_normalized_exceedance": percentile_summary(
            [row["max_normalized_exceedance"] for row in records]
        ),
        "mean_center_distance": percentile_summary(
            [row["mean_center_distance"] for row in records]
        ),
        "records": records,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"records", "errors"}},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
