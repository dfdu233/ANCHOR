#!/usr/bin/env python3
"""Audit local source projections without loading a VLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corrected_sgta.local_source_projection import (
    load_local_source_index,
    local_source_projection,
    source_mean_std_projection,
)
from corrected_sgta.mosec import model_visible_image
from corrected_sgta.run_huatuo_mosec import (
    image_name,
    load_rows,
    row_id,
    select_rows,
)


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--local-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--low-frequency-ratios",
        type=float,
        nargs="+",
        default=(0.03, 0.10),
    )
    parser.add_argument(
        "--source-stat-strengths",
        type=float,
        nargs="*",
        default=(),
    )
    parser.add_argument("--allow-truncated-images", action="store_true")
    args = parser.parse_args()
    ImageFile.LOAD_TRUNCATED_IMAGES = args.allow_truncated_images

    rows = select_rows(load_rows(args.input), args.max_samples, args.seed)
    source = load_local_source_index(args.local_index)
    records: list[dict[str, Any]] = []
    errors = []
    for index, row in enumerate(rows):
        path = args.image_root / image_name(row)
        try:
            with Image.open(path) as raw:
                image = model_visible_image(raw.convert("RGB"))
            for ratio in args.low_frequency_ratios:
                _, metadata = local_source_projection(
                    image, source, low_frequency_ratio=ratio
                )
                records.append(
                    {
                        "item_id": row_id(row, index),
                        "image": str(path),
                        "low_frequency_ratio": ratio,
                        **metadata,
                    }
                )
            for strength in args.source_stat_strengths:
                _, metadata = source_mean_std_projection(
                    image, source, strength=strength
                )
                records.append(
                    {
                        "item_id": row_id(row, index),
                        "image": str(path),
                        "source_stat_strength": strength,
                        **metadata,
                    }
                )
        except Exception as exc:
            errors.append(
                {
                    "item_id": row_id(row, index),
                    "image": str(path),
                    "error": repr(exc),
                }
            )

    methods = {}
    for ratio in args.low_frequency_ratios:
        selected = [
            row
            for row in records
            if row.get("low_frequency_ratio") == ratio
        ]
        methods[f"local_l{ratio:g}"] = {
            "n": len(selected),
            "psnr": summary(
                [float(row["structure"]["psnr"]) for row in selected]
            ),
            "edge_correlation": summary(
                [
                    float(row["structure"]["edge_correlation"])
                    for row in selected
                ]
            ),
            "blend": summary([float(row["blend"]) for row in selected]),
            "mean_shift_norm": summary(
                [float(row["mean_shift_norm"]) for row in selected]
            ),
        }
    for strength in args.source_stat_strengths:
        selected = [
            row
            for row in records
            if row.get("source_stat_strength") == strength
        ]
        methods[f"sourcestats_s{strength:g}"] = {
            "n": len(selected),
            "psnr": summary(
                [float(row["structure"]["psnr"]) for row in selected]
            ),
            "edge_correlation": summary(
                [
                    float(row["structure"]["edge_correlation"])
                    for row in selected
                ]
            ),
        }
    payload = {
        "version": "huatuo-local-source-structure-audit-v1",
        "input": str(args.input),
        "image_root": str(args.image_root),
        "local_index": str(args.local_index),
        "n_requested": len(rows),
        "n_errors": len(errors),
        "methods": methods,
        "records": records,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "records"},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
