#!/usr/bin/env python3
"""Measure image-render x prompt interaction in Huatuo claim states.

The 2x2 design changes one DICOM rendering factor and one semantically
equivalent prompt factor.  Main effects are allowed; the mixed finite
difference measures whether the two nominal nuisance factors interact.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from anchor.corrected_sgta.run_huatuo_domain_orbit_head_probe_v1 import (
    bbox_patch_mask,
    score_view,
)
from corrected_sgta.run_huatuo_dicom_render_pilot_v1 import (
    balanced_rows,
    build_render_views,
    read_dicom_pixels,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    atomic_json,
    import_huatuo,
    label_ids,
    load_jsonl,
    prompt_for,
    resolve_image,
    sha256_file,
)


VERSION = "huatuo-factorial-interaction-probe-v1"
RENDERS = ("center_minus_0p05w", "center_plus_0p05w")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bboxes", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("pilot", "dev", "confirmation"), default="pilot")
    parser.add_argument(
        "--findings",
        nargs="+",
        default=["aortic_enlargement", "cardiomegaly", "pleural_effusion", "pulmonary_fibrosis"],
    )
    parser.add_argument("--per-bin", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 7, 14, 21, 27])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    return parser.parse_args()


def paraphrase(finding: str) -> str:
    readable = finding.replace("_", " ")
    return (
        f"Based only on this chest radiograph, is {readable} present? "
        "Reply with exactly one word: Yes, No, or Maybe."
    )


def summarize(cells: dict[str, dict], layers: tuple[int, ...]) -> dict:
    # cell names are r0p0, r0p1, r1p0, r1p1.
    polarity = {key: float(value["polarity"]) for key, value in cells.items()}
    image_main = 0.5 * ((polarity["r1p0"] - polarity["r0p0"]) + (polarity["r1p1"] - polarity["r0p1"]))
    prompt_main = 0.5 * ((polarity["r0p1"] - polarity["r0p0"]) + (polarity["r1p1"] - polarity["r1p0"]))
    interaction = polarity["r1p1"] - polarity["r1p0"] - polarity["r0p1"] + polarity["r0p0"]
    layer_rows = {}
    for layer in layers:
        key = str(layer)
        h00 = np.asarray(cells["r0p0"]["layers"][key]["head_output"], dtype=np.float64)
        h01 = np.asarray(cells["r0p1"]["layers"][key]["head_output"], dtype=np.float64)
        h10 = np.asarray(cells["r1p0"]["layers"][key]["head_output"], dtype=np.float64)
        h11 = np.asarray(cells["r1p1"]["layers"][key]["head_output"], dtype=np.float64)
        image = 0.5 * ((h10 - h00) + (h11 - h01))
        prompt = 0.5 * ((h01 - h00) + (h11 - h10))
        mixed = h11 - h10 - h01 + h00
        image_norm = np.linalg.norm(image, axis=1)
        prompt_norm = np.linalg.norm(prompt, axis=1)
        mixed_norm = np.linalg.norm(mixed, axis=1)
        layer_rows[key] = {
            "image_main_norm_by_head": image_norm.tolist(),
            "prompt_main_norm_by_head": prompt_norm.tolist(),
            "mixed_norm_by_head": mixed_norm.tolist(),
            "mean_image_main_norm": float(image_norm.mean()),
            "mean_prompt_main_norm": float(prompt_norm.mean()),
            "mean_mixed_norm": float(mixed_norm.mean()),
            "mixed_to_main_ratio": float(
                mixed_norm.mean() / (image_norm.mean() + prompt_norm.mean() + 1e-8)
            ),
        }
    base_prediction = cells["r0p0"]["prediction"]
    return {
        "base_prediction": base_prediction,
        "base_correct": base_prediction == "supported",
        "any_cell_prediction_flip": any(value["prediction"] != base_prediction for value in cells.values()),
        "polarity_image_main": image_main,
        "polarity_prompt_main": prompt_main,
        "polarity_mixed_interaction": interaction,
        "polarity_mixed_to_main_ratio": abs(interaction) / (abs(image_main) + abs(prompt_main) + 1e-8),
        "layers": layer_rows,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    layers = tuple(sorted(set(args.layers)))
    rows = balanced_rows(args.manifest, args.split, args.findings, [3], args.per_bin, args.seed)
    rows = rows[: args.max_cases]
    bbox_rows = load_jsonl(args.bboxes)
    boxes_by_claim = {
        (str(row["image_id"]), str(row["finding"])): list(row.get("boxes", [])) for row in bbox_rows
    }
    rows = [row for row in rows if boxes_by_claim.get((str(row["image_id"]), str(row["finding"])))]
    klass = import_huatuo(args.huatuo_root)
    bot = klass(str(args.model_dir), device=args.device)
    ids = label_ids(bot)
    records = []
    for index, row in enumerate(rows):
        image_id, finding = str(row["image_id"]), str(row["finding"])
        pixels = read_dicom_pixels(resolve_image(row, args.image_root))
        boxes = boxes_by_claim[(image_id, finding)]
        render_views = build_render_views(pixels, boxes, boxes)
        by_name = {str(view["name"]): view for view in render_views}
        patch_mask = bbox_patch_mask(
            576, boxes, int(pixels.modality.shape[0]), int(pixels.modality.shape[1]), bot.model.device
        )
        prompts = (prompt_for(finding), paraphrase(finding))
        cells = {}
        for render_index, render_name in enumerate(RENDERS):
            for prompt_index, prompt in enumerate(prompts):
                cells[f"r{render_index}p{prompt_index}"] = score_view(
                    bot, prompt, by_name[render_name]["image"], ids, layers, patch_mask
                )
        for view in render_views:
            view["image"].close()
        summary = summarize(cells, layers)
        records.append(
            {
                "image_id": image_id,
                "finding": finding,
                "positive_votes": int(row["positive_votes"]),
                "renders": list(RENDERS),
                "prompts": list(prompts),
                "cells": cells,
                "summary": summary,
            }
        )
        print(
            f"[{index + 1}/{len(rows)}] {finding} correct={summary['base_correct']} "
            f"mixed={summary['polarity_mixed_interaction']:.4f}",
            flush=True,
        )
    payload = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "retrospective 2x2 image-render by prompt interaction screen",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "bboxes": str(args.bboxes.resolve()),
        "bboxes_sha256": sha256_file(args.bboxes),
        "image_root": str(args.image_root.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "split": args.split,
        "findings": args.findings,
        "layers": list(layers),
        "seed": args.seed,
        "n": len(records),
        "records": records,
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
