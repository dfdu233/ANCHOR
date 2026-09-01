#!/usr/bin/env python3
"""Focal VinDr dual-counterfactual pilot for C3-Guard.

For reader-supported focal findings, compare erasing the union of radiologist
boxes with erasing its horizontal mirror.  Both interventions use the same
blur-and-feather operator, so the paired contrast tests region specificity
rather than generic masking sensitivity.  This is an evidence-erasure pilot,
not a claim that blurring creates a clinically normal counterfactual image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    dicom_to_pil,
    hidden_trajectory,
    import_huatuo,
    label_ids,
    layer_logits,
    prepared_embeddings,
)


VERSION = "vindr-focal-evidence-erasure-v2"
FINDINGS = ("Nodule/Mass", "Calcification")


def claim_prompt(finding: str) -> str:
    phrase = {
        "Nodule/Mass": "a pulmonary nodule or mass",
        "Calcification": "thoracic calcification",
    }[finding]
    return (
        f"Does this chest X-ray show {phrase}? "
        "Answer with exactly one word: Yes, No, or Maybe."
    )


def stable(seed: int, image_id: str, finding: str) -> str:
    return hashlib.sha256(f"{seed}:{image_id}:{finding}".encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mask_for(rows: list[dict], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for row in rows:
        x0, y0, x1, y1 = [float(row[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
        pad_x = max(2, int(round((x1 - x0) * 0.05)))
        pad_y = max(2, int(round((y1 - y0) * 0.05)))
        xa, xb = max(0, int(np.floor(x0)) - pad_x), min(width, int(np.ceil(x1)) + pad_x)
        ya, yb = max(0, int(np.floor(y0)) - pad_y), min(height, int(np.ceil(y1)) + pad_y)
        mask[ya:yb, xa:xb] = 255
    return mask


def select_cases(csv_path: Path, dicom_root: Path, per_finding: int, seed: int) -> list[dict]:
    import pydicom

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["class_name"] in FINDINGS and row["x_min"]:
                grouped[(row["image_id"], row["class_name"])].append(row)
    output = []
    for finding in FINDINGS:
        candidates = sorted(
            (
                (image_id, rows)
                for (image_id, name), rows in grouped.items()
                if name == finding and len({row["rad_id"] for row in rows}) >= 2
            ),
            key=lambda item: stable(seed, item[0], finding),
        )
        accepted = 0
        for image_id, rows in candidates:
            path = dicom_root / f"{image_id}.dicom"
            if not path.is_file():
                continue
            header = pydicom.dcmread(str(path), stop_before_pixels=True)
            height, width = int(header.Rows), int(header.Columns)
            mask = mask_for(rows, width, height)
            mirror = mask[:, ::-1].copy()
            area = float((mask > 0).mean())
            intersection = int(np.logical_and(mask > 0, mirror > 0).sum())
            union = int(np.logical_or(mask > 0, mirror > 0).sum())
            iou = intersection / union if union else 1.0
            if not 0.0005 <= area <= 0.15 or iou > 0.10:
                continue
            output.append(
                {
                    "image_id": image_id,
                    "finding": finding,
                    "dicom": str(path.resolve()),
                    "reader_votes": len({row["rad_id"] for row in rows}),
                    "reader_ids": sorted({row["rad_id"] for row in rows}),
                    "boxes": [
                        {k: row[k] for k in ("rad_id", "x_min", "y_min", "x_max", "y_max")}
                        for row in rows
                    ],
                    "height": height,
                    "width": width,
                    "mask_area_fraction": area,
                    "lesion_mirror_iou": iou,
                }
            )
            accepted += 1
            if accepted == per_finding:
                break
        if accepted < per_finding:
            raise RuntimeError(f"{finding}: only {accepted} eligible cases")
    return output


def erase_region(image: Image.Image, mask: np.ndarray) -> Image.Image:
    radius = max(8.0, min(image.size) * 0.03)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    feather = Image.fromarray(mask, mode="L").filter(
        ImageFilter.GaussianBlur(radius=max(2.0, radius * 0.15))
    )
    return Image.composite(blurred, image, feather)


@torch.inference_mode()
def score(bot, ids: dict[str, int], finding: str, image: Image.Image) -> dict:
    tensor = torch.stack(bot.get_image_tensors([image])).to(
        device=bot.model.device, dtype=torch.bfloat16
    )
    embeddings, attention, positions, _ = prepared_embeddings(bot, claim_prompt(finding), tensor)
    hidden = hidden_trajectory(bot, embeddings, attention, positions)
    values = layer_logits(bot, hidden, (), ids)[len(hidden) - 1]
    logits = np.asarray([values["supported"], values["refuted"], values["undetermined"]])
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return {
        "yes": values["supported"],
        "no": values["refuted"],
        "maybe": values["undetermined"],
        "yes_minus_no": values["supported"] - values["refuted"],
        "probabilities": {
            "yes": float(probabilities[0]),
            "no": float(probabilities[1]),
            "maybe": float(probabilities[2]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("/workspace/vinbigdata/train.csv"))
    parser.add_argument("--dicom-root", type=Path, default=Path("/workspace/vinbigdata/train"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-finding", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    cases = select_cases(args.csv, args.dicom_root, args.per_finding, args.seed)
    config = {
        "version": VERSION,
        "created_at": now(),
        "selection": "stable hash; >=2 independent VinDr readers; focal area 0.05%-15%; lesion-vs-mirror IoU <=0.10",
        "findings": list(FINDINGS),
        "per_finding": args.per_finding,
        "n": len(cases),
        "intervention": "Gaussian local-evidence erasure with feathered mask; paired horizontal-mirror same-shape control",
        "prompts": {finding: claim_prompt(finding) for finding in FINDINGS},
        "interpretation_boundary": "not a clinically normal counterfactual and not a hallucination mitigation result",
        "cases": cases,
    }
    config_path = args.output_dir / "config.json"
    if config_path.exists() and args.resume:
        old = json.loads(config_path.read_text())
        if old["cases"] != cases or old["per_finding"] != args.per_finding:
            raise RuntimeError("resume selection drift")
    elif config_path.exists():
        raise FileExistsError("output exists; use --resume")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    done = set()
    if raw_path.exists() and args.resume:
        done = {json.loads(line)["image_id"] + "|" + json.loads(line)["finding"] for line in raw_path.read_text().splitlines() if line.strip()}

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    ids = label_ids(bot)
    for index, case in enumerate(cases):
        key = case["image_id"] + "|" + case["finding"]
        if key in done:
            continue
        record = {**case, "version": VERSION, "status": "error"}
        try:
            original = dicom_to_pil(Path(case["dicom"]))
            mask = mask_for(case["boxes"], case["width"], case["height"])
            lesion_erased = erase_region(original, mask)
            mirror_erased = erase_region(original, mask[:, ::-1].copy())
            record.update(
                {
                    "status": "ok",
                    "scores": {
                        "original": score(bot, ids, case["finding"], original),
                        "lesion_erased": score(bot, ids, case["finding"], lesion_erased),
                        "mirror_erased": score(bot, ids, case["finding"], mirror_erased),
                    },
                    "completed_at": now(),
                }
            )
        except Exception as error:
            record.update({"error": repr(error), "traceback": traceback.format_exc(), "completed_at": now()})
        with raw_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(f"[{index + 1}/{len(cases)}] {key} {record['status']}", flush=True)


if __name__ == "__main__":
    main()
