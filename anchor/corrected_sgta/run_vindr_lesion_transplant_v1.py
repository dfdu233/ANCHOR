#!/usr/bin/env python3
"""Patient-internal lesion deletion/insertion pilot using VinDr consensus boxes."""

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
import torch
from PIL import Image, ImageFilter

from corrected_sgta.run_huatuo_vindr_commitment_probe import dicom_to_pil, import_huatuo, label_ids, sha256_file
from corrected_sgta.run_hulu_vindr_commitment_probe import model_file_inventory
from corrected_sgta.run_vindr_focal_evidence_erasure_v1 import claim_prompt, mask_for, score


VERSION = "vindr-lesion-transplant-v2"
FINDINGS = ("Nodule/Mass",)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def box_iou(left: dict, right: dict) -> float:
    a = [float(left[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
    b = [float(right[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def one_connected_consensus(rows: list[dict]) -> bool:
    if not rows:
        return False
    seen = {0}
    stack = [0]
    while stack:
        current = stack.pop()
        for other in range(len(rows)):
            if other not in seen and box_iou(rows[current], rows[other]) > 0.10:
                seen.add(other)
                stack.append(other)
    return len(seen) == len(rows) and len({row["rad_id"] for row in rows}) >= 2


def stable(seed: int, image_id: str, finding: str) -> str:
    return hashlib.sha256(f"{seed}:{image_id}:{finding}:transplant".encode()).hexdigest()


def select_cases(csv_path: Path, dicom_root: Path, per_finding: int, seed: int) -> list[dict]:
    import pydicom

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["class_name"] in FINDINGS and row["x_min"]:
                grouped[(row["image_id"], row["class_name"])].append(row)
    selected = []
    for finding in FINDINGS:
        candidates = sorted(
            ((iid, rows) for (iid, name), rows in grouped.items() if name == finding and one_connected_consensus(rows)),
            key=lambda item: stable(seed, item[0], finding),
        )
        count = 0
        for image_id, rows in candidates:
            path = dicom_root / f"{image_id}.dicom"
            if not path.is_file():
                continue
            header = pydicom.dcmread(str(path), stop_before_pixels=True)
            height, width = int(header.Rows), int(header.Columns)
            mask = mask_for(rows, width, height)
            mirror_mask = mask[:, ::-1].copy()
            area = float((mask > 0).mean())
            union = np.logical_or(mask > 0, mirror_mask > 0).sum()
            overlap = float(np.logical_and(mask > 0, mirror_mask > 0).sum() / union) if union else 1.0
            # Small, unilateral, single-consensus-cluster findings only.
            if not 0.0003 <= area <= 0.03 or overlap > 0.02:
                continue
            selected.append(
                {
                    "image_id": image_id,
                    "finding": finding,
                    "dicom": str(path.resolve()),
                    "reader_votes": len({row["rad_id"] for row in rows}),
                    "boxes": [{k: row[k] for k in ("rad_id", "x_min", "y_min", "x_max", "y_max")} for row in rows],
                    "height": height,
                    "width": width,
                    "mask_area_fraction": area,
                    "lesion_mirror_iou": overlap,
                }
            )
            count += 1
            if count == per_finding:
                break
        if count < per_finding:
            raise RuntimeError(f"{finding}: only {count} eligible cases")
    return selected


def transplant_variants(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, Image.Image]:
    reflected = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    radius = max(2.0, min(image.size) * 0.004)
    target_alpha = Image.fromarray(mask, mode="L").filter(ImageFilter.GaussianBlur(radius=radius))
    mirror_alpha = Image.fromarray(mask[:, ::-1].copy(), mode="L").filter(ImageFilter.GaussianBlur(radius=radius))
    deletion = Image.composite(reflected, image, target_alpha)
    # Move, rather than duplicate, the lesion: start from deletion, then paste
    # the reflected original lesion into the contralateral homologous region.
    relocation = Image.composite(reflected, deletion, mirror_alpha)
    return deletion, relocation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("/workspace/vinbigdata/train.csv"))
    parser.add_argument("--dicom-root", type=Path, default=Path("/workspace/vinbigdata/train"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-finding", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cases = select_cases(args.csv, args.dicom_root, args.per_finding, args.seed)
    static = {
        "version": VERSION,
        "n": len(cases),
        "per_finding": args.per_finding,
        "seed": args.seed,
        "csv": str(args.csv.resolve()),
        "csv_sha256": sha256_file(args.csv),
        "dicom_root": str(args.dicom_root.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "model_inventory": model_file_inventory(args.model_dir),
        "huatuo_root": str(args.huatuo_root.resolve()),
        "code_sha256": sha256_file(Path(__file__)),
        "selection": "single connected consensus cluster; >=2 readers; 0.03%-3% area; lesion/mirror IoU <=2%",
        "counterfactuals": {
            "deletion": "replace consensus lesion region with horizontally reflected contralateral tissue",
            "relocation": "after deletion, paste the same reflected lesion appearance into the contralateral homologous region",
        },
        "success_law": "score(original)>score(deletion) and score(relocation)>score(deletion)",
        "boundary": "within-image appearance transplant; not guaranteed clinically normal or pathology-preserving outside the target",
        "prompts": {finding: claim_prompt(finding) for finding in FINDINGS},
        "cases": cases,
    }
    fingerprint = canonical_hash(static)
    config = {
        "created_at": now(),
        "command": " ".join(sys.argv),
        "fingerprint": fingerprint,
        "static": static,
    }
    config_path = args.output_dir / "config.json"
    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError(args.output_dir)
        prior = json.loads(config_path.read_text())
        if prior.get("fingerprint") != fingerprint or prior.get("static") != static:
            raise ValueError("refusing resume after configuration drift")
        config = prior
    else:
        args.output_dir.mkdir(parents=True)
        atomic_json(config_path, config)
    raw = args.output_dir / "raw.jsonl"
    completed = set()
    if raw.is_file():
        for line in raw.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                completed.add(str(row["image_id"]))
    if len(completed) == len(cases):
        print(json.dumps({"status": "already_complete", "n": len(cases)}))
        return
    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    ids = label_ids(bot)
    for index, case in enumerate(cases):
        if case["image_id"] in completed:
            print(f"[{index + 1}/{len(cases)}] {case['finding']} {case['image_id']} cached", flush=True)
            continue
        image = dicom_to_pil(Path(case["dicom"]))
        mask = mask_for(case["boxes"], case["width"], case["height"])
        deletion, relocation = transplant_variants(image, mask)
        record = {
            **case,
            "version": VERSION,
            "status": "ok",
            "scores": {
                "original": score(bot, ids, case["finding"], image),
                "deletion": score(bot, ids, case["finding"], deletion),
                "relocation": score(bot, ids, case["finding"], relocation),
            },
            "completed_at": now(),
        }
        with raw.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"[{index + 1}/{len(cases)}] {case['finding']} {case['image_id']}", flush=True)


if __name__ == "__main__":
    main()
