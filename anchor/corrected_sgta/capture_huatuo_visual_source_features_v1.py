#!/usr/bin/env python3
"""Capture visual-only Huatuo pre-projector features for a frozen JSONL cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


VERSION = "huatuo-visual-source-features-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(exist_ok=True)
    sys.path.insert(0, str(args.huatuo_root))
    from cli import HuatuoChatbot  # type: ignore

    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    vision_tower = bot.model.get_model().get_vision_tower()
    records = []
    for index, row in enumerate(rows):
        image = (args.image_root / row["image"]).resolve()
        expected = row.get("image_sha256")
        actual = sha256_file(image)
        if expected and actual != expected:
            raise RuntimeError(f"frozen image hash mismatch: {image}")
        target = feature_dir / f"{row['question_id']}.npz"
        if target.is_file():
            with np.load(target) as data:
                if data["visual_pre"].shape != (1024,):
                    raise RuntimeError(f"invalid resumed feature: {target}")
        else:
            image_tensor = torch.stack(bot.get_image_tensors([str(image)])).to(
                device=bot.model.device, dtype=torch.bfloat16
            )
            with torch.inference_mode():
                pre = vision_tower(image_tensor)
            vector = pre.float().mean(dim=(0, 1)).cpu().numpy().astype(np.float32)
            np.savez_compressed(target, visual_pre=vector)
        records.append(
            {
                "question_id": str(row["question_id"]),
                "patient_id": row["patient_id"],
                "image": str(image),
                "image_sha256": actual,
                "feature_file": str(target.resolve()),
            }
        )
        print(f"[{index + 1}/{len(rows)}] {row['question_id']}", flush=True)
    payload = {
        "version": VERSION,
        "manifest": str(args.manifest.resolve()),
        "n": len(records),
        "semantics": "global mean of model-native selected-layer patch tokens immediately before mm_projector",
        "model": str(args.model_dir.resolve()),
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
