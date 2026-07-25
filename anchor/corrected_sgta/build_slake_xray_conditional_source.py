#!/usr/bin/env python3
"""Build a leak-audited SLAKE X-ray conditional source hidden-state bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.build_conditional_source_bank import (
    TARGET_CACHE,
    canonical_rgb_sha256,
    target_hashes,
)
from corrected_sgta.models import load_adapter
from corrected_sgta.protocol import file_sha256
from corrected_sgta.protocol_v2 import build_prompt


VERSION = "slake-xray-conditional-source-v1"
ImageFile.LOAD_TRUNCATED_IMAGES = True
SLAKE_ROOT = Path("/root/autodl-tmp/MedHEval/images/Slake")
MM_TARGET = Path(
    "/root/autodl-tmp/MedHEval/benchmark_data/"
    "Visual_Misinterpretation_Hallucination/close-ended/MM-VisHal.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="llava")
    parser.add_argument("--slake-root", type=Path, default=SLAKE_ROOT)
    parser.add_argument("--mm-target", type=Path, default=MM_TARGET)
    parser.add_argument("--target-cache", type=Path, default=TARGET_CACHE)
    parser.add_argument("--max-per-cell", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def source_group(row: dict) -> str | None:
    content = str(row.get("content_type") or "").lower()
    if content == "abnormality":
        return "presence"
    if content == "organ":
        return "attribute"
    if content == "modality":
        return "modality"
    return None


def select_rows(args: argparse.Namespace) -> list[dict]:
    mm_rows = json.loads(args.mm_target.read_text())
    excluded_names = {
        str(row["img_name"])
        for row in mm_rows
        if str(row.get("source", "")).lower() == "slake"
    }
    candidates = []
    for qa_path in sorted(args.slake_root.glob("*/question.json")):
        for row in json.loads(qa_path.read_text()):
            answer = str(row.get("answer", "")).strip().lower()
            image_name = str(row.get("img_name", ""))
            if (
                row.get("q_lang") != "en"
                or row.get("modality") != "X-Ray"
                or answer not in {"yes", "no"}
                or image_name in excluded_names
            ):
                continue
            group = source_group(row)
            if group is None:
                continue
            rank = hashlib.sha256(
                f"{args.seed}:{group}:{answer}:{image_name}:{row['question']}".encode()
            ).hexdigest()
            candidates.append(
                {
                    "source_id": image_name,
                    "image_path": args.slake_root / image_name,
                    "question": str(row["question"]).strip(),
                    "answer": answer,
                    "label": 0 if answer == "yes" else 1,
                    "question_group": group,
                    "content_type": row.get("content_type"),
                    "location": row.get("location"),
                    "modality": row.get("modality"),
                    "rank": rank,
                }
            )
    selected = []
    used: dict[tuple[str, int], set[str]] = {}
    for row in sorted(candidates, key=lambda item: item["rank"]):
        cell = (row["question_group"], row["label"])
        cell_images = used.setdefault(cell, set())
        if len(cell_images) >= args.max_per_cell or row["source_id"] in cell_images:
            continue
        cell_images.add(row["source_id"])
        selected.append(row)
    return selected


def main() -> None:
    args = parse_args()
    forbidden_hashes = target_hashes(args.target_cache)
    selected = select_rows(args)
    adapter = load_adapter(args.model)
    features = []
    metadata = []
    leaked = 0
    try:
        for row in tqdm(selected, desc="SLAKE X-ray source hidden states"):
            with Image.open(row["image_path"]) as opened:
                image = opened.convert("RGB")
            image_hash = canonical_rgb_sha256(image)
            if image_hash in forbidden_hashes:
                leaked += 1
                continue
            sample = {
                "question": row["question"],
                "question_type": "binary",
                "choices": "",
            }
            evidence = adapter.forward_ce(
                [image], build_prompt(sample), ("Yes", "No")
            )[0]
            features.append(evidence.features.astype(np.float32))
            metadata.append(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in row.items()
                    if key != "rank"
                }
                | {
                    "canonical_rgb_sha256": image_hash,
                    "surface_logits": evidence.logits.tolist(),
                }
            )
    finally:
        adapter.close()
    if not features:
        raise RuntimeError("no leak-free SLAKE source rows extracted")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.stack(features),
        labels=np.asarray([row["label"] for row in metadata], dtype=np.int64),
        question_groups=np.asarray([row["question_group"] for row in metadata]),
    )
    meta = {
        "version": VERSION,
        "model": args.model,
        "slake_root": str(args.slake_root.resolve()),
        "mm_target": str(args.mm_target.resolve()),
        "mm_target_sha256": file_sha256(args.mm_target),
        "target_cache": str(args.target_cache.resolve()),
        "target_cache_sha256": file_sha256(args.target_cache),
        "selection": {
            "seed": args.seed,
            "modality": "X-Ray",
            "language": "English",
            "answers": ["Yes", "No"],
            "excluded_all_mm_vishal_slake_images_by_name": True,
            "max_per_question_group_answer": args.max_per_cell,
            "one_question_per_source_image_per_cell": True,
        },
        "canonical_cxr_target_hash_count": len(forbidden_hashes),
        "excluded_exact_cxr_target_images": leaked,
        "n": len(metadata),
        "cell_counts": {
            f"{group}:{answer}": sum(
                row["question_group"] == group and row["answer"] == answer
                for row in metadata
            )
            for group in sorted({row["question_group"] for row in metadata})
            for answer in ("yes", "no")
        },
        "rows": metadata,
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(
        json.dumps(
            {
                key: meta[key]
                for key in ("n", "cell_counts", "excluded_exact_cxr_target_images")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
