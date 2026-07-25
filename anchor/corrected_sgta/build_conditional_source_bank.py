#!/usr/bin/env python3
"""Build leak-audited question/answer-conditioned source hidden states."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import iter_successes
from corrected_sgta.models import load_adapter
from corrected_sgta.protocol import file_sha256, resolve_image
from corrected_sgta.protocol_v2 import build_prompt


VERSION = "conditional-source-bank-v1"
ImageFile.LOAD_TRUNCATED_IMAGES = True
TARGET_DATASET = Path(
    "/root/autodl-tmp/MedHEval/benchmark_data/"
    "Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json"
)
TARGET_CACHE = Path(
    "corrected_runs/confgen_visualdep_full_v1/"
    "llava_cxr_visualdep.full_v52.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="llava")
    parser.add_argument("--max-per-cell", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-dataset", type=Path, default=TARGET_DATASET)
    parser.add_argument("--target-cache", type=Path, default=TARGET_CACHE)
    return parser.parse_args()


def canonical_rgb_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    payload = (
        f"{rgb.width}x{rgb.height}:RGB:".encode()
        + np.asarray(rgb, dtype=np.uint8).tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def question_group(question: str) -> str:
    text = " ".join(str(question).lower().split())
    if any(token in text for token in ("modality", "x-ray", "xray", " mri", " ct ", "ultrasound")):
        return "modality"
    if any(token in text for token in ("where", "location", "located", "which side", "left or right")):
        return "location"
    if any(token in text for token in ("how many", "number of", "count")):
        return "count"
    if any(token in text for token in ("size", "shape", "color", "appearance", "plane", "view")):
        return "attribute"
    return "presence"


def target_hashes(cache_path: Path) -> set[str]:
    cache_meta = json.loads(
        cache_path.with_suffix(cache_path.suffix + ".meta.json").read_text()
    )
    rows = list(iter_successes(cache_path, cache_meta["fingerprint"]))
    hashes: set[str] = set()
    seen: set[str] = set()
    for row in tqdm(rows, desc="target leak hashes"):
        image_name = str(row["img_name"])
        if image_name in seen:
            continue
        seen.add(image_name)
        path = resolve_image(image_name)
        if path is None:
            raise FileNotFoundError(f"target image missing during leak audit: {image_name}")
        with Image.open(path) as image:
            hashes.add(canonical_rgb_sha256(image))
    return hashes


def select_rows(frame: pd.DataFrame, limit: int, seed: int) -> list[dict]:
    candidates: list[dict] = []
    for index, row in frame.iterrows():
        answer = str(row["answer"]).strip().lower()
        if answer not in {"yes", "no"}:
            continue
        image_record = row["image"]
        image_bytes = image_record["bytes"]
        source_id = str(image_record.get("path") or hashlib.sha256(image_bytes).hexdigest())
        question = str(row["question"]).strip()
        group = question_group(question)
        rank = hashlib.sha256(
            f"{seed}:{group}:{answer}:{source_id}:{question}".encode()
        ).hexdigest()
        candidates.append(
            {
                "source_index": int(index),
                "source_id": source_id,
                "question": question,
                "answer": answer,
                "label": 0 if answer == "yes" else 1,
                "question_group": group,
                "rank": rank,
                "image_bytes": image_bytes,
            }
        )
    selected: list[dict] = []
    used_per_cell: dict[tuple[str, int], set[str]] = {}
    for row in sorted(candidates, key=lambda item: item["rank"]):
        cell = (row["question_group"], row["label"])
        used = used_per_cell.setdefault(cell, set())
        if len(used) >= limit or row["source_id"] in used:
            continue
        used.add(row["source_id"])
        selected.append(row)
    return selected


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.parquet)
    selected = select_rows(frame, args.max_per_cell, args.seed)
    forbidden_hashes = target_hashes(args.target_cache)
    adapter = load_adapter(args.model)
    features: list[np.ndarray] = []
    metadata: list[dict] = []
    leaked = 0
    try:
        for row in tqdm(selected, desc="source hidden states"):
            with Image.open(io.BytesIO(row.pop("image_bytes"))) as opened:
                image = opened.convert("RGB")
            image_hash = canonical_rgb_sha256(image)
            if image_hash in forbidden_hashes:
                leaked += 1
                continue
            sample = {"question": row["question"], "question_type": "binary", "choices": ""}
            evidence = adapter.forward_ce(
                [image], build_prompt(sample), ("Yes", "No")
            )[0]
            features.append(evidence.features.astype(np.float32))
            metadata.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"rank"}
                }
                | {
                    "canonical_rgb_sha256": image_hash,
                    "surface_logits": evidence.logits.tolist(),
                }
            )
    finally:
        adapter.close()
    if not features:
        raise RuntimeError("no leak-free source rows were extracted")
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
        "parquet": str(args.parquet.resolve()),
        "parquet_sha256": file_sha256(args.parquet),
        "target_dataset": str(args.target_dataset.resolve()),
        "target_dataset_sha256": file_sha256(args.target_dataset),
        "target_cache": str(args.target_cache.resolve()),
        "target_cache_sha256": file_sha256(args.target_cache),
        "selection": {
            "seed": args.seed,
            "max_per_question_group_answer": args.max_per_cell,
            "one_question_per_source_image_per_cell": True,
        },
        "feature_space": "last multimodal prompt hidden state",
        "prompt": "<question> Please answer Yes or No.",
        "canonical_target_hash_count": len(forbidden_hashes),
        "excluded_exact_target_images": leaked,
        "n": len(metadata),
        "cell_counts": {
            f"{group}:{label}": sum(
                row["question_group"] == group and row["answer"] == label
                for row in metadata
            )
            for group in sorted({row["question_group"] for row in metadata})
            for label in ("yes", "no")
        },
        "rows": metadata,
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(json.dumps({key: meta[key] for key in ("n", "cell_counts", "excluded_exact_target_images")}, indent=2))


if __name__ == "__main__":
    main()
