#!/usr/bin/env python3
"""Prepare the official VQA-RAD test split's genuinely open questions.

The source parquet stores image bytes.  Images are content-addressed so that
multiple questions about the same image share one file and can be clustered
during statistical evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file


VERSION = "vqa-rad-official-test-oe-v1"


def normalized_answer(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def is_open_answer(text: str) -> bool:
    return normalized_answer(text) not in {"yes", "no"}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    table = pq.read_table(args.parquet)
    required = {"image", "question", "answer"}
    if not required.issubset(table.column_names):
        raise ValueError(f"missing columns: {required - set(table.column_names)}")
    args.image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    image_question_counts: Counter[str] = Counter()
    excluded = Counter()
    for source_index, source in enumerate(table.to_pylist()):
        answer = str(source["answer"]).strip()
        if not is_open_answer(answer):
            excluded[normalized_answer(answer)] += 1
            continue
        image = source["image"]
        image_bytes = image.get("bytes") if isinstance(image, dict) else None
        if not image_bytes:
            raise ValueError(f"row {source_index} has no embedded image bytes")
        digest = hashlib.sha256(image_bytes).hexdigest()
        image_name = f"{digest}.jpg"
        image_path = args.image_dir / image_name
        if image_path.exists():
            if sha256_file(image_path) != digest:
                raise RuntimeError(f"content-address collision: {image_path}")
        else:
            image_path.write_bytes(image_bytes)
        image_question_counts[digest] += 1
        qid = f"vqa-rad-test-{source_index:04d}"
        rows.append(
            {
                "id": qid,
                "qid": qid,
                "img_name": image_name,
                "image_sha256": digest,
                "question": str(source["question"]).strip(),
                "answer": answer,
                "source": "vqa-rad",
                "dataset": "vqa_rad_official_test_oe",
                "task": "open_vqa",
                "official_split": "test",
                "source_row": source_index,
            }
        )
    atomic_json(args.output, rows)
    manifest = {
        "version": VERSION,
        "source_parquet": str(args.parquet.resolve()),
        "source_parquet_sha256": sha256_file(args.parquet),
        "license": "CC0-1.0 per the VQA-RAD dataset card",
        "split": "official test",
        "filter": "normalized answer is neither yes nor no",
        "source_rows": table.num_rows,
        "open_rows": len(rows),
        "excluded_binary": dict(sorted(excluded.items())),
        "unique_images": len(image_question_counts),
        "max_questions_per_image": max(image_question_counts.values(), default=0),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "image_dir": str(args.image_dir.resolve()),
    }
    atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
