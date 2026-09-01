#!/usr/bin/env python3
"""Freeze PathVQA's complete official test shards with content-addressed images."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file


VERSION = "pathvqa-official-test-v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    shards = sorted(args.parquet_dir.glob("test-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no test shards under {args.parquet_dir}")
    args.image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    counts = Counter()
    seen_qids: set[str] = set()
    for shard_index, shard in enumerate(shards):
        table = pq.read_table(shard)
        required = {"image", "question", "answer"}
        if not required.issubset(table.column_names):
            raise ValueError(f"{shard} missing {required - set(table.column_names)}")
        for row_index, source in enumerate(table.to_pylist()):
            question = str(source.get("question", "")).strip()
            answer = str(source.get("answer", "")).strip()
            image = source.get("image")
            image_bytes = image.get("bytes") if isinstance(image, dict) else None
            if not question or not answer or not image_bytes:
                counts["invalid_source_row"] += 1
                continue
            digest = hashlib.sha256(image_bytes).hexdigest()
            source_path = str(image.get("path") or "") if isinstance(image, dict) else ""
            suffix = Path(source_path).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                suffix = ".jpg"
            image_path = args.image_dir / f"{digest}{suffix}"
            if image_path.exists():
                if sha256_file(image_path) != digest:
                    raise RuntimeError(f"content collision: {image_path}")
            else:
                image_path.write_bytes(image_bytes)
            qid = f"pathvqa-test-{shard_index:02d}-{row_index:06d}"
            if qid in seen_qids:
                raise RuntimeError(f"duplicate qid: {qid}")
            seen_qids.add(qid)
            binary = answer.lower() in {"yes", "no"}
            rows.append(
                {
                    "id": qid,
                    "qid": qid,
                    "question_id": qid,
                    "img_name": str(image_path.resolve()),
                    "image_sha256": digest,
                    "question": question,
                    "answer": answer,
                    "question_type": "binary" if binary else "open",
                    "source_question_type": "binary" if binary else "short_answer",
                    "task": "binary_ce" if binary else "open_vqa",
                    "dataset": "pathvqa",
                    "source": "PathVQA",
                    "official_split": "test",
                    "source_shard": shard.name,
                    "source_row": row_index,
                    "modality": "Pathology",
                }
            )
            counts["binary" if binary else "open"] += 1

    write_json(args.output, rows)
    manifest = {
        "version": VERSION,
        "source_shards": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in shards
        ],
        "rows": len(rows),
        "unique_images": len({row["image_sha256"] for row in rows}),
        "question_types": {key: counts[key] for key in ("binary", "open")},
        "invalid_source_rows": counts["invalid_source_row"],
        "split_contract": "all downloaded official test parquet shards in source order",
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "image_dir": str(args.image_dir.resolve()),
        "preparer": str(Path(__file__).resolve()),
        "preparer_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
