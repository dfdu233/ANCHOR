#!/usr/bin/env python3
"""Freeze the official PMC-VQA v2 test CSV and referenced archive images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from anchor.medeval.hashing import sha256_file


VERSION = "pmcvqa-official-v2-test-v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def clean_choice(value: str, letter: str) -> str:
    return re.sub(rf"^\s*{letter}\s*:\s*", "", value, flags=re.IGNORECASE).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--images-zip", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    args.image_dir.mkdir(parents=True, exist_ok=True)

    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        sources = list(csv.DictReader(handle))
    with zipfile.ZipFile(args.images_zip) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member in PMC-VQA archive: {bad}")
        members = {
            PurePosixPath(info.filename).name: info
            for info in archive.infolist()
            if not info.is_dir()
        }
        rows: list[dict[str, Any]] = []
        seen: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        for source_row, source in enumerate(sources):
            if str(source.get("split", "")).strip().lower() != "test":
                counts["outside_test"] += 1
                continue
            filename = PurePosixPath(str(source["Figure_path"]).strip()).name
            info = members.get(filename)
            if info is None:
                counts["missing_image"] += 1
                continue
            blob = archive.read(info)
            digest = hashlib.sha256(blob).hexdigest()
            suffix = Path(filename).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                suffix = ".jpg"
            image_path = args.image_dir / f"{digest}{suffix}"
            if not image_path.exists():
                image_path.write_bytes(blob)
            base_qid = f"pmcvqa-v2-test-{str(source['index']).strip()}"
            seen[base_qid] += 1
            qid = base_qid if seen[base_qid] == 1 else f"{base_qid}-{seen[base_qid]}"
            choices = [clean_choice(source[f"Choice {letter}"], letter) for letter in "ABCD"]
            answer = str(source["Answer"]).strip().upper()
            if answer not in "ABCD":
                raise ValueError(f"invalid answer {answer!r} at CSV row {source_row}")
            rows.append(
                {
                    "id": qid,
                    "qid": qid,
                    "question_id": qid,
                    "img_name": str(image_path.resolve()),
                    "image_sha256": digest,
                    "question": str(source["Question"]).strip(),
                    "choices": choices,
                    "answer": answer,
                    "answer_idx": answer,
                    "task": "multiple_choice",
                    "question_type": "multiple-choice",
                    "dataset": "pmcvqa",
                    "source": "PMC-VQA",
                    "official_split": "test",
                    "source_row": source_row,
                    "source_index": str(source["index"]).strip(),
                    "source_figure": filename,
                    "modality": "Mixed",
                }
            )

    if not rows:
        raise RuntimeError("PMC-VQA test preparation produced no evaluable rows")
    write_json(args.output, rows)
    manifest = {
        "version": VERSION,
        "source_csv": str(args.csv.resolve()),
        "source_csv_sha256": sha256_file(args.csv),
        "source_images_zip": str(args.images_zip.resolve()),
        "source_images_zip_sha256": sha256_file(args.images_zip),
        "source_rows": len(sources),
        "rows": len(rows),
        "unique_images": len({row["image_sha256"] for row in rows}),
        "missing_image_rows": counts["missing_image"],
        "outside_test_rows": counts["outside_test"],
        "duplicate_source_indices": sum(value - 1 for value in seen.values()),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "preparer": str(Path(__file__).resolve()),
        "preparer_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
