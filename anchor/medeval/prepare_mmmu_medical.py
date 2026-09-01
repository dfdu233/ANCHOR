#!/usr/bin/env python3
"""Freeze the five official MMMU medical validation subjects.

All original images are retained.  A labelled contact sheet is additionally
created for the five multi-image questions so single-image-only backends do
not silently drop visual evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from anchor.medeval.hashing import sha256_file


VERSION = "mmmu-medical-validation-v1"
MEDICAL_SUBJECTS = (
    "Basic_Medical_Science",
    "Clinical_Medicine",
    "Diagnostics_and_Laboratory_Medicine",
    "Pharmacy",
    "Public_Health",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def subject_from_id(qid: str) -> str:
    match = re.fullmatch(r"validation_(.+)_\d+", qid)
    if not match:
        raise ValueError(f"unexpected MMMU id: {qid}")
    return match.group(1)


def save_content_addressed(blob: bytes, output_dir: Path) -> Path:
    digest = hashlib.sha256(blob).hexdigest()
    path = output_dir / f"{digest}.png"
    if not path.exists():
        with Image.open(io.BytesIO(blob)) as image:
            image.convert("RGB").save(path, format="PNG")
    return path.resolve()


def make_contact_sheet(paths: list[Path], output_dir: Path) -> Path:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    label_height = 36
    canvas = Image.new("RGB", (width, sum(i.height + label_height for i in images)), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for index, image in enumerate(images, 1):
        draw.text((8, y + 8), f"Image {index}", fill="black")
        y += label_height
        canvas.paste(image, ((width - image.width) // 2, y))
        y += image.height
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    for image in images:
        image.close()
    return save_content_addressed(buffer.getvalue(), output_dir)


def main() -> None:
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    args.image_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.parquet)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for source in table.to_pylist():
        qid = str(source["id"])
        subject = subject_from_id(qid)
        if subject not in MEDICAL_SUBJECTS:
            continue
        blobs = [
            source[f"image_{index}"]["bytes"]
            for index in range(1, 8)
            if source.get(f"image_{index}") and source[f"image_{index}"].get("bytes")
        ]
        if not blobs:
            raise ValueError(f"medical question has no image: {qid}")
        image_paths = [save_content_addressed(blob, args.image_dir) for blob in blobs]
        fallback = image_paths[0] if len(image_paths) == 1 else make_contact_sheet(image_paths, args.image_dir)
        options = ast.literal_eval(source["options"])
        answer = str(source["answer"]).strip().upper()
        question_type = str(source["question_type"])
        if question_type == "multiple-choice" and answer not in [
            chr(ord("A") + i) for i in range(len(options))
        ]:
            raise ValueError(f"invalid answer {answer!r} for {qid}")
        question = str(source["question"]).strip()
        rows.append(
            {
                "id": qid,
                "qid": qid,
                "question_id": qid,
                "img_name": str(fallback),
                "image_paths": [str(path) for path in image_paths],
                "multi_image_fallback": "native" if len(image_paths) == 1 else "labelled_vertical_contact_sheet",
                "question": question,
                "choices": options,
                "answer": answer,
                "answer_idx": answer,
                "task": "multiple_choice" if question_type == "multiple-choice" else "open_vqa",
                "question_type": question_type,
                "dataset": "mmmu_medical",
                "source": "MMMU",
                "official_split": "validation",
                "subject": subject,
                "modality": "Mixed",
            }
        )
        counts[subject] += 1
        counts[f"type_{question_type}"] += 1
        counts[f"images_{len(blobs)}"] += 1

    expected = {subject: 30 for subject in MEDICAL_SUBJECTS}
    actual = {subject: counts[subject] for subject in MEDICAL_SUBJECTS}
    if actual != expected:
        raise RuntimeError(f"unexpected medical-subset counts: {actual}")
    write_json(args.output, rows)
    manifest = {
        "version": VERSION,
        "source": str(args.parquet.resolve()),
        "source_sha256": sha256_file(args.parquet),
        "subjects": actual,
        "rows": len(rows),
        "image_count_distribution": {
            key.removeprefix("images_"): value
            for key, value in sorted(counts.items())
            if key.startswith("images_")
        },
        "question_types": {
            key.removeprefix("type_"): value
            for key, value in sorted(counts.items())
            if key.startswith("type_")
        },
        "multi_image_policy": "retain image_paths; img_name is a labelled vertical contact-sheet fallback",
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "preparer": str(Path(__file__).resolve()),
        "preparer_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
