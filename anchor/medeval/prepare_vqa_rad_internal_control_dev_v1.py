#!/usr/bin/env python3
"""Build an outcome-blind, image-disjoint VQA-RAD OE development set.

The public VQA-RAD train and test splits reuse images.  A row-level split is
therefore not a valid development substrate for choosing decoding controls.
This builder reads *only* image bytes from the official test parquet, removes
every train row whose image content occurs in test, and then selects genuinely
open train questions.  Test questions and answers are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from anchor.medeval.hashing import sha256_file
from anchor.medeval.prepare_vqa_rad_oe import is_open_answer, normalized_answer


VERSION = "vqa-rad-internal-control-dev-v1"
SOURCE_REPO = "flaviagiammarino/vqa-rad"
SOURCE_COMMIT = "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9"
EXPECTED_TRAIN_SHA256 = "b07c3441467b99060e5ec412ddd05be06f86f01f23bfa3debfbbcab47874a06e"
EXPECTED_TEST_SHA256 = "eb520bdab1116dd4f420120da19049d2315389fa126d031f65ec42e153264ea7"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def image_bytes(row: dict[str, Any], *, source_index: int) -> bytes:
    image = row.get("image")
    payload = image.get("bytes") if isinstance(image, dict) else None
    if not payload:
        raise ValueError(f"row {source_index} has no embedded image bytes")
    return bytes(payload)


def image_hashes(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {
        hashlib.sha256(image_bytes(row, source_index=index)).hexdigest()
        for index, row in enumerate(rows)
    }


def build_rows(
    train_rows: list[dict[str, Any]],
    test_image_hashes: set[str],
    image_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter train rows and materialize content-addressed development images."""

    output: list[dict[str, Any]] = []
    train_hashes: set[str] = set()
    overlapping_hashes: set[str] = set()
    eligible_hashes: set[str] = set()
    open_hashes: set[str] = set()
    seen_triplets: set[tuple[str, str, str]] = set()
    excluded = Counter()
    overlap_rows = 0
    eligible_rows = 0
    duplicate_triplets = 0
    open_questions_per_image: Counter[str] = Counter()

    for source_index, source in enumerate(train_rows):
        payload = image_bytes(source, source_index=source_index)
        digest = hashlib.sha256(payload).hexdigest()
        train_hashes.add(digest)
        if digest in test_image_hashes:
            overlap_rows += 1
            overlapping_hashes.add(digest)
            continue

        eligible_rows += 1
        eligible_hashes.add(digest)
        answer = str(source["answer"]).strip()
        if not is_open_answer(answer):
            excluded[normalized_answer(answer)] += 1
            continue

        question = str(source["question"]).strip()
        triplet = (digest, normalized_answer(question), normalized_answer(answer))
        if triplet in seen_triplets:
            duplicate_triplets += 1
            continue
        seen_triplets.add(triplet)

        image_name = f"{digest}.jpg"
        image_path = image_dir / image_name
        if image_path.exists():
            if sha256_file(image_path) != digest:
                raise RuntimeError(f"content-address collision: {image_path}")
        else:
            atomic_bytes(image_path, payload)
        open_hashes.add(digest)
        open_questions_per_image[digest] += 1
        qid = f"vqa-rad-train-{source_index:04d}"
        output.append(
            {
                "id": qid,
                "qid": qid,
                "img_name": image_name,
                "image_sha256": digest,
                "question": question,
                "answer": answer,
                "source": "vqa-rad",
                "dataset": "vqa_rad_internal_control_dev",
                "task": "open_vqa",
                "official_split": "train",
                "split_role": "development_only",
                "source_row": source_index,
            }
        )

    stats = {
        "train_unique_images": len(train_hashes),
        "test_unique_images": len(test_image_hashes),
        "overlap_unique_images": len(overlapping_hashes),
        "overlap_train_rows": overlap_rows,
        "eligible_rows_after_image_exclusion": eligible_rows,
        "eligible_unique_images_after_image_exclusion": len(eligible_hashes),
        "open_rows": len(output),
        "open_unique_images": len(open_hashes),
        "excluded_binary": dict(sorted(excluded.items())),
        "duplicate_open_triplets_removed": duplicate_triplets,
        "max_open_questions_per_image": max(open_questions_per_image.values(), default=0),
        "test_image_overlap_after_filter": len(open_hashes & test_image_hashes),
    }
    return output, stats


def prepare(
    *,
    train_parquet: Path,
    test_parquet: Path,
    image_dir: Path,
    output: Path,
    manifest_path: Path,
    expected_train_sha256: str | None = EXPECTED_TRAIN_SHA256,
    expected_test_sha256: str | None = EXPECTED_TEST_SHA256,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    train_sha = sha256_file(train_parquet)
    test_sha = sha256_file(test_parquet)
    if expected_train_sha256 and train_sha != expected_train_sha256:
        raise ValueError(f"unexpected train parquet SHA-256: {train_sha}")
    if expected_test_sha256 and test_sha != expected_test_sha256:
        raise ValueError(f"unexpected test parquet SHA-256: {test_sha}")

    train_table = pq.read_table(train_parquet, columns=["image", "question", "answer"])
    # Leakage guard: neither test question nor test answer is loaded.
    test_image_table = pq.read_table(test_parquet, columns=["image"])
    test_hashes = image_hashes(test_image_table.to_pylist())
    rows, stats = build_rows(train_table.to_pylist(), test_hashes, image_dir)
    if stats["test_image_overlap_after_filter"] != 0:
        raise AssertionError("development output overlaps official test images")

    atomic_json(output, rows)
    manifest = {
        "version": VERSION,
        "source_repository": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "license": "CC0-1.0 per the VQA-RAD dataset card",
        "selection_contract": {
            "development_source": "official train",
            "held_out_source": "official test",
            "test_columns_read": ["image"],
            "test_questions_read": False,
            "test_answers_read": False,
            "test_labels_used_for_selection": False,
            "image_exclusion": "exclude every train row whose image SHA-256 occurs anywhere in official test",
            "question_filter": "normalized development answer is neither yes nor no",
            "duplicate_filter": "first occurrence of normalized (image, question, answer) triplet",
            "use": "development-only selection and calibration; never report as test efficacy",
        },
        "sources": {
            "train_parquet": str(train_parquet.resolve()),
            "train_parquet_size": train_parquet.stat().st_size,
            "train_parquet_sha256": train_sha,
            "train_rows": train_table.num_rows,
            "test_parquet": str(test_parquet.resolve()),
            "test_parquet_size": test_parquet.stat().st_size,
            "test_parquet_sha256": test_sha,
            "test_rows": test_image_table.num_rows,
        },
        "counts": stats,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "image_dir": str(image_dir.resolve()),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--test-parquet", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-train-sha256", default=EXPECTED_TRAIN_SHA256)
    parser.add_argument("--expected-test-sha256", default=EXPECTED_TEST_SHA256)
    args = parser.parse_args()
    manifest = prepare(
        train_parquet=args.train_parquet,
        test_parquet=args.test_parquet,
        image_dir=args.image_dir,
        output=args.output,
        manifest_path=args.manifest,
        expected_train_sha256=args.expected_train_sha256,
        expected_test_sha256=args.expected_test_sha256,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
