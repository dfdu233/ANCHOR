#!/usr/bin/env python3
"""Prepare the unverified HF CheXpert subset for report evaluation.

The upstream dataset exposes only embedded images and free-text reports.  This
converter never derives disease labels and deliberately marks the dataset as
ineligible for unknown-institution claims until its provenance is verified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

from corrected_sgta.rule_source_preference import file_sha256, stable_json_sha256


VERSION = "chexpert-subset-report-adapter-v1"
DATASET_ID = "ayyuce/chexpert-subset"
DATASET_NAME = "chexpert_subset_unverified"
DEFAULT_REVISION = "372166fb5f5004176fd0642f2290574958034629"
REPORT_PROMPT = (
    "You are a professional radiologist. You are provided with a chest X-ray "
    "image. Please generate a report based on the image. Please only include "
    "the content of the report in your response."
)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def stable_key(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def image_extension(image_bytes: bytes) -> str:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.verify()
        fmt = str(image.format or "").upper()
    extensions = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
    if fmt not in extensions:
        raise ValueError(f"unsupported embedded image format: {fmt!r}")
    return extensions[fmt]


def load_records(parquet: Path, seed: int, max_samples: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    table = pq.read_table(parquet, columns=["image", "report"])
    if table.column_names != ["image", "report"]:
        raise ValueError(f"unexpected parquet columns: {table.column_names}")
    records: list[dict[str, Any]] = []
    image_hashes: set[str] = set()
    empty_reports = 0
    for source_index, row in enumerate(table.to_pylist()):
        report = str(row.get("report") or "").strip()
        if not report:
            empty_reports += 1
            continue
        image = row.get("image")
        image_bytes = image.get("bytes") if isinstance(image, dict) else None
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError(f"row {source_index} has no embedded image bytes")
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        if image_sha256 in image_hashes:
            raise ValueError(
                "duplicate image bytes require an explicit multi-reference policy: "
                f"{image_sha256}"
            )
        image_hashes.add(image_sha256)
        report_sha256 = hashlib.sha256(report.encode()).hexdigest()
        identifier = hashlib.sha256(
            f"{image_sha256}:{report_sha256}".encode()
        ).hexdigest()
        records.append(
            {
                "id": identifier,
                "source_index": source_index,
                "image_sha256": image_sha256,
                "report_sha256": report_sha256,
                "extension": image_extension(image_bytes),
                "image_bytes": image_bytes,
                "report": report,
            }
        )
    records.sort(key=lambda row: stable_key(seed, row["id"]))
    if max_samples:
        records = records[:max_samples]
    audit = {
        "parquet_rows": table.num_rows,
        "empty_reports": empty_reports,
        "unique_valid_images": len(image_hashes),
        "selected_rows": len(records),
    }
    return records, audit


def write_image(path: Path, image_bytes: bytes, expected_sha256: str) -> None:
    if path.exists():
        if file_sha256(path) != expected_sha256:
            raise RuntimeError(f"existing extracted image hash mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(image_bytes)
    if file_sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"extracted image hash mismatch: {path}")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ayyuce/chexpert-subset and emit fingerprinted MMed-RAG, "
            "RULE-shaped report, and normalized ANCHOR report manifests."
        )
    )
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--dataset-readme", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if not args.parquet.is_file() or not args.dataset_readme.is_file():
        raise FileNotFoundError("parquet and dataset README must both exist")

    output_files = {
        "mmedrag": args.output_dir / "mmedrag_report.json",
        "rule": args.output_dir / "rule_report.jsonl",
        "normalized": args.output_dir / "anchor_report_manifest.json",
        "provenance": args.output_dir / "provenance.json",
    }
    existing = [str(path) for path in output_files.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite derived manifests: {existing}")

    records, audit = load_records(args.parquet, args.seed, args.max_samples)
    if not records:
        raise RuntimeError("no valid image-report records selected")
    image_root = args.output_dir / "images"
    for row in records:
        filename = row["image_sha256"] + row["extension"]
        write_image(image_root / filename, row["image_bytes"], row["image_sha256"])
        row["filename"] = filename

    provenance_core = {
        "version": VERSION,
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_revision": args.revision,
        "parquet_sha256": file_sha256(args.parquet),
        "dataset_readme_sha256": file_sha256(args.dataset_readme),
        "code_sha256": file_sha256(Path(__file__)),
        "seed": args.seed,
        "max_samples": args.max_samples,
        "selection": "sha256(seed, sha256(image_bytes):sha256(stripped_report))",
        "audit": audit,
        "provenance_verified": False,
        "patient_split_verified": False,
        "institution_verified": False,
        "eligible_as_unknown_institution": False,
        "official_disease_labels_available": False,
        "derived_vqa_labels_created": False,
        "reference_used_for_generation_or_selection": False,
    }
    fingerprint = stable_json_sha256(provenance_core)

    mmedrag_rows = []
    rule_rows = []
    normalized_rows = []
    for row in records:
        common = {
            "id": row["id"],
            "report": row["report"],
            "image_sha256": row["image_sha256"],
            "source_index": row["source_index"],
            "dataset": DATASET_NAME,
            "dataset_fingerprint": fingerprint,
            "provenance_verified": False,
        }
        mmedrag_rows.append(
            {
                **common,
                "image_path": [row["filename"]],
                "split": "test",
            }
        )
        rule_rows.append(
            {
                "question_id": row["id"],
                "question": REPORT_PROMPT + "\n<image>",
                "answer": row["report"],
                "image": row["filename"],
                "report": row["report"],
                "task": "report_generation",
                "dataset": DATASET_NAME,
                "dataset_fingerprint": fingerprint,
                "official_rule_binary_compatible": False,
                "provenance_verified": False,
            }
        )
        normalized_rows.append(
            {
                "id": row["id"],
                "image": str((image_root / row["filename"]).resolve()),
                "prompt": REPORT_PROMPT,
                "answer": row["report"],
                "domain": DATASET_NAME,
                "patient_id": row["id"],
                "patient_id_is_surrogate": True,
                "dataset_fingerprint": fingerprint,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(output_files["mmedrag"], json.dumps(mmedrag_rows, indent=2) + "\n")
    atomic_text(
        output_files["rule"],
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rule_rows),
    )
    normalized_payload = {
        **provenance_core,
        "fingerprint": fingerprint,
        "image_root": str(image_root.resolve()),
        "records": normalized_rows,
    }
    atomic_text(
        output_files["normalized"],
        json.dumps(normalized_payload, indent=2, ensure_ascii=False) + "\n",
    )
    provenance = {
        **provenance_core,
        "fingerprint": fingerprint,
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in output_files.items()
            if name != "provenance"
        },
    }
    atomic_text(
        output_files["provenance"],
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
    )
    print(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "selected": len(records),
                "image_root": str(image_root.resolve()),
                "outputs": {name: str(path.resolve()) for name, path in output_files.items()},
                "eligible_as_unknown_institution": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
