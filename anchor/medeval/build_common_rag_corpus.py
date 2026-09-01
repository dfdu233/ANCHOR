#!/usr/bin/env python3
"""Build a normalized, hash-closed report corpus for common-protocol RAG."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .hashing import sha256_file, sha256_json
from .store import atomic_write_json


VERSION = "common-medical-rag-corpus-v1"


def identities(image: str) -> dict[str, str | None]:
    normalized = image.replace("\\", "/")
    mimic = re.search(r"(?:^|/)p\d+/p(?P<patient>\d+)/s(?P<study>\d+)/(?P<image>[^/]+)", normalized)
    if mimic:
        return {
            "patient_id": mimic.group("patient"),
            "study_id": mimic.group("study"),
            "image_id": Path(mimic.group("image")).stem,
        }
    first = normalized.split("/")[0]
    return {"patient_id": None, "study_id": first, "image_id": normalized}


def build(source: Path, dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = json.loads(source.read_text())
    if not isinstance(raw, list):
        raise ValueError("retrieval source must be a JSON list")
    rows = []
    seen = set()
    for item in raw:
        report = " ".join(str(item.get("report", "")).split())
        images = item.get("image_path", item.get("image", []))
        if isinstance(images, str):
            images = [images]
        if not report or not images:
            continue
        doc_id = f"{dataset}:{item.get('id', len(rows))}"
        if doc_id in seen:
            raise ValueError(f"duplicate retrieval doc_id: {doc_id}")
        seen.add(doc_id)
        identity = identities(str(images[0]))
        identity["patient_id"] = str(item.get("subject_id", identity["patient_id"] or "")) or None
        identity["study_id"] = str(item.get("study_id", identity["study_id"] or "")) or None
        rows.append({
            "doc_id": doc_id,
            "dataset": dataset,
            "source_split": str(item.get("split", "train")),
            "report": report,
            "report_sha256": sha256_json(report),
            "image_paths": [str(value) for value in images],
            **identity,
        })
    if not rows:
        raise ValueError("retrieval corpus is empty")
    return rows, {
        "protocol_version": VERSION,
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "dataset": dataset,
        "documents": len(rows),
        "ordered_documents_sha256": sha256_json(rows),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest = build(args.source.resolve(), args.dataset)
    corpus = args.output_dir / "corpus.jsonl"
    manifest["corpus"] = str(corpus.resolve())
    write_jsonl(corpus, rows)
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
