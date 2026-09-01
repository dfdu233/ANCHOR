#!/usr/bin/env python3
"""Fail closed on image, study, patient, or exact-report retrieval leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .build_common_rag_corpus import identities
from .hashing import sha256_file, sha256_json
from .store import atomic_write_json


VERSION = "retrieval-split-leakage-audit-v1"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError("query manifest must be a JSON list or JSONL objects")
    return value


def target_keys(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    result = {name: set() for name in ("patient_id", "study_id", "image_id", "report_sha256")}
    for row in rows:
        images = row.get("image_paths", row.get("image", row.get("img_name", [])))
        if isinstance(images, str):
            images = [images]
        for image in images:
            for key, value in identities(str(image)).items():
                if value:
                    result[key].add(str(value))
        report = row.get("report")
        if report:
            result["report_sha256"].add(sha256_json(" ".join(str(report).split())))
    return result


def audit(corpus: Path, queries: Path) -> dict[str, Any]:
    corpus_rows = read_rows(corpus)
    query_rows = read_rows(queries)
    source = {name: set() for name in ("patient_id", "study_id", "image_id", "report_sha256")}
    for row in corpus_rows:
        for name in source:
            value = row.get(name)
            if value:
                source[name].add(str(value))
    target = target_keys(query_rows)
    overlaps = {name: sorted(source[name] & target[name]) for name in source}
    passed = not any(overlaps.values())
    return {
        "protocol_version": VERSION,
        "corpus": str(corpus.resolve()),
        "corpus_sha256": sha256_file(corpus),
        "queries": str(queries.resolve()),
        "queries_sha256": sha256_file(queries),
        "corpus_documents": len(corpus_rows),
        "query_samples": len(query_rows),
        "overlap_counts": {name: len(values) for name, values in overlaps.items()},
        "overlaps": overlaps,
        "passed": passed,
        "policy": "no exact image, study, patient, or normalized reference-report overlap",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.corpus.resolve(), args.queries.resolve())
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
