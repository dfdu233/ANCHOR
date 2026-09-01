#!/usr/bin/env python3
"""Remove target-overlapping documents before a retrieval index is built."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .audit_retrieval_split import read_rows, target_keys
from .hashing import sha256_file, sha256_json
from .store import atomic_write_json


VERSION = "rag-corpus-decontamination-v1"
KEYS = ("patient_id", "study_id", "image_id", "report_sha256")


def decontaminate(corpus_rows: list[dict[str, Any]], query_rows: list[dict[str, Any]]):
    forbidden = target_keys(query_rows)
    kept, removed = [], []
    for row in corpus_rows:
        reasons = [key for key in KEYS if row.get(key) and str(row[key]) in forbidden[key]]
        if reasons:
            removed.append({"doc_id": str(row["doc_id"]), "reasons": reasons})
        else:
            kept.append(row)
    return kept, removed


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-corpus", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args()
    corpus_rows, query_rows = read_rows(args.corpus), read_rows(args.queries)
    kept, removed = decontaminate(corpus_rows, query_rows)
    write_jsonl(args.output_corpus, kept)
    result = {
        "protocol_version": VERSION,
        "operation": "deletion only; target references never enter retrieval documents or scores",
        "source_corpus": str(args.corpus.resolve()),
        "source_corpus_sha256": sha256_file(args.corpus),
        "queries": str(args.queries.resolve()),
        "queries_sha256": sha256_file(args.queries),
        "output_corpus": str(args.output_corpus.resolve()),
        "output_corpus_sha256": sha256_file(args.output_corpus),
        "before": len(corpus_rows),
        "after": len(kept),
        "removed": len(removed),
        "removed_reason_counts": {
            key: sum(key in row["reasons"] for row in removed) for key in KEYS
        },
        "removed_doc_ids_sha256": sha256_json(sorted(row["doc_id"] for row in removed)),
    }
    atomic_write_json(args.output_audit, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
