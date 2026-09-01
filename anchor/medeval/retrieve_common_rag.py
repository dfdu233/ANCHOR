#!/usr/bin/env python3
"""Deterministic shared lexical retriever for the common-protocol RAG control."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .audit_retrieval_split import read_rows
from .hashing import sha256_file, sha256_json
from .schema import RetrievalRecord, as_canonical_dict
from .store import atomic_write_json


VERSION = "common-medical-rag-bm25-v1"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_index(rows: list[dict[str, Any]]):
    term_counts = [Counter(tokens(str(row["report"]))) for row in rows]
    lengths = [sum(counts.values()) for counts in term_counts]
    document_frequency = Counter(term for counts in term_counts for term in counts)
    average_length = sum(lengths) / len(lengths)
    return term_counts, lengths, document_frequency, average_length


def retrieve(
    rows: list[dict[str, Any]], query: str, top_k: int, index_data=None
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    counts, lengths, df, avg = index_data or build_index(rows)
    n, k1, b = len(rows), 1.2, 0.75
    query_terms = Counter(tokens(query))
    scores = []
    for index, document in enumerate(counts):
        score = 0.0
        for term, query_weight in query_terms.items():
            frequency = document.get(term, 0)
            if not frequency:
                continue
            inverse = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            norm = frequency + k1 * (1 - b + b * lengths[index] / avg)
            score += query_weight * inverse * frequency * (k1 + 1) / norm
        scores.append((score, str(rows[index]["doc_id"]), index))
    selected = sorted(scores, key=lambda value: (-value[0], value[1]))[:top_k]
    return [
        {
            "doc_id": doc_id,
            "rank": rank,
            "score": float(score),
            "sha256": str(rows[index]["report_sha256"]),
            "report": str(rows[index]["report"]),
            "dataset": str(rows[index]["dataset"]),
        }
        for rank, (score, doc_id, index) in enumerate(selected, 1)
    ]


def qid(row: dict[str, Any], index: int) -> str:
    return str(row.get("question_id", row.get("qid", row.get("id", index))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    corpus = read_rows(args.corpus)
    queries = read_rows(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    index_version = sha256_json({"protocol": VERSION, "corpus_sha256": sha256_file(args.corpus)})
    index_data = build_index(corpus)
    records = []
    for index, row in enumerate(queries):
        query = str(row.get("question", row.get("text", ""))).replace("<image>", "").strip()
        record = RetrievalRecord(
            sample_id=qid(row, index),
            query=query,
            split_policy="train-only; exact image/study/patient/reference-report decontamination",
            index_version=index_version,
            documents=tuple(retrieve(corpus, query, args.top_k, index_data)),
        )
        records.append(as_canonical_dict(record))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "retrieval.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    temporary.replace(output)
    manifest = {
        "protocol_version": VERSION,
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": sha256_file(args.corpus),
        "queries": str(args.queries.resolve()),
        "queries_sha256": sha256_file(args.queries),
        "retrieval": str(output.resolve()),
        "retrieval_sha256": sha256_file(output),
        "index_version": index_version,
        "query_schema": "question text only; answer/reference excluded",
        "context_schema": "ranked report text with doc_id/score/hash",
        "top_k": args.top_k,
        "n": len(records),
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
