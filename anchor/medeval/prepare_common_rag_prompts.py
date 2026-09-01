#!/usr/bin/env python3
"""Create matched no-context/RAG manifests with one model-independent schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from corrected_sgta.evaluate_medheval_answers import normalize_binary_reference

from .audit_retrieval_split import read_rows
from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "common-medical-rag-prompt-v2-observability"


def qid(row: dict[str, Any], index: int) -> str:
    return str(row.get("question_id", row.get("qid", row.get("id", index))))


def render(question: str, documents: list[dict[str, Any]], use_context: bool) -> str:
    question = question.replace("<image>", "").strip()
    if use_context:
        context = "\n".join(f"[{row['rank']}] {row['report']}" for row in documents)
    else:
        context = "[none]"
    return (
        "Use the medical image as the primary evidence. Retrieved reports, if present, "
        "come from different patients and may be irrelevant; never copy a finding that "
        "is not supported by the image.\n"
        f"Retrieved reports:\n{context}\n"
        f"Question: {question}\n"
        "Begin the answer with exactly Yes or No, then give at most one concise sentence."
    )


def prepare(queries: list[dict[str, Any]], retrieval: list[dict[str, Any]], use_context: bool):
    by_id = {str(row["sample_id"]): row for row in retrieval}
    if len(by_id) != len(retrieval):
        raise ValueError("duplicate retrieval sample_id")
    output = []
    for index, row in enumerate(queries):
        sample_id = qid(row, index)
        if row.get("observability") != "image_grounded":
            raise ValueError(f"non-image-grounded query in visual CE protocol: {sample_id}")
        reference = str(row.get("answer", row.get("gt_ans", "")))
        if normalize_binary_reference(reference) is None:
            raise ValueError(f"invalid leading-label reference: {sample_id}")
        retrieved = by_id.get(sample_id)
        if retrieved is None:
            raise ValueError(f"missing retrieval record for {sample_id}")
        image = row.get("img_name", row.get("image"))
        if not image:
            raise ValueError(f"missing image for {sample_id}")
        output.append({
            "qid": sample_id,
            "img_name": str(image),
            "question": render(str(row.get("question", row.get("text", ""))), retrieved["documents"], use_context),
            "answer": reference,
            "question_type": "binary",
            "observability": "image_grounded",
            "context_condition": "retrieved" if use_context else "none",
            "retrieval_index_version": retrieved["index_version"],
            "retrieved_doc_ids": [doc["doc_id"] for doc in retrieved["documents"]] if use_context else [],
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    queries, retrieval = read_rows(args.queries), read_rows(args.retrieval)
    if args.limit:
        queries, retrieval = queries[: args.limit], retrieval[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, use_context in (("no_context", False), ("rag", True)):
        path = args.output_dir / f"{name}.json"
        rows = prepare(queries, retrieval, use_context)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        outputs[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "n": len(rows)}
    manifest = {
        "protocol_version": VERSION,
        "queries": str(args.queries.resolve()),
        "queries_sha256": sha256_file(args.queries),
        "retrieval": str(args.retrieval.resolve()),
        "retrieval_sha256": sha256_file(args.retrieval),
        "outputs": outputs,
        "matched_prompt_except_context": True,
        "reference_used_in_prompt": False,
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
