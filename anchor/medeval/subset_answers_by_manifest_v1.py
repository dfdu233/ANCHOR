#!/usr/bin/env python3
"""Extract an answer subset in exact frozen-manifest order with provenance."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .audit_retrieval_split import read_rows
from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "answer-subset-by-manifest-v1"


def qid(row: dict, index: int) -> str:
    return str(row.get("question_id", row.get("qid", row.get("id", index))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_rows(args.manifest)
    answers = read_rows(args.answers)
    answer_map: dict[str, dict] = {}
    for index, row in enumerate(answers):
        sample_id = qid(row, index)
        if sample_id in answer_map:
            raise ValueError(f"duplicate answer qid: {sample_id}")
        answer_map[sample_id] = row
    order = [qid(row, index) for index, row in enumerate(manifest)]
    if len(order) != len(set(order)):
        raise ValueError("manifest qids are not unique")
    missing = [sample_id for sample_id in order if sample_id not in answer_map]
    if missing:
        raise ValueError(f"answers are missing {len(missing)} manifest qids; first={missing[:5]}")
    selected = [answer_map[sample_id] for sample_id in order]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    os.replace(temporary, args.output)
    audit = {
        "version": VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "answers": str(args.answers.resolve()),
        "answers_sha256": sha256_file(args.answers),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "source_rows": len(answers),
        "selected_rows": len(selected),
        "qid_order_exact": True,
    }
    atomic_write_json(args.output.with_suffix(args.output.suffix + ".audit.json"), audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
