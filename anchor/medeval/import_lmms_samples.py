#!/usr/bin/env python3
"""Convert lmms-eval per-sample logs into the unified answer contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _prediction(sample: dict[str, Any]) -> str:
    responses = sample.get("filtered_resps")
    while isinstance(responses, list) and len(responses) == 1:
        responses = responses[0]
    if not isinstance(responses, str):
        raise ValueError("sample does not contain one string prediction in filtered_resps")
    return responses


def convert_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        doc = sample.get("doc")
        if not isinstance(doc, dict):
            raise ValueError("sample is missing its source document")
        qid = str(doc.get("qid") or doc.get("id") or "")
        if not qid:
            raise ValueError("sample source document has no qid/id")
        if qid in seen:
            raise ValueError(f"duplicate qid in lmms-eval samples: {qid}")
        seen.add(qid)
        rows.append({
            "question_id": qid,
            "qid": qid,
            "id": qid,
            "prompt": str(doc.get("question", "")),
            "text": _prediction(sample),
            "answer": str(doc.get("answer", "")),
            "image_sha256": str(doc.get("image_sha256", "")),
            "source": "SECOND-official-lmms-eval",
        })
    return rows


def load_samples(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"non-object sample in {path}")
                    rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = convert_samples(load_samples(args.samples))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

