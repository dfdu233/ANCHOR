#!/usr/bin/env python3
"""Split a canonical JSONL answer stream by its explicit model_id."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.prepare_vqa_rad_internal_control_dev_v1 import atomic_json


VERSION = "split-canonical-answers-by-model-v1"


def split(source: Path, output_dir: Path) -> dict:
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        model = str(row.get("model_id", "")).strip()
        if not model:
            raise ValueError("every answer requires model_id")
        grouped[model].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for model, values in sorted(grouped.items()):
        path = output_dir / f"{model}.answers.jsonl"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in values))
        temporary.replace(path)
        outputs[model] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": len(values),
            "unique_qids": len({str(row["question_id"]) for row in values}),
        }
    result = {
        "protocol_version": VERSION,
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "rows": len(rows),
        "outputs": outputs,
    }
    result["fingerprint"] = sha256_json(result)
    atomic_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(split(args.source, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
