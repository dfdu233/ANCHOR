#!/usr/bin/env python3
"""Freeze deterministic, answer-blind smoke panels for reference benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file


VERSION = "reference-smoke-panels-v1"


def stable_rank(seed: int, qid: str) -> str:
    return hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()


def qid(row: dict[str, Any]) -> str:
    return str(row.get("qid", row.get("question_id", row.get("id"))))


def group_key(row: dict[str, Any], field: str) -> str:
    if field == "task_kind":
        choices = row.get("choices")
        if isinstance(choices, (list, tuple)) and choices:
            return "multiple_choice"
        return "binary" if str(row.get("question_type", "")).lower() == "binary" else "open"
    return str(row.get(field) or "unknown")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--group-by", default="task_kind")
    parser.add_argument("--per-group", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_group <= 0:
        raise ValueError("per-group must be positive")
    rows = json.loads(args.input.read_text())
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row, args.group_by)].append(row)
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for group in sorted(groups):
        ranked = sorted(groups[group], key=lambda row: stable_rank(args.seed, qid(row)))
        take = ranked[: min(args.per_group, len(ranked))]
        selected.extend(take)
        counts[group] = len(take)
    selected.sort(key=lambda row: qid(row))
    if len({qid(row) for row in selected}) != len(selected):
        raise RuntimeError("smoke selection produced duplicate qids")
    write_json(args.output, selected)
    manifest = {
        "version": VERSION,
        "source": str(args.input.resolve()),
        "source_sha256": sha256_file(args.input),
        "selection": "sha256(seed:qid), first per group; labels never inspected",
        "seed": args.seed,
        "group_by": args.group_by,
        "per_group": args.per_group,
        "group_counts": counts,
        "rows": len(selected),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "preparer": str(Path(__file__).resolve()),
        "preparer_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
