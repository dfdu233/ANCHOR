#!/usr/bin/env python3
"""Label-blind upper bound on exact-constraint overlap before physician review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corrected_sgta.compile_specificity_ratchet_mechanism_manifest_v1 import exact_constraint_spans
from corrected_sgta.compile_specificity_ratchet_replay_manifest_v1 import (
    SPLIT_SEED,
    _label_blind_case_pool,
)


PROTOCOL_ID = "specificity-ratchet-label-blind-lexical-overlap-ceiling-v1"
MIN_REQUIRED_BLOCKS = 10


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(candidates_path: Path) -> dict[str, Any]:
    candidates = [
        json.loads(line) for line in candidates_path.read_text().splitlines() if line.strip()
    ]
    case_split = {
        row["case_id"]: row["split"] for row in _label_blind_case_pool(candidates)
    }
    rows = []
    for candidate in candidates:
        key = " || ".join(
            str(span["text"]).strip().casefold() for span in exact_constraint_spans(candidate)
        )
        rows.append({"case_id": candidate["case_id"], "split": case_split[candidate["case_id"]], "key": key})

    def summarize(subset: list[dict[str, str]]) -> dict[str, Any]:
        by_key: dict[str, set[str]] = defaultdict(set)
        for row in subset:
            by_key[row["key"]].add(row["case_id"])
        counts = Counter(len(cases) for cases in by_key.values())
        repeated = {key: cases for key, cases in by_key.items() if len(cases) >= 2}
        return {
            "edges": len(subset),
            "cases": len({row["case_id"] for row in subset}),
            "lexical_keys": len(by_key),
            "singleton_keys": counts.get(1, 0),
            "keys_in_at_least_two_cases": len(repeated),
            "edges_in_repeated_keys": sum(
                1 for row in subset if row["key"] in repeated
            ),
            "maximum_possible_cross_role_exact_lexical_blocks": len(repeated),
        }

    summaries = {
        split: summarize([row for row in rows if split == "all" or row["split"] == split])
        for split in ("all", "dev", "test")
    }
    failed = [
        split for split in ("dev", "test")
        if summaries[split]["maximum_possible_cross_role_exact_lexical_blocks"] < MIN_REQUIRED_BLOCKS
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "confirmatory_overlap_ceiling_failed" if failed else "ceiling_not_failed",
        "dataset": "VQA-RAD public OE candidate pack",
        "model": "HuatuoGPT-Vision-7B source outputs",
        "method": "label-blind exact added-constraint overlap ceiling",
        "seed": SPLIT_SEED,
        "candidate_sha256": _sha(candidates_path),
        "source_sha256": _sha(Path(__file__).resolve()),
        "minimum_required_blocks_per_split": MIN_REQUIRED_BLOCKS,
        "failed_splits": failed,
        "summaries": summaries,
        "interpretation": (
            "This is an optimistic upper bound before physician exclusions; it supplies no clinical role label."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.candidates)
    result["command"] = [shlex.join(sys.argv)]
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
