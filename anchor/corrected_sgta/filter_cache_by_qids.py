"""Filter an SGTA JSONL cache to a qid subset while preserving row payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--qids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_qids(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise RuntimeError("qid file must be a JSON list")
    return {str(item) for item in payload}


def main() -> None:
    args = parse_args()
    qids = load_qids(args.qids)
    meta_path = args.cache.with_suffix(args.cache.suffix + ".meta.json")
    metadata = json.loads(meta_path.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_meta = {
        **metadata,
        "filtered_from_cache": str(args.cache.resolve()),
        "filtered_qids": str(args.qids.resolve()),
        "filtered_qid_count": len(qids),
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(output_meta, indent=2)
    )
    kept = 0
    seen: set[str] = set()
    with args.cache.open() as src, args.output.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid"))
            if qid in qids:
                dst.write(json.dumps(row, separators=(",", ":")) + "\n")
                kept += 1
                seen.add(qid)
    missing = sorted(qids - seen)
    report = {
        "cache": str(args.cache.resolve()),
        "qids": str(args.qids.resolve()),
        "output": str(args.output.resolve()),
        "requested": len(qids),
        "kept": kept,
        "missing": missing,
    }
    print(json.dumps(report, indent=2))
    if missing:
        raise RuntimeError(f"missing {len(missing)} requested qids")


if __name__ == "__main__":
    main()
