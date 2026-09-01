#!/usr/bin/env python3
"""Build a lightweight, hash-bound provenance record for a baseline gate."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from .hashing import sha256_file, sha256_json

VERSION = "baseline-gate-provenance-v1"
WEIGHT_SUFFIXES = {".bin", ".safetensors"}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def checkpoint_identity(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    metadata = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix not in WEIGHT_SUFFIXES
    )
    weights = sorted(
        ({"path": path.name, "bytes": path.stat().st_size}
         for path in root.iterdir()
         if path.is_file() and path.suffix in WEIGHT_SUFFIXES),
        key=lambda row: row["path"],
    )
    if not weights:
        raise ValueError(f"checkpoint has no top-level weight shards: {root}")
    referenced: set[str] = set()
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = root / name
        if index.is_file():
            payload = json.loads(index.read_text())
            referenced.update(str(value) for value in payload.get("weight_map", {}).values())
    present = {row["path"] for row in weights}
    missing = sorted(referenced - present)
    if missing:
        raise ValueError(f"checkpoint weight shards referenced by index are missing: {missing}")
    return {
        "path": str(root),
        "metadata_files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in metadata
        ],
        "weight_shard_inventory": weights,
        "index_referenced_weight_shards": sorted(referenced),
        "index_complete": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--dependency", action="append", default=[])
    parser.add_argument("--generation-json", required=True)
    args = parser.parse_args()

    sources = [path.resolve() for path in args.source]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"gate provenance sources missing: {missing}")
    generation = json.loads(args.generation_json)
    if not isinstance(generation, dict):
        raise ValueError("generation-json must decode to an object")
    payload = {
        "version": VERSION,
        "model": args.model,
        "method": args.method,
        "sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sources
        ],
        "frozen_config": {
            "path": str(args.config.resolve()),
            "sha256": sha256_file(args.config),
        },
        "gate_manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
        },
        "checkpoint": checkpoint_identity(args.checkpoint),
        "dependencies": {
            name: package_version(name)
            for name in sorted(set(args.dependency or ["torch", "transformers", "accelerate", "numpy"]))
        },
        "generation": generation,
    }
    payload["fingerprint"] = sha256_json(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(payload["fingerprint"])


if __name__ == "__main__":
    main()
