#!/usr/bin/env python3
"""Safely inspect the official FactMM-RAG retriever checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.prepare_vqa_rad_internal_control_dev_v1 import atomic_json


VERSION = "factmm-rag-official-checkpoint-audit-v1"


def audit(checkpoint: Path, download_provenance: Path) -> dict[str, Any]:
    provenance = json.loads(download_provenance.read_text())
    errors = []
    if provenance.get("sha256") != sha256_file(checkpoint):
        errors.append("download provenance hash mismatch")
    if checkpoint.stat().st_size < 1_000_000:
        errors.append("checkpoint is implausibly small")

    import torch

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict):
        errors.append("checkpoint root is not a mapping")
        payload = {}
    state = payload.get("model")
    if not isinstance(state, dict) or not state:
        errors.append("checkpoint lacks a nonempty model state mapping")
        state = {}
    tensor_rows = []
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            continue
        tensor_rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": int(value.numel()),
            }
        )
    if len(tensor_rows) != len(state):
        errors.append("model state includes non-tensor or non-string entries")
    prefixes = Counter(row["name"].split(".", 1)[0] for row in tensor_rows)
    result = {
        "protocol_version": VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "download_provenance": str(download_provenance.resolve()),
        "download_provenance_sha256": sha256_file(download_provenance),
        "safe_weights_only_load": True,
        "mmap_load": True,
        "root_keys": sorted(map(str, payload)),
        "model_state_entries": len(state),
        "tensor_entries": len(tensor_rows),
        "parameter_numel": sum(row["numel"] for row in tensor_rows),
        "top_level_prefix_counts": dict(sorted(prefixes.items())),
        "tensor_schema_sha256": sha256_json(tensor_rows),
        "paper_native_retriever_asset_admissible": not errors,
        "paper_native_generator_asset_admissible": False,
        "paper_native_end_to_end_efficacy_authorized": False,
        "errors": errors,
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--download-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint, args.download_provenance)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
