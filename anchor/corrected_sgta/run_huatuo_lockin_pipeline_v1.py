#!/usr/bin/env python3
"""Canary-gated Huatuo runtime for the frozen lock-in development split."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from corrected_sgta.clinical_autoregressive_lockin_probe_v1 import (
    _canonical,
    _row_sha,
    _sha,
    compute_row,
    load_manifest,
    run_runtime,
)
from corrected_sgta.huatuo_lockin_adapter_v1 import create_adapter


VERSION = "huatuo-lockin-canary-gated-dev-pipeline-v1"


def atomic_write_once_or_equal(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite drifted canary artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--huatuo-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    rows, metadata = load_manifest(args.manifest, args.metadata)
    adapter = create_adapter(
        {
            "model_dir": str(args.model_dir),
            "huatuo_root": str(args.huatuo_root),
            "device": args.device,
        }
    )
    canary_row = rows[0]
    canary = compute_row(adapter, canary_row, args.image_root)
    canary_summary = {
        "version": VERSION,
        "status": "passed",
        "sample_id": canary["sample_id"],
        "row_sha256": _row_sha(canary_row),
        "manifest_sha256": metadata["manifest_sha256"],
        "adapter_fingerprint": adapter.fingerprint(),
        "layer_ids": canary["layer_ids"],
        "layer_fractions": canary["layer_fractions"],
        "prompt_end_hidden_dimension": canary["prompt_end_readout"]["hidden_dimension"],
        "prefix_token_lengths": [
            len(step["prefix_token_ids"]) for step in canary["prefix_ladder"]
        ],
        "continuation_token_ids": [
            step["continuation_token_ids"] for step in canary["prefix_ladder"]
        ],
        "all_contextual_offsets_validated": True,
        "all_final_rows_match_standard_logits": True,
        "multimodal_prompt_boundary_certified": True,
        "scientific_direction_read": False,
    }
    canary_summary["canary_fingerprint"] = _sha(_canonical(canary_summary))
    atomic_write_once_or_equal(
        args.output_dir / "ADAPTER_CANARY.json",
        json.dumps(canary_summary, indent=2, sort_keys=True).encode() + b"\n",
    )
    result = run_runtime(
        manifest=args.manifest,
        metadata=args.metadata,
        image_root=args.image_root,
        output_dir=args.output_dir,
        adapter=adapter,
        command=os.sys.argv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

