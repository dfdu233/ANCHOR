#!/usr/bin/env python3
"""Freeze one held-out OE question per VQA-RAD test image without outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.prepare_vqa_rad_internal_control_dev_v1 import atomic_json


VERSION = "internal-control-t3-freeze-v1"
SEED = "vqa-rad-internal-controls-t3-heldout-v1"


def _rank(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode()).hexdigest()


def freeze_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_sha = str(row.get("image_sha256", ""))
        qid = str(row.get("qid", ""))
        if not image_sha or not qid:
            raise ValueError("every source row requires qid and image_sha256")
        by_image[image_sha].append(row)
    selected = [
        min(group, key=lambda row: _rank(SEED, str(row["qid"])))
        for group in by_image.values()
    ]
    return sorted(selected, key=lambda row: _rank(SEED, str(row["image_sha256"])))


def build(source: Path, development: Path, execution: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = json.loads(source.read_text())
    development_rows = json.loads(development.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("source must be a nonempty JSON list")
    selected = freeze_rows(rows)
    if len({row["image_sha256"] for row in selected}) != len(selected):
        raise AssertionError("held-out selection is not image-disjoint")
    development_images = {str(row["image_sha256"]) for row in development_rows}
    held_out_images = {str(row["image_sha256"]) for row in selected}
    overlap = development_images & held_out_images
    if overlap:
        raise ValueError(f"development/test image leakage: {len(overlap)} images")
    provenance = {
        "protocol_version": VERSION,
        "status": "frozen_before_held_out_T3_generation_or_scoring",
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": sha256_file(source),
        "source_rows": len(rows),
        "source_unique_images": len({row["image_sha256"] for row in rows}),
        "development_manifest": str(development.resolve()),
        "development_manifest_sha256": sha256_file(development),
        "held_out_manifest_sha256": sha256_file(source),
        "execution_contract": str(execution.resolve()),
        "execution_contract_sha256": sha256_file(execution),
        "selection_seed": SEED,
        "selection_fields": ["qid", "image_sha256"],
        "questions_used_for_selection": False,
        "answers_used_for_selection": False,
        "model_outputs_used_for_selection": False,
        "source_test_image_overlap": len(overlap),
        "pilot_rows": len(selected),
        "pilot_unique_images": len(selected),
        "selected_qids": [str(row["qid"]) for row in selected],
    }
    provenance["fingerprint"] = sha256_json(provenance)
    return selected, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--execution-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    args = parser.parse_args()
    rows, provenance = build(args.source, args.development, args.execution_contract)
    atomic_json(args.output, rows)
    provenance["pilot_manifest"] = str(args.output.resolve())
    provenance["pilot_manifest_sha256"] = sha256_file(args.output)
    provenance["fingerprint"] = sha256_json({k: v for k, v in provenance.items() if k != "fingerprint"})
    atomic_json(args.provenance, provenance)
    print(json.dumps({"rows": len(rows), "fingerprint": provenance["fingerprint"]}, indent=2))


if __name__ == "__main__":
    main()
