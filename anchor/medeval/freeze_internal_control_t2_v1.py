#!/usr/bin/env python3
"""Freeze an outcome-blind VQA-RAD T2 pilot and execution provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.prepare_vqa_rad_internal_control_dev_v1 import atomic_json


VERSION = "internal-control-t2-freeze-v1"


def _rank(seed: str, *values: object) -> str:
    payload = ":".join([seed, *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_one_per_image(rows: list[dict[str, Any]], size: int, seed: str) -> list[dict[str, Any]]:
    if size <= 0:
        raise ValueError("pilot size must be positive")
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        digest = str(row.get("image_sha256", ""))
        qid = str(row.get("qid", row.get("id", "")))
        if len(digest) != 64 or not qid:
            raise ValueError("every development row needs image_sha256 and qid")
        by_image[digest].append(row)
    chosen = [
        min(group, key=lambda row: _rank(seed, digest, row["qid"]))
        for digest, group in by_image.items()
    ]
    chosen.sort(key=lambda row: _rank(seed, row["image_sha256"]))
    if len(chosen) < size:
        raise ValueError(f"only {len(chosen)} unique development images for pilot size {size}")
    return [dict(row) for row in chosen[:size]]


def freeze(
    *,
    development_manifest: Path,
    development_audit: Path,
    held_out_manifest: Path,
    execution_contract: Path,
    output: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    rows = json.loads(development_manifest.read_text(encoding="utf-8"))
    audit = json.loads(development_audit.read_text(encoding="utf-8"))
    contract = json.loads(execution_contract.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("development manifest must be a JSON list")
    selection = contract["development"]
    pilot = select_one_per_image(rows, int(selection["pilot_size"]), str(selection["selection_seed"]))

    # Reading held-out bytes for hashing is allowed. Its JSON content is never
    # parsed here, preventing accidental answer-dependent pilot selection.
    dev_images = {str(row["image_sha256"]) for row in pilot}
    if len(dev_images) != len(pilot):
        raise AssertionError("T2 pilot is not image independent")
    if audit["counts"]["test_image_overlap_after_filter"] != 0:
        raise AssertionError("source development audit reports held-out image leakage")
    atomic_json(output, pilot)
    provenance = {
        "protocol_version": VERSION,
        "status": "frozen_before_held_out_test_generation",
        "development_manifest": str(development_manifest.resolve()),
        "development_manifest_sha256": sha256_file(development_manifest),
        "development_audit": str(development_audit.resolve()),
        "development_audit_sha256": sha256_file(development_audit),
        "held_out_manifest": str(held_out_manifest.resolve()),
        "held_out_manifest_sha256": sha256_file(held_out_manifest),
        "held_out_manifest_content_parsed": False,
        "held_out_questions_read_for_selection": False,
        "held_out_answers_read_for_selection": False,
        "execution_contract": str(execution_contract.resolve()),
        "execution_contract_sha256": sha256_file(execution_contract),
        "pilot_manifest": str(output.resolve()),
        "pilot_manifest_sha256": sha256_file(output),
        "pilot_rows": len(pilot),
        "pilot_unique_images": len(dev_images),
        "source_open_rows": len(rows),
        "source_open_unique_images": len({str(row["image_sha256"]) for row in rows}),
        "source_test_image_overlap": audit["counts"]["test_image_overlap_after_filter"],
        "selection_seed": selection["selection_seed"],
        "selected_qids": [str(row["qid"]) for row in pilot],
    }
    provenance["fingerprint"] = sha256_json(provenance)
    atomic_json(provenance_path, provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--development-audit", required=True, type=Path)
    parser.add_argument("--held-out-manifest", required=True, type=Path)
    parser.add_argument("--execution-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    args = parser.parse_args()
    result = freeze(
        development_manifest=args.development_manifest,
        development_audit=args.development_audit,
        held_out_manifest=args.held_out_manifest,
        execution_contract=args.execution_contract,
        output=args.output,
        provenance_path=args.provenance,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
