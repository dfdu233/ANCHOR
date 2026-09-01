#!/usr/bin/env python3
"""Outcome-blind schema skeleton for model-specific PIH head discovery.

This file validates split geometry and enumerates candidates.  It deliberately
has no outcome field, score function, ranking operation, selected-head writer,
or locked-test reader.  Consequently it cannot authorize formal selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "cecd-pih-dev-selection-schema-only-v1"
MODEL_GEOMETRY = {
    "huatuo": {"layers": 28, "query_heads": 28, "head_width": 128},
    "hulu": {"layers": 36, "query_heads": 32, "head_width": 128},
}
FORBIDDEN_FIELDS = frozenset(
    {"label", "outcome", "score", "prediction", "selected_heads", "test_metric"}
)


class SelectionSchemaError(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_outcome_blind_dev_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "model_family", "split", "records"}
    if set(payload) != required:
        raise SelectionSchemaError("manifest schema drift")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise SelectionSchemaError("schema version mismatch")
    family = str(payload["model_family"])
    if family not in MODEL_GEOMETRY:
        raise SelectionSchemaError("unsupported model family")
    if payload["split"] != "dev":
        raise SelectionSchemaError("only patient/image-disjoint dev is admissible")
    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise SelectionSchemaError("dev records must be a nonempty list")
    image_ids: set[str] = set()
    patient_ids: set[str] = set()
    for row in records:
        if not isinstance(row, Mapping) or set(row) != {
            "record_id",
            "image_id",
            "patient_id",
            "prompt_pair_id",
        }:
            raise SelectionSchemaError("record schema drift or outcome-bearing field")
        if set(row) & FORBIDDEN_FIELDS:
            raise SelectionSchemaError("outcome-bearing field is forbidden")
        image_id = str(row["image_id"])
        patient_id = str(row["patient_id"])
        if not image_id or not patient_id or image_id in image_ids:
            raise SelectionSchemaError("image IDs must be nonempty and unique")
        image_ids.add(image_id)
        patient_ids.add(patient_id)
    geometry = MODEL_GEOMETRY[family]
    candidates = [
        {"layer": layer, "query_head": head}
        for layer in range(geometry["layers"])
        for head in range(geometry["query_heads"])
    ]
    manifest_hash = canonical_sha256(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "schema_only_no_outcomes_no_selection",
        "model_family": family,
        "split": "dev",
        "manifest_hash": manifest_hash,
        "records": len(records),
        "unique_images": len(image_ids),
        "unique_patients": len(patient_ids),
        "candidate_count": len(candidates),
        "candidates_hash": canonical_sha256(candidates),
        "locked_test_scanned": False,
        "formal_head_selection_authorized": False,
        "selected_head_artifact_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.dev_manifest.read_text(encoding="utf-8"))
    print(json.dumps(validate_outcome_blind_dev_manifest(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
