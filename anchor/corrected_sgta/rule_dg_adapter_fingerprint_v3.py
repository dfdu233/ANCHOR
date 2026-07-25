"""Next-version RULE DG inference fingerprint helpers.

This module is intentionally not imported by ``infer_rule_dg_adapter.py`` so
an in-flight inference-v2 cache can resume under its original fingerprint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FINGERPRINT_VERSION = "rule-sequence-dg-inference-fingerprint-v3"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def tree_identity(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"empty base-model tree: {path}")
    manifest = [
        {
            "path": item.relative_to(path).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in files
    ]
    return {
        "root": str(path.resolve()),
        "files": len(manifest),
        "tree_sha256": stable_sha256(manifest),
    }


def image_manifest_identity(
    rows: list[dict[str, Any]], image_root: Path
) -> dict[str, Any]:
    relative_paths = sorted({str(row["image"]) for row in rows})
    manifest = []
    for relative in relative_paths:
        path = image_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing inference image: {path}")
        manifest.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "root": str(image_root.resolve()),
        "unique_images": len(manifest),
        "manifest_sha256": stable_sha256(manifest),
    }


def fingerprint_data_v3(
    *,
    questions: Path,
    checkpoint: Path,
    base_answers: Path | None,
    inference_code: Path,
    base_model: Path,
    image_root: Path,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    prompt_protocol: str,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    return {
        "version": FINGERPRINT_VERSION,
        "questions_sha256": file_sha256(questions),
        "checkpoint_sha256": file_sha256(checkpoint),
        "base_answers_sha256": (
            file_sha256(base_answers) if base_answers else None
        ),
        "inference_code": {
            "path": str(inference_code.resolve()),
            "sha256": file_sha256(inference_code),
        },
        "base_model": tree_identity(base_model),
        "images": image_manifest_identity(rows, image_root),
        "max_new_tokens": max_new_tokens,
        "prompt_protocol": prompt_protocol,
    }


def fingerprint_v3(payload: dict[str, Any]) -> str:
    if payload.get("version") != FINGERPRINT_VERSION:
        raise ValueError("not a RULE DG inference-v3 fingerprint payload")
    return stable_sha256(payload)
