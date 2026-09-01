#!/usr/bin/env python3
"""Fail-closed verification for a hash-bound experimental handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_handoff(path: Path, *, expected_schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != expected_schema:
        raise ValueError(
            f"schema drift: expected {expected_schema!r}, "
            f"got {payload.get('schema_version')!r}"
        )

    bindings = payload.get("source_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("source_bindings must be a non-empty object")

    checked: list[dict[str, Any]] = []
    for name, binding in sorted(bindings.items()):
        if not isinstance(binding, dict):
            raise ValueError(f"source binding {name!r} is not an object")
        source = Path(str(binding.get("path", "")))
        if not source.is_absolute() or not source.is_file():
            raise ValueError(f"bound source {name!r} is missing: {source}")
        expected_bytes = binding.get("bytes")
        expected_hash = binding.get("sha256")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise ValueError(f"invalid byte binding for {name!r}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"invalid SHA-256 binding for {name!r}")
        actual_bytes = source.stat().st_size
        actual_hash = _sha256(source)
        if actual_bytes != expected_bytes or actual_hash != expected_hash:
            raise ValueError(
                f"source drift for {name!r}: expected "
                f"{expected_bytes} bytes/{expected_hash}, got "
                f"{actual_bytes} bytes/{actual_hash}"
            )
        checked.append(
            {"name": name, "path": str(source), "bytes": actual_bytes, "sha256": actual_hash}
        )

    frozen_input = payload.get("frozen_input")
    if not isinstance(frozen_input, dict):
        raise ValueError("frozen_input must be an object")
    image = Path(str(frozen_input.get("image_path", "")))
    expected_image_hash = frozen_input.get("image_sha256")
    if not image.is_absolute() or not image.is_file():
        raise ValueError(f"frozen image is missing: {image}")
    if not isinstance(expected_image_hash, str) or len(expected_image_hash) != 64:
        raise ValueError("frozen image SHA-256 is invalid")
    actual_image_hash = _sha256(image)
    if actual_image_hash != expected_image_hash:
        raise ValueError(
            f"frozen image drift: expected {expected_image_hash}, got {actual_image_hash}"
        )

    return {
        "schema_version": "frozen-handoff-binding-verification-v1",
        "handoff": str(path.resolve()),
        "handoff_sha256": _sha256(path),
        "expected_schema": expected_schema,
        "bindings_checked": checked,
        "frozen_image": {"path": str(image), "sha256": actual_image_hash},
        "verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--expected-schema", required=True)
    parser.add_argument("--phase", choices=("pre_lock", "post_lock"), required=True)
    args = parser.parse_args()
    result = verify_handoff(args.handoff, expected_schema=args.expected_schema)
    result["phase"] = args.phase
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
