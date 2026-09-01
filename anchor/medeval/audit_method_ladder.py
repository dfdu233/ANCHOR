#!/usr/bin/env python3
"""Fail-closed T0 audit for the unified baseline and RAG ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .hashing import sha256_file, source_tree_fingerprint
from .store import atomic_write_json


VERSION = "method-ladder-t0-audit-v1"


def resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def audit_method(root: Path, method: dict[str, Any]) -> dict[str, Any]:
    source = resolve(root, method.get("source"))
    license_path = resolve(root, method.get("license"))
    implementation = resolve(root, method.get("implementation"))
    checkpoint = resolve(root, method.get("checkpoint"))
    reasons = []
    if method.get("official_code_required") and (source is None or not source.exists()):
        reasons.append("official_source_missing")
    if method.get("official_code_required") and (license_path is None or not license_path.is_file()):
        reasons.append("license_missing")
    if implementation is None or not implementation.exists():
        reasons.append("implementation_missing")
    required_data = [resolve(root, value) for value in method.get("required_data", [])]
    missing_data = [str(path) for path in required_data if path is None or not path.exists()]
    if missing_data:
        reasons.append("required_data_missing")
    if "paper_native" in method["tracks"] and "checkpoint" in method and (
        checkpoint is None or not checkpoint.exists()
    ):
        reasons.append("paper_native_checkpoint_missing")
    status = "pass" if not reasons else "not_admissible"
    return {
        **method,
        "t0_status": status,
        "t0_reasons": reasons,
        "source_resolved": None if source is None else str(source.resolve()),
        "source_fingerprint": (
            source_tree_fingerprint(source) if source is not None and source.is_dir() else None
        ),
        "license_resolved": None if license_path is None else str(license_path.resolve()),
        "license_sha256": (
            sha256_file(license_path) if license_path is not None and license_path.is_file() else None
        ),
        "implementation_resolved": (
            None if implementation is None else str(implementation.resolve())
        ),
        "required_data_resolved": [str(path.resolve()) for path in required_data if path is not None],
        "missing_data": missing_data,
    }


def audit(config: Path, root: Path) -> dict[str, Any]:
    payload = json.loads(config.read_text())
    methods = [audit_method(root, method) for method in payload["methods"]]
    return {
        "protocol_version": VERSION,
        "config": str(config.resolve()),
        "config_sha256": sha256_file(config),
        "repository_root": str(root.resolve()),
        "methods": methods,
        "summary": {
            "n": len(methods),
            "pass": sum(row["t0_status"] == "pass" for row in methods),
            "not_admissible": sum(row["t0_status"] != "pass" for row in methods),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config.resolve(), args.root.resolve())
    atomic_write_json(args.output, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
