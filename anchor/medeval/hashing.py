"""Canonical fingerprints used to decide whether two runs are comparable."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_file_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        resolved = path.resolve()
        rows.append({
            "path": str(resolved),
            "size": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        })
    return rows


def source_tree_fingerprint(root: Path) -> dict[str, Any]:
    """Hash a source tree by relative path and bytes, independent of mtimes."""

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    ignored_parts = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and not ignored_parts.intersection(path.relative_to(root).parts)
        and path.suffix not in {".pyc", ".pyo"}
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        content_hash = sha256_file(path)
        digest.update(relative + b"\0" + str(size).encode() + b"\0")
        digest.update(content_hash.encode() + b"\n")
        total_bytes += size
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root}", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "diff_sha256": sha256_bytes((run("diff", "--binary", "HEAD") or "").encode()),
    }


def runtime_fingerprint() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
