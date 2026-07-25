"""Compact, resumable evidence-cache encoding and validation."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def encode_array(array: np.ndarray, dtype: str = "float16") -> dict[str, Any]:
    value = np.ascontiguousarray(np.asarray(array).astype(dtype))
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "data": base64.b64encode(value.tobytes()).decode("ascii"),
    }


def decode_array(value: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(value["data"])
    array = np.frombuffer(raw, dtype=np.dtype(value["dtype"]))
    return array.reshape(value["shape"]).copy()



def repair_truncated_jsonl_tail(path: Path) -> dict[str, object]:
    """Make an interrupted JSONL tail append-safe without touching valid rows."""

    if not path.exists() or path.stat().st_size == 0:
        return {"action": "none", "bytes_removed": 0}
    data = path.read_bytes()
    stripped = data.rstrip(b"\r\n")
    if not stripped:
        return {"action": "none", "bytes_removed": 0}
    line_start = stripped.rfind(b"\n") + 1
    tail = stripped[line_start:]
    try:
        json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        with path.open("r+b") as handle:
            handle.truncate(line_start)
        return {"action": "truncated_invalid_tail", "bytes_removed": len(data) - line_start}
    if not data.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
        return {"action": "added_missing_newline", "bytes_removed": 0}
    return {"action": "none", "bytes_removed": 0}

def load_successful_qids(path: Path, fingerprint: str) -> set[str]:
    saved: set[str] = set()
    if not path.exists():
        return saved
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    f"cache protocol mismatch in {path}; refusing unsafe partial reuse"
                )
            if row.get("status") == "ok":
                saved.add(str(row["qid"]))
    return saved


def iter_successes(
    path: Path, fingerprint: str | None = None
) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if fingerprint is not None and row.get("fingerprint") != fingerprint:
                raise RuntimeError(f"cache protocol mismatch in {path}")
            if row.get("status") == "ok":
                yield row
