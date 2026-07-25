"""Leak-aware source-domain bank utilities for SGTA.

The source bank is intentionally independent of task labels.  Each entry
contains an amplitude center, a reproducible source-image index, provenance,
and an explicit benchmark-exclusion policy.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


SOURCE_BANK_VERSION = "sgta-source-bank-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_order(values: Iterable[Path], seed: int) -> list[Path]:
    return sorted(
        values,
        key=lambda path: hashlib.sha256(f"{seed}:{path}".encode()).hexdigest(),
    )


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("source_bank_version") != SOURCE_BANK_VERSION:
        raise RuntimeError(f"unsupported source bank: {path}")
    return manifest


def load_index(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_descriptor_image(descriptor: dict) -> Image.Image:
    kind = descriptor.get("kind", "path")
    if kind == "path":
        with Image.open(descriptor["path"]) as image:
            return image.convert("RGB").copy()
    if kind == "hex_json":
        rows = json.loads(Path(descriptor["path"]).read_text())
        payload = bytes.fromhex(rows[int(descriptor["row_index"])]["image_bytes"])
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB").copy()
    raise ValueError(f"unsupported source descriptor kind: {kind}")


def normalize_modality(value: str | None) -> str | None:
    text = str(value or "").strip().lower().replace("-", "")
    if text in {"xray", "cxr", "radiograph"}:
        return "xray"
    if text in {"ct", "computedtomography"}:
        return "ct"
    if text in {"mri", "mr", "magneticresonance"}:
        return "mri"
    return None


def entries_for_modality(
    manifest: dict, modality: str | None, formal_only: bool = True
) -> list[dict]:
    normalized = normalize_modality(modality)
    entries = []
    for entry in manifest.get("entries", []):
        if formal_only and not entry.get("formal", False):
            continue
        if normalized is not None and normalize_modality(entry.get("modality")) != normalized:
            continue
        entries.append(entry)
    return entries


def load_feature_centers(path: Path, expected_model: str | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    if expected_model is not None and metadata.get("model") != expected_model:
        raise RuntimeError(
            f"visual center/model mismatch: {metadata.get('model')} != {expected_model}"
        )
    payload = np.load(path, allow_pickle=False)
    centers = {
        item["source_id"]: np.asarray(payload[item["array_key"]], dtype=np.float32)
        for item in metadata.get("entries", [])
    }
    return metadata, centers


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 1.0
    return float(np.clip(1.0 - (a @ b) / denominator, 0.0, 2.0))
