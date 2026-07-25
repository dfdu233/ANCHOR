"""Strict, leak-aware Source Bank loading for processor-aware SGTA."""

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
        payload = bytes.fromhex(rows[int(descriptor["row_index"])] ["image_bytes"])
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


def verify_source_artifacts(manifest: dict) -> dict[str, str]:
    """Fail closed if a manifest-referenced artifact was mutated."""

    verified: dict[str, str] = {}
    for entry in manifest.get("entries", []):
        amplitude = Path(entry["amplitude_file"])
        actual = sha256_file(amplitude)
        expected = entry.get("amplitude_sha256")
        if actual != expected:
            raise RuntimeError(
                f"amplitude hash mismatch for {entry.get('source_id')}: {actual} != {expected}"
            )
        verified[str(amplitude.resolve())] = actual
        if entry.get("formal") and entry.get("image_index"):
            index = Path(entry["image_index"])
            actual_index = sha256_file(index)
            expected_index = entry.get("image_index_sha256")
            if actual_index != expected_index:
                raise RuntimeError(
                    f"index hash mismatch for {entry.get('source_id')}: "
                    f"{actual_index} != {expected_index}"
                )
            verified[str(index.resolve())] = actual_index
    return verified


def load_feature_centers(
    path: Path,
    expected_model: str | None = None,
    expected_source_bank_sha256: str | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    if expected_model is not None and metadata.get("model") != expected_model:
        raise RuntimeError(
            f"visual center/model mismatch: {metadata.get('model')} != {expected_model}"
        )
    if (
        expected_source_bank_sha256 is not None
        and metadata.get("source_bank_sha256") != expected_source_bank_sha256
    ):
        raise RuntimeError(
            "visual center/source-bank mismatch: "
            f"{metadata.get('source_bank_sha256')} != {expected_source_bank_sha256}"
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
