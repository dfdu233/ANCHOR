"""Content-verifying Source Bank access."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.source_bank_v2 import *  # noqa: F403
from corrected_sgta.source_bank_v2 import (
    load_descriptor_image as base_load_descriptor_image,
    load_index,
    verify_source_artifacts as base_verify_source_artifacts,
)


def verify_descriptor(descriptor: dict) -> None:
    if descriptor.get("kind", "path") != "path":
        raise RuntimeError("formal source descriptor is not content-verifiable")
    path = Path(descriptor["path"])
    expected = descriptor.get("file_sha256")
    if not expected:
        raise RuntimeError(f"missing source image hash: {path}")
    actual = sha256_file(path)  # noqa: F405
    if actual != expected:
        raise RuntimeError(f"source image hash mismatch: {path}")


def load_descriptor_image(descriptor: dict):
    verify_descriptor(descriptor)
    return base_load_descriptor_image(descriptor)


def verify_source_artifacts(manifest: dict) -> dict[str, str]:
    verified = base_verify_source_artifacts(manifest)
    for entry in manifest.get("entries", []):
        if not entry.get("formal"):
            continue
        descriptors = load_index(Path(entry["image_index"]))
        for descriptor in descriptors:
            verify_descriptor(descriptor)
        verified[f"source_images:{entry['source_id']}"] = str(len(descriptors))
    return verified
