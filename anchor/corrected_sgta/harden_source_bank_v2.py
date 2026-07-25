"""Add immutable per-source-image content provenance to an existing bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from corrected_sgta.source_bank_v2 import load_index, load_manifest, sha256_file


RELEASES = {
    "IU-Xray": "OpenI IU Chest X-ray collection; local MedHEval copy",
    "MIMIC-CXR parquet subset": "local parquet/extracted subset; source is MIMIC-CXR",
    "RadImageNet-VQA": "yixuantt/RadImageNet-VQA alignment parquet shards",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", required=True, type=Path)
    return parser.parse_args()


def canonical_pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.source_bank)
    for entry in manifest.get("entries", []):
        if not entry.get("formal") or not entry.get("image_index"):
            continue
        index_path = Path(entry["image_index"])
        descriptors = load_index(index_path)
        hardened = []
        for descriptor in tqdm(descriptors, desc=f"hash {entry['source_id']}"):
            if descriptor.get("kind", "path") != "path":
                raise RuntimeError(
                    f"formal descriptor lacks a directly hashable path: {descriptor}"
                )
            path = Path(descriptor["path"])
            descriptor = dict(descriptor)
            descriptor["file_sha256"] = sha256_file(path)
            descriptor["canonical_rgb_sha256"] = canonical_pixel_sha256(path)
            descriptor["bytes"] = path.stat().st_size
            hardened.append(descriptor)
        temporary_index = index_path.with_name(index_path.name + ".tmp")
        temporary_index.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in hardened)
        )
        temporary_index.replace(index_path)
        entry["image_index_sha256"] = sha256_file(index_path)
        entry["source_content_hashes"] = "file SHA256 + canonical RGB SHA256 per indexed image"
        entry["dataset_release"] = RELEASES.get(entry.get("dataset"), "local pinned source")
    manifest["provenance_version"] = "source-image-content-v2"
    temporary = args.source_bank.with_name(args.source_bank.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(args.source_bank)
    print(json.dumps({"source_bank": str(args.source_bank), "formal": manifest["formal_source_ids"]}, indent=2))


if __name__ == "__main__":
    main()
