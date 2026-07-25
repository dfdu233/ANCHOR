"""Build a provenance-complete PubMedVision X-ray source from local official zips."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import ijson
import numpy as np
from PIL import Image
from tqdm import tqdm

from corrected_sgta.augment_radimagenet_source_bank import (
    benchmark_pixel_hashes,
    pixel_digest,
)
from corrected_sgta.build_source_bank import DEFAULT_CXR, DEFAULT_MM
from corrected_sgta.source_bank_v2 import load_manifest, sha256_file


DEFAULT_METADATA = Path(
    "/root/autodl-tmp/source_data/pubmedvision/PubMedVision_Alignment_VQA.json"
)
DEFAULT_ZIPS = (
    Path("/autodl-fs/data/pubmedvision_images/images_0.zip"),
    Path("/autodl-fs/data/pubmedvision_images/images_1.zip"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source-bank", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--zip", dest="zips", type=Path, nargs="+", default=DEFAULT_ZIPS)
    parser.add_argument("--cxr-benchmark", type=Path, default=DEFAULT_CXR)
    parser.add_argument("--mm-benchmark", type=Path, default=DEFAULT_MM)
    parser.add_argument("--max-per-source", type=int, default=1024)
    parser.add_argument("--target-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalized_modality(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def zip_member_map(zips: list[Path]) -> dict[str, tuple[Path, str]]:
    members: dict[str, tuple[Path, str]] = {}
    for archive in zips:
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                if member.lower().endswith((".jpg", ".jpeg", ".png")):
                    members.setdefault(Path(member).name, (archive, member))
    return members


def resolve_member(name: str, members: dict[str, tuple[Path, str]]):
    base = Path(name).name
    if base in members:
        return base, members[base]
    stem = Path(base).stem
    if stem.endswith("_0"):
        fallback = stem[:-2] + Path(base).suffix
        if fallback in members:
            return fallback, members[fallback]
    return None


def metadata_candidates(
    metadata: Path, members: dict[str, tuple[Path, str]], seed: int
) -> tuple[list[tuple[str, str, Path, str]], dict]:
    candidates: dict[tuple[str, str], tuple[str, str, Path, str]] = {}
    records = 0
    official_xray_records = 0
    with metadata.open("rb") as handle:
        for item in tqdm(ijson.items(handle, "item"), desc="scan PubMedVision metadata"):
            records += 1
            if normalized_modality(item.get("modality")) not in {
                "xray",
                "radiography",
                "radiograph",
                "chestxray",
            }:
                continue
            official_xray_records += 1
            images = item.get("image") or []
            if isinstance(images, str):
                images = [images]
            for name in images:
                resolved = resolve_member(str(name), members)
                if resolved is None:
                    continue
                canonical_name, (archive, member) = resolved
                key = (str(archive.resolve()), member)
                rank = hashlib.sha256(
                    f"{seed}:{item.get('id')}:{canonical_name}:{member}".encode()
                ).hexdigest()
                candidates[key] = (
                    rank,
                    str(item.get("id")),
                    archive,
                    member,
                )
    ranked = sorted(candidates.values(), key=lambda value: value[0])
    return ranked, {
        "metadata_records": records,
        "official_xray_records": official_xray_records,
        "local_unique_xray_candidates": len(ranked),
    }


def build_entry(args: argparse.Namespace, benchmark_hashes: set[str]) -> tuple[dict, dict]:
    members = zip_member_map(list(args.zips))
    ranked, scan = metadata_candidates(args.metadata, members, args.seed)
    image_root = args.output_dir / "source_images" / "pubmedvision_xray_formal"
    amplitude_root = args.output_dir / "amplitudes"
    index_root = args.output_dir / "indices"
    image_root.mkdir(parents=True, exist_ok=True)
    amplitude_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)
    total = np.zeros((args.target_size, args.target_size), dtype=np.float64)
    descriptors = []
    excluded_benchmark = 0
    excluded_duplicate_pixels = 0
    seen_pixels: set[str] = set()
    open_archives: dict[Path, zipfile.ZipFile] = {}
    try:
        for _, record_id, archive, member in tqdm(ranked, desc="extract PubMedVision X-ray"):
            if len(descriptors) >= args.max_per_source:
                break
            handle = open_archives.setdefault(archive, zipfile.ZipFile(archive))
            try:
                payload = handle.read(member)
                with Image.open(io.BytesIO(payload)) as raw:
                    image = raw.convert("RGB")
                canonical = pixel_digest(image)
                if canonical in benchmark_hashes:
                    excluded_benchmark += 1
                    continue
                if canonical in seen_pixels:
                    excluded_duplicate_pixels += 1
                    continue
                seen_pixels.add(canonical)
                destination = image_root / f"{len(descriptors):06d}.png"
                image.save(destination, format="PNG", optimize=False)
                gray = np.asarray(
                    image.resize(
                        (args.target_size, args.target_size), Image.Resampling.LANCZOS
                    ),
                    dtype=np.float64,
                ).mean(axis=2)
                total += np.abs(np.fft.fft2(gray))
                descriptors.append(
                    {
                        "kind": "path",
                        "path": str(destination.resolve()),
                        "file_sha256": sha256_file(destination),
                        "canonical_rgb_sha256": canonical,
                        "bytes": destination.stat().st_size,
                        "pubmedvision_record_id": record_id,
                        "source_zip": str(archive.resolve()),
                        "source_member": member,
                    }
                )
            except Exception:
                continue
    finally:
        for handle in open_archives.values():
            handle.close()
    if len(descriptors) < args.max_per_source:
        raise RuntimeError(
            f"only {len(descriptors)} usable PubMedVision X-rays; requested {args.max_per_source}"
        )
    amplitude_path = amplitude_root / "pubmedvision_xray_formal.npy"
    index_path = index_root / "pubmedvision_xray_formal.jsonl"
    np.save(amplitude_path, total / len(descriptors))
    index_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in descriptors)
    )
    metadata_sha256 = sha256_file(args.metadata)
    entry = {
        "source_id": "pubmedvision_xray_formal",
        "modality": "xray",
        "dataset": "PubMedVision Alignment VQA",
        "formal": True,
        "n_candidates_after_exclusion": scan["local_unique_xray_candidates"]
        - excluded_benchmark
        - excluded_duplicate_pixels,
        "n_excluded": excluded_benchmark + excluded_duplicate_pixels,
        "n_used": len(descriptors),
        "amplitude_file": str(amplitude_path.resolve()),
        "amplitude_sha256": sha256_file(amplitude_path),
        "image_index": str(index_path.resolve()),
        "image_index_sha256": sha256_file(index_path),
        "source_metadata": str(args.metadata.resolve()),
        "source_metadata_sha256": metadata_sha256,
        "source_zips": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.zips
        ],
        "modality_selection": "official PubMedVision metadata modality field",
        "exclusion_policy": (
            "exclude canonical RGB hashes of all images resolved by CXR-VisHal and "
            "MM-VisHal, then remove within-source duplicate pixel hashes"
        ),
        "source_content_hashes": "file SHA256 + canonical RGB SHA256 per indexed image",
        "dataset_release": (
            "FreedomIntelligence/PubMedVision commit "
            "3c84e04b38bceb5341419b9a4f8ca37ba790cb84; Apache-2.0"
        ),
        "model_source_relation": {
            "llava": (
                "training-adjacent PMC publication-image domain; LLaVA-Med documents "
                "PMC-15M, but exact membership in PMC-15M is not claimed"
            ),
            "hulu": (
                "public medical multimodal proxy source; Hulu-Med has not released its "
                "16.7M sample-level training manifest"
            ),
        },
    }
    scan.update(
        {
            "excluded_benchmark_pixels": excluded_benchmark,
            "excluded_duplicate_pixels": excluded_duplicate_pixels,
            "selected_unique_images": len(descriptors),
        }
    )
    return entry, scan


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.base_source_bank)
    benchmark_hashes, exclusion = benchmark_pixel_hashes(
        [args.cxr_benchmark, args.mm_benchmark]
    )
    entry, scan = build_entry(args, benchmark_hashes)
    replace_ids = {"pubmedvision_xray_formal"}
    manifest["entries"] = [
        item
        for item in manifest.get("entries", [])
        if item.get("source_id") not in replace_ids
    ] + [entry]
    manifest["formal_source_ids"] = [
        item["source_id"] for item in manifest["entries"] if item.get("formal")
    ]
    manifest["pubmedvision_benchmark_exclusion"] = exclusion
    manifest["pubmedvision_scan"] = scan
    manifest["provenance_version"] = "source-image-content-v3-pubmedvision"
    manifest.setdefault("notes", {})["pubmedvision"] = (
        "formal X-ray source rebuilt from official per-image metadata and local pinned zips"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "source_bank.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output)
    print(json.dumps({"source_bank": str(output), "added": entry, "scan": scan}, indent=2))


if __name__ == "__main__":
    main()
