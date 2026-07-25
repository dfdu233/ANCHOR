"""Build a leak-checked Source Bank entry from official LLaVA-Med alignment images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.protocol_v2 import file_sha256, resolve_image
from corrected_sgta.source_bank_v2 import SOURCE_BANK_VERSION, sha256_file


VERSION = "llava-med-exact-source-bank-v1"
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-index", required=True, type=Path)
    parser.add_argument("--prepared-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cxr-benchmark", required=True, type=Path)
    parser.add_argument("--mm-benchmark", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=384)
    return parser.parse_args()


def canonical_rgb_sha256(path: Path) -> str:
    with Image.open(path) as source:
        payload = np.asarray(source.convert("RGB")).tobytes()
    return hashlib.sha256(payload).hexdigest()


def amplitude_center(paths: list[Path], size: int) -> np.ndarray:
    total = np.zeros((size, size), dtype=np.float64)
    for path in tqdm(paths, desc="exact-source amplitude center"):
        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                (size, size), Image.Resampling.LANCZOS
            )
        gray = np.asarray(image, dtype=np.float64).mean(axis=2)
        total += np.abs(np.fft.fft2(gray))
    if not paths:
        raise RuntimeError("no exact-source images after exclusions")
    return total / len(paths)


def benchmark_target_hashes(paths: list[Path]) -> tuple[set[str], dict]:
    hashes = set()
    rows_seen = 0
    resolved_images = 0
    unreadable_images = 0
    for dataset in paths:
        for sample in json.loads(dataset.read_text()):
            rows_seen += 1
            image = resolve_image(sample.get("img_name", ""))
            if image is None:
                continue
            resolved_images += 1
            try:
                hashes.add(canonical_rgb_sha256(image))
            except OSError:
                unreadable_images += 1
    return hashes, {
        "datasets": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in paths
        ],
        "rows_seen": rows_seen,
        "resolved_images": resolved_images,
        "unreadable_images": unreadable_images,
        "canonical_rgb_hashes": len(hashes),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    amplitude_dir = args.output_dir / "amplitudes"
    index_dir = args.output_dir / "indices"
    amplitude_dir.mkdir(exist_ok=True)
    index_dir.mkdir(exist_ok=True)

    target_hashes, exclusions = benchmark_target_hashes(
        [args.cxr_benchmark, args.mm_benchmark]
    )
    prepared = [
        json.loads(line)
        for line in args.prepared_index.read_text().splitlines()
        if line.strip()
    ]
    accepted = []
    seen = set()
    excluded_target = 0
    excluded_duplicate = 0
    for row in prepared:
        path = Path(row["local_path"])
        canonical = canonical_rgb_sha256(path)
        if canonical in target_hashes:
            excluded_target += 1
            continue
        if canonical in seen:
            excluded_duplicate += 1
            continue
        seen.add(canonical)
        accepted.append(
            {
                "kind": "path",
                "path": str(path.resolve()),
                "file_sha256": sha256_file(path),
                "canonical_rgb_sha256": canonical,
                "bytes": path.stat().st_size,
                "pair_id": row["pair_id"],
                "source_url": row["source_url"],
                "caption": row["caption"],
            }
        )

    amplitude_path = amplitude_dir / "llava_alignment_cxr_exact.npy"
    np.save(
        amplitude_path,
        amplitude_center([Path(row["path"]) for row in accepted], args.target_size),
    )
    index_path = index_dir / "llava_alignment_cxr_exact.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in accepted)
    )
    prepared_meta = json.loads(args.prepared_metadata.read_text())
    entry = {
        "source_id": "llava_alignment_cxr_exact",
        "modality": "xray",
        "dataset": "LLaVA-Med alignment 500k (official release)",
        "formal": True,
        "n_candidates_after_exclusion": len(prepared) - excluded_target,
        "n_excluded": excluded_target + excluded_duplicate,
        "n_used": len(accepted),
        "amplitude_file": str(amplitude_path.resolve()),
        "amplitude_sha256": sha256_file(amplitude_path),
        "image_index": str(index_path.resolve()),
        "image_index_sha256": sha256_file(index_path),
        "source_metadata": prepared_meta["config"]["alignment"],
        "source_metadata_sha256": prepared_meta["config"]["alignment_sha256"],
        "source_url_metadata": prepared_meta["config"]["image_urls"],
        "source_url_metadata_sha256": prepared_meta["config"]["image_urls_sha256"],
        "selection": prepared_meta["config"]["selection"],
        "prepared_fingerprint": prepared_meta["fingerprint"],
        "exclusion_policy": (
            "exclude canonical RGB hashes of CXR-VisHal and MM-VisHal, then "
            "remove within-source canonical RGB duplicates"
        ),
        "model_source_relation": {
            "llava": (
                "exact released alignment-stage source membership for "
                "LLaVA-Med; caption-filtered CXR candidate subset"
            ),
            "hulu": "not a claimed Hulu-Med training source",
        },
    }
    manifest = {
        "source_bank_version": SOURCE_BANK_VERSION,
        "builder_version": VERSION,
        "target_size": args.target_size,
        "benchmark_exclusion": exclusions,
        "entries": [entry],
        "formal_source_ids": [entry["source_id"]],
        "input_provenance": {
            "prepared_index": str(args.prepared_index.resolve()),
            "prepared_index_sha256": file_sha256(args.prepared_index),
            "prepared_metadata": str(args.prepared_metadata.resolve()),
            "prepared_metadata_sha256": file_sha256(args.prepared_metadata),
        },
    }
    manifest_path = args.output_dir / "source_bank.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "accepted": len(accepted),
                "excluded_target": excluded_target,
                "excluded_duplicate": excluded_duplicate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
