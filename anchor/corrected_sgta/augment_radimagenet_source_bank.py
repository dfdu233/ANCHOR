"""Add provenance-complete CT/MRI RadImageNet entries to a Source Bank."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from corrected_sgta.build_source_bank import DEFAULT_CXR, DEFAULT_MM
from corrected_sgta.protocol_v2 import file_sha256, resolve_image
from corrected_sgta.source_bank_v2 import load_manifest, sha256_file


DEFAULT_CT = Path(
    "/root/autodl-tmp/source_data/radimagenet/alignment/"
    "train-00000-of-00059.parquet"
)
DEFAULT_MRI = Path(
    "/root/autodl-tmp/source_data/radimagenet/alignment/"
    "train-00030-of-00059.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--ct-parquet", type=Path, default=DEFAULT_CT)
    parser.add_argument("--mri-parquet", type=Path, default=DEFAULT_MRI)
    parser.add_argument("--cxr-benchmark", type=Path, default=DEFAULT_CXR)
    parser.add_argument("--mm-benchmark", type=Path, default=DEFAULT_MM)
    parser.add_argument("--max-per-source", type=int, default=1024)
    parser.add_argument("--target-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pixel_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def benchmark_pixel_hashes(paths: list[Path]) -> tuple[set[str], dict]:
    hashes: set[str] = set()
    resolved: set[str] = set()
    for dataset in paths:
        for sample in json.loads(dataset.read_text()):
            path = resolve_image(sample.get("img_name", ""))
            if path is None or str(path.resolve()) in resolved:
                continue
            resolved.add(str(path.resolve()))
            try:
                with Image.open(path) as image:
                    hashes.add(pixel_digest(image))
            except OSError:
                continue
    return hashes, {
        "datasets": [{"path": str(p), "sha256": file_sha256(p)} for p in paths],
        "unique_resolved_images": len(resolved),
        "canonical_pixel_hashes": len(hashes),
    }


def deterministic_rows(parquet: Path, seed: int, limit: int) -> tuple[list[int], int]:
    table = pq.read_table(parquet, columns=["metadata"])
    ranked = []
    modalities = set()
    for index, metadata in enumerate(table.column("metadata").to_pylist()):
        modality = str((metadata or {}).get("modality") or "").lower()
        modalities.add(modality)
        qid = str((metadata or {}).get("question_id") or index)
        rank = hashlib.sha256(f"{seed}:{parquet.name}:{qid}:{index}".encode()).hexdigest()
        ranked.append((rank, index))
    ranked.sort()
    if not modalities:
        raise RuntimeError(f"missing modality metadata: {parquet}")
    return [index for _, index in ranked[:limit]], len(ranked)


def build_entry(
    source_bank: Path,
    parquet: Path,
    source_id: str,
    modality: str,
    benchmark_hashes: set[str],
    args: argparse.Namespace,
) -> dict:
    selected_indices, row_count = deterministic_rows(
        parquet, args.seed, max(args.max_per_source * 2, args.max_per_source + 64)
    )
    table = pq.read_table(parquet, columns=["image", "metadata"])
    image_root = source_bank.parent / "source_images" / source_id
    image_root.mkdir(parents=True, exist_ok=True)
    total = np.zeros((args.target_size, args.target_size), dtype=np.float64)
    descriptors = []
    overlap = 0
    for row_index in tqdm(selected_indices, desc=f"extract {source_id}"):
        if len(descriptors) >= args.max_per_source:
            break
        row = table.slice(row_index, 1).to_pylist()[0]
        payload = (row.get("image") or {}).get("bytes")
        if not payload:
            continue
        try:
            with Image.open(io.BytesIO(payload)) as raw:
                image = raw.convert("RGB")
            if pixel_digest(image) in benchmark_hashes:
                overlap += 1
                continue
            filename = f"{row_index:06d}.png"
            destination = image_root / filename
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
                    "parquet_row_index": row_index,
                    "question_id": str((row.get("metadata") or {}).get("question_id")),
                }
            )
        except Exception:
            continue
    if len(descriptors) < args.max_per_source:
        raise RuntimeError(
            f"only {len(descriptors)} usable images for {source_id}; "
            f"requested {args.max_per_source}"
        )
    amplitude = total / len(descriptors)
    amplitude_path = source_bank.parent / "amplitudes" / f"{source_id}.npy"
    index_path = source_bank.parent / "indices" / f"{source_id}.jsonl"
    np.save(amplitude_path, amplitude)
    index_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in descriptors)
    )
    return {
        "source_id": source_id,
        "modality": modality,
        "dataset": "RadImageNet-VQA",
        "formal": True,
        "n_candidates_after_exclusion": row_count - overlap,
        "n_excluded": overlap,
        "n_used": len(descriptors),
        "amplitude_file": str(amplitude_path.resolve()),
        "amplitude_sha256": sha256_file(amplitude_path),
        "image_index": str(index_path.resolve()),
        "image_index_sha256": sha256_file(index_path),
        "source_parquet": str(parquet.resolve()),
        "source_parquet_sha256": sha256_file(parquet),
        "exclusion_policy": (
            "exclude canonical RGB pixel hashes of every image resolved by "
            "CXR-VisHal or MM-VisHal; source dataset is otherwise disjoint"
        ),
    }


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.source_bank)
    benchmark_hashes, exclusion = benchmark_pixel_hashes(
        [args.cxr_benchmark, args.mm_benchmark]
    )
    additions = [
        build_entry(
            args.source_bank,
            args.ct_parquet,
            "radimagenet_ct_leaksafe",
            "ct",
            benchmark_hashes,
            args,
        ),
        build_entry(
            args.source_bank,
            args.mri_parquet,
            "radimagenet_mri_leaksafe",
            "mri",
            benchmark_hashes,
            args,
        ),
    ]
    replace_ids = {item["source_id"] for item in additions}
    manifest["entries"] = [
        item for item in manifest.get("entries", []) if item.get("source_id") not in replace_ids
    ] + additions
    manifest["formal_source_ids"] = [
        item["source_id"] for item in manifest["entries"] if item.get("formal")
    ]
    manifest["radimagenet_benchmark_exclusion"] = exclusion
    manifest["notes"]["radimagenet"] = (
        "CT/MRI entries use local downloaded shards with content-level benchmark exclusion"
    )
    temporary = args.source_bank.with_name(args.source_bank.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(args.source_bank)
    print(json.dumps({"source_bank": str(args.source_bank), "added": additions}, indent=2))


if __name__ == "__main__":
    main()
