"""Build a leak-aware medical source-domain amplitude bank from local data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from corrected_sgta.protocol_v2 import file_sha256, resolve_image
from corrected_sgta.source_bank import SOURCE_BANK_VERSION, deterministic_order, sha256_file


DEFAULT_CXR = Path(
    "/root/autodl-tmp/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/"
    "close-ended/CXR-VisHal.json"
)
DEFAULT_MM = Path(
    "/root/autodl-tmp/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/"
    "close-ended/MM-VisHal.json"
)
IU_ROOT = Path("/root/autodl-tmp/MedHEval/images/IU-Xray")
MIMIC_ROOT = Path("/root/autodl-tmp/data/extracted_images")
LEGACY_ROOT = Path("/root/autodl-tmp/multimodal_center_report/centers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-source", type=int, default=1024)
    parser.add_argument("--target-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cxr-benchmark", type=Path, default=DEFAULT_CXR)
    parser.add_argument("--mm-benchmark", type=Path, default=DEFAULT_MM)
    return parser.parse_args()


def image_paths(root: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg"}
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_exclusions(paths: list[Path]) -> tuple[set[str], set[str], dict]:
    resolved: set[str] = set()
    mimic_hashes: set[str] = set()
    rows_seen = 0
    for dataset in paths:
        for sample in json.loads(dataset.read_text()):
            rows_seen += 1
            image = resolve_image(sample.get("img_name", ""))
            if image is None:
                continue
            real = str(image.resolve())
            resolved.add(real)
            if "/MedHEval/images/p" in real:
                try:
                    mimic_hashes.add(file_digest(image))
                except OSError:
                    pass
    metadata = {
        "datasets": [
            {"path": str(path), "sha256": file_sha256(path)} for path in paths
        ],
        "rows_seen": rows_seen,
        "resolved_images": len(resolved),
        "mimic_content_hashes": len(mimic_hashes),
    }
    return resolved, mimic_hashes, metadata


def amplitude_center(paths: list[Path], size: int) -> tuple[np.ndarray, int, list[str]]:
    total = np.zeros((size, size), dtype=np.float64)
    used = []
    for path in tqdm(paths, desc="amplitude center"):
        try:
            with Image.open(path) as source:
                image = source.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
            gray = np.asarray(image, dtype=np.float64).mean(axis=2)
            total += np.abs(np.fft.fft2(gray))
            used.append(str(path.resolve()))
        except Exception:
            continue
    if not used:
        raise RuntimeError("no readable source images")
    return total / len(used), len(used), used


def write_entry(
    output: Path,
    source_id: str,
    modality: str,
    dataset: str,
    candidates: list[Path],
    excluded: int,
    args: argparse.Namespace,
    exclusion_policy: str,
) -> dict:
    chosen = deterministic_order(candidates, args.seed)[: args.max_per_source]
    center, count, used = amplitude_center(chosen, args.target_size)
    center_path = output / "amplitudes" / f"{source_id}.npy"
    index_path = output / "indices" / f"{source_id}.jsonl"
    np.save(center_path, center)
    index_path.write_text(
        "".join(json.dumps({"kind": "path", "path": path}) + "\n" for path in used)
    )
    return {
        "source_id": source_id,
        "modality": modality,
        "dataset": dataset,
        "formal": True,
        "n_candidates_after_exclusion": len(candidates),
        "n_excluded": excluded,
        "n_used": count,
        "amplitude_file": str(center_path.resolve()),
        "amplitude_sha256": sha256_file(center_path),
        "image_index": str(index_path.resolve()),
        "image_index_sha256": sha256_file(index_path),
        "exclusion_policy": exclusion_policy,
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir
    (output / "amplitudes").mkdir(parents=True, exist_ok=True)
    (output / "indices").mkdir(parents=True, exist_ok=True)
    resolved, mimic_hashes, exclusion = benchmark_exclusions(
        [args.cxr_benchmark, args.mm_benchmark]
    )

    iu_all = image_paths(IU_ROOT)
    iu_paths = [path for path in iu_all if str(path.resolve()) not in resolved]

    mimic_all = image_paths(MIMIC_ROOT)
    mimic_paths = []
    mimic_excluded = 0
    for path in tqdm(mimic_all, desc="MIMIC exclusion hashes"):
        try:
            if file_digest(path) in mimic_hashes:
                mimic_excluded += 1
            else:
                mimic_paths.append(path)
        except OSError:
            continue

    entries = [
        write_entry(
            output,
            "iuxray_xray_leaksafe",
            "xray",
            "IU-Xray",
            iu_paths,
            len(iu_all) - len(iu_paths),
            args,
            "exclude every image path resolved by CXR-VisHal or MM-VisHal",
        ),
        write_entry(
            output,
            "mimic_cxr_leaksafe",
            "xray",
            "MIMIC-CXR parquet subset",
            mimic_paths,
            mimic_excluded,
            args,
            "exclude exact SHA256 matches to benchmark MIMIC images",
        ),
    ]

    legacy = []
    for source_id, modality, dataset, filename in [
        ("pubmedvision_xray_legacy", "xray", "PubMedVision", "pubmedvision_xray.npy"),
        ("pubmedvision_ct_legacy", "ct", "PubMedVision", "pubmedvision_ct.npy"),
        ("pubmedvision_mri_legacy", "mri", "PubMedVision", "pubmedvision_mri.npy"),
        ("radimagenet_ct_legacy", "ct", "RadImageNet-VQA", "radimagenet_ct.npy"),
        ("radimagenet_mri_legacy", "mri", "RadImageNet-VQA", "radimagenet_mri.npy"),
    ]:
        path = LEGACY_ROOT / filename
        if path.is_file():
            legacy.append(
                {
                    "source_id": source_id,
                    "modality": modality,
                    "dataset": dataset,
                    "formal": False,
                    "amplitude_file": str(path.resolve()),
                    "amplitude_sha256": sha256_file(path),
                    "exclusion_policy": "legacy artifact; image-level provenance unavailable",
                }
            )

    manifest = {
        "source_bank_version": SOURCE_BANK_VERSION,
        "seed": args.seed,
        "target_size": args.target_size,
        "max_per_source": args.max_per_source,
        "benchmark_exclusion": exclusion,
        "entries": entries + legacy,
        "formal_source_ids": [entry["source_id"] for entry in entries],
        "notes": {
            "slake": "not used as a formal source: all 180 local SLAKE images occur in MM-VisHal",
            "legacy": "legacy centers remain diagnostic-only until image provenance is rebuilt",
        },
    }
    manifest_path = output / "source_bank.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": str(manifest_path), "formal": entries}, indent=2))


if __name__ == "__main__":
    main()
