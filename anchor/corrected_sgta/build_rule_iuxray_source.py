"""Build a test-disjoint IU-Xray source center from RULE's retriever split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-root", type=Path, default=Path("/root/autodl-tmp/RULE"))
    parser.add_argument("--image-root", type=Path, default=Path("/root/autodl-tmp/MedHEval/images/IU-Xray"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=1024)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_file = args.rule_root / "data/training/retriever/iuxray_train.json"
    test_file = args.rule_root / "data/test/iuxray_test.jsonl"
    train = json.loads(train_file.read_text())
    test = [json.loads(line) for line in test_file.read_text().splitlines() if line.strip()]
    train_paths = {path for row in train for path in row["image_path"]}
    test_paths = {row["image"] for row in test}
    overlap = train_paths & test_paths
    if overlap:
        raise RuntimeError(f"RULE train/test image leakage: {sorted(overlap)[:5]}")
    available = [path for path in train_paths if (args.image_root / path).is_file()]
    available.sort(key=lambda path: hashlib.sha256(f"{args.seed}:{path}".encode()).hexdigest())
    chosen = available[: args.max_images]
    if len(chosen) < min(args.max_images, len(train_paths)):
        raise RuntimeError(f"missing RULE train images: available={len(available)} expected={len(train_paths)}")

    amplitude = np.zeros((args.size, args.size), dtype=np.float64)
    for relative in tqdm(chosen, desc="RULE IU source center"):
        with Image.open(args.image_root / relative) as source:
            image = source.convert("RGB").resize((args.size, args.size), Image.Resampling.LANCZOS)
        gray = np.asarray(image, dtype=np.float64).mean(axis=2)
        amplitude += np.abs(np.fft.fft2(gray))
    amplitude /= len(chosen)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    center = args.output_dir / "rule_iuxray_train_amplitude.npy"
    np.save(center, amplitude)
    index = args.output_dir / "rule_iuxray_train_index.jsonl"
    index.write_text("".join(json.dumps({"path": path}) + "\n" for path in chosen))
    manifest = {
        "version": "rule-iuxray-source-v1",
        "source_id": "rule_iuxray_retriever_train",
        "modality": "xray",
        "train_annotation": str(train_file),
        "train_annotation_sha256": sha256(train_file),
        "test_annotation": str(test_file),
        "test_annotation_sha256": sha256(test_file),
        "train_unique_images": len(train_paths),
        "test_unique_images": len(test_paths),
        "train_test_overlap": 0,
        "available_train_images": len(available),
        "n_used": len(chosen),
        "seed": args.seed,
        "size": args.size,
        "amplitude_file": str(center.resolve()),
        "amplitude_sha256": sha256(center),
        "index_file": str(index.resolve()),
        "index_sha256": sha256(index),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
