"""Fit a source nuisance subspace from paired clean/domain-augmented images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import sha256_file
from corrected_sgta.source_nuisance import fit_nuisance_subspace


VERSION = "source-nuisance-subspace-v1"
TRANSFORMS = ("gamma_0.7", "gamma_1.4", "contrast_0.65", "blur_1.2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--explained-variance", type=float, default=0.90)
    return parser.parse_args()


def gamma_image(image: Image.Image, gamma: float) -> Image.Image:
    values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    corrected = np.power(values, gamma)
    return Image.fromarray(np.clip(corrected * 255.0, 0, 255).astype(np.uint8))


def domain_views(image: Image.Image) -> list[Image.Image]:
    rgb = image.convert("RGB")
    return [
        gamma_image(rgb, 0.7),
        gamma_image(rgb, 1.4),
        ImageEnhance.Contrast(rgb).enhance(0.65),
        rgb.filter(ImageFilter.GaussianBlur(radius=1.2)),
    ]


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.source_index.read_text().splitlines()
        if line.strip()
    ]
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row['path']}".encode()
        ).hexdigest()
    )
    rows = rows[: args.max_images]
    if not rows:
        raise RuntimeError("empty source index")

    adapter = LlavaMedAlignmentAdapter()
    clean_rows: list[np.ndarray] = []
    shifted_rows: list[np.ndarray] = []
    try:
        for row in tqdm(rows, desc="fit source nuisance"):
            with Image.open(row["path"]) as source:
                image = source.convert("RGB")
            views = domain_views(image)
            features = adapter.visual_features([image, *views])
            clean_rows.append(features[0])
            shifted_rows.append(features[1:])
            image.close()
            for view in views:
                view.close()
    finally:
        adapter.close()

    clean = np.stack(clean_rows)
    shifted = np.stack(shifted_rows)
    source_mean, basis, diagnostics = fit_nuisance_subspace(
        clean, shifted, args.explained_variance
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        source_mean=source_mean,
        nuisance_basis=basis,
        singular_input_clean=clean.astype(np.float16),
    )
    metadata = {
        "version": VERSION,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "model": "llava",
        "model_identity": model_identity("llava"),
        "source_index": str(args.source_index.resolve()),
        "source_index_sha256": sha256_file(args.source_index),
        "max_images": args.max_images,
        "n_images": len(rows),
        "seed": args.seed,
        "selection": "sha256(seed:path)",
        "transforms": list(TRANSFORMS),
        "explained_variance_target": args.explained_variance,
        "diagnostics": diagnostics,
        "claim_scope": (
            "paired label-preserving acquisition-style residuals on exact "
            "released LLaVA-Med alignment-source CXR images"
        ),
    }
    atomic_json(args.output.with_suffix(args.output.suffix + ".meta.json"), metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

