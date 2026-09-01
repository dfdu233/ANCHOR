#!/usr/bin/env python3
"""Build model-visible radial banks for public CXR proxy domains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from corrected_sgta.audit_public_domain_hypotheses import public_domain_paths
from corrected_sgta.mosec import model_visible_image, radial_log_amplitude


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-domain", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    paths_by_domain = public_domain_paths(
        args.repo_root.resolve(), args.samples_per_domain, args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "version": "ssrt-public-cxr-radial-banks-v1",
        "descriptor": "64-bin radial median log Fourier amplitude",
        "preprocess": "Huatuo CLIP-mean square pad then bicubic 336x336",
        "seed": args.seed,
        "domains": {},
        "limitations": [
            "Public dataset identity is a proxy for institution and export pipeline.",
            "CheXpert subset provenance is unverified.",
            "These banks are controls and shift donors, not Huatuo training-source banks.",
        ],
    }
    manifest_path = args.output_dir / "selected_images.jsonl"
    with manifest_path.open("w") as manifest:
        for domain, paths in paths_by_domain.items():
            descriptors = []
            errors = []
            for path in paths:
                try:
                    with Image.open(path) as source:
                        visible = model_visible_image(source.convert("RGB"))
                    descriptors.append(radial_log_amplitude(visible))
                    manifest.write(
                        json.dumps(
                            {
                                "domain": domain,
                                "path": str(path),
                                "sha256": file_sha256(path),
                            }
                        )
                        + "\n"
                    )
                except Exception as exc:
                    errors.append({"path": str(path), "error": repr(exc)})
            if not descriptors:
                raise RuntimeError(f"no readable images for {domain}")
            values = np.stack(descriptors).astype(np.float64)
            median = np.median(values, axis=0)
            scale = np.maximum(
                1.4826 * np.median(np.abs(values - median), axis=0), 1e-4
            )
            bank_path = args.output_dir / f"{domain}_radial_bank.npz"
            np.savez_compressed(
                bank_path,
                mean=values.mean(axis=0).astype(np.float32),
                median=median.astype(np.float32),
                scale=scale.astype(np.float32),
                lower=np.quantile(values, 0.05, axis=0).astype(np.float32),
                upper=np.quantile(values, 0.95, axis=0).astype(np.float32),
            )
            metadata["domains"][domain] = {
                "n": len(descriptors),
                "errors": errors,
                "bank": str(bank_path),
                "bank_sha256": file_sha256(bank_path),
            }
            print(json.dumps({"domain": domain, "n": len(descriptors)}), flush=True)

    metadata["manifest"] = str(manifest_path)
    metadata["manifest_sha256"] = file_sha256(manifest_path)
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
