#!/usr/bin/env python3
"""Outcome-blind channel geometry audit for high-bit chroma residual coding.

This script never loads images, labels, or a GPU.  It asks only whether the
first RGB patch projection of a frozen visual tower has non-degenerate gain on
the two-dimensional null space of a fixed linear-luma functional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


VERSION = "chroma-patch-channel-audit-v1"
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensor(model_dir: Path, key: str) -> tuple[np.ndarray, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard = model_dir / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        value = handle.get_tensor(key).float().numpy().astype(np.float64)
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"expected [out,3,h,w] patch kernel, got {value.shape}")
    return value, shard


def null_basis(vector: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(vector.reshape(1, -1), full_matrices=True)
    basis = vh[1:].T
    if not np.allclose(vector @ basis, 0.0, atol=1e-12):
        raise RuntimeError("failed to construct luma-null basis")
    return basis


def audit(name: str, model_dir: Path, key: str, std: np.ndarray) -> dict[str, object]:
    kernel, shard = load_tensor(model_dir, key)
    # A channel perturbation u is normalized before projection, so the
    # effective channel kernels are W_c / sigma_c.
    normalized = kernel / std.reshape(1, 3, 1, 1)
    flattened = normalized.transpose(1, 0, 2, 3).reshape(3, -1)
    gram = flattened @ flattened.T
    basis = null_basis(LUMA)
    restricted = basis.T @ gram @ basis
    eigenvalues, eigenvectors = np.linalg.eigh(restricted)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    directions = basis @ eigenvectors[:, order]
    gray = np.ones(3, dtype=np.float64)
    gray /= np.linalg.norm(gray)
    gray_gain_sq = float(gray @ gram @ gray)
    total_mean_gain_sq = float(np.trace(gram) / 3.0)
    records = []
    gray_effective = gray @ flattened
    for rank in range(2):
        direction = directions[:, rank]
        effective = direction @ flattened
        denominator = np.linalg.norm(effective) * np.linalg.norm(gray_effective)
        records.append(
            {
                "rank": rank + 1,
                "rgb_direction": direction.tolist(),
                "luma_inner_product": float(LUMA @ direction),
                "gain_squared": float(eigenvalues[rank]),
                "gain_over_gray": float(np.sqrt(eigenvalues[rank] / gray_gain_sq)),
                "gain_over_channel_rms": float(
                    np.sqrt(eigenvalues[rank] / total_mean_gain_sq)
                ),
                "effective_kernel_cosine_with_gray": (
                    float(effective @ gray_effective / denominator) if denominator else 0.0
                ),
            }
        )
    condition = float(eigenvalues[0] / max(eigenvalues[-1], 1e-30))
    return {
        "model": name,
        "model_dir": str(model_dir.resolve()),
        "patch_key": key,
        "patch_shape": list(kernel.shape),
        "checkpoint_shard": str(shard.resolve()),
        "checkpoint_shard_sha256": sha256(shard),
        "channel_std": std.tolist(),
        "normalized_channel_gram": gram.tolist(),
        "gray_gain_squared": gray_gain_sq,
        "chroma_restricted_condition_number": condition,
        "chroma_directions": records,
        "necessary_condition_passed": bool(eigenvalues[-1] > 1e-4 * gray_gain_sq),
        "guardrail": (
            "Non-degenerate first-layer gain proves visibility only; it does not prove "
            "clinical information, downstream use, or hallucination mitigation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    models = [
        (
            "huatuo",
            Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
            "model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight",
            np.asarray([0.26862954, 0.26130258, 0.27577711]),
        ),
        (
            "hulu",
            Path("/home/dbw/models/Hulu-Med-4B"),
            "model.vision_encoder.embeddings.patch_embedding.weight",
            np.asarray([0.5, 0.5, 0.5]),
        ),
    ]
    result = {
        "version": VERSION,
        "luma_functional": LUMA.tolist(),
        "models": [audit(*model) for model in models],
        "decision": "necessary_condition_only",
        "source_sha256": sha256(Path(__file__)),
        "command": " ".join(__import__("sys").argv),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
