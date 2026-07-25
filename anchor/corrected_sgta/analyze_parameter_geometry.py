"""Audit what source information is identifiable from frozen VLM parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from corrected_sgta.source_bank_v2 import sha256_file


MODEL_ROOTS = {
    "llava": Path("/root/autodl-tmp/LLaVA-Med/microsoft/llava-med-v1.5-mistral-7b"),
    "hulu": Path(
        "/root/autodl-tmp/Hulu-Med/MedUniEval/datas/hub/"
        "models--ZJU-AI4H--Hulu-Med-14B/snapshots/"
        "b30d9161b8c23a79e20e1eca3891f63697531904"
    ),
}

PROJECTOR_KEYS = {
    "llava": (
        "model.mm_projector.0.weight",
        "model.mm_projector.0.bias",
        "model.mm_projector.2.weight",
        "model.mm_projector.2.bias",
    ),
    "hulu": (
        "model.mm_projector.readout.0.weight",
        "model.mm_projector.readout.0.bias",
        "model.mm_projector.readout.2.weight",
        "model.mm_projector.readout.2.bias",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("llava", "hulu"), required=True)
    parser.add_argument("--visual-centers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_selected(root: Path, keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    index_path = root / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    tensors = {}
    for key in keys:
        with safe_open(root / weight_map[key], framework="pt", device="cpu") as handle:
            tensors[key] = handle.get_tensor(key).float()
    return tensors


def spectral_summary(weight: torch.Tensor, iterations: int = 30) -> dict:
    generator = torch.Generator().manual_seed(42)
    vector = torch.randn(weight.shape[1], generator=generator)
    vector /= vector.norm()
    for _ in range(iterations):
        left = weight @ vector
        left /= left.norm().clamp_min(1e-12)
        vector = weight.T @ left
        vector /= vector.norm().clamp_min(1e-12)
    spectral = float((weight @ vector).norm())
    frobenius = float(weight.norm())
    return {
        "shape": list(weight.shape),
        "spectral_norm_power_iteration": spectral,
        "frobenius_norm": frobenius,
        "stable_rank_estimate": (frobenius / max(spectral, 1e-12)) ** 2,
    }


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(1.0 - np.dot(left, right) / max(denominator, 1e-12))


def main() -> None:
    args = parse_args()
    root = MODEL_ROOTS[args.model]
    keys = PROJECTOR_KEYS[args.model]
    tensors = load_selected(root, keys)
    first_weight, first_bias, second_weight, second_bias = [tensors[key] for key in keys]
    zero_anchor = second_weight @ torch.nn.functional.gelu(first_bias) + second_bias
    arrays = np.load(args.visual_centers, allow_pickle=False)
    metadata_path = args.visual_centers.with_suffix(args.visual_centers.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    centers = {
        entry["source_id"]: arrays[entry["array_key"]].astype(np.float64)
        for entry in metadata["entries"]
    }
    anchor = zero_anchor.numpy().astype(np.float64)
    pairwise = []
    source_ids = sorted(centers)
    for index, left_id in enumerate(source_ids):
        for right_id in source_ids[index + 1 :]:
            pairwise.append(
                {
                    "left": left_id,
                    "right": right_id,
                    "cosine_distance": cosine_distance(
                        centers[left_id], centers[right_id]
                    ),
                }
            )
    weight_map = json.loads(
        (root / "model.safetensors.index.json").read_text()
    )["weight_map"]
    stored_population_statistics = sorted(
        key
        for key in weight_map
        if any(
            marker in key
            for marker in ("running_mean", "running_var", "num_batches_tracked")
        )
    )
    result = {
        "version": "parameter-native-geometry-audit-v1",
        "model": args.model,
        "model_root": str(root),
        "checkpoint_index_sha256": sha256_file(
            root / "model.safetensors.index.json"
        ),
        "visual_centers": str(args.visual_centers.resolve()),
        "visual_centers_sha256": sha256_file(args.visual_centers),
        "normalization_audit": {
            "stored_population_statistic_count": len(stored_population_statistics),
            "interpretation": (
                "No BatchNorm running moments are stored; LayerNorm affine "
                "parameters do not identify a training-distribution center."
            ),
        },
        "projector": {
            "architecture": "Linear-GELU-Linear",
            "first_linear": spectral_summary(first_weight),
            "second_linear": spectral_summary(second_weight),
            "zero_input_anchor_norm": float(zero_anchor.norm()),
            "zero_input_anchor_definition": "W2 GELU(b1) + b2",
            "interpretation": (
                "The anchor is the projector output for a zero vision feature. "
                "It is parameter-identifiable but is a visual-null anchor, not "
                "an identified source-data mean."
            ),
        },
        "anchor_to_source_centers": {
            source_id: {
                "cosine_distance": cosine_distance(anchor, center),
                "anchor_to_center_l2": float(np.linalg.norm(anchor - center)),
                "center_norm": float(np.linalg.norm(center)),
                "anchor_norm_over_center_norm": float(
                    np.linalg.norm(anchor) / max(np.linalg.norm(center), 1e-12)
                ),
            }
            for source_id, center in sorted(centers.items())
        },
        "source_center_pairwise": pairwise,
        "mean_pairwise_source_cosine_distance": float(
            np.mean([item["cosine_distance"] for item in pairwise])
        ),
        "non_claims": [
            "The zero-input anchor is not claimed to equal the VLM training mean.",
            "Projector singular geometry identifies transmitted directions, not a data density.",
            "Moving toward the anchor is visual-evidence shrinkage, not source alignment.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
