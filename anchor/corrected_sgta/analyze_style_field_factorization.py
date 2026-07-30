"""Measure whether acquisition-style displacement is globally reusable.

For a displacement tensor ``Delta[case, style, dimension]``, the least-squares
optimal image-independent correction for each style is its case mean.  The
fraction of squared displacement explained by that projection is an exact
upper bound for the observed sample on any additive style-only correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


VERSION = "style-field-factorization-v1"
LAYERS = ("llm_27_image", "llm_27_prompt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def displacement_factorization(features: np.ndarray) -> dict[str, float]:
    """Return exact least-squares projections of the observed style field."""
    if features.ndim != 3 or features.shape[1] < 3:
        raise ValueError("features must have shape [case, real/null/style, dim]")
    delta = (
        features[:, 2:].astype(np.float64)
        - features[:, [0]].astype(np.float64)
    )
    total = float(np.square(delta).sum())
    if total <= 0:
        raise ValueError("style displacement has zero energy")
    global_mean = delta.mean(axis=(0, 1), keepdims=True)
    style_mean = delta.mean(axis=0, keepdims=True)
    case_mean = delta.mean(axis=1, keepdims=True)
    global_explained = float(
        np.square(np.broadcast_to(global_mean, delta.shape)).sum() / total
    )
    style_explained = float(
        np.square(np.broadcast_to(style_mean, delta.shape)).sum() / total
    )
    case_explained = float(
        np.square(np.broadcast_to(case_mean, delta.shape)).sum() / total
    )
    style_residual = float(np.square(delta - style_mean).sum() / total)
    return {
        "global_offset_explained": global_explained,
        "style_specific_offsets_explained": style_explained,
        "case_specific_offsets_explained": case_explained,
        "residual_after_optimal_style_offsets": style_residual,
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def analyze_files(paths: list[Path]) -> dict[str, Any]:
    lineages: dict[str, Any] = {}
    sources = []
    for path in paths:
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        data = np.load(path)
        variants = [row["name"] for row in metadata["variants"]]
        sources.append(
            {
                "features": str(path),
                "features_sha256": sha256(path),
                "metadata": str(metadata_path),
                "metadata_fingerprint": metadata["fingerprint"],
            }
        )
        for variant_index, variant in enumerate(variants):
            label = variant
            if label in lineages:
                label = f"{path.stem}:{variant}"
            lineages[label] = {
                layer: displacement_factorization(
                    data[layer][variant_index].astype(np.float32)
                )
                for layer in LAYERS
            }
    layer_summary = {}
    for layer in LAYERS:
        layer_summary[layer] = {
            metric: summarize(
                [values[layer][metric] for values in lineages.values()]
            )
            for metric in (
                "global_offset_explained",
                "style_specific_offsets_explained",
                "case_specific_offsets_explained",
                "residual_after_optimal_style_offsets",
            )
        }
    prompt = layer_summary["llm_27_prompt"]
    return {
        "version": VERSION,
        "sources": sources,
        "lineages": lineages,
        "layers": layer_summary,
        "decision": {
            "style_only_explains_below_10pct_in_every_lineage": (
                prompt["style_specific_offsets_explained"]["maximum"] < 0.10
            ),
            "over_90pct_remains_after_style_only_correction": (
                prompt["residual_after_optimal_style_offsets"]["minimum"] > 0.90
            ),
        },
        "interpretation": (
            "On these paired synthetic views, a global per-style additive "
            "correction cannot represent most late-fusion displacement; the "
            "field is predominantly image-conditioned."
        ),
        "claim_ceiling": (
            "exact finite-sample least-squares statement for fixed Fourier "
            "views on exposed MIMIC development images; not a natural-domain "
            "or clinical-utility claim"
        ),
    }


def plot_result(result: dict[str, Any], output: Path) -> None:
    labels = list(result["lineages"])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.9), sharey=True)
    colors = {
        "global_offset_explained": "#BDBDBD",
        "style_specific_offsets_explained": "#2166AC",
        "case_specific_offsets_explained": "#D6604D",
    }
    names = {
        "global_offset_explained": "Single global offset",
        "style_specific_offsets_explained": "Per-style offsets",
        "case_specific_offsets_explained": "Per-image offsets",
    }
    for axis, layer in zip(axes, LAYERS, strict=True):
        x = np.arange(len(labels))
        width = 0.24
        for index, metric in enumerate(colors):
            axis.bar(
                x + (index - 1) * width,
                [
                    result["lineages"][label][layer][metric] * 100
                    for label in labels
                ],
                width=width,
                color=colors[metric],
                label=names[metric],
            )
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=42, ha="right", fontsize=7)
        axis.set_title(
            "Final image tokens" if layer.endswith("image") else "Final prompt token"
        )
        axis.set_ylabel("Observed displacement explained (%)")
        axis.grid(axis="y", alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.suptitle(
        "Acquisition-style displacement is predominantly image-conditioned",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    paths = [path.expanduser().resolve() for path in args.features]
    for path in paths:
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise FileNotFoundError(path)
    result = analyze_files(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    plot_result(result, args.figure)
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
