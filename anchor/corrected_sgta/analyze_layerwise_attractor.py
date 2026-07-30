"""Trace case-anchored style drift through language-model layers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from anchor.corrected_sgta.analyze_conditional_style_field import (
    endpoint_projection_controls,
    null_prior_projection,
    patient_ids,
    sha256,
)


VERSION = "layerwise-attractor-v1"
LAYER_INDICES = (0, 7, 14, 21, 27)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def centroid_distance_controls(
    features: np.ndarray, patients: list[str], eps: float = 1e-12
) -> dict[str, float]:
    """Measure whether styled states become closer to a patient-LOO centroid.

    Positive projection onto the centroid direction is not sufficient for
    contraction: a large orthogonal displacement can still increase the total
    distance.  The log distance ratio directly distinguishes those cases.
    """
    if features.ndim != 3 or features.shape[1] < 3:
        raise ValueError("features must have shape [case, real/null/style, dim]")
    if features.shape[0] != len(patients):
        raise ValueError("patient count does not match feature cases")
    real = features[:, 0].astype(np.float64)
    styled = features[:, 2:].astype(np.float64)
    log_ratios = []
    projection_coefficients = []
    absolute_projection_energy = []
    for case_index, patient in enumerate(patients):
        eligible = np.asarray(
            [other != patient for other in patients], dtype=bool
        )
        if not eligible.any():
            raise ValueError("patient-LOO centroid has no eligible cases")
        centroid = real[eligible].mean(axis=0)
        direction = centroid - real[case_index]
        before_sq = float(np.dot(direction, direction))
        deltas = styled[case_index] - real[case_index]
        after_sq = np.square(styled[case_index] - centroid).sum(axis=1)
        log_ratios.extend(
            np.log((after_sq + eps) / (before_sq + eps)).tolist()
        )
        direction_sq = before_sq + eps
        coefficients = (deltas @ direction) / direction_sq
        projection_coefficients.extend(coefficients.tolist())
        projected_sq = np.square(deltas @ direction) / direction_sq
        absolute_projection_energy.extend(projected_sq.tolist())
    ratio_array = np.asarray(log_ratios, dtype=np.float64)
    coefficient_array = np.asarray(
        projection_coefficients, dtype=np.float64
    )
    projection_array = np.asarray(
        absolute_projection_energy, dtype=np.float64
    )
    return {
        "mean_log_centroid_distance_ratio": float(ratio_array.mean()),
        "median_log_centroid_distance_ratio": float(np.median(ratio_array)),
        "fraction_closer_to_centroid": float((ratio_array < 0).mean()),
        "mean_centroid_projection_coefficient": float(
            coefficient_array.mean()
        ),
        "median_centroid_projection_coefficient": float(
            np.median(coefficient_array)
        ),
        "mean_absolute_centroid_projection_energy": float(
            projection_array.mean()
        ),
    }


def analyze_files(
    paths: list[Path], permutation_repeats: int = 200
) -> dict[str, Any]:
    lineages: dict[str, Any] = {}
    sources = []
    for path_index, path in enumerate(paths):
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        patients = patient_ids(metadata)
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
            label = variant if variant not in lineages else f"{path.stem}:{variant}"
            lineages[label] = {}
            for layer_index in LAYER_INDICES:
                lineages[label][str(layer_index)] = {}
                for token_type in ("image", "prompt"):
                    layer = f"llm_{layer_index}_{token_type}"
                    null_metrics = null_prior_projection(
                            data[layer][variant_index].astype(np.float32),
                            seed=2027 + 101 * path_index + 7 * variant_index
                            + layer_index,
                            permutation_repeats=permutation_repeats,
                    )
                    endpoint_metrics = endpoint_projection_controls(
                        data[layer][variant_index].astype(np.float32),
                        patients,
                    )
                    distance_metrics = centroid_distance_controls(
                        data[layer][variant_index].astype(np.float32),
                        patients,
                    )
                    lineages[label][str(layer_index)][token_type] = {
                        **null_metrics,
                        **endpoint_metrics,
                        **distance_metrics,
                    }
    layer_summary = {}
    for layer_index in LAYER_INDICES:
        layer_summary[str(layer_index)] = {}
        for token_type in ("image", "prompt"):
            layer_summary[str(layer_index)][token_type] = {
                metric: summarize(
                    [
                        lineage[str(layer_index)][token_type][metric]
                        for lineage in lineages.values()
                    ]
                )
                for metric in (
                    "same_case_null_projection_energy",
                    "clean_centroid_projection_energy",
                    "fraction_toward_null",
                    "fraction_toward_clean_centroid",
                    "permuted_projection_median",
                    "mean_log_centroid_distance_ratio",
                    "median_log_centroid_distance_ratio",
                    "fraction_closer_to_centroid",
                    "mean_centroid_projection_coefficient",
                    "mean_absolute_centroid_projection_energy",
                )
            }
    prompt_control = {
        str(layer_index): all(
            lineage[str(layer_index)]["prompt"][
                "same_case_null_projection_energy"
            ]
            > lineage[str(layer_index)]["prompt"][
                "permuted_projection_95pct"
            ][1]
            for lineage in lineages.values()
        )
        for layer_index in LAYER_INDICES
    }
    return {
        "version": VERSION,
        "sources": sources,
        "lineages": lineages,
        "layers": layer_summary,
        "decision": {
            "prompt_alignment_exceeds_permuted_control_by_layer": prompt_control,
            "not_identifiable_at_layer_0": not prompt_control["0"],
            "identifiable_from_layer_7_onward": all(
                prompt_control[str(index)] for index in LAYER_INDICES[1:]
            ),
            "final_prompt_exceeds_initial_prompt_every_lineage": all(
                lineage["27"]["prompt"]["same_case_null_projection_energy"]
                > lineage["0"]["prompt"]["same_case_null_projection_energy"]
                for lineage in lineages.values()
            ),
            "prompt_exceeds_image_at_every_sampled_layer_and_lineage": all(
                lineage[str(index)]["prompt"][
                    "same_case_null_projection_energy"
                ]
                > lineage[str(index)]["image"][
                    "same_case_null_projection_energy"
                ]
                for lineage in lineages.values()
                for index in LAYER_INDICES
            ),
            "clean_centroid_exceeds_null_at_every_layer_and_lineage": all(
                lineage[str(index)]["prompt"][
                    "clean_centroid_projection_energy"
                ]
                > lineage[str(index)]["prompt"][
                    "same_case_null_projection_energy"
                ]
                for lineage in lineages.values()
                for index in LAYER_INDICES
            ),
            "normalized_prompt_centroid_alignment_exceeds_image_tokens": all(
                lineage[str(index)]["prompt"][
                    "clean_centroid_projection_energy"
                ]
                > lineage[str(index)]["image"][
                    "clean_centroid_projection_energy"
                ]
                for lineage in lineages.values()
                for index in LAYER_INDICES
            ),
            "prompt_distance_increases_layers_0_7_14_every_lineage": all(
                lineage[str(index)]["prompt"][
                    "mean_log_centroid_distance_ratio"
                ]
                > 0
                for lineage in lineages.values()
                for index in LAYER_INDICES[:3]
            ),
            "prompt_distance_decreases_layers_21_27_every_lineage": all(
                lineage[str(index)]["prompt"][
                    "mean_log_centroid_distance_ratio"
                ]
                < 0
                for lineage in lineages.values()
                for index in LAYER_INDICES[3:]
            ),
        },
        "interpretation": (
            "On the fixed synthetic probe, style displacement contains a "
            "consistent component aligned toward a patient-LOO clean-state "
            "centroid, stronger than null-state alignment. Positive alignment "
            "does not imply contraction: prompt states move farther from the "
            "centroid at layers 0, 7, and 14, and become closer only at layers "
            "21 and 27 in every lineage. The nonmonotonic late-layer onset "
            "supports a descriptive contraction dynamic, not a null-specific "
            "clinical prior or a causal language-fusion claim."
        ),
        "claim_ceiling": (
            "layerwise pooled-state diagnostic on 40 exposed MIMIC "
            "development images, six fixed Fourier views, and correlated "
            "Qwen2.5-VL-7B lineages; not a causal or output-utility claim"
        ),
    }


def plot_result(result: dict[str, Any], output: Path) -> None:
    x = np.asarray(LAYER_INDICES)
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))

    for lineage in result["lineages"].values():
        values = [
            lineage[str(index)]["prompt"][
                "mean_log_centroid_distance_ratio"
            ]
            for index in LAYER_INDICES
        ]
        axes[0].plot(x, values, color="#2166AC", alpha=0.18, linewidth=0.9)
    distance_median = [
        result["layers"][str(index)]["prompt"][
            "mean_log_centroid_distance_ratio"
        ]["median"]
        for index in LAYER_INDICES
    ]
    axes[0].plot(
        x,
        distance_median,
        color="#2166AC",
        marker="o",
        linewidth=2.4,
        label="Median across lineages",
    )
    axes[0].axhline(0, color="#4D4D4D", linewidth=1.0, linestyle="--")
    axes[0].fill_between(
        [20.5, 27.5], -0.2, 0, color="#67A9CF", alpha=0.12
    )
    axes[0].set_title("True contraction starts only in late layers")
    axes[0].set_ylabel(
        r"Mean log distance ratio to LOO centroid"
    )
    axes[0].legend(frameon=False, fontsize=8)

    null_median = [
        result["layers"][str(index)]["prompt"][
            "same_case_null_projection_energy"
        ]["median"]
        * 100
        for index in LAYER_INDICES
    ]
    centroid_median = [
        result["layers"][str(index)]["prompt"][
            "clean_centroid_projection_energy"
        ]["median"]
        * 100
        for index in LAYER_INDICES
    ]
    axes[1].plot(
        x,
        centroid_median,
        color="#B2182B",
        marker="^",
        linewidth=2.4,
        label="LOO clean centroid",
    )
    axes[1].plot(
        x,
        null_median,
        color="#E08214",
        marker="o",
        linewidth=2.0,
        label="Null endpoint",
    )
    axes[1].set_title("Centroid alignment exceeds null alignment")
    axes[1].set_ylabel("Projected displacement energy (%)")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xlabel("Language-model layer")
        axis.set_xticks(x)
        axis.grid(alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        "Acquisition-style drift becomes centroid-contracting late",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--permutation-repeats", type=int, default=200)
    args = parser.parse_args()
    paths = [path.expanduser().resolve() for path in args.features]
    for path in paths:
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise FileNotFoundError(path)
    result = analyze_files(paths, args.permutation_repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    plot_result(result, args.figure)
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
