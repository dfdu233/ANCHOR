"""Analyze layerwise acquisition-style orbits across controlled model lineages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


VERSION = "layerwise-style-orbit-analysis-v1"


def rms(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=axis))


def per_case_components(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return style drift, real-null leverage, and their ratio per case."""
    style_delta = features[:, 2:] - features[:, [0]]
    numerator = rms(style_delta, axis=(1, 2))
    denominator = rms(
        features[:, 0] - features[:, 1], axis=1
    )
    return (
        numerator,
        denominator,
        numerator / np.maximum(denominator, 1e-12),
    )


def per_case_susceptibility(features: np.ndarray) -> np.ndarray:
    """RMS style displacement normalized by the real-to-null displacement."""
    return per_case_components(features)[2]


def variance_decomposition(features: np.ndarray) -> dict[str, float]:
    """Balanced two-way ANOVA of same-image style displacement vectors."""
    delta = features[:, 2:].astype(np.float64) - features[:, [0]].astype(
        np.float64
    )
    grand = delta.mean(axis=(0, 1), keepdims=True)
    case = delta.mean(axis=1, keepdims=True) - grand
    style = delta.mean(axis=0, keepdims=True) - grand
    interaction = delta - grand - case - style
    sums = {
        "case": float(delta.shape[1] * np.square(case).sum()),
        "style": float(delta.shape[0] * np.square(style).sum()),
        "case_by_style": float(np.square(interaction).sum()),
    }
    total = sum(sums.values())
    return {name: value / max(total, 1e-18) for name, value in sums.items()}


def participation_rank(features: np.ndarray) -> float:
    delta = features[:, 2:].astype(np.float64) - features[:, [0]].astype(
        np.float64
    )
    matrix = delta.reshape(-1, delta.shape[-1])
    matrix -= matrix.mean(axis=0, keepdims=True)
    # Non-zero eigenvalues of X X^T and X^T X coincide.  The Gram form is
    # substantially cheaper because observations << hidden dimensions.
    eigenvalues = np.linalg.eigvalsh(matrix @ matrix.T)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return float(
        np.square(eigenvalues.sum())
        / max(np.square(eigenvalues).sum(), 1e-18)
    )


def cluster_bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    clusters: list[str],
    draws: int = 10_000,
    seed: int = 2027,
) -> tuple[float, list[float]]:
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("paired bootstrap inputs must be one-dimensional")
    unique = sorted(set(clusters))
    differences = first - second
    # The estimand is explicitly paired and patient-level: first average any
    # repeated images within a patient, then take the median patient effect.
    cluster_effect = np.asarray(
        [
            differences[
                np.asarray(
                    [
                        position
                        for position, value in enumerate(clusters)
                        if value == cluster
                    ]
                )
            ].mean()
            for cluster in unique
        ]
    )
    point = float(np.median(cluster_effect))
    generator = np.random.default_rng(seed)
    sampled = generator.integers(
        0, len(cluster_effect), size=(draws, len(cluster_effect))
    )
    values = np.median(cluster_effect[sampled], axis=1)
    return point, [float(value) for value in np.quantile(values, [0.025, 0.975])]


def cluster_bootstrap_relative_effect(
    first: np.ndarray,
    second: np.ndarray,
    clusters: list[str],
    draws: int = 10_000,
    seed: int = 2027,
) -> tuple[float, list[float]]:
    relative = (first - second) / np.maximum(np.abs(second), 1e-12)
    unique = sorted(set(clusters))
    cluster_effect = np.asarray(
        [
            relative[
                np.asarray(
                    [
                        position
                        for position, value in enumerate(clusters)
                        if value == cluster
                    ]
                )
            ].mean()
            for cluster in unique
        ]
    )
    point = float(np.median(cluster_effect))
    generator = np.random.default_rng(seed)
    sampled = generator.integers(
        0, len(cluster_effect), size=(draws, len(cluster_effect))
    )
    values = np.median(cluster_effect[sampled], axis=1)
    return point, [float(value) for value in np.quantile(values, [0.025, 0.975])]


def style_direction_split_half(
    features: np.ndarray,
    draws: int = 2_000,
    seed: int = 2027,
) -> dict[str, float | list[float]]:
    delta = features[:, 2:].astype(np.float64) - features[:, [0]].astype(
        np.float64
    )
    generator = np.random.default_rng(seed)
    flat = delta.reshape(delta.shape[0], -1)
    half = delta.shape[0] // 2
    values = []
    for start in range(0, draws, 128):
        count = min(128, draws - start)
        masks = np.zeros((count, delta.shape[0]), dtype=np.float64)
        for row in range(count):
            chosen = generator.choice(delta.shape[0], size=half, replace=False)
            masks[row, chosen] = 1.0
        left = masks @ flat / half
        right = (1.0 - masks) @ flat / (delta.shape[0] - half)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        values.extend(
            np.sum(left * right, axis=1) / np.maximum(denominator, 1e-18)
        )
    return {
        "median": float(np.median(values)),
        "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "fraction_positive": float(np.mean(np.asarray(values) > 0)),
    }


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear centered-kernel alignment between paired observation matrices."""
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    left -= left.mean(axis=0, keepdims=True)
    right -= right.mean(axis=0, keepdims=True)
    # The feature-space expression would materialize D x D matrices for the
    # 3,584-dimensional LLM states.  Its algebraically identical Gram form
    # needs only N x N matrices, where N is the paired case-view count.
    left_gram = left @ left.T
    right_gram = right @ right.T
    denominator = np.linalg.norm(left_gram) * np.linalg.norm(right_gram)
    return float(
        np.sum(left_gram * right_gram) / max(denominator, 1e-18)
    )


def evidence_matrix(path: Path, case_ids: list[str], views: list[str]) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    diseases = sorted({row["disease"] for row in rows})
    values = {
        (
            row["case_id"],
            row["view"],
            row["disease"],
            row["polarity"],
        ): -float(row["sequence_nll"])
        for row in rows
    }
    output = np.empty((len(case_ids), len(views), len(diseases)), dtype=np.float64)
    for case_index, case in enumerate(case_ids):
        for view_index, view in enumerate(views):
            for disease_index, disease in enumerate(diseases):
                output[case_index, view_index, disease_index] = (
                    values[(case, view, disease, "positive")]
                    - values[(case, view, disease, "negative")]
                )
    return output


def plot_summary(result: dict[str, Any], output: Path) -> None:
    layers = result["layers"]
    names = list(layers)
    variants = result["variants"]
    colors = {"base": "#666666", "matched": "#1874CD", "permuted": "#D95F02"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    x = np.arange(len(names))
    for variant in variants:
        color = colors.get(variant, None)
        axes[0].plot(
            x,
            [layers[layer]["variants"][variant]["median_kappa"] for layer in names],
            marker="o",
            label=variant,
            color=color,
        )
        axes[1].plot(
            x,
            [
                layers[layer]["variants"][variant]["variance_fraction"][
                    "style"
                ]
                for layer in names
            ],
            marker="o",
            label=variant,
            color=color,
        )
        axes[2].plot(
            x,
            [
                layers[layer]["variants"][variant]["output_drift_cka"]
                for layer in names
            ],
            marker="o",
            label=variant,
            color=color,
        )
    titles = [
        "Style susceptibility / visual leverage",
        "Reusable style fraction",
        "Alignment with clinical-evidence drift",
    ]
    ylabels = [r"$\kappa_\ell$", "Variance fraction", "Linear CKA"]
    for axis, title, ylabel in zip(axes, titles, ylabels, strict=True):
        axis.set_xticks(x)
        axis.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Where acquisition-style orbits survive in Qwen2.5-VL-7B",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism(result: dict[str, Any], output: Path) -> None:
    prompt_layers = [
        layer for layer in result["layers"] if layer.endswith("_prompt")
    ]
    x = np.arange(len(prompt_layers))
    blue = "#1874CD"
    orange = "#D95F02"
    teal = "#1B9E77"
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))

    axes[0].plot(
        x,
        [
            result["layers"][layer]["variants"]["matched"]["median_kappa"]
            for layer in prompt_layers
        ],
        marker="o",
        color=blue,
        label="Correctly matched",
    )
    axes[0].plot(
        x,
        [
            result["layers"][layer]["variants"]["permuted"]["median_kappa"]
            for layer in prompt_layers
        ],
        marker="o",
        color=orange,
        label="Image-permuted",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [layer.split("_")[1] for layer in prompt_layers]
    )
    axes[0].set_xlabel("Language layer")
    axes[0].set_ylabel(r"Style susceptibility $\kappa_\ell$")
    axes[0].set_title("Contraction emerges after fusion")
    axes[0].legend(frameon=False, fontsize=8)

    final = result["layers"][prompt_layers[-1]]
    components = [
        ("Style drift", final["relative_effects"]["style_drift"], blue),
        ("Visual leverage", final["relative_effects"]["real_null_leverage"], teal),
        ("Normalized $\\kappa$", final["relative_effects"]["kappa"], orange),
    ]
    points = np.asarray([item[1]["point"] for item in components]) * 100
    lower = np.asarray(
        [item[1]["patient_cluster_bootstrap_ci95"][0] for item in components]
    ) * 100
    upper = np.asarray(
        [item[1]["patient_cluster_bootstrap_ci95"][1] for item in components]
    ) * 100
    axes[1].bar(
        np.arange(3),
        points,
        color=[item[2] for item in components],
        width=0.66,
        alpha=0.9,
    )
    axes[1].errorbar(
        np.arange(3),
        points,
        yerr=np.vstack([points - lower, upper - points]),
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(3))
    axes[1].set_xticklabels([item[0] for item in components], rotation=18)
    axes[1].set_ylabel("Matched vs permuted (%)")
    axes[1].set_title("Paired associations at the final layer")

    readouts = [
        ("Final hidden\nstate", final["relative_effects"]["kappa"], blue),
        (
            "Clinical sentence\nevidence",
            result["output_evidence"]["relative_kappa_effect"],
            orange,
        ),
    ]
    points = np.asarray([item[1]["point"] for item in readouts]) * 100
    lower = np.asarray(
        [item[1]["patient_cluster_bootstrap_ci95"][0] for item in readouts]
    ) * 100
    upper = np.asarray(
        [item[1]["patient_cluster_bootstrap_ci95"][1] for item in readouts]
    ) * 100
    axes[2].bar(
        np.arange(2),
        points,
        color=[item[2] for item in readouts],
        width=0.58,
        alpha=0.9,
    )
    axes[2].errorbar(
        np.arange(2),
        points,
        yerr=np.vstack([points - lower, upper - points]),
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xticks(np.arange(2))
    axes[2].set_xticklabels([item[0] for item in readouts])
    axes[2].set_ylabel("Matched vs permuted (%)")
    axes[2].set_title("Hidden-state effect is absent at output")

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(
        "Matched and image-permuted lineages diverge only after late fusion",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--base-evidence", type=Path, required=True)
    parser.add_argument("--matched-evidence", type=Path, required=True)
    parser.add_argument("--permuted-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--mechanism-figure", type=Path)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    variants = [row["name"] for row in metadata["variants"]]
    if variants != ["base", "matched", "permuted"]:
        raise ValueError(
            "analysis expects variants in exact order: base, matched, permuted"
        )
    case_ids = []
    patients = []
    for row in metadata["rows"]:
        if row["view"] == "real":
            case_ids.append(row["case_id"])
            patients.append(row["patient_id"])
    views = metadata["views"]
    style_views = views[2:]
    evidence_paths = {
        "base": args.base_evidence,
        "matched": args.matched_evidence,
        "permuted": args.permuted_evidence,
    }
    evidence = {
        name: evidence_matrix(path, case_ids, views)
        for name, path in evidence_paths.items()
    }
    data = np.load(args.features)
    layer_results: dict[str, Any] = {}
    for layer in data.files:
        tensor = data[layer].astype(np.float32)
        layer_result: dict[str, Any] = {"variants": {}}
        for variant_index, variant in enumerate(variants):
            features = tensor[variant_index]
            style_drift, visual_leverage, susceptibility = per_case_components(
                features
            )
            representation_delta = (
                features[:, 2:].astype(np.float64)
                - features[:, [0]].astype(np.float64)
            ).reshape(-1, features.shape[-1])
            evidence_delta = (
                evidence[variant][:, 2:] - evidence[variant][:, [0]]
            ).reshape(-1, evidence[variant].shape[-1])
            layer_result["variants"][variant] = {
                "median_kappa": float(np.median(susceptibility)),
                "mean_kappa": float(np.mean(susceptibility)),
                "per_case_kappa": susceptibility.tolist(),
                "median_style_drift": float(np.median(style_drift)),
                "per_case_style_drift": style_drift.tolist(),
                "median_real_null_leverage": float(np.median(visual_leverage)),
                "per_case_real_null_leverage": visual_leverage.tolist(),
                "variance_fraction": variance_decomposition(features),
                "style_orbit_participation_rank": participation_rank(features),
                "split_half_style_direction": style_direction_split_half(features),
                "output_drift_cka": linear_cka(
                    representation_delta, evidence_delta
                ),
            }
        point, interval = cluster_bootstrap_difference(
            np.asarray(
                layer_result["variants"]["matched"]["per_case_kappa"]
            ),
            np.asarray(
                layer_result["variants"]["permuted"]["per_case_kappa"]
            ),
            patients,
        )
        layer_result["matched_minus_permuted_kappa"] = {
            "point": point,
            "patient_cluster_bootstrap_ci95": interval,
        }
        layer_result["relative_effects"] = {}
        point, interval = cluster_bootstrap_relative_effect(
            np.asarray(layer_result["variants"]["matched"]["per_case_kappa"]),
            np.asarray(layer_result["variants"]["permuted"]["per_case_kappa"]),
            patients,
        )
        layer_result["relative_effects"]["kappa"] = {
            "point": point,
            "patient_cluster_bootstrap_ci95": interval,
        }
        for quantity, key in (
            ("style_drift", "per_case_style_drift"),
            ("real_null_leverage", "per_case_real_null_leverage"),
        ):
            point, interval = cluster_bootstrap_difference(
                np.asarray(layer_result["variants"]["matched"][key]),
                np.asarray(layer_result["variants"]["permuted"][key]),
                patients,
            )
            layer_result[f"matched_minus_permuted_{quantity}"] = {
                "point": point,
                "patient_cluster_bootstrap_ci95": interval,
            }
            point, interval = cluster_bootstrap_relative_effect(
                np.asarray(layer_result["variants"]["matched"][key]),
                np.asarray(layer_result["variants"]["permuted"][key]),
                patients,
            )
            layer_result["relative_effects"][quantity] = {
                "point": point,
                "patient_cluster_bootstrap_ci95": interval,
            }
        layer_results[layer] = layer_result

    preferred_order = [
        "vision_block_0",
        "vision_block_7",
        "vision_block_15",
        "vision_block_23",
        "vision_block_31",
        "merger",
        "llm_0_image",
        "llm_0_prompt",
        "llm_7_image",
        "llm_7_prompt",
        "llm_14_image",
        "llm_14_prompt",
        "llm_21_image",
        "llm_21_prompt",
        "llm_27_image",
        "llm_27_prompt",
    ]
    layer_results = {
        layer: layer_results[layer]
        for layer in preferred_order
        if layer in layer_results
    }
    significant = [
        layer
        for layer, values in layer_results.items()
        if (
            values["matched_minus_permuted_kappa"][
                "patient_cluster_bootstrap_ci95"
            ][1]
            < 0
        )
    ]
    output_kappa = {
        variant: per_case_susceptibility(evidence[variant])
        for variant in variants
    }
    output_point, output_interval = cluster_bootstrap_relative_effect(
        output_kappa["matched"], output_kappa["permuted"], patients
    )
    output = {
        "version": VERSION,
        "features": str(args.features.resolve()),
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "metadata_fingerprint": metadata["fingerprint"],
        "variants": variants,
        "cases": len(case_ids),
        "patients": len(set(patients)),
        "views": views,
        "styles": style_views,
        "definition": {
            "kappa": (
                "per-case coordinate-RMS style displacement divided by "
                "real-to-null representation displacement"
            ),
            "variance": (
                "balanced two-way ANOVA over same-image style displacement"
            ),
            "cka": (
                "linear CKA between paired layer displacements and "
                "complete-sentence clinical-evidence displacements"
            ),
        },
        "layers": layer_results,
        "output_evidence": {
            "median_kappa": {
                variant: float(np.median(values))
                for variant, values in output_kappa.items()
            },
            "relative_kappa_effect": {
                "point": output_point,
                "patient_cluster_bootstrap_ci95": output_interval,
            },
        },
        "decision": {
            "layers_with_certified_matched_contraction": significant,
            "alignment_contraction_detected": bool(significant),
        },
        "claim_ceiling": (
            "paired layerwise mechanism diagnostic on exposed MIMIC "
            "development images; it does not establish hallucination "
            "mitigation or general domain generalization"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    plot_summary(output, args.figure)
    if args.mechanism_figure:
        plot_mechanism(output, args.mechanism_figure)
    print(
        json.dumps(
            {
                "decision": output["decision"],
                "output": str(args.output),
                "figure": str(args.figure),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
