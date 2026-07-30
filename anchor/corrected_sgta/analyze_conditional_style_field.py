"""Audit whether a medical VLM's acquisition-style field is predictable.

The paired displacement tensor

    Delta[i, s] = h(T_s x_i) - h(x_i)

can be decomposed exactly into grand, case, style, and interaction terms.  A
dominant case term is not by itself deployable: at inference time the other
style views are unavailable.  This module therefore reports both a crossed
cell-completion ceiling and a patient-held-out kernel-ridge predictor from the
unmodified image state.

The analysis is a mechanism diagnostic.  It neither changes predictions nor
uses target answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


VERSION = "conditional-style-field-v1"
LAYERS = ("llm_27_image", "llm_27_prompt")
RIDGE_MULTIPLIERS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_ids(metadata: dict[str, Any]) -> list[str]:
    """Return one patient identifier per case in activation-array order."""
    result = []
    seen = set()
    for row in metadata["rows"]:
        case_id = row["case_id"]
        if case_id in seen:
            continue
        seen.add(case_id)
        result.append(str(row["patient_id"]))
    expected = int(metadata["cases"])
    if len(result) != expected:
        raise ValueError(f"expected {expected} cases, found {len(result)}")
    return result


def displacement_tensor(features: np.ndarray) -> np.ndarray:
    if features.ndim != 3 or features.shape[1] < 3:
        raise ValueError("features must have shape [case, real/null/style, dim]")
    return (
        features[:, 2:].astype(np.float64)
        - features[:, [0]].astype(np.float64)
    )


def anova_factorization(delta: np.ndarray) -> dict[str, float]:
    """Return the exact balanced two-way ANOVA energy decomposition."""
    total = float(np.square(delta).sum())
    if total <= 0:
        raise ValueError("style displacement has zero energy")
    grand = delta.mean(axis=(0, 1), keepdims=True)
    case = delta.mean(axis=1, keepdims=True) - grand
    style = delta.mean(axis=0, keepdims=True) - grand
    interaction = delta - grand - case - style
    components = {
        "grand": np.broadcast_to(grand, delta.shape),
        "case": np.broadcast_to(case, delta.shape),
        "style": np.broadcast_to(style, delta.shape),
        "interaction": interaction,
    }
    ratios = {
        f"{name}_energy_fraction": float(np.square(value).sum() / total)
        for name, value in components.items()
    }
    ratios["sum"] = float(sum(ratios.values()))
    return ratios


def crossed_cell_prediction(
    delta: np.ndarray, patients: list[str]
) -> dict[str, float]:
    """Predict each case-style cell without using that cell.

    The case term uses the same image under the other styles.  The style and
    grand terms exclude every case from the held-out patient.  This is a
    multi-view structural ceiling, not a single-view deployable estimator.
    """
    n_cases, n_styles, _ = delta.shape
    if len(patients) != n_cases:
        raise ValueError("patient count does not match displacement tensor")
    prediction = np.empty_like(delta)
    case_only = np.empty_like(delta)
    style_only = np.empty_like(delta)
    patient_array = np.asarray(patients)
    for case_index in range(n_cases):
        train_cases = patient_array != patients[case_index]
        if int(train_cases.sum()) < 2:
            raise ValueError("too few patient-disjoint cases")
        for style_index in range(n_styles):
            other_styles = np.arange(n_styles) != style_index
            case_mean = delta[case_index, other_styles].mean(axis=0)
            style_mean = delta[train_cases, style_index].mean(axis=0)
            grand_other = delta[train_cases][:, other_styles].mean(axis=(0, 1))
            prediction[case_index, style_index] = (
                case_mean + style_mean - grand_other
            )
            case_only[case_index, style_index] = case_mean
            style_only[case_index, style_index] = style_mean
    total = float(np.square(delta).sum())
    return {
        "crossed_cell_r2_zero": float(
            1.0 - np.square(delta - prediction).sum() / total
        ),
        "case_other_styles_r2_zero": float(
            1.0 - np.square(delta - case_only).sum() / total
        ),
        "patient_loo_style_r2_zero": float(
            1.0 - np.square(delta - style_only).sum() / total
        ),
    }


def group_folds(
    groups: list[str], n_splits: int, seed: int
) -> list[np.ndarray]:
    unique = np.asarray(sorted(set(groups)))
    if len(unique) < n_splits:
        raise ValueError("fewer groups than folds")
    shuffled = np.random.default_rng(seed).permutation(unique)
    buckets = [set(shuffled[index::n_splits]) for index in range(n_splits)]
    array = np.asarray(groups)
    return [np.asarray([value in bucket for value in array]) for bucket in buckets]


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def kernel_ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """Linear-kernel ridge with a fitted intercept and scale-free penalty."""
    mean_x = train_x.mean(axis=0, keepdims=True)
    centered_train = train_x - mean_x
    centered_test = test_x - mean_x
    kernel = centered_train @ centered_train.T
    test_kernel = centered_test @ centered_train.T
    scale = max(float(np.trace(kernel) / len(train_x)), 1e-12)
    mean_y = train_y.mean(axis=0, keepdims=True)
    coefficients = np.linalg.solve(
        kernel + multiplier * scale * np.eye(len(train_x)),
        train_y - mean_y,
    )
    return mean_y + test_kernel @ coefficients


def select_ridge_multiplier(
    inputs: np.ndarray,
    targets: np.ndarray,
    groups: list[str],
    seed: int,
) -> float:
    folds = group_folds(groups, min(4, len(set(groups))), seed)
    losses = []
    for multiplier in RIDGE_MULTIPLIERS:
        squared_error = 0.0
        for validation in folds:
            train = ~validation
            prediction = kernel_ridge_predict(
                inputs[train], targets[train], inputs[validation], multiplier
            )
            squared_error += float(
                np.square(targets[validation] - prediction).sum()
            )
        losses.append(squared_error)
    return float(RIDGE_MULTIPLIERS[int(np.argmin(losses))])


def patient_cv_predictability(
    features: np.ndarray,
    patients: list[str],
    seed: int = 2027,
    permutation_repeats: int = 20,
) -> dict[str, Any]:
    """Predict a case-conditioned style field from the clean image state."""
    delta = displacement_tensor(features)
    inputs = _normalized_rows(features[:, 0].astype(np.float64))
    case_targets = delta.mean(axis=1)
    folds = group_folds(patients, 5, seed)
    prediction = np.empty_like(delta)
    style_prediction = np.empty_like(delta)
    chosen = []
    fold_records = []
    for fold_index, test in enumerate(folds):
        train = ~test
        train_groups = list(np.asarray(patients)[train])
        multiplier = select_ridge_multiplier(
            inputs[train],
            case_targets[train],
            train_groups,
            seed + fold_index + 1,
        )
        chosen.append(multiplier)
        predicted_case = kernel_ridge_predict(
            inputs[train], case_targets[train], inputs[test], multiplier
        )
        residual_style = (
            delta[train] - case_targets[train, None, :]
        ).mean(axis=0)
        prediction[test] = predicted_case[:, None, :] + residual_style[None, :, :]
        style_prediction[test] = delta[train].mean(axis=0, keepdims=True)
        fold_records.append(
            {
                "fold": fold_index,
                "test_patients": sorted(set(np.asarray(patients)[test])),
                "ridge_multiplier": multiplier,
            }
        )
    total = float(np.square(delta).sum())
    conditional_error = float(np.square(delta - prediction).sum())
    style_error = float(np.square(delta - style_prediction).sum())

    rng = np.random.default_rng(seed + 991)
    permutation_incremental = []
    for _ in range(permutation_repeats):
        permuted_prediction = np.empty_like(delta)
        for fold_index, test in enumerate(folds):
            train = ~test
            permuted_targets = case_targets[train][
                rng.permutation(int(train.sum()))
            ]
            predicted_case = kernel_ridge_predict(
                inputs[train],
                permuted_targets,
                inputs[test],
                chosen[fold_index],
            )
            residual_style = (
                delta[train] - case_targets[train, None, :]
            ).mean(axis=0)
            permuted_prediction[test] = (
                predicted_case[:, None, :] + residual_style[None, :, :]
            )
        permuted_error = float(np.square(delta - permuted_prediction).sum())
        permutation_incremental.append(1.0 - permuted_error / style_error)
    return {
        "patient_cv_conditional_r2_zero": float(1.0 - conditional_error / total),
        "patient_cv_style_only_r2_zero": float(1.0 - style_error / total),
        "patient_cv_incremental_over_style_only": float(
            1.0 - conditional_error / style_error
        ),
        "permuted_incremental_median": float(
            np.median(permutation_incremental)
        ),
        "permuted_incremental_95pct": [
            float(value)
            for value in np.quantile(permutation_incremental, [0.025, 0.975])
        ],
        "folds": fold_records,
    }


def null_prior_projection(
    features: np.ndarray,
    seed: int = 2027,
    permutation_repeats: int = 200,
) -> dict[str, Any]:
    """Measure drift toward the same-case null-image state."""
    delta = displacement_tensor(features)
    direction = (
        features[:, 1].astype(np.float64)
        - features[:, 0].astype(np.float64)
    )

    def projection_energy(candidate: np.ndarray) -> float:
        denominator = np.einsum("id,id->i", candidate, candidate)
        coefficient = np.einsum(
            "isd,id->is", delta, candidate
        ) / np.maximum(denominator[:, None], 1e-12)
        projected = coefficient[:, :, None] * candidate[:, None, :]
        return float(np.square(projected).sum() / np.square(delta).sum())

    denominator = np.linalg.norm(delta, axis=2) * np.linalg.norm(
        direction, axis=1
    )[:, None]
    cosine = np.einsum("isd,id->is", delta, direction) / np.maximum(
        denominator, 1e-12
    )
    observed = projection_energy(direction)
    rng = np.random.default_rng(seed)
    permuted = np.asarray(
        [
            projection_energy(direction[rng.permutation(len(direction))])
            for _ in range(permutation_repeats)
        ]
    )
    return {
        "same_case_null_projection_energy": observed,
        "median_signed_cosine": float(np.median(cosine)),
        "fraction_toward_null": float(np.mean(cosine > 0)),
        "permuted_projection_median": float(np.median(permuted)),
        "permuted_projection_95pct": [
            float(value) for value in np.quantile(permuted, [0.025, 0.975])
        ],
        "permutation_p_one_sided": float(
            (1 + int(np.sum(permuted >= observed))) / (len(permuted) + 1)
        ),
    }


def endpoint_projection_controls(
    features: np.ndarray, patients: list[str]
) -> dict[str, float]:
    """Separate null-state alignment from generic clean-centroid contraction."""
    delta = displacement_tensor(features)
    real = features[:, 0].astype(np.float64)
    null_direction = features[:, 1].astype(np.float64) - real
    patient_array = np.asarray(patients)
    centroid_direction = np.stack(
        [
            real[patient_array != patient].mean(axis=0) - real[index]
            for index, patient in enumerate(patients)
        ]
    )

    def projection_energy(direction: np.ndarray) -> float:
        denominator = np.einsum("id,id->i", direction, direction)
        coefficient = np.einsum(
            "isd,id->is", delta, direction
        ) / np.maximum(denominator[:, None], 1e-12)
        projected = coefficient[:, :, None] * direction[:, None, :]
        return float(np.square(projected).sum() / np.square(delta).sum())

    null_on_centroid = np.einsum(
        "id,id->i", null_direction, centroid_direction
    ) / np.maximum(
        np.einsum("id,id->i", centroid_direction, centroid_direction), 1e-12
    )
    null_unique = (
        null_direction - null_on_centroid[:, None] * centroid_direction
    )
    centroid_on_null = np.einsum(
        "id,id->i", centroid_direction, null_direction
    ) / np.maximum(
        np.einsum("id,id->i", null_direction, null_direction), 1e-12
    )
    centroid_unique = (
        centroid_direction - centroid_on_null[:, None] * null_direction
    )
    denominator = np.linalg.norm(delta, axis=2) * np.linalg.norm(
        centroid_direction, axis=1
    )[:, None]
    centroid_cosine = np.einsum(
        "isd,id->is", delta, centroid_direction
    ) / np.maximum(denominator, 1e-12)
    direction_cosine = np.einsum(
        "id,id->i", null_direction, centroid_direction
    ) / np.maximum(
        np.linalg.norm(null_direction, axis=1)
        * np.linalg.norm(centroid_direction, axis=1),
        1e-12,
    )
    return {
        "clean_centroid_projection_energy": projection_energy(
            centroid_direction
        ),
        "null_projection_energy": projection_energy(null_direction),
        "null_unique_after_centroid_energy": projection_energy(null_unique),
        "centroid_unique_after_null_energy": projection_energy(centroid_unique),
        "fraction_toward_clean_centroid": float(np.mean(centroid_cosine > 0)),
        "median_null_centroid_direction_cosine": float(
            np.median(direction_cosine)
        ),
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
            for layer in LAYERS:
                features = data[layer][variant_index].astype(np.float32)
                delta = displacement_tensor(features)
                lineages[label][layer] = {
                    "anova": anova_factorization(delta),
                    "crossed_cell": crossed_cell_prediction(delta, patients),
                    "clean_state_predictability": patient_cv_predictability(
                        features, patients
                    ),
                    "null_prior": null_prior_projection(features),
                    "endpoint_controls": endpoint_projection_controls(
                        features, patients
                    ),
                }
    metrics = {
        "case_energy_fraction": ("anova", "case_energy_fraction"),
        "interaction_energy_fraction": ("anova", "interaction_energy_fraction"),
        "crossed_cell_r2_zero": ("crossed_cell", "crossed_cell_r2_zero"),
        "patient_cv_incremental_over_style_only": (
            "clean_state_predictability",
            "patient_cv_incremental_over_style_only",
        ),
        "same_case_null_projection_energy": (
            "null_prior",
            "same_case_null_projection_energy",
        ),
        "fraction_toward_null": ("null_prior", "fraction_toward_null"),
        "clean_centroid_projection_energy": (
            "endpoint_controls",
            "clean_centroid_projection_energy",
        ),
        "null_unique_after_centroid_energy": (
            "endpoint_controls",
            "null_unique_after_centroid_energy",
        ),
        "centroid_unique_after_null_energy": (
            "endpoint_controls",
            "centroid_unique_after_null_energy",
        ),
    }
    layer_summary = {}
    for layer in LAYERS:
        layer_summary[layer] = {
            name: summarize(
                [
                    values[layer][section][metric]
                    for values in lineages.values()
                ]
            )
            for name, (section, metric) in metrics.items()
        }
    prompt = layer_summary["llm_27_prompt"]
    return {
        "version": VERSION,
        "sources": sources,
        "lineages": lineages,
        "layers": layer_summary,
        "decision": {
            "case_conditioning_is_structurally_dominant": (
                prompt["case_energy_fraction"]["minimum"] > 0.60
            ),
            "crossed_multiview_field_is_predictable": (
                prompt["crossed_cell_r2_zero"]["minimum"] > 0.50
            ),
            "clean_single_view_predictor_is_positive_in_every_lineage": (
                prompt["patient_cv_incremental_over_style_only"]["minimum"] > 0.0
            ),
            "clean_single_view_predictor_remains_below_2pct_everywhere": (
                prompt["patient_cv_incremental_over_style_only"]["maximum"] < 0.02
            ),
            "same_case_null_component_exceeds_permuted_control_everywhere": all(
                values["llm_27_prompt"]["null_prior"][
                    "same_case_null_projection_energy"
                ]
                > values["llm_27_prompt"]["null_prior"][
                    "permuted_projection_95pct"
                ][1]
                for values in lineages.values()
            ),
            "clean_centroid_explains_more_than_null_everywhere": all(
                values["llm_27_prompt"]["endpoint_controls"][
                    "clean_centroid_projection_energy"
                ]
                > values["llm_27_prompt"]["endpoint_controls"][
                    "null_projection_energy"
                ]
                for values in lineages.values()
            ),
            "centroid_unique_component_exceeds_null_unique_everywhere": all(
                values["llm_27_prompt"]["endpoint_controls"][
                    "centroid_unique_after_null_energy"
                ]
                > values["llm_27_prompt"]["endpoint_controls"][
                    "null_unique_after_centroid_energy"
                ]
                for values in lineages.values()
            ),
        },
        "interpretation": (
            "The synthetic style field has a strong case-conditioned common "
            "mode. Its displacement aligns with the case-anchored null "
            "direction, but aligns more strongly with a leave-one-patient "
            "clean-state centroid; the current evidence therefore supports "
            "generic case-conditioned contraction, not a null-specific prior. "
            "A simple patient-held-out clean-state map explains little "
            "incremental energy."
        ),
        "claim_ceiling": (
            "paired synthetic Fourier views on 40 exposed MIMIC development "
            "cases and correlated Qwen2.5-VL-7B lineages; no clinical-utility "
            "or natural-scanner claim"
        ),
    }


def plot_result(result: dict[str, Any], output: Path) -> None:
    labels = list(result["lineages"])
    prompt_rows = [result["lineages"][label]["llm_27_prompt"] for label in labels]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))

    components = ("grand", "style", "case", "interaction")
    colors = ("#BDBDBD", "#2166AC", "#D6604D", "#7B3294")
    bottom = np.zeros(len(labels))
    for component, color in zip(components, colors, strict=True):
        values = np.asarray(
            [
                row["anova"][f"{component}_energy_fraction"] * 100
                for row in prompt_rows
            ]
        )
        axes[0].bar(labels, values, bottom=bottom, color=color, label=component)
        bottom += values
    axes[0].set_title("Exact style-field decomposition")
    axes[0].set_ylabel("Displacement energy (%)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    crossed = [
        row["crossed_cell"]["crossed_cell_r2_zero"] * 100
        for row in prompt_rows
    ]
    deployable = [
        row["clean_state_predictability"][
            "patient_cv_incremental_over_style_only"
        ]
        * 100
        for row in prompt_rows
    ]
    x = np.arange(len(labels))
    axes[1].bar(
        x,
        crossed,
        width=0.55,
        color="#1B7837",
        alpha=0.88,
        label="Crossed multi-view",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Transductive structure vs single-view prediction")
    axes[1].set_ylabel("Crossed-view explained energy (%)", color="#1B7837")
    axes[1].tick_params(axis="y", colors="#1B7837")
    deployable_axis = axes[1].twinx()
    deployable_axis.plot(
        x,
        deployable,
        color="#762A83",
        marker="o",
        linewidth=1.6,
        markersize=4,
        label="Clean-state KRR",
    )
    deployable_axis.set_ylim(0, max(1.5, max(deployable) * 1.2))
    deployable_axis.set_ylabel(
        "Single-view incremental energy (%)", color="#762A83"
    )
    deployable_axis.tick_params(axis="y", colors="#762A83")
    deployable_axis.spines["top"].set_visible(False)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    deployable_axis.legend(frameon=False, fontsize=7, loc="upper right")

    observed = [
        row["endpoint_controls"]["clean_centroid_projection_energy"] * 100
        for row in prompt_rows
    ]
    null_observed = [
        row["endpoint_controls"]["null_projection_energy"] * 100
        for row in prompt_rows
    ]
    axes[2].bar(
        x - 0.18,
        observed,
        width=0.36,
        color="#D73027",
        label="LOO clean centroid",
    )
    axes[2].bar(
        x + 0.18,
        null_observed,
        width=0.36,
        color="#91BFDB",
        label="Null state",
    )
    axes[2].set_title("Generic contraction dominates null alignment")
    axes[2].set_ylabel("Projected displacement energy (%)")
    axes[2].legend(frameon=False, fontsize=7)

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=48, ha="right", fontsize=6.5)
        axis.grid(axis="y", alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        "Acquisition style induces a case-conditioned contraction field",
        fontsize=12,
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
