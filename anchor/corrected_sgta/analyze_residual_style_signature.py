"""Test for a reusable style signature after generic contraction is removed.

The complete-sentence evidence displacement is decomposed cellwise as

    Delta[i,s] = beta[i,s] r[i] + rho[i,s],

where r[i] points from the clean evidence vector to its patient-LOO clean
centroid. The residual rho is orthogonal to this generic contraction direction.
If style selects a reproducible clinical prior, the same style should retain a
cross-patient direction in rho rather than only an image-specific response.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from anchor.corrected_sgta.analyze_clinical_evidence_attractor import (
    load_evidence,
    sha256,
)


VERSION = "residual-style-signature-v1"


def radial_residual(
    evidence: np.ndarray, patients: list[str], eps: float = 1e-12
) -> tuple[np.ndarray, float]:
    """Remove each cell's projection onto its clean-centroid direction."""
    real = evidence[:, 0]
    delta = evidence[:, 2:] - real[:, None, :]
    centroids = []
    for patient in patients:
        eligible = np.asarray(
            [other != patient for other in patients], dtype=bool
        )
        if not eligible.any():
            raise ValueError("patient-LOO centroid has no eligible cases")
        centroids.append(real[eligible].mean(axis=0))
    radial = np.asarray(centroids) - real
    coefficient = np.einsum("csd,cd->cs", delta, radial) / (
        np.square(radial).sum(axis=1)[:, None] + eps
    )
    projected = coefficient[:, :, None] * radial[:, None, :]
    residual = delta - projected
    radial_fraction = float(
        np.square(projected).sum() / (np.square(delta).sum() + eps)
    )
    orthogonality = float(
        np.max(np.abs(np.einsum("csd,cd->cs", residual, radial)))
    )
    if orthogonality > 1e-8:
        raise AssertionError(f"radial residual is not orthogonal: {orthogonality}")
    return residual, radial_fraction


def anova_energy(residual: np.ndarray) -> dict[str, float]:
    """Exact balanced case-by-style energy decomposition."""
    total = float(np.square(residual).sum())
    if total <= 0:
        raise ValueError("residual field has zero energy")
    grand = residual.mean(axis=(0, 1), keepdims=True)
    case = residual.mean(axis=1, keepdims=True) - grand
    style = residual.mean(axis=0, keepdims=True) - grand
    interaction = residual - grand - case - style
    values = {
        "grand": np.broadcast_to(grand, residual.shape),
        "case": np.broadcast_to(case, residual.shape),
        "style": np.broadcast_to(style, residual.shape),
        "interaction": interaction,
    }
    result = {
        key: float(np.square(value).sum() / total)
        for key, value in values.items()
    }
    if not np.isclose(sum(result.values()), 1.0, atol=1e-10):
        raise AssertionError("ANOVA energy does not sum to one")
    return result


def case_center_style_residual(residual: np.ndarray) -> np.ndarray:
    """Remove each case's residual component shared by all fixed styles."""
    centered = residual - residual.mean(axis=1, keepdims=True)
    if not np.allclose(centered.mean(axis=1), 0.0, atol=1e-12):
        raise AssertionError("case-centered style residual does not sum to zero")
    return centered


def cross_patient_style_metrics(
    residual: np.ndarray, patients: list[str], eps: float = 1e-12
) -> dict[str, float]:
    """Predict held-patient residuals using style prototypes from other patients."""
    prediction = np.zeros_like(residual)
    truth = []
    guessed = []
    for case_index, patient in enumerate(patients):
        eligible = np.asarray(
            [other != patient for other in patients], dtype=bool
        )
        prototypes = residual[eligible].mean(axis=0)
        prediction[case_index] = prototypes
        normalized_cell = residual[case_index] / (
            np.linalg.norm(residual[case_index], axis=1, keepdims=True) + eps
        )
        normalized_prototype = prototypes / (
            np.linalg.norm(prototypes, axis=1, keepdims=True) + eps
        )
        guessed.extend(
            (normalized_cell @ normalized_prototype.T).argmax(axis=1).tolist()
        )
        truth.extend(range(residual.shape[1]))
    residual_sse = float(np.square(residual - prediction).sum())
    zero_sse = float(np.square(residual).sum()) + eps
    cosine = np.einsum("csd,csd->cs", residual, prediction) / (
        np.linalg.norm(residual, axis=2)
        * np.linalg.norm(prediction, axis=2)
        + eps
    )
    return {
        "style_prototype_r2_zero": float(1.0 - residual_sse / zero_sse),
        "style_identification_accuracy": float(
            np.mean(np.asarray(truth) == np.asarray(guessed))
        ),
        "mean_style_prototype_cosine": float(cosine.mean()),
        "median_style_prototype_cosine": float(np.median(cosine)),
    }


def patient_blocked_style_permutation(
    residual: np.ndarray,
    patients: list[str],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Break cross-patient style correspondence while preserving each patient."""
    observed = cross_patient_style_metrics(residual, patients)
    groups = sorted(set(patients))
    patient_array = np.asarray(patients)
    indices = {
        patient: np.flatnonzero(patient_array == patient) for patient in groups
    }
    rng = np.random.default_rng(seed)
    null = {
        "style_prototype_r2_zero": [],
        "style_identification_accuracy": [],
    }
    for _ in range(repeats):
        permuted = residual.copy()
        for patient in groups:
            permutation = rng.permutation(residual.shape[1])
            permuted[indices[patient]] = permuted[indices[patient]][
                :, permutation
            ]
        metrics = cross_patient_style_metrics(permuted, patients)
        for key in null:
            null[key].append(metrics[key])
    result: dict[str, Any] = {"observed": observed, "null": {}}
    for key, values in null.items():
        array = np.asarray(values, dtype=np.float64)
        result["null"][key] = {
            "mean": float(array.mean()),
            "95pct": [
                float(value) for value in np.quantile(array, [0.025, 0.975])
            ],
            "one_sided_p": float(
                (1 + np.sum(array >= observed[key])) / (repeats + 1)
            ),
        }
    return result


def cross_model_signature(
    first: np.ndarray, second: np.ndarray, eps: float = 1e-12
) -> dict[str, Any]:
    """Compare global residual style directions between paired checkpoints."""
    first_mean = first.mean(axis=0)
    second_mean = second.mean(axis=0)
    first_unit = first_mean / (
        np.linalg.norm(first_mean, axis=1, keepdims=True) + eps
    )
    second_unit = second_mean / (
        np.linalg.norm(second_mean, axis=1, keepdims=True) + eps
    )
    cosine = first_unit @ second_unit.T
    observed = float(np.trace(cosine) / cosine.shape[0])
    assignments = np.asarray(
        [
            np.mean(
                [
                    cosine[index, permutation[index]]
                    for index in range(cosine.shape[0])
                ]
            )
            for permutation in itertools.permutations(range(cosine.shape[0]))
        ],
        dtype=np.float64,
    )
    first_norm = np.linalg.norm(first_mean, axis=1)
    second_norm = np.linalg.norm(second_mean, axis=1)
    norm_ratio = first_norm / (second_norm + eps)
    return {
        "first_style_signature_vectors": first_mean.tolist(),
        "second_style_signature_vectors": second_mean.tolist(),
        "cosine_matrix": cosine.tolist(),
        "matched_style_mean_cosine": observed,
        "style_assignment_exact_p": float(
            np.sum(assignments >= observed) / len(assignments)
        ),
        "assignment_null_mean": float(assignments.mean()),
        "assignment_null_95pct": [
            float(value) for value in np.quantile(assignments, [0.025, 0.975])
        ],
        "first_to_second_signature_norm_ratio": norm_ratio.tolist(),
        "mean_signature_norm_ratio": float(norm_ratio.mean()),
        "median_signature_norm_ratio": float(np.median(norm_ratio)),
        "all_first_signatures_smaller": bool(np.all(norm_ratio < 1)),
    }


def analyze_files(
    huatuo_path: Path,
    base_path: Path,
    permutation_repeats: int = 1000,
    seed: int = 2027,
) -> dict[str, Any]:
    models = {}
    residuals = {}
    residual_scales = {}
    common_pairing = None
    common_diseases = None
    common_styles = None
    for name, path in (("huatuo", huatuo_path), ("qwen_base", base_path)):
        (
            evidence,
            cases,
            patients,
            diseases,
            model,
            pairing,
        ) = load_evidence(path)
        if common_pairing is None:
            common_pairing = pairing
            common_diseases = diseases
            common_styles = [
                f"style_{index}" for index in range(evidence.shape[1] - 2)
            ]
        elif pairing != common_pairing or diseases != common_diseases:
            raise ValueError("model probes are not exactly paired")
        residual, radial_fraction = radial_residual(evidence, patients)
        style_centered_residual = case_center_style_residual(residual)
        permutation = patient_blocked_style_permutation(
            style_centered_residual, patients, permutation_repeats, seed
        )
        disease_scale = evidence[:, 0].std(axis=0, ddof=1)
        if np.any(disease_scale <= 0):
            raise ValueError("clean evidence contains a zero-variance disease")
        whitened_evidence = evidence / disease_scale[None, None, :]
        whitened_residual, whitened_radial_fraction = radial_residual(
            whitened_evidence, patients
        )
        whitened_style_centered = case_center_style_residual(
            whitened_residual
        )
        whitened_permutation = patient_blocked_style_permutation(
            whitened_style_centered,
            patients,
            permutation_repeats,
            seed + 10000,
        )
        residuals[name] = style_centered_residual
        residuals[f"{name}_whitened"] = whitened_style_centered
        residual_scales[name] = {
            "full_residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
            "case_centered_residual_rms": float(
                np.sqrt(np.mean(np.square(style_centered_residual)))
            ),
        }
        models[name] = {
            "model": model,
            "raw": str(path),
            "raw_sha256": sha256(path),
            "cases": len(cases),
            "patients": len(set(patients)),
            "radial_projection_energy": radial_fraction,
            "residual_anova": anova_energy(residual),
            "style_test_case_centered": True,
            "cross_patient_style_test": permutation,
            "disease_whitened_sensitivity": {
                "clean_disease_scale": disease_scale.tolist(),
                "radial_projection_energy": whitened_radial_fraction,
                "residual_anova": anova_energy(whitened_residual),
                "style_test_case_centered": True,
                "cross_patient_style_test": whitened_permutation,
            },
        }
    cross_model = cross_model_signature(
        residuals["huatuo"], residuals["qwen_base"]
    )
    cross_model_whitened = cross_model_signature(
        residuals["huatuo_whitened"], residuals["qwen_base_whitened"]
    )
    raw_norm_ratio = np.asarray(
        cross_model["first_to_second_signature_norm_ratio"]
    )
    full_rms_ratio = raw_norm_ratio * (
        residual_scales["qwen_base"]["full_residual_rms"]
        / residual_scales["huatuo"]["full_residual_rms"]
    )
    centered_rms_ratio = raw_norm_ratio * (
        residual_scales["qwen_base"]["case_centered_residual_rms"]
        / residual_scales["huatuo"]["case_centered_residual_rms"]
    )
    return {
        "version": VERSION,
        "paired_probe_fingerprint": common_pairing,
        "diseases": common_diseases,
        "styles": common_styles,
        "models": models,
        "cross_model": cross_model,
        "cross_model_disease_whitened_sensitivity": cross_model_whitened,
        "cross_model_magnitude_scale_sensitivity": {
            "raw_coordinate": {
                "ratios": raw_norm_ratio.tolist(),
                "mean": float(raw_norm_ratio.mean()),
                "styles_below_one": int(np.sum(raw_norm_ratio < 1)),
            },
            "diseasewise_clean_state_standardized": {
                "ratios": cross_model_whitened[
                    "first_to_second_signature_norm_ratio"
                ],
                "mean": cross_model_whitened["mean_signature_norm_ratio"],
                "styles_below_one": int(
                    np.sum(
                        np.asarray(
                            cross_model_whitened[
                                "first_to_second_signature_norm_ratio"
                            ]
                        )
                        < 1
                    )
                ),
            },
            "full_residual_rms_standardized": {
                "ratios": full_rms_ratio.tolist(),
                "mean": float(full_rms_ratio.mean()),
                "styles_below_one": int(np.sum(full_rms_ratio < 1)),
            },
            "case_centered_residual_rms_standardized": {
                "ratios": centered_rms_ratio.tolist(),
                "mean": float(centered_rms_ratio.mean()),
                "styles_below_one": int(np.sum(centered_rms_ratio < 1)),
            },
        },
        "decision": {
            "huatuo_residual_style_identity_above_permutation": models[
                "huatuo"
            ]["cross_patient_style_test"]["null"][
                "style_identification_accuracy"
            ][
                "one_sided_p"
            ]
            < 0.05,
            "both_style_prototype_r2_above_permutation": all(
                models[name]["cross_patient_style_test"]["null"][
                    "style_prototype_r2_zero"
                ]["one_sided_p"]
                < 0.05
                for name in models
            ),
            "matched_cross_model_style_assignment_nonrandom": cross_model[
                "style_assignment_exact_p"
            ]
            < 0.05,
            "matched_cross_model_assignment_survives_whitening": (
                cross_model_whitened["style_assignment_exact_p"] < 0.05
            ),
            "huatuo_whitened_style_identity_above_permutation": models[
                "huatuo"
            ]["disease_whitened_sensitivity"][
                "cross_patient_style_test"
            ]["null"][
                "style_identification_accuracy"
            ][
                "one_sided_p"
            ]
            < 0.05,
            "huatuo_whitened_prototype_r2_above_permutation": models[
                "huatuo"
            ]["disease_whitened_sensitivity"][
                "cross_patient_style_test"
            ]["null"][
                "style_prototype_r2_zero"
            ][
                "one_sided_p"
            ]
            < 0.05,
            "huatuo_raw_coordinate_signature_smaller_all_styles": cross_model[
                "all_first_signatures_smaller"
            ],
            "residual_style_energy_below_two_percent_both_models": all(
                models[name]["residual_anova"]["style"] < 0.02
                for name in models
            ),
        },
        "interpretation": (
            "After model-specific generic centroid contraction is removed, a "
            "weak but cross-patient reproducible style signature remains in "
            "complete-sentence disease evidence. The six supplied labels show "
            "aggregate cross-checkpoint alignment, but not every matched style "
            "direction is positively aligned. "
            "Cross-model style matching and prototype R2 survive disease-wise "
            "whitening after removing each case's common residual. "
            "Raw-coordinate Huatuo signature norms are smaller, but this "
            "ordering is scale-sensitive. This supports an exploratory, "
            "cohort-conditional residual spectral signature, not a "
            "medical-training-specific clinical prior or an effective decoder."
        ),
        "claim_ceiling": (
            "target-transductive teacher-forced evidence audit on 64 exposed "
            "MIMIC development images and six fixed synthetic styles; "
            "exploratory unadjusted tests; no independent replication, "
            "generated accuracy, causal tuning, natural acquisition, or "
            "deployment claim"
        ),
    }


def plot_result(result: dict[str, Any], output: Path) -> None:
    labels = ["Qwen base", "Huatuo"]
    keys = ["qwen_base", "huatuo"]
    colors = ["#4393C3", "#D6604D"]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))

    components = ["grand", "case", "style", "interaction"]
    component_colors = ["#BDBDBD", "#2166AC", "#FDAE61", "#B2182B"]
    bottom = np.zeros(2)
    for component, color in zip(components, component_colors):
        values = np.asarray(
            [
                result["models"][key]["residual_anova"][component] * 100
                for key in keys
            ]
        )
        axes[0].bar(labels, values, bottom=bottom, color=color, label=component)
        bottom += values
    axes[0].set_ylabel("Residual field energy (%)")
    axes[0].set_title("Case dependence still dominates")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    observed = [
        result["models"][key]["cross_patient_style_test"]["observed"][
            "style_identification_accuracy"
        ]
        * 100
        for key in keys
    ]
    lower = [
        result["models"][key]["cross_patient_style_test"]["null"][
            "style_identification_accuracy"
        ]["95pct"][0]
        * 100
        for key in keys
    ]
    upper = [
        result["models"][key]["cross_patient_style_test"]["null"][
            "style_identification_accuracy"
        ]["95pct"][1]
        * 100
        for key in keys
    ]
    axes[1].bar(labels, observed, color=colors, width=0.6)
    for index in range(2):
        axes[1].fill_between(
            [index - 0.3, index + 0.3],
            lower[index],
            upper[index],
            color="#777777",
            alpha=0.2,
        )
    axes[1].axhline(100 / 6, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Held-patient style ID (%)")
    axes[1].set_title("A weak residual signature is identifiable")

    cosine = np.asarray(result["cross_model"]["cosine_matrix"])
    image = axes[2].imshow(cosine, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    for row in range(cosine.shape[0]):
        for column in range(cosine.shape[1]):
            axes[2].text(
                column,
                row,
                f"{cosine[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
            )
    axes[2].set_xlabel("Qwen base style")
    axes[2].set_ylabel("Huatuo style")
    axes[2].set_title("Style directions transfer across checkpoints")
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)

    for axis in axes[:2]:
        axis.grid(axis="y", alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        "Residual style signatures survive generic centroid correction",
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
    parser.add_argument("--huatuo", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--permutation-repeats", type=int, default=1000)
    args = parser.parse_args()
    result = analyze_files(
        args.huatuo.expanduser().resolve(),
        args.base.expanduser().resolve(),
        args.permutation_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    plot_result(result, args.figure)
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
