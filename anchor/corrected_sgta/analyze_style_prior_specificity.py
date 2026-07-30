"""Distinguish disease-specific style priors from a uniform answer bias.

The residual style audit identifies a weak reusable six-disease signature.
This module asks a stricter question: does that signature survive removal of
the uniform disease axis, or does each style merely raise/lower all disease
evidence together?
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
from anchor.corrected_sgta.analyze_residual_style_signature import (
    anova_energy,
    case_center_style_residual,
    cross_model_signature,
    patient_blocked_style_permutation,
    radial_residual,
)


VERSION = "style-prior-specificity-v1"


def split_uniform_disease_axis(
    field: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, float]:
    """Orthogonally split a field into uniform and disease-contrast parts."""
    disease_count = field.shape[-1]
    uniform_axis = np.ones(disease_count, dtype=np.float64) / np.sqrt(
        disease_count
    )
    coefficient = np.einsum("...d,d->...", field, uniform_axis)
    uniform = coefficient[..., None] * uniform_axis
    contrast = field - uniform
    energy_fraction = float(
        np.square(uniform).sum() / (np.square(field).sum() + eps)
    )
    if not np.allclose(
        np.einsum("...d,d->...", contrast, uniform_axis),
        0.0,
        atol=1e-10,
    ):
        raise AssertionError("disease contrast is not orthogonal to uniform axis")
    return uniform, contrast, energy_fraction


def joint_nuisance_residual(
    evidence: np.ndarray,
    patients: list[str],
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Remove the joint span of patient radial and uniform disease axes."""
    real = evidence[:, 0]
    delta = evidence[:, 2:] - real[:, None, :]
    uniform_axis = np.ones(evidence.shape[-1], dtype=np.float64)
    uniform_axis /= np.linalg.norm(uniform_axis)
    residual = np.zeros_like(delta)
    removed_energy = 0.0
    for case_index, patient in enumerate(patients):
        eligible = np.asarray(
            [other != patient for other in patients], dtype=bool
        )
        radial = real[eligible].mean(axis=0) - real[case_index]
        nuisance = np.stack([radial, uniform_axis], axis=1)
        left, singular_values, _ = np.linalg.svd(
            nuisance, full_matrices=False
        )
        tolerance = max(nuisance.shape) * singular_values[0] * 1e-12
        rank = int(np.sum(singular_values > tolerance))
        basis = left[:, :rank]
        projected = (delta[case_index] @ basis) @ basis.T
        residual[case_index] = delta[case_index] - projected
        removed_energy += float(np.square(projected).sum())
        if not np.allclose(
            residual[case_index] @ basis, 0.0, atol=1e-10
        ):
            raise AssertionError("joint nuisance residual is not orthogonal")
    centered = case_center_style_residual(residual)
    return centered, float(
        removed_energy / (np.square(delta).sum() + eps)
    )


def signature_spectrum(field: np.ndarray, eps: float = 1e-12) -> dict[str, Any]:
    """Return scale-free spectrum of the case-averaged style signature."""
    signature = field.mean(axis=0)
    singular_values = np.linalg.svd(signature, compute_uv=False)
    energy = np.square(singular_values)
    proportions = energy / (energy.sum() + eps)
    positive = proportions[proportions > eps]
    entropy_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    participation_rank = float(
        np.square(energy.sum()) / (np.square(energy).sum() + eps)
    )
    return {
        "singular_values": singular_values.tolist(),
        "energy_proportions": proportions.tolist(),
        "entropy_effective_rank": entropy_rank,
        "participation_effective_rank": participation_rank,
    }


def exact_disease_assignment(
    first: np.ndarray, second: np.ndarray, eps: float = 1e-12
) -> dict[str, Any]:
    """Test matched disease profiles after column-wise normalization."""
    first_signature = first.mean(axis=0)
    second_signature = second.mean(axis=0)
    first_unit = first_signature / (
        np.linalg.norm(first_signature, axis=0, keepdims=True) + eps
    )
    second_unit = second_signature / (
        np.linalg.norm(second_signature, axis=0, keepdims=True) + eps
    )
    cosine = first_unit.T @ second_unit

    def score(permutation: tuple[int, ...]) -> float:
        return float(
            np.mean(
                [
                    cosine[index, permutation[index]]
                    for index in range(cosine.shape[0])
                ]
            )
        )

    identity = tuple(range(first.shape[-1]))
    observed = score(identity)
    assignments = np.asarray(
        [score(permutation) for permutation in itertools.permutations(identity)],
        dtype=np.float64,
    )
    return {
        "matched_disease_mean_cosine": observed,
        "disease_cosine_matrix": cosine.tolist(),
        "disease_assignment_exact_p": float(
            np.sum(assignments >= observed) / len(assignments)
        ),
        "assignment_count": int(len(assignments)),
        "assignment_null_mean": float(assignments.mean()),
        "assignment_null_95pct": [
            float(value) for value in np.quantile(assignments, [0.025, 0.975])
        ],
    }


def patient_cluster_bootstrap_alignment(
    first: np.ndarray,
    second: np.ndarray,
    patients: list[str],
    repeats: int,
    seed: int,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Bootstrap cross-checkpoint alignment over patient clusters."""
    if first.shape != second.shape or first.shape[0] != len(patients):
        raise ValueError("paired fields and patients do not align")
    groups = sorted(set(patients))
    patient_array = np.asarray(patients)
    group_rows = {
        group: np.flatnonzero(patient_array == group) for group in groups
    }
    permutations = list(itertools.permutations(range(first.shape[-1])))
    identity = tuple(range(first.shape[-1]))
    non_identity = [
        permutation for permutation in permutations if permutation != identity
    ]

    def scores(rows: np.ndarray) -> tuple[float, float, float]:
        first_signature = first[rows].mean(axis=0)
        second_signature = second[rows].mean(axis=0)
        first_style_unit = first_signature / (
            np.linalg.norm(first_signature, axis=1, keepdims=True) + eps
        )
        second_style_unit = second_signature / (
            np.linalg.norm(second_signature, axis=1, keepdims=True) + eps
        )
        style_cosine = first_style_unit @ second_style_unit.T
        style_observed = float(
            np.trace(style_cosine) / style_cosine.shape[0]
        )
        style_competitor = max(
            float(
                np.mean(
                    [
                        style_cosine[index, permutation[index]]
                        for index in range(style_cosine.shape[0])
                    ]
                )
            )
            for permutation in non_identity
        )
        first_disease_unit = first_signature / (
            np.linalg.norm(first_signature, axis=0, keepdims=True) + eps
        )
        second_disease_unit = second_signature / (
            np.linalg.norm(second_signature, axis=0, keepdims=True) + eps
        )
        disease_cosine = first_disease_unit.T @ second_disease_unit
        disease_observed = float(
            np.trace(disease_cosine) / disease_cosine.shape[0]
        )
        disease_competitor = max(
            float(
                np.mean(
                    [
                        disease_cosine[index, permutation[index]]
                        for index in range(disease_cosine.shape[0])
                    ]
                )
            )
            for permutation in non_identity
        )
        return (
            style_observed,
            style_observed - style_competitor,
            disease_observed - disease_competitor,
        )

    all_rows = np.arange(first.shape[0])
    observed = scores(all_rows)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled])
        estimates.append(scores(rows))
    array = np.asarray(estimates)
    names = (
        "matched_style_mean_cosine",
        "style_identity_assignment_margin",
        "disease_identity_assignment_margin",
    )
    return {
        name: {
            "observed": float(observed[index]),
            "patient_cluster_bootstrap_95pct": [
                float(value)
                for value in np.quantile(array[:, index], [0.025, 0.975])
            ],
            "bootstrap_positive_fraction": float(
                np.mean(array[:, index] > 0)
            ),
        }
        for index, name in enumerate(names)
    }


def patient_blocked_spectrum_permutation(
    field: np.ndarray,
    patients: list[str],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Test whether observed signature rank is unusual under style relabeling."""
    observed_spectrum = signature_spectrum(field)
    observed = {
        "entropy_effective_rank": observed_spectrum[
            "entropy_effective_rank"
        ],
        "participation_effective_rank": observed_spectrum[
            "participation_effective_rank"
        ],
        "first_two_energy": float(
            np.sum(observed_spectrum["energy_proportions"][:2])
        ),
    }
    groups = sorted(set(patients))
    patient_array = np.asarray(patients)
    indices = {
        patient: np.flatnonzero(patient_array == patient) for patient in groups
    }
    rng = np.random.default_rng(seed)
    null = {key: [] for key in observed}
    for _ in range(repeats):
        permuted = field.copy()
        for patient in groups:
            permutation = rng.permutation(field.shape[1])
            permuted[indices[patient]] = permuted[indices[patient]][
                :, permutation
            ]
        spectrum = signature_spectrum(permuted)
        null["entropy_effective_rank"].append(
            spectrum["entropy_effective_rank"]
        )
        null["participation_effective_rank"].append(
            spectrum["participation_effective_rank"]
        )
        null["first_two_energy"].append(
            float(np.sum(spectrum["energy_proportions"][:2]))
        )
    result: dict[str, Any] = {"observed": observed, "null": {}}
    for key, values in null.items():
        array = np.asarray(values)
        lower_is_extreme = key != "first_two_energy"
        count = (
            np.sum(array <= observed[key])
            if lower_is_extreme
            else np.sum(array >= observed[key])
        )
        result["null"][key] = {
            "mean": float(array.mean()),
            "95pct": [
                float(value) for value in np.quantile(array, [0.025, 0.975])
            ],
            "one_sided_p": float((1 + count) / (repeats + 1)),
        }
    return result


def patient_cluster_bootstrap_uniform_fraction(
    field: np.ndarray,
    patients: list[str],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the uniform-axis fraction of the reusable style signature."""
    groups = sorted(set(patients))
    patient_array = np.asarray(patients)
    group_rows = {
        group: np.flatnonzero(patient_array == group) for group in groups
    }

    def estimate(rows: np.ndarray) -> float:
        signature = field[rows].mean(axis=0)
        return split_uniform_disease_axis(signature)[2]

    observed = estimate(np.arange(field.shape[0]))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled])
        estimates.append(estimate(rows))
    array = np.asarray(estimates)
    return {
        "observed": observed,
        "patient_cluster_bootstrap_95pct": [
            float(value) for value in np.quantile(array, [0.025, 0.975])
        ],
        "bootstrap_fraction_below_half": float(np.mean(array < 0.5)),
    }


def _model_field(
    path: Path,
    repeats: int,
    seed: int,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    str,
    list[str],
    str,
    list[str],
]:
    evidence, cases, patients, diseases, model, pairing = load_evidence(path)
    residual, radial_fraction = radial_residual(evidence, patients)
    centered = case_center_style_residual(residual)
    uniform, contrast, cell_uniform_fraction = split_uniform_disease_axis(
        centered
    )
    signature = centered.mean(axis=0)
    _, _, signature_uniform_fraction = split_uniform_disease_axis(signature)
    signature_uniform_bootstrap = patient_cluster_bootstrap_uniform_fraction(
        centered, patients, repeats, seed + 500
    )
    full_test = patient_blocked_style_permutation(
        centered, patients, repeats, seed
    )
    contrast_test = patient_blocked_style_permutation(
        contrast, patients, repeats, seed + 1000
    )
    contrast_spectrum_test = patient_blocked_spectrum_permutation(
        contrast, patients, repeats, seed + 1500
    )
    joint_residual, joint_removed_fraction = joint_nuisance_residual(
        evidence, patients
    )
    joint_test = patient_blocked_style_permutation(
        joint_residual, patients, repeats, seed + 1750
    )
    joint_spectrum_test = patient_blocked_spectrum_permutation(
        joint_residual, patients, repeats, seed + 1900
    )

    disease_scale = evidence[:, 0].std(axis=0, ddof=1)
    whitened_evidence = evidence / disease_scale[None, None, :]
    whitened_residual, _ = radial_residual(whitened_evidence, patients)
    whitened_centered = case_center_style_residual(whitened_residual)
    (
        whitened_uniform,
        whitened_contrast,
        whitened_cell_uniform_fraction,
    ) = split_uniform_disease_axis(whitened_centered)
    whitened_signature = whitened_centered.mean(axis=0)
    _, _, whitened_signature_uniform_fraction = split_uniform_disease_axis(
        whitened_signature
    )
    whitened_contrast_test = patient_blocked_style_permutation(
        whitened_contrast, patients, repeats, seed + 2000
    )
    whitened_joint_residual, whitened_joint_removed_fraction = (
        joint_nuisance_residual(whitened_evidence, patients)
    )
    whitened_joint_test = patient_blocked_style_permutation(
        whitened_joint_residual, patients, repeats, seed + 2250
    )
    result = {
        "model": model,
        "raw": str(path),
        "raw_sha256": sha256(path),
        "cases": len(cases),
        "patients": len(set(patients)),
        "radial_projection_energy": radial_fraction,
        "case_centered_residual_anova": anova_energy(centered),
        "uniform_axis": {
            "cell_field_energy_fraction": cell_uniform_fraction,
            "style_signature_energy_fraction": signature_uniform_fraction,
            "style_signature_patient_cluster_bootstrap": (
                signature_uniform_bootstrap
            ),
        },
        "full_style_test": full_test,
        "disease_contrast_style_test": contrast_test,
        "disease_contrast_spectrum": signature_spectrum(contrast),
        "disease_contrast_spectrum_permutation": contrast_spectrum_test,
        "joint_nuisance_complement": {
            "removed_displacement_energy_fraction": joint_removed_fraction,
            "style_test": joint_test,
            "spectrum": signature_spectrum(joint_residual),
            "spectrum_permutation": joint_spectrum_test,
        },
        "disease_whitened_sensitivity": {
            "clean_disease_scale": disease_scale.tolist(),
            "cell_field_uniform_energy_fraction": (
                whitened_cell_uniform_fraction
            ),
            "style_signature_uniform_energy_fraction": (
                whitened_signature_uniform_fraction
            ),
            "disease_contrast_style_test": whitened_contrast_test,
            "disease_contrast_spectrum": signature_spectrum(
                whitened_contrast
            ),
            "joint_nuisance_complement": {
                "removed_displacement_energy_fraction": (
                    whitened_joint_removed_fraction
                ),
                "style_test": whitened_joint_test,
                "spectrum": signature_spectrum(
                    whitened_joint_residual
                ),
            },
        },
    }
    fields = {
        "centered": centered,
        "uniform": uniform,
        "contrast": contrast,
        "whitened_centered": whitened_centered,
        "whitened_uniform": whitened_uniform,
        "whitened_contrast": whitened_contrast,
        "joint_nuisance": joint_residual,
        "whitened_joint_nuisance": whitened_joint_residual,
    }
    return result, fields, pairing, diseases, model, patients


def analyze_files(
    huatuo_path: Path,
    base_path: Path,
    permutation_repeats: int = 1000,
    seed: int = 2027,
) -> dict[str, Any]:
    huatuo, huatuo_fields, pairing, diseases, _, patients = _model_field(
        huatuo_path, permutation_repeats, seed
    )
    base, base_fields, base_pairing, base_diseases, _, base_patients = (
        _model_field(
        base_path, permutation_repeats, seed
        )
    )
    if (
        pairing != base_pairing
        or diseases != base_diseases
        or patients != base_patients
    ):
        raise ValueError("model probes are not exactly paired")

    contrast_cross_model = cross_model_signature(
        huatuo_fields["contrast"], base_fields["contrast"]
    )
    contrast_disease_assignment = exact_disease_assignment(
        huatuo_fields["contrast"], base_fields["contrast"]
    )
    whitened_contrast_cross_model = cross_model_signature(
        huatuo_fields["whitened_contrast"],
        base_fields["whitened_contrast"],
    )
    whitened_contrast_disease_assignment = exact_disease_assignment(
        huatuo_fields["whitened_contrast"],
        base_fields["whitened_contrast"],
    )
    joint_cross_model = cross_model_signature(
        huatuo_fields["joint_nuisance"],
        base_fields["joint_nuisance"],
    )
    joint_disease_assignment = exact_disease_assignment(
        huatuo_fields["joint_nuisance"],
        base_fields["joint_nuisance"],
    )
    whitened_joint_cross_model = cross_model_signature(
        huatuo_fields["whitened_joint_nuisance"],
        base_fields["whitened_joint_nuisance"],
    )
    whitened_joint_disease_assignment = exact_disease_assignment(
        huatuo_fields["whitened_joint_nuisance"],
        base_fields["whitened_joint_nuisance"],
    )
    contrast_patient_bootstrap = patient_cluster_bootstrap_alignment(
        huatuo_fields["contrast"],
        base_fields["contrast"],
        patients,
        permutation_repeats,
        seed + 3000,
    )
    whitened_contrast_patient_bootstrap = (
        patient_cluster_bootstrap_alignment(
            huatuo_fields["whitened_contrast"],
            base_fields["whitened_contrast"],
            patients,
            permutation_repeats,
            seed + 4000,
        )
    )
    joint_patient_bootstrap = patient_cluster_bootstrap_alignment(
        huatuo_fields["joint_nuisance"],
        base_fields["joint_nuisance"],
        patients,
        permutation_repeats,
        seed + 4500,
    )
    whitened_joint_patient_bootstrap = patient_cluster_bootstrap_alignment(
        huatuo_fields["whitened_joint_nuisance"],
        base_fields["whitened_joint_nuisance"],
        patients,
        permutation_repeats,
        seed + 5000,
    )

    models = {"huatuo": huatuo, "qwen_base": base}
    decision = {
        "contrast_style_identity_above_permutation_both": all(
            model["disease_contrast_style_test"]["null"][
                "style_identification_accuracy"
            ]["one_sided_p"]
            < 0.05
            for model in models.values()
        ),
        "contrast_style_prototype_r2_above_permutation_both": all(
            model["disease_contrast_style_test"]["null"][
                "style_prototype_r2_zero"
            ]["one_sided_p"]
            < 0.05
            for model in models.values()
        ),
        "contrast_cross_model_style_assignment_nonrandom": (
            contrast_cross_model["style_assignment_exact_p"] < 0.05
        ),
        "contrast_cross_model_disease_assignment_nonrandom": (
            contrast_disease_assignment["disease_assignment_exact_p"] < 0.05
        ),
        "whitened_contrast_cross_model_style_assignment_nonrandom": (
            whitened_contrast_cross_model["style_assignment_exact_p"] < 0.05
        ),
        "whitened_contrast_cross_model_disease_assignment_nonrandom": (
            whitened_contrast_disease_assignment[
                "disease_assignment_exact_p"
            ]
            < 0.05
        ),
        "contrast_matched_cosine_patient_ci_positive": (
            contrast_patient_bootstrap["matched_style_mean_cosine"][
                "patient_cluster_bootstrap_95pct"
            ][0]
            > 0
        ),
        "contrast_style_identity_margin_patient_ci_positive": (
            contrast_patient_bootstrap["style_identity_assignment_margin"][
                "patient_cluster_bootstrap_95pct"
            ][0]
            > 0
        ),
        "contrast_disease_identity_margin_patient_ci_positive": (
            contrast_patient_bootstrap[
                "disease_identity_assignment_margin"
            ]["patient_cluster_bootstrap_95pct"][0]
            > 0
        ),
        "whitened_contrast_style_identity_margin_patient_ci_positive": (
            whitened_contrast_patient_bootstrap[
                "style_identity_assignment_margin"
            ]["patient_cluster_bootstrap_95pct"][0]
            > 0
        ),
        "whitened_contrast_disease_identity_margin_patient_ci_positive": (
            whitened_contrast_patient_bootstrap[
                "disease_identity_assignment_margin"
            ]["patient_cluster_bootstrap_95pct"][0]
            > 0
        ),
        "contrast_effective_rank_below_permutation_both": all(
            model["disease_contrast_spectrum_permutation"]["null"][
                "participation_effective_rank"
            ]["one_sided_p"]
            < 0.05
            for model in models.values()
        ),
        "uniform_style_signature_ci_upper_below_half_both": all(
            model["uniform_axis"][
                "style_signature_patient_cluster_bootstrap"
            ]["patient_cluster_bootstrap_95pct"][1]
            < 0.5
            for model in models.values()
        ),
        "joint_nuisance_style_identity_above_permutation_both": all(
            model["joint_nuisance_complement"]["style_test"]["null"][
                "style_identification_accuracy"
            ]["one_sided_p"]
            < 0.05
            for model in models.values()
        ),
        "joint_nuisance_prototype_r2_above_permutation_both": all(
            model["joint_nuisance_complement"]["style_test"]["null"][
                "style_prototype_r2_zero"
            ]["one_sided_p"]
            < 0.05
            for model in models.values()
        ),
        "joint_nuisance_matched_cosine_patient_ci_positive": (
            joint_patient_bootstrap["matched_style_mean_cosine"][
                "patient_cluster_bootstrap_95pct"
            ][0]
            > 0
        ),
        "joint_nuisance_style_identity_margin_patient_ci_positive": (
            joint_patient_bootstrap["style_identity_assignment_margin"][
                "patient_cluster_bootstrap_95pct"
            ][0]
            > 0
        ),
        "joint_nuisance_disease_identity_margin_patient_ci_positive": (
            joint_patient_bootstrap["disease_identity_assignment_margin"][
                "patient_cluster_bootstrap_95pct"
            ][0]
            > 0
        ),
    }
    return {
        "version": VERSION,
        "paired_probe_fingerprint": pairing,
        "styles": [f"style_{index}" for index in range(6)],
        "diseases": diseases,
        "models": models,
        "disease_contrast_cross_model": contrast_cross_model,
        "disease_contrast_disease_assignment": (
            contrast_disease_assignment
        ),
        "whitened_disease_contrast_cross_model": (
            whitened_contrast_cross_model
        ),
        "whitened_disease_contrast_disease_assignment": (
            whitened_contrast_disease_assignment
        ),
        "disease_contrast_patient_cluster_bootstrap": (
            contrast_patient_bootstrap
        ),
        "whitened_disease_contrast_patient_cluster_bootstrap": (
            whitened_contrast_patient_bootstrap
        ),
        "joint_nuisance_cross_model": joint_cross_model,
        "joint_nuisance_disease_assignment": joint_disease_assignment,
        "joint_nuisance_patient_cluster_bootstrap": joint_patient_bootstrap,
        "whitened_joint_nuisance_cross_model": (
            whitened_joint_cross_model
        ),
        "whitened_joint_nuisance_disease_assignment": (
            whitened_joint_disease_assignment
        ),
        "whitened_joint_nuisance_patient_cluster_bootstrap": (
            whitened_joint_patient_bootstrap
        ),
        "decision": decision,
        "claim_ceiling": (
            "exploratory target-transductive teacher-forced audit on six "
            "fixed diseases and styles; uniform-axis removal tests "
            "disease specificity, but identity-assignment margins are not "
            "assumed from sequential noncommuting projections; the joint "
            "nuisance complement is primary. No clinical prior, natural "
            "acquisition shift, generation utility, or deployment is established"
        ),
    }


def plot_result(result: dict[str, Any], output: Path) -> None:
    keys = ["qwen_base", "huatuo"]
    labels = ["Qwen base", "Huatuo"]
    colors = ["#4393C3", "#D6604D"]
    figure, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))

    signature_uniform = np.asarray(
        [
            result["models"][key]["uniform_axis"][
                "style_signature_energy_fraction"
            ]
            for key in keys
        ]
    )
    axes[0].bar(
        labels,
        signature_uniform * 100,
        color="#BDBDBD",
        label="uniform bias",
    )
    axes[0].bar(
        labels,
        (1 - signature_uniform) * 100,
        bottom=signature_uniform * 100,
        color=colors,
        label="disease contrast",
    )
    axes[0].set_ylabel("Style-signature energy (%)")
    axes[0].set_title("Uniform bias vs disease contrast")
    axes[0].legend(frameon=False, fontsize=7)

    x = np.arange(2)
    width = 0.34
    full_id = [
        result["models"][key]["full_style_test"]["observed"][
            "style_identification_accuracy"
        ]
        * 100
        for key in keys
    ]
    contrast_id = [
        result["models"][key]["joint_nuisance_complement"]["style_test"][
            "observed"
        ][
            "style_identification_accuracy"
        ]
        * 100
        for key in keys
    ]
    axes[1].bar(x - width / 2, full_id, width, color="#999999", label="full")
    axes[1].bar(
        x + width / 2,
        contrast_id,
        width,
        color=colors,
        label="joint nuisance-free",
    )
    axes[1].axhline(100 / 6, color="#222222", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Held-patient style ID (%)")
    axes[1].set_title("Does identification survive?")
    axes[1].legend(frameon=False, fontsize=7)

    cosine = np.asarray(
        result["joint_nuisance_cross_model"]["cosine_matrix"]
    )
    image = axes[2].imshow(cosine, vmin=-1, vmax=1, cmap="RdBu_r")
    for row in range(6):
        for column in range(6):
            axes[2].text(
                column,
                row,
                f"{cosine[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
            )
    axes[2].set_xlabel("Qwen base style")
    axes[2].set_ylabel("Huatuo style")
    axes[2].set_title("Joint-complement transfer")
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)

    for key, label, color in zip(keys, labels, colors):
        spectrum = np.asarray(
            result["models"][key]["joint_nuisance_complement"]["spectrum"][
                "energy_proportions"
            ]
        )
        axes[3].plot(
            np.arange(1, len(spectrum) + 1),
            np.cumsum(spectrum),
            marker="o",
            color=color,
            label=label,
        )
    axes[3].set_ylim(0, 1.04)
    axes[3].set_xlabel("Number of singular directions")
    axes[3].set_ylabel("Cumulative residual energy")
    axes[3].set_title("Joint-complement spectrum")
    axes[3].legend(frameon=False, fontsize=7)

    for axis in (axes[0], axes[1], axes[3]):
        axis.grid(axis="y", alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        "Is the residual style signature disease-specific?",
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
