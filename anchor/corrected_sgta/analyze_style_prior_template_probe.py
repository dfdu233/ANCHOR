"""Test whether residual style-prior directions survive answer rephrasing."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from anchor.corrected_sgta.analyze_clinical_evidence_attractor import (
    patient_from_image,
    sha256,
)
from anchor.corrected_sgta.analyze_style_prior_specificity import (
    exact_disease_assignment,
    joint_nuisance_residual,
    patient_cluster_bootstrap_alignment,
)
from anchor.corrected_sgta.analyze_residual_style_signature import (
    cross_model_signature,
    patient_blocked_style_permutation,
)


VERSION = "style-prior-template-invariance-v1"
REFERENCE_TEMPLATE = "original"
STYLE_CHANCE = 1.0 / 6.0
MIN_GATE_COSINE = 0.50


def load_template_evidence(
    path: Path,
    template_id: str | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[str], list[str], str, dict]:
    """Load a complete balanced case×view×disease evidence tensor."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if template_id is not None:
        rows = [
            row
            for row in rows
            if str(row.get("template_id", REFERENCE_TEMPLATE)) == template_id
        ]
    if not rows:
        raise ValueError(f"no rows for template {template_id!r} in {path}")
    templates = {
        str(row.get("template_id", REFERENCE_TEMPLATE)) for row in rows
    }
    if len(templates) != 1:
        raise ValueError(f"expected one template, found {sorted(templates)}")
    resolved_template = next(iter(templates))
    cases = sorted({str(row["case_id"]) for row in rows})
    diseases = sorted({str(row["disease"]) for row in rows})
    styles = sorted(
        {
            str(row["view"])
            for row in rows
            if str(row["view"]).startswith("style_")
        },
        key=lambda value: int(value.split("_")[-1]),
    )
    views = ["real", "null", *styles]
    case_index = {value: index for index, value in enumerate(cases)}
    disease_index = {value: index for index, value in enumerate(diseases)}
    view_index = {value: index for index, value in enumerate(views)}
    nll: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    patients: dict[str, str] = {}
    token_counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    models = set()
    for row in rows:
        key = (str(row["case_id"]), str(row["view"]), str(row["disease"]))
        polarity = str(row["polarity"])
        if polarity in nll[key]:
            raise ValueError(f"duplicate polarity cell: {key}/{polarity}")
        nll[key][polarity] = float(row["sequence_nll"])
        token_counts[(str(row["disease"]), polarity)].append(
            int(row["answer_token_count"])
        )
        patients[str(row["case_id"])] = patient_from_image(
            str(row["image_relative"])
        )
        models.add(str(row["model"]))
    if len(models) != 1:
        raise ValueError(f"expected one model, found {sorted(models)}")
    evidence = np.full(
        (len(cases), len(views), len(diseases)), np.nan, dtype=np.float64
    )
    for (case, view, disease), values in nll.items():
        if set(values) != {"positive", "negative"}:
            raise ValueError(f"incomplete polarity pair: {case}/{view}/{disease}")
        evidence[
            case_index[case], view_index[view], disease_index[disease]
        ] = values["negative"] - values["positive"]
    if not np.isfinite(evidence).all():
        raise ValueError("evidence tensor is incomplete")
    counts = {
        disease: {
            polarity: float(np.mean(token_counts[(disease, polarity)]))
            for polarity in ("positive", "negative")
        }
        for disease in diseases
    }
    return (
        evidence,
        cases,
        [patients[case] for case in cases],
        diseases,
        styles,
        next(iter(models)),
        {
            "template_id": resolved_template,
            "mean_answer_token_counts": counts,
        },
    )


def subset_cases(
    evidence: np.ndarray,
    cases: list[str],
    requested: list[str],
) -> np.ndarray:
    lookup = {case: index for index, case in enumerate(cases)}
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise ValueError(f"missing paired cases: {missing[:3]}")
    return evidence[[lookup[case] for case in requested]]


def held_patient_template_style_identification(
    first: np.ndarray,
    second: np.ndarray,
    patients: list[str],
    repeats: int,
    seed: int,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Classify style in one template using prototypes from another template."""
    if first.shape != second.shape or first.shape[0] != len(patients):
        raise ValueError("paired fields and patients do not align")
    patient_array = np.asarray(patients)
    groups = sorted(set(patients))
    group_rows = {
        group: np.flatnonzero(patient_array == group) for group in groups
    }

    def accuracy(source: np.ndarray, target: np.ndarray) -> float:
        predictions = []
        truth = []
        for case_index, patient in enumerate(patients):
            eligible = patient_array != patient
            prototypes = source[eligible].mean(axis=0)
            prototype_unit = prototypes / (
                np.linalg.norm(prototypes, axis=1, keepdims=True) + eps
            )
            target_unit = target[case_index] / (
                np.linalg.norm(target[case_index], axis=1, keepdims=True) + eps
            )
            predictions.extend(
                (target_unit @ prototype_unit.T).argmax(axis=1).tolist()
            )
            truth.extend(range(target.shape[1]))
        return float(np.mean(np.asarray(predictions) == np.asarray(truth)))

    forward = accuracy(first, second)
    reverse = accuracy(second, first)
    observed = 0.5 * (forward + reverse)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(repeats):
        permuted = second.copy()
        for patient in groups:
            permutation = rng.permutation(second.shape[1])
            permuted[group_rows[patient]] = permuted[group_rows[patient]][
                :, permutation
            ]
        null.append(0.5 * (accuracy(first, permuted) + accuracy(permuted, first)))
    null_array = np.asarray(null, dtype=np.float64)
    return {
        "forward_accuracy": forward,
        "reverse_accuracy": reverse,
        "symmetric_accuracy": observed,
        "chance": 1.0 / first.shape[1],
        "patient_blocked_null_mean": float(null_array.mean()),
        "patient_blocked_null_95pct": [
            float(value) for value in np.quantile(null_array, [0.025, 0.975])
        ],
        "one_sided_p": float(
            (1 + np.sum(null_array >= observed)) / (repeats + 1)
        ),
    }


def pair_metrics(
    reference: np.ndarray,
    alternative: np.ndarray,
    patients: list[str],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Compute pre-registered template-pair direction tests."""
    style = cross_model_signature(reference, alternative)
    disease = exact_disease_assignment(reference, alternative)
    bootstrap = patient_cluster_bootstrap_alignment(
        reference, alternative, patients, repeats, seed
    )
    identification = held_patient_template_style_identification(
        reference, alternative, patients, repeats, seed + 1009
    )
    return {
        "style_direction": style,
        "disease_profile": disease,
        "patient_cluster_bootstrap": bootstrap,
        "held_patient_cross_template_style_id": identification,
    }


def analyze(
    reference_path: Path,
    probe_path: Path,
    repeats: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    (
        reference_evidence,
        reference_cases,
        reference_patients,
        reference_diseases,
        reference_styles,
        reference_model,
        reference_meta,
    ) = load_template_evidence(reference_path)
    probe_rows = [
        json.loads(line) for line in probe_path.read_text().splitlines() if line
    ]
    template_ids = sorted({str(row["template_id"]) for row in probe_rows})
    if len(template_ids) < 2:
        raise ValueError("probe must contain at least two alternative templates")
    common_cases = sorted(
        set(reference_cases)
        & {
            str(row["case_id"])
            for row in probe_rows
            if str(row["template_id"]) in template_ids
        }
    )
    if len(common_cases) < 8:
        raise ValueError("fewer than eight paired cases")
    reference = subset_cases(
        reference_evidence, reference_cases, common_cases
    )
    patient_lookup = dict(zip(reference_cases, reference_patients, strict=True))
    patients = [patient_lookup[case] for case in common_cases]
    reference_field, reference_removed = joint_nuisance_residual(
        reference, patients
    )
    fields = {REFERENCE_TEMPLATE: reference_field}
    within = {
        REFERENCE_TEMPLATE: {
            "joint_nuisance_removed_energy": reference_removed,
            "style_reproducibility": patient_blocked_style_permutation(
                reference_field, patients, repeats, seed
            ),
            "metadata": reference_meta,
        }
    }
    pairs = {}
    models = {reference_model}
    for offset, template_id in enumerate(template_ids):
        (
            evidence,
            cases,
            template_patients,
            diseases,
            styles,
            model,
            metadata,
        ) = load_template_evidence(probe_path, template_id)
        if diseases != reference_diseases or styles != reference_styles:
            raise ValueError("template disease/style axes do not match")
        template_patient_lookup = dict(
            zip(cases, template_patients, strict=True)
        )
        if [template_patient_lookup[case] for case in common_cases] != patients:
            raise ValueError("template patient pairing does not match")
        aligned = subset_cases(evidence, cases, common_cases)
        field, removed = joint_nuisance_residual(aligned, patients)
        fields[template_id] = field
        models.add(model)
        within[template_id] = {
            "joint_nuisance_removed_energy": removed,
            "style_reproducibility": patient_blocked_style_permutation(
                field, patients, repeats, seed + 17 + offset
            ),
            "metadata": metadata,
        }
        pairs[f"{REFERENCE_TEMPLATE}_vs_{template_id}"] = pair_metrics(
            reference_field,
            field,
            patients,
            repeats,
            seed + 101 + offset,
        )
    if len(models) != 1:
        raise ValueError(f"reference and probe models differ: {sorted(models)}")

    pair_values = list(pairs.values())
    matched_cosines = [
        row["style_direction"]["matched_style_mean_cosine"]
        for row in pair_values
    ]
    bootstrap_lowers = [
        row["patient_cluster_bootstrap"]["matched_style_mean_cosine"][
            "patient_cluster_bootstrap_95pct"
        ][0]
        for row in pair_values
    ]
    style_id_p = [
        row["held_patient_cross_template_style_id"]["one_sided_p"]
        for row in pair_values
    ]
    gate = {
        "pre_registered_rules": {
            "each_matched_style_cosine_gt": MIN_GATE_COSINE,
            "each_patient_bootstrap_lower_gt": 0.0,
            "each_cross_template_style_id_p_le": 0.05,
            "each_cross_template_style_id_gt_chance": STYLE_CHANCE,
        },
        "matched_style_cosines": matched_cosines,
        "patient_bootstrap_lowers": bootstrap_lowers,
        "cross_template_style_id_p_values": style_id_p,
        "passed": bool(
            all(value > MIN_GATE_COSINE for value in matched_cosines)
            and all(value > 0 for value in bootstrap_lowers)
            and all(value <= 0.05 for value in style_id_p)
            and all(
                row["held_patient_cross_template_style_id"][
                    "symmetric_accuracy"
                ]
                > STYLE_CHANCE
                for row in pair_values
            )
        ),
    }
    summary = {
        "version": VERSION,
        "scope": {
            "mechanism_only": True,
            "teacher_forced_complete_sentences_not_predictions": True,
            "exposed_target_development_images": True,
            "target_labels_used": False,
            "claim_boundary": (
                "Tests lexical/template invariance of a synthetic-style "
                "evidence signature; it does not establish a clinical prior "
                "switch or justify an intervention."
            ),
        },
        "inputs": {
            "reference": str(reference_path.resolve()),
            "reference_sha256": sha256(reference_path),
            "probe": str(probe_path.resolve()),
            "probe_sha256": sha256(probe_path),
        },
        "model": next(iter(models)),
        "cases": len(common_cases),
        "patients": len(set(patients)),
        "diseases": reference_diseases,
        "styles": reference_styles,
        "templates": [REFERENCE_TEMPLATE, *template_ids],
        "within_template": within,
        "cross_template": pairs,
        "gate": gate,
    }
    return summary, fields


def plot_summary(
    summary: dict[str, Any],
    fields: dict[str, np.ndarray],
    output: Path,
) -> None:
    templates = summary["templates"]
    reference = templates[0]
    alternatives = templates[1:]
    figure, axes = plt.subplots(
        2, len(alternatives), figsize=(5.4 * len(alternatives), 8.0)
    )
    if len(alternatives) == 1:
        axes = np.asarray(axes)[:, None]
    for column, alternative in enumerate(alternatives):
        pair = summary["cross_template"][f"{reference}_vs_{alternative}"]
        style_cosine = np.asarray(pair["style_direction"]["cosine_matrix"])
        disease_cosine = np.asarray(
            pair["disease_profile"]["disease_cosine_matrix"]
        )
        for row, (matrix, labels, title) in enumerate(
            (
                (
                    style_cosine,
                    summary["styles"],
                    f"{reference} → {alternative}: style directions",
                ),
                (
                    disease_cosine,
                    summary["diseases"],
                    f"{reference} → {alternative}: disease profiles",
                ),
            )
        ):
            axis = axes[row, column]
            image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
            axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
            axis.set_yticks(range(len(labels)), labels)
            axis.set_title(title)
            for y, x in itertools.product(
                range(matrix.shape[0]), range(matrix.shape[1])
            ):
                axis.text(
                    x,
                    y,
                    f"{matrix[y, x]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        "Does the residual style signature survive equivalent wording?",
        fontsize=14,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    summary, fields = analyze(
        args.reference, args.probe, args.repeats, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    plot_summary(summary, fields, args.figure)
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()
