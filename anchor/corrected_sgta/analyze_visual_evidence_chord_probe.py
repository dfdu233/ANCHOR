"""Analyze the visual-evidence chord law against style-specific prior rotation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


VERSION = "visual-evidence-chord-analysis-v2"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator < 1e-12 else float(left @ right / denominator)


def evidence_table(rows: list[dict]) -> tuple[dict, list[str], list[str]]:
    diseases = sorted({row["disease"] for row in rows})
    views = sorted({row["view"] for row in rows})
    values: dict[tuple[str, str, str], dict[str, float]] = {}
    image_metrics: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["case_id"], row["view"], row["disease"])
        values.setdefault(key, {})[row["polarity"]] = float(
            row["sequence_nll"]
        )
        image_metrics[(row["case_id"], row["view"])] = row["image_metrics"]
    cases = sorted({row["case_id"] for row in rows})
    table = {}
    for case in cases:
        table[case] = {}
        for view in views:
            vector = []
            for disease in diseases:
                pair = values[(case, view, disease)]
                vector.append(pair["negative"] - pair["positive"])
            table[case][view] = np.asarray(vector, dtype=np.float64)
    return {
        "evidence": table,
        "image_metrics": image_metrics,
    }, diseases, views


def chord_projection(
    style: np.ndarray,
    real: np.ndarray,
    null: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    direction = real - null
    denominator = float(direction @ direction)
    alpha = (
        0.0
        if denominator < 1e-12
        else float((style - null) @ direction / denominator)
    )
    predicted = null + alpha * direction
    return alpha, predicted, style - predicted


def style_case_variance_decomposition(
    evidence: dict[str, dict[str, np.ndarray]],
    styles: list[str],
) -> dict:
    """Decompose style-induced evidence drift in a balanced two-way design.

    For Delta[i, s] = evidence[i, s] - evidence[i, real], the orthogonal
    ANOVA decomposition is case + style + case-by-style interaction.  The
    style-specific reducible fraction is the maximum fraction of squared
    drift removable by a single additive offset shared across cases.
    """

    cases = sorted(evidence)
    drift = np.asarray(
        [
            [
                evidence[case][style] - evidence[case]["real"]
                for style in styles
            ]
            for case in cases
        ],
        dtype=np.float64,
    )
    grand = drift.mean(axis=(0, 1), keepdims=True)
    case_effect = drift.mean(axis=1, keepdims=True) - grand
    style_effect = drift.mean(axis=0, keepdims=True) - grand
    interaction = drift - grand - case_effect - style_effect
    centered = drift - grand
    sums_of_squares = {
        "case": float(
            len(styles) * np.square(case_effect).sum()
        ),
        "style": float(
            len(cases) * np.square(style_effect).sum()
        ),
        "case_by_style": float(np.square(interaction).sum()),
        "total": float(np.square(centered).sum()),
    }
    total = max(sums_of_squares["total"], 1e-12)
    fractions = {
        name: value / total
        for name, value in sums_of_squares.items()
        if name != "total"
    }
    reducible_by_style = {}
    for index, style in enumerate(styles):
        style_drift = drift[:, index]
        shared_offset = style_drift.mean(axis=0)
        total_energy = float(np.square(style_drift).sum(axis=1).mean())
        reducible_by_style[style] = {
            "shared_offset_norm_squared": float(
                shared_offset @ shared_offset
            ),
            "mean_drift_energy": total_energy,
            "maximum_reducible_fraction": float(
                (shared_offset @ shared_offset)
                / max(total_energy, 1e-12)
            ),
        }
    return {
        "definition": (
            "Delta[i,s] = evidence[i,s] - evidence[i,real]; balanced "
            "two-way Euclidean ANOVA over the complete-sentence evidence "
            "vectors"
        ),
        "sums_of_squares": sums_of_squares,
        "fraction_of_centered_variance": fractions,
        "maximum_global_additive_correction_by_style": reducible_by_style,
        "theorem": (
            "For a fixed style s, the least-squares global additive "
            "correction is E_i[Delta[i,s]], and its maximum reducible "
            "fraction is ||E_i Delta||^2 / E_i||Delta||^2."
        ),
    }


def susceptibility_correlations(
    evidence: dict[str, dict[str, np.ndarray]],
    image_metrics: dict[tuple[str, str], dict],
    styles: list[str],
) -> dict:
    """Test simple, unlabeled predictors of patient-specific style response."""

    cases = sorted(evidence)
    susceptibility = []
    features = {
        "real_null_evidence_norm": [],
        "mean_absolute_visual_evidence": [],
        "mean_absolute_clinical_evidence": [],
        "minimum_absolute_clinical_evidence": [],
        "clinical_evidence_std": [],
        "mean_pixel_change": [],
        "mean_edge_change": [],
    }
    for case in cases:
        real = evidence[case]["real"]
        null = evidence[case]["null"]
        drift = np.asarray(
            [evidence[case][style] - real for style in styles]
        )
        susceptibility.append(float(np.sqrt(np.square(drift).mean())))
        features["real_null_evidence_norm"].append(
            float(np.linalg.norm(real - null))
        )
        features["mean_absolute_visual_evidence"].append(
            float(np.mean(np.abs(real - null)))
        )
        features["mean_absolute_clinical_evidence"].append(
            float(np.mean(np.abs(real)))
        )
        features["minimum_absolute_clinical_evidence"].append(
            float(np.min(np.abs(real)))
        )
        features["clinical_evidence_std"].append(float(np.std(real)))
        features["mean_pixel_change"].append(
            float(
                np.mean(
                    [
                        image_metrics[(case, style)][
                            "mean_absolute_change"
                        ]
                        for style in styles
                    ]
                )
            )
        )
        features["mean_edge_change"].append(
            float(
                np.mean(
                    [
                        1
                        - image_metrics[(case, style)][
                            "edge_correlation"
                        ]
                        for style in styles
                    ]
                )
            )
        )
    response = np.asarray(susceptibility)
    correlations = {}
    for name, values in features.items():
        if np.ptp(values) < 1e-12 or np.ptp(response) < 1e-12:
            correlations[name] = {
                "spearman_rho": 0.0,
                "p_value": 1.0,
            }
        else:
            statistic = spearmanr(values, response)
            correlations[name] = {
                "spearman_rho": float(statistic.statistic),
                "p_value": float(statistic.pvalue),
            }
    return {
        "definition": (
            "patient susceptibility = RMS complete-sentence evidence drift "
            "over source styles and clinical concepts"
        ),
        "median_susceptibility": float(np.median(response)),
        "correlations": correlations,
    }


def cross_validated_mse(
    evidence: dict[str, dict[str, np.ndarray]],
    styles: list[str],
    style_assignment: dict[tuple[str, str], str] | None = None,
) -> dict:
    cases = sorted(evidence)
    assignment = style_assignment or {
        (case, style): style for case in cases for style in styles
    }
    methods = (
        "identity",
        "chord",
        "style_offset",
        "diagonal_filter",
        "full_linear_filter",
    )
    errors = {method: [] for method in methods}
    per_case = {}
    for holdout in cases:
        train = [case for case in cases if case != holdout]
        alpha_by_label: dict[str, float] = {}
        offset_by_label: dict[str, np.ndarray] = {}
        diagonal_by_label: dict[str, np.ndarray] = {}
        linear_by_label: dict[str, np.ndarray] = {}
        for label in styles:
            numerator = 0.0
            denominator = 0.0
            residuals = []
            directions = []
            shifted_vectors = []
            for case in train:
                direction = evidence[case]["real"] - evidence[case]["null"]
                for style in styles:
                    if assignment[(case, style)] != label:
                        continue
                    shifted = (
                        evidence[case][style] - evidence[case]["null"]
                    )
                    numerator += float(shifted @ direction)
                    denominator += float(direction @ direction)
                    directions.append(direction)
                    shifted_vectors.append(shifted)
            alpha = numerator / max(denominator, 1e-12)
            alpha_by_label[label] = alpha
            for case in train:
                direction = evidence[case]["real"] - evidence[case]["null"]
                for style in styles:
                    if assignment[(case, style)] != label:
                        continue
                    shifted = (
                        evidence[case][style] - evidence[case]["null"]
                    )
                    residuals.append(shifted - alpha * direction)
            offset_by_label[label] = np.mean(residuals, axis=0)
            design = np.asarray(directions)
            targets = np.asarray(shifted_vectors)
            diagonal_by_label[label] = (
                (design * targets).sum(axis=0)
                / np.square(design).sum(axis=0).clip(min=1e-12)
            )
            gram = design.T @ design
            ridge = 1e-3 * float(np.trace(gram)) / max(gram.shape[0], 1)
            linear_by_label[label] = np.linalg.solve(
                gram + ridge * np.eye(gram.shape[0]),
                design.T @ targets,
            )
        held = {method: [] for method in methods}
        real = evidence[holdout]["real"]
        null = evidence[holdout]["null"]
        direction = real - null
        for style in styles:
            target = evidence[holdout][style]
            # The held-out style identity is observed.  Only source cases are
            # shuffled in the negative control.
            label = style
            chord = null + alpha_by_label[label] * direction
            predictions = {
                "identity": real,
                "chord": chord,
                "style_offset": chord + offset_by_label[label],
                "diagonal_filter": (
                    null + diagonal_by_label[label] * direction
                ),
                "full_linear_filter": (
                    null + direction @ linear_by_label[label]
                ),
            }
            for method, prediction in predictions.items():
                held[method].append(float(np.mean((target - prediction) ** 2)))
        per_case[holdout] = {
            method: float(np.mean(values))
            for method, values in held.items()
        }
        for method in errors:
            errors[method].append(per_case[holdout][method])
    return {
        "mean": {
            method: float(np.mean(values))
            for method, values in errors.items()
        },
        "per_case": per_case,
    }


def bootstrap_interval(
    values: np.ndarray,
    draws: int = 5000,
    seed: int = 2027,
) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [
        float(rng.choice(values, size=len(values), replace=True).mean())
        for _ in range(draws)
    ]
    return [
        float(value)
        for value in np.quantile(means, [0.025, 0.975])
    ]


def permutation_test(
    evidence: dict[str, dict[str, np.ndarray]],
    styles: list[str],
    method: str,
    observed_improvement: float,
    draws: int = 1000,
) -> tuple[float, list[float]]:
    rng = np.random.default_rng(2027)
    improvements = []
    for _ in range(draws):
        assignment = {}
        for case in sorted(evidence):
            shuffled = list(styles)
            rng.shuffle(shuffled)
            assignment.update(
                {
                    (case, style): label
                    for style, label in zip(styles, shuffled, strict=True)
                }
            )
        result = cross_validated_mse(evidence, styles, assignment)
        improvements.append(
            result["mean"]["chord"] - result["mean"][method]
        )
    p_value = (
        1
        + sum(value >= observed_improvement for value in improvements)
    ) / (draws + 1)
    return float(p_value), improvements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--view-labels", type=Path)
    parser.add_argument(
        "--view-category", default="a frontal chest radiograph"
    )
    parser.add_argument("--permutation-draws", type=int, default=200)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    selected_cases = None
    view_audit = None
    if args.view_labels:
        labels = read_jsonl(args.view_labels)
        selected_cases = {
            row["case_id"]
            for row in labels
            if row["predicted_category"] == args.view_category
        }
        rows = [row for row in rows if row["case_id"] in selected_cases]
        view_audit = {
            "path": str(args.view_labels.resolve()),
            "sha256": sha256(args.view_labels),
            "selected_category": args.view_category,
            "selected_cases": len(selected_cases),
            "total_cases": len({row["case_id"] for row in labels}),
        }
    packed, diseases, views = evidence_table(rows)
    evidence = packed["evidence"]
    styles = [view for view in views if view.startswith("style_")]
    diagnostics = []
    for case in sorted(evidence):
        real = evidence[case]["real"]
        null = evidence[case]["null"]
        null_direction = null - real
        for style in styles:
            styled = evidence[case][style]
            alpha, _, residual = chord_projection(styled, real, null)
            shifted = styled - real
            diagnostics.append(
                {
                    "case": case,
                    "style": style,
                    "alpha": alpha,
                    "attenuation_cosine": cosine(shifted, null_direction),
                    "relative_style_shift": float(
                        np.linalg.norm(shifted)
                        / max(np.linalg.norm(real - null), 1e-12)
                    ),
                    "orthogonal_fraction": float(
                        np.linalg.norm(residual)
                        / max(np.linalg.norm(styled - null), 1e-12)
                    ),
                }
            )
    cv = cross_validated_mse(evidence, styles)
    variance_decomposition = style_case_variance_decomposition(
        evidence, styles
    )
    susceptibility = susceptibility_correlations(
        evidence, packed["image_metrics"], styles
    )
    case_order = sorted(evidence)
    chord_minus_style = np.asarray(
        [
            cv["per_case"][case]["chord"]
            - cv["per_case"][case]["style_offset"]
            for case in case_order
        ]
    )
    style_improvement = float(chord_minus_style.mean())
    style_interval = bootstrap_interval(chord_minus_style)
    permutation_p, permutation_improvements = permutation_test(
        evidence,
        styles,
        "style_offset",
        style_improvement,
        draws=args.permutation_draws,
    )
    diagonal_difference = np.asarray(
        [
            cv["per_case"][case]["chord"]
            - cv["per_case"][case]["diagonal_filter"]
            for case in case_order
        ]
    )
    diagonal_improvement = float(diagonal_difference.mean())
    diagonal_interval = bootstrap_interval(diagonal_difference)
    diagonal_p, diagonal_permutations = permutation_test(
        evidence,
        styles,
        "diagonal_filter",
        diagonal_improvement,
        draws=args.permutation_draws,
    )
    identity = cv["mean"]["identity"]
    chord = cv["mean"]["chord"]
    style_offset = cv["mean"]["style_offset"]
    content_metrics = [
        packed["image_metrics"][(case, style)]
        for case in sorted(evidence)
        for style in styles
    ]
    cosine_values = np.asarray(
        [item["attenuation_cosine"] for item in diagnostics]
    )
    relative_values = np.asarray(
        [item["relative_style_shift"] for item in diagnostics]
    )
    result = {
        "version": VERSION,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "n_cases": len(evidence),
        "n_styles": len(styles),
        "diseases": diseases,
        "styles": styles,
        "view_audit": view_audit,
        "definition": (
            "evidence = mean log p(positive complete sentence) - mean log "
            "p(negative complete sentence)"
        ),
        "content_preservation": {
            "median_pixel_correlation": float(
                np.median(
                    [item["pixel_correlation"] for item in content_metrics]
                )
            ),
            "median_edge_correlation": float(
                np.median(
                    [item["edge_correlation"] for item in content_metrics]
                )
            ),
            "median_mean_absolute_change": float(
                np.median(
                    [item["mean_absolute_change"] for item in content_metrics]
                )
            ),
        },
        "direct_geometry": {
            "median_attenuation_cosine": float(np.median(cosine_values)),
            "attenuation_cosine_ci95": bootstrap_interval(cosine_values),
            "median_relative_style_shift": float(
                np.median(relative_values)
            ),
            "median_orthogonal_fraction": float(
                np.median(
                    [item["orthogonal_fraction"] for item in diagnostics]
                )
            ),
        },
        "leave_one_image_out_mse": cv["mean"],
        "style_offset_over_chord": {
            "mse_improvement": style_improvement,
            "relative_improvement": (
                style_improvement / max(chord, 1e-12)
            ),
            "image_bootstrap_ci95": style_interval,
            "style_label_permutation_p": permutation_p,
            "permutation_draws": args.permutation_draws,
            "permutation_mean": float(
                np.mean(permutation_improvements)
            ),
        },
        "concept_filter_over_chord": {
            "mse_improvement": diagonal_improvement,
            "relative_improvement": (
                diagonal_improvement / max(chord, 1e-12)
            ),
            "image_bootstrap_ci95": diagonal_interval,
            "style_label_permutation_p": diagonal_p,
            "permutation_draws": args.permutation_draws,
            "permutation_mean": float(np.mean(diagonal_permutations)),
        },
        "style_case_variance_decomposition": variance_decomposition,
        "patient_style_susceptibility": susceptibility,
        "decision": {
            "intervention_nontrivial": bool(
                np.median(relative_values) >= 0.05
            ),
            "content_preserved": bool(
                np.median(
                    [item["edge_correlation"] for item in content_metrics]
                )
                >= 0.90
            ),
            "scalar_attenuation_supported": bool(
                chord <= 0.90 * identity
                and np.median(cosine_values) >= 0.50
                and not (
                    style_interval[0] > 0
                    and style_improvement >= 0.10 * chord
                )
            ),
            "style_prior_rotation_supported": bool(
                style_interval[0] > 0
                and style_improvement >= 0.10 * chord
                and permutation_p < 0.05
            ),
            "concept_selective_filter_supported": bool(
                diagonal_interval[0] > 0
                and diagonal_improvement >= 0.10 * chord
                and diagonal_p < 0.05
            ),
        },
        "claim_ceiling": (
            "mechanism diagnostic on fixed MIMIC development images; "
            "teacher-forced sentence likelihoods are not predictions"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1, 4, figsize=(14.5, 3.4), constrained_layout=True
    )
    axes[0].hist(cosine_values, bins=12, color="#33658a")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("cos(style−real, null−real)")
    axes[0].set_ylabel("Paired views")
    axes[0].set_title("Evidence attenuation")
    axes[1].bar(
        ["Identity", "Scalar\nchord", "Style\noffset", "Concept\nfilter"],
        [
            identity,
            chord,
            style_offset,
            cv["mean"]["diagonal_filter"],
        ],
        color=["#9e9e9e", "#2a9d8f", "#e76f51", "#6a4c93"],
    )
    axes[1].set_ylabel("Leave-one-image-out MSE")
    axes[1].set_title("Competing geometric models")
    matrix = np.asarray(
        [
            [
                next(
                    item["orthogonal_fraction"]
                    for item in diagnostics
                    if item["case"] == case and item["style"] == style
                )
                for style in styles
            ]
            for case in sorted(evidence)
        ]
    )
    image = axes[2].imshow(matrix, aspect="auto", cmap="magma", vmin=0)
    axes[2].set_xticks(range(len(styles)), styles, rotation=45, ha="right")
    axes[2].set_yticks(range(len(evidence)))
    axes[2].set_yticklabels(sorted(evidence), fontsize=7)
    axes[2].set_title("Chord-orthogonal fraction")
    figure.colorbar(image, ax=axes[2], fraction=0.046)
    fractions = variance_decomposition[
        "fraction_of_centered_variance"
    ]
    axes[3].bar(
        ["Patient", "Style", "Patient×style"],
        [
            100 * fractions["case"],
            100 * fractions["style"],
            100 * fractions["case_by_style"],
        ],
        color=["#457b9d", "#e9c46a", "#e76f51"],
    )
    axes[3].set_ylabel("Explained evidence drift (%)")
    axes[3].set_title("What determines style response?")
    axes[3].tick_params(axis="x", rotation=22)
    figure.savefig(args.figure, dpi=220)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
