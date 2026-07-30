"""Compare acquisition-style evidence geometry across matched VLM lineages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anchor.corrected_sgta.analyze_visual_evidence_chord_probe import (
    cosine,
    evidence_table,
    read_jsonl,
)


VERSION = "visual-evidence-chord-lineage-comparison-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def drift_tensor(
    evidence: dict[str, dict[str, np.ndarray]],
    cases: list[str],
    styles: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [
                evidence[case][style] - evidence[case]["real"]
                for style in styles
            ]
            for case in cases
        ],
        dtype=np.float64,
    )


def variance_fractions(drift: np.ndarray) -> np.ndarray:
    """Return patient, style, and patient-by-style variance fractions."""

    grand = drift.mean(axis=(0, 1), keepdims=True)
    case = drift.mean(axis=1, keepdims=True) - grand
    style = drift.mean(axis=0, keepdims=True) - grand
    interaction = drift - grand - case - style
    sums = np.asarray(
        [
            drift.shape[1] * np.square(case).sum(),
            drift.shape[0] * np.square(style).sum(),
            np.square(interaction).sum(),
        ],
        dtype=np.float64,
    )
    return sums / max(float(sums.sum()), 1e-12)


def paired_lineage_comparison(
    first: np.ndarray,
    second: np.ndarray,
    styles: list[str],
    first_visual_scale: np.ndarray | None = None,
    second_visual_scale: np.ndarray | None = None,
    draws: int = 5000,
    seed: int = 2027,
) -> dict:
    """Compare two models on exactly paired cases, styles, and concepts."""

    if first.shape != second.shape:
        raise ValueError(
            f"paired drift tensors differ: {first.shape} != {second.shape}"
        )
    first_fraction = variance_fractions(first)
    second_fraction = variance_fractions(second)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(draws):
        indices = rng.integers(0, first.shape[0], size=first.shape[0])
        bootstrap.append(
            variance_fractions(first[indices])
            - variance_fractions(second[indices])
        )
    bootstrap_array = np.asarray(bootstrap)
    style_directions = {}
    for index, style in enumerate(styles):
        first_mean = first[:, index].mean(axis=0)
        second_mean = second[:, index].mean(axis=0)
        style_directions[style] = {
            "first_offset_norm": float(np.linalg.norm(first_mean)),
            "second_offset_norm": float(np.linalg.norm(second_mean)),
            "cross_model_cosine": cosine(first_mean, second_mean),
        }
    susceptibility = None
    if first_visual_scale is not None and second_visual_scale is not None:
        first_case = (
            np.sqrt(np.square(first).sum(axis=2).mean(axis=1))
            / np.asarray(first_visual_scale).clip(min=1e-12)
        )
        second_case = (
            np.sqrt(np.square(second).sum(axis=2).mean(axis=1))
            / np.asarray(second_visual_scale).clip(min=1e-12)
        )
        observed = float(np.median(first_case) - np.median(second_case))
        differences = []
        for _ in range(draws):
            indices = rng.integers(0, first.shape[0], size=first.shape[0])
            differences.append(
                float(
                    np.median(first_case[indices])
                    - np.median(second_case[indices])
                )
            )
        susceptibility = {
            "definition": (
                "per-patient RMS style evidence drift divided by the "
                "real-null evidence norm"
            ),
            "first_median": float(np.median(first_case)),
            "second_median": float(np.median(second_case)),
            "first_minus_second": observed,
            "paired_case_bootstrap_ci95": [
                float(value)
                for value in np.quantile(differences, [0.025, 0.975])
            ],
            "first_per_case": [float(value) for value in first_case],
            "second_per_case": [float(value) for value in second_case],
        }
    names = ["patient", "style", "patient_by_style"]
    return {
        "first_fraction": dict(zip(names, first_fraction, strict=True)),
        "second_fraction": dict(zip(names, second_fraction, strict=True)),
        "first_minus_second": {
            name: {
                "point": float(first_fraction[index] - second_fraction[index]),
                "paired_case_bootstrap_ci95": [
                    float(value)
                    for value in np.quantile(
                        bootstrap_array[:, index], [0.025, 0.975]
                    )
                ],
            }
            for index, name in enumerate(names)
        },
        "style_offset_alignment": style_directions,
        "normalized_style_susceptibility": susceptibility,
        "bootstrap_draws": draws,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-name", default="medical")
    parser.add_argument("--second-name", default="base")
    parser.add_argument("--view-labels", type=Path)
    parser.add_argument(
        "--view-category", default="a frontal chest radiograph"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()

    first_rows = read_jsonl(args.first)
    second_rows = read_jsonl(args.second)
    selected = None
    if args.view_labels:
        labels = read_jsonl(args.view_labels)
        selected = {
            row["case_id"]
            for row in labels
            if row["predicted_category"] == args.view_category
        }
        first_rows = [
            row for row in first_rows if row["case_id"] in selected
        ]
        second_rows = [
            row for row in second_rows if row["case_id"] in selected
        ]
    first_pack, first_diseases, first_views = evidence_table(first_rows)
    second_pack, second_diseases, second_views = evidence_table(second_rows)
    first_evidence = first_pack["evidence"]
    second_evidence = second_pack["evidence"]
    cases = sorted(set(first_evidence) & set(second_evidence))
    styles = sorted(
        view for view in first_views if view.startswith("style_")
    )
    if first_diseases != second_diseases or first_views != second_views:
        raise RuntimeError("models do not share the same concepts and views")
    if cases != sorted(first_evidence) or cases != sorted(second_evidence):
        raise RuntimeError("models do not share the exact same cases")
    first_drift = drift_tensor(first_evidence, cases, styles)
    second_drift = drift_tensor(second_evidence, cases, styles)
    first_scale = np.asarray(
        [
            np.linalg.norm(
                first_evidence[case]["real"]
                - first_evidence[case]["null"]
            )
            for case in cases
        ]
    )
    second_scale = np.asarray(
        [
            np.linalg.norm(
                second_evidence[case]["real"]
                - second_evidence[case]["null"]
            )
            for case in cases
        ]
    )
    comparison = paired_lineage_comparison(
        first_drift,
        second_drift,
        styles,
        first_visual_scale=first_scale,
        second_visual_scale=second_scale,
        draws=args.bootstrap_draws,
    )
    result = {
        "version": VERSION,
        "first": {
            "name": args.first_name,
            "path": str(args.first.resolve()),
            "sha256": sha256(args.first),
        },
        "second": {
            "name": args.second_name,
            "path": str(args.second.resolve()),
            "sha256": sha256(args.second),
        },
        "n_paired_cases": len(cases),
        "styles": styles,
        "diseases": first_diseases,
        "view_category": args.view_category if selected is not None else None,
        "comparison": comparison,
        "decision_rule": (
            "A medical-training lineage effect requires a positive paired "
            "bootstrap lower bound for the style variance fraction; matching "
            "fractions indicate an architectural or transform-level effect."
        ),
        "decision": {
            "medical_training_increases_style_fraction": bool(
                comparison["first_minus_second"]["style"][
                    "paired_case_bootstrap_ci95"
                ][0]
                > 0
            ),
            "medical_training_reduces_normalized_style_susceptibility": bool(
                comparison["normalized_style_susceptibility"][
                    "paired_case_bootstrap_ci95"
                ][1]
                < 0
            ),
        },
        "claim_ceiling": (
            "paired mechanism diagnostic on exposed MIMIC development images"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    labels = ["Patient", "Style", "Patient×style"]
    first_values = list(comparison["first_fraction"].values())
    second_values = list(comparison["second_fraction"].values())
    locations = np.arange(3)
    figure, axes = plt.subplots(
        1, 2, figsize=(10.6, 3.7), constrained_layout=True
    )
    axes[0].bar(
        locations - 0.18,
        np.asarray(first_values) * 100,
        width=0.36,
        label=args.first_name,
        color="#457b9d",
    )
    axes[0].bar(
        locations + 0.18,
        np.asarray(second_values) * 100,
        width=0.36,
        label=args.second_name,
        color="#e9c46a",
    )
    axes[0].set_xticks(locations, labels)
    axes[0].set_ylabel("Evidence-drift variance (%)")
    axes[0].set_title("Style is not the dominant component")
    axes[0].legend(frameon=False)
    susceptibility = comparison["normalized_style_susceptibility"]
    first_case = np.asarray(susceptibility["first_per_case"])
    second_case = np.asarray(susceptibility["second_per_case"])
    for first_value, second_value in zip(
        first_case, second_case, strict=True
    ):
        axes[1].plot(
            [0, 1],
            [first_value, second_value],
            color="#b0b0b0",
            alpha=0.35,
            linewidth=0.7,
        )
    axes[1].scatter(
        np.zeros_like(first_case),
        first_case,
        color="#457b9d",
        alpha=0.75,
        s=16,
        zorder=3,
    )
    axes[1].scatter(
        np.ones_like(second_case),
        second_case,
        color="#e9c46a",
        alpha=0.75,
        s=16,
        zorder=3,
    )
    axes[1].hlines(
        [np.median(first_case), np.median(second_case)],
        [-0.18, 0.82],
        [0.18, 1.18],
        color=["#1d3557", "#8c6d1f"],
        linewidth=2.4,
    )
    axes[1].set_xticks(
        [0, 1], [args.first_name, args.second_name], rotation=8
    )
    axes[1].set_ylabel("Normalized style susceptibility $\\kappa$")
    axes[1].set_title("Medical training contracts style sensitivity")
    figure.savefig(args.figure, dpi=220)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
