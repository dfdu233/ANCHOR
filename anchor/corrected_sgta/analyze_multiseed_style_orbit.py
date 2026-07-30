"""Confirm a prespecified style-orbit effect across training lineages.

The exploratory layerwise audit identified ``llm_27_prompt`` as a candidate
mechanism location. This module deliberately does not search across layers.
It tests that one endpoint across independently trained matched/image-permuted
lineages and uses a crossed seed-by-patient bootstrap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from anchor.corrected_sgta.analyze_layerwise_style_orbit import (
    per_case_components,
)


VERSION = "multiseed-style-orbit-confirmation-v1"
PRIMARY_LAYER = "llm_27_prompt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run(specification: str) -> tuple[str, Path, str, str]:
    """Parse ``label[,matched-name,permuted-name]=features.npz``."""
    if "=" not in specification:
        raise ValueError("run must have form label=/path/to/features.npz")
    raw_names, raw_path = specification.split("=", 1)
    names = [name.strip() for name in raw_names.split(",")]
    if len(names) == 1:
        label = names[0]
        matched_name, permuted_name = "matched", "permuted"
    elif len(names) == 3:
        label, matched_name, permuted_name = names
    else:
        raise ValueError(
            "run names must be label or label,matched-name,permuted-name"
        )
    path = Path(raw_path).expanduser().resolve()
    if not label or any(character.isspace() for character in label):
        raise ValueError(f"invalid run label: {label!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = path.with_suffix(".json")
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    if not matched_name or not permuted_name or matched_name == permuted_name:
        raise ValueError("matched and permuted variant names must be distinct")
    return label, path, matched_name, permuted_name


def patient_effects(
    matched: np.ndarray,
    permuted: np.ndarray,
    patients: list[str],
) -> tuple[list[str], np.ndarray]:
    """Average repeated images, then return relative effects per patient."""
    if matched.shape != permuted.shape or matched.ndim != 1:
        raise ValueError("matched and permuted must be paired 1-D arrays")
    if matched.shape[0] != len(patients):
        raise ValueError("patient labels do not match paired observations")
    relative = (matched - permuted) / np.maximum(np.abs(permuted), 1e-12)
    unique = sorted(set(patients))
    effects = np.asarray(
        [
            np.mean(
                relative[
                    np.asarray(
                        [
                            index
                            for index, patient in enumerate(patients)
                            if patient == target
                        ]
                    )
                ]
            )
            for target in unique
        ],
        dtype=np.float64,
    )
    return unique, effects


def crossed_seed_patient_bootstrap(
    effects: np.ndarray,
    draws: int = 20_000,
    seed: int = 2027,
) -> tuple[float, list[float], np.ndarray]:
    """Bootstrap independent training lineages and patient clusters."""
    if effects.ndim != 2 or min(effects.shape) < 2:
        raise ValueError("effects must contain at least two seeds and patients")
    point = float(np.mean(np.median(effects, axis=1)))
    generator = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        seeds = generator.integers(0, effects.shape[0], size=effects.shape[0])
        patients = generator.integers(
            0, effects.shape[1], size=effects.shape[1]
        )
        sampled = effects[seeds][:, patients]
        values[index] = np.mean(np.median(sampled, axis=1))
    interval = [float(value) for value in np.quantile(values, [0.025, 0.975])]
    return point, interval, values


def load_run(
    label: str,
    path: Path,
    matched_name: str,
    permuted_name: str,
) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    variants = [row["name"] for row in metadata["variants"]]
    if not {matched_name, permuted_name}.issubset(variants):
        raise ValueError(
            f"{label}: missing {matched_name}/{permuted_name}, got {variants}"
        )
    if PRIMARY_LAYER not in metadata["layers"]:
        raise ValueError(f"{label}: missing prespecified {PRIMARY_LAYER}")
    data = np.load(path)
    tensor = data[PRIMARY_LAYER].astype(np.float32)
    if tensor.shape[0] != len(variants):
        raise ValueError(f"{label}: variant dimension mismatch")
    values: dict[str, dict[str, np.ndarray]] = {}
    requested = {matched_name: "matched", permuted_name: "permuted"}
    for variant_index, variant in enumerate(variants):
        if variant not in requested:
            continue
        style, leverage, kappa = per_case_components(tensor[variant_index])
        values[requested[variant]] = {
            "style_drift": style,
            "real_null_leverage": leverage,
            "kappa": kappa,
        }
    patients = [
        row["patient_id"] for row in metadata["rows"] if row["view"] == "real"
    ]
    case_ids = [
        row["case_id"] for row in metadata["rows"] if row["view"] == "real"
    ]
    return {
        "label": label,
        "path": path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "patients": patients,
        "case_ids": case_ids,
        "values": values,
        "matched_variant": matched_name,
        "permuted_variant": permuted_name,
    }


def metric_confirmation(
    runs: list[dict[str, Any]], metric: str, draws: int, seed: int
) -> dict[str, Any]:
    patient_order: list[str] | None = None
    seed_effects = []
    per_seed = {}
    for run in runs:
        patients, effects = patient_effects(
            run["values"]["matched"][metric],
            run["values"]["permuted"][metric],
            run["patients"],
        )
        if patient_order is None:
            patient_order = patients
        elif patients != patient_order:
            raise ValueError("runs do not share the exact patient clusters")
        seed_effects.append(effects)
        per_seed[run["label"]] = {
            "patient_median_relative_effect": float(np.median(effects)),
            "matched_median": float(
                np.median(run["values"]["matched"][metric])
            ),
            "permuted_median": float(
                np.median(run["values"]["permuted"][metric])
            ),
            "patients": len(patients),
        }
    matrix = np.stack(seed_effects)
    point, interval, bootstrap = crossed_seed_patient_bootstrap(
        matrix, draws=draws, seed=seed
    )
    seed_points = np.median(matrix, axis=1)
    seed_mean = float(np.mean(seed_points))
    seed_sd = float(np.std(seed_points, ddof=1))
    half_width = float(
        student_t.ppf(0.975, df=len(seed_points) - 1)
        * seed_sd
        / np.sqrt(len(seed_points))
    )
    negative_count = int(np.sum(seed_points < 0))
    sign_test_p = float(
        sum(
            math.comb(len(seed_points), count)
            for count in range(negative_count, len(seed_points) + 1)
        )
        / (2 ** len(seed_points))
    )
    return {
        "estimand": (
            "mean across training lineages of the within-lineage median "
            "patient relative effect (matched - permuted) / |permuted|"
        ),
        "point": point,
        "crossed_seed_patient_bootstrap_ci95": interval,
        "lineages_negative": negative_count,
        "lineages_total": int(len(seed_points)),
        "seed_point_mean": seed_mean,
        "seed_point_sd": seed_sd,
        "seed_t_ci95": [seed_mean - half_width, seed_mean + half_width],
        "one_sided_exact_sign_test_p": sign_test_p,
        "bootstrap_probability_negative": float(np.mean(bootstrap < 0)),
        "per_seed": per_seed,
    }


def plot_confirmation(result: dict[str, Any], output: Path) -> None:
    metrics = [
        ("kappa", r"Normalized susceptibility $\kappa$"),
        ("style_drift", "Style drift"),
        ("real_null_leverage", "Visual leverage"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    palette = ["#2166AC", "#67A9CF", "#D1E5F0", "#FDDBC7", "#B2182B"]
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = result["metrics_all_lineages"][metric]
        confirmation = result["metrics_confirmation_only"][metric]
        labels = list(values["per_seed"])
        points = np.asarray(
            [
                values["per_seed"][label]["patient_median_relative_effect"]
                for label in labels
            ]
        )
        axis.scatter(
            np.arange(len(points)),
            points * 100,
            s=45,
            color=palette[: len(points)],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        axis.axhline(0, color="black", linewidth=0.8)
        aggregate = confirmation["point"] * 100
        lower, upper = (
            np.asarray(
                confirmation["crossed_seed_patient_bootstrap_ci95"]
            )
            * 100
        )
        axis.errorbar(
            [len(points) + 0.35],
            [aggregate],
            yerr=[[aggregate - lower], [upper - aggregate]],
            fmt="D",
            color="#1B7837",
            capsize=4,
            markersize=6,
            label="New-lineage bootstrap",
        )
        axis.set_xticks(
            list(range(len(points))) + [len(points) + 0.35],
            labels + ["New"],
            rotation=32,
            ha="right",
        )
        axis.set_title(title)
        axis.set_ylabel("Matched vs permuted (%)")
        axis.grid(axis="y", alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Exploratory seed 42 and prespecified late-fusion replication",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--discovery-label",
        action="append",
        default=[],
        help="Exploratory lineage excluded from confirmatory inference.",
    )
    args = parser.parse_args()
    parsed = [parse_run(specification) for specification in args.run]
    if len(parsed) < 2:
        raise ValueError("confirmation requires at least two lineages")
    if len({label for label, *_ in parsed}) != len(parsed):
        raise ValueError("run labels must be unique")
    runs = [load_run(*run) for run in parsed]
    unknown_discovery = sorted(
        set(args.discovery_label) - {run["label"] for run in runs}
    )
    if unknown_discovery:
        raise ValueError(f"unknown discovery labels: {unknown_discovery}")
    confirmation_runs = [
        run for run in runs if run["label"] not in args.discovery_label
    ]
    if len(confirmation_runs) < 2:
        raise ValueError("confirmation requires at least two new lineages")
    reference_cases = runs[0]["case_ids"]
    for run in runs[1:]:
        if run["case_ids"] != reference_cases:
            raise ValueError("runs do not share the exact case manifest")

    metrics_all = {
        metric: metric_confirmation(runs, metric, args.draws, args.seed + index)
        for index, metric in enumerate(
            ("kappa", "style_drift", "real_null_leverage")
        )
    }
    metrics_confirmation = {
        metric: metric_confirmation(
            confirmation_runs, metric, args.draws, args.seed + 10 + index
        )
        for index, metric in enumerate(
            ("kappa", "style_drift", "real_null_leverage")
        )
    }
    primary = metrics_confirmation["kappa"]
    output = {
        "version": VERSION,
        "primary_endpoint": PRIMARY_LAYER,
        "primary_metric": (
            "style RMS displacement divided by real-null RMS displacement"
        ),
        "discovery_labels_excluded_from_confirmation": args.discovery_label,
        "runs": [
            {
                "label": run["label"],
                "features": str(run["path"]),
                "features_sha256": sha256(run["path"]),
                "metadata": str(run["metadata_path"]),
                "metadata_fingerprint": run["metadata"]["fingerprint"],
                "matched_variant": run["matched_variant"],
                "permuted_variant": run["permuted_variant"],
            }
            for run in runs
        ],
        "cases": len(reference_cases),
        "patients": len(set(runs[0]["patients"])),
        "metrics_all_lineages": metrics_all,
        "metrics_confirmation_only": metrics_confirmation,
        "decision": {
            "all_confirmation_lineages_contract": (
                primary["lineages_negative"] == primary["lineages_total"]
            ),
            "crossed_ci_excludes_zero": (
                primary["crossed_seed_patient_bootstrap_ci95"][1] < 0
            ),
            "prespecified_effect_replicated": (
                primary["lineages_negative"] == primary["lineages_total"]
                and primary["crossed_seed_patient_bootstrap_ci95"][1] < 0
            ),
        },
        "claim_ceiling": (
            "multi-lineage representation mechanism confirmation on exposed "
            "MIMIC development images; no generated-answer utility or external "
            "domain-generalization claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    plot_confirmation(output, args.figure)
    print(json.dumps({"decision": output["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
