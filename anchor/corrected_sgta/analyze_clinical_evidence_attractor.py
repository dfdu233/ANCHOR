"""Audit centroid contraction in complete-sentence clinical evidence space.

For each image/view, the six-dimensional evidence vector contains the
teacher-forced complete-sentence log-likelihood contrast

    mean log p(positive sentence) - mean log p(negative sentence).

The analysis is read-only and label-free. It asks whether synthetic
acquisition-style views move this clinical evidence vector toward the
patient-LOO clean evidence centroid. It does not select an answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


VERSION = "clinical-evidence-attractor-v1"
STYLE_PREFIX = "style_"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_from_image(image_relative: str) -> str:
    parts = Path(image_relative).parts
    if len(parts) < 2 or not parts[1].startswith("p"):
        raise ValueError(f"cannot parse patient from {image_relative}")
    return parts[1]


def load_evidence(
    path: Path,
) -> tuple[np.ndarray, list[str], list[str], list[str], str, str]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    cases = sorted({str(row["case_id"]) for row in rows})
    diseases = sorted({str(row["disease"]) for row in rows})
    views = ["real", "null"] + sorted(
        {
            str(row["view"])
            for row in rows
            if str(row["view"]).startswith(STYLE_PREFIX)
        },
        key=lambda value: int(value.split("_")[-1]),
    )
    case_index = {value: index for index, value in enumerate(cases)}
    disease_index = {value: index for index, value in enumerate(diseases)}
    view_index = {value: index for index, value in enumerate(views)}
    nll: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    patients: dict[str, str] = {}
    models = set()
    pairing_rows = []
    for row in rows:
        case = str(row["case_id"])
        view = str(row["view"])
        disease = str(row["disease"])
        polarity = str(row["polarity"])
        if polarity in nll[(case, view, disease)]:
            raise ValueError(
                f"duplicate cell: {case}/{view}/{disease}/{polarity}"
            )
        nll[(case, view, disease)][polarity] = float(row["sequence_nll"])
        patients[case] = patient_from_image(str(row["image_relative"]))
        models.add(str(row["model"]))
        pairing_rows.append(
            {
                key: row.get(key)
                for key in (
                    "case_id",
                    "image_relative",
                    "source_question_id",
                    "view",
                    "disease",
                    "polarity",
                    "question",
                    "answer",
                    "answer_token_count",
                    "questions_sha256",
                    "image_manifest_sha256",
                    "style_manifest_sha256",
                    "radius",
                    "strength",
                )
            }
        )
    if len(models) != 1:
        raise ValueError(f"expected one model, found {sorted(models)}")
    evidence = np.full(
        (len(cases), len(views), len(diseases)), np.nan, dtype=np.float64
    )
    for (case, view, disease), values in nll.items():
        if set(values) != {"positive", "negative"}:
            raise ValueError(f"incomplete polarity pair: {case}/{view}/{disease}")
        # -NLL_positive - (-NLL_negative)
        evidence[
            case_index[case], view_index[view], disease_index[disease]
        ] = values["negative"] - values["positive"]
    if not np.isfinite(evidence).all():
        raise ValueError("evidence tensor is incomplete")
    pairing_payload = json.dumps(
        sorted(
            pairing_rows,
            key=lambda row: (
                str(row["case_id"]),
                str(row["view"]),
                str(row["disease"]),
                str(row["polarity"]),
            ),
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return (
        evidence,
        cases,
        [patients[case] for case in cases],
        diseases,
        next(iter(models)),
        hashlib.sha256(pairing_payload).hexdigest(),
    )


def cluster_bootstrap_mean(
    values: np.ndarray,
    patients: list[str],
    repeats: int,
    seed: int,
) -> tuple[float, list[float]]:
    """Bootstrap a case-by-style statistic at the patient level."""
    if values.shape[0] != len(patients):
        raise ValueError("patient count does not match values")
    groups = sorted(set(patients))
    group_rows = {
        group: np.flatnonzero(np.asarray(patients) == group) for group in groups
    }
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        chunks = [values[group_rows[group]].reshape(-1) for group in sampled]
        estimates.append(float(np.concatenate(chunks).mean()))
    return float(values.mean()), [
        float(value)
        for value in np.quantile(estimates, [0.025, 0.975])
    ]


def patient_balanced_bootstrap_mean(
    values: np.ndarray,
    patients: list[str],
    repeats: int,
    seed: int,
) -> tuple[float, list[float]]:
    """Bootstrap an estimand that gives each patient equal weight."""
    groups = sorted(set(patients))
    patient_means = np.asarray(
        [
            values[np.asarray(patients) == group].mean()
            for group in groups
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    estimates = [
        float(
            rng.choice(
                patient_means, size=len(patient_means), replace=True
            ).mean()
        )
        for _ in range(repeats)
    ]
    return float(patient_means.mean()), [
        float(value)
        for value in np.quantile(estimates, [0.025, 0.975])
    ]


def endpoint_projection_energy(
    delta: np.ndarray, endpoint: np.ndarray, eps: float = 1e-12
) -> tuple[float, float]:
    """Return normalized projection energy and positive-sign fraction."""
    coefficients = np.einsum("csd,cd->cs", delta, endpoint)
    endpoint_sq = np.square(endpoint).sum(axis=1)[:, None] + eps
    projected_sq = np.square(coefficients) / endpoint_sq
    total = float(np.square(delta).sum()) + eps
    return float(projected_sq.sum() / total), float((coefficients > 0).mean())


def analyze_evidence(
    evidence: np.ndarray,
    patients: list[str],
    repeats: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    real = evidence[:, 0]
    null = evidence[:, 1]
    styles = evidence[:, 2:]
    centroids = []
    for patient in patients:
        eligible = np.asarray(
            [other != patient for other in patients], dtype=bool
        )
        if not eligible.any():
            raise ValueError("patient-LOO centroid has no eligible cases")
        centroids.append(real[eligible].mean(axis=0))
    centroid = np.asarray(centroids)
    before_sq = np.square(real - centroid).sum(axis=1)[:, None]
    after_sq = np.square(styles - centroid[:, None, :]).sum(axis=2)
    log_ratio = np.log((after_sq + 1e-12) / (before_sq + 1e-12))
    estimate, interval = cluster_bootstrap_mean(
        log_ratio, patients, repeats, seed
    )
    balanced_estimate, balanced_interval = patient_balanced_bootstrap_mean(
        log_ratio, patients, repeats, seed + 17
    )
    per_style = {}
    for style_index in range(log_ratio.shape[1]):
        style_estimate, style_interval = cluster_bootstrap_mean(
            log_ratio[:, [style_index]],
            patients,
            repeats,
            seed + 101 + style_index,
        )
        per_style[f"style_{style_index}"] = {
            "mean_log_squared_centroid_distance_ratio": style_estimate,
            "patient_cluster_bootstrap_95pct": style_interval,
        }
    delta = styles - real[:, None, :]
    centroid_direction = centroid - real
    null_direction = null - real
    centroid_projection, centroid_positive = endpoint_projection_energy(
        delta, centroid_direction
    )
    null_projection, null_positive = endpoint_projection_energy(
        delta, null_direction
    )
    return (
        {
            "cases": int(evidence.shape[0]),
            "patients": len(set(patients)),
            "styles": int(styles.shape[1]),
            "mean_log_squared_centroid_distance_ratio": estimate,
            "patient_cluster_bootstrap_95pct": interval,
            "patient_balanced_sensitivity": {
                "mean_log_squared_centroid_distance_ratio": balanced_estimate,
                "patient_cluster_bootstrap_95pct": balanced_interval,
            },
            "per_style": per_style,
            "styles_ci_below_zero": sum(
                row["patient_cluster_bootstrap_95pct"][1] < 0
                for row in per_style.values()
            ),
            "median_log_squared_centroid_distance_ratio": float(
                np.median(log_ratio)
            ),
            "fraction_closer_to_centroid": float((log_ratio < 0).mean()),
            "clean_centroid_projection_energy": centroid_projection,
            "fraction_toward_clean_centroid": centroid_positive,
            "null_projection_energy": null_projection,
            "fraction_toward_null": null_positive,
        },
        log_ratio,
    )


def analyze_files(
    huatuo_path: Path,
    base_path: Path,
    repeats: int = 2000,
    seed: int = 2027,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    raw = {}
    common_cases = None
    common_patients = None
    common_diseases = None
    common_pairing = None
    for name, path in (("huatuo", huatuo_path), ("qwen_base", base_path)):
        (
            evidence,
            cases,
            patients,
            diseases,
            model,
            pairing_fingerprint,
        ) = load_evidence(path)
        if common_cases is None:
            common_cases = cases
            common_patients = patients
            common_diseases = diseases
            common_pairing = pairing_fingerprint
        elif (
            cases != common_cases
            or patients != common_patients
            or diseases != common_diseases
            or pairing_fingerprint != common_pairing
        ):
            raise ValueError("model probes are not exactly paired")
        metrics, log_ratio = analyze_evidence(
            evidence, patients, repeats, seed
        )
        models[name] = {
            "model": model,
            "raw": str(path),
            "raw_sha256": sha256(path),
            **metrics,
        }
        raw[name] = log_ratio
    assert common_patients is not None
    paired = raw["huatuo"] - raw["qwen_base"]
    paired_mean, paired_interval = cluster_bootstrap_mean(
        paired, common_patients, repeats, seed + 1
    )
    return {
        "version": VERSION,
        "diseases": common_diseases,
        "paired_probe_fingerprint": common_pairing,
        "models": models,
        "paired_huatuo_minus_base": {
            "mean_log_distance_ratio_difference": paired_mean,
            "patient_cluster_bootstrap_95pct": paired_interval,
        },
        "decision": {
            "huatuo_centroid_contraction_ci_below_zero": models["huatuo"][
                "patient_cluster_bootstrap_95pct"
            ][1]
            < 0,
            "base_centroid_contraction_ci_below_zero": models["qwen_base"][
                "patient_cluster_bootstrap_95pct"
            ][1]
            < 0,
            "medical_checkpoint_more_contracting_than_base": paired_interval[1]
            < 0,
            "centroid_projection_exceeds_null_both_models": all(
                models[name]["clean_centroid_projection_energy"]
                > models[name]["null_projection_energy"]
                for name in models
            ),
        },
        "interpretation": (
            "Complete-sentence six-disease evidence vectors contract toward a "
            "patient-LOO clean centroid for Huatuo on this synthetic probe. "
            "The exact Qwen base shows the same point-estimate direction but "
            "its interval includes zero. Their paired difference is uncertain, "
            "so no medical-tuning amplification claim is supported."
        ),
        "claim_ceiling": (
            "teacher-forced, unstandardized six-disease evidence geometry on "
            "64 exposed "
            "MIMIC development images and six fixed Fourier styles; not "
            "generated accuracy, factuality, natural-scanner robustness, or "
            "a causal effect of medical tuning"
        ),
    }


def plot_result(result: dict[str, Any], output: Path) -> None:
    labels = ["Qwen2.5-VL base", "Huatuo medical"]
    keys = ["qwen_base", "huatuo"]
    colors = ["#4393C3", "#D6604D"]
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    means = [
        result["models"][key][
            "mean_log_squared_centroid_distance_ratio"
        ]
        for key in keys
    ]
    intervals = [
        result["models"][key]["patient_cluster_bootstrap_95pct"] for key in keys
    ]
    errors = np.asarray(
        [
            [mean - interval[0] for mean, interval in zip(means, intervals)],
            [interval[1] - mean for mean, interval in zip(means, intervals)],
        ]
    )
    axes[0].bar(labels, means, color=colors, width=0.6)
    axes[0].errorbar(
        np.arange(2), means, yerr=errors, fmt="none", color="#333333", capsize=4
    )
    axes[0].axhline(0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Mean log squared-distance ratio")
    axes[0].set_title("Clinical evidence contracts under style")

    width = 0.34
    x = np.arange(2)
    centroid = [
        result["models"][key]["clean_centroid_projection_energy"] * 100
        for key in keys
    ]
    null = [
        result["models"][key]["null_projection_energy"] * 100 for key in keys
    ]
    axes[1].bar(
        x - width / 2, centroid, width, color="#B2182B", label="Clean centroid"
    )
    axes[1].bar(
        x + width / 2, null, width, color="#F4A582", label="Null endpoint"
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Style displacement projected (%)")
    axes[1].set_title("Centroid direction exceeds null direction")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
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
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    result = analyze_files(
        args.huatuo.expanduser().resolve(),
        args.base.expanduser().resolve(),
        args.bootstrap_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    plot_result(result, args.figure)
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
