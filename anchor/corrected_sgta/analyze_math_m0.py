"""Mathematics-first M0 audit for source geometry and decoder-visible risk.

This module is cache-only.  It deliberately separates:

1. source/target distribution geometry;
2. per-sample error ranking;
3. finite-difference decoder sensitivity along an aligned view.

No center distance is interpreted as evidence of improved correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np

from corrected_sgta.cache import decode_array


VERSION = "sgta-math-m0-identifiability-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxr-support", required=True, type=Path)
    parser.add_argument("--modality-support", required=True, type=Path)
    parser.add_argument("--manifold-ct", required=True, type=Path)
    parser.add_argument("--manifold-mri", required=True, type=Path)
    parser.add_argument(
        "--paired-control-final", action="append", default=[], type=Path
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--risk-auroc-gate", type=float, default=0.70)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def error_auroc(scores: np.ndarray, errors: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.int64)
    finite = np.isfinite(scores)
    scores = scores[finite]
    errors = errors[finite]
    positive = scores[errors == 1]
    negative = scores[errors == 0]
    if not len(positive) or not len(negative):
        return None
    greater = (positive[:, None] > negative[None, :]).sum()
    equal = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(positive) * len(negative)))


def bootstrap_auroc_ci(
    scores: np.ndarray,
    errors: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> list[float] | None:
    scores = np.asarray(scores, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.int64)
    if len(np.unique(errors)) < 2:
        return None
    rng = random.Random(seed)
    values = []
    n = len(errors)
    for _ in range(repeats):
        indices = np.asarray([rng.randrange(n) for _ in range(n)])
        value = error_auroc(scores[indices], errors[indices])
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    lo = values[max(0, int(0.025 * len(values)))]
    hi = values[min(len(values) - 1, int(0.975 * len(values)))]
    return [float(lo), float(hi)]


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max()
    exponent = np.exp(shifted)
    return exponent / np.clip(exponent.sum(), 1e-12, None)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    middle = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, middle) + 0.5 * kl_divergence(q, middle)


def psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def gaussian_bures_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bures-Wasserstein distance between empirical Gaussian measures."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return math.nan
    mean_term = float(np.sum((a.mean(axis=0) - b.mean(axis=0)) ** 2))
    cov_a = np.atleast_2d(np.cov(a, rowvar=False))
    cov_b = np.atleast_2d(np.cov(b, rowvar=False))
    root_a = psd_sqrt(cov_a)
    middle_root = psd_sqrt(root_a @ cov_b @ root_a)
    covariance_term = float(np.trace(cov_a + cov_b - 2.0 * middle_root))
    return float(math.sqrt(max(0.0, mean_term + covariance_term)))


def source_geometry(
    source: np.ndarray,
    target: np.ndarray,
    train_fraction: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Fit source-only geometry and return population and sample scores."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n_train = max(2, min(len(source) - 2, int(len(source) * train_fraction)))
    train = source[:n_train]
    heldout = source[n_train:]

    train_unit = normalize_rows(train)
    target_unit = normalize_rows(target)
    center = normalize_rows(train_unit.mean(axis=0, keepdims=True))[0]
    center_distance = 1.0 - target_unit @ center
    similarities = target_unit @ train_unit.T
    ordered = np.sort(similarities, axis=1)
    nearest = 1.0 - ordered[:, -1]
    five_nearest = 1.0 - ordered[:, -min(5, len(train)) :].mean(axis=1)

    mean = train.mean(axis=0)
    centered = train - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    energy = singular**2
    cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-12)
    rank90 = int(np.searchsorted(cumulative, 0.90) + 1)
    basis = vt[:rank90].T

    def components(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = values - mean
        tangent = delta @ basis
        residual = delta - tangent @ basis.T
        return tangent, np.linalg.norm(residual, axis=1)

    heldout_tangent, heldout_normal = components(heldout)
    target_tangent, target_normal = components(target)
    tangent_scale = np.var(centered @ basis, axis=0, ddof=1)
    tangent_scale = np.clip(tangent_scale, 1e-8, None)
    target_mahalanobis = np.sum((target_tangent**2) / tangent_scale, axis=1)
    heldout_mahalanobis = np.sum((heldout_tangent**2) / tangent_scale, axis=1)

    population = {
        "n_source_train": n_train,
        "n_source_heldout": len(heldout),
        "n_target": len(target),
        "source_rank90": rank90,
        "source_explained_variance": float(cumulative[rank90 - 1]),
        "source_pca_bures_target": gaussian_bures_distance(
            heldout_tangent, target_tangent
        ),
        "normal_residual_ratio_target_over_heldout": float(
            np.median(target_normal) / max(float(np.median(heldout_normal)), 1e-12)
        ),
        "mahalanobis_ratio_target_over_heldout": float(
            np.median(target_mahalanobis)
            / max(float(np.median(heldout_mahalanobis)), 1e-12)
        ),
    }
    scores = {
        "center_cosine": center_distance,
        "local_1nn_cosine": nearest,
        "local_5nn_cosine": five_nearest,
        "source_affine_normal": target_normal,
        "source_pca_mahalanobis": target_mahalanobis,
    }
    return population, scores


def risk_summary(
    scores: dict[str, np.ndarray],
    errors: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> dict:
    return {
        name: {
            "error_auroc": error_auroc(values, errors),
            "bootstrap_95ci": bootstrap_auroc_ci(
                values, errors, seed=seed + index, repeats=repeats
            ),
        }
        for index, (name, values) in enumerate(sorted(scores.items()))
    }


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def prediction(row: dict, interface: str, style: int) -> int | None:
    if interface == "decoded":
        value = row["style_decoded_prediction"][style]
        return None if value is None or value < 0 else int(value)
    if interface == "sequence_nll":
        return int(np.argmin(row["style_sequence_nll"][style]))
    raise ValueError(interface)


def manifold_path_scores(rows: list[dict]) -> tuple[dict[str, np.ndarray], dict]:
    scores: dict[str, list[float]] = {
        "path_js": [],
        "path_symmetric_kl": [],
        "decoder_sensitivity": [],
        "sequence_score_delta": [],
        "visual_delta_norm": [],
    }
    changed = {"decoded": 0, "sequence_nll": 0}
    oracle = {"decoded": 0, "sequence_nll": 0}
    baseline_correct = {"decoded": 0, "sequence_nll": 0}
    for row in rows:
        visual = decode_array(row["style_visual_features"]).astype(np.float64)
        visual_delta = float(np.linalg.norm(visual[1] - visual[0]))
        sequence_scores = -np.asarray(row["style_sequence_nll"], dtype=np.float64)
        p0 = softmax(sequence_scores[0])
        p1 = softmax(sequence_scores[1])
        score_delta = float(np.linalg.norm(sequence_scores[1] - sequence_scores[0]))
        scores["path_js"].append(js_divergence(p0, p1))
        scores["path_symmetric_kl"].append(
            0.5 * (kl_divergence(p0, p1) + kl_divergence(p1, p0))
        )
        scores["decoder_sensitivity"].append(
            score_delta / max(visual_delta, 1e-12)
        )
        scores["sequence_score_delta"].append(score_delta)
        scores["visual_delta_norm"].append(visual_delta)
        for interface in changed:
            original = prediction(row, interface, 0)
            aligned = prediction(row, interface, 1)
            gt = int(row["gt_index"])
            changed[interface] += int(original != aligned)
            baseline_correct[interface] += int(original == gt)
            oracle[interface] += int(original == gt or aligned == gt)
    n = len(rows)
    headroom = {
        interface: {
            "disagreements": changed[interface],
            "disagreement_rate": changed[interface] / n,
            "oracle_headroom_pp": 100.0
            * (oracle[interface] - baseline_correct[interface])
            / n,
            "mathematical_upper_bound_pp": 100.0 * changed[interface] / n,
        }
        for interface in changed
    }
    return {key: np.asarray(value) for key, value in scores.items()}, headroom


def errors_from_rows(rows: list[dict], interface: str) -> np.ndarray:
    return np.asarray(
        [
            prediction(row, interface, 0) != int(row["gt_index"])
            for row in rows
        ],
        dtype=np.int64,
    )


def best_metric(payload: dict) -> tuple[str | None, float | None]:
    candidates = []
    for family, metrics in payload.items():
        for name, value in metrics.items():
            auroc = value.get("error_auroc")
            if auroc is not None:
                candidates.append((f"{family}/{name}", float(auroc)))
    return max(candidates, key=lambda item: item[1]) if candidates else (None, None)


def paired_control_summary(final_path: Path) -> tuple[dict, list[Path]]:
    final = json.loads(final_path.read_text())
    analysis_path = Path(final["analysis"])
    audit_path = Path(final.get("audit") or final["structure"])
    analysis = json.loads(analysis_path.read_text())
    audit = json.loads(audit_path.read_text())
    fingerprints = {
        "final": final.get("fingerprint"),
        "analysis": analysis.get("fingerprint"),
        "audit": audit.get("fingerprint"),
    }
    fingerprint_match = len(set(fingerprints.values())) == 1
    complete = bool(
        audit.get("formal_matched_structure_pass")
        or final.get("checks", {}).get("strict_cache_and_pixel_identity_audit")
    )
    domain = analysis["domain_diagnostics"]
    return {
        "final": str(final_path.resolve()),
        "n": int(final["n"]),
        "fingerprints": fingerprints,
        "fingerprint_match": fingerprint_match,
        "complete_paired_audit": complete,
        "final_pass": bool(final["pass"]),
        "decision": final["decision"],
        "matched_minus_wrong_control_accuracy": float(
            analysis["matched_minus_wrong_control_uniform_accuracy"]
        ),
        "matched_oracle_headroom_pp": 100.0
        * float(analysis["matched_style_oracle_headroom_diagnostic_only"]),
        "matched_prediction_disagreement_rate": float(
            domain["matched_cross_view_prediction_disagreement_rate"]
        ),
        "matched_relative_closure_median": float(
            domain["matched_relative_closure_median"]
        ),
        "wrong_relative_closure_median": float(
            domain["wrong_control_relative_closure_median"]
        ),
    }, [analysis_path, audit_path]


def main() -> None:
    args = parse_args()
    paths = [
        args.cxr_support,
        args.modality_support,
        args.manifold_ct,
        args.manifold_mri,
    ] + args.paired_control_final
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    cxr_analysis = json.loads(args.cxr_support.read_text())
    cxr_features_path = Path(cxr_analysis["features"])
    cxr_arrays = np.load(cxr_features_path, allow_pickle=False)
    greedy = json.loads(Path(cxr_analysis["config"]["greedy_eval"]).read_text())
    correct = {
        str(item["question_id"]): bool(item["correct"])
        for item in greedy["details"]
    }
    cxr_ids = [str(value) for value in cxr_analysis["ids"]["target"]]
    cxr_errors = np.asarray([not correct[qid] for qid in cxr_ids], dtype=np.int64)
    cxr = {}
    for layer, source_key, target_key in (
        ("raw_clip", "exact_raw", "target_raw"),
        ("projected", "exact_projected", "target_projected"),
    ):
        population, scores = source_geometry(
            cxr_arrays[source_key], cxr_arrays[target_key], 0.75
        )
        cxr[layer] = {
            "population_geometry": population,
            "error_risk": risk_summary(
                scores,
                cxr_errors,
                seed=args.seed,
                repeats=args.bootstrap_repeats,
            ),
        }

    modality_analysis = json.loads(args.modality_support.read_text())
    modality_arrays = np.load(
        Path(modality_analysis["features"]), allow_pickle=False
    )
    modalities = {}
    all_risk_families = {"cxr_raw": cxr["raw_clip"]["error_risk"], "cxr_projected": cxr["projected"]["error_risk"]}
    for offset, (modality, rows_path) in enumerate(
        (("ct", args.manifold_ct), ("mri", args.manifold_mri))
    ):
        rows = [row for row in load_jsonl(rows_path) if row.get("status") == "ok"]
        target_ids = [
            str(value) for value in modality_analysis["ids"][modality]["target"]
        ]
        row_ids = [str(row["qid"]) for row in rows]
        overlap = sorted(set(row_ids) & set(target_ids))
        row_projected = np.stack(
            [
                decode_array(row["style_visual_features"]).astype(np.float64)[0]
                for row in rows
            ]
        )
        layer_payload = {}
        raw_population, _ = source_geometry(
            modality_arrays[f"{modality}_source_raw"],
            modality_arrays[f"{modality}_target_raw"],
            modality_analysis["config"]["source_train_fraction"],
        )
        layer_payload["raw"] = {
            "population_geometry": raw_population,
            "error_risk_sequence_nll": None,
            "unavailable_reason": (
                "manifold rows and raw target cache use different deterministic "
                "subsets; row-level joining is forbidden"
            ),
        }
        projected_population, projected_scores = source_geometry(
            modality_arrays[f"{modality}_source_projected"],
            row_projected,
            modality_analysis["config"]["source_train_fraction"],
        )
        layer_payload["projected"] = {
            "population_geometry": projected_population,
            "error_risk_sequence_nll": risk_summary(
                projected_scores,
                errors_from_rows(rows, "sequence_nll"),
                seed=args.seed + 100 * (offset + 1),
                repeats=args.bootstrap_repeats,
            ),
        }
        all_risk_families[f"{modality}_projected_geometry"] = layer_payload[
            "projected"
        ]["error_risk_sequence_nll"]

        path_scores, headroom = manifold_path_scores(rows)
        path_descriptives = {
            name: {
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": float(np.max(values)),
            }
            for name, values in path_scores.items()
        }
        path_risk = {}
        for interface in ("decoded", "sequence_nll"):
            path_risk[interface] = risk_summary(
                path_scores,
                errors_from_rows(rows, interface),
                seed=args.seed + 1000 * (offset + 1),
                repeats=args.bootstrap_repeats,
            )
            all_risk_families[f"{modality}_path_{interface}"] = path_risk[
                interface
            ]
        modalities[modality] = {
            "n_rows": len(rows),
            "cache_identity": {
                "manifold_rows": len(row_ids),
                "support_target_rows": len(target_ids),
                "qid_overlap": len(overlap),
                "projected_row_source": "manifold.style_visual_features[original]",
            },
            "layers": layer_payload,
            "path_descriptives": path_descriptives,
            "path_risk": path_risk,
            "candidate_headroom": headroom,
        }

    paired_controls = []
    referenced_control_paths = []
    for final_path in args.paired_control_final:
        summary, referenced = paired_control_summary(final_path)
        paired_controls.append(summary)
        referenced_control_paths.extend(referenced)
    paths.extend(referenced_control_paths)

    metric_name, metric_auroc = best_metric(all_risk_families)
    control_available = bool(paired_controls) and all(
        item["fingerprint_match"] and item["complete_paired_audit"]
        for item in paired_controls
    )
    matched_beats_control = control_available and any(
        item["matched_minus_wrong_control_accuracy"] > 0.0
        and item["matched_oracle_headroom_pp"] >= 5.0
        for item in paired_controls
    )
    checks = {
        "risk_metric_auroc_at_least_gate": (
            metric_auroc is not None and metric_auroc >= args.risk_auroc_gate
        ),
        "matched_path_beats_shuffled_control": matched_beats_control,
        "candidate_headroom_at_least_5pp_for_any_modality": any(
            value["candidate_headroom"]["sequence_nll"]["oracle_headroom_pp"]
            >= 5.0
            for value in modalities.values()
        ),
    }
    config = {
        "version": VERSION,
        "seed": args.seed,
        "bootstrap_repeats": args.bootstrap_repeats,
        "risk_auroc_gate": args.risk_auroc_gate,
        "inputs": {
            str(path): sha256_file(path)
            for path in paths
        },
        "labels_used_for": "post-hoc error-risk audit only",
        "source_geometry_role": "diagnostic proxy, never a correctness guarantee",
        "decoder_metric": "finite-difference full-label-sequence score sensitivity and JS/Fisher local proxy",
    }
    payload = {
        "analysis_version": VERSION,
        "fingerprint": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "config": config,
        "cxr": cxr,
        "modalities": modalities,
        "paired_controls": paired_controls,
        "gate": {
            "best_error_risk_metric": metric_name,
            "best_error_risk_auroc": metric_auroc,
            "checks": checks,
            "pass": all(checks.values()),
            "decision": (
                "proceed_to_matched_vs_shuffled_n64"
                if all(checks.values())
                else "do_not_tune_or_expand"
            ),
            "missing_evidence": [
                "audited paired matched-versus-shuffled path cache"
            ]
            if not control_available
            else [],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "analysis.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "fingerprint": payload["fingerprint"],
                "gate": payload["gate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
