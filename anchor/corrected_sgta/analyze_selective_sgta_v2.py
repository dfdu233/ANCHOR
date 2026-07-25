"""Image-disjoint evaluation of single-view Selective SGTA.

The method keeps the original point prediction. A single modality-matched,
structure-safe source view is used only to estimate risk. Development images
select one mixing weight, calibration images set operating thresholds, and
test images are used once for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


METHOD_VERSION = "selective-sgta-v2"
MIXING_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGET_COVERAGES = (0.9, 0.8)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def _entropy(probabilities: np.ndarray) -> float:
    values = np.clip(probabilities, 1e-12, 1.0)
    return float(-(values * np.log(values)).sum())


def _js_pair(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)
    return _entropy(midpoint) - 0.5 * (_entropy(left) + _entropy(right))


def _hash_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _group_split(
    records: list[dict], seed: int, development_fraction: float, calibration_fraction: float
) -> tuple[list[dict], list[dict], list[dict]]:
    groups = sorted({row["group"] for row in records}, key=lambda value: _hash_key(seed, value))
    if len(groups) < 3:
        raise ValueError("at least three image groups are required")
    development_end = max(1, int(math.floor(development_fraction * len(groups))))
    calibration_end = max(
        development_end + 1,
        int(math.floor((development_fraction + calibration_fraction) * len(groups))),
    )
    calibration_end = min(calibration_end, len(groups) - 1)
    assignments = {
        group: "development"
        if index < development_end
        else "calibration"
        if index < calibration_end
        else "test"
        for index, group in enumerate(groups)
    }
    splits = tuple(
        [row for row in records if assignments[row["group"]] == split]
        for split in ("development", "calibration", "test")
    )
    group_sets = [{row["group"] for row in split} for split in splits]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("image groups overlap across splits")
    return splits


def _is_structure_safe(metadata: dict, min_psnr: float, min_edge: float) -> bool:
    structure = metadata.get("structure") or {}
    psnr = structure.get("psnr")
    edge = structure.get("edge_correlation")
    return (
        psnr is not None
        and edge is not None
        and float(psnr) >= min_psnr
        and float(edge) >= min_edge
    )


def _distortion(metadata: dict) -> float:
    return float((metadata.get("structure") or {}).get("pixel_mse") or 0.0)


def _select_views(
    row: dict, min_psnr: float, min_edge: float
) -> tuple[int, int | None] | None:
    names = list(row["style_names"])
    metadata = list(row["style_metadata"])
    source = [
        index
        for index, (name, item) in enumerate(zip(names, metadata))
        if index > 0
        and name.startswith("feddg_")
        and _is_structure_safe(item, min_psnr, min_edge)
    ]
    if not source:
        return None
    # The strongest intervention that still passes the frozen structure gate
    # maximizes the chance of exposing decoder-visible domain sensitivity.
    source_index = max(source, key=lambda index: (_distortion(metadata[index]), -index))
    generic = [
        index
        for index, (name, item) in enumerate(zip(names, metadata))
        if index > 0
        and name.startswith("gamma_")
        and _is_structure_safe(item, min_psnr, min_edge)
    ]
    generic_index = (
        min(
            generic,
            key=lambda index: (
                abs(_distortion(metadata[index]) - _distortion(metadata[source_index])),
                index,
            ),
        )
        if generic
        else None
    )
    return source_index, generic_index


def _record(row: dict, min_psnr: float, min_edge: float) -> dict | None:
    selected = _select_views(row, min_psnr, min_edge)
    if selected is None:
        return None
    source_index, generic_index = selected
    if row.get("style_sequence_nll") is not None:
        probabilities = [
            _softmax(-np.asarray(values, dtype=np.float64))
            for values in row["style_sequence_nll"]
        ]
        score_channel = "complete_label_sequence_nll"
    else:
        probabilities = [
            _softmax(np.asarray(values, dtype=np.float64)) for values in row["style_logits"]
        ]
        score_channel = "surface_logits"
    original = probabilities[0]
    metadata = row["style_metadata"]
    result = {
        "qid": str(row["qid"]),
        "group": str(row.get("img_name") or row["qid"]),
        "fingerprint": str(row.get("fingerprint")),
        "score_channel": score_channel,
        "correct": int(np.argmax(original)) == int(row["gt_index"]),
        "entropy": _entropy(original),
        "source_js": _js_pair(original, probabilities[source_index]),
        "source_style": row["style_names"][source_index],
        "source_distortion": _distortion(metadata[source_index]),
        "generic_js": None,
        "generic_style": None,
    }
    if generic_index is not None:
        result["generic_js"] = _js_pair(original, probabilities[generic_index])
        result["generic_style"] = row["style_names"][generic_index]
    return result


def _ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="right") / max(1, len(ordered))


def _aurc(correct: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(risk, kind="stable")
    cumulative_errors = np.cumsum(~correct[order])
    return float(np.mean(cumulative_errors / np.arange(1, len(order) + 1)))


def _fixed_coverage(correct: np.ndarray, risk: np.ndarray, coverage: float) -> float:
    count = max(1, min(len(correct), int(math.floor(coverage * len(correct)))))
    accepted = np.argsort(risk, kind="stable")[:count]
    return float(correct[accepted].mean())


def _calibration_threshold(risk: np.ndarray, coverage: float) -> float:
    rank = min(len(risk) - 1, max(0, math.ceil((len(risk) + 1) * coverage) - 1))
    return float(np.sort(risk)[rank])


def _fit_weight(
    correct: np.ndarray, entropy_rank: np.ndarray, view_rank: np.ndarray
) -> tuple[float, list[dict]]:
    candidates = []
    for weight in MIXING_WEIGHTS:
        risk = (1.0 - weight) * entropy_rank + weight * view_rank
        candidates.append({"weight": weight, "development_aurc": _aurc(correct, risk)})
    selected = min(candidates, key=lambda row: (row["development_aurc"], row["weight"]))
    return float(selected["weight"]), candidates


def _cluster_bootstrap(
    rows: list[dict],
    left_risk: np.ndarray,
    right_risk: np.ndarray,
    coverage: float,
    samples: int,
    seed: int,
) -> dict:
    groups = sorted({row["group"] for row in rows})
    indices = {
        group: np.asarray([index for index, row in enumerate(rows) if row["group"] == group])
        for group in groups
    }
    correct = np.asarray([row["correct"] for row in rows], dtype=bool)
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        drawn_groups = rng.choice(groups, size=len(groups), replace=True)
        draw = np.concatenate([indices[group] for group in drawn_groups])
        deltas[sample_index] = _fixed_coverage(
            correct[draw], left_risk[draw], coverage
        ) - _fixed_coverage(correct[draw], right_risk[draw], coverage)
    return {
        "mean": float(deltas.mean()),
        "ci95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
        "probability_gt_zero": float(np.mean(deltas > 0)),
        "samples": samples,
        "cluster": "img_name",
        "n_clusters": len(groups),
    }


def analyze(
    input_path: Path,
    question_type: str,
    seed: int,
    development_fraction: float,
    calibration_fraction: float,
    min_psnr: float,
    min_edge: float,
    bootstrap_samples: int,
) -> dict:
    parsed = []
    total = 0
    with input_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            if question_type != "all" and row.get("question_type") != question_type:
                continue
            total += 1
            value = _record(row, min_psnr, min_edge)
            if value is not None:
                parsed.append(value)
    if len(parsed) < 30:
        raise ValueError(f"need at least 30 usable rows, found {len(parsed)}")
    fingerprints = sorted({row["fingerprint"] for row in parsed})
    channels = sorted({row["score_channel"] for row in parsed})
    if len(fingerprints) != 1 or len(channels) != 1:
        raise RuntimeError("input must contain one fingerprint and one score channel")

    development, calibration, test = _group_split(
        parsed, seed, development_fraction, calibration_fraction
    )
    development_entropy = np.asarray([row["entropy"] for row in development])
    development_source = np.asarray([row["source_js"] for row in development])
    development_correct = np.asarray([row["correct"] for row in development], dtype=bool)
    entropy_development_rank = _ecdf(development_entropy, development_entropy)
    source_development_rank = _ecdf(development_source, development_source)
    source_weight, source_search = _fit_weight(
        development_correct, entropy_development_rank, source_development_rank
    )

    has_generic_control = all(row["generic_js"] is not None for row in parsed)
    generic_weight = None
    generic_search = None
    development_generic = None
    if has_generic_control:
        development_generic = np.asarray([row["generic_js"] for row in development])
        generic_weight, generic_search = _fit_weight(
            development_correct,
            entropy_development_rank,
            _ecdf(development_generic, development_generic),
        )

    def risks(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        entropy_rank = _ecdf(
            development_entropy, np.asarray([row["entropy"] for row in rows])
        )
        source_rank = _ecdf(
            development_source, np.asarray([row["source_js"] for row in rows])
        )
        source_risk = (1.0 - source_weight) * entropy_rank + source_weight * source_rank
        generic_risk = None
        if has_generic_control and development_generic is not None and generic_weight is not None:
            generic_rank = _ecdf(
                development_generic, np.asarray([row["generic_js"] for row in rows])
            )
            generic_risk = (
                (1.0 - generic_weight) * entropy_rank + generic_weight * generic_rank
            )
        return entropy_rank, source_risk, generic_risk

    calibration_risks = risks(calibration)
    test_risks = risks(test)
    test_correct = np.asarray([row["correct"] for row in test], dtype=bool)
    results = {}
    for coverage in TARGET_COVERAGES:
        entropy_accuracy = _fixed_coverage(test_correct, test_risks[0], coverage)
        source_accuracy = _fixed_coverage(test_correct, test_risks[1], coverage)
        source_vs_entropy = _cluster_bootstrap(
            test,
            test_risks[1],
            test_risks[0],
            coverage,
            bootstrap_samples,
            seed + int(coverage * 100),
        )
        row = {
            "evaluation_only_fixed_coverage": {
                "coverage": coverage,
                "entropy_accuracy": entropy_accuracy,
                "source_accuracy": source_accuracy,
                "source_increment_pp": 100.0 * (source_accuracy - entropy_accuracy),
                "source_vs_entropy_cluster_bootstrap": source_vs_entropy,
            },
            "calibration_threshold": {},
        }
        for name, calibration_risk, test_risk in (
            ("entropy", calibration_risks[0], test_risks[0]),
            ("source", calibration_risks[1], test_risks[1]),
        ):
            threshold = _calibration_threshold(calibration_risk, coverage)
            accepted = test_risk <= threshold
            row["calibration_threshold"][name] = {
                "threshold": threshold,
                "coverage": float(accepted.mean()),
                "accuracy": float(test_correct[accepted].mean()) if accepted.any() else None,
                "n_accepted": int(accepted.sum()),
            }
        if test_risks[2] is not None:
            generic_accuracy = _fixed_coverage(test_correct, test_risks[2], coverage)
            row["evaluation_only_fixed_coverage"]["generic_accuracy"] = generic_accuracy
            row["evaluation_only_fixed_coverage"]["source_vs_generic_cluster_bootstrap"] = (
                _cluster_bootstrap(
                    test,
                    test_risks[1],
                    test_risks[2],
                    coverage,
                    bootstrap_samples,
                    seed + 1000 + int(coverage * 100),
                )
            )
        results[str(coverage)] = row

    entropy_aurc = _aurc(test_correct, test_risks[0])
    source_aurc = _aurc(test_correct, test_risks[1])
    generic_aurc = _aurc(test_correct, test_risks[2]) if test_risks[2] is not None else None
    primary = results["0.8"]["evaluation_only_fixed_coverage"]
    gate = {
        "development_selected_nonzero_source_weight": source_weight > 0,
        "test_source_aurc_better_than_entropy": source_aurc < entropy_aurc,
        "source_increment_cluster_ci_lower_above_zero": (
            primary["source_vs_entropy_cluster_bootstrap"]["ci95"][0] > 0
        ),
        "source_aurc_better_than_generic_control": (
            generic_aurc is not None and source_aurc < generic_aurc
        ),
        "source_vs_generic_cluster_ci_lower_above_zero": (
            primary.get("source_vs_generic_cluster_bootstrap", {}).get("ci95", [-math.inf])[0]
            > 0
        ),
    }
    gate["pass"] = all(gate.values())
    return {
        "method_version": METHOD_VERSION,
        "input": str(input_path),
        "question_type": question_type,
        "fingerprint": fingerprints[0],
        "score_channel": channels[0],
        "structure_gate": {"min_psnr": min_psnr, "min_edge_correlation": min_edge},
        "single_source_policy": "maximum pixel MSE among matched FedDG views passing structure gate",
        "generic_control_policy": "safe gamma view with pixel MSE nearest to selected source view",
        "n_total_scope": total,
        "n_usable": len(parsed),
        "n_excluded_without_safe_source": total - len(parsed),
        "split": {
            "unit": "img_name",
            "seed": seed,
            "development_fraction_of_groups": development_fraction,
            "calibration_fraction_of_groups": calibration_fraction,
            "development": {"rows": len(development), "groups": len({r["group"] for r in development})},
            "calibration": {"rows": len(calibration), "groups": len({r["group"] for r in calibration})},
            "test": {"rows": len(test), "groups": len({r["group"] for r in test})},
        },
        "selected_source_weight": source_weight,
        "source_weight_search": source_search,
        "selected_generic_weight": generic_weight,
        "generic_weight_search": generic_search,
        "selected_source_styles": Counter(row["source_style"] for row in parsed),
        "selected_generic_styles": Counter(row["generic_style"] for row in parsed),
        "test": {
            "raw_accuracy": float(test_correct.mean()),
            "entropy_aurc": entropy_aurc,
            "source_aurc": source_aurc,
            "generic_aurc": generic_aurc,
            "coverage_results": results,
        },
        "predeclared_gate": gate,
        "claim_scope": (
            "A passing result supports an incremental, matched-source-specific selective-risk "
            "signal only. Point prediction and conformal coverage are not changed or claimed."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--question-type", choices=("all", "binary", "multichoice"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development-fraction", type=float, default=0.25)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--min-psnr", type=float, default=20.0)
    parser.add_argument("--min-edge-correlation", type=float, default=0.90)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        args.input,
        args.question_type,
        args.seed,
        args.development_fraction,
        args.calibration_fraction,
        args.min_psnr,
        args.min_edge_correlation,
        args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["test"], indent=2))
    print(json.dumps(result["predeclared_gate"], indent=2))


if __name__ == "__main__":
    main()
