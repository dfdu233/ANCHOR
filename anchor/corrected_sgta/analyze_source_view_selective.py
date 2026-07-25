"""Locked evaluation of source-view inconsistency for selective prediction.

The analyzer combines original prediction entropy with disagreement among the
original and matched FedDG views. A single mixing weight is selected on a
deterministic calibration split and then frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


METHOD_VERSION = "source-view-selective-v1"
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def _entropy(probabilities: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _generalized_js(probabilities: np.ndarray) -> float:
    """Equal-weight generalized Jensen--Shannon divergence."""
    mean_p = probabilities.mean(axis=0)
    return _entropy(mean_p) - float(np.mean([_entropy(p) for p in probabilities]))


def _split_key(qid: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="right") / max(1, len(ordered))


def _aurc(correct: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(risk, kind="stable")
    cumulative_errors = np.cumsum(~correct[order])
    risks = cumulative_errors / np.arange(1, len(order) + 1)
    return float(risks.mean())


def _selective_accuracy(correct: np.ndarray, risk: np.ndarray, coverage: float) -> dict:
    k = max(1, min(len(correct), int(math.floor(coverage * len(correct)))))
    accepted = np.argsort(risk, kind="stable")[:k]
    accuracy = float(correct[accepted].mean())
    return {
        "target_coverage": coverage,
        "n_accepted": int(k),
        "empirical_coverage": float(k / len(correct)),
        "accuracy": accuracy,
        "risk": float(1.0 - accuracy),
    }


def _bootstrap_increment(
    correct: np.ndarray,
    baseline_risk: np.ndarray,
    method_risk: np.ndarray,
    coverage: float,
    *,
    samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(correct)
    deltas = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, n, size=n)
        drawn_correct = correct[draw]
        base = _selective_accuracy(drawn_correct, baseline_risk[draw], coverage)["accuracy"]
        method = _selective_accuracy(drawn_correct, method_risk[draw], coverage)["accuracy"]
        deltas[index] = method - base
    return {
        "mean": float(deltas.mean()),
        "ci95": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
        "probability_gt_zero": float(np.mean(deltas > 0)),
        "samples": samples,
        "seed": seed,
    }


def _iter_rows(path: Path, question_type: str) -> Iterable[dict]:
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            if question_type != "all" and row.get("question_type") != question_type:
                continue
            yield row


def _record_from_row(row: dict) -> dict | None:
    names = list(row["style_names"])
    source_indices = [0] + [i for i, name in enumerate(names) if name.startswith("feddg_")]
    if len(source_indices) < 2:
        return None
    if row.get("style_sequence_nll") is not None:
        probabilities = _softmax(-np.asarray(row["style_sequence_nll"], dtype=np.float64))
        score_channel = "complete_label_sequence_nll"
    else:
        probabilities = _softmax(np.asarray(row["style_logits"], dtype=np.float64))
        score_channel = "surface_logits"
    original = probabilities[0]
    prediction = int(np.argmax(original))
    return {
        "qid": str(row["qid"]),
        "fingerprint": row.get("fingerprint"),
        "score_channel": score_channel,
        "correct": prediction == int(row["gt_index"]),
        "original_entropy": _entropy(original),
        "source_view_js": _generalized_js(probabilities[source_indices]),
    }


def analyze(
    input_path: Path,
    *,
    question_type: str,
    seed: int,
    calibration_fraction: float,
    bootstrap_samples: int,
) -> dict:
    parsed = [_record_from_row(row) for row in _iter_rows(input_path, question_type)]
    records = [record for record in parsed if record is not None]
    excluded_without_source_views = len(parsed) - len(records)
    if len(records) < 20:
        raise ValueError(f"Need at least 20 usable records, found {len(records)}")
    records.sort(key=lambda row: _split_key(row["qid"], seed))
    cut = int(math.floor(calibration_fraction * len(records)))
    calibration, test = records[:cut], records[cut:]

    cal_entropy = np.asarray([r["original_entropy"] for r in calibration])
    cal_source = np.asarray([r["source_view_js"] for r in calibration])
    test_entropy = np.asarray([r["original_entropy"] for r in test])
    test_source = np.asarray([r["source_view_js"] for r in test])
    cal_u0 = _ecdf(cal_entropy, cal_entropy)
    cal_udg = _ecdf(cal_source, cal_source)
    test_u0 = _ecdf(cal_entropy, test_entropy)
    test_udg = _ecdf(cal_source, test_source)
    cal_correct = np.asarray([r["correct"] for r in calibration], dtype=bool)
    test_correct = np.asarray([r["correct"] for r in test], dtype=bool)

    candidates = []
    for mixing_weight in LAMBDA_GRID:
        risk = (1.0 - mixing_weight) * cal_u0 + mixing_weight * cal_udg
        candidates.append({
            "mixing_weight": mixing_weight,
            "calibration_aurc": _aurc(cal_correct, risk),
        })
    selected = min(candidates, key=lambda item: (item["calibration_aurc"], item["mixing_weight"]))
    mixing_weight = float(selected["mixing_weight"])
    combined_risk = (1.0 - mixing_weight) * test_u0 + mixing_weight * test_udg

    calibrated_coverages = {}
    for coverage in (0.9, 0.8):
        rank = min(len(calibration) - 1, max(0, math.ceil((len(calibration) + 1) * coverage) - 1))
        entropy_threshold = float(np.sort(cal_u0)[rank])
        combined_cal_risk = (1.0 - mixing_weight) * cal_u0 + mixing_weight * cal_udg
        combined_threshold = float(np.sort(combined_cal_risk)[rank])
        calibrated_coverages[str(coverage)] = {}
        for name, risk, threshold in (
            ("entropy_only", test_u0, entropy_threshold),
            ("source_view_combined", combined_risk, combined_threshold),
        ):
            accepted = risk <= threshold
            calibrated_coverages[str(coverage)][name] = {
                "calibration_threshold": threshold,
                "n_accepted": int(accepted.sum()),
                "empirical_coverage": float(accepted.mean()),
                "accuracy": float(test_correct[accepted].mean()) if accepted.any() else None,
            }

    baseline_aurc = _aurc(test_correct, test_u0)
    combined_aurc = _aurc(test_correct, combined_risk)
    coverages = {}
    for coverage in (0.9, 0.8):
        baseline = _selective_accuracy(test_correct, test_u0, coverage)
        combined = _selective_accuracy(test_correct, combined_risk, coverage)
        coverages[str(coverage)] = {
            "entropy_only": baseline,
            "source_view_combined": combined,
            "gain_over_raw_pp": 100.0 * (combined["accuracy"] - float(test_correct.mean())),
            "increment_over_entropy_pp": 100.0 * (combined["accuracy"] - baseline["accuracy"]),
            "paired_bootstrap_increment": _bootstrap_increment(
                test_correct,
                test_u0,
                combined_risk,
                coverage,
                samples=bootstrap_samples,
                seed=seed + int(100 * coverage),
            ),
        }

    fingerprints = sorted({str(r["fingerprint"]) for r in records})
    channels = sorted({r["score_channel"] for r in records})
    relative_aurc_reduction = (
        (baseline_aurc - combined_aurc) / baseline_aurc if baseline_aurc > 0 else None
    )
    gate = {
        "selected_nonzero_source_weight": mixing_weight > 0,
        "locked_aurc_better_than_entropy": combined_aurc < baseline_aurc,
        "raw_gain_at_80pct_exceeds_3pp": coverages["0.8"]["gain_over_raw_pp"] > 3.0,
        "source_increment_ci95_lower_above_zero_at_80pct": (
            coverages["0.8"]["paired_bootstrap_increment"]["ci95"][0] > 0
        ),
    }
    gate["pass"] = all(gate.values())
    return {
        "method_version": METHOD_VERSION,
        "input": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "fingerprints": fingerprints,
        "score_channels": channels,
        "question_type": question_type,
        "excluded_without_source_views": excluded_without_source_views,
        "split": {
            "method": "sha256(seed:qid), sorted; first fraction is calibration",
            "seed": seed,
            "calibration_fraction": calibration_fraction,
            "calibration_n": len(calibration),
            "locked_test_n": len(test),
            "calibration_qids": [r["qid"] for r in calibration],
            "locked_test_qids": [r["qid"] for r in test],
        },
        "calibration_candidates": candidates,
        "selected_mixing_weight": mixing_weight,
        "locked_test": {
            "raw_accuracy": float(test_correct.mean()),
            "entropy_only_aurc": baseline_aurc,
            "source_view_combined_aurc": combined_aurc,
            "relative_aurc_reduction": relative_aurc_reduction,
            "coverage_results": coverages,
            "calibration_threshold_results": calibrated_coverages,
        },
        "predeclared_gate": gate,
        "interpretation": (
            "Passing supports source-view inconsistency as an incremental selective-risk "
            "signal; it does not support improved full-coverage point accuracy."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--question-type", choices=("all", "binary", "multichoice"), default="all"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze(
        args.input,
        question_type=args.question_type,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["locked_test"], indent=2))
    print(json.dumps(payload["predeclared_gate"], indent=2))


if __name__ == "__main__":
    main()
