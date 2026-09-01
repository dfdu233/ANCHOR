#!/usr/bin/env python3
"""Reader-grounded sampled-layer recoverability audit for VinDr claims.

The native oracle uses the unmodified support-minus-refute logit-lens margin.
The calibrated oracle subtracts a finding/layer threshold fitted only on a
separate development split.  Both are necessary: a raw early-layer class bias
can otherwise make every positive error look "recoverable" without containing
case-specific clinical evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


VERSION = "evidence-recoverability-audit-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty metadata: {path}")
    return rows


def layers_of(rows: list[dict[str, Any]]) -> tuple[int, ...]:
    layers = tuple(sorted(int(value) for value in rows[0]["diagnostic_plain_logit_lens"]))
    for row in rows:
        if tuple(sorted(int(value) for value in row["diagnostic_plain_logit_lens"])) != layers:
            raise ValueError("inconsistent layer inventory")
    return layers


def margin(row: dict[str, Any], layer: int) -> float:
    logits = row["diagnostic_plain_logit_lens"][str(layer)]
    return float(logits["supported"]) - float(logits["refuted"])


def truth(row: dict[str, Any]) -> int:
    votes = int(row["positive_votes"])
    if votes == 3:
        return 1
    if votes == 0:
        return -1
    return 0


def balanced_accuracy(values: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    positive = values[labels == 1] > threshold
    negative = values[labels == -1] <= threshold
    return 0.5 * (float(positive.mean()) + float(negative.mean()))


def fit_threshold(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if set(labels.tolist()) != {-1, 1}:
        raise ValueError("threshold fitting requires both clear polarities")
    unique = np.unique(values)
    candidates = np.concatenate(
        ([np.nextafter(unique[0], -np.inf)], (unique[:-1] + unique[1:]) / 2.0,
         [np.nextafter(unique[-1], np.inf)])
    )
    scores = np.asarray([balanced_accuracy(values, labels, value) for value in candidates])
    best = np.flatnonzero(scores == scores.max())
    # Deterministic tie break: closest to the median margin, then smallest value.
    center = float(np.median(values))
    index = min(best.tolist(), key=lambda i: (abs(float(candidates[i]) - center), float(candidates[i])))
    return {"threshold": float(candidates[index]), "dev_balanced_accuracy": float(scores[index])}


def fit_thresholds(
    rows: list[dict[str, Any]], layers: tuple[int, ...], *, per_finding: bool
) -> dict[str, dict[str, float]]:
    clear = [row for row in rows if truth(row) != 0]
    findings = sorted({str(row["finding"]) for row in clear}) if per_finding else ["__global__"]
    output: dict[str, dict[str, float]] = {}
    for finding in findings:
        selected = clear if finding == "__global__" else [row for row in clear if row["finding"] == finding]
        labels = np.asarray([truth(row) for row in selected], dtype=np.int8)
        for layer in layers:
            values = np.asarray([margin(row, layer) for row in selected], dtype=np.float64)
            output[f"{finding}:{layer}"] = fit_threshold(values, labels)
    return output


def threshold_for(
    thresholds: dict[str, dict[str, float]], row: dict[str, Any], layer: int, per_finding: bool
) -> float:
    key = f"{row['finding'] if per_finding else '__global__'}:{layer}"
    return float(thresholds[key]["threshold"])


def classify_error(row: dict[str, Any], final_layer: int) -> str:
    label = truth(row)
    if label == 0:
        return "ambiguous"
    prediction = 1 if margin(row, final_layer) > 0.0 else -1
    if label == 1:
        return "tp" if prediction == 1 else "fn"
    return "tn" if prediction == -1 else "fp"


def recovery_flags(
    row: dict[str, Any], candidate_layers: tuple[int, ...], thresholds: dict[str, dict[str, float]],
    *, per_finding: bool,
) -> dict[str, Any]:
    label = truth(row)
    native = [label * margin(row, layer) > 0.0 for layer in candidate_layers]
    calibrated = [
        label * (margin(row, layer) - threshold_for(thresholds, row, layer, per_finding)) > 0.0
        for layer in candidate_layers
    ]
    if not any(calibrated):
        trajectory = "absent"
    elif calibrated[-1]:
        trajectory = "prefinal_correct_final_reversal"
    else:
        trajectory = "transient_early_correct"
    return {
        "native_recoverable": bool(any(native)),
        "calibrated_recoverable": bool(any(calibrated)),
        "native_correct_layers": [layer for layer, value in zip(candidate_layers, native) if value],
        "calibrated_correct_layers": [layer for layer, value in zip(candidate_layers, calibrated) if value],
        "sampled_trajectory": trajectory,
    }


def cluster_bootstrap(
    records: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool], key: str,
    *, draws: int, seed: int,
) -> dict[str, float | int | None]:
    selected = [row for row in records if selector(row)]
    if not selected:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n": 0, "images": 0}
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        clusters[str(row["image_id"])].append(row)
    ids = sorted(clusters)
    estimate = float(np.mean([float(row[key]) for row in selected]))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        values = [float(row[key]) for cluster in chosen for row in clusters[str(cluster)]]
        samples.append(float(np.mean(values)))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": estimate, "ci_low": float(low), "ci_high": float(high),
        "n": len(selected), "images": len(ids), "bootstrap_draws": draws,
    }


def shuffled_null(
    rows: list[dict[str, Any]], candidate_layers: tuple[int, ...],
    thresholds: dict[str, dict[str, float]], role: str, *, per_finding: bool,
    draws: int, seed: int,
) -> dict[str, float | int | None]:
    # Error roles are attached by ``analyze`` using the actual final layer.
    selected = [row for row in rows if row["error_role"] == role]
    if not selected:
        return {"mean": None, "ci_low": None, "ci_high": None, "draws": draws}
    all_by_finding_truth: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if truth(row) != 0:
            all_by_finding_truth[(str(row["finding"]), truth(row))].append(row)
    observed = float(np.mean([
        any(
            truth(row) * (
                margin(row, layer) - threshold_for(thresholds, row, layer, per_finding)
            ) > 0.0
            for layer in candidate_layers
        )
        for row in selected
    ]))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        recovered = []
        for row in selected:
            pool = [
                donor
                for donor in all_by_finding_truth[(str(row["finding"]), truth(row))]
                if donor["record_key"] != row["record_key"]
            ]
            if not pool:
                raise ValueError("same-finding/same-truth shuffled donor pool is empty")
            # Draw one donor for the complete trajectory.  Sampling a different
            # donor at every layer destroys the natural layer covariance and
            # gives an unfairly easy null for an "any layer" oracle.
            donor = pool[int(rng.integers(0, len(pool)))]
            signs = []
            for layer in candidate_layers:
                centered = margin(donor, layer) - threshold_for(thresholds, row, layer, per_finding)
                signs.append(truth(row) * centered > 0.0)
            recovered.append(any(signs))
        values.append(float(np.mean(recovered)))
    low, high = np.quantile(values, [0.025, 0.975])
    null_mean = float(np.mean(values))
    return {
        "observed": observed,
        "mean": null_mean,
        "observed_minus_mean": observed - null_mean,
        "ci_low": float(low),
        "ci_high": float(high),
        "one_sided_p": float((1 + sum(value >= observed for value in values)) / (draws + 1)),
        "draws": draws,
        "unit": "whole within-finding/same-truth sampled-layer trajectory",
    }


def analyze(
    dev_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    layers = layers_of(dev_rows)
    if layers_of(test_rows) != layers:
        raise ValueError("development and test layer inventories differ")
    final_layer, candidate_layers = layers[-1], layers[:-1]
    thresholds = fit_thresholds(dev_rows, layers, per_finding=True)
    global_thresholds = fit_thresholds(dev_rows, layers, per_finding=False)
    records = []
    for row in test_rows:
        role = classify_error(row, final_layer)
        output = {
            "record_key": row["record_key"], "image_id": row["image_id"],
            "finding": row["finding"], "positive_votes": int(row["positive_votes"]),
            "error_role": role,
            "margins": {str(layer): margin(row, layer) for layer in layers},
        }
        if role in {"fp", "fn"}:
            output.update(recovery_flags(row, candidate_layers, thresholds, per_finding=True))
            global_flags = recovery_flags(row, candidate_layers, global_thresholds, per_finding=False)
            output["global_calibrated_recoverable"] = global_flags["calibrated_recoverable"]
        records.append(output)
    for source, attached in zip(test_rows, records):
        source["error_role"] = attached["error_role"]
    layer_diagnostics = {}
    for layer in layers:
        clear = [row for row in test_rows if truth(row) != 0]
        raw = np.asarray([margin(row, layer) for row in clear])
        labels = np.asarray([truth(row) for row in clear])
        calibrated_predictions = np.asarray([
            margin(row, layer) > threshold_for(thresholds, row, layer, True) for row in clear
        ])
        layer_diagnostics[str(layer)] = {
            "positive_margin_rate_all": float(np.mean([margin(row, layer) > 0 for row in test_rows])),
            "native_clear_accuracy": float(np.mean((raw > 0) == (labels == 1))),
            "calibrated_clear_accuracy": float(np.mean(calibrated_predictions == (labels == 1))),
        }
    role_counts = Counter(row["error_role"] for row in records)
    summaries = {}
    for role in ("fp", "fn"):
        summaries[role] = {
            "native": cluster_bootstrap(
                records, lambda row, value=role: row["error_role"] == value,
                "native_recoverable", draws=draws, seed=seed + (1 if role == "fp" else 2),
            ),
            "finding_calibrated": cluster_bootstrap(
                records, lambda row, value=role: row["error_role"] == value,
                "calibrated_recoverable", draws=draws, seed=seed + (3 if role == "fp" else 4),
            ),
            "global_calibrated": cluster_bootstrap(
                records, lambda row, value=role: row["error_role"] == value,
                "global_calibrated_recoverable", draws=draws, seed=seed + (5 if role == "fp" else 6),
            ),
            "finding_calibrated_shuffled_null": shuffled_null(
                test_rows, candidate_layers, thresholds, role, per_finding=True,
                draws=draws, seed=seed + (7 if role == "fp" else 8),
            ),
            "trajectory_counts": dict(sorted(Counter(
                row["sampled_trajectory"] for row in records if row["error_role"] == role
            ).items())),
        }
    return {
        "version": VERSION,
        "scope": "sampled-layer logit-lens screen; not causal evidence and not an all-layer oracle",
        "n_dev": len(dev_rows), "n_test": len(test_rows), "layers": list(layers),
        "candidate_layers": list(candidate_layers), "final_layer": final_layer,
        "truth_definition": "VinDr 3/3=supported, 0/3=refuted; 1/3 and 2/3 excluded from FP/FN",
        "actual_error_definition": "sign of unmodified final support-minus-refute logit",
        "role_counts": dict(sorted(role_counts.items())),
        "thresholds": thresholds,
        "global_thresholds": global_thresholds,
        "layer_diagnostics": layer_diagnostics,
        "recoverability": summaries,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-metadata", type=Path, required=True)
    parser.add_argument("--test-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(
        load_rows(args.dev_metadata), load_rows(args.test_metadata),
        draws=args.bootstrap_draws, seed=args.seed,
    )
    result["provenance"] = {
        "dev_metadata": str(args.dev_metadata.resolve()),
        "dev_metadata_sha256": sha256_file(args.dev_metadata),
        "test_metadata": str(args.test_metadata.resolve()),
        "test_metadata_sha256": sha256_file(args.test_metadata),
        "command": " ".join(__import__("sys").argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
