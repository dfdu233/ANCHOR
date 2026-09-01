#!/usr/bin/env python3
"""Patient-clustered analysis for the natural-counterfactual score probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


VERSION = "natural-counterfactual-analysis-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effects(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            float(row["direction"])
            * (
                float(row["scores"]["current"]["yes_minus_no"])
                - float(row["scores"]["prior"]["yes_minus_no"])
            )
            for row in rows
        ],
        dtype=float,
    )


def cluster_bootstrap(rows: list[dict], draws: int, seed: int) -> dict[str, list[float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["patient_id"])].append(row)
    patients = sorted(grouped)
    rng = np.random.default_rng(seed)
    means, rates = [], []
    for _ in range(draws):
        sampled = rng.choice(patients, len(patients), replace=True)
        values = effects([row for patient in sampled for row in grouped[str(patient)]])
        means.append(float(values.mean()))
        rates.append(float((values > 0).mean()))
    return {
        "mean_95_ci": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "positive_rate_95_ci": [float(x) for x in np.quantile(rates, [0.025, 0.975])],
    }


def cluster_sign_flip_p(rows: list[dict], draws: int, seed: int) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, effects(rows), strict=True):
        grouped[str(row["patient_id"])].append(float(value))
    patients = sorted(grouped)
    patient_sums = np.asarray([sum(grouped[patient]) for patient in patients])
    observed = abs(float(patient_sums.sum()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        signs = rng.choice((-1.0, 1.0), size=len(patient_sums))
        exceed += abs(float((patient_sums * signs).sum())) >= observed
    return float((exceed + 1) / (draws + 1))


def summarize(rows: list[dict], draws: int, seed: int) -> dict[str, object]:
    values = effects(rows)
    return {
        "n": len(rows),
        "n_patients": len({row["patient_id"] for row in rows}),
        "mean_signed_delta": float(values.mean()),
        "median_signed_delta": float(np.median(values)),
        "positive_n": int((values > 0).sum()),
        "positive_rate": float((values > 0).mean()),
        "tie_n": int((values == 0).sum()),
        **cluster_bootstrap(rows, draws, seed),
        "patient_cluster_sign_flip_two_sided_p": cluster_sign_flip_p(rows, draws, seed + 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    all_rows = [json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()]
    rows = [row for row in all_rows if row.get("status") == "ok"]
    if not rows:
        raise RuntimeError("no successful rows")
    duplicate_keys = [key for key, n in Counter(row["record_key"] for row in rows).items() if n > 1]
    if duplicate_keys:
        raise RuntimeError(f"duplicate record keys: {duplicate_keys[:5]}")

    overall = summarize(rows, args.bootstrap_draws, args.seed)
    by_finding = {}
    for index, finding in enumerate(sorted({row["finding"] for row in rows})):
        subset = [row for row in rows if row["finding"] == finding]
        by_finding[finding] = summarize(subset, args.bootstrap_draws, args.seed + 100 + index)
    by_direction = {}
    for index, direction in enumerate(("new", "resolved")):
        subset = [row for row in rows if row["direction_name"] == direction]
        if subset:
            by_direction[direction] = summarize(
                subset, args.bootstrap_draws, args.seed + 200 + index
            )

    mean_ci = overall["mean_95_ci"]
    preliminary = bool(
        overall["n"] >= 40
        and overall["n_patients"] >= 20
        and mean_ci[0] > 0
        and overall["positive_rate"] > 0.5
        and overall["patient_cluster_sign_flip_two_sided_p"] < 0.05
    )
    result = {
        "version": VERSION,
        "status": "preliminary_signal" if preliminary else "preliminary_no_go",
        "estimand": (
            "direction * (current Yes-minus-No margin - prior Yes-minus-No margin); "
            "positive means the single-image claim score follows the reported clinical transition"
        ),
        "overall": overall,
        "by_finding": by_finding,
        "by_direction": by_direction,
        "errors": len(all_rows) - len(rows),
        "audit": {
            "all_images_distinct_within_pair": all(
                row["current_image"] != row["prior_image"] for row in rows
            ),
            "same_view_position_within_pair": all(row.get("view_position") in {"AP", "PA"} for row in rows),
            "label_source": "Medical-Diff-VQA, automatically derived from MIMIC reports",
            "independent_doctor_adjudication": False,
            "raw_sha256": sha256(args.raw),
        },
        "decision": {
            "preliminary_directional_signal_admitted": preliminary,
            "authorize_evidence_head_training": False,
            "reason": (
                "Even a positive silver result requires replication on unanimous expert labels "
                "before training; a negative result kills this native-score route immediately."
            ),
        },
        "interpretation_boundary": (
            "Real images remove edit artifacts, but report-derived labels, small n, one model, "
            "and residual clinical/acquisition confounding prevent a causal or general claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
