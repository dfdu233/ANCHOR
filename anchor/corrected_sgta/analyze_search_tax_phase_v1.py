#!/usr/bin/env python3
"""Audit claim×region search inflation on a reader-unanimous global null.

This is a phenomenon screen, not a mitigation result.  It deliberately uses
only images on which all seven frozen claims received 0/3 reader votes, so an
off-claim response cannot silently be a true positive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "search-tax-phase-audit-v2-global-reader-null"
SEED = 20260812
BOOTSTRAP_DRAWS = 5000
FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
CSV_COLUMNS = {
    "aortic_enlargement": "Aortic enlargement",
    "cardiomegaly": "Cardiomegaly",
    "lung_opacity": "Lung Opacity",
    "nodule_mass": "Nodule/Mass",
    "pleural_effusion": "Pleural effusion",
    "pleural_thickening": "Pleural thickening",
    "pulmonary_fibrosis": "Pulmonary fibrosis",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_ids(directory: Path) -> set[str]:
    return {json.loads(line)["image_id"] for line in (directory / "metadata.jsonl").read_text().splitlines()}


def globally_null_ids(path: Path) -> tuple[set[str], dict[str, Any]]:
    votes: dict[str, dict[str, list[int]]] = {}
    reader_ids: dict[str, set[str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            image = row["image_id"]
            votes.setdefault(image, {finding: [] for finding in FINDINGS})
            reader_ids.setdefault(image, set()).add(row["rad_id"])
            for finding, column in CSV_COLUMNS.items():
                votes[image][finding].append(int(row[column]))
    eligible = {
        image
        for image, finding_votes in votes.items()
        if len(reader_ids[image]) == 3
        and all(len(finding_votes[finding]) == 3 and sum(finding_votes[finding]) == 0 for finding in FINDINGS)
    }
    return eligible, {
        "images_in_csv": len(votes),
        "globally_reader_unanimous_null_images": len(eligible),
        "required_readers": 3,
        "required_vote_sum_per_claim": 0,
    }


def patch_artifact(directory: Path) -> tuple[np.ndarray, dict[str, int], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in (directory / "metadata.jsonl").read_text().splitlines()]
    scores = np.asarray(np.load(directory / "patch_scores.npz")["patch_scores"], dtype=float)
    if len(rows) != scores.shape[0]:
        raise ValueError("patch score/metadata length mismatch")
    index = {row["image_id"]: i for i, row in enumerate(rows)}
    if len(index) != len(rows):
        raise ValueError("patch artifact repeats image ids")
    return scores, index, rows


def estimate_patch_null(
    ids: list[str], scores: np.ndarray, image_index: dict[str, int]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    null = {}
    for column, finding in enumerate(FINDINGS):
        values = np.stack([scores[image_index[image], :, column] for image in ids])
        mean, std = values.mean(0), values.std(0)
        positive = std[std > 0]
        floor = float(np.quantile(positive, 0.10)) if len(positive) else 1.0
        null[finding] = (mean, np.maximum(std, floor))
    return null


def responses(
    ids: list[str],
    scores: np.ndarray,
    image_index: dict[str, int],
    null: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[int, dict[str, np.ndarray]]:
    output = {k: {"raw": [], "analytic": []} for k in (1, 2, 4, 7)}
    for image in ids:
        image_scores = scores[image_index[image]]
        seed = int(hashlib.sha256(f"{SEED}|{image}".encode()).hexdigest()[:16], 16)
        order = np.random.default_rng(seed).permutation(len(FINDINGS))
        accumulated: list[float] = []
        for count, column in enumerate(order, start=1):
            finding = FINDINGS[int(column)]
            mean, std = null[finding]
            accumulated.extend(((image_scores[:, column] - mean) / std).tolist())
            if count not in output:
                continue
            maximum = float(np.max(accumulated))
            output[count]["raw"].append(maximum)
            output[count]["analytic"].append(maximum - math.sqrt(2.0 * math.log(len(accumulated))))
    return {
        k: {name: np.asarray(values, dtype=float) for name, values in payload.items()}
        for k, payload in output.items()
    }


def ci(values: list[float]) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu"), required=True)
    parser.add_argument("--development-hidden", type=Path, required=True)
    parser.add_argument("--confirmation-hidden", type=Path, required=True)
    parser.add_argument("--patch-scores", type=Path, required=True)
    parser.add_argument("--reader-labels-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    scores, image_index, patch_rows = patch_artifact(args.patch_scores)
    global_null, label_audit = globally_null_ids(args.reader_labels_csv)
    dev_ids = sorted(metadata_ids(args.development_hidden) & global_null & image_index.keys())
    test_ids = sorted(metadata_ids(args.confirmation_hidden) & global_null & image_index.keys())
    if len(dev_ids) < 100 or len(test_ids) < 50:
        raise ValueError(f"underpowered global null: development={len(dev_ids)}, confirmation={len(test_ids)}")
    null = estimate_patch_null(dev_ids, scores, image_index)
    dev, test = responses(dev_ids, scores, image_index, null), responses(test_ids, scores, image_index, null)

    metrics: dict[str, Any] = {}
    for k in (1, 2, 4, 7):
        threshold = float(np.quantile(dev[k]["raw"], 0.95))
        metrics[str(k)] = {
            "development_n": len(dev_ids),
            "confirmation_n": len(test_ids),
            "search_size": int(k * scores.shape[1]),
            "raw_mean": float(test[k]["raw"].mean()),
            "raw_p95": float(np.quantile(test[k]["raw"], 0.95)),
            "analytic_mean": float(test[k]["analytic"].mean()),
            "analytic_p95": float(np.quantile(test[k]["analytic"], 0.95)),
            "development_empirical_p95_threshold": threshold,
            "confirmation_exceedance_of_dev_p95": float(np.mean(test[k]["raw"] > threshold)),
        }

    rng = np.random.default_rng(SEED)
    raw_delta, analytic_delta = [], []
    indices = np.arange(len(test_ids))
    for _ in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(indices, len(indices), replace=True)
        raw_delta.append(float(np.quantile(test[7]["raw"][sample], 0.95) - np.quantile(test[1]["raw"][sample], 0.95)))
        analytic_delta.append(float(np.quantile(test[7]["analytic"][sample], 0.95) - np.quantile(test[1]["analytic"][sample], 0.95)))
    raw_growth = metrics["7"]["raw_p95"] - metrics["1"]["raw_p95"]
    analytic_growth = metrics["7"]["analytic_p95"] - metrics["1"]["analytic_p95"]
    phase = {
        "raw_p95_growth_k1_to_k7": raw_growth,
        "raw_growth_ci95": ci(raw_delta),
        "analytic_p95_growth_k1_to_k7": analytic_growth,
        "analytic_growth_ci95": ci(analytic_delta),
        "gate": bool(ci(raw_delta)[0] > 0),
        "gate_rule": "raw K=7 minus K=1 p95 bootstrap lower 95% bound > 0",
    }
    result = {
        "version": VERSION,
        "status": "complete",
        "model": args.model,
        "scope": "claim×patch search inflation on images with 0/3 reader votes for every searched claim",
        "reader_null_audit": {**label_audit, "development_n": len(dev_ids), "confirmation_n": len(test_ids)},
        "metrics_by_claim_count": metrics,
        "phase_test": phase,
        "configuration": {
            "claim_counts": [1, 2, 4, 7],
            "claims": list(FINDINGS),
            "claim_order": "nested image-specific permutation frozen by sha256(seed|image_id)",
            "analytic_reference_only": "subtract sqrt(2 log(K * patch_tokens)); not claimed valid under correlated patches",
            "empirical_reference": "development global-null p95 replayed on confirmation",
            "seed": SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "command": " ".join(sys.argv),
            "source_sha256": sha256(Path(__file__)),
            "reader_labels_csv": str(args.reader_labels_csv.resolve()),
            "reader_labels_sha256": sha256(args.reader_labels_csv),
            "patch_score_sha256": sha256(args.patch_scores / "patch_scores.npz"),
            "patch_metadata_sha256": sha256(args.patch_scores / "metadata.jsonl"),
        },
        "boundary": (
            "A pass establishes raw internal search inflation only. It does not establish selection-reuse "
            "inflation in a second validator, propagation to decoder margin, or hallucination mitigation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(phase, indent=2))


if __name__ == "__main__":
    main()
