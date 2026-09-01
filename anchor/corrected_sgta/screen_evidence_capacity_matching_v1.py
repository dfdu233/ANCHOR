#!/usr/bin/env python3
"""CPU fatal screen for claim-to-region capacity matching.

The method under test treats claim selection as a capacitated bipartite
matching problem: a coarse visual region cannot independently support an
unlimited number of distinct findings.  It is evaluated at the same positive
claim budget K as independent per-finding top-K selection, so gains cannot come
from saying less.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
CSV_NAMES = {
    "Aortic enlargement": "aortic_enlargement",
    "Cardiomegaly": "cardiomegaly",
    "Lung Opacity": "lung_opacity",
    "Nodule/Mass": "nodule_mass",
    "Pleural effusion": "pleural_effusion",
    "Pleural thickening": "pleural_thickening",
    "Pulmonary fibrosis": "pulmonary_fibrosis",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_votes(path: Path) -> dict[str, dict[str, set[str]]]:
    votes: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            finding = CSV_NAMES.get(row["class_name"])
            if finding is not None:
                votes[row["image_id"]][finding].add(row["rad_id"])
    return votes


def coarse_regions(scores: np.ndarray, side: int) -> np.ndarray:
    # scores: image x 24 x 24 x finding.  sqrt(area)-normalized sums retain
    # comparable null scale while allowing weak evidence to accumulate.
    grid = scores.shape[1]
    if grid % side:
        raise ValueError(f"coarse side {side} must divide patch side {grid}")
    width = grid // side
    output = scores.reshape(len(scores), side, width, side, width, len(FINDINGS))
    output = output.mean(axis=(2, 4)) * width
    return output.reshape(len(scores), side * side, len(FINDINGS)).transpose(0, 2, 1)


def topk_independent(matrix: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(matrix.max(axis=1), kind="stable")[-k:].tolist())


def capacity_match(matrix: np.ndarray, k: int, capacity: int) -> set[int]:
    expanded = np.repeat(matrix, capacity, axis=1)
    rows, _ = linear_sum_assignment(-expanded)
    # linear_sum_assignment assigns every claim row when rows <= columns; pick
    # the globally best K claims by padding with K dummy rows is unnecessary at
    # seven claims, so enumerate subsets exactly and keep the definition clear.
    del rows
    from itertools import combinations

    best_subset: tuple[int, ...] | None = None
    best_score = -np.inf
    for subset in combinations(range(len(FINDINGS)), k):
        row, col = linear_sum_assignment(-expanded[list(subset)])
        value = float(expanded[list(subset)][row, col].sum())
        if value > best_score:
            best_score = value
            best_subset = subset
    if best_subset is None:
        raise RuntimeError("empty matching subset")
    return set(best_subset)


def paired_bootstrap(rows: list[tuple[int, int, int]], draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    delta = np.empty(draws, dtype=float)
    for draw in range(draws):
        sample = rng.integers(0, len(rows), len(rows))
        subset = [rows[index] for index in sample]
        denominator = sum(value[2] for value in subset)
        delta[draw] = sum(value[1] - value[0] for value in subset) / denominator
    return {
        "draws": draws,
        "unit": "image",
        "micro_recall_delta_ci95": [float(x) for x in np.quantile(delta, [0.025, 0.975])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--vindr-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coarse-side", type=int, default=6)
    parser.add_argument("--capacities", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    metadata_path = args.patch_dir / "metadata.jsonl"
    scores_path = args.patch_dir / "patch_scores.npz"
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    scores = np.load(scores_path)["patch_scores"].astype(np.float64)
    scores = scores.reshape(len(metadata), 24, 24, len(FINDINGS))
    votes = load_votes(args.vindr_csv)
    labels = np.asarray(
        [[len(votes[row["image_id"]][finding]) for finding in FINDINGS] for row in metadata],
        dtype=int,
    )

    # Development-only, finding- and position-specific null standardization.
    zscores = np.empty_like(scores)
    for finding in range(len(FINDINGS)):
        indexes = np.asarray(
            [row["split"] == "development" and labels[i, finding] == 0 for i, row in enumerate(metadata)]
        )
        mean = scores[indexes, :, :, finding].mean(axis=0)
        std = scores[indexes, :, :, finding].std(axis=0)
        positive = std[std > 0]
        floor = float(np.quantile(positive, 0.1))
        zscores[:, :, :, finding] = (scores[:, :, :, finding] - mean) / np.maximum(std, floor)
    regions = coarse_regions(zscores, args.coarse_side)

    results: dict[str, dict] = {}
    for capacity in args.capacities:
        split_results = {}
        for split in ("development", "confirmation"):
            rows: list[tuple[int, int, int]] = []
            exact_independent = 0
            exact_capacity = 0
            collision_true = []
            collision_false = []
            for index, row in enumerate(metadata):
                if row["split"] != split or not np.all(np.isin(labels[index], (0, 3))):
                    continue
                truth = set(np.where(labels[index] == 3)[0].tolist())
                k = len(truth)
                if not 0 < k < len(FINDINGS):
                    continue
                independent = topk_independent(regions[index], k)
                matched = capacity_match(regions[index], k, capacity)
                rows.append((len(independent & truth), len(matched & truth), k))
                exact_independent += independent == truth
                exact_capacity += matched == truth
                peaks = np.argmax(regions[index], axis=1)
                collision_true.append(len({int(peaks[c]) for c in truth}) < len(truth))
                false_claims = set(range(len(FINDINGS))) - truth
                collision_false.append(len({int(peaks[c]) for c in false_claims}) < len(false_claims))
            denominator = sum(value[2] for value in rows)
            independent_hits = sum(value[0] for value in rows)
            capacity_hits = sum(value[1] for value in rows)
            split_results[split] = {
                "n_images": len(rows),
                "n_true_claims": denominator,
                "independent_micro_recall": independent_hits / denominator,
                "capacity_micro_recall": capacity_hits / denominator,
                "micro_recall_delta": (capacity_hits - independent_hits) / denominator,
                "independent_exact_set_accuracy": exact_independent / len(rows),
                "capacity_exact_set_accuracy": exact_capacity / len(rows),
                "true_claim_peak_collision_rate": float(np.mean(collision_true)),
                "false_claim_peak_collision_rate": float(np.mean(collision_false)),
                "bootstrap": paired_bootstrap(rows, args.bootstrap_draws, args.seed + capacity),
            }
        results[str(capacity)] = split_results

    primary = results["1"]["confirmation"]
    passed = (
        primary["micro_recall_delta"] >= 0.02
        and primary["bootstrap"]["micro_recall_delta_ci95"][0] > 0
        and primary["capacity_exact_set_accuracy"] >= primary["independent_exact_set_accuracy"]
    )
    payload = {
        "version": "evidence-capacity-matching-screen-v1",
        "hypothesis": "distinct clinical claims should compete for finite visual-region support instead of independently reusing the same evidence peak",
        "fixed_content_contract": "both methods select exactly the ground-truth number K of positive claims per image",
        "method": "maximum-weight capacitated bipartite claim-region matching versus independent per-claim max followed by top-K",
        "inputs": {
            "patch_scores": str(scores_path.resolve()),
            "patch_scores_sha256": sha256(scores_path),
            "metadata_sha256": sha256(metadata_path),
            "vindr_csv_sha256": sha256(args.vindr_csv),
        },
        "configuration": {
            "findings": list(FINDINGS),
            "coarse_regions": args.coarse_side * args.coarse_side,
            "capacities": args.capacities,
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
            "null_standardization": "development vote-0, finding x patch-position mean/std",
        },
        "results": results,
        "gate": {
            "rule": "confirmation capacity-1 micro recall delta >=0.02, image-bootstrap CI lower>0, exact-set accuracy non-inferior",
            "pass": passed,
        },
        "decision": "GO_TO_COLLISION_AND_OPEN_GENERATION" if passed else "NO_GO",
        "claim_boundary": "Even a pass would establish only a fixed-K claim-selection signal from supervised patch directions, not natural open-generation mitigation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["gate"] | {"decision": payload["decision"], "primary": primary}, indent=2))


if __name__ == "__main__":
    main()
