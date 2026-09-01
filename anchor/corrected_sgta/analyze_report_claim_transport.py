#!/usr/bin/env python3
"""Audit fixed-K report claim transport without treating missing truth as negative."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.analyze_no_free_grounding import sha256_file


VERSION = "unknown-aware-report-claim-transport-v1"
STATES = ("supported", "refuted", "undetermined", "unverified")


def _counts(image: dict[str, Any], selected: set[str]) -> dict[str, Any]:
    states = image["reference_states"]
    counts = {state: sum(states[finding] == state for finding in selected) for state in STATES}
    supported_total = sum(state == "supported" for state in states.values())
    verified = counts["supported"] + counts["refuted"]
    return {
        "k": len(selected),
        **counts,
        "verified": verified,
        "verified_precision": counts["supported"] / verified if verified else None,
        "supported_recall": counts["supported"] / supported_total if supported_total else None,
        "verified_coverage": verified / len(selected) if selected else None,
        "supported_total": supported_total,
    }


def evaluate_images(
    manifest: dict[str, Any], score_rows: list[dict[str, Any]], score_name: str, split: str
) -> list[dict[str, Any]]:
    scores = {
        (str(row["image"]), str(row["question"])): float(row["scores"][score_name])
        for row in score_rows if row.get("status") == "ok"
    }
    images = []
    for image in manifest["result"]["images"]:
        if split != "all" and image["split"] != split:
            continue
        score_by_finding = {}
        for finding in manifest["result"]["findings"]:
            question = f"Does this chest X-ray show {finding.replace('_', ' ')}?"
            key = (str(image["image"]), question)
            if key not in scores:
                raise ValueError(f"missing score for {key}")
            score_by_finding[finding] = scores[key]
        order = sorted(score_by_finding, key=lambda finding: (-score_by_finding[finding], finding))
        baseline_set = set(image["draft_selected"])
        candidate_set = set(order[: image["k"]])
        baseline = _counts(image, baseline_set)
        candidate = _counts(image, candidate_set)
        if baseline["k"] != candidate["k"]:
            raise AssertionError("claim budget changed")
        images.append({
            "image_id": image["image_id"],
            "baseline": baseline,
            "candidate": candidate,
            "removed": sorted(baseline_set - candidate_set),
            "added": sorted(candidate_set - baseline_set),
        })
    if not images:
        raise ValueError("no images in selected split")
    return images


def aggregate(images: list[dict[str, Any]], name: str) -> dict[str, Any]:
    keys = ("k", *STATES, "verified", "supported_total")
    total = {key: sum(image[name][key] for image in images) for key in keys}
    total.update({
        "verified_precision": total["supported"] / total["verified"] if total["verified"] else None,
        "supported_recall": total["supported"] / total["supported_total"] if total["supported_total"] else None,
        "verified_coverage": total["verified"] / total["k"] if total["k"] else None,
    })
    return total


def _optional_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _deltas(images: list[dict[str, Any]]) -> dict[str, float | None]:
    left, right = aggregate(images, "candidate"), aggregate(images, "baseline")
    return {
        "supported_count": float(left["supported"] - right["supported"]),
        "refuted_count": float(left["refuted"] - right["refuted"]),
        "undetermined_count": float(left["undetermined"] - right["undetermined"]),
        "unverified_count": float(left["unverified"] - right["unverified"]),
        "verified_precision": _optional_delta(
            left["verified_precision"], right["verified_precision"]
        ),
        "supported_recall": _optional_delta(
            left["supported_recall"], right["supported_recall"]
        ),
        "verified_coverage": _optional_delta(
            left["verified_coverage"], right["verified_coverage"]
        ),
    }


def analyze(images: list[dict[str, Any]], *, draws: int, seed: int) -> dict[str, Any]:
    observed = _deltas(images)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in observed}
    for _ in range(draws):
        indices = rng.integers(0, len(images), len(images))
        delta = _deltas([images[index] for index in indices])
        for key, value in delta.items():
            if value is not None and np.isfinite(value):
                samples[key].append(value)
    intervals = {}
    for key, value in observed.items():
        finite = samples[key]
        intervals[key] = {
            "estimate": value,
            "ci_low": float(np.quantile(finite, 0.025)) if finite else None,
            "ci_high": float(np.quantile(finite, 0.975)) if finite else None,
            "valid_bootstrap_draws": len(finite),
        }
    required = (
        intervals["supported_recall"]["ci_low"],
        intervals["refuted_count"]["ci_high"],
        intervals["unverified_count"]["ci_high"],
    )
    passed = (
        all(value is not None for value in required)
        and required[0] > 0
        and required[1] < 0
        and required[2] <= 0
    )
    return {
        "n_images": len(images),
        "baseline": aggregate(images, "baseline"),
        "candidate": aggregate(images, "candidate"),
        "candidate_minus_baseline_image_bootstrap": intervals,
        "screening_gate": {
            "passed": passed,
            "rule": (
                "supported recall CI_low > 0, refuted-count CI_high < 0, and "
                "unverified-count CI_high <= 0"
            ),
        },
        "conservation": (
            "For every image K is fixed: delta(supported + refuted + undetermined + "
            "unverified) = 0. Only refuted-to-supported transport is a clean gain."
        ),
        "evidence_ceiling": "Single-reference report extraction is grade C screening evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-name", default="original_margin")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1223)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    score_rows = [json.loads(line) for line in args.scores.read_text().splitlines() if line.strip()]
    images = evaluate_images(manifest, score_rows, args.score_name, args.split)
    payload = {
        "config": {
            "version": VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": sha256_file(args.manifest),
            "scores_sha256": sha256_file(args.scores),
            "score_name": args.score_name,
            "split": args.split,
            "code_sha256": sha256_file(Path(__file__)),
        },
        "result": analyze(images, draws=args.bootstrap_draws, seed=args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
