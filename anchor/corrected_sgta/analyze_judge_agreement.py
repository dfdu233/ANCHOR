#!/usr/bin/env python3
"""Check local-judge or annotator agreement before OE evidence is admissible."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from corrected_sgta.clinical_judgments import load_judgments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--score-field", default="hallucination_score")
    parser.add_argument("--min-kappa", type=float, default=0.60)
    parser.add_argument("--min-spearman", type=float, default=0.70)
    parser.add_argument("--min-shared", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_bytes, right_bytes = args.left.read_bytes(), args.right.read_bytes()
    left_hash = hashlib.sha256(left_bytes).hexdigest()
    right_hash = hashlib.sha256(right_bytes).hexdigest()
    if left_hash == right_hash:
        raise RuntimeError("left and right annotation files must be distinct")
    left, right = load_judgments(args.left), load_judgments(args.right)
    shared = sorted(set(left) & set(right))
    values = []
    left_judges, right_judges, bundles, fingerprints = set(), set(), set(), set()
    for item in shared:
        left_row, right_row = left[item], right[item]
        left_score, right_score = left_row.get(args.score_field), right_row.get(args.score_field)
        if left_score is None or right_score is None:
            continue
        if (
            type(left_score) is not int or type(right_score) is not int
            or not 0 <= left_score <= 5 or not 0 <= right_score <= 5
        ):
            raise ValueError(f"{item}: hallucination scores must be JSON integers in [0, 5]")
        if (
            left_row.get("rubric_version") != "MedHEval-0-5-v1"
            or right_row.get("rubric_version") != "MedHEval-0-5-v1"
        ):
            raise ValueError(f"{item}: unexpected or missing rubric_version")
        if left_row.get("annotation_bundle_id") != right_row.get("annotation_bundle_id"):
            raise ValueError(f"{item}: annotation bundle mismatch")
        if left_row.get("cache_fingerprint") != right_row.get("cache_fingerprint"):
            raise ValueError(f"{item}: cache fingerprint mismatch")
        left_judges.add(str(left_row.get("annotator_id", "")).strip())
        right_judges.add(str(right_row.get("annotator_id", "")).strip())
        bundles.add(str(left_row.get("annotation_bundle_id", "")).strip())
        fingerprints.add(str(left_row.get("cache_fingerprint", "")).strip())
        values.append((left_score, right_score))
    if len(values) < 2:
        raise RuntimeError("at least two shared scored items are required")
    if (
        len(left_judges) != 1 or len(right_judges) != 1
        or "" in left_judges or "" in right_judges
        or left_judges == right_judges
    ):
        raise RuntimeError("annotation files require distinct, non-empty single annotator_id values")
    if len(bundles) != 1 or "" in bundles or len(fingerprints) != 1 or "" in fingerprints:
        raise RuntimeError("all shared rows must bind to one non-empty bundle and fingerprint")
    left_values = [value[0] for value in values]
    right_values = [value[1] for value in values]
    kappa = float(cohen_kappa_score(left_values, right_values, weights="quadratic"))
    spearman = float(spearmanr(left_values, right_values).statistic)
    passed = (
        len(values) >= args.min_shared
        and kappa >= args.min_kappa
        and spearman >= args.min_spearman
    )
    report = {
        "validation_type": "knowledge_judge_agreement",
        "left_sha256": left_hash,
        "right_sha256": right_hash,
        "left_annotator_id": next(iter(left_judges)),
        "right_annotator_id": next(iter(right_judges)),
        "annotation_bundle_id": next(iter(bundles)),
        "cache_fingerprint": next(iter(fingerprints)),
        "rubric_version": "MedHEval-0-5-v1",
        "n_shared_scored": len(values),
        "quadratic_cohen_kappa": kappa,
        "spearman": spearman,
        "thresholds": {
            "min_shared": args.min_shared,
            "min_kappa": args.min_kappa,
            "min_spearman": args.min_spearman,
        },
        "passed": passed,
        "paper_status": "eligible" if passed else "exploratory_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
