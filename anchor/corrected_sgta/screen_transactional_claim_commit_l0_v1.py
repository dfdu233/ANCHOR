#!/usr/bin/env python3
"""CPU-only L0 for specialist-vetoed transactional claim replacement.

This is an optimistic diagnostic, not a proposed method.  It reuses a
development-fitted XRV veto from the existing one-bit falsification audit.  On
at most 32 hash-selected confirmation images, it preserves the VLM's number of
positive claims K: vetoed claims are rolled back and the next highest-margin
non-vetoed VLM claims are committed.

The candidate already collides with verifier-guided rejection/resampling.  The
purpose of this script is only to test whether the local evidence contains a
cross-model safety signal strong enough to justify any further thought.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from anchor.corrected_sgta.analyze_xrv_one_bit_veto_v1 import (
    choose_veto_threshold,
    fit_expert_probability,
    load_logits,
)
from anchor.corrected_sgta.screen_external_visual_increment_v1 import load_claims


VERSION = "transactional-claim-commit-l0-v1"


def hash_key(image_id: str) -> str:
    return hashlib.sha256(f"{VERSION}:{image_id}".encode()).hexdigest()


def audit_model(
    model_name: str,
    development_path: Path,
    confirmation_path: Path,
    logits: dict[str, Any],
    limit: int,
    harm_budget: float,
    seed: int,
) -> dict[str, Any]:
    development = load_claims(development_path, "development", model_name)
    confirmation = load_claims(confirmation_path, "confirmation", model_name)
    p_development, p_confirmation = fit_expert_probability(
        development, confirmation, logits, seed
    )
    threshold = choose_veto_threshold(development, p_development, harm_budget)

    groups: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, probability in zip(confirmation, p_confirmation):
        groups[row["image_id"]].append((row, float(probability)))

    eligible = []
    for image_id, rows in groups.items():
        k = sum(row["margin"] > 0 for row, _ in rows)
        accepted = sum(probability >= threshold for _, probability in rows)
        if len(rows) >= 2 and 0 < k < len(rows) and accepted >= k:
            eligible.append((hash_key(image_id), image_id, rows))
    selected = sorted(eligible)[:limit]

    counts = {
        "baseline_fp": 0,
        "baseline_fn": 0,
        "transactional_fp": 0,
        "transactional_fn": 0,
        "vetoed_fp": 0,
        "vetoed_tp": 0,
        "replaced_claims": 0,
        "content_budget_k": 0,
        "claims": 0,
    }
    image_rows = []
    for _, image_id, rows in selected:
        k = sum(row["margin"] > 0 for row, _ in rows)
        baseline = {i for i, (row, _) in enumerate(rows) if row["margin"] > 0}
        allowed = [i for i, (_, probability) in enumerate(rows) if probability >= threshold]
        transactional = set(
            sorted(allowed, key=lambda i: rows[i][0]["margin"], reverse=True)[:k]
        )
        if len(transactional) != k:
            raise AssertionError("fixed-K content invariant failed")

        before_errors = after_errors = 0
        for i, (row, probability) in enumerate(rows):
            label = int(row["label"])
            counts["baseline_fp"] += int(i in baseline and label == 0)
            counts["baseline_fn"] += int(i not in baseline and label == 1)
            counts["transactional_fp"] += int(i in transactional and label == 0)
            counts["transactional_fn"] += int(i not in transactional and label == 1)
            if i in baseline and probability < threshold:
                counts["vetoed_fp"] += int(label == 0)
                counts["vetoed_tp"] += int(label == 1)
            before_errors += int((i in baseline) != bool(label))
            after_errors += int((i in transactional) != bool(label))

        replaced = len(baseline - transactional)
        counts["replaced_claims"] += replaced
        counts["content_budget_k"] += k
        counts["claims"] += len(rows)
        image_rows.append(
            {
                "image_id": image_id,
                "claims": len(rows),
                "k": k,
                "replaced": replaced,
                "baseline_errors": before_errors,
                "transactional_errors": after_errors,
            }
        )

    before = counts["baseline_fp"] + counts["baseline_fn"]
    after = counts["transactional_fp"] + counts["transactional_fn"]
    return {
        "model": model_name,
        "threshold": threshold,
        "images": len(selected),
        "eligible_images": len(eligible),
        **counts,
        "baseline_total_errors": before,
        "transactional_total_errors": after,
        "error_delta_after_minus_before": after - before,
        "relative_error_reduction": (before - after) / max(before, 1),
        "fixed_k_verified": all(row["k"] >= row["replaced"] for row in image_rows),
        "image_rows": image_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-dev", type=Path, required=True)
    parser.add_argument("--huatuo-confirmation", type=Path, required=True)
    parser.add_argument("--hulu-dev", type=Path, required=True)
    parser.add_argument("--hulu-confirmation", type=Path, required=True)
    parser.add_argument("--xrv-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--harm-budget", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.limit <= 32:
        raise ValueError("L0 limit must be between 1 and 32")

    logits = load_logits(args.xrv_logits)
    analyses = {
        "huatuo": audit_model(
            "huatuo",
            args.huatuo_dev,
            args.huatuo_confirmation,
            logits,
            args.limit,
            args.harm_budget,
            args.seed,
        ),
        "hulu": audit_model(
            "hulu",
            args.hulu_dev,
            args.hulu_confirmation,
            logits,
            args.limit,
            args.harm_budget,
            args.seed,
        ),
    }
    passes = [
        row["relative_error_reduction"] >= 0.20
        and row["transactional_fn"] <= row["baseline_fn"]
        and row["fixed_k_verified"]
        for row in analyses.values()
    ]
    result = {
        "version": VERSION,
        "status": "complete_cpu_cache_only",
        "decision": "GO" if all(passes) else "NO_GO",
        "decision_rule": (
            "Both models must reduce fixed-K total claim errors by at least 20%, "
            "with no FN increase. This small L0 has no inferential CI and cannot establish a method."
        ),
        "collision_boundary": (
            "The operation is hard verifier-guided rejection/resampling and is not novel even if positive."
        ),
        "optimism_warning": (
            "The specialist probability and veto threshold are fitted with development labels; "
            "therefore this is an optimistic upper-bound diagnostic and violates the final "
            "calibration-free requirement."
        ),
        "analyses": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
