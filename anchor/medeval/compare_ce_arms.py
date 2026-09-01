#!/usr/bin/env python3
"""Paired, image-clustered CE-G comparison with a no-shortening cutoff."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from corrected_sgta.evaluate_medheval_answers import normalize_binary_reference, parse_answer

from .audit_retrieval_split import read_rows
from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "paired-ce-g-arm-comparison-v2-provenance"


def qid(row, index):
    return str(row.get("question_id", row.get("qid", row.get("id", index))))


def compare(manifest: list[dict], baseline: list[dict], candidate: list[dict], draws=5000, seed=42):
    refs = {qid(row, i): row for i, row in enumerate(manifest)}
    left, right = ({qid(row, i): row for i, row in enumerate(rows)} for rows in (baseline, candidate))
    if set(refs) != set(left) or set(refs) != set(right):
        raise ValueError("manifest and paired answer qids are not identical")
    records = []
    for sample_id in refs:
        truth = normalize_binary_reference(refs[sample_id].get("answer"))
        if truth is None:
            raise ValueError(f"non-binary reference: {sample_id}")
        values = []
        for row in (left[sample_id], right[sample_id]):
            text = str(row.get("text", ""))
            parsed = parse_answer(text, answer_type="binary")
            decision = parsed.labels[0] if parsed.labels else None
            tokens = int((row.get("metadata") or {}).get("generated_token_count", 0))
            values.append((decision == truth, decision is not None, tokens))
        records.append({"sample_id": sample_id, "cluster": str(refs[sample_id].get("img_name", refs[sample_id].get("image", sample_id))), "baseline":values[0], "candidate":values[1]})
    clusters = defaultdict(list)
    for row in records:
        clusters[row["cluster"]].append(row)
    cluster_ids = sorted(clusters)
    delta = sum(row["candidate"][0] - row["baseline"][0] for row in records) / len(records)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = rng.choice(cluster_ids, len(cluster_ids), replace=True)
        rows = [row for cluster in selected for row in clusters[str(cluster)]]
        samples.append(sum(row["candidate"][0] - row["baseline"][0] for row in rows) / len(rows))
    def rate(arm, field):
        return sum(row[arm][field] for row in records) / len(records)
    baseline_tokens, candidate_tokens = rate("baseline", 2), rate("candidate", 2)
    safety = {
        "candidate_parse_rate_not_lower_by_more_than_0.01": rate("candidate", 1) >= rate("baseline", 1) - .01,
        "candidate_mean_tokens_at_least_90pct_baseline": candidate_tokens >= .9 * baseline_tokens,
    }
    ci_low, ci_high = float(np.quantile(samples, .025)), float(np.quantile(samples, .975))
    return {
        "protocol_version": VERSION,
        "n": len(records),
        "clusters": len(clusters),
        "baseline_accuracy": rate("baseline", 0),
        "candidate_accuracy": rate("candidate", 0),
        "accuracy_delta": delta,
        "accuracy_delta_ci_low": ci_low,
        "accuracy_delta_ci_high": ci_high,
        "baseline_parse_rate": rate("baseline", 1),
        "candidate_parse_rate": rate("candidate", 1),
        "baseline_mean_generated_tokens": baseline_tokens,
        "candidate_mean_generated_tokens": candidate_tokens,
        "safety_gates": safety,
        "full_run_authorized": delta > 0 and ci_low > 0 and all(safety.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = compare(
        read_rows(args.manifest),
        read_rows(args.baseline),
        read_rows(args.candidate),
        args.bootstrap_draws,
        args.seed,
    )
    result["provenance"] = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "baseline_answers": str(args.baseline.resolve()),
        "baseline_answers_sha256": sha256_file(args.baseline),
        "candidate_answers": str(args.candidate.resolve()),
        "candidate_answers_sha256": sha256_file(args.candidate),
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "binary_parser_contract": "leading explicit Yes/No; invalid generations count as errors",
        "bootstrap_unit": "manifest image identity",
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
