#!/usr/bin/env python3
"""Analyze fixed-K reranking when draft predictions and scores share one run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from corrected_sgta.analyze_claim_reranking import (
    SCORE_NAMES,
    aggregate,
    bootstrap_deltas,
    evaluate_image,
)
from corrected_sgta.analyze_no_free_grounding import _binary, sha256_file


VERSION = "draft-conditioned-fixed-k-reranking-screen-v1"


def analyze(
    records: list[dict], *, draws: int, seed: int,
    draft_records: list[dict] | None = None,
) -> dict:
    records = [record for record in records if record.get("status") == "ok"]
    if not records:
        raise ValueError("successful score records are required")
    drafts = None
    if draft_records is not None:
        drafts = {
            int(record["question_id"]): record
            for record in draft_records if record.get("status") == "ok"
        }
        if set(drafts) != {int(record["question_id"]) for record in records}:
            raise ValueError("draft and score records are not exactly aligned")
    elif any("draft" not in record for record in records):
        raise ValueError("generated drafts are required")
    rows = []
    for record in records:
        truth = _binary(record["truth"])
        draft = drafts[int(record["question_id"])] if drafts is not None else record
        baseline = _binary(draft["draft"].get("prediction"))
        if truth == "invalid":
            raise ValueError(f"invalid truth for qid={record['question_id']}")
        rows.append({
            "question_id": int(record["question_id"]),
            "image": str(record["image"]),
            "truth": truth,
            "baseline": baseline,
            "scores": {name: float(record["scores"][name]) for name in SCORE_NAMES},
        })
    grouped = {}
    for row in rows:
        grouped.setdefault(row["image"], []).append(row)
    images = [evaluate_image(grouped[name]) for name in sorted(grouped)]
    names = ("baseline", *SCORE_NAMES)
    metrics = {name: aggregate(images, name) for name in names}
    deltas = {
        name: bootstrap_deltas(images, name, "baseline", draws, seed + index * 10)
        for index, name in enumerate(SCORE_NAMES)
    }
    centered_vs_original = bootstrap_deltas(
        images, "null_centered_margin", "original_margin", draws, seed + 100
    )
    centered = deltas["null_centered_margin"]
    return {
        "n_claims": len(rows),
        "n_images": len(images),
        "draft_parse_rate": sum(row["baseline"] != "invalid" for row in rows) / len(rows),
        "metrics": metrics,
        "candidate_minus_draft_cluster_bootstrap": deltas,
        "null_centered_minus_original_same_K_cluster_bootstrap": centered_vs_original,
        "screening_gate": {
            "passed": centered["precision"]["ci_low"] > 0 and centered["recall"]["ci_low"] > 0,
            "rule": "null-centered fixed-K reranking improves precision and recall with image-bootstrap 95% confidence",
        },
        "identity": "Per-image draft K is conserved: delta(FP)=delta(FN)=-delta(TP).",
        "claim_ceiling": "Grouped SLAKE CE claims proxy an OE ontology; no free-text claim extraction is evaluated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--draft-raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=733)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()]
    draft_records = None
    if args.draft_raw:
        draft_records = [
            json.loads(line) for line in args.draft_raw.read_text().splitlines()
            if line.strip()
        ]
    payload = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": analyze(
            records, draws=args.bootstrap_draws, seed=args.seed,
            draft_records=draft_records,
        ),
        "provenance": {
            "code_sha256": sha256_file(Path(__file__)),
            "raw_sha256": sha256_file(args.raw),
            "draft_raw_sha256": sha256_file(args.draft_raw) if args.draft_raw else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
