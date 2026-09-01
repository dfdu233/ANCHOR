#!/usr/bin/env python3
"""Build patient-disjoint, finding-matched three-state MIMIC screening triplets.

The labels are extracted from a single radiology report, so this is explicitly
screening evidence for linguistic commitment—not a substitute for VinDr
multi-reader visual support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from corrected_sgta.run_huatuo_vindr_commitment_probe import sha256_file


VERSION = "mimic-report-uncertainty-triplets-v1"
STATE_FROM_CLAIM = {
    ("present", "definite"): "supported",
    ("absent", "definite"): "refuted",
    ("present", "uncertain"): "undetermined",
}
ANSWER_FROM_STATE = {"supported": "yes", "refuted": "no", "undetermined": "maybe"}


def stable_hash(*values: object) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()


def patient_split(subject_id: object, seed: int) -> str:
    value = int(stable_hash(seed, subject_id)[:8], 16) / 0xFFFFFFFF
    return "dev" if value < 0.5 else "holdout"


def claim_state(claim: dict[str, Any]) -> str | None:
    if claim.get("provenance") != "image_grounded":
        return None
    return STATE_FROM_CLAIM.get((claim.get("polarity"), claim.get("uncertainty")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--cap-per-finding-state", type=int, default=60)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    reports = json.loads(args.claims.read_text())["reports"]
    source = {str(row["id"]): row for row in json.loads(args.dataset.read_text())}
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    excluded = Counter()
    for report in reports:
        sample_id = str(report["id"])
        row = source.get(sample_id)
        if row is None:
            excluded["missing_source_row"] += 1
            continue
        image_values = row.get("image_path") or []
        image_name = str(image_values[0]) if image_values else ""
        if not image_name or not (args.image_root / image_name).is_file():
            excluded["missing_image"] += 1
            continue
        by_finding: dict[str, set[str]] = defaultdict(set)
        for claim in report.get("claims", []):
            state = claim_state(claim)
            if state:
                by_finding[str(claim["finding"])].add(state)
        split = patient_split(row["subject_id"], args.seed)
        for finding, states in by_finding.items():
            if len(states) != 1:
                excluded["conflicting_same_report_finding"] += 1
                continue
            state = next(iter(states))
            buckets[(split, finding, state)].append({
                "sample_id": sample_id,
                "subject_id": str(row["subject_id"]),
                "study_id": str(row["study_id"]),
                "image": image_name,
                "finding": finding,
                "state": state,
                "split": split,
            })
    selected: list[dict[str, Any]] = []
    balance = []
    for split in ("dev", "holdout"):
        findings = sorted({key[1] for key in buckets if key[0] == split})
        for finding in findings:
            state_rows = {state: buckets[(split, finding, state)] for state in ANSWER_FROM_STATE}
            n = min((len(rows) for rows in state_rows.values()), default=0)
            n = min(n, args.cap_per_finding_state)
            if n == 0:
                continue
            balance.append({
                "split": split, "finding": finding, "per_state": n,
                "available": {state: len(rows) for state, rows in state_rows.items()},
            })
            for state, rows in state_rows.items():
                ordered = sorted(
                    rows,
                    key=lambda item: stable_hash(
                        args.seed, split, finding, state, item["subject_id"], item["sample_id"]
                    ),
                )
                selected.extend(ordered[:n])
    selected.sort(key=lambda row: stable_hash(args.seed, row["split"], row["sample_id"], row["finding"]))
    questions = []
    manifest_rows = []
    for qid, row in enumerate(selected):
        question = f"Does this chest X-ray show {row['finding'].replace('_', ' ')}?"
        questions.append({
            "qid": qid,
            "img_name": row["image"],
            "question": question,
            "answer": ANSWER_FROM_STATE[row["state"]],
        })
        manifest_rows.append({"qid": qid, "question": question, **row})
    if not questions:
        raise ValueError("no complete finding-matched three-state blocks")
    qpath = args.output_dir / "questions.json"
    qpath.write_text(json.dumps(questions, indent=2) + "\n")
    payload = {
        "config": {
            "version": VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "claims": str(args.claims.resolve()),
            "claims_sha256": sha256_file(args.claims),
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": sha256_file(args.dataset),
            "image_root": str(args.image_root.resolve()),
            "seed": args.seed,
            "split_unit": "MIMIC subject_id",
            "truth_ceiling": (
                "Single-report RadGraph uncertainty; screening-only linguistic "
                "commitment target, not multi-reader visual support."
            ),
        },
        "summary": {
            "n": len(manifest_rows),
            "patients": len({row["subject_id"] for row in manifest_rows}),
            "images": len({row["sample_id"] for row in manifest_rows}),
            "state_counts": Counter(row["state"] for row in manifest_rows),
            "split_counts": Counter(row["split"] for row in manifest_rows),
            "balance": balance,
            "excluded": excluded,
            "questions_sha256": sha256_file(qpath),
        },
        "rows": manifest_rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, default=dict) + "\n"
    )
    print(json.dumps(payload["summary"], indent=2, default=dict))


if __name__ == "__main__":
    main()
