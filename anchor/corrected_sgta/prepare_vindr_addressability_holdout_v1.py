#!/usr/bin/env python3
"""Create a fresh fixed-panel VinDr holdout outside the prior 100/cell manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from corrected_sgta.prepare_vindr_reader_manifest import (
    build_records,
    read_votes,
    sha256_file,
    stable_key,
)


VERSION = "vindr-addressability-fresh-holdout-v2"
PANEL = ("R8", "R9", "R10")
FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.per_cell != 19:
        raise ValueError("v1 freezes exactly 19 claims per finding/vote cell")
    old_rows = [
        json.loads(line)
        for line in args.exclusion_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    excluded_images = {str(row["image_id"]) for row in old_rows}
    votes, source_findings, _image_column, _reader_column = read_votes(args.labels_csv)
    normalized_to_source = {
        str(name).strip().lower().replace("/", "_").replace(" ", "_"): name
        for name in source_findings
    }
    missing = set(FINDINGS) - set(normalized_to_source)
    if missing:
        raise ValueError(f"source CSV lacks frozen findings: {sorted(missing)}")
    selected_source = [normalized_to_source[name] for name in FINDINGS]
    records, _statistics = build_records(votes, selected_source, "https://unused.invalid")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    candidate_counts: Counter[tuple[str, int]] = Counter()
    for source in records:
        row = dict(source)
        if str(row["image_id"]) in excluded_images:
            continue
        if set(row["reader_ids"]) != set(PANEL):
            continue
        key = (str(row["finding"]), int(row["positive_votes"]))
        grouped[key].append(row)
        candidate_counts[key] += 1
    chosen: list[dict[str, Any]] = []
    used_images: set[str] = set()
    cells = sorted(
        ((finding, vote) for finding in FINDINGS for vote in range(4)),
        key=lambda cell: (candidate_counts[cell], cell[0], cell[1]),
    )
    for finding, vote in cells:
            candidates = sorted(
                grouped[(finding, vote)],
                key=lambda row: stable_key(
                    args.seed,
                    "addressability-fresh-holdout-v2-unique-images",
                    finding,
                    str(vote),
                    str(row["image_id"]),
                ),
            )
            candidates = [
                row for row in candidates if str(row["image_id"]) not in used_images
            ]
            if len(candidates) < args.per_cell:
                raise ValueError(
                    f"insufficient unseen unique fixed-panel cases for {finding}:{vote}: "
                    f"{len(candidates)} < {args.per_cell}"
                )
            for row in candidates[: args.per_cell]:
                used_images.add(str(row["image_id"]))
                row.pop("dicom_url", None)
                row.update(
                    {
                        "experiment_split": "confirmation",
                        "addressability_split": "fresh_confirmation_v2",
                        "selection_outcome_blind": True,
                        "one_claim_per_image": True,
                        "exclusion_scope": "all images in prior reader_vote_manifest_v2",
                    }
                )
                chosen.append(row)
    chosen.sort(
        key=lambda row: (
            str(row["finding"]), int(row["positive_votes"]), str(row["image_id"])
        )
    )
    record_keys = [f"{row['finding']}:{row['image_id']}" for row in chosen]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("fresh holdout contains duplicate claim keys")
    if len(used_images) != len(chosen):
        raise ValueError("fresh holdout violates one-claim-per-image contract")
    if set(str(row["image_id"]) for row in chosen) & excluded_images:
        raise ValueError("fresh holdout overlaps exclusion manifest")
    manifest_path = args.output_dir / "reader_vote_holdout.jsonl"
    atomic_text(
        manifest_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chosen),
    )
    cell_counts = Counter(
        f"{row['finding']}:{row['positive_votes']}" for row in chosen
    )
    receipt = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "locked_unopened_holdout",
        "labels_csv": str(args.labels_csv.resolve()),
        "labels_csv_sha256": sha256_file(args.labels_csv),
        "exclusion_manifest": str(args.exclusion_manifest.resolve()),
        "exclusion_manifest_sha256": sha256_file(args.exclusion_manifest),
        "excluded_unique_images": len(excluded_images),
        "reader_panel": list(PANEL),
        "findings": list(FINDINGS),
        "per_cell": args.per_cell,
        "claims": len(chosen),
        "unique_images": len(set(str(row["image_id"]) for row in chosen)),
        "cell_counts": dict(sorted(cell_counts.items())),
        "candidate_counts": {
            f"{finding}:{vote}": candidate_counts[(finding, vote)]
            for finding in FINDINGS
            for vote in range(4)
        },
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "record_keys_sha256": hashlib.sha256("\n".join(record_keys).encode()).hexdigest(),
        "seed": args.seed,
        "selection_rule": "stable SHA256 rank; no model output read",
        "cell_selection_order": [f"{finding}:{vote}" for finding, vote in cells],
        "one_claim_per_image": True,
        "model_output_read": False,
    }
    atomic_text(args.output_dir / "lock_receipt.json", json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=19)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite holdout: {args.output_dir}")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
