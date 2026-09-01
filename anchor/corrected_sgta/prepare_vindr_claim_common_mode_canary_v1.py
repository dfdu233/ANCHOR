#!/usr/bin/env python3
"""Select a balanced, image-unique VinDr claim common-mode canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


VERSION = "vindr-claim-common-mode-canary-v1"


def stable(seed: int, row: dict) -> str:
    return hashlib.sha256(
        f"{seed}:{row['finding']}:{row['positive_votes']}:{row['image_id']}".encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dicom-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metadata.read_text().splitlines() if line.strip()]
    findings = sorted({str(row["finding"]) for row in rows})
    selected = []
    used_images = set()
    counts = Counter()
    # Rare cross-finding image overlap makes a stable cell-by-cell greedy
    # allocation sufficient while preserving exact target-cell balance.
    for finding in findings:
        for votes in range(4):
            candidates = sorted(
                [
                    row
                    for row in rows
                    if row["finding"] == finding
                    and int(row["positive_votes"]) == votes
                    and (args.dicom_root / f"{row['image_id']}.dicom").is_file()
                ],
                key=lambda row: stable(args.seed, row),
            )
            accepted = []
            for row in candidates:
                if row["image_id"] in used_images:
                    continue
                accepted.append(
                    {
                        "record_key": row["record_key"],
                        "image_id": row["image_id"],
                        "image_path": str(
                            (args.dicom_root / f"{row['image_id']}.dicom").resolve()
                        ),
                        "finding": row["finding"],
                        "positive_votes": int(row["positive_votes"]),
                        "reader_support": float(row["reader_support"]),
                        "reader_state": row["reader_state"],
                        "experiment_split": row.get("experiment_split"),
                    }
                )
                used_images.add(row["image_id"])
                if len(accepted) == args.per_cell:
                    break
            if len(accepted) != args.per_cell:
                raise RuntimeError(
                    f"{finding}|{votes}: only {len(accepted)} image-unique rows"
                )
            selected.extend(accepted)
            counts[f"{finding}|{votes}"] = len(accepted)
    selected.sort(key=lambda row: (row["finding"], row["positive_votes"], row["image_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    )
    audit = {
        "version": VERSION,
        "n": len(selected),
        "n_unique_images": len(used_images),
        "findings": findings,
        "per_finding_vote_cell": args.per_cell,
        "cell_counts": dict(sorted(counts.items())),
        "selection": "stable hash within finding x 0/1/2/3 reader-vote cells; image unique",
        "metadata": str(args.metadata.resolve()),
        "metadata_sha256": sha256(args.metadata),
        "manifest_sha256": sha256(manifest),
        "seed": args.seed,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
