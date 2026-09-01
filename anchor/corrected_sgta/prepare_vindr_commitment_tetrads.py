#!/usr/bin/env python3
"""Build matched VinDr tetrads for reader-support commitment analysis.

Each tetrad holds majority polarity fixed and contains two unanimous images
and two reader-disagreement images.  The within-state pairs estimate nuisance
drift, while their difference estimates the local reader-support response:
0/3 versus 1/3 for negative claims and 3/3 versus 2/3 for positive claims.
No image is reused anywhere in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from corrected_sgta.clinical_claims import VERSION as CLAIM_VERSION
from corrected_sgta.prepare_vindr_selectivity_triplets import (
    acquisition_stratum,
    atomic_text,
    enrich_rows,
    load_jsonl,
    resolution_key,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import sha256_file


VERSION = "vindr-commitment-tetrads-v1"
ROLE_ORDER = (
    "clear_a",
    "clear_b",
    "ambiguous_a",
    "ambiguous_b",
)


def _stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join((str(seed), *parts)).encode()).hexdigest()


def _adjacent_pairs(
    rows: list[dict[str, Any]], seed: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: resolution_key(row, seed))
    return [
        (ordered[index], ordered[index + 1])
        for index in range(0, len(ordered) - 1, 2)
    ]


def build_commitment_tetrads(
    rows: Iterable[Mapping[str, Any]],
    seed: int,
    match_manufacturer: bool,
    max_tetrads_per_branch: int | None,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Create image-disjoint 2-clear/2-ambiguous matched units."""

    grouped: dict[
        tuple[str, str, tuple[str, ...]], dict[int, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for source in rows:
        row = dict(source)
        if row.get("reference_source") != "vindr_reader_votes":
            raise ValueError("formal tetrads require vindr_reader_votes provenance")
        if row.get("formal_reference") is not True:
            raise ValueError("formal tetrads require formal_reference=true")
        votes = int(row["positive_votes"])
        readers = int(row["reader_count"])
        split = str(row.get("experiment_split", ""))
        if readers != 3 or votes not in range(4):
            raise ValueError(f"invalid reader votes: {votes}/{readers}")
        if split not in {"dev", "test"}:
            raise ValueError("every input row must have a frozen dev/test split")
        stratum = acquisition_stratum(row["dicom_metadata"], match_manufacturer)
        grouped[(str(row["finding"]), split, stratum)][votes].append(row)

    output: list[dict[str, Any]] = []
    allocation = []
    used_images: set[str] = set()
    group_keys = sorted(
        grouped,
        key=lambda key: _stable_key(
            seed, "group-order", key[0], key[1], "|".join(key[2])
        ),
    )
    for finding, split, stratum in group_keys:
        bins = grouped[(finding, split, stratum)]
        for branch, clear_votes, ambiguous_votes in (
            ("negative", 0, 1),
            ("positive", 3, 2),
        ):
            clear = [
                row
                for row in bins[clear_votes]
                if str(row["image_id"]) not in used_images
            ]
            ambiguous = [
                row
                for row in bins[ambiguous_votes]
                if str(row["image_id"]) not in used_images
            ]
            clear_pairs = _adjacent_pairs(clear, seed)
            ambiguous_pairs = _adjacent_pairs(ambiguous, seed)
            count = min(len(clear_pairs), len(ambiguous_pairs))
            if max_tetrads_per_branch is not None:
                count = min(count, max_tetrads_per_branch)
            if count == 0:
                continue
            allocation.append(
                {
                    "finding": finding,
                    "experiment_split": split,
                    "stratum": list(stratum),
                    "majority_polarity": branch,
                    "clear_votes": clear_votes,
                    "ambiguous_votes": ambiguous_votes,
                    "available": {
                        "clear": len(clear),
                        "ambiguous": len(ambiguous),
                    },
                    "tetrads": count,
                }
            )
            for index in range(count):
                members = (*clear_pairs[index], *ambiguous_pairs[index])
                digest = _stable_key(
                    seed,
                    finding,
                    split,
                    "|".join(stratum),
                    branch,
                    str(index),
                    *(str(row["image_id"]) for row in members),
                )[:16]
                tetrad_id = f"{finding}:{split}:{branch}:{digest}"
                for role, member in zip(ROLE_ORDER, members):
                    record = dict(member)
                    record.update(
                        {
                            "version": VERSION,
                            "tetrad_id": tetrad_id,
                            "tetrad_role": role,
                            "majority_polarity": branch,
                            "clear_positive_votes": clear_votes,
                            "ambiguous_positive_votes": ambiguous_votes,
                            "matching_stratum": list(stratum),
                            "matching_policy": (
                                "exact view/source stratum; adjacent resolution "
                                "pairs within vote bin; no global image reuse"
                            ),
                        }
                    )
                    output.append(record)
                    used_images.add(str(member["image_id"]))

    role_counts = Counter(str(row["tetrad_role"]) for row in output)
    image_counts = Counter(str(row["image_id"]) for row in output)
    reused = [image_id for image_id, count in image_counts.items() if count != 1]
    if reused:
        raise RuntimeError(f"global image reuse detected: {reused[:5]}")
    if not output or len(output) % 4:
        raise RuntimeError("no complete formal commitment tetrads could be built")
    expected = len(output) // 4
    if role_counts != Counter({role: expected for role in ROLE_ORDER}):
        raise RuntimeError(f"unbalanced tetrad roles: {dict(role_counts)}")
    summary = {
        "tetrads": expected,
        "records": len(output),
        "unique_images": len(image_counts),
        "role_counts": dict(role_counts),
        "branch_counts": dict(
            Counter(str(row["majority_polarity"]) for row in output[::4])
        ),
        "allocation": allocation,
    }
    return sorted(
        output,
        key=lambda row: (str(row["tetrad_id"]), ROLE_ORDER.index(str(row["tetrad_role"]))),
    ), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-manufacturer", action="store_true")
    parser.add_argument("--allow-unknown-view", action="store_true")
    parser.add_argument("--max-tetrads-per-branch", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_tetrads_per_branch is not None and args.max_tetrads_per_branch <= 0:
        raise ValueError("max-tetrads-per-branch must be positive")

    source = load_jsonl(args.reader_manifest)
    enriched, rejected = enrich_rows(source, args.image_root, args.allow_unknown_view)
    tetrads, summary = build_commitment_tetrads(
        enriched, args.seed, args.match_manufacturer, args.max_tetrads_per_branch
    )
    manifest_path = args.output_dir / "commitment_tetrads.jsonl"
    atomic_text(
        manifest_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tetrads),
    )
    atomic_text(
        args.output_dir / "metadata_rejections.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected),
    )
    payload = {
        "version": VERSION,
        "claim_contract_version": CLAIM_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reader_manifest": str(args.reader_manifest.resolve()),
        "reader_manifest_sha256": sha256_file(args.reader_manifest),
        "image_root": str(args.image_root.resolve()),
        "matching_fields": (
            ["view_position", "manufacturer", "manufacturer_model"]
            if args.match_manufacturer
            else ["view_position"]
        ),
        "metadata_rejections": len(rejected),
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
        **summary,
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    atomic_text(
        args.output_dir / "summary.json",
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
