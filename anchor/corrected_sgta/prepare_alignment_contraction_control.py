"""Build matched and image-permuted CXR alignment manifests.

Both branches contain exactly the same image and text marginals.  The
permuted branch changes only the image--instruction pairing, providing a
compute- and data-matched control for the causal effect of visual-semantic
alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "alignment-contraction-control-v1"
IMAGE_FIELDS = (
    "source_parquet",
    "parquet_row_index",
    "image_sha256",
    "dhash64",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def select_strict_cxr(
    rows: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    strict = [dict(row) for row in rows if bool(row.get("is_strict_cxr"))]
    strict.sort(key=lambda row: stable_key(seed, str(row["id"])))
    if len(strict) < limit:
        raise ValueError(
            f"requested {limit} strict CXR records, found only {len(strict)}"
        )
    return strict[:limit]


def deranged_group_permutation(
    rows: list[dict[str, Any]], seed: int
) -> list[int]:
    """Find a deterministic permutation with no shared source group."""

    groups = [str(row.get("group_id", row["id"])) for row in rows]
    counts = Counter(groups)
    largest = max(counts.values())
    if largest * 2 > len(rows):
        raise RuntimeError(
            "group derangement is impossible because one group exceeds half "
            "of the selected records"
        )
    ordered = sorted(
        range(len(rows)),
        key=lambda index: (
            groups[index],
            stable_key(seed, str(rows[index]["id"])),
        ),
    )
    for shift in range(largest, len(rows) - largest + 1):
        donors = ordered[shift:] + ordered[:shift]
        permutation = [0] * len(rows)
        for target, donor in zip(ordered, donors, strict=True):
            permutation[target] = donor
        if all(
            groups[index] != groups[permutation[index]]
            for index in range(len(rows))
        ):
            return permutation
    raise RuntimeError("could not construct a group-deranged image permutation")


def image_permuted_rows(
    selected: list[dict[str, Any]],
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[int]]:
    permutation = deranged_group_permutation(selected, seed)
    output: list[dict[str, Any]] = []
    for index, target in enumerate(selected):
        donor = selected[permutation[index]]
        copied = dict(target)
        for field in IMAGE_FIELDS:
            copied[field] = donor[field]
        copied["alignment_control"] = "image_permuted"
        copied["text_record_id"] = str(target["id"])
        copied["image_record_id"] = str(donor["id"])
        copied["image_group_id"] = str(
            donor.get("group_id", donor["id"])
        )
        output.append(copied)
    return output, permutation


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    selected = select_strict_cxr(
        read_jsonl(args.input), args.limit, args.seed
    )
    matched = []
    for row in selected:
        copied = dict(row)
        copied["alignment_control"] = "matched"
        copied["text_record_id"] = str(row["id"])
        copied["image_record_id"] = str(row["id"])
        copied["image_group_id"] = str(row.get("group_id", row["id"]))
        matched.append(copied)
    permuted, permutation = image_permuted_rows(selected, args.seed)

    matched_path = args.output / "matched.jsonl"
    permuted_path = args.output / "image_permuted.jsonl"
    write_jsonl(matched_path, matched)
    write_jsonl(permuted_path, permuted)

    matched_images = sorted(row["image_sha256"] for row in matched)
    permuted_images = sorted(row["image_sha256"] for row in permuted)
    matched_texts = sorted(row["text_record_id"] for row in matched)
    permuted_texts = sorted(row["text_record_id"] for row in permuted)
    summary = {
        "version": VERSION,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "seed": args.seed,
        "limit": args.limit,
        "permutation_sha256": hashlib.sha256(
            json.dumps(permutation).encode()
        ).hexdigest(),
        "matched_path": str(matched_path.resolve()),
        "matched_sha256": sha256(matched_path),
        "permuted_path": str(permuted_path.resolve()),
        "permuted_sha256": sha256(permuted_path),
        "same_image_marginal": matched_images == permuted_images,
        "same_text_marginal": matched_texts == permuted_texts,
        "fixed_image_pairs": sum(
            left["text_record_id"] == right["image_record_id"]
            for left, right in zip(matched, permuted, strict=True)
        ),
        "same_group_pairs": sum(
            str(left.get("group_id", left["id"]))
            == right["image_group_id"]
            for left, right in zip(matched, permuted, strict=True)
        ),
        "claim_ceiling": (
            "controlled image-text alignment intervention; target data absent"
        ),
    }
    if not summary["same_image_marginal"] or not summary["same_text_marginal"]:
        raise AssertionError("matched branches do not share exact marginals")
    if summary["fixed_image_pairs"] or summary["same_group_pairs"]:
        raise AssertionError("permuted branch is not group-deranged")
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
