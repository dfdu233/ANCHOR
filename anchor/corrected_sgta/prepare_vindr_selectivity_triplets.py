#!/usr/bin/env python3
"""Build image-disjoint, metadata-matched VinDr selectivity triplets.

Across all findings, every selected image appears in at most one role.  Each
support bin is used in three balanced
roles: anchor, same-support control, and opposite-support control for its
complementary bin (0/3, 1/2).  This prevents pseudo-replication from recycling
one convenient negative across many anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from corrected_sgta.clinical_claims import VERSION as CLAIM_VERSION
from corrected_sgta.run_huatuo_vindr_commitment_probe import sha256_file


VERSION = "vindr-clinical-selectivity-triplets-v1"


def _stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join((str(seed), *parts)).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def normalized_text(value: object) -> str:
    return "_".join(str(value or "unknown").strip().lower().split()) or "unknown"


def read_dicom_metadata(path: Path) -> dict[str, object]:
    import pydicom

    dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    rows = int(getattr(dataset, "Rows", 0) or 0)
    columns = int(getattr(dataset, "Columns", 0) or 0)
    return {
        "view_position": normalized_text(getattr(dataset, "ViewPosition", "unknown")),
        "manufacturer": normalized_text(getattr(dataset, "Manufacturer", "unknown")),
        "manufacturer_model": normalized_text(
            getattr(dataset, "ManufacturerModelName", "unknown")
        ),
        "rows": rows,
        "columns": columns,
        "aspect_ratio": (columns / rows) if rows > 0 and columns > 0 else None,
    }


def acquisition_stratum(
    metadata: Mapping[str, object], match_manufacturer: bool
) -> tuple[str, ...]:
    values = [str(metadata["view_position"])]
    if match_manufacturer:
        values.extend(
            [str(metadata["manufacturer"]), str(metadata["manufacturer_model"])]
        )
    return tuple(values)


def resolution_key(row: Mapping[str, Any], seed: int) -> tuple[float, int, int, str]:
    metadata = row["dicom_metadata"]
    aspect = metadata.get("aspect_ratio")
    return (
        float(aspect) if aspect is not None else float("inf"),
        int(metadata.get("rows", 0)),
        int(metadata.get("columns", 0)),
        _stable_key(seed, str(row["finding"]), str(row["image_id"])),
    )


def enrich_rows(
    rows: Iterable[Mapping[str, Any]],
    image_root: Path,
    allow_unknown_view: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    metadata_by_image: dict[str, dict[str, object]] = {}
    enriched = []
    rejected = []
    for source in rows:
        row = dict(source)
        image_id = str(row["image_id"])
        path = image_root / str(row["dicom_relpath"]).removeprefix("/")
        if not path.is_file():
            rejected.append({"image_id": image_id, "reason": "missing_dicom"})
            continue
        try:
            if image_id not in metadata_by_image:
                metadata_by_image[image_id] = read_dicom_metadata(path)
            metadata = metadata_by_image[image_id]
        except Exception as error:
            rejected.append(
                {"image_id": image_id, "reason": f"dicom_metadata_error:{error!r}"}
            )
            continue
        if metadata["view_position"] == "unknown" and not allow_unknown_view:
            rejected.append({"image_id": image_id, "reason": "unknown_view_position"})
            continue
        row["dicom_metadata"] = metadata
        enriched.append(row)
    return enriched, rejected


def build_triplets(
    rows: Iterable[Mapping[str, Any]],
    seed: int,
    match_manufacturer: bool,
    max_triplets_per_bin: int | None,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    grouped: dict[
        tuple[str, str, tuple[str, ...]], dict[int, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for source in rows:
        row = dict(source)
        if row.get("reference_source") != "vindr_reader_votes":
            raise ValueError("formal triplets require vindr_reader_votes provenance")
        if row.get("formal_reference") is not True:
            raise ValueError("formal triplets require formal_reference=true")
        votes = int(row["positive_votes"])
        readers = int(row["reader_count"])
        if readers != 3 or votes not in range(4):
            raise ValueError(f"invalid reader votes: {votes}/{readers}")
        split = str(row.get("experiment_split", ""))
        if split not in {"dev", "test"}:
            raise ValueError("every input row must have a frozen dev/test split")
        stratum = acquisition_stratum(row["dicom_metadata"], match_manufacturer)
        grouped[(str(row["finding"]), split, stratum)][votes].append(row)

    output = []
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
        for low, high in ((0, 3), (1, 2)):
            low_rows = sorted(
                (
                    row
                    for row in bins[low]
                    if str(row["image_id"]) not in used_images
                ),
                key=lambda row: resolution_key(row, seed),
            )
            high_rows = sorted(
                (
                    row
                    for row in bins[high]
                    if str(row["image_id"]) not in used_images
                ),
                key=lambda row: resolution_key(row, seed),
            )
            per_direction = min(len(low_rows), len(high_rows)) // 3
            if max_triplets_per_bin is not None:
                per_direction = min(per_direction, max_triplets_per_bin)
            if per_direction == 0:
                continue
            usable = 3 * per_direction
            pools = {
                low: {
                    "anchor": low_rows[:per_direction],
                    "same": low_rows[per_direction : 2 * per_direction],
                    "opposite": low_rows[2 * per_direction : usable],
                },
                high: {
                    "anchor": high_rows[:per_direction],
                    "same": high_rows[per_direction : 2 * per_direction],
                    "opposite": high_rows[2 * per_direction : usable],
                },
            }
            allocation.append(
                {
                    "finding": finding,
                    "experiment_split": split,
                    "stratum": list(stratum),
                    "support_pair": [low, high],
                    "available": {str(low): len(low_rows), str(high): len(high_rows)},
                    "triplets_per_direction": per_direction,
                }
            )
            for anchor_votes, opposite_votes in ((low, high), (high, low)):
                anchors = pools[anchor_votes]["anchor"]
                same = pools[anchor_votes]["same"]
                opposite = pools[opposite_votes]["opposite"]
                for index, (anchor, same_row, opposite_row) in enumerate(
                    zip(anchors, same, opposite)
                ):
                    triplet_digest = _stable_key(
                        seed,
                        finding,
                        split,
                        "|".join(stratum),
                        str(anchor_votes),
                        str(anchor["image_id"]),
                    )[:16]
                    triplet_id = f"{finding}:{split}:{anchor_votes}:{triplet_digest}"
                    for role, member in (
                        ("anchor", anchor),
                        ("same_state_swap", same_row),
                        ("opposite_state_swap", opposite_row),
                    ):
                        record = dict(member)
                        record.update(
                            {
                                "version": VERSION,
                                "triplet_id": triplet_id,
                                "swap_role": role,
                                "anchor_positive_votes": anchor_votes,
                                "matching_stratum": list(stratum),
                                "matching_policy": (
                                    "exact view/source stratum; resolution-ordered; "
                                    "no image reuse across any finding"
                                ),
                            }
                        )
                        output.append(record)
            used_images.update(
                str(row["image_id"])
                for vote in (low, high)
                for pool in pools[vote].values()
                for row in pool
            )

    role_counts = Counter(str(row["swap_role"]) for row in output)
    image_role_counts = Counter(str(row["image_id"]) for row in output)
    reused = [key for key, count in image_role_counts.items() if count != 1]
    if reused:
        raise RuntimeError(f"global image reuse detected: {reused[:5]}")
    if not output or len(output) % 3:
        raise RuntimeError("no complete formal triplets could be built")
    if set(role_counts.values()) != {len(output) // 3}:
        raise RuntimeError(f"unbalanced triplet roles: {dict(role_counts)}")
    summary = {
        "triplets": len(output) // 3,
        "records": len(output),
        "unique_images": len(image_role_counts),
        "role_counts": dict(role_counts),
        "allocation": allocation,
    }
    return sorted(
        output,
        key=lambda row: (str(row["triplet_id"]), str(row["swap_role"])),
    ), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-manufacturer", action="store_true")
    parser.add_argument("--allow-unknown-view", action="store_true")
    parser.add_argument("--max-triplets-per-bin", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_triplets_per_bin is not None and args.max_triplets_per_bin <= 0:
        raise ValueError("max-triplets-per-bin must be positive")

    source = load_jsonl(args.reader_manifest)
    enriched, rejected = enrich_rows(
        source, args.image_root, args.allow_unknown_view
    )
    triplets, summary = build_triplets(
        enriched,
        args.seed,
        args.match_manufacturer,
        args.max_triplets_per_bin,
    )
    manifest_path = args.output_dir / "clinical_selectivity_triplets.jsonl"
    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in triplets
    )
    atomic_text(manifest_path, manifest_text)
    rejection_path = args.output_dir / "metadata_rejections.jsonl"
    atomic_text(
        rejection_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected),
    )
    payload = {
        "version": VERSION,
        "claim_contract_version": CLAIM_VERSION,
        "reader_manifest": str(args.reader_manifest.resolve()),
        "reader_manifest_sha256": sha256_file(args.reader_manifest),
        "image_root": str(args.image_root.resolve()),
        "matching_fields": (
            ["view_position", "manufacturer", "manufacturer_model"]
            if args.match_manufacturer
            else ["view_position"]
        ),
        "resolution_control": "deterministic ordering by aspect ratio, rows, columns",
        "unknown_view_allowed": args.allow_unknown_view,
        "metadata_rejections": len(rejected),
        "seed": args.seed,
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
