#!/usr/bin/env python3
"""Build the fixed-panel, three-way VinDr mechanism manifest.

The pilot split is expendable, development is the only model-selection split,
and confirmation remains locked.  Split assignment is global at image level
before sampling, so a radiograph can never leak through a second finding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from corrected_sgta.clinical_claims import normalize_term
from corrected_sgta.prepare_vindr_reader_manifest import (
    atomic_text,
    build_oe_listing_records,
    build_records,
    load_ontology_findings,
    read_votes,
    reader_effect_summary,
    select_ontology_columns,
    sha256_file,
    stable_key,
)


VERSION = "vindr-reader-vote-manifest-v5-fixed-panel-three-way"
SPLIT_ORDER = ("pilot", "dev", "confirmation")


def three_way_split(image_id: str, seed: int) -> str:
    """Assign one image globally using frozen 20/20/60 hash intervals."""

    # Preserve the v1 split namespace; the three-way extension changes only
    # the frozen interval boundaries, not the image hash convention.
    digest = stable_key(seed, "experiment-split", str(image_id))
    uniform = int(digest[:16], 16) / float(16**16)
    if uniform < 0.2:
        return "pilot"
    if uniform < 0.4:
        return "dev"
    return "confirmation"


def fixed_panel_records(
    records: Iterable[dict[str, object]], panel: tuple[str, ...]
) -> list[dict[str, object]]:
    required = set(panel)
    if not panel or len(required) != len(panel):
        raise ValueError("reader panel must contain unique reader IDs")
    selected = []
    for source in records:
        readers = {str(value) for value in source["reader_ids"]}
        if readers != required:
            continue
        row = dict(source)
        row["reader_panel"] = list(panel)
        row["panel_policy"] = "exact_fixed_panel"
        selected.append(row)
    if not selected:
        raise ValueError(f"no rows use exact reader panel {panel}")
    return selected


def select_split_balanced(
    records: Iterable[dict[str, object]],
    findings: set[str],
    split_counts: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, dict[str, int]]]]:
    """Sample exact quotas after globally assigning every image to one split."""

    if set(split_counts) != set(SPLIT_ORDER) or any(
        int(value) <= 0 for value in split_counts.values()
    ):
        raise ValueError(f"split_counts must give positive quotas for {SPLIT_ORDER}")
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    available: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for source in records:
        finding = str(source["finding"])
        if finding not in findings:
            continue
        vote = int(source["positive_votes"])
        split = three_way_split(str(source["image_id"]), seed)
        grouped[(finding, vote, split)].append(source)
    output = []
    for finding in sorted(findings):
        for vote in range(4):
            for split in SPLIT_ORDER:
                candidates = sorted(
                    grouped[(finding, vote, split)],
                    key=lambda row: stable_key(
                        seed,
                        "within-split-sample",
                        finding,
                        str(vote),
                        split,
                        str(row["image_id"]),
                    ),
                )
                available[finding][f"{vote}/3"][split] = len(candidates)
                required = int(split_counts[split])
                if len(candidates) < required:
                    raise RuntimeError(
                        f"insufficient {finding} {vote}/3 {split}: "
                        f"need {required}, found {len(candidates)}"
                    )
                for source in candidates[:required]:
                    row = dict(source)
                    row["experiment_split"] = split
                    row["split_assignment"] = "global_image_sha256_20_20_60"
                    output.append(row)
    output.sort(
        key=lambda row: (
            SPLIT_ORDER.index(str(row["experiment_split"])),
            str(row["finding"]),
            int(row["positive_votes"]),
            str(row["image_id"]),
        )
    )
    image_splits: dict[str, set[str]] = defaultdict(set)
    for row in output:
        image_splits[str(row["image_id"])].add(str(row["experiment_split"]))
    leaking = sorted(image for image, splits in image_splits.items() if len(splits) != 1)
    if leaking:
        raise AssertionError(f"images cross splits: {leaking[:5]}")
    return output, {k: dict(v) for k, v in available.items()}


def read_bbox_annotations(
    path: Path,
    selected_keys: set[tuple[str, str]],
    panel: set[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Retain valid boxes from the fixed readers for ROI causal controls."""

    boxes: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image_id", "class_name", "rad_id", "x_min", "y_min", "x_max", "y_max"
        }
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"bbox CSV missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            key = (str(row["image_id"]), normalize_term(str(row["class_name"])))
            if key not in selected_keys or str(row["rad_id"]) not in panel:
                continue
            raw = [str(row[name]).strip() for name in ("x_min", "y_min", "x_max", "y_max")]
            if not all(raw):
                continue
            x_min, y_min, x_max, y_max = map(float, raw)
            if not (x_max > x_min and y_max > y_min):
                continue
            boxes[key].append(
                {
                    "rad_id": str(row["rad_id"]),
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
            )
    rows = [
        {"image_id": image, "finding": finding, "boxes": values}
        for (image, finding), values in sorted(boxes.items())
    ]
    positive_keys = {key for key in selected_keys if key in boxes}
    return rows, {
        "selected_claim_keys": len(selected_keys),
        "claim_keys_with_valid_fixed_panel_bbox": len(positive_keys),
        "valid_boxes": sum(len(row["boxes"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--bbox-csv", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--reader-panel", nargs=3, default=("R8", "R9", "R10"))
    parser.add_argument(
        "--findings",
        nargs="+",
        help="Frozen normalized finding subset; defaults to every ontology finding",
    )
    parser.add_argument("--pilot-per-bin", type=int, default=20)
    parser.add_argument("--dev-per-bin", type=int, default=20)
    parser.add_argument("--confirmation-per-bin", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    panel = tuple(str(value) for value in args.reader_panel)
    split_counts = {
        "pilot": args.pilot_per_bin,
        "dev": args.dev_per_bin,
        "confirmation": args.confirmation_per_bin,
    }
    votes, source_findings, image_col, reader_col = read_votes(args.labels_csv)
    findings, excluded = select_ontology_columns(
        source_findings, load_ontology_findings(args.ontology)
    )
    all_records, _ = build_records(votes, findings, "local-only")
    panel_records = fixed_panel_records(all_records, panel)
    ontology_findings = {normalize_term(value) for value in findings}
    normalized_findings = (
        {normalize_term(value) for value in args.findings}
        if args.findings
        else ontology_findings
    )
    unknown = normalized_findings - ontology_findings
    if unknown:
        raise ValueError(f"requested findings absent from ontology: {sorted(unknown)}")
    selected, availability = select_split_balanced(
        panel_records, normalized_findings, split_counts, args.seed
    )

    missing = sorted(
        {
            str(row["image_id"])
            for row in selected
            if not (args.image_root / f"{row['image_id']}.dicom").is_file()
        }
    )
    if missing:
        raise FileNotFoundError(f"selected DICOMs absent from image root: {missing[:5]}")
    for row in selected:
        row["dicom_relpath"] = f"train/{row['image_id']}.dicom"
        row.pop("dicom_url", None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(
        args.output_dir / "reader_vote_manifest_v2.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
    )
    selected_images = {str(row["image_id"]) for row in selected}
    eligible_source_findings = {
        str(value)
        for value in findings
        if normalize_term(str(value)) in normalized_findings
    }
    oe_rows = build_oe_listing_records(
        panel_records, selected_images, eligible_source_findings, args.seed, 0.2
    )
    for row in oe_rows:
        row["experiment_split"] = three_way_split(str(row["image_id"]), args.seed)
        row["split_assignment"] = "global_image_sha256_20_20_60"
        row["dicom_relpath"] = f"train/{row['image_id']}.dicom"
        row.pop("dicom_url", None)
    atomic_text(
        args.output_dir / "oe_listing_reference_v2.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in oe_rows),
    )

    selected_positive_keys = {
        (str(row["image_id"]), str(row["finding"]))
        for row in selected
        if int(row["positive_votes"]) > 0
    }
    bbox_rows, bbox_audit = read_bbox_annotations(
        args.bbox_csv, selected_positive_keys, set(panel)
    )
    atomic_text(
        args.output_dir / "bbox_annotations_v2.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bbox_rows),
    )

    split_claims = Counter(str(row["experiment_split"]) for row in selected)
    image_split = {
        image: three_way_split(image, args.seed) for image in selected_images
    }
    summary = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(args.labels_csv.resolve()),
        "source_csv_sha256": sha256_file(args.labels_csv),
        "bbox_csv": str(args.bbox_csv.resolve()),
        "bbox_csv_sha256": sha256_file(args.bbox_csv),
        "ontology": str(args.ontology.resolve()),
        "ontology_sha256": sha256_file(args.ontology),
        "image_root": str(args.image_root.resolve()),
        "image_root_policy": "immutable_external_reference_no_copy",
        "source_columns": {"image": image_col, "reader": reader_col},
        "excluded_source_findings": excluded,
        "eligible_findings": sorted(normalized_findings),
        "reader_panel": list(panel),
        "panel_policy": "exact_fixed_panel_primary",
        "seed": args.seed,
        "split_contract": {
            "assignment": "global image-level SHA256 intervals 20/20/60 before sampling",
            "quotas_per_finding_vote_bin": split_counts,
            "claim_rows": dict(split_claims),
            "unique_images": dict(Counter(image_split.values())),
            "image_disjoint": True,
        },
        "availability_before_sampling": availability,
        "selected_claim_rows": len(selected),
        "selected_unique_images": len(selected_images),
        "oe_claim_rows": len(oe_rows),
        "bbox_audit": bbox_audit,
        "reader_effect_readiness": reader_effect_summary(
            row for row in panel_records if str(row["finding"]) in normalized_findings
        ),
        "command": " ".join(sys.argv),
    }
    summary["fingerprint"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode()
    ).hexdigest()
    atomic_text(args.output_dir / "summary_v2.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
