#!/usr/bin/env python3
"""Build a fail-closed manifest of real within-patient CXR claim transitions.

The source Medical-Diff-VQA labels are silver labels derived from MIMIC-CXR
reports.  A row is admitted only when the current and reference reports are
both locally available and independently satisfy strict, opposite polarity
rules for the same finding.  This is a small mechanism canary, not clinical
gold and not an open-ended hallucination benchmark.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "mimic-natural-counterfactual-manifest-v1"
FINDINGS = (
    "pleural effusion",
    "pneumothorax",
    "edema",
    "consolidation",
    "pneumonia",
)
ADDITIONAL = re.compile(
    r"main image has (?:an )?additional findings? of (.+?) than the reference image"
)
MISSING = re.compile(
    r"main image is missing the findings? of (.+?) than the reference image"
)
UNCERTAIN = re.compile(
    r"\b(?:possible|possibly|probable|probably|likely|may|might|could|or|cannot exclude|can not exclude|"
    r"concern(?:ing)? for|suspect(?:ed)?|question of|versus|vs\.?|and/or)\b"
)
MENTIONS = {
    "pleural effusion": re.compile(r"(?<!pericardial )\b(?:pleural )?effusions?\b"),
    "pneumothorax": re.compile(r"\bpneumothorax\b"),
    "edema": re.compile(r"\b(?:pulmonary|interstitial) edema\b"),
    "consolidation": re.compile(r"\bconsolidation\b"),
    "pneumonia": re.compile(r"\bpneumonia\b"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sentences(text: str) -> list[str]:
    # MIMIC reports contain hard line wraps inside sentences.  Treating every
    # newline as a sentence boundary silently loses negation scope ("no\n+
    # pleural effusion"), so normalize whitespace before sentence splitting.
    text = re.sub(r"\s+", " ", text.strip())
    return [
        part.strip().lower()
        for part in re.split(r"(?<=[.!?])\s+", text)
        if part.strip()
    ]


def report_state(text: str, finding: str) -> tuple[str, list[str]]:
    """Return definite present/absent only; every other report is ambiguous."""

    positive: list[str] = []
    negative: list[str] = []
    mention = MENTIONS[finding]
    for sentence in sentences(text):
        if not mention.search(sentence):
            continue
        spans = list(mention.finditer(sentence))
        is_negative = False
        for match in spans:
            prefix = sentence[max(0, match.start() - 70) : match.start()]
            qualified_only = re.search(
                r"\bno\s+(?:large|sizable|significant|definite)\b[^.;:]{0,55}$",
                prefix,
            )
            if not qualified_only and re.search(r"\b(?:no|without|absent|negative for|free of)\b[^.;:]{0,65}$", prefix):
                is_negative = True
        if is_negative:
            negative.append(sentence)
        elif not UNCERTAIN.search(sentence):
            positive.append(sentence)
    if positive and not negative:
        return "present", positive
    if negative and not positive:
        return "absent", negative
    return "ambiguous", sorted(set(positive + negative))


def parsed_silver_claims(answer: str) -> dict[str, int]:
    text = answer.lower()
    claims: dict[str, int] = {}
    for direction, pattern in ((1, ADDITIONAL), (-1, MISSING)):
        for phrase in pattern.findall(text):
            for finding in FINDINGS:
                if re.search(rf"(?<![a-z]){re.escape(finding)}(?![a-z])", phrase):
                    if finding in claims and claims[finding] != direction:
                        claims[finding] = 0
                    else:
                        claims[finding] = direction
    return claims


def ids_from_image_path(value: str) -> tuple[str, str, str]:
    match = re.search(r"/(p\d+)/s(\d+)/([^/]+)\.jpg$", "/" + value.lstrip("/"))
    if not match:
        raise ValueError(f"unexpected MIMIC image path: {value}")
    return match.group(1), match.group(2), match.group(3)


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return {row["dicom_id"]: row for row in csv.DictReader(handle)}


def load_local_images(root: Path) -> dict[str, Path]:
    values: dict[str, Path] = {}
    for path in root.rglob("*.jpg"):
        values.setdefault(path.stem, path.resolve())
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--exclude-raw", type=Path, required=True)
    parser.add_argument(
        "--patient-scope",
        choices=("exclude", "only", "all"),
        default="exclude",
        help="use patients outside, inside, or regardless of the previous target cohort",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    local_images = load_local_images(args.image_root)
    metadata = load_metadata(args.metadata)
    report_rows = json.loads(args.reports.read_text())
    reports = {str(row["study_id"]): str(row["report"]) for row in report_rows}
    excluded_patients = {
        str(json.loads(line).get("patient_id"))
        for line in args.exclude_raw.read_text().splitlines()
        if line.strip()
    }
    source = json.loads(args.sft.read_text())
    exclusion = Counter()
    admitted: list[dict[str, Any]] = []
    silver_admitted: list[dict[str, Any]] = []

    for row in source:
        if row.get("dataset_name") != "meddiffvqa":
            continue
        image_paths = row.get("image_path") or []
        if len(image_paths) != 2:
            exclusion["not_exactly_two_images"] += 1
            continue
        try:
            current_id = ids_from_image_path(image_paths[0])
            prior_id = ids_from_image_path(image_paths[1])
        except ValueError:
            exclusion["bad_image_path"] += 1
            continue
        patient, current_study, current_dicom = current_id
        prior_patient, prior_study, prior_dicom = prior_id
        if patient != prior_patient:
            exclusion["different_patient"] += 1
            continue
        if args.patient_scope == "exclude" and patient in excluded_patients:
            exclusion["patient_outside_requested_scope"] += 1
            continue
        if args.patient_scope == "only" and patient not in excluded_patients:
            exclusion["patient_outside_requested_scope"] += 1
            continue
        if current_dicom not in local_images or prior_dicom not in local_images:
            exclusion["image_not_local"] += 1
            continue
        if current_dicom not in metadata or prior_dicom not in metadata:
            exclusion["metadata_missing"] += 1
            continue
        current_meta, prior_meta = metadata[current_dicom], metadata[prior_dicom]
        current_view = current_meta.get("ViewPosition", "")
        prior_view = prior_meta.get("ViewPosition", "")
        if current_view not in {"AP", "PA"} or prior_view != current_view:
            exclusion["view_not_matched_frontal"] += 1
            continue
        current_time = (current_meta.get("StudyDate", ""), current_meta.get("StudyTime", ""))
        prior_time = (prior_meta.get("StudyDate", ""), prior_meta.get("StudyTime", ""))
        if not all(current_time + prior_time) or current_time <= prior_time:
            exclusion["not_forward_chronology"] += 1
            continue

        silver = parsed_silver_claims(str(row.get("answer", "")))
        if not silver:
            exclusion["no_target_silver_claim"] += 1
            continue
        for finding, direction in silver.items():
            if direction == 0:
                exclusion["contradictory_silver_claim"] += 1
                continue
            base_row = {
                "record_key": f"{patient}:{prior_study}:{current_study}:{finding}",
                "patient_id": patient,
                "finding": finding,
                "direction": direction,
                "direction_name": "new" if direction == 1 else "resolved",
                "current_image": str(local_images[current_dicom]),
                "prior_image": str(local_images[prior_dicom]),
                "current_study": current_study,
                "prior_study": prior_study,
                "view_position": current_view,
                "silver_item_id": row.get("item_id"),
                "silver_answer": row.get("answer"),
            }
            silver_admitted.append(
                {
                    **base_row,
                    "label_boundary": (
                        "report-derived Medical-Diff-VQA silver transition; "
                        "not independently adjudicated"
                    ),
                }
            )
            if current_study not in reports or prior_study not in reports:
                exclusion["report_not_local"] += 1
                continue
            current_state, current_evidence = report_state(reports[current_study], finding)
            prior_state, prior_evidence = report_state(reports[prior_study], finding)
            expected = ("present", "absent") if direction == 1 else ("absent", "present")
            if (current_state, prior_state) != expected:
                exclusion["report_polarity_not_strictly_confirmed"] += 1
                continue
            admitted.append(
                {
                    **base_row,
                    "current_state": current_state,
                    "prior_state": prior_state,
                    "current_report_evidence": current_evidence,
                    "prior_report_evidence": prior_evidence,
                    "silver_item_id": row.get("item_id"),
                    "silver_answer": row.get("answer"),
                    "label_boundary": "radiologist-report-explicit silver; not independently adjudicated",
                }
            )

    # Fail closed on duplicate pair/finding labels.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in admitted:
        grouped.setdefault(row["record_key"], []).append(row)
    final = []
    for rows in grouped.values():
        directions = {row["direction"] for row in rows}
        if len(directions) != 1:
            exclusion["duplicate_direction_conflict"] += len(rows)
            continue
        final.append(sorted(rows, key=lambda row: str(row["silver_item_id"]))[0])
        exclusion["duplicate_same_direction_removed"] += len(rows) - 1
    final.sort(key=lambda row: row["record_key"])

    silver_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in silver_admitted:
        silver_grouped.setdefault(row["record_key"], []).append(row)
    silver_final = []
    for rows in silver_grouped.values():
        directions = {row["direction"] for row in rows}
        if len(directions) == 1:
            silver_final.append(sorted(rows, key=lambda row: str(row["silver_item_id"]))[0])
        else:
            exclusion["silver_duplicate_direction_conflict"] += len(rows)
    silver_final.sort(key=lambda row: row["record_key"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final))
    silver_path = args.output_dir / "silver_manifest.jsonl"
    silver_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in silver_final)
    )
    audit = {
        "version": VERSION,
        "status": "ready" if final else "no_admissible_rows",
        "n": len(final),
        "n_patients": len({row["patient_id"] for row in final}),
        "silver_n": len(silver_final),
        "silver_n_patients": len({row["patient_id"] for row in silver_final}),
        "silver_by_finding_direction": dict(
            sorted(
                Counter(
                    f"{row['finding']}|{row['direction_name']}" for row in silver_final
                ).items()
            )
        ),
        "by_finding_direction": dict(
            sorted(Counter(f"{row['finding']}|{row['direction_name']}" for row in final).items())
        ),
        "exclusions": dict(sorted(exclusion.items())),
        "rules": {
            "patient_scope": args.patient_scope,
            "acquisition": "same AP/PA view and strictly forward acquisition time",
            "label": "Medical-Diff-VQA direction plus opposite definite states in both source reports",
            "uncertainty": "uncertain or internally contradictory report mentions are excluded",
            "boundary": "silver mechanism canary only; no doctor review and no OE hallucination truth",
        },
        "inputs": {
            "sft": str(args.sft.resolve()),
            "sft_sha256": sha256(args.sft),
            "reports": str(args.reports.resolve()),
            "reports_sha256": sha256(args.reports),
            "metadata": str(args.metadata.resolve()),
            "metadata_sha256": sha256(args.metadata),
            "exclude_raw": str(args.exclude_raw.resolve()),
            "exclude_raw_sha256": sha256(args.exclude_raw),
        },
        "manifest_sha256": sha256(manifest_path),
        "silver_manifest_sha256": sha256(silver_path),
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
