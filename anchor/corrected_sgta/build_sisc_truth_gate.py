"""Build the outcome-blind Study--Image Supervision Collision truth gate.

This module deliberately does *not* parse a study report into image-level
truth.  A report hash is retained only to audit whether one study target was
duplicated across image rows.  Claim truth is admitted only from an explicit
image-local expert/grounding annotation, and an unannotated sibling is never
converted to ``refuted`` or ``unassessable``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TRUTH_STATES = frozenset({"visible", "refuted", "unassessable"})
MODEL_OUTCOME_KEYS = frozenset(
    {
        "prediction",
        "predictions",
        "generated_text",
        "model_answer",
        "model_output",
        "response",
        "scores",
        "logits",
    }
)
DEFAULT_SALT = "sisc-truth-gate-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _assert_no_outcomes(row: Mapping[str, Any], *, context: str) -> None:
    overlap = MODEL_OUTCOME_KEYS.intersection(row)
    if overlap:
        raise ValueError(f"{context} contains forbidden model-outcome keys: {sorted(overlap)}")


def subject_split(subject_id: str, *, salt: str = DEFAULT_SALT) -> str:
    """Return a deterministic patient-disjoint 70/15/15 split."""

    bucket = int(sha256_bytes(f"{salt}|{subject_id}".encode())[:8], 16) % 10_000
    if bucket < 7_000:
        return "train"
    if bucket < 8_500:
        return "dev"
    return "test"


def load_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load optional official image metadata keyed by DICOM/image id.

    No view is inferred from pixels, dimensions, filenames, or report prose.
    """

    if path is None:
        return {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        result: dict[str, dict[str, Any]] = {}
        for row in reader:
            key = str(row.get("dicom_id") or row.get("image_id") or row.get("id") or "").strip()
            if not key:
                continue
            result[key] = {
                "view_position": str(row.get("ViewPosition") or row.get("view_position") or "unknown").strip()
                or "unknown",
                "rows": _optional_int(row.get("Rows") or row.get("rows")),
                "columns": _optional_int(row.get("Columns") or row.get("columns")),
            }
        return result


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def build_study_view_manifest(
    report_json: Path,
    *,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
    image_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create an outcome-free image/study manifest from MIMIC report rows."""

    source_rows = _read_json(report_json)
    if not isinstance(source_rows, list):
        raise ValueError("MIMIC report input must be a JSON list")
    metadata = metadata or {}
    manifest: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    report_hashes_by_study: dict[str, set[str]] = defaultdict(set)

    for index, source in enumerate(source_rows):
        if not isinstance(source, dict):
            raise ValueError(f"row {index} is not an object")
        _assert_no_outcomes(source, context=f"MIMIC row {index}")
        image_paths = source.get("image_path")
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if not image_paths:
            image_paths = [source.get("image")]
        if len(image_paths) != 1 or not image_paths[0]:
            raise ValueError(f"row {index} must identify exactly one image")
        image_path = str(image_paths[0])
        image_id = str(source.get("id") or Path(image_path).stem)
        if image_id in seen_images:
            raise ValueError(f"duplicate image id: {image_id}")
        seen_images.add(image_id)
        subject_id = str(source["subject_id"])
        study_id = str(source["study_id"])
        report_hash = sha256_bytes(str(source.get("report", "")).encode("utf-8"))
        report_hashes_by_study[study_id].add(report_hash)
        meta = metadata.get(image_id, {})
        resolved = image_root / image_path if image_root is not None else None
        manifest.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "image_present": bool(resolved and resolved.is_file()),
                "subject_id": subject_id,
                "study_id": study_id,
                "source_split": str(source.get("split", "unknown")),
                "gate_split": subject_split(subject_id),
                "view_position": str(meta.get("view_position") or "unknown"),
                "rows": meta.get("rows"),
                "columns": meta.get("columns"),
                "view_metadata_source": "official_metadata" if image_id in metadata else "missing",
                "study_report_sha256": report_hash,
            }
        )

    inconsistent = sorted(study for study, hashes in report_hashes_by_study.items() if len(hashes) != 1)
    if inconsistent:
        raise ValueError(f"same study has non-identical report targets: {inconsistent[:5]}")
    by_study = Counter(row["study_id"] for row in manifest)
    audit = {
        "input_path": str(report_json.resolve()),
        "input_sha256": sha256_file(report_json),
        "rows": len(manifest),
        "unique_images": len(seen_images),
        "unique_studies": len(by_study),
        "unique_subjects": len({row["subject_id"] for row in manifest}),
        "paired_view_studies": sum(count >= 2 for count in by_study.values()),
        "paired_view_rows": sum(count for count in by_study.values() if count >= 2),
        "max_images_per_study": max(by_study.values(), default=0),
        "known_view_rows": sum(row["view_position"] != "unknown" for row in manifest),
        "image_present_rows": sum(row["image_present"] for row in manifest),
        "report_target_identity_within_study": True,
        "report_text_used_as_claim_truth": False,
    }
    return sorted(manifest, key=lambda row: (row["subject_id"], row["study_id"], row["image_id"])), audit


def load_tam2020_visible_truth(
    annotation_json: Path,
    manifest: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load Tam et al. expert boxes as *visible-only* truth.

    These annotations contain only positive boxes.  Missing annotations and
    sibling images therefore create no negative or unassessable records.
    """

    payload = _read_json(annotation_json)
    categories = {int(row["id"]): str(row["name"]) for row in payload.get("categories", [])}
    images = {int(row["id"]): row for row in payload.get("images", [])}
    manifest_by_filename = {Path(str(row["image_path"])).name: row for row in manifest}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_images: set[str] = set()
    overlap_images: set[str] = set()

    for annotation in payload.get("annotations", []):
        image = images.get(int(annotation["image_id"]))
        if image is None:
            raise ValueError("annotation refers to a missing image entry")
        filename = str(image.get("file_name") or Path(str(image.get("path", ""))).name)
        source_images.add(filename)
        match = manifest_by_filename.get(filename)
        if match is None:
            continue
        overlap_images.add(filename)
        finding = categories.get(int(annotation["category_id"]))
        if not finding:
            raise ValueError("annotation refers to a missing category")
        grouped[(str(match["image_id"]), finding)].append(
            {
                "bbox": annotation.get("bbox"),
                "annotator_id": str(annotation.get("creator") or "undisclosed"),
                "annotation_id": str(annotation.get("id")),
            }
        )

    by_image_id = {str(row["image_id"]): row for row in manifest}
    truth: list[dict[str, Any]] = []
    for (image_id, finding), boxes in sorted(grouped.items()):
        image = by_image_id[image_id]
        truth.append(
            {
                "image_id": image_id,
                "study_id": image["study_id"],
                "subject_id": image["subject_id"],
                "gate_split": image["gate_split"],
                "view_position": image["view_position"],
                "finding": finding,
                "truth_state": "visible",
                "truth_source": "tam2020_board_certified_radiologist_bbox",
                "truth_source_sha256": sha256_file(annotation_json),
                "independent_image_local_evidence": True,
                "bounding_boxes": boxes,
                "shared_report_used": False,
            }
        )

    audit = {
        "input_path": str(annotation_json.resolve()),
        "input_sha256": sha256_file(annotation_json),
        "source_images": len(source_images),
        "source_categories": sorted(categories.values()),
        "overlap_images": len(overlap_images),
        "admitted_image_finding_records": len(truth),
        "admitted_states": ["visible"] if truth else [],
        "missing_box_interpreted_as_refuted": False,
        "missing_box_interpreted_as_unassessable": False,
    }
    return truth, audit


def assess_gate(
    manifest: Sequence[Mapping[str, Any]],
    truth: Sequence[Mapping[str, Any]],
    *,
    min_paired_studies: int = 100,
    min_findings: int = 3,
    min_view_exclusive_per_finding: int = 30,
) -> dict[str, Any]:
    for row in truth:
        if row.get("truth_state") not in TRUTH_STATES:
            raise ValueError(f"invalid truth state: {row.get('truth_state')}")
        if not row.get("independent_image_local_evidence"):
            raise ValueError("all admitted truth must be independent and image-local")

    by_study: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_study[str(row["study_id"])].append(row)
    paired_studies = {study for study, rows in by_study.items() if len(rows) >= 2}
    image_to_study = {str(row["image_id"]): str(row["study_id"]) for row in manifest}
    states_by_study_finding: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in truth:
        image_id = str(row["image_id"])
        study_id = image_to_study.get(image_id)
        if study_id is None:
            raise ValueError(f"truth image is absent from manifest: {image_id}")
        states_by_study_finding[(study_id, str(row["finding"]))][str(row["truth_state"])].add(image_id)

    exclusive_by_finding: Counter[str] = Counter()
    exclusive_records: list[dict[str, Any]] = []
    for (study_id, finding), state_images in states_by_study_finding.items():
        if study_id not in paired_studies or not state_images.get("visible"):
            continue
        nonvisible = state_images.get("refuted", set()) | state_images.get("unassessable", set())
        if not nonvisible:
            continue
        exclusive_by_finding[finding] += 1
        exclusive_records.append(
            {
                "study_id": study_id,
                "finding": finding,
                "visible_image_ids": sorted(state_images["visible"]),
                "nonvisible_image_ids": sorted(nonvisible),
            }
        )

    eligible_findings = sorted(
        finding
        for finding, count in exclusive_by_finding.items()
        if count >= min_view_exclusive_per_finding
    )
    subject_splits: dict[str, set[str]] = defaultdict(set)
    study_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        subject_splits[str(row["subject_id"])].add(str(row["gate_split"]))
        study_splits[str(row["study_id"])].add(str(row["gate_split"]))
    split_leakage = {
        "subject_leakage_count": sum(len(value) > 1 for value in subject_splits.values()),
        "study_leakage_count": sum(len(value) > 1 for value in study_splits.values()),
    }
    truth_states = Counter(str(row["truth_state"]) for row in truth)
    gates = {
        "paired_studies_at_least_100": len(paired_studies) >= min_paired_studies,
        "at_least_3_eligible_findings": len(eligible_findings) >= min_findings,
        "each_eligible_finding_at_least_30": len(eligible_findings) >= min_findings,
        "absence_and_unassessable_separable": truth_states["refuted"] > 0
        and truth_states["unassessable"] > 0,
        "patient_and_study_disjoint_split": not any(split_leakage.values()),
    }
    return {
        "protocol": "sisc_outcome_blind_truth_gate_v1",
        "thresholds": {
            "min_paired_studies": min_paired_studies,
            "min_findings": min_findings,
            "min_view_exclusive_or_unassessable_per_finding": min_view_exclusive_per_finding,
        },
        "counts": {
            "manifest_rows": len(manifest),
            "paired_studies": len(paired_studies),
            "truth_records": len(truth),
            "truth_states": dict(sorted(truth_states.items())),
            "view_exclusive_studies_by_finding": dict(sorted(exclusive_by_finding.items())),
            "eligible_findings": eligible_findings,
        },
        "split_audit": split_leakage,
        "gates": gates,
        "decision": "GO" if all(gates.values()) else "NO-GO",
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "view_exclusive_records": exclusive_records,
        "outcomes_opened": False,
        "gpu_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-report-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--tam-annotations-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata_csv)
    manifest, manifest_audit = build_study_view_manifest(
        args.mimic_report_json,
        metadata=metadata,
        image_root=args.image_root,
    )
    truth: list[dict[str, Any]] = []
    source_audits: list[dict[str, Any]] = []
    if args.tam_annotations_json:
        truth, source_audit = load_tam2020_visible_truth(args.tam_annotations_json, manifest)
        source_audits.append(source_audit)
    feasibility = assess_gate(manifest, truth)
    feasibility["manifest_audit"] = manifest_audit
    feasibility["truth_source_audits"] = source_audits
    feasibility["construct_locks"] = {
        "shared_report_may_audit_target_duplication_only": True,
        "shared_report_may_define_claim_view_truth": False,
        "missing_grounding_box_may_define_refuted": False,
        "missing_grounding_box_may_define_unassessable": False,
        "model_outputs_may_be_opened_before_go": False,
    }
    fingerprint_payload = {
        "protocol": feasibility["protocol"],
        "thresholds": feasibility["thresholds"],
        "mimic_input_sha256": manifest_audit["input_sha256"],
        "metadata_sha256": sha256_file(args.metadata_csv) if args.metadata_csv else None,
        "truth_source_sha256": [row["input_sha256"] for row in source_audits],
        "split_salt": DEFAULT_SALT,
    }
    feasibility["provenance"] = {
        "dataset": "local_mmedrag_mimic_test_plus_independent_tam2020_boxes",
        "model": "none_outcome_blind",
        "method": "sisc_truth_gate_v1",
        "seed": "not_applicable_deterministic_sha256_split",
        "command": " ".join(shlex.quote(part) for part in [sys.executable, "-m", __name__, *sys.argv[1:]]),
        "fingerprint": sha256_bytes(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "fingerprint_payload": fingerprint_payload,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "study_view_manifest.jsonl", manifest)
    _write_jsonl(args.output_dir / "claim_view_truth_candidates.jsonl", truth)
    _write_json(args.output_dir / "sisc_feasibility.json", feasibility)
    print(json.dumps({"decision": feasibility["decision"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
