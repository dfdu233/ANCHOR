#!/usr/bin/env python3
"""Build an outcome-blind VinDr multi-claim CECD listing substrate.

This is deliberately a *closed-ontology, open-cardinality listing* task.  It
is not free-form OE: every candidate claim is one of the 14 VinBigData local
finding classes, while ``none_of_the_listed_findings`` serializes the empty
set and is never treated as a fifteenth independent clinical claim.

The builder reads only official reader labels and DICOM metadata.  It never
reads model outputs, computes model scores, or authorizes GPU execution.
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
from typing import Any, Iterable, Mapping, Sequence

from corrected_sgta.clinical_claims import normalize_term, reader_state
from corrected_sgta.prepare_vindr_reader_manifest import sha256_file, stable_key
from corrected_sgta.prepare_vindr_reader_manifest_v2 import three_way_split


VERSION = "vindr-cecd-ontology-listing-substrate-v1"
REFERENCE_VERSION = "vindr-cecd-ontology-listing-reference-v1"
SCHEMA_VERSION = "vindr-cecd-ontology-listing-pack-v1"
SPLITS = ("pilot", "dev", "confirmation")
STRATA = (
    "unanimous_no_finding",
    "one_unanimous_target_finding",
    "multiple_unanimous_target_findings",
)
PANEL = ("R8", "R9", "R10")

# The 14-class VinBigData detection ontology.  The official wide VinDr table
# contains 27 abnormal findings/diagnoses plus No finding; these 14 therefore
# define a relative closed world only when the prompt names the ontology.
TARGET_FINDINGS: tuple[tuple[str, str], ...] = (
    ("aortic_enlargement", "Aortic enlargement"),
    ("atelectasis", "Atelectasis"),
    ("calcification", "Calcification"),
    ("cardiomegaly", "Cardiomegaly"),
    ("consolidation", "Consolidation"),
    ("ild", "ILD"),
    ("infiltration", "Infiltration"),
    ("lung_opacity", "Lung Opacity"),
    ("nodule_mass", "Nodule/Mass"),
    ("other_lesion", "Other lesion"),
    ("pleural_effusion", "Pleural effusion"),
    ("pleural_thickening", "Pleural thickening"),
    ("pneumothorax", "Pneumothorax"),
    ("pulmonary_fibrosis", "Pulmonary fibrosis"),
)
NONE_TOKEN = "None of the listed findings"

SCIENCE_RENDER_IDS = (
    "baseline_percentile",
    "native_linear",
    "center_minus_0p05w",
    "center_plus_0p05w",
    "width_x1p25",
)
IDENTITY_RENDER_ID = "identity_lossless_duplicate"
SCIENCE_PROMPT_IDS = (
    "inspect_and_list",
    "which_are_visible",
    "report_all_from_ontology",
)
DUPLICATE_PROMPT_ID = "inspect_and_list_exact_duplicate"


class SubstrateError(RuntimeError):
    """The source cannot support the frozen structural contract."""


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_texts() -> dict[str, str]:
    labels = "; ".join(source for _, source in TARGET_FINDINGS)
    suffix = (
        f"Allowed ontology: [{labels}]. Return only a comma-separated list using "
        f"those exact labels, or exactly '{NONE_TOKEN}' when none of the listed "
        "findings is visible. Do not add explanations."
    )
    return {
        "inspect_and_list": (
            "Inspect this chest X-ray and list every visible abnormality from the "
            f"allowed ontology. {suffix}"
        ),
        "which_are_visible": (
            "Which findings from the allowed ontology are visible on this chest "
            f"X-ray? Report all that apply. {suffix}"
        ),
        "report_all_from_ontology": (
            "Using only the allowed ontology, report all abnormalities visible on "
            f"this chest X-ray. {suffix}"
        ),
    }


def orbit_cells() -> list[dict[str, str]]:
    prompts = prompt_texts()
    cells: list[dict[str, str]] = []
    for render_id in SCIENCE_RENDER_IDS:
        for prompt_id in SCIENCE_PROMPT_IDS:
            cells.append(
                {
                    "cell_id": f"science__{render_id}__{prompt_id}",
                    "render_id": render_id,
                    "prompt_id": prompt_id,
                    "prompt_text": prompts[prompt_id],
                    "role": "science_factorial",
                }
            )
    for prompt_id in SCIENCE_PROMPT_IDS:
        cells.append(
            {
                "cell_id": f"control_identity_image__{prompt_id}",
                "render_id": IDENTITY_RENDER_ID,
                "prompt_id": prompt_id,
                "prompt_text": prompts[prompt_id],
                "role": "identity_image_control",
            }
        )
    cells.append(
        {
            "cell_id": f"control_duplicate_prompt__{DUPLICATE_PROMPT_ID}",
            "render_id": SCIENCE_RENDER_IDS[0],
            "prompt_id": DUPLICATE_PROMPT_ID,
            "prompt_text": prompts[SCIENCE_PROMPT_IDS[0]],
            "role": "exact_duplicate_prompt_control",
        }
    )
    if len(cells) != 19 or len({row["cell_id"] for row in cells}) != 19:
        raise AssertionError("5x3 science orbit plus controls must contain 19 cells")
    return cells


def _read_binary(value: Any, *, where: str) -> int:
    text = str(value).strip()
    if text not in {"0", "1", "0.0", "1.0"}:
        raise SubstrateError(f"{where}: expected binary label, got {value!r}")
    return int(float(text))


def read_reader_vectors(
    labels_csv: Path,
) -> tuple[dict[str, dict[str, dict[str, int]]], list[str]]:
    """Read the complete official image x reader x label tensor."""

    with labels_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SubstrateError("wide VinDr label CSV has no header")
        required = {"image_id", "rad_id", "No finding"} | {
            source for _, source in TARGET_FINDINGS
        }
        missing = required - set(reader.fieldnames)
        if missing:
            raise SubstrateError(f"wide VinDr label CSV misses {sorted(missing)}")
        label_names = [
            name for name in reader.fieldnames if name not in {"image_id", "rad_id"}
        ]
        if len(label_names) != 28:
            raise SubstrateError(
                f"official wide label tensor must contain 28 labels, found {len(label_names)}"
            )
        vectors: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
        for line_number, row in enumerate(reader, 2):
            image_id = str(row["image_id"]).strip()
            rad_id = str(row["rad_id"]).strip()
            if not image_id or not rad_id:
                raise SubstrateError(f"line {line_number}: empty image or reader ID")
            if rad_id in vectors[image_id]:
                raise SubstrateError(
                    f"line {line_number}: duplicate reader {rad_id} for {image_id}"
                )
            vector = {
                name: _read_binary(row[name], where=f"line {line_number}/{name}")
                for name in label_names
            }
            abnormal_count = sum(
                value for name, value in vector.items() if name != "No finding"
            )
            if vector["No finding"] and abnormal_count:
                raise SubstrateError(
                    f"line {line_number}: No finding co-occurs with an abnormal label"
                )
            if not vector["No finding"] and abnormal_count == 0:
                raise SubstrateError(
                    f"line {line_number}: neither No finding nor an abnormal label is set"
                )
            vectors[image_id][rad_id] = vector
    reader_counts = Counter(len(rows) for rows in vectors.values())
    if reader_counts != {3: len(vectors)}:
        raise SubstrateError(
            f"every training image must have exactly three readers, got {dict(reader_counts)}"
        )
    return dict(vectors), label_names


def reference_relevance(positive_votes: int) -> str:
    if positive_votes == 3:
        return "required"
    if positive_votes == 0:
        return "out_of_scope"
    if positive_votes in {1, 2}:
        return "optional"
    raise ValueError(f"invalid 3-reader vote count {positive_votes}")


def build_image_reference(
    image_id: str,
    readers: Mapping[str, Mapping[str, int]],
    label_names: Sequence[str],
) -> dict[str, Any]:
    if set(readers) != set(PANEL):
        raise ValueError("build_image_reference requires the exact fixed panel")
    target_sources = {source for _, source in TARGET_FINDINGS}
    claims: list[dict[str, Any]] = []
    for finding_id, source_name in TARGET_FINDINGS:
        votes = [int(readers[rad_id][source_name]) for rad_id in PANEL]
        positive = sum(votes)
        claims.append(
            {
                "finding_id": finding_id,
                "source_name": source_name,
                "reader_votes": [
                    {"rad_id": rad_id, "vote": vote}
                    for rad_id, vote in zip(PANEL, votes)
                ],
                "positive_votes": positive,
                "reader_count": 3,
                "reader_support": positive / 3.0,
                "reader_state": reader_state(positive, 3),
                "listing_relevance": reference_relevance(positive),
            }
        )
    outside_names = [
        name
        for name in label_names
        if name not in target_sources and name != "No finding"
    ]
    outside = []
    for source_name in outside_names:
        votes = [int(readers[rad_id][source_name]) for rad_id in PANEL]
        if any(votes):
            outside.append(
                {
                    "source_name": source_name,
                    "positive_votes": sum(votes),
                    "reader_support": sum(votes) / 3.0,
                }
            )
    no_finding_votes = sum(int(readers[rad_id]["No finding"]) for rad_id in PANEL)
    required = [row["finding_id"] for row in claims if row["positive_votes"] == 3]
    optional = [row["finding_id"] for row in claims if row["positive_votes"] in {1, 2}]
    if no_finding_votes == 3:
        stratum = "unanimous_no_finding"
        if required or optional or outside:
            raise AssertionError("unanimous No finding must have an empty abnormal universe")
    elif len(required) == 1:
        stratum = "one_unanimous_target_finding"
    elif len(required) >= 2:
        stratum = "multiple_unanimous_target_findings"
    else:
        # Images with no unanimous target finding but reader disagreement or
        # unanimous findings outside the 14-label ontology are valid source
        # cases, but they are not part of the three frozen primary strata.
        stratum = "zero_unanimous_target_but_not_unanimous_normal"
    return {
        "reference_version": REFERENCE_VERSION,
        "dataset": "vindr-cxr-1.0.0",
        "image_id": image_id,
        "reader_panel": list(PANEL),
        "panel_policy": "exact_fixed_panel",
        "claims": claims,
        "required_finding_ids": required,
        "optional_finding_ids": optional,
        "refuted_finding_ids": [
            row["finding_id"] for row in claims if row["positive_votes"] == 0
        ],
        "outside_target_ontology_reader_positive": outside,
        "no_finding_positive_votes": no_finding_votes,
        "sampling_stratum": stratum,
        "target_ontology_closed_world_only_under_explicit_prompt": True,
    }


def select_records(
    records: Iterable[dict[str, Any]],
    quotas: Mapping[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        split = three_way_split(str(row["image_id"]), seed)
        stratum = str(row["sampling_stratum"])
        if stratum in STRATA:
            grouped[(split, stratum)].append(row)
    selected: list[dict[str, Any]] = []
    availability: dict[str, dict[str, int]] = defaultdict(dict)
    for split in SPLITS:
        for stratum in STRATA:
            candidates = sorted(
                grouped[(split, stratum)],
                key=lambda row: stable_key(
                    seed,
                    "vindr-cecd-ontology-listing",
                    split,
                    stratum,
                    str(row["image_id"]),
                ),
            )
            availability[split][stratum] = len(candidates)
            needed = int(quotas[split])
            if len(candidates) < needed:
                raise SubstrateError(
                    f"insufficient {split}/{stratum}: need {needed}, found {len(candidates)}"
                )
            for source in candidates[:needed]:
                row = dict(source)
                row["experiment_split"] = split
                row["split_assignment"] = "global_image_sha256_20_20_60_before_sampling"
                row["sampling_probability"] = needed / len(candidates)
                row["inverse_sampling_weight"] = len(candidates) / needed
                selected.append(row)
    selected.sort(
        key=lambda row: (
            SPLITS.index(str(row["experiment_split"])),
            STRATA.index(str(row["sampling_stratum"])),
            str(row["image_id"]),
        )
    )
    if len({str(row["image_id"]) for row in selected}) != len(selected):
        raise AssertionError("an image was selected into more than one stratum or split")
    return selected, {key: dict(value) for key, value in availability.items()}


def audit_bbox_ontology(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "class_name" not in reader.fieldnames:
            raise SubstrateError("bbox CSV lacks class_name")
        classes = sorted({str(row["class_name"]).strip() for row in reader})
    target = {source for _, source in TARGET_FINDINGS}
    missing = target - set(classes)
    if missing:
        raise SubstrateError(f"bbox source misses target classes {sorted(missing)}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_classes": classes,
        "all_14_target_classes_present": True,
    }


def audit_dicom_identity(
    selected: Sequence[dict[str, Any]], image_root: Path
) -> dict[str, Any]:
    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - environment contract
        raise SubstrateError("pydicom is required for the identity audit") from error

    keywords = ("PatientID", "StudyInstanceUID", "AccessionNumber")
    nonempty = Counter()
    sizes: list[int] = []
    for row in selected:
        path = image_root / f"{row['image_id']}.dicom"
        if not path.is_file():
            raise FileNotFoundError(path)
        sizes.append(path.stat().st_size)
        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=list(keywords),
        )
        for keyword in keywords:
            if str(getattr(dataset, keyword, "")).strip():
                nonempty[keyword] += 1
        row["dicom_relpath"] = f"train/{row['image_id']}.dicom"
        row["dicom_size_bytes"] = path.stat().st_size
        row["patient_group_id"] = None
    patient_available = nonempty["PatientID"] == len(selected)
    return {
        "selected_dicoms_checked": len(selected),
        "nonempty_identity_tag_counts": dict(nonempty),
        "patient_identity_available_for_every_selected_image": patient_available,
        "patient_disjoint_split_verifiable": patient_available,
        "image_disjoint_split_verifiable": True,
        "minimum_dicom_size_bytes": min(sizes),
        "maximum_dicom_size_bytes": max(sizes),
        "identity_claim": (
            "patient grouping available"
            if patient_available
            else "patient identity absent; only image-level disjointness is provable"
        ),
    }


def source_census(
    records: Sequence[dict[str, Any]], label_names: Sequence[str], seed: int
) -> dict[str, Any]:
    stratum_counts = Counter(str(row["sampling_stratum"]) for row in records)
    split_strata: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    per_finding: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        for row in records:
            if three_way_split(str(row["image_id"]), seed) == split:
                split_strata[split][str(row["sampling_stratum"])] += 1
    for finding_id, _ in TARGET_FINDINGS:
        rows = [
            claim
            for row in records
            for claim in row["claims"]
            if claim["finding_id"] == finding_id
        ]
        bins = Counter(int(row["positive_votes"]) for row in rows)
        split_unanimous = {
            split: sum(
                1
                for row in records
                if three_way_split(str(row["image_id"]), seed) == split
                and finding_id in row["required_finding_ids"]
            )
            for split in SPLITS
        }
        per_finding[finding_id] = {
            "reader_vote_bins": {f"{vote}/3": bins[vote] for vote in range(4)},
            "unanimous_positive_by_split": split_unanimous,
            "per_finding_confirmation_ready": (
                split_unanimous["dev"] >= 10
                and split_unanimous["confirmation"] >= 20
            ),
        }
    outside_any = sum(
        bool(row["outside_target_ontology_reader_positive"]) for row in records
    )
    outside_unanimous = sum(
        any(item["positive_votes"] == 3 for item in row["outside_target_ontology_reader_positive"])
        for row in records
    )
    return {
        "fixed_panel_images": len(records),
        "source_label_count_including_no_finding": len(label_names),
        "target_abnormality_count": len(TARGET_FINDINGS),
        "stratum_counts": dict(stratum_counts),
        "split_stratum_counts": {
            split: dict(counts) for split, counts in split_strata.items()
        },
        "images_with_at_least_two_unanimous_target_findings": stratum_counts[
            "multiple_unanimous_target_findings"
        ],
        "images_with_any_reader_positive_outside_target_ontology": outside_any,
        "images_with_unanimous_positive_outside_target_ontology": outside_unanimous,
        "per_finding": per_finding,
        "per_finding_confirmation_ready_count": sum(
            bool(value["per_finding_confirmation_ready"])
            for value in per_finding.values()
        ),
    }


def build_pack(
    *,
    labels_csv: Path,
    bbox_csv: Path,
    image_root: Path,
    output_dir: Path,
    quotas: Mapping[str, int],
    seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    vectors, label_names = read_reader_vectors(labels_csv)
    fixed_panel = [
        build_image_reference(image_id, readers, label_names)
        for image_id, readers in sorted(vectors.items())
        if set(readers) == set(PANEL)
    ]
    if len(fixed_panel) != 5501:
        raise SubstrateError(
            f"expected 5501 exact R8/R9/R10 images, found {len(fixed_panel)}"
        )
    selected, availability = select_records(fixed_panel, quotas, seed)
    identity_audit = audit_dicom_identity(selected, image_root)
    bbox_audit = audit_bbox_ontology(bbox_csv)
    census = source_census(fixed_panel, label_names, seed)
    prompts = prompt_texts()
    cells = orbit_cells()

    output_dir.mkdir(parents=True)
    references_path = output_dir / "reference_images.jsonl"
    atomic_text(
        references_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
    )
    prompt_hashes = {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in prompts.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "outcome_blind": True,
        "model_outputs_read": False,
        "model_scores_read": False,
        "gpu_used": False,
        "source": {
            "dataset": "vindr-cxr-1.0.0",
            "labels_csv": str(labels_csv.resolve()),
            "labels_csv_sha256": sha256_file(labels_csv),
            "bbox": bbox_audit,
            "image_root": str(image_root.resolve()),
            "reader_panel": list(PANEL),
            "panel_policy": "exact_fixed_panel",
        },
        "task_contract": {
            "task_id": "vindr_14_closed_ontology_open_cardinality_listing_v1",
            "formal_task_type": "ontology_constrained_open_cardinality_listing",
            "free_form_oe": False,
            "native_report_generation": False,
            "candidate_claim_identity_fixed": True,
            "target_ontology_is_exhaustive": False,
            "target_ontology_closed_world_only_under_explicit_prompt": True,
            "target_finding_ids": [value for value, _ in TARGET_FINDINGS],
            "target_source_names": [value for _, value in TARGET_FINDINGS],
            "empty_set_token": NONE_TOKEN,
            "empty_set_token_is_independent_clinical_claim": False,
            "automatic_parse_policy": "exact_labels_comma_separated_fail_visible",
            "out_of_ontology_policy": "retain_and_count_as_format_violation_never_silently_drop",
            "certainty_semantics": "every listed finding is a definite positive claim",
        },
        "reference_contract": {
            "reference_file": references_path.name,
            "reference_file_sha256": sha256_file(references_path),
            "reference_rows": len(selected),
            "claims_per_image": len(TARGET_FINDINGS),
            "reader_state": "0/3 refuted; 1/3 or 2/3 undetermined; 3/3 supported",
            "listing_relevance": "3/3 required; 1/3 or 2/3 optional; 0/3 out_of_scope",
            "truth_source": "three independent official VinDr reader vectors",
        },
        "split_contract": {
            "assignment": "global image-level SHA256 intervals 20/20/60 before sampling",
            "seed": seed,
            "quotas_per_split_per_stratum": dict(quotas),
            "strata": list(STRATA),
            "availability": availability,
            "selected_images_by_split": dict(
                Counter(str(row["experiment_split"]) for row in selected)
            ),
            "selected_images_by_stratum": dict(
                Counter(str(row["sampling_stratum"]) for row in selected)
            ),
            "image_disjoint": True,
            "patient_disjoint_verifiable": bool(
                identity_audit["patient_disjoint_split_verifiable"]
            ),
            "sampling_weights_required_for_population_metrics": True,
        },
        "orbit_contract": {
            "science_render_ids": list(SCIENCE_RENDER_IDS),
            "science_prompt_ids": list(SCIENCE_PROMPT_IDS),
            "baseline_render_id": SCIENCE_RENDER_IDS[0],
            "baseline_prompt_id": SCIENCE_PROMPT_IDS[0],
            "identity_render_id": IDENTITY_RENDER_ID,
            "duplicate_prompt_id": DUPLICATE_PROMPT_ID,
            "science_cells": 15,
            "total_cells": 19,
            "cells": cells,
            "prompt_text_sha256": prompt_hashes,
            "behavioral_unit": (
                "generate one strict list per cell; deterministically expand each list "
                "to 14 fixed claim-membership indicators"
            ),
            "fresh_generation_identity_guard": (
                "claim identity remains the frozen ontology atom; cell-to-cell set "
                "membership is the OE content-selection outcome"
            ),
            "interaction_estimand": (
                "per-finding centered render-by-prompt mixed derivative of membership, "
                "conditional on complete valid product orbit"
            ),
        },
        "admission_contract": {
            "existing_binary_ce_prompt_admission_transfers": False,
            "existing_render_implementation_may_be_reused": True,
            "new_listing_prompt_admission_required": True,
            "new_multiclaim_render_admission_required": True,
            "minimum_new_admission_images": 60,
            "status": "pending_independent_human_admission",
            "model_orbit_scoring_authorized": False,
            "gpu_authorized": False,
        },
        "fixed_k_contract": {
            "baseline_K": "number of valid in-ontology findings in canonical-cell draft",
            "K_preserved_per_image_model": True,
            "candidate_universe_preserved": True,
            "required_claim_coverage_must_not_decrease": True,
            "out_of_ontology_outputs_may_not_be_erased_to_fake_conservation": True,
            "refusal_and_format_violation_rates_reported": True,
            "mandatory_controls": [
                "canonical membership",
                "render marginal",
                "prompt marginal",
                "full-orbit membership mean",
                "random tie-matched reranking",
                "answer shortening and K checks",
            ],
            "candidate_cecd_score": (
                "canonical-cell additive projection row_mean + column_mean - grand_mean; "
                "this is a causal probe, not standalone method novelty"
            ),
        },
        "scope_guards": {
            "free_oe_hallucination_claim_authorized": False,
            "report_generation_claim_authorized": False,
            "patient_disjoint_generalization_authorized": bool(
                identity_audit["patient_disjoint_split_verifiable"]
            ),
            "aggregate_ontology_listing_claim_possible_after_gates": True,
            "per_finding_claim_limited_to_ready_findings": True,
            "why_not_free_oe": (
                "the 14-label ontology omits 13 official abnormal finding/diagnosis labels "
                "and has no anatomy/attribute truth for arbitrary emitted claims"
            ),
        },
        "source_census": census,
        "dicom_identity_audit": identity_audit,
        "command": " ".join(sys.argv),
    }
    manifest["fingerprint"] = canonical_hash(manifest)
    manifest_path = output_dir / "experiment_manifest.json"
    atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    result = {
        "version": VERSION,
        "status": "conditional_go_closed_ontology_listing_only",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "reference": str(references_path.resolve()),
        "reference_sha256": sha256_file(references_path),
        "selected_images": len(selected),
        "selected_claim_cells": len(selected) * len(TARGET_FINDINGS),
        "fixed_panel_images": len(fixed_panel),
        "true_multiclaim_available": census[
            "images_with_at_least_two_unanimous_target_findings"
        ],
        "free_oe_authorized": False,
        "patient_disjoint_verifiable": bool(
            identity_audit["patient_disjoint_split_verifiable"]
        ),
        "model_or_gpu_authorized": False,
    }
    atomic_text(output_dir / "build_result.json", json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--bbox-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-per-stratum", type=int, default=20)
    parser.add_argument("--dev-per-stratum", type=int, default=40)
    parser.add_argument("--confirmation-per-stratum", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    quotas = {
        "pilot": args.pilot_per_stratum,
        "dev": args.dev_per_stratum,
        "confirmation": args.confirmation_per_stratum,
    }
    if any(value <= 0 for value in quotas.values()):
        raise ValueError("all split/stratum quotas must be positive")
    result = build_pack(
        labels_csv=args.labels_csv,
        bbox_csv=args.bbox_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        quotas=quotas,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
