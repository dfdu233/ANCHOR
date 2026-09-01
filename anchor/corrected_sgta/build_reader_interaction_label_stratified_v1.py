#!/usr/bin/env python3
"""Build an explicit label-stratified VinDr reader-interaction substrate.

Unlike the separate hash-blind source-by-polarity main-effect substrate, this
design intentionally uses the fixed-panel VinDr vote bin for stratification.
That use is frozen and disclosed in a separate audit.  Runner manifests contain
no votes, bins, labels, answers, or target fields.

The old hash-blind images are excluded.  Cells are processed from rarest to
most common and images are globally unique, so an image cannot leak across a
finding, cell, or discovery/confirmation split.  Each finding x 0/3--3/3 cell
is capped at 16; any shortfall is retained and quantified rather than filled
from another bin.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from anchor.corrected_sgta.build_reader_grounded_controlled_source_injection_v1 import (
    ARMS,
    CORE_FINDINGS,
    LABELS_CSV,
    MIMIC_CORPUS,
    MODEL_TOKENIZERS,
    PANEL,
    SOURCE_COLUMNS,
    STATE_TAGS,
    VINDR_IMAGE_ROOT,
    build_donor_pool,
    donor_for_arm,
    prompt,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_json,
)
from anchor.corrected_sgta.run_target_blind_canary_v1 import (
    load_target_blind_manifest,
    preflight_inputs,
)


PROTOCOL = "reader-interaction-label-stratified-source-injection-v1"
HASH_BLIND_DIR = Path("corrected_runs/reader_grounded_controlled_source_injection_v1")
HASH_BLIND_FREEZE = HASH_BLIND_DIR / "query_selection_freeze.json"
HASH_BLIND_DONORS = HASH_BLIND_DIR / "donor_provenance_audit.jsonl"
OUT_DIR = Path("corrected_runs/reader_interaction_label_stratified_v1")
MANIFEST = OUT_DIR / "target_blind_manifest.json"
DISCOVERY_MANIFEST = OUT_DIR / "target_blind_discovery.json"
CONFIRMATION_MANIFEST = OUT_DIR / "target_blind_confirmation.json"
SELECTION_AUDIT = OUT_DIR / "label_stratified_selection_audit.jsonl"
DONOR_AUDIT = OUT_DIR / "donor_provenance_audit.jsonl"
RESULT = OUT_DIR / "result.json"
SPLITS = ("discovery", "confirmation")
MAX_PER_FINDING_VOTE_BIN = 16
FRESH_DONOR_EXTRA_EXCLUDE = re.compile(
    r"\b(?:residual|dedicated|evaluat\w*)\b", re.I
)


def read_fixed_panel_votes(path: Path) -> dict[str, dict[str, Any]]:
    per_image: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "rad_id", *SOURCE_COLUMNS.values()}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("VinDr CSV lacks required fixed-panel vote fields")
        for row in reader:
            image, rad = row["image_id"].strip(), row["rad_id"].strip()
            if rad in per_image[image]:
                raise ValueError(f"duplicate reader {rad} for {image}")
            for finding, column in SOURCE_COLUMNS.items():
                raw = row[column].strip()
                if raw not in {"0", "1", "0.0", "1.0"}:
                    raise ValueError(f"non-binary {finding} vote for {image}/{rad}")
                per_image[image][rad][finding] = int(float(raw))
    output = {}
    for image, readers in per_image.items():
        if set(readers) != set(PANEL):
            continue
        output[image] = {
            finding: {
                "positive_votes": sum(readers[rad][finding] for rad in PANEL),
                "reader_votes": [
                    {"rad_id": rad, "vote": readers[rad][finding]} for rad in PANEL
                ],
            }
            for finding in CORE_FINDINGS
        }
    return output


def select_cells(
    records: dict[str, dict[str, Any]], excluded_images: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    for image, findings in records.items():
        if image in excluded_images:
            continue
        for finding in CORE_FINDINGS:
            grouped[(finding, int(findings[finding]["positive_votes"]))].append(image)

    raw_available = {cell: len(images) for cell, images in grouped.items()}
    # Rarest cells reserve images first.  Ties are stable by finding and bin.
    cells = sorted(
        [(finding, vote) for finding in CORE_FINDINGS for vote in range(4)],
        key=lambda cell: (raw_available.get(cell, 0), cell[0], cell[1]),
    )
    used: set[str] = set()
    selected: list[dict[str, Any]] = []
    cell_audit = {}
    for finding, vote in cells:
        candidates = sorted(
            grouped.get((finding, vote), []),
            key=lambda image: stable_hash(
                f"{PROTOCOL}:cell:{finding}:{vote}:{image}"
            ),
        )
        available_after_global_uniqueness = [image for image in candidates if image not in used]
        chosen = available_after_global_uniqueness[:MAX_PER_FINDING_VOTE_BIN]
        # Alternate split assignment so a short cell differs by at most one.
        split_counts = Counter()
        for index, image in enumerate(chosen):
            split = SPLITS[index % len(SPLITS)]
            value = records[image][finding]
            selected.append({
                "finding": finding,
                "image_id": image,
                "experiment_split": split,
                "positive_votes": vote,
                "reader_count": 3,
                "vote_bin": f"{vote}/3",
                "reader_votes": value["reader_votes"],
                "selection_uses_votes_for_stratification": True,
            })
            split_counts[split] += 1
            used.add(image)
        cell_audit[f"{finding}:{vote}/3"] = {
            "available_after_hash_blind_exclusion_before_global_uniqueness": len(candidates),
            "available_when_cell_processed_after_global_uniqueness": len(
                available_after_global_uniqueness
            ),
            "requested_cap": MAX_PER_FINDING_VOTE_BIN,
            "selected": len(chosen),
            "shortfall": MAX_PER_FINDING_VOTE_BIN - len(chosen),
            "selected_by_split": dict(sorted(split_counts.items())),
        }
    if len(selected) != len({row["image_id"] for row in selected}):
        raise AssertionError("label-stratified query images are not globally unique")
    return selected, cell_audit


def select_fresh_donors(
    pool: dict[tuple[str, str], list[dict[str, Any]]], excluded_patients: set[str]
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    used_patients = set(excluded_patients)
    used_studies: set[str] = set()
    used_images: set[str] = set()
    shortage = {}
    requirements = [
        (finding, state)
        for finding in CORE_FINDINGS
        for state in ("positive", "negative")
    ] + [("fracture", "positive")]
    for finding, state in requirements:
        candidates = pool.get((finding, state), [])
        chosen = []
        for row in candidates:
            if FRESH_DONOR_EXTRA_EXCLUDE.search(row["sentence"]):
                continue
            if (
                row["patient_id"] in used_patients
                or row["study_id"] in used_studies
                or row["image_id"] in used_images
            ):
                continue
            chosen.append(row)
            used_patients.add(row["patient_id"])
            used_studies.add(row["study_id"])
            used_images.add(row["image_id"])
            if len(chosen) == len(SPLITS):
                break
        if len(chosen) != len(SPLITS):
            shortage[f"{finding}:{state}"] = {
                "required": len(SPLITS), "selected": len(chosen)
            }
        for split, row in zip(SPLITS, chosen):
            selected[(finding, state, split)] = row
    return selected, {
        "selected": len(selected),
        "unique_patients": len({row["patient_id"] for row in selected.values()}),
        "unique_studies": len({row["study_id"] for row in selected.values()}),
        "unique_images": len({row["image_id"] for row in selected.values()}),
        "excluded_hash_blind_donor_patients": len(excluded_patients),
        "patient_overlap_with_hash_blind_donors": len(
            {row["patient_id"] for row in selected.values()} & excluded_patients
        ),
        "shortage": shortage,
    }


def main() -> None:
    hash_blind = json.loads(HASH_BLIND_FREEZE.read_text(encoding="utf-8"))
    old_images = {row["image_id"] for row in hash_blind["selected"]}
    old_donor_rows = read_jsonl(HASH_BLIND_DONORS)
    old_donor_patients = {str(row["patient_id"]) for row in old_donor_rows}

    records = read_fixed_panel_votes(LABELS_CSV)
    selected, cell_audit = select_cells(records, old_images)
    donor_pool = build_donor_pool(
        read_jsonl(MIMIC_CORPUS), set(CORE_FINDINGS) | {"fracture"}
    )
    donors, donor_audit = select_fresh_donors(donor_pool, old_donor_patients)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_AUDIT.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    DONOR_AUDIT.write_text(
        "".join(
            json.dumps({"experiment_split": key[2], **row}, sort_keys=True) + "\n"
            for key, row in sorted(donors.items())
        ),
        encoding="utf-8",
    )

    gate_failures = []
    required_donors = (len(CORE_FINDINGS) * 2 + 1) * len(SPLITS)
    if len(donors) != required_donors or donor_audit["shortage"]:
        gate_failures.append("fresh_mimic_donor_shortage")
    if set(old_images) & {row["image_id"] for row in selected}:
        gate_failures.append("hash_blind_query_image_overlap")
    if donor_audit["patient_overlap_with_hash_blind_donors"]:
        gate_failures.append("hash_blind_donor_patient_overlap")
    if gate_failures:
        # Never leave a stale runnable manifest beside a stopped result from a
        # prior successful build attempt.
        for path in (MANIFEST, DISCOVERY_MANIFEST, CONFIRMATION_MANIFEST):
            if path.is_file():
                path.unlink()
        result = {
            "status": "stopped_pre_manifest_gate_failure",
            "protocol": PROTOCOL,
            "gate_failures": gate_failures,
            "selection_uses_votes_for_stratification": True,
            "cell_audit": cell_audit,
            "donor_audit": donor_audit,
            "manifest_generated": False,
            "gpu_execution": "not_run",
        }
        write_json(RESULT, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    tokenizers = {
        name: AutoTokenizer.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True
        )
        for name, path in MODEL_TOKENIZERS.items()
    }
    manifest_rows = []
    token_gaps: dict[str, list[int]] = defaultdict(list)
    token_values: dict[str, list[int]] = defaultdict(list)
    for source in selected:
        finding, image, split = source["finding"], source["image_id"], source["experiment_split"]
        questions = {arm: prompt(finding, arm) for arm in ARMS}
        for model, tokenizer in tokenizers.items():
            counts = [
                len(tokenizer.encode(questions[arm], add_special_tokens=False))
                for arm in ARMS
            ]
            token_gaps[model].append(max(counts) - min(counts))
            token_values[model].extend(counts)
            if len(set(counts)) != 1:
                raise RuntimeError(f"{finding}/{image}: {model} token mismatch")
        for arm in ARMS:
            donor = donor_for_arm(donors, finding, split, arm)
            row = {
                "arm": arm,
                "controlled_source_injection_not_natural_rag": True,
                "dataset": "vindr-cxr-1.0.0",
                "experiment_split": split,
                "finding": finding,
                "id": f"{finding}:{image}:{arm}",
                "img_name": f"train/{image}.dicom",
                "pair_id": f"{finding}:{image}",
                "prompt_contract": PROTOCOL,
                "qid": f"{finding}:{image}:{arm}",
                "question": questions[arm],
                "question_type": "binary_target_blinded",
                "selection_design": "reader_vote_stratified_audit_separate",
                "task": "controlled_reader_interaction_generation",
            }
            if donor is None:
                row["source_provenance_dataset"] = "canonical_control_no_donor"
            else:
                row.update({
                    "source_provenance_dataset": "mimic_train",
                    "source_provenance_report_sha256": donor["report_sha256"],
                    "source_provenance_patient_hash": stable_hash(donor["patient_id"]),
                })
            manifest_rows.append(row)

    write_json(MANIFEST, manifest_rows)
    write_json(
        DISCOVERY_MANIFEST,
        [row for row in manifest_rows if row["experiment_split"] == "discovery"],
    )
    write_json(
        CONFIRMATION_MANIFEST,
        [row for row in manifest_rows if row["experiment_split"] == "confirmation"],
    )
    loaded = load_target_blind_manifest(MANIFEST, 0)
    discovery_loaded = load_target_blind_manifest(DISCOVERY_MANIFEST, 0)
    confirmation_loaded = load_target_blind_manifest(CONFIRMATION_MANIFEST, 0)
    unique_rows = []
    seen = set()
    for row in loaded:
        if row["img_name"] not in seen:
            unique_rows.append(row)
            seen.add(row["img_name"])
    image_preflight = preflight_inputs(unique_rows, VINDR_IMAGE_ROOT)

    discovery_images = {
        row["img_name"] for row in manifest_rows if row["experiment_split"] == "discovery"
    }
    confirmation_images = {
        row["img_name"] for row in manifest_rows if row["experiment_split"] == "confirmation"
    }
    short_cells = {
        cell: value for cell, value in cell_audit.items() if value["shortfall"] > 0
    }
    result = {
        "status": (
            "completed_with_declared_cell_shortage" if short_cells else "completed_all_cells_full"
        ),
        "protocol": PROTOCOL,
        "selection_contract": {
            "selection_uses_votes_for_stratification": True,
            "selection_use_is_declared_design_not_leakage": True,
            "cell_cap": MAX_PER_FINDING_VOTE_BIN,
            "no_cross_bin_substitution": True,
            "rarest_cell_first_then_cell_hash": True,
            "runner_manifest_contains_votes_bins_or_labels": False,
            "cell_audit": cell_audit,
            "short_cells": short_cells,
        },
        "counts": {
            "selected_query_images": len(selected),
            "globally_unique_query_images": len({row["image_id"] for row in selected}),
            "manifest_rows": len(manifest_rows),
            "rows_by_split": dict(Counter(row["experiment_split"] for row in manifest_rows)),
            "rows_by_arm": dict(Counter(row["arm"] for row in manifest_rows)),
        },
        "independence": {
            "hash_blind_query_images_excluded": len(old_images),
            "query_image_overlap_with_hash_blind": len(
                old_images & {row["image_id"] for row in selected}
            ),
            "discovery_confirmation_query_image_overlap": len(
                discovery_images & confirmation_images
            ),
            "new_mimic_donor_patient_overlap_with_hash_blind": donor_audit[
                "patient_overlap_with_hash_blind_donors"
            ],
            "new_donor_audit": donor_audit,
        },
        "factorial": {
            "primary_2x2": [
                "current_present", "current_absent", "other_present", "other_absent"
            ],
            "controls": [
                "current_uncertain", "other_uncertain", "plain", "random_unrelated_state"
            ],
            "state_tags": STATE_TAGS,
        },
        "token_length_audit": {
            model: {
                "all_eight_arms_exact_within_query": all(gap == 0 for gap in token_gaps[model]),
                "maximum_within_query_gap": max(token_gaps[model]),
                "minimum_prompt_tokens": min(token_values[model]),
                "maximum_prompt_tokens": max(token_values[model]),
                "tokenizer_path": str(MODEL_TOKENIZERS[model]),
            }
            for model in tokenizers
        },
        "runner_validation": {
            "combined_rows": len(loaded),
            "discovery_rows": len(discovery_loaded),
            "confirmation_rows": len(confirmation_loaded),
            "target_blind_loader_passed": True,
            "unique_image_cpu_preflight": image_preflight,
            "target_fields_present": 0,
        },
        "gpu_execution": "not_run",
        "artifacts": {
            "combined_manifest": str(MANIFEST),
            "combined_manifest_sha256": sha256_file(MANIFEST),
            "discovery_manifest": str(DISCOVERY_MANIFEST),
            "discovery_manifest_sha256": sha256_file(DISCOVERY_MANIFEST),
            "confirmation_manifest": str(CONFIRMATION_MANIFEST),
            "confirmation_manifest_sha256": sha256_file(CONFIRMATION_MANIFEST),
            "selection_audit": str(SELECTION_AUDIT),
            "selection_audit_sha256": sha256_file(SELECTION_AUDIT),
            "donor_audit": str(DONOR_AUDIT),
            "donor_audit_sha256": sha256_file(DONOR_AUDIT),
        },
    }
    write_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
