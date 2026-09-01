#!/usr/bin/env python3
"""Fail-closed integrity/blinding verifier for the VinDr listing admission pack."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from corrected_sgta.build_vindr_cecd_listing_admission_pack_v1 import (
    CLINICAL_DECISION_FIELDS,
    PROFESSIONAL_ROLE,
    PROMPT_DECISION_FIELDS,
    ROLES,
    VERSION as PACK_VERSION,
    return_schema,
)
from corrected_sgta.prepare_vindr_cecd_ontology_listing_v1 import (
    IDENTITY_RENDER_ID,
    SCIENCE_RENDER_IDS,
    STRATA,
    canonical_hash,
)
from corrected_sgta.prepare_vindr_reader_manifest import sha256_file


VERSION = "vindr-cecd-listing-admission-pack-integrity-v1"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{path}: CSV header missing")
        return list(reader.fieldnames), list(reader)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(pack_dir: Path) -> dict[str, Any]:
    manifest_path = pack_dir / "manifest.json"
    mapping_path = pack_dir / "sealed_mapping.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    require(manifest.get("version") == PACK_VERSION, "wrong pack version")
    require(mapping.get("version") == PACK_VERSION, "mapping version drift")
    fingerprint = str(manifest.get("fingerprint", ""))
    unsigned = dict(manifest)
    unsigned.pop("fingerprint", None)
    require(fingerprint == canonical_hash(unsigned), "manifest fingerprint drift")
    for field in ("outcome_blind",):
        require(manifest.get(field) is True, f"unsafe manifest flag {field}")
    for field in ("model_outputs_read", "model_scores_read", "gpu_used"):
        require(manifest.get(field) is False, f"unsafe manifest flag {field}")

    source = manifest["source"]
    for path_field, hash_field in (
        ("breadth_manifest", "breadth_manifest_sha256"),
        ("bbox_csv", "bbox_csv_sha256"),
    ):
        path = Path(source[path_field])
        require(path.is_file(), f"source missing: {path_field}")
        require(sha256_file(path) == source[hash_field], f"source hash drift: {path_field}")
    source_manifest = json.loads(Path(source["breadth_manifest"]).read_text())
    require(source_manifest["fingerprint"] == source["breadth_fingerprint"], "breadth fingerprint drift")
    require(
        source_manifest["reference_contract"]["reference_file_sha256"]
        == source["reference_sha256"],
        "breadth reference hash binding drift",
    )
    require(Path(source["image_root"]).is_dir(), "source image root missing")
    for relative, expected in manifest["artifact_sha256"].items():
        path = pack_dir / relative
        require(path.is_file(), f"frozen artifact missing: {relative}")
        require(sha256_file(path) == expected, f"frozen artifact stale: {relative}")

    selected_path = pack_dir / "selected_images.sealed.jsonl"
    selected = _load_jsonl(selected_path)
    require(len(selected) == 60, "selected pilot must contain 60 images")
    require(len({str(row["image_id"]) for row in selected}) == 60, "selected image duplicated")
    require(all(row["experiment_split"] == "pilot" for row in selected), "non-pilot image selected")
    require(
        Counter(str(row["sampling_stratum"]) for row in selected)
        == {stratum: 20 for stratum in STRATA},
        "selected stratum balance drift",
    )
    source_reference = Path(source["breadth_manifest"]).parent / source_manifest[
        "reference_contract"
    ]["reference_file"]
    source_pilot = [
        row for row in _load_jsonl(source_reference) if row["experiment_split"] == "pilot"
    ]
    require(
        {canonical_hash(row) for row in selected} == {canonical_hash(row) for row in source_pilot},
        "selected pilot differs from the entire pre-frozen source pilot",
    )
    require(
        manifest["selection"]["selection_sha256"]
        == canonical_hash([(row["image_id"], row["sampling_stratum"]) for row in selected]),
        "selection hash drift",
    )

    inventory_path = pack_dir / manifest["images"]["inventory"]
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    require(len(inventory) == manifest["images"]["pngs"] == 504, "PNG inventory count drift")
    require(sha256_file(inventory_path) == manifest["images"]["inventory_sha256"], "inventory hash drift")
    expected_pngs = set()
    for item in inventory:
        path = pack_dir / str(item["name"])
        require(path.is_file(), f"PNG missing: {item['name']}")
        require(path.stat().st_size == int(item["size_bytes"]), f"PNG size drift: {item['name']}")
        require(sha256_file(path) == item["sha256"], f"PNG hash drift: {item['name']}")
        expected_pngs.add(path.resolve())
    actual_pngs = {path.resolve() for path in (pack_dir / "images").glob("*.png")}
    require(actual_pngs == expected_pngs, "PNG directory is not inventory-closed")

    pairs = mapping["clinical_pairs"]
    require(len(pairs) == 252, "clinical pair count drift")
    pair_ids = {str(row["pair_id"]) for row in pairs}
    require(len(pair_ids) == 252, "clinical pair IDs duplicated")
    transforms = Counter(str(row["transform"]) for row in pairs)
    require(transforms.pop(IDENTITY_RENDER_ID, 0) == 12, "identity control count drift")
    require(
        transforms == {name: 60 for name in SCIENCE_RENDER_IDS[1:]},
        "science render balance drift",
    )
    guard_failures = sorted(
        str(row["pair_id"])
        for row in pairs
        if not bool(row["transform_guard"].get("clinical_guard_pass"))
    )
    clinical_contract = manifest["clinical_review"]
    require(
        int(clinical_contract["computational_guard_pass_pairs"])
        == len(pairs) - len(guard_failures),
        "computational guard pass count drift",
    )
    require(
        int(clinical_contract["computational_guard_fail_pairs"])
        == len(guard_failures),
        "computational guard failure count drift",
    )
    require(
        clinical_contract["computational_guard_failure_pair_ids_sha256"]
        == canonical_hash(guard_failures),
        "computational guard failure identity drift",
    )
    require(
        clinical_contract["guard_failures_retained_for_blinded_clinical_review"] is True,
        "guard failure review policy drift",
    )
    for row in pairs:
        left = pack_dir / "images" / f"{row['pair_id']}_A.png"
        right = pack_dir / "images" / f"{row['pair_id']}_B.png"
        require(sha256_file(left) == row["left_png_sha256"], "sealed left PNG hash drift")
        require(sha256_file(right) == row["right_png_sha256"], "sealed right PNG hash drift")
        if row["transform"] == IDENTITY_RENDER_ID:
            require(row["left_png_sha256"] == row["right_png_sha256"], "identity pixels differ")

    for reviewer in (1, 2):
        path = pack_dir / f"clinical_reviewer_{reviewer}.csv"
        header, rows = _csv(path)
        require(len(rows) == 252, f"{path.name}: wrong row count")
        require({row["pair_id"] for row in rows} == pair_ids, f"{path.name}: pair IDs drift")
        require(set(CLINICAL_DECISION_FIELDS) <= set(header), f"{path.name}: decisions missing")
        for row in rows:
            require(
                all(not row[field].strip() for field in CLINICAL_DECISION_FIELDS),
                f"{path.name}: review decision prefilled",
            )
            hidden = next(item for item in pairs if item["pair_id"] == row["pair_id"])
            visible = json.dumps(row, sort_keys=True)
            require(hidden["image_id"] not in visible, f"{path.name}: image ID leaked")
            require(hidden["transform"] not in visible, f"{path.name}: transform leaked")

    prompt_mapping = mapping["prompt_pairs"]
    prompt_ids = {str(row["item_id"]) for row in prompt_mapping}
    require(len(prompt_mapping) == len(prompt_ids) == 3, "prompt pair count/IDs drift")
    require(sum(bool(row["exact_duplicate_control"]) for row in prompt_mapping) == 1, "prompt identity control drift")
    for filename in ("clinical_template_reviewer.csv", "language_reviewer.csv"):
        header, rows = _csv(pack_dir / filename)
        require(len(rows) == 3, f"{filename}: wrong row count")
        require({row["item_id"] for row in rows} == prompt_ids, f"{filename}: item IDs drift")
        require(set(PROMPT_DECISION_FIELDS) <= set(header), f"{filename}: decisions missing")
        require(
            all(not row[field].strip() for row in rows for field in PROMPT_DECISION_FIELDS),
            f"{filename}: review decision prefilled",
        )

    schema = json.loads((pack_dir / "RETURN_SCHEMA.json").read_text(encoding="utf-8"))
    require(schema == return_schema(), "return schema drift")
    for role in ROLES:
        template = json.loads(
            (pack_dir / f"{role}.attestation.template.json").read_text(encoding="utf-8")
        )
        require(template["protocol_id"] == PACK_VERSION, f"{role}: attestation protocol drift")
        require(template["review_role"] == role, f"{role}: attestation role drift")
        require(
            template["reviewer"]["professional_role"] == PROFESSIONAL_ROLE[role],
            f"{role}: professional role drift",
        )
        require(not template["reviewer"]["reviewer_id"], f"{role}: reviewer ID prefilled")
        require(template["reviewer"]["independent_review"] is None, f"{role}: independence prefilled")

    authorization = manifest["scientific_authorization"]
    require(all(value is False for value in authorization.values()), "pack preauthorizes science")
    return {
        "version": VERSION,
        "passed": True,
        "status": "structurally_valid_awaiting_four_independent_human_returns",
        "pack": str(pack_dir.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "sealed_mapping_sha256": sha256_file(mapping_path),
        "source_pilot_frozen_verbatim": True,
        "selected_images": 60,
        "selected_strata": {stratum: 20 for stratum in STRATA},
        "clinical_pairs": 252,
        "primary_render_pairs": 240,
        "identity_render_controls": 12,
        "computational_guard_fail_pairs_retained_for_review": len(guard_failures),
        "computational_guard_fail_cells_model_eligible": False,
        "prompt_pairs": 3,
        "exact_prompt_duplicate_controls": 1,
        "pngs": len(actual_pngs),
        "review_sheets_blank": True,
        "reviewer_visible_leakage_checks_passed": True,
        "prior_polar_prompt_admission_reused": False,
        "model_or_gpu_authorized": False,
        "clinical_equivalence_established": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify(args.pack_dir)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
