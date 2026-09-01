#!/usr/bin/env python3
"""Fail-closed structural validator for the VinDr CECD listing pack.

Validation proves only that the fixed reader-vector claim universe, split,
strict parser, and 5x3 orbit are internally coherent.  It never establishes
human equivalence admission or model efficacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from corrected_sgta.clinical_claims import reader_state
from corrected_sgta.prepare_vindr_cecd_ontology_listing_v1 import (
    DUPLICATE_PROMPT_ID,
    IDENTITY_RENDER_ID,
    NONE_TOKEN,
    PANEL,
    SCHEMA_VERSION,
    SCIENCE_PROMPT_IDS,
    SCIENCE_RENDER_IDS,
    SPLITS,
    STRATA,
    TARGET_FINDINGS,
    canonical_hash,
    orbit_cells,
    reference_relevance,
)
from corrected_sgta.prepare_vindr_reader_manifest import sha256_file
from corrected_sgta.prepare_vindr_reader_manifest_v2 import three_way_split


VERSION = "vindr-cecd-ontology-listing-structural-validator-v1"
MODELS = {"huatuo", "hulu"}


class ValidationError(RuntimeError):
    """A pack violates the frozen substrate contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid JSON {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                require(
                    isinstance(value, Mapping),
                    f"{path}:{line_number}: row must be an object",
                )
                rows.append(dict(value))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid JSONL {path}: {error}") from error
    return rows


def parse_ontology_listing(text: str) -> dict[str, Any]:
    """Parse only the exact grammar named by the prompt.

    Unknown or prose-bearing items remain visible in ``unknown_items`` and
    make the whole output a format violation.  Valid ontology labels are still
    returned for transparent diagnostics, but downstream formal analysis may
    not silently treat a violation as a clean closed-world list.
    """

    raw = str(text).strip()
    by_label = {source.casefold(): finding for finding, source in TARGET_FINDINGS}
    if raw == NONE_TOKEN:
        return {
            "status": "valid",
            "finding_ids": [],
            "unknown_items": [],
            "duplicate_finding_ids": [],
            "empty_set": True,
        }
    if not raw:
        return {
            "status": "format_violation",
            "finding_ids": [],
            "unknown_items": ["<empty_output>"],
            "duplicate_finding_ids": [],
            "empty_set": False,
        }
    items = [item.strip() for item in raw.split(",")]
    finding_ids: list[str] = []
    unknown: list[str] = []
    for item in items:
        if not item or item == NONE_TOKEN:
            unknown.append(item or "<empty_item>")
            continue
        finding = by_label.get(item.casefold())
        if finding is None:
            unknown.append(item)
        else:
            finding_ids.append(finding)
    counts = Counter(finding_ids)
    duplicates = sorted(finding for finding, count in counts.items() if count > 1)
    status = "valid" if not unknown and not duplicates else "format_violation"
    return {
        "status": status,
        "finding_ids": sorted(counts),
        "unknown_items": unknown,
        "duplicate_finding_ids": duplicates,
        "empty_set": False,
    }


def _claim_map(row: Mapping[str, Any], where: str) -> dict[str, Mapping[str, Any]]:
    claims = row.get("claims")
    require(isinstance(claims, list), f"{where}: claims must be a list")
    mapped: dict[str, Mapping[str, Any]] = {}
    for claim in claims:
        require(isinstance(claim, Mapping), f"{where}: claim must be object")
        finding = str(claim.get("finding_id", ""))
        require(finding not in mapped, f"{where}: duplicate finding {finding}")
        mapped[finding] = claim
    expected = {finding for finding, _ in TARGET_FINDINGS}
    require(set(mapped) == expected, f"{where}: claim universe is not exact 14")
    return mapped


def validate_reference_row(
    row: Mapping[str, Any], *, seed: int, image_root: Path, where: str
) -> tuple[str, str]:
    image_id = str(row.get("image_id", ""))
    split = str(row.get("experiment_split", ""))
    stratum = str(row.get("sampling_stratum", ""))
    require(bool(image_id), f"{where}: empty image_id")
    require(split in SPLITS, f"{where}: invalid split")
    require(stratum in STRATA, f"{where}: non-primary stratum selected")
    require(
        three_way_split(image_id, seed) == split,
        f"{where}: split is not the frozen image hash split",
    )
    require(row.get("reader_panel") == list(PANEL), f"{where}: reader panel drift")
    require(row.get("patient_group_id") is None, f"{where}: unexpected patient ID")
    relative = str(row.get("dicom_relpath", ""))
    require(relative == f"train/{image_id}.dicom", f"{where}: DICOM path drift")
    dicom = image_root / f"{image_id}.dicom"
    require(dicom.is_file(), f"{where}: DICOM missing")
    require(
        int(row.get("dicom_size_bytes", -1)) == dicom.stat().st_size,
        f"{where}: DICOM size drift",
    )

    claims = _claim_map(row, where)
    required: list[str] = []
    optional: list[str] = []
    refuted: list[str] = []
    for finding_id, _ in TARGET_FINDINGS:
        claim = claims[finding_id]
        votes = claim.get("reader_votes")
        require(isinstance(votes, list) and len(votes) == 3, f"{where}: bad votes")
        require(
            [str(item.get("rad_id")) for item in votes if isinstance(item, Mapping)]
            == list(PANEL),
            f"{where}: vote reader order drift",
        )
        values = [int(item.get("vote", -1)) for item in votes]
        require(set(values) <= {0, 1}, f"{where}: nonbinary vote")
        positive = sum(values)
        require(int(claim.get("positive_votes", -1)) == positive, f"{where}: vote sum drift")
        require(int(claim.get("reader_count", -1)) == 3, f"{where}: reader count drift")
        require(
            float(claim.get("reader_support", -1)) == positive / 3.0,
            f"{where}: support drift",
        )
        require(
            claim.get("reader_state") == reader_state(positive, 3),
            f"{where}: reader state drift",
        )
        require(
            claim.get("listing_relevance") == reference_relevance(positive),
            f"{where}: relevance drift",
        )
        if positive == 3:
            required.append(finding_id)
        elif positive == 0:
            refuted.append(finding_id)
        else:
            optional.append(finding_id)
    require(row.get("required_finding_ids") == required, f"{where}: required set drift")
    require(row.get("optional_finding_ids") == optional, f"{where}: optional set drift")
    require(row.get("refuted_finding_ids") == refuted, f"{where}: refuted set drift")
    no_finding = int(row.get("no_finding_positive_votes", -1))
    if stratum == "unanimous_no_finding":
        require(no_finding == 3, f"{where}: normal stratum lacks 3/3 No finding")
        require(not required and not optional, f"{where}: normal stratum has target finding")
        require(
            not row.get("outside_target_ontology_reader_positive"),
            f"{where}: normal stratum has outside finding",
        )
    elif stratum == "one_unanimous_target_finding":
        require(len(required) == 1, f"{where}: single stratum cardinality drift")
    else:
        require(len(required) >= 2, f"{where}: multi stratum cardinality drift")
    return split, stratum


def validate_orbit(manifest: Mapping[str, Any]) -> None:
    orbit = manifest.get("orbit_contract")
    require(isinstance(orbit, Mapping), "orbit_contract missing")
    require(
        orbit.get("science_render_ids") == list(SCIENCE_RENDER_IDS),
        "science render contract drift",
    )
    require(
        orbit.get("science_prompt_ids") == list(SCIENCE_PROMPT_IDS),
        "science prompt contract drift",
    )
    require(orbit.get("identity_render_id") == IDENTITY_RENDER_ID, "identity render drift")
    require(
        orbit.get("duplicate_prompt_id") == DUPLICATE_PROMPT_ID,
        "duplicate prompt drift",
    )
    require(int(orbit.get("science_cells", -1)) == 15, "science cell count drift")
    require(int(orbit.get("total_cells", -1)) == 19, "total cell count drift")
    require(orbit.get("cells") == orbit_cells(), "orbit cell payload drift")


def validate_outputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_ids: set[str],
    cell_ids: set[str],
) -> dict[str, Any]:
    """Validate future output coverage and grammar without reading truth labels."""

    expected = {
        (model, image_id, cell_id)
        for model in MODELS
        for image_id in image_ids
        for cell_id in cell_ids
    }
    observed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    parse_counts = Counter()
    unknown_items = Counter()
    for index, row in enumerate(rows):
        key = (
            str(row.get("model", "")),
            str(row.get("image_id", "")),
            str(row.get("cell_id", "")),
        )
        require(key not in observed, f"outputs[{index}]: duplicate cell {key}")
        require(key in expected, f"outputs[{index}]: unexpected cell {key}")
        observed[key] = row
        require(row.get("status") == "ok", f"outputs[{index}]: non-ok model cell")
        parsed = parse_ontology_listing(str(row.get("text", "")))
        parse_counts[parsed["status"]] += 1
        unknown_items.update(parsed["unknown_items"])
    require(set(observed) == expected, "future outputs do not cover both models/full orbit")
    return {
        "rows": len(rows),
        "full_two_model_product_orbit": True,
        "parse_status_counts": dict(parse_counts),
        "unknown_item_counts": dict(unknown_items),
        "all_cells_strictly_parseable": parse_counts["format_violation"] == 0,
        "efficacy_or_truth_comparison_performed": False,
    }


def validate_pack(
    manifest_path: Path, outputs_path: Path | None = None
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    require(isinstance(manifest, Mapping), "manifest must be object")
    require(manifest.get("schema_version") == SCHEMA_VERSION, "schema version drift")
    require(manifest.get("outcome_blind") is True, "pack not outcome blind")
    require(manifest.get("model_outputs_read") is False, "builder read model output")
    require(manifest.get("model_scores_read") is False, "builder read model score")
    require(manifest.get("gpu_used") is False, "builder used GPU")
    fingerprint = str(manifest.get("fingerprint", ""))
    unsigned = dict(manifest)
    unsigned.pop("fingerprint", None)
    require(fingerprint == canonical_hash(unsigned), "manifest fingerprint drift")

    source = manifest.get("source")
    require(isinstance(source, Mapping), "source contract missing")
    labels_csv = Path(str(source.get("labels_csv", "")))
    require(labels_csv.is_file(), "wide source labels missing")
    require(
        source.get("labels_csv_sha256") == sha256_file(labels_csv),
        "wide source labels hash drift",
    )
    require(source.get("reader_panel") == list(PANEL), "source reader panel drift")
    image_root = Path(str(source.get("image_root", "")))
    require(image_root.is_dir(), "image root missing")

    task = manifest.get("task_contract")
    require(isinstance(task, Mapping), "task_contract missing")
    require(
        task.get("formal_task_type") == "ontology_constrained_open_cardinality_listing",
        "task type drift",
    )
    require(task.get("free_form_oe") is False, "closed ontology mislabeled free OE")
    require(task.get("target_ontology_is_exhaustive") is False, "14 labels mislabeled exhaustive")
    require(
        task.get("empty_set_token_is_independent_clinical_claim") is False,
        "empty-set token mislabeled as a clinical claim",
    )
    require(
        task.get("target_finding_ids") == [value for value, _ in TARGET_FINDINGS],
        "target finding IDs drift",
    )
    require(task.get("empty_set_token") == NONE_TOKEN, "empty-set token drift")

    reference = manifest.get("reference_contract")
    require(isinstance(reference, Mapping), "reference_contract missing")
    reference_path = manifest_path.parent / str(reference.get("reference_file", ""))
    require(reference_path.is_file(), "reference file missing")
    require(
        reference.get("reference_file_sha256") == sha256_file(reference_path),
        "reference file hash drift",
    )
    rows = load_jsonl(reference_path)
    require(len(rows) == int(reference.get("reference_rows", -1)), "reference row count drift")
    require(int(reference.get("claims_per_image", -1)) == 14, "claims/image drift")

    split_contract = manifest.get("split_contract")
    require(isinstance(split_contract, Mapping), "split_contract missing")
    seed = int(split_contract.get("seed", -1))
    quotas = split_contract.get("quotas_per_split_per_stratum")
    require(isinstance(quotas, Mapping), "split quotas missing")
    observed = Counter()
    image_ids: set[str] = set()
    for index, row in enumerate(rows):
        image_id = str(row.get("image_id", ""))
        require(image_id not in image_ids, f"reference[{index}]: duplicate image")
        image_ids.add(image_id)
        split, stratum = validate_reference_row(
            row,
            seed=seed,
            image_root=image_root,
            where=f"reference[{index}]",
        )
        observed[(split, stratum)] += 1
    for split in SPLITS:
        for stratum in STRATA:
            require(
                observed[(split, stratum)] == int(quotas[split]),
                f"quota drift for {split}/{stratum}",
            )
    require(split_contract.get("image_disjoint") is True, "image split not disjoint")
    identity = manifest.get("dicom_identity_audit")
    require(isinstance(identity, Mapping), "DICOM identity audit missing")
    patient_disjoint = bool(identity.get("patient_disjoint_split_verifiable"))
    require(
        split_contract.get("patient_disjoint_verifiable") is patient_disjoint,
        "patient-disjoint flag drift",
    )
    # The released local files have no patient identifier.  If a future source
    # differs, this validator forces a new version rather than silently changing
    # the scientific unit.
    require(not patient_disjoint, "v1 contract expects image-only, not patient, identity")

    validate_orbit(manifest)
    admission = manifest.get("admission_contract")
    require(isinstance(admission, Mapping), "admission_contract missing")
    require(
        admission.get("existing_binary_ce_prompt_admission_transfers") is False,
        "binary CE admission cannot transfer to listing prompts",
    )
    require(
        admission.get("status") == "pending_independent_human_admission",
        "unexpected admission state",
    )
    require(admission.get("model_orbit_scoring_authorized") is False, "model scoring preauthorized")
    require(admission.get("gpu_authorized") is False, "GPU preauthorized")
    scope = manifest.get("scope_guards")
    require(isinstance(scope, Mapping), "scope_guards missing")
    require(scope.get("free_oe_hallucination_claim_authorized") is False, "free OE overclaim")
    require(
        scope.get("patient_disjoint_generalization_authorized") is False,
        "patient generalization overclaim",
    )

    output_audit = None
    if outputs_path is not None:
        output_rows = load_jsonl(outputs_path)
        output_audit = validate_outputs(
            output_rows,
            image_ids=image_ids,
            cell_ids={str(row["cell_id"]) for row in orbit_cells()},
        )
    return {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "structurally_valid_conditional_go_closed_ontology_only",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "reference": str(reference_path.resolve()),
        "reference_sha256": sha256_file(reference_path),
        "selected_images": len(rows),
        "selected_atomic_claims": len(rows) * 14,
        "selected_true_multiclaim_images": sum(
            row["sampling_stratum"] == "multiple_unanimous_target_findings"
            for row in rows
        ),
        "gates": {
            "complete_three_reader_vectors": True,
            "fixed_14_claim_universe": True,
            "true_multiclaim_cases_present": True,
            "image_disjoint_split": True,
            "patient_disjoint_split": False,
            "strict_automatic_parser_defined": True,
            "five_by_three_orbit_structurally_defined": True,
            "new_listing_equivalence_admitted": False,
            "free_oe_claim_authorized": False,
            "model_or_gpu_authorized": False,
            "efficacy_claim_authorized": False,
        },
        "future_output_structural_audit": output_audit,
        "claim_ceiling": (
            "After independent listing-prompt and multi-claim render admission, this "
            "pack can test aggregate closed-ontology content selection and fixed-K "
            "coverage. It cannot establish free-form OE, report, or patient-disjoint "
            "generalization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = validate_pack(args.manifest, args.outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
