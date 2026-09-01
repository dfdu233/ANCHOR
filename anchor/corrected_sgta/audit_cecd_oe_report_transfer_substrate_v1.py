#!/usr/bin/env python3
"""Outcome-blind census for a possible CECD OE/report transfer.

This audit deliberately does not extract claims, compare answers with a
reference, or inspect method efficacy.  Its only question is whether the
cached substrate already contains the identities and independent clinical
truth required to authorize an atomic-claim product-orbit experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "cecd-oe-report-transfer-substrate-audit-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AuditError(RuntimeError):
    """An input violates the outcome-blind census contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise AuditError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid JSON {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise AuditError(f"{path}:{line_number}: row must be an object")
            rows.append(dict(value))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid JSONL {path}: {error}") from error
    return rows


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path)


def _mimic_identifier(path: str, prefix: str) -> str | None:
    if prefix == "p":
        pattern = re.compile(r"^p\d{8}$")
    elif prefix == "s":
        pattern = re.compile(r"^s\d{8}$")
    else:  # pragma: no cover - internal misuse guard
        raise ValueError(prefix)
    return next((part for part in Path(path).parts if pattern.fullmatch(part)), None)


def summarize_vqa(
    manifest_path: Path, output_paths: Sequence[tuple[str, Path]]
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if not isinstance(manifest, list) or not manifest:
        raise AuditError("VQA manifest must be a non-empty list")
    if any(not isinstance(row, Mapping) for row in manifest):
        raise AuditError("VQA manifest contains a non-object row")
    qids = [str(row.get("qid", "")) for row in manifest]
    image_ids = [str(row.get("image_sha256", "")) for row in manifest]
    if any(not value for value in qids + image_ids):
        raise AuditError("VQA manifest lacks qid/image_sha256")
    if len(set(qids)) != len(qids):
        raise AuditError("VQA manifest contains duplicate qids")
    patient_fields = ("patient_id", "subject_id", "patient_group_id")
    patient_rows = sum(
        any(row.get(field) not in {None, ""} for field in patient_fields)
        for row in manifest
    )
    by_image = Counter(image_ids)
    outputs: dict[str, Any] = {}
    for name, path in output_paths:
        rows = load_jsonl(path)
        row_ids = [str(row.get("question_id", row.get("qid", ""))) for row in rows]
        outputs[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "unique_question_ids": len(set(row_ids)),
            "question_id_alignment": set(row_ids) == set(qids),
            "nonempty_answer_field_rows": sum(
                bool(str(row.get("text", row.get("answer", ""))).strip())
                for row in rows
            ),
            "claim_truth_fields_present": sum(
                isinstance(row.get("independent_clinical_truth"), Mapping)
                for row in rows
            ),
        }
    return {
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "rows": len(manifest),
            "unique_questions": len(set(qids)),
            "unique_images": len(set(image_ids)),
            "questions_per_image_histogram": {
                str(key): value
                for key, value in sorted(Counter(by_image.values()).items())
            },
            "rows_with_patient_group_identity": patient_rows,
            "rows_with_independent_atomic_visual_truth": sum(
                isinstance(row.get("independent_atomic_visual_truth"), list)
                for row in manifest
            ),
            "image_formats": sorted(
                {
                    Path(str(row.get("img_name", ""))).suffix.lower()
                    for row in manifest
                    if row.get("img_name")
                }
            ),
        },
        "outputs": outputs,
        "all_models_complete": bool(outputs)
        and all(row["question_id_alignment"] for row in outputs.values()),
        "patient_disjoint_split_verifiable": patient_rows == len(manifest),
        "independent_atomic_visual_truth_available": False,
        "render_prompt_equivalence_bound_to_cases": False,
        "interpretation": (
            "The benchmark answer is a direct-answer reference, not exhaustive "
            "independent truth for every clinical claim emitted by a long answer."
        ),
    }


def summarize_reports(output_paths: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    common_images: set[str] | None = None
    for name, path in output_paths:
        rows = load_jsonl(path)
        image_ids = [str(row.get("qid", "")) for row in rows]
        patients = [_mimic_identifier(str(row.get("img_name", "")), "p") for row in rows]
        studies = [_mimic_identifier(str(row.get("img_name", "")), "s") for row in rows]
        if any(not value for value in image_ids):
            raise AuditError(f"report output {path} lacks qid")
        current = set(image_ids)
        common_images = current if common_images is None else common_images & current
        outputs[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "unique_images": len(current),
            "unique_patients_from_path": len({value for value in patients if value}),
            "unique_studies_from_path": len({value for value in studies if value}),
            "rows_with_patient_id_from_path": sum(value is not None for value in patients),
            "rows_with_study_id_from_path": sum(value is not None for value in studies),
            "rows_with_nonempty_native_generation": sum(
                bool(str((row.get("greedy") or {}).get("text", "")).strip())
                for row in rows
            ),
            "rows_with_independent_atomic_visual_truth": sum(
                isinstance(row.get("independent_atomic_visual_truth"), list)
                for row in rows
            ),
            "input_image_formats": sorted(
                {
                    Path(str(row.get("img_name", ""))).suffix.lower()
                    for row in rows
                    if row.get("img_name")
                }
            ),
        }
    return {
        "outputs": outputs,
        "target_cecd_model_pair_available": {name for name, _ in output_paths}
        == {"huatuo", "hulu"},
        "common_image_count": len(common_images or set()),
        "patient_identity_recoverable": bool(outputs)
        and all(
            row["rows_with_patient_id_from_path"] == row["rows"]
            for row in outputs.values()
        ),
        "independent_atomic_visual_truth_available": False,
        "reference_report_is_visual_truth": False,
        "render_equivalence_admission_for_this_jpeg_cohort": False,
        "interpretation": (
            "MIMIC paths retain patient/study groups, but one clinical report and "
            "automatic report extraction cannot define image-grounded claim truth."
        ),
    }


def summarize_physician_pack(pack_root: Path) -> dict[str, Any]:
    metadata_path = pack_root / "review.metadata.json"
    template_path = pack_root / "review.template.jsonl"
    metadata = load_json(metadata_path)
    rows = load_jsonl(template_path)
    if not isinstance(metadata, Mapping):
        raise AuditError("physician pack metadata must be an object")
    answer_units = sum(
        len(row.get("candidate_answers", []))
        for row in rows
        if isinstance(row.get("candidate_answers"), list)
    )
    filled_claim_units = 0
    for row in rows:
        for answer in row.get("candidate_answers", []):
            annotation = answer.get("annotation", {}) if isinstance(answer, Mapping) else {}
            if isinstance(annotation, Mapping) and annotation.get("atomic_claims"):
                filled_claim_units += 1
    returns_root = pack_root / "clinical_returns_v1"
    completed = sorted(
        path
        for path in returns_root.glob("*")
        if path.is_file() and ".completed." in path.name
    )
    return {
        "root": str(pack_root.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "template_sha256": sha256_file(template_path),
        "groups": len(rows),
        "candidate_answer_units": answer_units,
        "declared_model_assignments": metadata.get("n_model_assignments"),
        "declared_blinded_answer_units": metadata.get("n_answer_units"),
        "template_answer_units_with_filled_atomic_claims": filled_claim_units,
        "completed_return_files": len(completed),
        "completed_return_names": [path.name for path in completed],
        "patient_group_identity_present": all(
            row.get("patient_id") or row.get("patient_group_id") for row in rows
        ),
        "cecd_product_orbit_bound": False,
        "interpretation": (
            "This is a 24-image baseline-review pilot. Empty templates are not "
            "clinical truth, and the pack predates any CECD product-orbit transfer."
        ),
    }


def audit_current_substrate(
    vqa_manifest: Path,
    vqa_outputs: Sequence[tuple[str, Path]],
    report_outputs: Sequence[tuple[str, Path]],
    physician_pack: Path,
) -> dict[str, Any]:
    if len({name for name, _ in vqa_outputs}) != len(vqa_outputs):
        raise AuditError("VQA output names must be unique")
    if len({name for name, _ in report_outputs}) != len(report_outputs):
        raise AuditError("report output names must be unique")
    vqa = summarize_vqa(vqa_manifest, vqa_outputs)
    reports = summarize_reports(report_outputs)
    physician = summarize_physician_pack(physician_pack)
    reasons = [
        "VQA-RAD has no patient-group identifier, so patient-disjoint dev/test cannot be proven.",
        "The 200 VQA references do not label every atomic claim in the three long answers.",
        "The physician pack has only 24 images, no completed clinical return, and no CECD orbit binding.",
        "MIMIC has patient IDs and two complete caches, but they are Hulu/LLaVA rather than the CECD Huatuo/Hulu pair and have no independent image-level atomic truth.",
        "Neither cached track has a cohort-specific clinically admitted render x prompt product orbit.",
        "No cache contains content-frozen claim targets or paired intervention outputs with structural conservation checks.",
    ]
    return {
        "protocol_version": PROTOCOL_ID,
        "status": "strict_no_go_current_vqa_rad_mimic_physician_pack",
        "scope_guard": (
            "This verdict covers only the audited VQA-RAD OE, MIMIC report, "
            "and existing Physician-OE artifacts. It does not adjudicate a "
            "separately constructed VinDr reader-vector ontology-listing track."
        ),
        "outcome_blind": True,
        "generated_reference_matching_performed": False,
        "automatic_claim_extraction_performed": False,
        "llm_judge_called": False,
        "model_scores_or_efficacy_metrics_read": False,
        "gpu_used": False,
        "sources": {
            "vqa_rad_oe": vqa,
            "mimic_report": reports,
            "physician_oe_pack": physician,
        },
        "gates": {
            "two_native_model_caches_available": (
                len(vqa_outputs) >= 2 and len(report_outputs) >= 2
            ),
            "target_cecd_model_pair_available_in_both_tasks": (
                {name for name, _ in vqa_outputs} >= {"huatuo", "hulu"}
                and {name for name, _ in report_outputs} == {"huatuo", "hulu"}
            ),
            "independent_atomic_clinical_truth": False,
            "patient_disjoint_dev_test_proven": False,
            "clinical_equivalence_product_orbit_bound": False,
            "atomic_teacher_forcing_targets_frozen": False,
            "fixed_content_coverage_contract_verifiable": False,
            "formal_oe_transfer_authorized": False,
            "formal_report_transfer_authorized": False,
            "gpu_authorized": False,
        },
        "no_go_reasons": reasons,
        "allowed_reuse": {
            "vqa_native_answers": "candidate-generation and annotation-pilot inputs only",
            "mimic_native_reports": "candidate-generation inputs with patient grouping only",
            "benchmark_answers_or_reports": "reviewer context only; never atomic visual truth",
            "radgraph_or_llm_judge": "candidate proposal/auxiliary analysis only; never truth",
        },
        "one_admissible_new_pack_option": {
            "substrate": "MIMIC-CXR, one frontal study/image per patient",
            "patients": {"dev": 80, "test": 120, "total": 200},
            "tasks_on_same_patient_cohort": ["oe_abnormality_listing", "report"],
            "models": ["huatuo", "hulu"],
            "native_answer_or_report_units": 800,
            "truth": (
                "two independent source-model-blinded radiologists plus a third "
                "radiologist adjudicator; image inspected; reference hidden while "
                "assigning supported/refuted/undetermined/unobservable and required/optional/out_of_scope"
            ),
            "render_prompt_admission": (
                "at least 60 separate MIMIC images, two independent reviewers plus "
                "adjudication, before any orbit score; admission hashes bound to pack"
            ),
            "claim_cell_minima_per_model_task": {
                "dev": {"supported": 40, "refuted": 20, "undetermined": 20},
                "test": {"supported": 60, "refuted": 30, "undetermined": 30},
            },
            "patient_bootstrap": True,
            "limitation": (
                "OE and report on the same 200 patients are paired output-form "
                "transfer, not independent-dataset replication."
            ),
        },
        "separate_near_term_track_not_audited": {
            "name": "VinDr reader-vector fixed-ontology abnormality listing",
            "required_independent_checks": [
                "patient/group-disjoint split provenance",
                "reader-vector truth retained per ontology atom",
                "native multi-claim output admission",
                "the same content/coverage and product-orbit conservation gates",
            ],
        },
    }


def atomic_write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-manifest", type=Path, required=True)
    parser.add_argument(
        "--vqa-output", action="append", type=parse_named_path, default=[]
    )
    parser.add_argument(
        "--report-output", action="append", type=parse_named_path, default=[]
    )
    parser.add_argument("--physician-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = audit_current_substrate(
            args.vqa_manifest,
            args.vqa_output,
            args.report_output,
            args.physician_pack,
        )
        encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        atomic_write_new(args.output, encoded)
    except (AuditError, FileExistsError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
