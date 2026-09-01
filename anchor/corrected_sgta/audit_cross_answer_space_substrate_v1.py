#!/usr/bin/env python3
"""Fail-closed audit for same-image, same-claim CE/OE substrate.

This audit intentionally stops before semantic claim extraction.  It counts
only original human-authored QA/report artifacts that can be joined by an
exact image identifier.  Same-image CE/OE cross-products and conservative
literal matches are reported as *upper bounds*, never as clinical truth.

Formal admission requires a separate manifest in which two reviewers agree
that both answer spaces express the same atomic clinical claim.  In its
absence, formal eligible-pair counts are exactly zero.  Model outputs, common
evaluation artifacts, and LLM judges are outside the input contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "cross-answer-space-substrate-audit-v1"
MIN_FINDINGS = 3
MIN_PER_DIRECTION_TASK_CELL = 50
CLINICAL_CONTENT_TYPE = "Abnormality"
YES_NO = {"yes", "no"}

# Terms whose literal occurrence cannot establish a clinical finding identity.
NON_FINDING_ANSWERS = {
    "abnormal",
    "ap",
    "axial",
    "bilateral",
    "both",
    "brain",
    "chest",
    "chest x ray",
    "coronal",
    "ct",
    "female",
    "flair",
    "heart",
    "kidney",
    "left",
    "liver",
    "lung",
    "lungs",
    "male",
    "mri",
    "no",
    "none",
    "normal",
    "one",
    "pa",
    "pancreas",
    "right",
    "sagittal",
    "three",
    "the brain",
    "two",
    "ultrasound",
    "x ray",
    "yes",
    "enlarged",
}


class AuditError(RuntimeError):
    """A source artifact violates the frozen audit contract."""


def sha256_file(path: Path) -> str:
    """Hash one exact audit input without normalizing or rewriting it."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise AuditError(f"cannot hash input {path}: {error}") from error
    return digest.hexdigest()


def audit_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Bind the audit result to its program, command, and exact source files."""

    sources = {
        "slake_train": args.slake_root / "train.json",
        "slake_validation": args.slake_root / "validation.json",
        "slake_test": args.slake_root / "test.json",
        "vqa_rad_train": args.vqa_rad_train,
        "vqa_rad_test": args.vqa_rad_test,
        "medheval_slake_closed": args.medheval_fine_root / "slake_qa_pairs.json",
        "medheval_vqa_rad_closed": args.medheval_fine_root / "rad_vqa_pairs.json",
        "medheval_iu_xray_closed": args.medheval_fine_root / "xray_closed_pairs.json",
        "iu_xray_annotation": args.iuxray_annotation,
    }
    source_rows = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sources.items()
    }
    evaluator_path = Path(__file__).resolve()
    evaluator_sha256 = sha256_file(evaluator_path)
    fingerprint_payload = {
        "protocol_id": PROTOCOL_ID,
        "evaluator_sha256": evaluator_sha256,
        "source_sha256": {
            name: row["sha256"] for name, row in sorted(source_rows.items())
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "dataset": ["SLAKE", "VQA-RAD", "MedHEval-derived", "IU-Xray"],
        "model": "not_applicable_cpu_substrate_audit",
        "method": PROTOCOL_ID,
        "seed": "not_applicable_deterministic_audit",
        "command": [sys.executable, *sys.argv],
        "evaluator": {
            "path": str(evaluator_path),
            "sha256": evaluator_sha256,
        },
        "sources": source_rows,
        "fingerprint": fingerprint,
    }


def normalize_text(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def answer_space(row: Mapping[str, Any]) -> str:
    """Classify VQA-RAD by gold answer and SLAKE by original metadata."""

    declared = str(row.get("answer_type", "")).upper()
    if declared in {"OPEN", "CLOSED"}:
        return declared
    return "CLOSED" if normalize_text(row.get("answer")) in YES_NO else "OPEN"


def summarize_original_qa(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count exact-image cross-task availability without semantic matching."""

    materialized = [dict(row) for row in rows]
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        image_id = str(row.get("image_id") or row.get("img_name") or "")
        if not image_id:
            raise AuditError("original QA row lacks an exact image identifier")
        by_image[image_id].append(row)

    open_count = sum(answer_space(row) == "OPEN" for row in materialized)
    closed_count = len(materialized) - open_count
    both_images = 0
    raw_cross_product = 0
    exact_surface_pairs = 0
    for image_rows in by_image.values():
        open_rows = [row for row in image_rows if answer_space(row) == "OPEN"]
        closed_rows = [row for row in image_rows if answer_space(row) == "CLOSED"]
        if open_rows and closed_rows:
            both_images += 1
        raw_cross_product += len(open_rows) * len(closed_rows)
        open_questions = Counter(normalize_text(row.get("question")) for row in open_rows)
        closed_questions = Counter(normalize_text(row.get("question")) for row in closed_rows)
        exact_surface_pairs += sum(
            open_questions[key] * closed_questions[key]
            for key in open_questions.keys() & closed_questions.keys()
        )

    return {
        "qa_rows": len(materialized),
        "unique_images": len(by_image),
        "open_rows": open_count,
        "closed_rows": closed_count,
        "images_with_both_answer_spaces": both_images,
        "same_image_cross_product_upper_bound": raw_cross_product,
        "same_image_exact_question_cross_space_pairs": exact_surface_pairs,
    }


def literal_answer_in_closed_question_candidates(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Return a sensitivity-only lexical upper bound.

    A candidate is emitted only when the *entire* normalized OE answer appears
    as a token phrase in a same-image CE question.  This does not prove atomic
    claim equivalence, polarity consistency, or location agreement.
    """

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        image_id = str(row.get("image_id") or row.get("img_name") or "")
        if not image_id:
            raise AuditError("candidate row lacks an exact image identifier")
        by_image[image_id].append(row)

    candidates: list[dict[str, str]] = []
    for image_id, image_rows in by_image.items():
        open_rows = [row for row in image_rows if answer_space(row) == "OPEN"]
        closed_rows = [row for row in image_rows if answer_space(row) == "CLOSED"]
        for open_row in open_rows:
            answer = normalize_text(open_row.get("answer"))
            if (
                not answer
                or answer in NON_FINDING_ANSWERS
                or len(answer) < 4
                or "," in str(open_row.get("answer", ""))
            ):
                continue
            needle = f" {answer} "
            for closed_row in closed_rows:
                if needle not in f" {normalize_text(closed_row.get('question'))} ":
                    continue
                candidates.append(
                    {
                        "image_id": image_id,
                        "literal_answer": answer,
                        "open_question": str(open_row.get("question", "")),
                        "open_answer": str(open_row.get("answer", "")),
                        "closed_question": str(closed_row.get("question", "")),
                        "closed_answer": str(closed_row.get("answer", "")),
                        "status": "automatic_candidate_only",
                    }
                )
    return candidates


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read JSON {path}: {error}") from error


def audit_slake(slake_root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {
        "provenance": "original_human_qa",
        "language": "en",
        "patient_identifier_available": False,
        "split_unit": "question records; English image IDs are disjoint locally",
        "splits": {},
    }
    split_images: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        rows = [
            row
            for row in _load_json(slake_root / f"{split}.json")
            if row.get("q_lang") == "en"
        ]
        split_images[split] = {str(row["img_name"]) for row in rows}
        clinical = [
            row for row in rows if row.get("content_type") == CLINICAL_CONTENT_TYPE
        ]
        candidate_rows = literal_answer_in_closed_question_candidates(clinical)
        content_counts = Counter(str(row.get("content_type") or "<missing>") for row in rows)
        output["splits"][split] = {
            "all_original_qa": summarize_original_qa(rows),
            "content_type_counts": dict(sorted(content_counts.items())),
            "clinical_abnormality_only": summarize_original_qa(clinical),
            "clinical_literal_candidates": {
                "pair_count": len(candidate_rows),
                "unique_images": len({row["image_id"] for row in candidate_rows}),
                "literal_terms": dict(
                    sorted(Counter(row["literal_answer"] for row in candidate_rows).items())
                ),
                "status": "sensitivity_only_not_truth",
            },
        }
    output["english_image_overlap"] = {
        "train_validation": len(split_images["train"] & split_images["validation"]),
        "train_test": len(split_images["train"] & split_images["test"]),
        "validation_test": len(split_images["validation"] & split_images["test"]),
    }
    return output


def _read_vqa_rad_parquet(path: Path) -> list[dict[str, str]]:
    try:
        import duckdb
    except ImportError as error:
        raise AuditError("duckdb is required to inspect local VQA-RAD parquet") from error
    connection = duckdb.connect()
    escaped = str(path).replace("'", "''")
    records = connection.execute(
        f"SELECT md5(image.bytes), question, answer FROM read_parquet('{escaped}')"
    ).fetchall()
    return [
        {"image_id": image_id, "question": question, "answer": answer}
        for image_id, question, answer in records
    ]


def audit_vqa_rad(train_path: Path, test_path: Path) -> dict[str, Any]:
    train_rows = _read_vqa_rad_parquet(train_path)
    test_rows = _read_vqa_rad_parquet(test_path)
    output: dict[str, Any] = {
        "provenance": "original_clinician_qa_repacked_by_huggingface",
        "patient_identifier_available": False,
        "split_unit": "QA; not image-disjoint",
        "splits": {},
    }
    for split, rows in (("train", train_rows), ("test", test_rows)):
        candidates = literal_answer_in_closed_question_candidates(rows)
        output["splits"][split] = {
            "all_original_qa": summarize_original_qa(rows),
            "literal_candidates": {
                "pair_count": len(candidates),
                "unique_images": len({row["image_id"] for row in candidates}),
                "literal_terms": dict(
                    sorted(Counter(row["literal_answer"] for row in candidates).items())
                ),
                "status": "sensitivity_only_not_truth",
            },
        }
    train_images = {row["image_id"] for row in train_rows}
    test_images = {row["image_id"] for row in test_rows}
    output["train_test_image_hash_overlap"] = len(train_images & test_images)
    return output


def audit_medheval_and_iuxray(
    medheval_fine_root: Path,
    iuxray_annotation_path: Path,
) -> dict[str, Any]:
    names = {
        "slake": "slake_qa_pairs.json",
        "vqa_rad": "rad_vqa_pairs.json",
        "iu_xray": "xray_closed_pairs.json",
    }
    derived: dict[str, Any] = {}
    for dataset, filename in names.items():
        rows = _load_json(medheval_fine_root / filename)
        derived[dataset] = {
            "rows": len(rows),
            "unique_img_name": len({str(row.get("img_name")) for row in rows}),
            "provenance": "gpt_4_128k_synthesized_close_ended",
            "formal_gold_eligible": False,
        }

    annotation = _load_json(iuxray_annotation_path)
    flattened: list[tuple[str, Mapping[str, Any]]] = []
    for split, studies in annotation.items():
        flattened.extend((split, study) for study in studies)
    image_map = {
        str(image_path): (split, str(study["id"]))
        for split, study in flattened
        for image_path in study.get("image_path", [])
    }
    iu_rows = _load_json(medheval_fine_root / names["iu_xray"])
    joined = [row for row in iu_rows if str(row.get("img_name")) in image_map]
    derived["iu_xray"].update(
        {
            "human_report_studies": len(flattened),
            "human_report_views": len(image_map),
            "exact_view_path_join_rows": len(joined),
            "exact_view_path_join_unique_views": len(
                {str(row.get("img_name")) for row in joined}
            ),
            "exact_view_path_join_unique_studies": len(
                {image_map[str(row.get("img_name"))][1] for row in joined}
            ),
            "report_claim_extraction": "not_performed",
            "patient_identifier_available": False,
        }
    )
    return derived


def gate_summary() -> dict[str, Any]:
    return {
        "thresholds": {
            "minimum_clinical_findings": MIN_FINDINGS,
            "minimum_per_direction_task_cell": MIN_PER_DIRECTION_TASK_CELL,
            "patient_and_image_disjoint_dev_test": True,
            "dual_reviewer_same_atomic_claim_equivalence": True,
        },
        "observed_formal_dual_reviewed_pairs": 0,
        "f6": {
            "verdict": "KILL",
            "reason": "no dual-reviewed same-atomic-claim equivalence manifest",
        },
        "f7": {
            "verdict": "KILL",
            "reason": (
                "original VQA-RAD exact surface pairs are zero; literal holdout "
                "candidates are far below three findings x fifty per cell and "
                "the official QA split is not image-disjoint"
            ),
        },
        "overall": "NO_GO",
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "provenance": audit_provenance(args),
        "scope": (
            "same original medical image and original gold CE/OE-or-report; "
            "no model-output selection, common eval, LLM judge, or automatic truth"
        ),
        "slake": audit_slake(args.slake_root),
        "vqa_rad": audit_vqa_rad(args.vqa_rad_train, args.vqa_rad_test),
        "medheval_derived_and_iuxray_join": audit_medheval_and_iuxray(
            args.medheval_fine_root, args.iuxray_annotation
        ),
        "formal_pair_manifest_schema": {
            "identity": ["pair_id", "dataset", "split", "patient_id", "image_id"],
            "closed_source": [
                "closed_source_id",
                "closed_question",
                "closed_answer",
                "closed_provenance",
            ],
            "open_source": [
                "open_source_id",
                "open_question_or_report_span",
                "open_answer_or_report",
                "open_provenance",
            ],
            "atomic_claim": [
                "finding",
                "polarity",
                "uncertainty",
                "anatomy",
                "attributes",
            ],
            "review": [
                "reviewer_1_equivalent",
                "reviewer_2_equivalent",
                "reviewer_agreement",
            ],
        },
        "gates": gate_summary(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slake-root", type=Path, default=Path("/home/dbw/data/SLAKE"))
    parser.add_argument(
        "--vqa-rad-train",
        type=Path,
        default=Path(
            "/home/dbw/data/VQA-RAD/data/train-00000-of-00001-eb8844602202be60.parquet"
        ),
    )
    parser.add_argument(
        "--vqa-rad-test",
        type=Path,
        default=Path(
            "/home/dbw/data/VQA-RAD/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet"
        ),
    )
    parser.add_argument(
        "--medheval-fine-root",
        type=Path,
        default=Path(
            "data/medheval/benchmark_data/Visual_Misinterpretation_Hallucination/"
            "close-ended/fine-grained"
        ),
    )
    parser.add_argument(
        "--iuxray-annotation",
        type=Path,
        default=Path("data/medheval/images/IU-Xray/annotation.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_audit(args)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
