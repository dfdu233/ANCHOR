import csv
import hashlib
import json
import tarfile
from pathlib import Path

from anchor.corrected_sgta.merge_specificity_ratchet_reviews_v1 import COPY_FIELDS
from anchor.medeval.package_specificity_ratchet_deliveries import package_deliveries
from anchor.medeval.verify_specificity_ratchet_deliveries import verify_delivery_dir
from scripts.smoke_specificity_ratchet_review_form import (
    immutable_projection,
    synthetic_completed_rows,
)


def _fixture(tmp_path: Path):
    pack = tmp_path / "pack"
    pack.mkdir()
    image_root = tmp_path / "images"
    (image_root / "test_images").mkdir(parents=True)
    image_bytes = b"fake-jpeg-for-packaging-test"
    digest = hashlib.sha256(image_bytes).hexdigest()
    image_relpath = f"test_images/{digest}.jpg"
    (image_root / image_relpath).write_bytes(image_bytes)
    candidate = {
        "case_id": "case-1",
        "edge_id": "edge-1",
        "question": "What is present?",
        "image_relpath": image_relpath,
        "answer_span": "A left opacity is present.",
        "parent_proposal": "An opacity is present.",
        "child_proposal": "A left opacity is present.",
        "added_constraint_proposal": "left",
        "edge_type": "laterality",
        "modality_stratum": "XR",
        "anatomy_stratum": "thorax",
        "answer_length_stratum": "short_le_50",
        "observability_screen": "potentially_single_image_decidable",
        "prompt_requested_increment": True,
        "proposal_only": True,
    }
    (pack / "candidates.blinded.jsonl").write_text(json.dumps(candidate) + "\n")
    fields = {
        "edge_entailment_admitted": ["yes", "no", "uncertain"],
        "parent_visual_support": ["supported", "refuted", "undetermined", "unobservable"],
        "child_visual_support": ["supported", "refuted", "undetermined", "unobservable"],
        "increment_observability": [
            "observable_on_supplied_image",
            "requires_other_view_or_sequence",
            "requires_history_lab_pathology_or_prior",
            "fundamentally_nonvisual_knowledge",
            "uncertain",
        ],
        "logical_scope_preserved": ["yes", "no", "not_applicable"],
        "reviewer_confidence": ["low", "medium", "high"],
        "clinical_usefulness_if_backed_off": [
            "improves", "unchanged", "minor_loss", "major_loss", "uncertain"
        ],
        "clinically_harmful_if_wrong": ["no", "minor", "major", "uncertain"],
    }
    schema = {
        "protocol_id": "specificity-ratchet-physician-pack-v2",
        "fields": fields,
    }
    (pack / "annotation_schema.json").write_text(json.dumps(schema))
    header = [*candidate, "reviewer_id", *COPY_FIELDS]
    row = {
        **{
            key: "True" if value is True else "False" if value is False else str(value)
            for key, value in candidate.items()
        },
        "reviewer_id": "",
        **{field: "" for field in COPY_FIELDS},
    }
    for role in (1, 2):
        with (pack / f"annotations.reviewer_{role}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
    return pack, image_root, schema, header, [row]


def test_v3_archives_are_deterministic_role_isolated_and_verified(tmp_path: Path):
    pack, image_root, _, _, _ = _fixture(tmp_path)
    output = tmp_path / "delivery"
    first = package_deliveries(pack_dir=pack, image_root=image_root, output_dir=output)
    assert verify_delivery_dir(output)["passed"] is True
    hashes = {row["role"]: row["archive_sha256"] for row in first["archives"]}
    second = package_deliveries(pack_dir=pack, image_root=image_root, output_dir=output)
    assert hashes == {row["role"]: row["archive_sha256"] for row in second["archives"]}
    for record in first["archives"]:
        with tarfile.open(output / record["archive"], "r:gz") as archive:
            names = archive.getnames()
            form = archive.extractfile(f"{record['root']}/REVIEW_FORM.html").read()
        other = 2 if record["role"] == 1 else 1
        assert not any(f"reviewer_{other}.csv" in name for name in names)
        assert not any("provenance" in name.lower() for name in names)
        assert b"http://" not in form and b"https://" not in form


def test_synthetic_smoke_completion_preserves_only_frozen_content(tmp_path: Path):
    _, _, schema, header, rows = _fixture(tmp_path)
    completed = synthetic_completed_rows(rows, schema, "SYNTHETIC")
    review_fields = list(schema["fields"])
    assert immutable_projection(completed, header, review_fields) == immutable_projection(
        rows, header, review_fields
    )
    assert completed[0]["reviewer_id"] == "SYNTHETIC"
    assert completed[0]["rationale"].startswith("Synthetic")
