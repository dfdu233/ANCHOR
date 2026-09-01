import json
import tarfile
from pathlib import Path

from anchor.medeval.package_physician_oe_deliveries import (
    _review_form,
    package_deliveries,
    sha256_file,
)
from anchor.medeval.verify_physician_oe_delivery_archives import verify_delivery_dir


def _annotation():
    return {
        "direct_answer_correctness": None,
        "direct_answer_state": None,
        "atomic_claims": [],
        "no_clinical_claims": None,
        "omitted_required_claim_ids": [],
        "overall_clinically_harmful": None,
        "reviewer_confidence": None,
        "rationale": "",
    }


def test_packages_role_isolated_hash_bound_archives(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    image_data = b"not-a-real-jpeg"
    import hashlib

    image_hash = hashlib.sha256(image_data).hexdigest()
    image_name = f"{image_hash}.jpg"
    (images / image_name).write_bytes(image_data)

    source_template = tmp_path / "review.template.jsonl"
    source_template.write_text("source\n")
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    reviewers = {}
    for role in ("A", "B"):
        row = {
            "bundle_id": "bundle-1",
            "group_id": "group-1",
            "review_order": 0,
            "image": {"relative_path": image_name, "sha256": image_hash},
            "question": "question",
            "benchmark_reference": "reference",
            "reference_annotation": {},
            "candidate_answers": [
                {"answer_id": "answer-1", "answer_text": "answer", "annotation": _annotation()}
            ],
            "reviewer_slot": role,
            "review_phase": "calibration",
        }
        path = delivery / f"reviewer_{role}.blinded.jsonl"
        path.write_text(json.dumps(row) + "\n")
        reviewers[role] = {"sha256": sha256_file(path), "groups": 1, "answer_units": 1}
    clarification = delivery / "clarification_log.template.md"
    clarification.write_text("# Clarification\n")
    (delivery / "delivery_manifest.json").write_text(
        json.dumps(
                {
                    "version": "anchor-physician-oe-review-deliveries-v1",
                    "bundle_id": "bundle-1",
                    "source_template_sha256": sha256_file(source_template),
                    "calibration_groups": 1,
                "reviewers": reviewers,
                "private_mapping_in_delivery": False,
            }
        )
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-1",
                "bundle_sha256": sha256_file(source_template),
                "image_root": str(images),
            }
        )
    )
    runbook = tmp_path / "runbook.md"
    runbook.write_text("# Review rules\n")
    output = tmp_path / "archives"
    first = package_deliveries(
        delivery_dir=delivery,
        metadata_path=metadata,
        runbook=runbook,
        output_dir=output,
    )
    archive = output / first["archives"][0]["archive"]
    with tarfile.open(archive, "r:gz") as handle:
        instructions = next(
            member for member in handle.getmembers() if member.name.endswith("/INSTRUCTIONS.md")
        )
        text = handle.extractfile(instructions).read().decode("utf-8")
    assert "first 1 `calibration` groups" in text
    verification = verify_delivery_dir(output)
    assert verification["passed"] is True
    assert {record["role"] for record in first["archives"]} == {"A", "B"}

    hashes = {record["role"]: record["archive_sha256"] for record in first["archives"]}
    second = package_deliveries(
        delivery_dir=delivery,
        metadata_path=metadata,
        runbook=runbook,
        output_dir=output,
    )
    assert hashes == {
        record["role"]: record["archive_sha256"] for record in second["archives"]
    }

    with tarfile.open(output / first["archives"][0]["archive"], "r:gz") as archive:
        names = archive.getnames()
    assert not any("reviewer_B" in name for name in names)
    assert not any("mapping" in name.lower() for name in names)
    assert any(name.endswith("REVIEW_FORM.html") for name in names)


def test_offline_form_embeds_only_blinded_rows_and_protects_script_boundary():
    rows = [
        {
            "bundle_id": "bundle-1",
            "group_id": "group-1",
            "review_order": 0,
            "review_phase": "calibration",
            "reviewer_slot": "A",
            "image": {"relative_path": "abc.jpg", "sha256": "abc"},
            "question": "is </script> escaped?",
            "benchmark_reference": "yes",
            "reference_annotation": {
                "visual_observability": None,
                "benchmark_reference_correctness": None,
                "required_answer_claims": [],
                "notes": "",
            },
            "candidate_answers": [],
        }
    ]
    form = _review_form(rows, "A", "bundle-1")
    assert b"REVIEW_FORM" not in form  # no source path or coordination metadata
    assert b"\\u003c/script>" in form
    assert b"source_model" not in form
    assert b"https://" not in form and b"http://" not in form
