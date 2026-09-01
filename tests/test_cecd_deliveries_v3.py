import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from anchor.corrected_sgta.build_cecd_reviewer_deliveries_v1 import (
    CLINICAL_FIELDS,
    LANGUAGE_FIELDS,
    ROLES,
    SOURCE_VERSION,
)
from anchor.medeval.package_cecd_deliveries_v3 import (
    PROFESSIONAL_ROLE,
    package_deliveries,
    v3_root,
)
from anchor.medeval.validate_cecd_returns_v3 import validate_all, validate_return
from anchor.medeval.verify_cecd_deliveries_v3 import verify_deliveries


def _csv_bytes(header: list[str], rows: list[list[str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _source_pack(tmp_path: Path) -> Path:
    pack = tmp_path / "pack"
    (pack / "images").mkdir(parents=True)
    clinical_header = [
        "pair_id",
        "image_A",
        "image_B",
        "finding",
        *CLINICAL_FIELDS,
    ]
    clinical_rows = []
    for index in range(252):
        pair = f"pair-{index:03d}"
        a, b = f"images/{pair}-A.png", f"images/{pair}-B.png"
        (pack / a).write_bytes(f"A-{index}".encode())
        (pack / b).write_bytes(f"B-{index}".encode())
        clinical_rows.append([pair, a, b, "finding", "", "", "", "", ""])
    clinical = _csv_bytes(clinical_header, clinical_rows)
    language_header = ["item_id", "wording_A", "wording_B", *LANGUAGE_FIELDS]
    language_rows = [
        [f"item-{index}", f"wording A {index}", f"wording B {index}", "", "", "", "", ""]
        for index in range(8)
    ]
    language = _csv_bytes(language_header, language_rows)
    payloads = {
        "clinical_reviewer_1.csv": clinical,
        "clinical_reviewer_2.csv": clinical,
        "clinical_template_reviewer.csv": language,
        "language_annotator.csv": language,
    }
    for name, payload in payloads.items():
        (pack / name).write_bytes(payload)
    manifest = {
        "version": SOURCE_VERSION,
        "artifact_sha256": {
            name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
        },
    }
    (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pack


def _completed(pack: Path, role: str, output: Path) -> None:
    spec = ROLES[role]
    with (pack / spec["sheet"]).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header, rows = list(reader.fieldnames or []), list(reader)
    if spec["kind"] == "clinical":
        values = {
            "support_state_same_supported_refuted_undetermined": "yes",
            "lesion_visibility": "unchanged",
            "clinically_interchangeable": "yes",
            "unable_to_judge": "no",
            "comments": "",
        }
    else:
        values = {
            "same_clinical_proposition": "yes",
            "same_speech_act": "yes",
            "same_certainty_demand": "yes",
            "same_answer_space": "yes",
            "comments": "",
        }
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\r\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **values})


def _attestation(role: str, reviewer_id: str, output: Path) -> None:
    output.write_text(
        json.dumps(
            {
                "protocol_id": SOURCE_VERSION,
                "review_role": role,
                "reviewer": {
                    "reviewer_id": reviewer_id,
                    "professional_role": PROFESSIONAL_ROLE[role],
                    "independent_review": True,
                    "blinded_to_sealed_mapping": True,
                    "completed_at_utc": "2026-08-03T00:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )


def test_v3_deliveries_are_deterministic_isolated_and_fail_closed(tmp_path: Path) -> None:
    pack = _source_pack(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    first_index = package_deliveries(pack, first)
    second_index = package_deliveries(pack, second)
    assert [row["archive_sha256"] for row in first_index["archives"]] == [
        row["archive_sha256"] for row in second_index["archives"]
    ]
    result = verify_deliveries(pack, first)
    assert result["passed"] is True
    assert {row["role"] for row in result["roles"]} == set(ROLES)
    for record in first_index["archives"]:
        root = v3_root(record["role"])
        with tarfile.open(first / record["archive"], "r:gz") as archive:
            names = archive.getnames()
            form = archive.extractfile(f"{root}/REVIEW_FORM.html").read()
            instructions = archive.extractfile(f"{root}/INSTRUCTIONS.md").read()
        assert [name for name in names if name.endswith(".csv")] == [
            f"{root}/{ROLES[record['role']]['sheet']}"
        ]
        assert b"http://" not in form and b"https://" not in form
        completed_name = ROLES[record["role"]]["sheet"].replace(
            ".csv", ".completed.csv"
        )
        assert completed_name.encode() in instructions
        assert b"Extract the entire assigned archive" in instructions
        assert not any("sealed_mapping" in name or "selected_claims" in name for name in names)


def test_v3_returns_require_frozen_rows_valid_decisions_and_four_distinct_attestations(
    tmp_path: Path,
) -> None:
    pack = _source_pack(tmp_path)
    completed, attestations = {}, {}
    for index, role in enumerate(ROLES):
        completed[role] = tmp_path / f"{role}.completed.csv"
        attestations[role] = tmp_path / f"{role}.attestation.json"
        _completed(pack, role, completed[role])
        _attestation(role, f"reviewer-{index}", attestations[role])
    result = validate_all(pack_dir=pack, completed=completed, attestations=attestations)
    assert result["status"] == "four_independent_returns_validated"
    assert result["clinical_or_language_labels_synthesized"] is False

    rows = list(csv.DictReader(completed["clinical_reviewer_1"].open(newline="", encoding="utf-8")))
    rows[0]["image_A"] = "images/tampered.png"
    with completed["clinical_reviewer_1"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="immutable field changed"):
        validate_return(
            pack_dir=pack,
            role="clinical_reviewer_1",
            completed_path=completed["clinical_reviewer_1"],
            attestation_path=attestations["clinical_reviewer_1"],
        )

    _completed(pack, "clinical_reviewer_1", completed["clinical_reviewer_1"])
    _attestation("language_reviewer", "reviewer-0", attestations["language_reviewer"])
    with pytest.raises(ValueError, match="distinct reviewer IDs"):
        validate_all(pack_dir=pack, completed=completed, attestations=attestations)
