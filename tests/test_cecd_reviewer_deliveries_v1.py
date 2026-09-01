import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

from anchor.corrected_sgta.build_cecd_reviewer_deliveries_v1 import (
    ROLES,
    SOURCE_VERSION,
    build_deliveries,
)
from anchor.corrected_sgta.verify_cecd_reviewer_delivery_v1 import verify_deliveries


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv_bytes(header: list[str], rows: list[list[str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _source_pack(tmp_path: Path) -> Path:
    pack = tmp_path / "pack"
    images = pack / "images"
    images.mkdir(parents=True)

    clinical_header = [
        "pair_id",
        "image_A",
        "image_B",
        "finding",
        "support_state_same_supported_refuted_undetermined",
        "lesion_visibility",
        "clinically_interchangeable",
        "unable_to_judge",
        "comments",
    ]
    clinical_rows = []
    for index in range(252):
        pair = f"{index:016x}"
        a = f"images/{pair}_A.png"
        b = f"images/{pair}_B.png"
        (pack / a).write_bytes(f"png-A-{index}".encode())
        (pack / b).write_bytes(f"png-B-{index}".encode())
        clinical_rows.append([pair, a, b, "finding", "", "", "", "", ""])
    clinical = _csv_bytes(clinical_header, clinical_rows)

    language_header = [
        "item_id",
        "wording_A",
        "wording_B",
        "same_clinical_proposition",
        "same_speech_act",
        "same_certainty_demand",
        "same_answer_space",
        "comments",
    ]
    language_rows = [[f"item-{index}", "A", "B", "", "", "", "", ""] for index in range(8)]
    template = _csv_bytes(language_header, language_rows)
    language = _csv_bytes(language_header, list(reversed(language_rows)))

    payloads = {
        "clinical_reviewer_1.csv": clinical,
        "clinical_reviewer_2.csv": clinical,
        "clinical_template_reviewer.csv": template,
        "language_annotator.csv": language,
    }
    for name, data in payloads.items():
        (pack / name).write_bytes(data)
    (pack / "manifest.json").write_text(
        json.dumps(
            {
                "version": SOURCE_VERSION,
                "artifact_sha256": {name: _sha(data) for name, data in payloads.items()},
            }
        ),
        encoding="utf-8",
    )
    return pack


def test_deliveries_are_deterministic_frozen_role_isolated_and_verifiable(tmp_path: Path) -> None:
    pack = _source_pack(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_index = build_deliveries(pack, first)
    second_index = build_deliveries(pack, second)
    assert [row["archive_sha256"] for row in first_index["archives"]] == [
        row["archive_sha256"] for row in second_index["archives"]
    ]

    result = verify_deliveries(pack, first)
    assert result["passed"] is True
    assert result["archives_verified"] == 4

    for role, spec in ROLES.items():
        archive_path = first / f"{spec['root']}.tar.gz"
        with tarfile.open(archive_path, "r:gz") as archive:
            names = archive.getnames()
            csvs = [name for name in names if name.endswith(".csv")]
            assert csvs == [f"{spec['root']}/{spec['sheet']}"]
            assert not any("sealed" in name or "selected_claims" in name for name in names)
            archived_sheet = archive.extractfile(csvs[0]).read()
            assert archived_sheet == (pack / spec["sheet"]).read_bytes()
            png_count = sum(name.endswith(".png") for name in names)
            assert png_count == (504 if spec["images"] else 0)
