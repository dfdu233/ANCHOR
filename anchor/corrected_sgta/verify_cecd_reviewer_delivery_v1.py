"""Independent safety and integrity verifier for CECD reviewer deliveries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from anchor.corrected_sgta.build_cecd_reviewer_deliveries_v1 import (
        CLINICAL_FIELDS,
        LANGUAGE_FIELDS,
        ROLES,
        SOURCE_VERSION,
        VERSION,
        canonical_json,
        sha256_bytes,
        sha256_file,
    )
except ModuleNotFoundError:  # Support direct ``python path/to/script.py`` execution.
    from build_cecd_reviewer_deliveries_v1 import (  # type: ignore[no-redef]
        CLINICAL_FIELDS,
        LANGUAGE_FIELDS,
        ROLES,
        SOURCE_VERSION,
        VERSION,
        canonical_json,
        sha256_bytes,
        sha256_file,
    )


FORBIDDEN_MEMBER_FRAGMENTS = (
    "sealed_mapping",
    "selected_claims",
    "model_output",
    "private_provenance",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _member_bytes(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = archive.extractfile(member)
    _require(handle is not None, f"cannot read archive member: {member.name}")
    return handle.read()


def _inventory_hash(entries: list[tuple[str, str]]) -> str:
    payload = "".join(f"{digest}  {name}\n" for name, digest in sorted(entries))
    return sha256_bytes(payload.encode("utf-8"))


def _blank_sheet(data: bytes, fields: tuple[str, ...], expected_rows: int, role: str) -> None:
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))
    _require(len(rows) == expected_rows, f"{role}: wrong reviewer row count")
    for row_number, row in enumerate(rows, start=2):
        for field in fields:
            _require(field in row, f"{role}: missing decision field {field}")
            _require(not row[field].strip(), f"{role}: nonblank decision at row {row_number}, {field}")


def verify_archive(pack_dir: Path, archive_path: Path, role: str) -> dict[str, Any]:
    spec = ROLES[role]
    root = spec["root"]
    source_manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    _require(source_manifest.get("version") == SOURCE_VERSION, "wrong source pack version")

    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        _require(len(names) == len(set(names)), f"{role}: duplicate archive member")
        _require(all(member.isfile() for member in members), f"{role}: archive contains non-regular member")
        for name in names:
            pure = PurePosixPath(name)
            _require(not pure.is_absolute() and ".." not in pure.parts, f"{role}: unsafe path {name}")
            _require(pure.parts and pure.parts[0] == root, f"{role}: wrong or multiple archive roots")
            lowered = name.lower()
            _require(
                not any(fragment in lowered for fragment in FORBIDDEN_MEMBER_FRAGMENTS),
                f"{role}: forbidden member {name}",
            )

        member_by_name = {member.name: member for member in members}
        manifest_name = f"{root}/DELIVERY_MANIFEST.json"
        _require(manifest_name in member_by_name, f"{role}: delivery manifest absent")
        manifest_bytes = _member_bytes(archive, member_by_name[manifest_name])
        manifest = json.loads(manifest_bytes)
        _require(manifest.get("version") == VERSION, f"{role}: wrong delivery version")
        _require(manifest.get("role") == role, f"{role}: manifest role mismatch")
        _require(manifest.get("blinded_role_isolation") is True, f"{role}: isolation flag missing")

        sheet_name = f"{root}/{spec['sheet']}"
        instruction_name = f"{root}/INSTRUCTIONS.md"
        expected_names = {sheet_name, instruction_name, manifest_name}
        if spec["images"]:
            source_sheet = (pack_dir / spec["sheet"]).read_bytes()
            rows = list(csv.DictReader(io.StringIO(source_sheet.decode("utf-8"), newline="")))
            relative_images = sorted({row[key] for row in rows for key in ("image_A", "image_B")})
            expected_names.update(f"{root}/{name}" for name in relative_images)
        else:
            relative_images = []
        _require(set(names) == expected_names, f"{role}: missing or extra role-visible files")

        sheet_bytes = _member_bytes(archive, member_by_name[sheet_name])
        source_sheet_bytes = (pack_dir / spec["sheet"]).read_bytes()
        _require(sheet_bytes == source_sheet_bytes, f"{role}: frozen CSV bytes changed")
        _require(
            sha256_bytes(sheet_bytes) == source_manifest["artifact_sha256"][spec["sheet"]],
            f"{role}: source CSV hash does not match frozen manifest",
        )
        fields = CLINICAL_FIELDS if spec["kind"] == "clinical" else LANGUAGE_FIELDS
        expected_rows = 252 if spec["kind"] == "clinical" else 8
        _blank_sheet(sheet_bytes, fields, expected_rows, role)

        instruction_bytes = _member_bytes(archive, member_by_name[instruction_name])
        _require(
            sha256_bytes(instruction_bytes) == manifest["instructions"]["sha256"],
            f"{role}: instruction hash mismatch",
        )
        instruction_text = instruction_bytes.decode("utf-8").lower()
        _require(spec["sheet"].lower() in instruction_text, f"{role}: instructions are not self-contained")
        _require(
            not any(fragment in instruction_text for fragment in FORBIDDEN_MEMBER_FRAGMENTS),
            f"{role}: instructions disclose forbidden material",
        )

        image_hashes: list[tuple[str, str]] = []
        for relative in relative_images:
            archived = _member_bytes(archive, member_by_name[f"{root}/{relative}"])
            archived_hash = sha256_bytes(archived)
            _require(archived_hash == sha256_file(pack_dir / relative), f"{role}: changed PNG {relative}")
            image_hashes.append((relative, archived_hash))

        _require(manifest["review_sheet"]["filename"] == spec["sheet"], f"{role}: sheet name mismatch")
        _require(manifest["review_sheet"]["rows"] == expected_rows, f"{role}: manifest row count mismatch")
        _require(manifest["review_sheet"]["sha256"] == sha256_bytes(sheet_bytes), f"{role}: sheet hash mismatch")
        _require(manifest["images"]["included"] is bool(spec["images"]), f"{role}: image inclusion mismatch")
        _require(manifest["images"]["count"] == len(relative_images), f"{role}: image count mismatch")
        _require(
            manifest["images"]["inventory_sha256"] == _inventory_hash(image_hashes),
            f"{role}: image inventory mismatch",
        )
        _require(manifest["archive_member_file_count"] == len(names), f"{role}: member count mismatch")
        _require(manifest["payload_file_count"] + 1 == len(names), f"{role}: payload count mismatch")

        visible_csvs = [name for name in names if name.lower().endswith(".csv")]
        _require(visible_csvs == [sheet_name], f"{role}: another reviewer's sheet is visible")

    return {
        "version": VERSION,
        "role": role,
        "passed": True,
        "archive": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_member_file_count": len(names),
        "review_sheet_bytes_match_frozen_source": True,
        "review_sheet_blank": True,
        "role_isolation_passed": True,
        "unsafe_or_forbidden_members": 0,
        "images_verified": len(relative_images),
    }


def verify_deliveries(pack_dir: Path, delivery_dir: Path) -> dict[str, Any]:
    index_path = delivery_dir / "delivery_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    _require(index.get("version") == VERSION, "wrong delivery index version")
    index_by_role = {row["role"]: row for row in index.get("archives", [])}
    _require(set(index_by_role) == set(ROLES), "delivery index has missing or extra roles")

    results = []
    expected_archives = set()
    for role, spec in ROLES.items():
        archive_path = delivery_dir / f"{spec['root']}.tar.gz"
        expected_archives.add(archive_path.name)
        result = verify_archive(pack_dir, archive_path, role)
        indexed = index_by_role[role]
        _require(indexed["archive"] == archive_path.name, f"{role}: indexed archive name mismatch")
        _require(indexed["archive_sha256"] == result["archive_sha256"], f"{role}: indexed archive hash mismatch")
        _require(
            indexed["archive_member_file_count"] == result["archive_member_file_count"],
            f"{role}: indexed member count mismatch",
        )
        results.append(result)

    actual_archives = {path.name for path in delivery_dir.glob("*.tar.gz")}
    _require(actual_archives == expected_archives, "delivery directory has missing or extra archives")
    return {
        "version": VERSION,
        "passed": True,
        "archives_verified": len(results),
        "roles": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--delivery-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_deliveries(args.pack_dir, args.delivery_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
