#!/usr/bin/env python3
"""Fail-closed verifier for CECD v3 offline reviewer archives."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from anchor.corrected_sgta.build_cecd_reviewer_deliveries_v1 import (
    CLINICAL_FIELDS,
    LANGUAGE_FIELDS,
    ROLES,
    SOURCE_VERSION,
    sha256_bytes,
    sha256_file,
)
from anchor.medeval.package_cecd_deliveries_v3 import (
    ALLOWED,
    PROFESSIONAL_ROLE,
    VERSION,
    v3_root,
)
from anchor.medeval.package_physician_oe_deliveries import canonical_json


FORBIDDEN = (
    "sealed_mapping",
    "selected_claims",
    "model_output",
    "private_provenance",
    "reader_votes",
    "transform_name",
)
VERIFICATION_VERSION = "cecd-reviewer-delivery-verification-v3.1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    handle = archive.extractfile(member)
    _require(handle is not None, f"cannot read archive member: {name}")
    return handle.read()


def verify_archive(
    *,
    pack_dir: Path,
    archive_path: Path,
    indexed: dict[str, Any],
    image_hash_cache: dict[Path, str],
) -> dict[str, Any]:
    role = str(indexed["role"])
    spec = ROLES[role]
    root = v3_root(role)
    _require(archive_path.is_file(), f"{role}: archive missing")
    _require(archive_path.name == f"{root}.tar.gz", f"{role}: archive filename mismatch")
    _require(sha256_file(archive_path) == indexed["archive_sha256"], f"{role}: archive hash mismatch")
    _require(archive_path.stat().st_size == indexed["archive_size_bytes"], f"{role}: archive size mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        _require(len(names) == len(set(names)), f"{role}: duplicate members")
        for member in members:
            pure = PurePosixPath(member.name)
            _require(member.isfile(), f"{role}: non-regular member")
            _require(not pure.is_absolute() and ".." not in pure.parts, f"{role}: unsafe path")
            _require(pure.parts and pure.parts[0] == root, f"{role}: wrong archive root")
            _require(not any(value in member.name.lower() for value in FORBIDDEN), f"{role}: forbidden member")

        manifest_name = f"{root}/DELIVERY_MANIFEST.json"
        form_name = f"{root}/REVIEW_FORM.html"
        schema_name = f"{root}/REVIEW_SCHEMA.json"
        sheet_name = f"{root}/{spec['sheet']}"
        inventory_name = f"{root}/IMAGE_SHA256SUMS"
        instructions_name = f"{root}/INSTRUCTIONS.md"
        for required in (
            manifest_name,
            form_name,
            schema_name,
            sheet_name,
            inventory_name,
            instructions_name,
        ):
            _require(required in names, f"{role}: missing {required}")
        manifest = json.loads(_member_bytes(archive, manifest_name))
        schema = json.loads(_member_bytes(archive, schema_name))
        form = _member_bytes(archive, form_name)
        sheet = _member_bytes(archive, sheet_name)
        inventory = _member_bytes(archive, inventory_name)
        instructions = _member_bytes(archive, instructions_name)
        source_sheet = (pack_dir / spec["sheet"]).read_bytes()
        source_manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
        _require(source_manifest.get("version") == SOURCE_VERSION, f"{role}: stale source manifest")
        _require(
            source_manifest.get("artifact_sha256", {}).get(spec["sheet"])
            == sha256_bytes(source_sheet),
            f"{role}: source sheet differs from frozen manifest",
        )
        _require(sheet == source_sheet, f"{role}: source CSV bytes changed")
        reader = csv.DictReader(io.StringIO(sheet.decode("utf-8"), newline=""))
        header, rows = list(reader.fieldnames or []), list(reader)

        _require(manifest.get("version") == VERSION, f"{role}: wrong manifest version")
        _require(manifest.get("source_version") == SOURCE_VERSION, f"{role}: wrong source version")
        _require(manifest.get("role") == role, f"{role}: role mismatch")
        _require(manifest.get("kind") == spec["kind"], f"{role}: kind mismatch")
        _require(
            manifest.get("professional_role") == PROFESSIONAL_ROLE[role],
            f"{role}: professional role mismatch",
        )
        for flag in (
            "sealed_mapping_in_archive",
            "model_outputs_in_archive",
            "clinical_or_language_labels_created",
        ):
            _require(manifest.get(flag) is False, f"{role}: unsafe manifest flag {flag}")
        _require(manifest.get("attestation_export_required") is True, f"{role}: attestation not required")
        _require(manifest.get("form_sha256") == sha256_bytes(form), f"{role}: form hash mismatch")
        _require(manifest.get("schema_sha256") == sha256_bytes(_member_bytes(archive, schema_name)), f"{role}: schema hash mismatch")
        _require(manifest.get("instructions_sha256") == sha256_bytes(instructions), f"{role}: instructions hash mismatch")
        _require(manifest["review_sheet"]["header"] == header, f"{role}: header mismatch")
        _require(manifest["review_sheet"]["rows"] == len(rows), f"{role}: row count mismatch")
        _require(manifest["review_sheet"]["sha256"] == sha256_bytes(sheet), f"{role}: sheet hash mismatch")
        expected_decisions = CLINICAL_FIELDS if spec["kind"] == "clinical" else LANGUAGE_FIELDS
        expected_allowed = {
            field: ALLOWED[field] for field in expected_decisions if field != "comments"
        }
        _require(
            schema.get("protocol_id") == SOURCE_VERSION
            and schema.get("delivery_version") == VERSION
            and schema.get("role") == role
            and schema.get("kind") == spec["kind"]
            and schema.get("professional_role") == PROFESSIONAL_ROLE[role]
            and schema.get("sheet") == spec["sheet"]
            and schema.get("header") == header
            and schema.get("decision_fields") == list(expected_decisions)
            and schema.get("allowed") == expected_allowed,
            f"{role}: schema mismatch",
        )
        _require(b"http://" not in form and b"https://" not in form, f"{role}: network URL in form")
        match = re.search(rb'<script id="seed" type="application/json">(.*?)</script>', form, re.DOTALL)
        _require(match is not None, f"{role}: form seed absent")
        seed = json.loads(match.group(1).decode("utf-8"))
        _require(seed.get("role") == role and seed.get("rows") == rows, f"{role}: form seed drift")
        _require(seed.get("header") == header, f"{role}: form header drift")

        image_names = sorted(name for name in names if name.startswith(f"{root}/images/"))
        expected_relative = sorted(
            {row[field] for row in rows for field in ("image_A", "image_B")}
            if spec["kind"] == "clinical"
            else []
        )
        if spec["kind"] == "clinical":
            actual_source_images = sorted(
                f"images/{path.name}" for path in (pack_dir / "images").glob("*.png")
            )
            _require(actual_source_images == expected_relative, f"{role}: source image closure mismatch")
        _require(image_names == [f"{root}/{relative}" for relative in expected_relative], f"{role}: image closure mismatch")
        lines: list[str] = []
        for relative in expected_relative:
            source = pack_dir / relative
            digest = image_hash_cache.get(source)
            if digest is None:
                digest = sha256_file(source)
                image_hash_cache[source] = digest
            archived = _member_bytes(archive, f"{root}/{relative}")
            _require(sha256_bytes(archived) == digest, f"{role}: changed image {relative}")
            lines.append(f"{digest}  {relative}\n")
        expected_inventory = "".join(lines).encode("utf-8")
        _require(inventory == expected_inventory, f"{role}: image inventory mismatch")
        _require(manifest["images"]["inventory_sha256"] == sha256_bytes(inventory), f"{role}: inventory hash mismatch")
        _require(manifest["images"]["count"] == len(image_names), f"{role}: image count mismatch")
        _require(manifest["archive_member_file_count"] == len(names), f"{role}: member count mismatch")
        _require(indexed["archive_member_file_count"] == len(names), f"{role}: indexed member count mismatch")
        visible_csvs = [name for name in names if name.endswith(".csv")]
        _require(visible_csvs == [sheet_name], f"{role}: reviewer sheet isolation failure")
    return {
        "role": role,
        "archive": archive_path.name,
        "archive_sha256": indexed["archive_sha256"],
        "archive_size_bytes": archive_path.stat().st_size,
        "members": len(names),
        "rows": len(rows),
        "images": len(image_names),
        "passed": True,
    }


def verify_deliveries(pack_dir: Path, delivery_dir: Path) -> dict[str, Any]:
    index_path = delivery_dir / "delivery_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    _require(index.get("version") == VERSION, "wrong delivery index version")
    _require(index.get("source_version") == SOURCE_VERSION, "wrong source version")
    _require(
        index.get("source_manifest_sha256") == sha256_file(pack_dir / "manifest.json"),
        "source manifest hash mismatch",
    )
    _require(index.get("clinical_or_language_labels_created") is False, "unsafe index flag")
    records = index.get("archives", [])
    _require(
        isinstance(records, list)
        and len(records) == len(ROLES)
        and {row.get("role") for row in records} == set(ROLES),
        "role set mismatch",
    )
    _require(
        len({row.get("archive") for row in records}) == len(ROLES),
        "duplicate archive records",
    )
    cache: dict[Path, str] = {}
    results = [
        verify_archive(
            pack_dir=pack_dir,
            archive_path=delivery_dir / row["archive"],
            indexed=row,
            image_hash_cache=cache,
        )
        for row in records
    ]
    return {
        "version": VERIFICATION_VERSION,
        "delivery_version": VERSION,
        "source_version": SOURCE_VERSION,
        "source_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
        "delivery_index_sha256": sha256_file(index_path),
        "verifier_sha256": sha256_file(Path(__file__)),
        "roles": results,
        "passed": all(row["passed"] for row in results),
        "clinical_or_language_labels_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--delivery-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_deliveries(args.pack_dir, args.delivery_dir)
    args.output.write_bytes(canonical_json(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
