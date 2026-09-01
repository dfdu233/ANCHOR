#!/usr/bin/env python3
"""Fail-closed verification for Specificity Ratchet v3 delivery archives."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from anchor.medeval.package_physician_oe_deliveries import sha256_bytes, sha256_file
from anchor.medeval.package_specificity_ratchet_deliveries import VERSION


FORBIDDEN = ("provenance", "adjudication", "source_model", "question_id", "private")


def verify_archive(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not path.is_file() or sha256_file(path) != record.get("archive_sha256"):
        errors.append("archive SHA-256 mismatch")
        return {"archive": path.name, "passed": False, "errors": errors}
    root = str(record["root"])
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            errors.append("duplicate archive members")
        if len(names) != record.get("archive_member_file_count"):
            errors.append("archive member count mismatch")
        for member in members:
            parts = PurePosixPath(member.name).parts
            if (
                not member.isfile()
                or not parts
                or parts[0] != root
                or member.name.startswith("/")
                or ".." in parts
            ):
                errors.append(f"unsafe member: {member.name}")
            lowered = member.name.lower()
            if any(token in lowered for token in FORBIDDEN):
                errors.append(f"forbidden member name: {member.name}")
        manifest_name = f"{root}/DELIVERY_MANIFEST.json"
        if manifest_name not in names:
            errors.append("missing internal manifest")
            manifest: dict[str, Any] = {}
        else:
            handle = archive.extractfile(manifest_name)
            manifest = json.loads(handle.read()) if handle else {}
        if manifest.get("version") != VERSION:
            errors.append("internal version mismatch")
        if manifest.get("reviewer_role") != record.get("role"):
            errors.append("reviewer role mismatch")
        for flag in ("private_provenance_in_archive", "model_identity_in_archive", "clinical_labels_created"):
            if manifest.get(flag) is not False:
                errors.append(f"unsafe manifest flag: {flag}")
        role = record.get("role")
        csv_name = f"{root}/annotations.reviewer_{role}.csv"
        other_name = f"annotations.reviewer_{2 if role == 1 else 1}.csv"
        if csv_name not in names or any(name.endswith(other_name) for name in names):
            errors.append("reviewer CSV isolation failure")
            csv_rows: list[dict[str, str]] = []
            header: list[str] = []
        else:
            handle = archive.extractfile(csv_name)
            csv_bytes = handle.read() if handle else b""
            if sha256_bytes(csv_bytes) != manifest.get("csv", {}).get("sha256"):
                errors.append("reviewer CSV hash mismatch")
            reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
            header, csv_rows = list(reader.fieldnames or []), list(reader)
            if header != manifest.get("csv", {}).get("header"):
                errors.append("reviewer CSV header mismatch")
            if len(csv_rows) != manifest.get("csv", {}).get("rows"):
                errors.append("reviewer CSV row count mismatch")
        form_name = f"{root}/REVIEW_FORM.html"
        if form_name not in names:
            errors.append("missing offline form")
            form = b""
        else:
            handle = archive.extractfile(form_name)
            form = handle.read() if handle else b""
            if sha256_bytes(form) != manifest.get("review_form", {}).get("sha256"):
                errors.append("review form hash mismatch")
            if b"http://" in form or b"https://" in form:
                errors.append("review form contains network URL")
            match = re.search(
                rb'<script id="seed" type="application/json">(.*?)</script>',
                form,
                re.DOTALL,
            )
            if match is None:
                errors.append("review form seed missing")
            else:
                try:
                    seed = json.loads(match.group(1).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    errors.append("review form seed invalid")
                else:
                    if seed.get("header") != header or seed.get("rows") != csv_rows:
                        errors.append("review form seed differs from frozen CSV")
                    serialized = json.dumps(seed).lower()
                    if any(token in serialized for token in ("source_model", "question_id", "private_provenance")):
                        errors.append("private/model field leaked in form seed")
        image_names = sorted(name for name in names if name.startswith(f"{root}/test_images/"))
        if len(image_names) != manifest.get("images", {}).get("count"):
            errors.append("image count mismatch")
        inventory_name = f"{root}/IMAGE_SHA256SUMS"
        if inventory_name not in names:
            errors.append("missing image inventory")
        else:
            handle = archive.extractfile(inventory_name)
            inventory = handle.read() if handle else b""
            if sha256_bytes(inventory) != manifest.get("images", {}).get("inventory_sha256"):
                errors.append("image inventory hash mismatch")
            expected = []
            for name in image_names:
                handle = archive.extractfile(name)
                data = handle.read() if handle else b""
                expected.append(f"{sha256_bytes(data)}  test_images/{PurePosixPath(name).name}\n")
            if inventory != "".join(expected).encode("utf-8"):
                errors.append("image bytes differ from inventory")
        referenced = {f"{root}/{row.get('image_relpath', '')}" for row in csv_rows}
        if referenced != set(image_names):
            errors.append("CSV/image closure mismatch")
    return {"archive": path.name, "passed": not errors, "errors": errors}


def verify_delivery_dir(directory: Path) -> dict[str, Any]:
    index_path = directory / "delivery_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    errors = []
    if index.get("version") != VERSION:
        errors.append("delivery index version mismatch")
    records = index.get("archives", [])
    if {record.get("role") for record in records} != {1, 2}:
        errors.append("delivery index must contain roles 1 and 2")
    results = [verify_archive(directory / record["archive"], record) for record in records]
    return {
        "version": VERSION,
        "delivery_index_sha256": sha256_file(index_path),
        "passed": not errors and len(results) == 2 and all(row["passed"] for row in results),
        "errors": errors,
        "archives": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_delivery_dir(args.delivery_dir)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
