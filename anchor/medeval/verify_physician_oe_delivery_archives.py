#!/usr/bin/env python3
"""Fail-closed verification for role-isolated physician-OE review archives."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from .package_physician_oe_deliveries import VERSION, sha256_bytes, sha256_file


FORBIDDEN_NAMES = ("private", "mapping", "model_score", "unblind")
FORBIDDEN_JSON_KEYS = {
    "source_model",
    "source_method",
    "model_id",
    "model_name",
    "method",
    "method_id",
    "method_name",
    "private_mapping",
}


def _forbidden_json_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_JSON_KEYS:
                found.append(child_prefix)
            found.extend(_forbidden_json_keys(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_json_keys(child, f"{prefix}[{index}]"))
    return found


def verify_archive(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if sha256_file(path) != record["archive_sha256"]:
        errors.append("archive SHA-256 mismatch")
    root = record["root"]
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            errors.append("duplicate archive members")
        if len(names) != record["archive_member_file_count"]:
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
                errors.append(f"unsafe archive member: {member.name}")
            lowered = member.name.lower()
            if any(token in lowered for token in FORBIDDEN_NAMES):
                errors.append(f"forbidden member name: {member.name}")
        manifest_name = f"{root}/DELIVERY_MANIFEST.json"
        if manifest_name not in names:
            errors.append("missing delivery manifest")
            manifest = {}
        else:
            extracted = archive.extractfile(manifest_name)
            manifest = json.loads(extracted.read()) if extracted is not None else {}
        if manifest.get("version") != VERSION:
            errors.append("wrong internal manifest version")
        if manifest.get("reviewer_role") != record["role"]:
            errors.append("role mismatch")
        if manifest.get("private_mapping_in_archive") is not False:
            errors.append("private mapping blindness not certified")
        if manifest.get("method_identity_in_archive") is not False:
            errors.append("method blindness not certified")
        if manifest.get("unblinding_authorized") is not False:
            errors.append("archive incorrectly authorizes unblinding")
        form_name = f"{root}/REVIEW_FORM.html"
        if form_name not in names:
            errors.append("offline review form missing")
        else:
            extracted_form = archive.extractfile(form_name)
            form_bytes = extracted_form.read() if extracted_form is not None else b""
            if sha256_bytes(form_bytes) != manifest.get("review_form", {}).get("sha256"):
                errors.append("offline review form hash mismatch")
            if manifest.get("review_form", {}).get("offline") is not True:
                errors.append("review form is not certified offline")
            if b"http://" in form_bytes or b"https://" in form_bytes:
                errors.append("review form contains a network URL")
        review_name = f"{root}/reviewer_{record['role']}.blinded.jsonl"
        other_role = "B" if record["role"] == "A" else "A"
        if review_name not in names:
            errors.append("assigned reviewer JSONL missing")
            review_rows = []
        else:
            extracted_review = archive.extractfile(review_name)
            review_bytes = extracted_review.read() if extracted_review is not None else b""
            if sha256_bytes(review_bytes) != manifest.get("review_jsonl", {}).get("sha256"):
                errors.append("review JSONL hash mismatch")
            try:
                review_rows = [
                    json.loads(line)
                    for line in review_bytes.decode("utf-8").splitlines()
                    if line.strip()
                ]
            except (UnicodeDecodeError, json.JSONDecodeError):
                review_rows = []
                errors.append("review JSONL is not valid UTF-8 JSONL")
            leaked_keys = _forbidden_json_keys(review_rows)
            if leaked_keys:
                errors.append(f"method/private JSON fields leaked: {leaked_keys[:3]}")
            if any(row.get("reviewer_slot") != record["role"] for row in review_rows):
                errors.append("review JSONL contains the wrong reviewer slot")
            if form_name in names:
                match = re.search(
                    rb'<script id="seed" type="application/json">(.*?)</script>',
                    form_bytes,
                    flags=re.DOTALL,
                )
                if match is None:
                    errors.append("review form has no embedded blinded seed")
                else:
                    try:
                        form_rows = json.loads(match.group(1).decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        errors.append("review form seed is invalid JSON")
                    else:
                        if form_rows != review_rows:
                            errors.append("review form seed differs from frozen reviewer JSONL")
        if any(f"reviewer_{other_role}" in name for name in names):
            errors.append("other reviewer sheet leaked")
        image_names = sorted(name for name in names if name.startswith(f"{root}/images/"))
        if len(image_names) != manifest.get("images", {}).get("count"):
            errors.append("image count differs from manifest")
        inventory_name = f"{root}/IMAGE_SHA256SUMS"
        if inventory_name not in names:
            errors.append("image inventory missing")
        else:
            extracted = archive.extractfile(inventory_name)
            inventory = extracted.read() if extracted is not None else b""
            if sha256_bytes(inventory) != manifest.get("images", {}).get("inventory_sha256"):
                errors.append("image inventory hash mismatch")
            expected_lines = []
            for name in image_names:
                extracted_image = archive.extractfile(name)
                data = extracted_image.read() if extracted_image is not None else b""
                expected_lines.append(
                    f"{sha256_bytes(data)}  images/{PurePosixPath(name).name}\n"
                )
            if inventory != "".join(sorted(expected_lines)).encode("utf-8"):
                errors.append("image bytes do not match inventory")
        referenced_images = {
            f"{root}/images/{row.get('image', {}).get('relative_path', '')}"
            for row in review_rows
        }
        if referenced_images != set(image_names):
            errors.append("review JSONL image references differ from archived images")
    return {"archive": path.name, "passed": not errors, "errors": errors}


def verify_delivery_dir(directory: Path) -> dict[str, Any]:
    index_path = directory / "delivery_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if index.get("version") != VERSION:
        errors.append("wrong delivery index version")
    records = index.get("archives", [])
    if {record.get("role") for record in records} != {"A", "B"}:
        errors.append("delivery index must contain exactly roles A and B")
    results = [
        verify_archive(directory / record["archive"], record) for record in records
    ]
    passed = not errors and len(results) == 2 and all(result["passed"] for result in results)
    return {
        "version": VERSION,
        "delivery_index_sha256": sha256_file(index_path),
        "passed": passed,
        "errors": errors,
        "archives": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_delivery_dir(args.delivery_dir)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
