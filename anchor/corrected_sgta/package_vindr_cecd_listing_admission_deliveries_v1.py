#!/usr/bin/env python3
"""Build and verify role-isolated VinDr listing-admission deliveries."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from corrected_sgta.build_cecd_reviewer_deliveries_v1 import (
    _write_deterministic_tar_gz,
    canonical_json,
    sha256_bytes,
    sha256_file,
)
from corrected_sgta.build_vindr_cecd_listing_admission_pack_v1 import (
    PROFESSIONAL_ROLE,
    ROLES,
    VERSION as SOURCE_VERSION,
)
from corrected_sgta.verify_vindr_cecd_listing_admission_pack_v1 import verify


VERSION = "vindr-cecd-listing-admission-role-delivery-v1"
VERIFICATION_VERSION = "vindr-cecd-listing-admission-role-delivery-verification-v1"
FORBIDDEN = (
    "sealed_mapping",
    "selected_images",
    "reader_votes",
    "sampling_stratum",
    "transform",
    "baseline_side",
    "model_output",
    "model_score",
)


def role_kind(role: str) -> str:
    return "clinical" if role.startswith("clinical_reviewer_") else "prompt"


def role_root(role: str) -> str:
    return f"vindr_cecd_listing_{role}_v1"


def role_sheet(role: str) -> str:
    return f"{role}.csv"


def build_role(pack_dir: Path, output_dir: Path, role: str) -> dict[str, Any]:
    if role not in ROLES:
        raise ValueError(role)
    kind = role_kind(role)
    root = role_root(role)
    sheet = pack_dir / role_sheet(role)
    instructions = pack_dir / (
        "CLINICAL_INSTRUCTIONS.md" if kind == "clinical" else "PROMPT_INSTRUCTIONS.md"
    )
    schema = pack_dir / "RETURN_SCHEMA.json"
    attestation = pack_dir / f"{role}.attestation.template.json"
    byte_members = {
        f"{root}/{sheet.name}": sheet.read_bytes(),
        f"{root}/INSTRUCTIONS.md": instructions.read_bytes(),
        f"{root}/RETURN_SCHEMA.json": schema.read_bytes(),
        f"{root}/{attestation.name}": attestation.read_bytes(),
    }
    file_members: dict[str, Path] = {}
    image_inventory = []
    if kind == "clinical":
        for path in sorted((pack_dir / "images").glob("*.png")):
            relative = f"images/{path.name}"
            file_members[f"{root}/{relative}"] = path
            image_inventory.append({"name": relative, "sha256": sha256_file(path)})
    delivery_manifest = {
        "version": VERSION,
        "source_version": SOURCE_VERSION,
        "source_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
        "role": role,
        "kind": kind,
        "professional_role": PROFESSIONAL_ROLE[role],
        "review_sheet": {"filename": sheet.name, "sha256": sha256_file(sheet)},
        "return_schema_sha256": sha256_file(schema),
        "attestation_template_sha256": sha256_file(attestation),
        "instructions_sha256": sha256_file(instructions),
        "images": {
            "included": kind == "clinical",
            "count": len(image_inventory),
            "inventory_sha256": sha256_bytes(canonical_json(image_inventory)),
        },
        "sealed_mapping_included": False,
        "source_truth_included": False,
        "other_reviewer_sheets_included": False,
        "model_outputs_or_scores_included": False,
        "human_decisions_created": False,
        "archive_member_file_count": len(byte_members) + len(file_members) + 1,
    }
    byte_members[f"{root}/DELIVERY_MANIFEST.json"] = canonical_json(delivery_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{root}.tar.gz"
    _write_deterministic_tar_gz(archive, byte_members, file_members)
    return {
        "role": role,
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_member_file_count": delivery_manifest["archive_member_file_count"],
    }


def package(pack_dir: Path, output_dir: Path) -> dict[str, Any]:
    verify(pack_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("delivery output must be a new empty directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [build_role(pack_dir, output_dir, role) for role in ROLES]
    index = {
        "version": VERSION,
        "source_version": SOURCE_VERSION,
        "source_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
        "archives": records,
        "human_decisions_created": False,
        "model_or_gpu_authorized": False,
    }
    (output_dir / "delivery_index.json").write_bytes(canonical_json(index))
    return index


def _member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    handle = archive.extractfile(name)
    if handle is None:
        raise RuntimeError(f"cannot read archive member {name}")
    return handle.read()


def verify_archive(
    pack_dir: Path, archive_path: Path, indexed: dict[str, Any]
) -> dict[str, Any]:
    role = str(indexed["role"])
    if role not in ROLES:
        raise RuntimeError(f"unknown indexed role {role}")
    kind, root = role_kind(role), role_root(role)
    if archive_path.name != f"{root}.tar.gz":
        raise RuntimeError(f"{role}: archive name drift")
    if sha256_file(archive_path) != indexed["archive_sha256"]:
        raise RuntimeError(f"{role}: archive hash drift")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError(f"{role}: duplicate archive members")
        for member in members:
            pure = PurePosixPath(member.name)
            if not member.isfile() or pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(f"{role}: unsafe archive member")
            if not pure.parts or pure.parts[0] != root:
                raise RuntimeError(f"{role}: wrong archive root")
            if any(term in member.name.lower() for term in FORBIDDEN):
                raise RuntimeError(f"{role}: forbidden archive member {member.name}")
        required = {
            f"{root}/DELIVERY_MANIFEST.json",
            f"{root}/INSTRUCTIONS.md",
            f"{root}/RETURN_SCHEMA.json",
            f"{root}/{role_sheet(role)}",
            f"{root}/{role}.attestation.template.json",
        }
        if not required <= set(names):
            raise RuntimeError(f"{role}: required delivery members missing")
        delivery = json.loads(_member_bytes(archive, f"{root}/DELIVERY_MANIFEST.json"))
        if delivery.get("version") != VERSION or delivery.get("role") != role:
            raise RuntimeError(f"{role}: delivery manifest drift")
        for flag in (
            "sealed_mapping_included",
            "source_truth_included",
            "other_reviewer_sheets_included",
            "model_outputs_or_scores_included",
            "human_decisions_created",
        ):
            if delivery.get(flag) is not False:
                raise RuntimeError(f"{role}: unsafe delivery flag {flag}")
        sheet_bytes = _member_bytes(archive, f"{root}/{role_sheet(role)}")
        if sheet_bytes != (pack_dir / role_sheet(role)).read_bytes():
            raise RuntimeError(f"{role}: reviewer sheet bytes drift")
        visible_csv = [name for name in names if name.endswith(".csv")]
        if visible_csv != [f"{root}/{role_sheet(role)}"]:
            raise RuntimeError(f"{role}: reviewer sheet isolation failed")
        image_names = sorted(name for name in names if name.startswith(f"{root}/images/"))
        expected_images = (
            sorted(f"{root}/images/{path.name}" for path in (pack_dir / "images").glob("*.png"))
            if kind == "clinical"
            else []
        )
        if image_names != expected_images:
            raise RuntimeError(f"{role}: image closure drift")
        for name, source in zip(image_names, sorted((pack_dir / "images").glob("*.png"))):
            if sha256_bytes(_member_bytes(archive, name)) != sha256_file(source):
                raise RuntimeError(f"{role}: image bytes drift")
        if delivery["archive_member_file_count"] != len(names):
            raise RuntimeError(f"{role}: member count drift")
        if indexed["archive_member_file_count"] != len(names):
            raise RuntimeError(f"{role}: indexed member count drift")
    return {
        "role": role,
        "passed": True,
        "members": len(names),
        "images": len(image_names),
        "archive_sha256": indexed["archive_sha256"],
    }


def verify_deliveries(pack_dir: Path, delivery_dir: Path) -> dict[str, Any]:
    verify(pack_dir)
    index_path = delivery_dir / "delivery_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("version") != VERSION or index.get("source_version") != SOURCE_VERSION:
        raise RuntimeError("delivery index version drift")
    if index.get("source_manifest_sha256") != sha256_file(pack_dir / "manifest.json"):
        raise RuntimeError("delivery source manifest hash drift")
    if index.get("human_decisions_created") is not False:
        raise RuntimeError("delivery index claims created human decisions")
    records = index.get("archives")
    if not isinstance(records, list) or {row.get("role") for row in records} != set(ROLES):
        raise RuntimeError("delivery role closure drift")
    results = [
        verify_archive(pack_dir, delivery_dir / row["archive"], row) for row in records
    ]
    return {
        "version": VERIFICATION_VERSION,
        "passed": all(row["passed"] for row in results),
        "status": "four_role_delivery_skeleton_verified_awaiting_humans",
        "source_manifest_sha256": index["source_manifest_sha256"],
        "delivery_index_sha256": sha256_file(index_path),
        "roles": results,
        "sealed_or_truth_material_delivered": False,
        "human_decisions_created": False,
        "model_or_gpu_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--pack-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    check = subparsers.add_parser("verify")
    check.add_argument("--pack-dir", type=Path, required=True)
    check.add_argument("--delivery-dir", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = package(args.pack_dir, args.output_dir)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        result = verify_deliveries(args.pack_dir, args.delivery_dir)
        args.output.write_bytes(canonical_json(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
