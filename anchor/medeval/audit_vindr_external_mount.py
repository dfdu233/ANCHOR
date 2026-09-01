#!/usr/bin/env python3
"""Certify the immutable VinDr mount used by the v2 reader experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .audit_retrieval_split import read_rows
from .audit_vindr_download import inspect_one
from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "vindr-readonly-external-subset-audit-v1"


def mount_record(path: Path, mounts_path: Path = Path("/proc/mounts")) -> dict:
    resolved = path.resolve()
    candidates = []
    for line in mounts_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        mountpoint = Path(fields[1].replace("\\040", " ")).resolve()
        try:
            resolved.relative_to(mountpoint)
        except ValueError:
            continue
        candidates.append((len(mountpoint.parts), fields[0], mountpoint, fields[2], fields[3].split(",")))
    if not candidates:
        raise ValueError(f"no mount record contains {resolved}")
    _, device, mountpoint, filesystem, options = max(candidates, key=lambda row: row[0])
    return {
        "device": device,
        "mountpoint": str(mountpoint),
        "filesystem": filesystem,
        "options": options,
        "read_only": "ro" in options,
    }


def source_image_names(source_csv: Path) -> set[str]:
    with source_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_id" not in reader.fieldnames:
            raise ValueError("source CSV lacks image_id")
        names = {f"{str(row['image_id']).strip()}.dicom" for row in reader}
    if not names:
        raise ValueError("source CSV contains no image ids")
    return names


def selected_image_names(manifest: Path) -> set[str]:
    names = {
        f"{str(row.get('image_id', '')).strip()}.dicom"
        for row in read_rows(manifest)
        if str(row.get("image_id", "")).strip()
    }
    if not names:
        raise ValueError("reader manifest contains no image ids")
    return names


def audit(
    manifest: Path,
    source_csv: Path,
    image_root: Path,
    *,
    workers: int = 4,
    require_read_only: bool = True,
    mounts_path: Path = Path("/proc/mounts"),
) -> dict:
    selected = selected_image_names(manifest)
    source = source_image_names(source_csv)
    observed = {path.name for path in image_root.glob("*.dicom") if path.is_file()}
    missing_source = sorted(source - observed)
    extra_source = sorted(observed - source)
    missing_selected = sorted(selected - observed)
    unreferenced_selected = sorted(selected - source)
    mount = mount_record(image_root, mounts_path)

    inspected: list[dict] = []
    if not missing_selected and not unreferenced_selected:
        paths = (image_root / name for name in sorted(selected))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            inspected = list(pool.map(inspect_one, paths))
    invalid = [row for row in inspected if row.get("ok") is not True]
    content_lines = [
        f"{row['name']}\t{row['size']}\t{row['sha256']}"
        for row in inspected
        if row.get("ok") is True
    ]
    selected_fingerprint = hashlib.sha256("\n".join(content_lines).encode()).hexdigest()
    passed = bool(
        not missing_source
        and not extra_source
        and not missing_selected
        and not unreferenced_selected
        and not invalid
        and len(inspected) == len(selected)
        and (mount["read_only"] or not require_read_only)
    )
    return {
        "protocol_version": VERSION,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "source_csv": str(source_csv.resolve()),
        "source_csv_sha256": sha256_file(source_csv),
        "image_root": str(image_root.resolve()),
        "mount": mount,
        "read_only_required": require_read_only,
        "source_images": len(source),
        "observed_dicoms": len(observed),
        "selected_images": len(selected),
        "validated_selected_images": sum(row.get("ok") is True for row in inspected),
        "missing_source": missing_source,
        "extra_source": extra_source,
        "missing_selected": missing_selected,
        "unreferenced_selected": unreferenced_selected,
        "invalid_selected": invalid,
        "ordered_selected_dicom_sha256": selected_fingerprint if passed else None,
        "passed": passed,
        "completion_claim": (
            "the immutable mount exactly matches the official annotation image-id set, "
            "and every v2-selected DICOM has matching file-meta identity, complete "
            "native pixels or decodable encapsulated pixels, and a frozen ordered hash"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = audit(args.manifest, args.source_csv, args.image_root, workers=args.workers)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
