#!/usr/bin/env python3
"""Verify that a selective VinDr download is complete and parseable.

The audit is deliberately independent of wget's exit status.  A filename can
exist while a connection is still writing it, so file counts alone are not an
admissible completion criterion for the formal reader-grounded experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "vindr-selective-dicom-audit-v2"


def expected_names(urls: list[str]) -> list[str]:
    names = [Path(urlparse(url.strip()).path).name for url in urls if url.strip()]
    if not names or any(not name.endswith(".dicom") for name in names):
        raise ValueError("image URL manifest must contain DICOM URLs")
    if len(set(names)) != len(names):
        raise ValueError("image URL manifest contains duplicate DICOM filenames")
    return sorted(names)


def inspect_one(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"name": path.name, "size": path.stat().st_size}
    try:
        import pydicom
        from pydicom.encaps import generate_frames

        # VinDr removes the dataset-level SOPInstanceUID.  Its file-meta
        # MediaStorageSOPInstanceUID is the anonymized image id (and filename),
        # but is intentionally not a standards-conforming dotted UID.  Ignore
        # only that known warning; all structural and pixel checks remain hard
        # failures.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Invalid value for VR UI:.*")
            dataset = pydicom.dcmread(path, defer_size="1 KB", force=False)
            file_meta = dataset.file_meta
            transfer_syntax = getattr(file_meta, "TransferSyntaxUID", None)
            storage_class = str(getattr(file_meta, "MediaStorageSOPClassUID", ""))
            storage_instance = str(getattr(file_meta, "MediaStorageSOPInstanceUID", ""))
        if transfer_syntax is None or not transfer_syntax.is_transfer_syntax:
            raise ValueError("missing or invalid TransferSyntaxUID")
        if not storage_class:
            raise ValueError("missing MediaStorageSOPClassUID")
        if storage_instance != path.stem:
            raise ValueError(
                "MediaStorageSOPInstanceUID does not match anonymized filename"
            )

        rows = int(getattr(dataset, "Rows", 0))
        columns = int(getattr(dataset, "Columns", 0))
        samples = int(getattr(dataset, "SamplesPerPixel", 0))
        frames = int(getattr(dataset, "NumberOfFrames", 1))
        bits_allocated = int(getattr(dataset, "BitsAllocated", 0))
        photometric = str(getattr(dataset, "PhotometricInterpretation", ""))
        if rows <= 0 or columns <= 0:
            raise ValueError("invalid image dimensions")
        if samples != 1 or frames != 1:
            raise ValueError("VinDr CXR audit requires a single grayscale frame")
        if bits_allocated not in {8, 16}:
            raise ValueError("unexpected BitsAllocated for VinDr CXR")
        if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
            raise ValueError("unexpected PhotometricInterpretation for VinDr CXR")

        pixel_data = dataset.get("PixelData")
        if not isinstance(pixel_data, bytes) or not pixel_data:
            raise ValueError("missing or empty PixelData")
        if transfer_syntax.is_compressed:
            decoded_frames = list(generate_frames(pixel_data, number_of_frames=frames))
            if len(decoded_frames) != frames or any(not frame for frame in decoded_frames):
                raise ValueError("invalid encapsulated PixelData frame structure")
            # Encapsulation can be structurally valid while its compressed
            # codestream is truncated.  Decode once because these exact files
            # will be consumed by the formal mechanism experiment.
            decoded_pixels = dataset.pixel_array
            if tuple(decoded_pixels.shape) != (rows, columns):
                raise ValueError(
                    f"decoded pixel shape {tuple(decoded_pixels.shape)} != {(rows, columns)}"
                )
        else:
            expected_bytes = (rows * columns * samples * frames * bits_allocated + 7) // 8
            if len(pixel_data) not in {expected_bytes, expected_bytes + 1}:
                raise ValueError(
                    f"uncompressed PixelData length {len(pixel_data)} != {expected_bytes}"
                )

        result.update(
            {
                "transfer_syntax_uid": str(transfer_syntax),
                "rows": rows,
                "columns": columns,
                "bits_allocated": bits_allocated,
                "photometric_interpretation": photometric,
                "pixel_data_bytes": len(pixel_data),
            }
        )
        result["sha256"] = sha256_file(path)
        result["ok"] = True
    except Exception as error:  # the error is evidence retained in the audit
        result["ok"] = False
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def audit(url_manifest: Path, image_root: Path, workers: int = 4) -> dict[str, object]:
    names = expected_names(url_manifest.read_text(encoding="utf-8").splitlines())
    expected = set(names)
    observed = {path.name for path in image_root.glob("*.dicom") if path.is_file()}
    missing, extra = sorted(expected - observed), sorted(observed - expected)
    inspected: list[dict[str, object]] = []
    if not missing:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            inspected = list(pool.map(inspect_one, (image_root / name for name in names)))
    invalid = [row for row in inspected if row.get("ok") is not True]
    content_lines = [
        f"{row['name']}\t{row['size']}\t{row['sha256']}"
        for row in inspected
        if row.get("ok") is True
    ]
    fingerprint = hashlib.sha256("\n".join(content_lines).encode()).hexdigest()
    passed = not missing and not extra and not invalid and len(inspected) == len(names)
    return {
        "protocol_version": VERSION,
        "url_manifest": str(url_manifest.resolve()),
        "url_manifest_sha256": sha256_file(url_manifest),
        "image_root": str(image_root.resolve()),
        "expected": len(names),
        "observed": len(observed),
        "missing": missing,
        "extra": extra,
        "invalid": invalid,
        "validated": sum(row.get("ok") is True for row in inspected),
        "ordered_dicom_content_sha256": fingerprint if passed else None,
        "passed": passed,
        "completion_claim": (
            "all selected files are present, no unselected DICOM is present, and "
            "every file has matching anonymized file-meta identity, valid VinDr CXR "
            "image metadata, and complete native or encapsulated PixelData"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = audit(args.url_manifest, args.image_root, args.workers)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
