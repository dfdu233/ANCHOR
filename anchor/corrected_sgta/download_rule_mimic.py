"""Download only the MIMIC-CXR-JPG files referenced by RULE's test split.

Authentication is read by curl from a netrc file so credentials never appear
in this program's arguments, logs, or manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
from pathlib import Path, PurePosixPath

from PIL import Image

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing-paths", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--netrc-file", type=Path, default=Path("/root/.netrc"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_relative_path(raw: str) -> Path:
    value = PurePosixPath(raw.strip())
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"unsafe relative path: {raw!r}")
    return Path(*value.parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def main() -> None:
    args = parse_args()
    paths = [
        safe_relative_path(line)
        for line in args.missing_paths.read_text().splitlines()
        if line.strip()
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("missing-path list contains duplicates")
    if not args.dry_run and not args.netrc_file.is_file():
        raise FileNotFoundError(
            f"{args.netrc_file} is absent; create a chmod-600 netrc entry for "
            "physionet.org instead of passing a password on the command line"
        )
    if not args.dry_run and stat.S_IMODE(args.netrc_file.stat().st_mode) & 0o077:
        raise PermissionError(f"{args.netrc_file} must not be readable by group or others")

    completed: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    for index, relative in enumerate(paths, start=1):
        destination = args.image_root / relative
        url = f"{BASE_URL}/{relative.as_posix()}"
        if args.dry_run:
            status = "present" if destination.is_file() else "would_download"
        elif valid_image(destination):
            status = "present"
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            command = [
                "curl",
                "--fail",
                "--location",
                "--continue-at",
                "-",
                "--retry",
                "5",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "--netrc-file",
                str(args.netrc_file),
                "--output",
                str(destination),
                url,
            ]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode != 0:
                failed.append(
                    {
                        "path": relative.as_posix(),
                        "error": (result.stderr or result.stdout).strip()[-1000:],
                    }
                )
                print(f"[{index}/{len(paths)}] failed: {relative}", flush=True)
                continue
            if not valid_image(destination):
                failed.append(
                    {
                        "path": relative.as_posix(),
                        "error": "download completed but PIL verification failed",
                    }
                )
                print(f"[{index}/{len(paths)}] invalid image: {relative}", flush=True)
                continue
            status = "downloaded"
        item: dict[str, object] = {
            "path": relative.as_posix(),
            "status": status,
        }
        if valid_image(destination):
            item["bytes"] = destination.stat().st_size
            item["sha256"] = sha256_file(destination)
        completed.append(item)
        print(f"[{index}/{len(paths)}] {status}: {relative}", flush=True)

    manifest = {
        "protocol": "rule-mimic-targeted-download-v1",
        "base_url": BASE_URL,
        "requested": len(paths),
        "completed": len(completed),
        "failed": len(failed),
        "files": completed,
        "failures": failed,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
