"""Build deterministic, role-isolated CECD reviewer delivery archives.

The source admission pack contains sealed analysis material and every reviewer's
sheet.  None of that directory should be handed to a reviewer directly.  This
builder emits one self-contained archive per reviewer role, preserving the
frozen CSV bytes and including only the images required by that role.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Iterable


VERSION = "cecd-blinded-reviewer-delivery-v1"
SOURCE_VERSION = "cecd-blinded-human-admission-pack-v2"

CLINICAL_FIELDS = (
    "support_state_same_supported_refuted_undetermined",
    "lesion_visibility",
    "clinically_interchangeable",
    "unable_to_judge",
    "comments",
)
LANGUAGE_FIELDS = (
    "same_clinical_proposition",
    "same_speech_act",
    "same_certainty_demand",
    "same_answer_space",
    "comments",
)

ROLES: dict[str, dict[str, Any]] = {
    "clinical_reviewer_1": {
        "sheet": "clinical_reviewer_1.csv",
        "root": "cecd_clinical_reviewer_1_v2",
        "kind": "clinical",
        "images": True,
    },
    "clinical_reviewer_2": {
        "sheet": "clinical_reviewer_2.csv",
        "root": "cecd_clinical_reviewer_2_v2",
        "kind": "clinical",
        "images": True,
    },
    "clinical_template_reviewer": {
        "sheet": "clinical_template_reviewer.csv",
        "root": "cecd_clinical_template_reviewer_v2",
        "kind": "clinical_template",
        "images": False,
    },
    "language_reviewer": {
        "sheet": "language_annotator.csv",
        "root": "cecd_language_reviewer_v2",
        "kind": "language",
        "images": False,
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_csv_bytes(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))


def _require_blank(rows: list[dict[str, str]], fields: Iterable[str], role: str) -> None:
    for row_number, row in enumerate(rows, start=2):
        for field in fields:
            if field not in row:
                raise RuntimeError(f"{role}: missing decision field {field}")
            if row[field].strip():
                raise RuntimeError(f"{role}: frozen sheet is not blank at row {row_number}, {field}")


def _instructions(kind: str, sheet: str) -> bytes:
    if kind == "clinical":
        text = f"""# Independent blinded CECD clinical image review

Review all 252 rows in `{sheet}` independently. For each row, open the two relative image paths in `image_A` and `image_B`. The `finding` column names the clinical finding to compare.

Fill only these fields:

- `support_state_same_supported_refuted_undetermined`: `yes`, `no`, or `unable`.
- `lesion_visibility`: `unchanged`, `A_clearer`, `B_clearer`, or `unable`.
- `clinically_interchangeable`: `yes`, `no`, or `unable`.
- `unable_to_judge`: `yes` or `no`; enter `yes` if any primary field is `unable`.
- `comments`: optional concise rationale; do not place patient identifiers here.

Judge the displayed images only. Do not infer hidden conditions from filenames, reorder rows, consult another reviewer, or attempt to identify the source images. Preserve the CSV filename, columns, row order, and item IDs. Return only the completed `{sheet}` file to the study coordinator.
"""
    elif kind == "clinical_template":
        text = f"""# Independent blinded CECD clinical wording review

Review all 8 wording pairs in `{sheet}` independently from a clinical interpretation perspective. No images are required. Do not decide whether the named finding is present; compare what the two questions ask.

Fill `same_clinical_proposition`, `same_speech_act`, `same_certainty_demand`, and `same_answer_space` separately with `yes`, `no`, or `unable`. The optional `comments` field may contain a concise rationale.

Do not consult another reviewer or infer why a wording pair was included. Preserve the CSV filename, columns, row order, and item IDs. Return only the completed `{sheet}` file to the study coordinator.
"""
    elif kind == "language":
        text = f"""# Independent blinded CECD language review

Review all 8 wording pairs in `{sheet}` independently. No images are required. Compare the literal task semantics of each pair rather than answering either medical question.

Fill `same_clinical_proposition`, `same_speech_act`, `same_certainty_demand`, and `same_answer_space` separately with `yes`, `no`, or `unable`. The optional `comments` field may contain a concise rationale.

Do not consult another reviewer or infer why a wording pair was included. Preserve the CSV filename, columns, row order, and item IDs. Return only the completed `{sheet}` file to the study coordinator.
"""
    else:  # pragma: no cover - guarded by the frozen role table
        raise ValueError(f"unknown reviewer kind: {kind}")
    return text.encode("utf-8")


def _inventory_hash(entries: list[tuple[str, str]]) -> str:
    payload = "".join(f"{digest}  {name}\n" for name, digest in sorted(entries))
    return sha256_bytes(payload.encode("utf-8"))


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    archive.addfile(_tar_info(name, len(data)), io.BytesIO(data))


def _add_file(archive: tarfile.TarFile, name: str, path: Path) -> None:
    with path.open("rb") as handle:
        archive.addfile(_tar_info(name, path.stat().st_size), handle)


def _write_deterministic_tar_gz(
    output: Path,
    byte_members: dict[str, bytes],
    file_members: dict[str, Path],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as raw:
        temporary = Path(raw.name)
        try:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                    for name in sorted(set(byte_members) | set(file_members)):
                        if name in byte_members:
                            _add_bytes(archive, name, byte_members[name])
                        else:
                            _add_file(archive, name, file_members[name])
            raw.flush()
            os.fsync(raw.fileno())
            os.replace(temporary, output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _source_role(pack_dir: Path, role: str) -> dict[str, Any]:
    spec = ROLES[role]
    source_manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("version") != SOURCE_VERSION:
        raise RuntimeError("wrong CECD source pack version")

    sheet_path = pack_dir / spec["sheet"]
    sheet_bytes = sheet_path.read_bytes()
    expected_hash = source_manifest.get("artifact_sha256", {}).get(spec["sheet"])
    if not expected_hash or sha256_bytes(sheet_bytes) != expected_hash:
        raise RuntimeError(f"stale frozen source sheet: {spec['sheet']}")
    rows = _read_csv_bytes(sheet_bytes)

    if spec["kind"] == "clinical":
        if len(rows) != 252:
            raise RuntimeError(f"{role}: expected 252 clinical rows")
        _require_blank(rows, CLINICAL_FIELDS, role)
        expected_images = {
            row[field]
            for row in rows
            for field in ("image_A", "image_B")
        }
        if len(expected_images) != 504 or any(not name.startswith("images/") for name in expected_images):
            raise RuntimeError(f"{role}: invalid clinical image references")
        actual_images = {
            f"images/{path.name}" for path in (pack_dir / "images").glob("*.png")
        }
        if actual_images != expected_images:
            raise RuntimeError(f"{role}: source images do not exactly match the frozen sheet")
    else:
        if len(rows) != 8:
            raise RuntimeError(f"{role}: expected 8 language rows")
        _require_blank(rows, LANGUAGE_FIELDS, role)
        expected_images = set()

    return {
        "spec": spec,
        "sheet_path": sheet_path,
        "sheet_bytes": sheet_bytes,
        "rows": rows,
        "images": sorted(expected_images),
    }


def build_role(pack_dir: Path, output_dir: Path, role: str) -> dict[str, Any]:
    source = _source_role(pack_dir, role)
    spec = source["spec"]
    root = spec["root"]
    instructions = _instructions(spec["kind"], spec["sheet"])

    image_members: dict[str, Path] = {}
    image_hashes: list[tuple[str, str]] = []
    for relative in source["images"]:
        source_path = pack_dir / relative
        archive_relative = relative
        image_members[f"{root}/{archive_relative}"] = source_path
        image_hashes.append((archive_relative, sha256_file(source_path)))

    payload_file_count = 2 + len(image_members)
    delivery_manifest = {
        "version": VERSION,
        "role": role,
        "review_sheet": {
            "filename": spec["sheet"],
            "rows": len(source["rows"]),
            "sha256": sha256_bytes(source["sheet_bytes"]),
        },
        "instructions": {
            "filename": "INSTRUCTIONS.md",
            "sha256": sha256_bytes(instructions),
        },
        "images": {
            "included": bool(spec["images"]),
            "count": len(image_members),
            "inventory_sha256": _inventory_hash(image_hashes),
        },
        "payload_file_count": payload_file_count,
        "archive_member_file_count": payload_file_count + 1,
        "blinded_role_isolation": True,
    }

    byte_members = {
        f"{root}/INSTRUCTIONS.md": instructions,
        f"{root}/{spec['sheet']}": source["sheet_bytes"],
        f"{root}/DELIVERY_MANIFEST.json": canonical_json(delivery_manifest),
    }
    archive_path = output_dir / f"{root}.tar.gz"
    _write_deterministic_tar_gz(archive_path, byte_members, image_members)
    return {
        "role": role,
        "archive": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_member_file_count": delivery_manifest["archive_member_file_count"],
        "root": root,
    }


def build_deliveries(pack_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [build_role(pack_dir, output_dir, role) for role in ROLES]
    index = {
        "version": VERSION,
        "source_pack_version": SOURCE_VERSION,
        "archives": records,
    }
    index_path = output_dir / "delivery_index.json"
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_bytes(canonical_json(index))
    os.replace(temporary, index_path)
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_deliveries(args.pack_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
