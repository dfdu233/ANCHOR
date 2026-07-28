"""Prepare a CXR-only PubMedVision subset for MiniMind-V.

The script reads PubMedVision JSON annotations plus downloaded ``images_*.zip``
archives and writes a MiniMind-V-compatible parquet file containing
``image_bytes`` and ``conversations`` columns. It avoids full extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import io
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

import pyarrow as pa
import pyarrow.parquet as pq


CXR_MODALITY = re.compile(r"(x.?ray|radiograph|chest radiography)", re.I)
CXR_BODY = re.compile(r"(chest|thorax|thoracic|lung|pulmonary)", re.I)
CXR_TEXT = re.compile(
    r"(chest|thorax|thoracic|lung|pulmonary).{0,100}(x[- ]?ray|radiograph|radiography)"
    r"|\b(cx?r|chest x[- ]?ray|chest radiograph|portable chest|pa chest|ap chest)\b",
    re.I,
)
NEGATIVE_MODALITY_TEXT = re.compile(
    r"\b(ct|computed tomography|mri|magnetic resonance|microscop|ultrasound|endoscopy|fundus|oct)\b",
    re.I,
)


@dataclass
class PreparedRecord:
    id: str
    image_path: str
    modality: str
    body_part: str
    source_json: str
    zip_path: str
    conversations: list[dict[str, str]]


def normalize_conversations(raw: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for turn in raw:
        role = turn.get("role") or turn.get("from")
        value = turn.get("content") or turn.get("value") or ""
        if role in {"human", "user"}:
            role = "user"
            if "<image>" not in value:
                value = "<image>\n" + value
        elif role in {"gpt", "assistant"}:
            role = "assistant"
        else:
            role = str(role or "user")
        out.append({"role": role, "content": value})
    return out


def row_text(row: dict) -> str:
    parts = [str(row.get("modality") or ""), str(row.get("body_part") or "")]
    if row.get("conversations"):
        for turn in row["conversations"]:
            parts.append(str(turn.get("value") or turn.get("content") or ""))
    if row.get("Original_Caption"):
        parts.append(str(row["Original_Caption"]))
    return " ".join(parts)


def is_cxr(row: dict, mode: str, allow_negative_text: bool = False, require_body_cxr: bool = False) -> bool:
    modality = str(row.get("modality") or "")
    body = str(row.get("body_part") or "")
    if require_body_cxr and not CXR_BODY.search(body):
        return False
    text = row_text(row)
    metadata_hit = bool(CXR_MODALITY.search(modality) and CXR_BODY.search(body))
    text_hit = bool(CXR_TEXT.search(text))
    if mode == "metadata":
        hit = metadata_hit
    elif mode == "text":
        hit = text_hit
    elif mode == "metadata_or_text":
        hit = metadata_hit or text_hit
    else:
        raise ValueError(f"unknown filter mode: {mode}")
    if hit and not allow_negative_text and NEGATIVE_MODALITY_TEXT.search(text):
        return False
    return hit


def iter_json_rows(paths: Iterable[Path], limit_scan: int | None = None):
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for idx, row in enumerate(data):
            if limit_scan is not None and idx >= limit_scan:
                break
            yield path.name, row


def index_zips(zip_paths: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    index[name] = zp
    return index


def sample_records(
    json_paths: list[Path],
    image_to_zip: dict[str, Path],
    max_records: int,
    seed: int,
    limit_scan: int | None,
    filter_mode: str,
    allow_negative_text: bool,
    require_body_cxr: bool,
) -> list[PreparedRecord]:
    rng = random.Random(seed)
    reservoir: list[PreparedRecord] = []
    seen = 0
    for source_json, row in iter_json_rows(json_paths, limit_scan=limit_scan):
        if not is_cxr(row, mode=filter_mode, allow_negative_text=allow_negative_text, require_body_cxr=require_body_cxr):
            continue
        images = row.get("image") or []
        if isinstance(images, str):
            images = [images]
        image_path = next((p for p in images if p in image_to_zip), None)
        if image_path is None:
            continue
        conversations = row.get("conversations")
        if not conversations and row.get("Original_Caption"):
            conversations = [
                {"from": "human", "value": "Describe this chest radiograph."},
                {"from": "gpt", "value": str(row["Original_Caption"])},
            ]
        if not conversations:
            continue
        rec = PreparedRecord(
            id=str(row.get("id") or f"{source_json}:{seen}"),
            image_path=image_path,
            modality=str(row.get("modality") or ""),
            body_part=str(row.get("body_part") or ""),
            source_json=source_json,
            zip_path=str(image_to_zip[image_path]),
            conversations=normalize_conversations(conversations),
        )
        seen += 1
        if len(reservoir) < max_records:
            reservoir.append(rec)
        else:
            j = rng.randint(0, seen - 1)
            if j < max_records:
                reservoir[j] = rec
    rng.shuffle(reservoir)
    return reservoir


def to_rgb_jpeg(raw: bytes) -> bytes:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=95)
    return out.getvalue()


def read_image_bytes(records: list[PreparedRecord]) -> list[bytes]:
    grouped: dict[str, list[tuple[int, PreparedRecord]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        grouped[rec.zip_path].append((idx, rec))
    blobs: list[bytes | None] = [None] * len(records)
    for zip_path, items in grouped.items():
        with zipfile.ZipFile(zip_path) as zf:
            for idx, rec in items:
                blobs[idx] = to_rgb_jpeg(zf.read(rec.image_path))
    return [b if b is not None else b"" for b in blobs]


def write_parquet(records: list[PreparedRecord], image_bytes: list[bytes], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "image_bytes": image_bytes,
            "conversations": [json.dumps(r.conversations, ensure_ascii=False) for r in records],
            "id": [r.id for r in records],
            "image_path": [r.image_path for r in records],
            "modality": [r.modality for r in records],
            "body_part": [r.body_part for r in records],
            "source_json": [r.source_json for r in records],
        }
    )
    pq.write_table(table, out_path, compression="zstd")


def fingerprint(paths: list[Path], records: list[PreparedRecord]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p).encode())
        if p.exists():
            st = p.stat()
            h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    for r in records[:1000]:
        h.update(r.id.encode())
        h.update(r.image_path.encode())
    return h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pubmed-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--limit-scan", type=int, default=None)
    parser.add_argument("--filter-mode", choices=["metadata", "text", "metadata_or_text"], default="metadata_or_text")
    parser.add_argument("--allow-negative-text", action="store_true")
    parser.add_argument("--require-body-cxr", action="store_true")
    parser.add_argument(
        "--json",
        dest="json_names",
        nargs="+",
        default=["PubMedVision_InstructionTuning_VQA.json", "PubMedVision_Alignment_VQA.json"],
    )
    args = parser.parse_args()

    json_paths = [args.pubmed_root / name for name in args.json_names]
    zip_paths = sorted(args.pubmed_root.glob("images_*.zip"))
    if not zip_paths:
        raise SystemExit(f"No images_*.zip found under {args.pubmed_root}")

    image_to_zip = index_zips(zip_paths)
    records = sample_records(
        json_paths,
        image_to_zip,
        args.max_records,
        args.seed,
        args.limit_scan,
        args.filter_mode,
        args.allow_negative_text,
        args.require_body_cxr,
    )
    if not records:
        raise SystemExit("No CXR records found in available zips. Wait for more zips or relax filters.")
    image_bytes = read_image_bytes(records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out_dir / f"pubmedvision_cxr_minimind_{len(records)}.parquet"
    manifest_path = args.out_dir / f"pubmedvision_cxr_minimind_{len(records)}.manifest.jsonl"
    summary_path = args.out_dir / "pubmedvision_cxr_summary.json"
    write_parquet(records, image_bytes, parquet_path)
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    summary = {
        "num_records": len(records),
        "parquet": str(parquet_path),
        "manifest": str(manifest_path),
        "pubmed_root": str(args.pubmed_root),
        "zip_count": len(zip_paths),
        "json": [str(p) for p in json_paths],
        "filter_mode": args.filter_mode,
        "allow_negative_text": args.allow_negative_text,
        "require_body_cxr": args.require_body_cxr,
        "modality": Counter(r.modality for r in records).most_common(20),
        "body_part": Counter(r.body_part for r in records).most_common(20),
        "fingerprint": fingerprint(json_paths + zip_paths, records),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
