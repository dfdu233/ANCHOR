"""Prepare a resumable, provenance-tracked sample of LLaVA-Med alignment images.

The official metadata still records the retired PMC FTP layout.  This module
maps those records to PMC's 2026 AWS Open Data layout and downloads individual
images rather than whole article archives.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import ijson
import requests
from PIL import Image
from tqdm import tqdm

from corrected_sgta.source_bank_v2 import sha256_file


VERSION = "llava-med-alignment-source-v1"
CXR_PATTERN_VERSION = "strict-cxr-caption-v1"
CXR_PATTERN = re.compile(
    r"(?:(?:chest|thorax|lung|pulmonary).{0,80}(?:x[- ]?ray|radiograph)"
    r"|(?:x[- ]?ray|radiograph).{0,80}(?:chest|thorax|lung|pulmonary)"
    r"|\bcxr\b)",
    re.IGNORECASE,
)
MODALITY_PATTERN_VERSIONS = {
    "cxr": CXR_PATTERN_VERSION,
    "ct": "strict-single-clinical-ct-caption-v1",
    "mri": "strict-single-clinical-mri-caption-v1",
}
CT_PATTERN = re.compile(
    r"\b(?:CT|computed tomograph(?:y|ic)|MDCT)\b",
    re.IGNORECASE,
)
MRI_PATTERN = re.compile(
    r"\b(?:MRI|magnetic resonance(?: imaging)?|FLAIR"
    r"|T[12](?:WI|[- ]weighted))\b",
    re.IGNORECASE,
)
XRAY_PATTERN = re.compile(
    r"\b(?:x[- ]?rays?|radiographs?|CXR)\b",
    re.IGNORECASE,
)
RADIOLOGY_CUE_PATTERN = re.compile(
    r"\b(?:axial|coronal|sagittal|transverse|image|imaging|scan|section"
    r"|sequence|contrast|enhanc\w*|hyperintens\w*|hypointens\w*"
    r"|attenuation|window|lesion|mass|nodule|fracture)\b",
    re.IGNORECASE,
)
CLINICAL_ANATOMY_PATTERN = re.compile(
    r"\b(?:patient|brain|head|neck|spine|spinal|chest|thorax|lung"
    r"|pulmonary|heart|cardiac|coronary|abdomen|abdominal|liver|hepatic"
    r"|kidney|renal|pancrea\w*|pelvis|pelvic|prostate|breast|arter\w*"
    r"|vein|vascular|tumou?r|lesion|mass|nodule|cyst|fracture|joint"
    r"|knee|shoulder|hip|femur|aorta|colon|bowel|uter\w*|ovary"
    r"|placenta|fetal|foetal)\b",
    re.IGNORECASE,
)
NONCLINICAL_OR_COMPOSITE_PATTERN = re.compile(
    r"\b(?:micro[- ]?(?:CT|computed tomography)|mouse|mice|rat|rats"
    r"|canine|dog|dogs|rabbit|rabbits|porcine|swine|sheep|goat|bovine"
    r"|zebrafish|phantom|simulation|specimen|cadaver|graph|plot|diagram"
    r"|flowchart|bar chart|histogram|illustration|PET|SPECT|ultrasound"
    r"|sonograph\w*|histolog\w*|patholog\w*|immunohistochem\w*)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", required=True, type=Path)
    parser.add_argument("--image-urls", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--max-pmc-version", type=int, default=3)
    parser.add_argument(
        "--modality",
        choices=tuple(MODALITY_PATTERN_VERSIONS),
        default="cxr",
    )
    return parser.parse_args()


def caption_text(item: dict) -> str:
    # The misspelling ``conversatons`` is part of the official release schema.
    return " ".join(
        row.get("value", "")
        for row in item.get("conversatons", [])
        if row.get("from") == "gpt"
    )


def is_strict_cxr_caption(text: str) -> bool:
    """High-recall, label-free CXR candidate rule frozen before VLM evaluation."""

    return bool(CXR_PATTERN.search(text))


def is_strict_modality_caption(text: str, modality: str) -> bool:
    """Select a conservative, label-free, single-modality source caption."""

    if modality == "cxr":
        return is_strict_cxr_caption(text)
    selected = CT_PATTERN if modality == "ct" else MRI_PATTERN
    other = MRI_PATTERN if modality == "ct" else CT_PATTERN
    return bool(
        selected.search(text)
        and not other.search(text)
        and not XRAY_PATTERN.search(text)
        and RADIOLOGY_CUE_PATTERN.search(text)
        and CLINICAL_ANATOMY_PATTERN.search(text)
        and not NONCLINICAL_OR_COMPOSITE_PATTERN.search(text)
    )


def deterministic_key(pair_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()


def pmc_s3_urls(image_file_path: str, max_version: int = 3) -> list[str]:
    parts = Path(image_file_path).parts
    if len(parts) < 2 or not re.fullmatch(r"PMC\d+", parts[0]):
        raise ValueError(f"unsupported PMC image path: {image_file_path}")
    pmcid = parts[0]
    relative = "/".join(quote(part, safe="._-()") for part in parts[1:])
    return [
        f"https://pmc-oa-opendata.s3.amazonaws.com/{pmcid}.{version}/{relative}"
        for version in range(1, max_version + 1)
    ]


def alignment_candidates(path: Path, modality: str = "cxr") -> dict[str, dict]:
    selected: dict[str, dict] = {}
    with path.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            caption = caption_text(item)
            if is_strict_modality_caption(caption, modality):
                selected[item["id"]] = {
                    "pair_id": item["id"],
                    "released_image_name": item.get("image"),
                    "caption": caption,
                    "modality": modality,
                }
    return selected


def join_urls(path: Path, candidates: dict[str, dict]) -> list[dict]:
    joined = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            pair_id = row.get("pair_id")
            if pair_id in candidates:
                joined.append({**candidates[pair_id], **row})
    return joined


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def load_completed(path: Path) -> dict[str, dict]:
    completed = {}
    if not path.is_file():
        return completed
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_path = Path(row["local_path"])
        if image_path.is_file() and sha256_file(image_path) == row["file_sha256"]:
            completed[row["pair_id"]] = row
    return completed


def download_one(row: dict, args: argparse.Namespace, images_dir: Path) -> dict:
    proxies = (
        {"http": args.proxy, "https": args.proxy}
        if args.proxy
        else None
    )
    suffix = Path(row["image_file_path"]).suffix.lower() or ".img"
    local_path = images_dir / f"{row['pair_id']}{suffix}"
    errors = []
    for source_url in pmc_s3_urls(row["image_file_path"], args.max_pmc_version):
        for attempt in range(args.retries):
            try:
                response = requests.get(
                    source_url,
                    timeout=args.timeout,
                    proxies=proxies,
                )
                if response.status_code == 404:
                    errors.append(f"{source_url}:404")
                    break
                response.raise_for_status()
                payload = response.content
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                temporary = local_path.with_name(local_path.name + ".tmp")
                temporary.write_bytes(payload)
                temporary.replace(local_path)
                with Image.open(local_path) as image:
                    width, height = image.size
                return {
                    **row,
                    "local_path": str(local_path.resolve()),
                    "source_url": source_url,
                    "file_sha256": sha256_file(local_path),
                    "n_bytes": local_path.stat().st_size,
                    "width": width,
                    "height": height,
                    "download_status": "ok",
                }
            except Exception as exc:
                errors.append(f"{source_url}:{type(exc).__name__}:{exc}"[:300])
                if attempt + 1 < args.retries:
                    time.sleep(0.5 * (attempt + 1))
    return {
        "pair_id": row["pair_id"],
        "image_file_path": row["image_file_path"],
        "download_status": "error",
        "errors": errors[-8:],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    index_path = args.output_dir / "index.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    metadata_path = args.output_dir / "metadata.json"

    config = {
        "version": VERSION,
        "caption_pattern_version": MODALITY_PATTERN_VERSIONS[args.modality],
        "modality": args.modality,
        "alignment": str(args.alignment.resolve()),
        "alignment_sha256": sha256_file(args.alignment),
        "image_urls": str(args.image_urls.resolve()),
        "image_urls_sha256": sha256_file(args.image_urls),
        "max_candidates": args.max_candidates,
        "seed": args.seed,
        "max_pmc_version": args.max_pmc_version,
        "selection": (
            f"sha256(seed:pair_id) over frozen strict-{args.modality} "
            "single-modality clinical caption matches"
        ),
        "download_source": "PMC AWS Open Data individual image objects",
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    if metadata_path.is_file():
        old = json.loads(metadata_path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"metadata mismatch; choose a new output directory: {args.output_dir}"
            )

    candidates = alignment_candidates(args.alignment, args.modality)
    joined = join_urls(args.image_urls, candidates)
    url_matches = len(joined)
    joined.sort(key=lambda row: deterministic_key(row["pair_id"], args.seed))
    if args.max_candidates:
        joined = joined[: args.max_candidates]
    completed = load_completed(index_path)
    eligible = [row for row in joined if row["pair_id"] not in completed]
    metadata = {
        "fingerprint": fingerprint,
        "config": config,
        "strict_caption_matches": len(candidates),
        "url_matches": url_matches,
        "selected_candidates": len(joined),
        "completed_before_run": len(completed),
    }
    atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    if not eligible:
        return

    with (
        index_path.open("a") as success_handle,
        errors_path.open("a") as error_handle,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {
            pool.submit(download_one, row, args, images_dir): row["pair_id"]
            for row in eligible
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="LLaVA exact-source images"
        ):
            result = future.result()
            handle = (
                success_handle
                if result["download_status"] == "ok"
                else error_handle
            )
            handle.write(json.dumps(result, separators=(",", ":")) + "\n")
            handle.flush()

    final_completed = load_completed(index_path)
    metadata["completed_after_run"] = len(final_completed)
    metadata["errors_this_run"] = len(eligible) - (
        len(final_completed) - len(completed)
    )
    atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
