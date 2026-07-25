#!/usr/bin/env python3
"""Build a deterministic, leakage-audited multi-source RULE training manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile


VERSION = "rule-source-manifest-v2"
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_RULE = Path("/root/autodl-tmp/RULE")
DEFAULT_MEDHEVAL = Path("/root/autodl-tmp/MedHEval")
DEFAULT_VQARAD = Path(
    "/root/autodl-tmp/Hulu-Med/MedUniEval/corrected_runs/"
    "conditional_source_eb_v1/data/vqa_rad_train.parquet"
)


@dataclass(frozen=True)
class BuildConfig:
    iu_json: Path
    iu_image_root: Path
    slake_root: Path
    vqarad_parquet: Path
    locked_test: Path
    locked_image_root: Path
    output_dir: Path
    seed: int = 42
    dev_fraction: float = 0.2
    max_images_per_domain: int = 128
    qas_per_image: int = 1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_rgb_sha256(path: Path) -> str:
    """Hash decoded RGB content so re-encoding cannot evade leakage checks."""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:RGB:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def stable_digest(*values: object) -> str:
    return hashlib.sha256(
        "\x1f".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def canonical_answer(value: object) -> str:
    """Convert explicit or RULE-style first-sentence answers to Yes./No."""
    text = str(value).replace("\n", " ").strip()
    direct = text.lower().rstrip(".")
    if direct in {"yes", "no"}:
        return direct.capitalize() + "."
    first = text.split(".", 1)[0]
    tokens = first.replace(",", " ").replace(";", " ").split()
    return "No." if any(token.lower() in {"no", "not"} for token in tokens) else "Yes."


def extract_iu_question(value: object) -> str:
    """Remove RULE alignment RAG context and retain only the explicit question."""
    text = str(value).replace("<image>", "").strip()
    if "Question:" not in text:
        raise ValueError("IU alignment prompt has no Question: delimiter")
    question = text.rsplit("Question:", 1)[1].split("\n", 1)[0].strip()
    if not question:
        raise ValueError("empty IU question after removing RAG context")
    return question


def llava_row(
    *,
    domain: str,
    source_id: str,
    image_path: Path,
    image_sha256: str,
    image_blob_sha256: str,
    question: str,
    answer: str,
    original_answer: str,
) -> dict[str, Any]:
    row_id = stable_digest(domain, image_sha256, question, answer)[:24]
    return {
        "id": f"{domain}-{row_id}",
        "image": str(image_path.resolve()),
        "source_domain": domain,
        "source_id": source_id,
        "image_sha256": image_sha256,
        "image_blob_sha256": image_blob_sha256,
        "original_answer": original_answer,
        "conversations": [
            {"from": "human", "value": f"{question.strip()}\n<image>"},
            {"from": "gpt", "value": answer},
        ],
    }


def load_iu(config: BuildConfig) -> list[dict[str, Any]]:
    rows = json.loads(config.iu_json.read_text())
    output = []
    for row in rows:
        image_name = str(row.get("image", "")).strip()
        conversations = row.get("conversations")
        if not image_name or not isinstance(conversations, list) or len(conversations) != 2:
            continue
        image_path = config.iu_image_root / image_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        output.append(
            llava_row(
                domain="rule_iuxray",
                source_id=str(row.get("id", image_name)),
                image_path=image_path,
                image_sha256=canonical_rgb_sha256(image_path),
                image_blob_sha256=sha256_file(image_path),
                question=extract_iu_question(conversations[0].get("value", "")),
                answer=canonical_answer(conversations[1].get("value", "")),
                original_answer=str(conversations[1].get("value", "")),
            )
        )
    return output


def load_slake(config: BuildConfig) -> list[dict[str, Any]]:
    output = []
    for question_path in sorted(config.slake_root.glob("*/question.json")):
        for row in json.loads(question_path.read_text()):
            answer = str(row.get("answer", "")).strip().lower()
            if (
                row.get("q_lang") != "en"
                or row.get("answer_type") != "CLOSED"
                or row.get("modality") != "X-Ray"
                or answer not in {"yes", "no"}
            ):
                continue
            image_name = str(row.get("img_name", "")).strip()
            image_path = config.slake_root / image_name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            output.append(
                llava_row(
                    domain="slake_xray",
                    source_id=image_name,
                    image_path=image_path,
                    image_sha256=canonical_rgb_sha256(image_path),
                    image_blob_sha256=sha256_file(image_path),
                    question=str(row.get("question", "")).strip(),
                    answer=canonical_answer(answer),
                    original_answer=str(row.get("answer", "")),
                )
            )
    return output


def image_extension(value: bytes) -> str:
    with Image.open(io.BytesIO(value)) as image:
        image.verify()
        image_format = str(image.format or "").upper()
    extensions = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp", "GIF": ".gif"}
    if image_format not in extensions:
        raise ValueError(f"unsupported embedded image format: {image_format}")
    return extensions[image_format]


def materialize_content_addressed(value: bytes, directory: Path) -> tuple[Path, str]:
    image_hash = sha256_bytes(value)
    destination = directory / f"{image_hash}{image_extension(value)}"
    directory.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != image_hash:
            raise RuntimeError(f"content-addressed image hash mismatch: {destination}")
        return destination, image_hash
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if sha256_file(temporary) != image_hash:
            raise RuntimeError("temporary embedded image hash mismatch")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination, image_hash


def load_vqarad(config: BuildConfig) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(config.vqarad_parquet)
    image_dir = config.output_dir / "images" / "vqa_rad"
    output = []
    for index, row in frame.iterrows():
        answer = str(row.get("answer", "")).strip().lower()
        if answer not in {"yes", "no"}:
            continue
        image = row.get("image")
        value = image.get("bytes") if isinstance(image, dict) else None
        if not value:
            raise ValueError(f"VQA-RAD row {index} has no embedded image bytes")
        image_path, image_hash = materialize_content_addressed(bytes(value), image_dir)
        output.append(
            llava_row(
                domain="vqa_rad_train",
                source_id=str(index),
                image_path=image_path,
                image_sha256=canonical_rgb_sha256(image_path),
                image_blob_sha256=image_hash,
                question=str(row.get("question", "")).strip(),
                answer=canonical_answer(answer),
                original_answer=str(row.get("answer", "")),
            )
        )
    return output


def locked_hashes(config: BuildConfig) -> tuple[set[str], int]:
    hashes: set[str] = set()
    missing = 0
    for line in config.locked_test.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_path = config.locked_image_root / str(row.get("image", ""))
        if not image_path.is_file():
            missing += 1
            continue
        hashes.add(canonical_rgb_sha256(image_path))
    if missing:
        raise FileNotFoundError(f"{missing} locked-test images are missing")
    return hashes, len(hashes)


def select_and_split(
    rows: list[dict[str, Any]], config: BuildConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["source_domain"]][row["image_sha256"]].append(row)

    selected = []
    for domain in sorted(grouped):
        image_hashes = sorted(
            grouped[domain],
            key=lambda value: stable_digest(config.seed, "image", domain, value),
        )
        if config.max_images_per_domain > 0:
            image_hashes = image_hashes[: config.max_images_per_domain]
        for image_hash in image_hashes:
            candidates = sorted(
                grouped[domain][image_hash],
                key=lambda row: stable_digest(
                    config.seed, "qa", row["id"], row["conversations"][0]["value"]
                ),
            )
            if config.qas_per_image > 0:
                candidates = candidates[: config.qas_per_image]
            selected.extend(candidates)

    train, dev = [], []
    threshold = int(config.dev_fraction * (1 << 64))
    for row in selected:
        split_value = int(
            stable_digest(config.seed, "split", row["image_sha256"])[:16], 16
        )
        (dev if split_value < threshold else train).append(row)
    if not train or not dev:
        raise RuntimeError("hash split produced an empty train or dev partition")
    return sorted(train, key=lambda row: row["id"]), sorted(dev, key=lambda row: row["id"])


def write_json_and_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    json_path = Path(f"{path}.json")
    jsonl_path = Path(f"{path}.jsonl")
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    jsonl_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {
        "json": str(json_path.resolve()),
        "json_sha256": sha256_file(json_path),
        "jsonl": str(jsonl_path.resolve()),
        "jsonl_sha256": sha256_file(jsonl_path),
    }


def source_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "qas": len(rows),
        "images": len({row["image_sha256"] for row in rows}),
        "labels": dict(
            sorted(Counter(row["conversations"][1]["value"] for row in rows).items())
        ),
    }


def build(config: BuildConfig) -> dict[str, Any]:
    if not 0.0 < config.dev_fraction < 1.0:
        raise ValueError("dev_fraction must be strictly between zero and one")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_iu(config) + load_slake(config) + load_vqarad(config)
    forbidden, locked_count = locked_hashes(config)
    overlaps = sorted({row["image_sha256"] for row in rows} & forbidden)
    if overlaps:
        raise RuntimeError(
            f"source images overlap locked test by content hash: {overlaps[:3]}"
        )
    train, dev = select_and_split(rows, config)
    if {row["image_sha256"] for row in train} & {
        row["image_sha256"] for row in dev
    }:
        raise RuntimeError("image content leaked across train/dev")

    all_domains = sorted({row["source_domain"] for row in rows})
    outputs = {
        "train": write_json_and_jsonl(config.output_dir / "train", train),
        "dev": write_json_and_jsonl(config.output_dir / "dev", dev),
        "by_domain": {
            domain: {
                "train": write_json_and_jsonl(
                    config.output_dir / f"train.{domain}",
                    [row for row in train if row["source_domain"] == domain],
                ),
                "dev": write_json_and_jsonl(
                    config.output_dir / f"dev.{domain}",
                    [row for row in dev if row["source_domain"] == domain],
                ),
            }
            for domain in all_domains
        },
    }
    record_protocol = [
        {
            "id": row["id"],
            "domain": row["source_domain"],
            "image_sha256": row["image_sha256"],
            "question": row["conversations"][0]["value"],
            "answer": row["conversations"][1]["value"],
            "split": split,
        }
        for split, split_rows in (("train", train), ("dev", dev))
        for row in split_rows
    ]
    protocol = {
        "version": VERSION,
        "seed": config.seed,
        "dev_fraction": config.dev_fraction,
        "max_images_per_domain": config.max_images_per_domain,
        "qas_per_image": config.qas_per_image,
        "inputs": {
            "iu_json_sha256": sha256_file(config.iu_json),
            "vqarad_parquet_sha256": sha256_file(config.vqarad_parquet),
            "locked_test_sha256": sha256_file(config.locked_test),
        },
        "records": sorted(record_protocol, key=lambda row: row["id"]),
    }
    fingerprint = sha256_bytes(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    )
    manifest = {
        "version": VERSION,
        "fingerprint": fingerprint,
        "image_field": "absolute path; use --image_folder / with LLaVA loaders",
        "question_policy": "question-only; RULE alignment RAG reports removed",
        "split_policy": "deterministic SHA-256(image-content) split; no image crosses splits",
        "config": {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(config).items()
        },
        "locked_test": {
            "unique_image_hashes": locked_count,
            "source_overlap": 0,
            "labels_read_for_selection": False,
        },
        "available": {
            domain: source_stats(
                [row for row in rows if row["source_domain"] == domain]
            )
            for domain in all_domains
        },
        "train": {
            "total": source_stats(train),
            "domains": {
                domain: source_stats(
                    [row for row in train if row["source_domain"] == domain]
                )
                for domain in all_domains
            },
        },
        "dev": {
            "total": source_stats(dev),
            "domains": {
                domain: source_stats(
                    [row for row in dev if row["source_domain"] == domain]
                )
                for domain in all_domains
            },
        },
        "outputs": outputs,
    }
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iu-json",
        type=Path,
        default=DEFAULT_RULE / "data/training/alignment/iuxray.json",
    )
    parser.add_argument(
        "--iu-image-root", type=Path, default=DEFAULT_MEDHEVAL / "images/IU-Xray"
    )
    parser.add_argument(
        "--slake-root", type=Path, default=DEFAULT_MEDHEVAL / "images/Slake"
    )
    parser.add_argument("--vqarad-parquet", type=Path, default=DEFAULT_VQARAD)
    parser.add_argument(
        "--locked-test",
        type=Path,
        default=DEFAULT_RULE / "data/test/mimic_test.jsonl",
    )
    parser.add_argument(
        "--locked-image-root", type=Path, default=DEFAULT_MEDHEVAL / "images"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--max-images-per-domain", type=int, default=128)
    parser.add_argument("--qas-per-image", type=int, default=1)
    return BuildConfig(**vars(parser.parse_args()))


def main() -> None:
    manifest = build(parse_args())
    print(
        json.dumps(
            {
                "fingerprint": manifest["fingerprint"],
                "train": manifest["train"]["total"],
                "dev": manifest["dev"]["total"],
                "output": manifest["config"]["output_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
