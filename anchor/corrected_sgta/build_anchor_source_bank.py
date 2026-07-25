#!/usr/bin/env python3
"""Build a source-only ANCHOR visual-evidence bank."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from corrected_sgta.anchor_models import load_anchor_adapter
from corrected_sgta.anchor_transport import (
    FEATURE_NAMES,
    VERSION,
    file_sha256,
    load_json_or_jsonl,
    model_artifact_fingerprint,
    normalize_manifest_record,
    normalize_trajectory,
    resolve_image_path,
    robust_feature_statistics,
    stable_json_sha256,
)
from corrected_sgta.cache import repair_truncated_jsonl_tail


BANK_VERSION = "anchor-source-bank-v1"


def parse_source_spec(value: str) -> tuple[str, Path]:
    domain, separator, path = value.partition("=")
    if not separator or not domain.strip() or not path.strip():
        raise ValueError("--source must use DOMAIN=PATH")
    return domain.strip(), Path(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_sources(
    specifications: list[str],
    maximum_per_domain: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    seen_domains: set[str] = set()
    for value in specifications:
        default_domain, path = parse_source_spec(value)
        if default_domain in seen_domains:
            raise ValueError(f"duplicate source domain specification: {default_domain}")
        seen_domains.add(default_domain)
        inputs[str(path.resolve())] = file_sha256(path)
        rows = [
            normalize_manifest_record(
                row, require_answer=True, default_domain=default_domain
            )
            for row in load_json_or_jsonl(path)
        ]
        if any(row["domain"] != default_domain for row in rows):
            raise ValueError(
                f"manifest domain disagrees with source specification {default_domain}"
            )
        if maximum_per_domain:
            rows = rows[:maximum_per_domain]
        records.extend(rows)
    keys = [(row["domain"], row["id"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (domain,id) in source manifests")
    if not records:
        raise ValueError("source manifests contain no records")
    return records, inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build source-reference evidence trajectories. Source answer text "
            "is hashed but never stored in the final bank."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="DOMAIN=PATH",
        help="Source JSON/JSONL in normalized full-sentence schema.",
    )
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--model", choices=("hulu", "llava"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequence-tokens", type=int, default=1024)
    parser.add_argument("--max-samples-per-domain", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_sequence_tokens <= 0 or args.max_samples_per_domain < 0:
        raise ValueError("token and sample limits must be non-negative")
    rows, input_hashes = load_sources(
        args.source, args.max_samples_per_domain
    )
    model_path = args.model_path
    if model_path is None:
        from corrected_sgta.models import HULU_PATH, LLAVA_PATH

        model_path = HULU_PATH if args.model == "hulu" else LLAVA_PATH
    artifact = model_artifact_fingerprint(model_path)
    resolved_images = {}
    image_hash_cache = {}
    source_image_sha256 = []
    for row in rows:
        key = (row["domain"], row["id"])
        image_path = resolve_image_path(row["image"], args.image_root)
        resolved_images[key] = image_path
        path_key = str(image_path)
        if path_key not in image_hash_cache:
            image_hash_cache[path_key] = file_sha256(image_path)
        source_image_sha256.append(
            {"domain": row["domain"], "id": row["id"],
             "sha256": image_hash_cache[path_key]}
        )
    fingerprint_payload = {
        "version": BANK_VERSION,
        "method_version": VERSION,
        "model": args.model,
        "model_artifact_fingerprint": artifact["fingerprint"],
        "source_inputs": input_hashes,
        "source_order": [
            {"domain": row["domain"], "id": row["id"]} for row in rows
        ],
        "source_image_sha256": source_image_sha256,
        "image_root": str(args.image_root.resolve()) if args.image_root else None,
        "max_sequence_tokens": args.max_sequence_tokens,
        "max_samples_per_domain": args.max_samples_per_domain,
        "code_sha256": {
            "builder": file_sha256(Path(__file__)),
            "core": file_sha256(
                Path(__file__).with_name("anchor_transport.py")
            ),
            "models": file_sha256(Path(__file__).with_name("anchor_models.py")),
        },
    }
    fingerprint = stable_json_sha256(fingerprint_payload)
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    if args.raw.exists():
        repair_truncated_jsonl_tail(args.raw)
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if args.raw.exists():
        for line in args.raw.read_text().splitlines():
            record = json.loads(line)
            if record.get("fingerprint") != fingerprint:
                raise RuntimeError("source raw-cache fingerprint mismatch")
            key = (record["domain"], record["id"])
            if key in completed:
                raise RuntimeError(f"duplicate source raw-cache key: {key}")
            completed[key] = record
    if completed and not args.resume:
        raise FileExistsError("source raw cache exists; use --resume")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)

    adapter = load_anchor_adapter(args.model, model_path)
    try:
        with args.raw.open("a") as handle:
            for row in tqdm(
                rows,
                desc=f"ANCHOR source bank ({args.model})",
                initial=len(completed),
                total=len(rows),
            ):
                key = (row["domain"], row["id"])
                if key in completed:
                    continue
                image_path = resolved_images[key]
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                embedding = adapter.input_embedding(image, row["prompt"])
                evidence = adapter.sequence_evidence(
                    image,
                    row["prompt"],
                    row["answer"],
                    args.max_sequence_tokens,
                )
                record = {
                    "version": BANK_VERSION,
                    "fingerprint": fingerprint,
                    "domain": row["domain"],
                    "id": row["id"],
                    "patient_id": row["patient_id"],
                    "image_sha256": image_hash_cache[str(image_path)],
                    "prompt_sha256": stable_json_sha256(row["prompt"]),
                    "reference_sha256": stable_json_sha256(row["answer"]),
                    "embedding": embedding.astype(float).tolist(),
                    "trajectory": evidence.trajectory.astype(float).tolist(),
                    "mean_reference_log_probability": (
                        evidence.mean_image_log_probability
                    ),
                    "eos_included": evidence.eos_included,
                    "source_answer_text_stored": False,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed[key] = record
    finally:
        adapter.close()

    ordered = [completed[(row["domain"], row["id"])] for row in rows]
    statistics = robust_feature_statistics(
        np.asarray(record["trajectory"], dtype=np.float64) for record in ordered
    )
    bank_records = []
    for record in ordered:
        trajectory = normalize_trajectory(
            np.asarray(record["trajectory"], dtype=np.float64), statistics
        )
        bank_records.append(
            {
                key: record[key]
                for key in (
                    "domain",
                    "id",
                    "patient_id",
                    "image_sha256",
                    "prompt_sha256",
                    "reference_sha256",
                    "embedding",
                    "mean_reference_log_probability",
                    "eos_included",
                    "source_answer_text_stored",
                )
            }
            | {"trajectory": trajectory.astype(float).tolist()}
        )
    payload = {
        "version": BANK_VERSION,
        "method_version": VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "model": args.model,
        "model_artifacts": artifact,
        "feature_names": list(FEATURE_NAMES),
        "feature_statistics": statistics,
        "max_sequence_tokens": args.max_sequence_tokens,
        "retrieval_embedding": (
            "equal-weight concatenation of L2-normalized frozen visual and "
            "prompt-token embeddings"
        ),
        "source_answer_text_stored": False,
        "counts_by_domain": dict(
            sorted(Counter(record["domain"] for record in bank_records).items())
        ),
        "records": bank_records,
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fingerprint": fingerprint,
                "records": len(bank_records),
                "counts_by_domain": payload["counts_by_domain"],
                "source_answer_text_stored": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
