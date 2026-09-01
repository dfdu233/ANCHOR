#!/usr/bin/env python3
"""Crash-safe Huatuo generation for the frozen pragmatic-commitment substrate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from .audit_diagnostic_completion_substrate_v1 import sha256_file
from .prepare_pragmatic_commitment_confirmatory_v1 import (
    CONDITIONS,
    VERSION as SUBSTRATE_VERSION,
    canonical_hash,
)


VERSION = "huatuo-pragmatic-commitment-generation-v1"
DEFAULT_SUBSTRATE = Path(
    "/home/dbw/ANCHOR/corrected_runs/pragmatic_commitment/confirmatory_substrate_v1"
)
DEFAULT_IMAGE_ROOT = Path("/workspace/vinbigdata/train")
DEFAULT_MODEL_DIR = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO_ROOT = Path("/home/dbw/HuatuoGPT-Vision")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def record_key(item_id: str, condition: str) -> str:
    return hashlib.sha256(f"{VERSION}\0{item_id}\0{condition}".encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_substrate(
    substrate_dir: Path, image_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path = substrate_dir / "substrate_config.json"
    manifest_path = substrate_dir / "selected_manifest.jsonl"
    if not config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("substrate_config.json and selected_manifest.jsonl are required")
    config = json.loads(config_path.read_text())
    if config.get("version") != SUBSTRATE_VERSION:
        raise ValueError("wrong substrate version")
    expected_fingerprint = str(config.get("fingerprint", ""))
    payload = {key: value for key, value in config.items() if key != "fingerprint"}
    if expected_fingerprint != canonical_hash(payload):
        raise ValueError("substrate fingerprint mismatch")
    if config.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("substrate manifest hash mismatch")
    if config.get("prompt_conditions") != list(CONDITIONS):
        raise ValueError("frozen prompt conditions changed")
    rows = load_jsonl(manifest_path)
    if len(rows) != int(config["diagnostics"]["eligible_after_discovery_exclusion"]):
        raise ValueError("substrate row count mismatch")
    if len({str(row["image_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate image in substrate")
    if {str(row["experiment_split"]) for row in rows} != {"dev", "test"}:
        raise ValueError("substrate must contain dev and test")
    missing = [
        str(image_root / f"{row['image_id']}.dicom")
        for row in rows
        if not (image_root / f"{row['image_id']}.dicom").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{len(missing)} DICOMs missing; first={missing[0]}")
    return config, rows


def validate_shard(
    row: Mapping[str, Any], item: Mapping[str, Any], condition: Mapping[str, str], fingerprint: str
) -> None:
    required = {
        "version",
        "item_id",
        "image_id",
        "experiment_split",
        "prompt_condition",
        "prompt",
        "text",
        "generated_token_ids",
        "generated_token_count",
        "generation_fingerprint",
        "substrate_fingerprint",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"shard missing fields: {sorted(missing)}")
    if row["version"] != VERSION:
        raise ValueError("wrong shard version")
    if row["item_id"] != item["item_id"] or row["image_id"] != item["image_id"]:
        raise ValueError("shard item identity mismatch")
    if row["experiment_split"] != item["experiment_split"]:
        raise ValueError("shard split mismatch")
    if row["prompt_condition"] != condition["name"] or row["prompt"] != condition["prompt"]:
        raise ValueError("shard prompt identity mismatch")
    if row["generation_fingerprint"] != fingerprint:
        raise ValueError("shard generation fingerprint mismatch")
    if not str(row["text"]).strip():
        raise ValueError("empty model output")
    ids = row["generated_token_ids"]
    if not isinstance(ids, list) or len(ids) != int(row["generated_token_count"]):
        raise ValueError("invalid token accounting")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate-dir", type=Path, default=DEFAULT_SUBSTRATE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7319)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    substrate, items = validate_substrate(args.substrate_dir, args.image_root)
    runner_path = Path(__file__).resolve()
    generation_settings = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": 1,
        "repetition_penalty": 1.2,
    }
    candidate = {
        "version": VERSION,
        "created_at": utc_now(),
        "command": sys.argv,
        "substrate_dir": str(args.substrate_dir.resolve()),
        "substrate_fingerprint": substrate["fingerprint"],
        "substrate_manifest_sha256": substrate["manifest_sha256"],
        "selected_images": len(items),
        "generation_jobs": len(items) * len(CONDITIONS),
        "image_root": str(args.image_root.resolve()),
        "model_id": "huatuo",
        "model_dir": str(args.model_dir.resolve()),
        "huatuo_root": str(args.huatuo_root.resolve()),
        "device": args.device,
        "seed": args.seed,
        "generation": generation_settings,
        "prompt_conditions": list(CONDITIONS),
        "runner_sha256": sha256_file(runner_path),
        "clinical_evaluation_performed": False,
        "reader_votes_visible_to_model": False,
    }
    immutable = {key: value for key, value in candidate.items() if key not in {"created_at", "command"}}
    candidate["fingerprint"] = canonical_hash(immutable)
    print(
        json.dumps(
            {
                "preflight_passed": True,
                "selected_images": len(items),
                "generation_jobs": len(items) * len(CONDITIONS),
                "split_counts": dict(Counter(row["experiment_split"] for row in items)),
                "fingerprint": candidate["fingerprint"],
            },
            indent=2,
        ),
        flush=True,
    )
    if args.preflight_only:
        return

    # GPU/model imports are intentionally delayed until every CPU identity gate passes.
    import torch

    from .run_clinical_presupposition_generation_v1 import exact_generate, surface_refusal
    from .run_huatuo_vindr_commitment_probe import (
        atomic_json,
        dicom_to_pil,
        import_huatuo,
    )
    from .run_huatuo_dicom_render_pilot_v1 import model_artifact_fingerprint

    candidate["model_artifact_fingerprint"] = model_artifact_fingerprint(args.model_dir)
    immutable = {key: value for key, value in candidate.items() if key not in {"created_at", "command", "fingerprint"}}
    candidate["fingerprint"] = canonical_hash(immutable)
    fingerprint = str(candidate["fingerprint"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "generation_config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("output already exists; use --resume")
        existing = json.loads(config_path.read_text())
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("refusing incompatible resume")
        candidate = existing
    else:
        if args.resume:
            raise FileNotFoundError("--resume requires generation_config.json")
        atomic_json(config_path, candidate)

    shards = args.output_dir / "shards"
    errors = args.output_dir / "errors"
    shards.mkdir(exist_ok=True)
    errors.mkdir(exist_ok=True)
    order = list(items)
    random.Random(args.seed).shuffle(order)
    completed = 0
    for item in order:
        for condition in CONDITIONS:
            path = shards / f"{record_key(str(item['item_id']), condition['name'])}.json"
            if path.exists():
                validate_shard(json.loads(path.read_text()), item, condition, fingerprint)
                completed += 1
    print(f"strict resume: {completed}/{len(items) * len(CONDITIONS)} shards", flush=True)

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device=args.device)
    bot.gen_kwargs.update(generation_settings)
    try:
        conformance_path = args.output_dir / "generation_conformance.json"
        first = order[0]
        first_condition = CONDITIONS[0]
        if conformance_path.exists():
            conformance = json.loads(conformance_path.read_text())
            if conformance.get("fingerprint") != fingerprint or conformance.get("passed") is not True:
                raise ValueError("invalid conformance artifact")
        else:
            image = dicom_to_pil(args.image_root / f"{first['image_id']}.dicom")
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            direct = exact_generate(
                bot,
                first_condition["prompt"],
                image,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=1.2,
            )
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            standard = bot.inference(first_condition["prompt"], [image])
            standard_text = str(standard[0] if standard else "").strip()
            conformance = {
                "version": "huatuo-pragmatic-direct-standard-conformance-v1",
                "fingerprint": fingerprint,
                "item_id": first["item_id"],
                "prompt_condition": first_condition["name"],
                "direct_text": direct["text"],
                "standard_text": standard_text,
                "passed": bool(direct["text"] and direct["text"] == standard_text),
                "created_at": utc_now(),
            }
            atomic_json(conformance_path, conformance)
            if not conformance["passed"]:
                raise RuntimeError("direct generation differs from Huatuo bot.inference")

        for item_index, item in enumerate(order, 1):
            image = None
            for condition in CONDITIONS:
                key = record_key(str(item["item_id"]), condition["name"])
                path = shards / f"{key}.json"
                if path.exists():
                    continue
                if image is None:
                    image = dicom_to_pil(args.image_root / f"{item['image_id']}.dicom")
                sample_seed = int(
                    hashlib.sha256(
                        f"{args.seed}:{item['image_id']}:{condition['name']}".encode()
                    ).hexdigest()[:16],
                    16,
                ) % (2**31)
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                try:
                    direct = exact_generate(
                        bot,
                        condition["prompt"],
                        image,
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=1.2,
                    )
                    text = str(direct["text"]).strip()
                    record = {
                        "version": VERSION,
                        "item_id": item["item_id"],
                        "image_id": item["image_id"],
                        "dicom_relpath": item["dicom_relpath"],
                        "experiment_split": item["experiment_split"],
                        "prompt_condition": condition["name"],
                        "prompt": condition["prompt"],
                        "pragmatic_task": condition["pragmatic_task"],
                        "answer_space_focus": condition["answer_space_focus"],
                        "text": text,
                        "generated_token_ids": direct["generated_token_ids"],
                        "generated_token_count": direct["generated_token_count"],
                        "visible_answer_token_count": len(
                            bot.tokenizer(text, add_special_tokens=False).input_ids
                        ),
                        "prompt_token_count": direct["prompt_token_count"],
                        "elapsed_seconds": direct["elapsed_seconds"],
                        "hit_max_new_tokens": direct["generated_token_count"]
                        >= args.max_new_tokens,
                        "sample_seed": sample_seed,
                        "lung_opacity_votes": item["lung_opacity_votes"],
                        "pneumonia_votes": item["pneumonia_votes"],
                        "reader_votes_visible_to_model": False,
                        "clinical_evaluation_status": "pending_frozen_pair_audit",
                        "surface_refusal_diagnostic": surface_refusal(text),
                        "substrate_fingerprint": substrate["fingerprint"],
                        "generation_fingerprint": fingerprint,
                        "created_at": utc_now(),
                    }
                    validate_shard(record, item, condition, fingerprint)
                    atomic_json(path, record)
                    completed += 1
                    print(
                        f"[{completed}/{len(items) * len(CONDITIONS)}] "
                        f"{item_index}/{len(items)} {item['image_id']} {condition['name']} "
                        f"tokens={record['generated_token_count']} seconds={record['elapsed_seconds']:.2f}",
                        flush=True,
                    )
                except Exception as error:
                    atomic_json(
                        errors / f"{key}.json",
                        {
                            "version": VERSION,
                            "item_id": item["item_id"],
                            "prompt_condition": condition["name"],
                            "generation_fingerprint": fingerprint,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                            "created_at": utc_now(),
                        },
                    )
                    raise
    finally:
        del bot
        torch.cuda.empty_cache()

    records = []
    for item in sorted(items, key=lambda row: row["image_id"]):
        for condition in CONDITIONS:
            path = shards / f"{record_key(str(item['item_id']), condition['name'])}.json"
            payload = json.loads(path.read_text())
            validate_shard(payload, item, condition, fingerprint)
            records.append(payload)
    generations_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    atomic_write(args.output_dir / "generations.jsonl", generations_text)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_condition[str(row["prompt_condition"])].append(row)
    summary = {
        "version": VERSION,
        "fingerprint": fingerprint,
        "records": len(records),
        "images": len(items),
        "generations_sha256": hashlib.sha256(generations_text.encode()).hexdigest(),
        "conditions": {
            condition: {
                "records": len(rows),
                "mean_generated_tokens": mean(int(row["generated_token_count"]) for row in rows),
                "mean_elapsed_seconds": mean(float(row["elapsed_seconds"]) for row in rows),
                "hit_max_new_tokens": sum(bool(row["hit_max_new_tokens"]) for row in rows),
            }
            for condition, rows in sorted(by_condition.items())
        },
        "clinical_evaluation_status": "pending_frozen_pair_audit",
        "created_at": utc_now(),
    }
    atomic_write(args.output_dir / "generation_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
