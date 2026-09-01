#!/usr/bin/env python3
"""Resumable Qwen2.5-VL inference for the frozen metric-calibration probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


VERSION = "qwen25vl-metric-calibration-inference-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def model_contract(model_dir: Path) -> dict[str, Any]:
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json", "model.safetensors.index.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise ValueError(f"model snapshot missing metadata: {missing}")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing_shards = [name for name in shards if not (model_dir / name).is_file()]
    incomplete = [str(path) for path in model_dir.rglob("*.incomplete")]
    if missing_shards or incomplete:
        raise ValueError(f"model snapshot incomplete: missing={missing_shards} incomplete={incomplete[:3]}")
    git_head = subprocess.run(
        ["git", "-C", str(model_dir), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    git_status = subprocess.run(
        ["git", "-C", str(model_dir), "status", "--porcelain"], capture_output=True, text=True
    )
    lfs_files = subprocess.run(
        ["git", "-C", str(model_dir), "lfs", "ls-files", "-l"], capture_output=True, text=True
    )
    lfs_oids = {}
    if lfs_files.returncode == 0:
        for line in lfs_files.stdout.splitlines():
            fields = line.split(maxsplit=2)
            if len(fields) == 3:
                lfs_oids[fields[2]] = fields[0]
    return {
        "root": str(model_dir.resolve()),
        "git_head": git_head.stdout.strip() if git_head.returncode == 0 else None,
        "git_status_porcelain": git_status.stdout if git_status.returncode == 0 else None,
        "metadata_hashes": {name: sha256_file(model_dir / name) for name in required},
        "lfs_oids": dict(sorted(lfs_oids.items())),
        "shard_count": len(shards),
        "shard_sizes": {name: (model_dir / name).stat().st_size for name in shards},
    }


def select_rows(
    manifest: Path,
    *,
    max_images: int,
    contracts: set[str],
    arms: set[str],
    conditions: set[str],
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    image_order = list(dict.fromkeys(row["image_id"] for row in rows))[:max_images]
    selected_images = set(image_order)
    return [
        row
        for row in rows
        if row["image_id"] in selected_images
        and row["question_contract"] in contracts
        and row["arm"] in arms
        and row["condition"] in conditions
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("corrected_runs/metric_calibration_probe_v1/prompt_manifest.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--question-contract", nargs="+", default=["structured_neutral"])
    parser.add_argument("--arm", nargs="+", default=["oracle_coordinate", "vision_coordinate"])
    parser.add_argument("--condition", nargs="+", default=["certified_x1", "missing", "detector_only"])
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    rows = select_rows(
        args.manifest,
        max_images=args.max_images,
        contracts=set(args.question_contract),
        arms=set(args.arm),
        conditions=set(args.condition),
    )
    if not rows:
        raise ValueError("selection is empty")
    contract = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inference_code_sha256": sha256_file(Path(__file__)),
        "model": model_contract(args.model),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "selection": {
            "max_images": args.max_images,
            "question_contract": sorted(set(args.question_contract)),
            "arms": sorted(set(args.arm)),
            "conditions": sorted(set(args.condition)),
            "item_ids": [row["item_id"] for row in rows],
        },
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "max_pixels": args.max_pixels,
            "stop": "model_eos_only",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
        },
    }
    fingerprint_payload = {key: value for key, value in contract.items() if key != "created_at"}
    contract["fingerprint"] = stable_hash(fingerprint_payload)
    config_path = args.output / "config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text())
        if prior.get("fingerprint") != contract["fingerprint"]:
            raise ValueError("output directory belongs to a different run contract")
        contract = prior
    else:
        atomic_json(config_path, contract)

    heartbeat_path = args.output / "heartbeat.json"

    answers_path = args.output / "answers.jsonl"
    completed = {}
    if answers_path.exists():
        for line in answers_path.read_text().splitlines():
            row = json.loads(line)
            if row["item_id"] in completed:
                raise ValueError("duplicate completed item")
            completed[row["item_id"]] = row

    def write_heartbeat(stage: str, sequence_index: int | None = None) -> None:
        atomic_json(
            heartbeat_path,
            {
                "version": VERSION,
                "time": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "stage": stage,
                "sequence_index": sequence_index,
                "completed": len(completed),
                "total": len(rows),
                "run_fingerprint": contract["fingerprint"],
            },
        )

    write_heartbeat("loading_model")

    processor = AutoProcessor.from_pretrained(
        str(args.model),
        local_files_only=True,
        use_fast=False,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()

    for index, row in enumerate(rows):
        if row["item_id"] in completed:
            continue
        image_path = Path(row["image_path"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": row["prompt"]},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        suffix = generated[:, inputs.input_ids.shape[1] :]
        token_ids = suffix[0].tolist()
        text = processor.batch_decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        stop_reason = "eos" if token_ids and token_ids[-1] in eos_ids else "max_new_tokens" if len(token_ids) >= args.max_new_tokens else "other"
        result = {
            "version": VERSION,
            "run_fingerprint": contract["fingerprint"],
            "sequence_index": index,
            "item_id": row["item_id"],
            "image_id": row["image_id"],
            "image_sha256": sha256_file(image_path),
            "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
            "arm": row["arm"],
            "condition": row["condition"],
            "question_contract": row["question_contract"],
            "raw_text": text,
            "generated_token_ids": token_ids,
            "generated_tokens": len(token_ids),
            "stop_reason": stop_reason,
            "expected_measurement_type": row["expected_measurement_type"],
            "expected_physical_value": row["expected_physical_value"],
            "expected_unit": row["expected_unit"],
            "patient_value_identifiable": row["patient_value_identifiable"],
        }
        append_jsonl(answers_path, result)
        completed[row["item_id"]] = result
        write_heartbeat("generating", index)

    final_rows = [json.loads(line) for line in answers_path.read_text().splitlines() if line.strip()]
    expected_ids = [row["item_id"] for row in rows]
    if [row["item_id"] for row in final_rows] != expected_ids:
        raise ValueError("answer rows are not the exact selected sequence")
    summary = {
        "version": VERSION,
        "run_fingerprint": contract["fingerprint"],
        "rows": len(final_rows),
        "answers_sha256": sha256_file(answers_path),
        "nonempty_rate": sum(bool(row["raw_text"].strip()) for row in final_rows) / len(final_rows),
        "cap_hit_rate": sum(row["stop_reason"] == "max_new_tokens" for row in final_rows) / len(final_rows),
        "gpu_authorized_downstream": False,
    }
    atomic_json(args.output / "summary.json", summary)
    write_heartbeat("done", len(rows) - 1)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
