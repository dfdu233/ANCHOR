#!/usr/bin/env python3
"""Resumable canonical decoding controls for a frozen OE-VQA manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile, UnidentifiedImageError

# MedHEval distributes a small number of JPEGs missing only their terminal
# bytes. Decode those identically for every model instead of silently changing
# the denominator per backend. Zero-byte/unidentifiable files still fail.
ImageFile.LOAD_TRUNCATED_IMAGES = True

from anchor.medeval.hashing import sha256_file, sha256_json


def generation_contract(args: argparse.Namespace) -> dict[str, Any]:
    if args.decode_mode == "greedy":
        if args.num_beams != 1:
            raise ValueError("greedy requires --num-beams 1")
        return {"do_sample": False, "num_beams": 1, "temperature": 1.0, "top_p": 1.0}
    if args.decode_mode == "beam":
        if args.num_beams < 2:
            raise ValueError("beam requires --num-beams >=2")
        return {
            "do_sample": False,
            "num_beams": args.num_beams,
            "temperature": 1.0,
            "top_p": 1.0,
        }
    if args.num_beams != 1:
        raise ValueError("sampling controls require --num-beams 1")
    if args.temperature <= 0:
        raise ValueError("sampling requires --temperature >0")
    return {
        "do_sample": True,
        "num_beams": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("manifest must be a JSON list of objects")
    return rows[:limit] if limit else rows


def qid(row: dict[str, Any]) -> str:
    return str(row.get("qid", row.get("question_id", row.get("id"))))


def load_resume(path: Path, expected: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    observed = [qid(row) for row in rows]
    if observed != expected[: len(observed)] or len(observed) != len(set(observed)):
        raise ValueError("existing answers are not an exact unique manifest prefix")
    return rows


def stable_seed(seed: int, item_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu", "llava", "qwen"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-mode", choices=("greedy", "beam", "sample"), default="greedy")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    generation = generation_contract(args)

    rows = load_rows(args.manifest, args.limit)
    expected = [qid(row) for row in rows]
    if len(expected) != len(set(expected)):
        raise ValueError("manifest qids are not unique")
    config = {
        "protocol": "native-oe-vqa-controls-v1",
        "model": args.model,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "image_root": str(args.image_root.resolve()),
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "prompt": "exact source question",
        "generation": generation,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "adapter_code_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "corrected_sgta" / "models_oe.py"
        ),
    }
    fingerprint = sha256_json(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "generation_config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text())
        if prior.get("fingerprint") != fingerprint:
            raise ValueError("refusing to resume an incompatible native OE run")
    else:
        config_path.write_text(json.dumps({**config, "fingerprint": fingerprint}, indent=2) + "\n")
    answers_path = args.output_dir / "answers.jsonl"
    completed = load_resume(answers_path, expected)

    from corrected_sgta.models_oe import load_oe_adapter

    adapter = load_oe_adapter(args.model, llava_conv_mode="mistral_instruct")
    try:
        with answers_path.open("a") as handle:
            for index, row in enumerate(rows[len(completed):], len(completed)):
                image_path = args.image_root / str(row["img_name"])
                try:
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                except (UnidentifiedImageError, OSError) as exc:
                    record = {
                        "question_id": expected[index],
                        "text": "",
                        "gt_ans": str(row["answer"]),
                        "model_id": args.model,
                        "metadata": {
                            "generated_token_count": 0,
                            "generated_token_ids": [],
                            "hit_max_new_tokens": False,
                            "stop_reason": "input_unavailable",
                            "empty_generation": True,
                            "input_error": f"{type(exc).__name__}: {exc}",
                            "fingerprint": fingerprint,
                        },
                    }
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    print(f"[{index + 1}/{len(rows)}] {expected[index]} input_unavailable", flush=True)
                    continue
                result = adapter.generate_control(
                    image=image,
                    prompt=str(row["question"]),
                    do_sample=generation["do_sample"],
                    temperature=generation["temperature"],
                    top_p=generation["top_p"],
                    num_beams=generation["num_beams"],
                    max_new_tokens=args.max_new_tokens,
                    seed=stable_seed(args.seed, expected[index]),
                )
                record = {
                    "question_id": expected[index],
                    "text": result.text,
                    "gt_ans": str(row["answer"]),
                    "model_id": args.model,
                    "metadata": {
                        "generated_token_count": result.token_count,
                        "generated_token_ids": list(result.token_ids),
                        "hit_max_new_tokens": result.token_count >= args.max_new_tokens,
                        "stop_reason": (
                            "length"
                            if result.token_count >= args.max_new_tokens
                            else "eos_or_template"
                        ),
                        "empty_generation": not bool(result.text.strip()),
                        "mean_token_nll": result.uncertainty,
                        "fingerprint": fingerprint,
                    },
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(f"[{index + 1}/{len(rows)}] {expected[index]}", flush=True)
    finally:
        adapter.close()
    final = load_resume(answers_path, expected)
    if len(final) != len(rows):
        raise RuntimeError(f"native OE run incomplete: {len(final)}/{len(rows)}")


if __name__ == "__main__":
    main()
