#!/usr/bin/env python3
"""Legacy Huatuo entry point retained for historical artifact replay.

New common-protocol controls use :mod:`anchor.medeval.run_native_oe_vqa`, whose
Huatuo adapter records actual generated token IDs and processed-token NLL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.run_native_oe_vqa import (
    generation_contract,
    load_resume,
    load_rows,
    qid,
    stable_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
    )
    parser.add_argument(
        "--huatuo-root",
        type=Path,
        default=Path("/home/dbw/HuatuoGPT-Vision"),
    )
    args = parser.parse_args()
    generation = generation_contract(args)

    rows = load_rows(args.manifest, args.limit)
    expected = [qid(row) for row in rows]
    if len(expected) != len(set(expected)):
        raise ValueError("manifest qids are not unique")
    config = {
        "protocol": "native-oe-vqa-controls-v1",
        "model": "huatuo",
        "model_dir": str(args.model_dir.resolve()),
        "model_config_sha256": sha256_file(args.model_dir / "config.json"),
        "huatuo_root": str(args.huatuo_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "image_root": str(args.image_root.resolve()),
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "prompt": "exact source question; Huatuo inserts its image placeholder",
        "generation": {
            **generation,
            "min_new_tokens": 1,
            "repetition_penalty": 1.2,
        },
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    fingerprint = sha256_json(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "generation_config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text())
        if prior.get("fingerprint") != fingerprint:
            raise ValueError("refusing to resume an incompatible Huatuo OE run")
    else:
        config_path.write_text(
            json.dumps({**config, "fingerprint": fingerprint}, indent=2) + "\n"
        )
    answers_path = args.output_dir / "answers.jsonl"
    completed = load_resume(answers_path, expected)

    sys.path.insert(0, str(args.huatuo_root.resolve()))
    from cli import HuatuoChatbot  # type: ignore

    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    bot.gen_kwargs.update(config["generation"])
    bot.gen_kwargs["max_new_tokens"] = args.max_new_tokens
    if not generation["do_sample"]:
        bot.gen_kwargs.pop("temperature", None)
        bot.gen_kwargs.pop("top_p", None)
    try:
        with answers_path.open("a") as handle:
            for index, row in enumerate(rows[len(completed) :], len(completed)):
                item_id = expected[index]
                image_path = args.image_root / str(row["img_name"])
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                sample_seed = stable_seed(args.seed, item_id)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                response = bot.inference(str(row["question"]), [image])
                text = str(response[0] if response else "").strip()
                if not text:
                    raise RuntimeError(f"empty generation for qid {item_id}")
                token_count = len(
                    bot.tokenizer(text, add_special_tokens=False).input_ids
                )
                record = {
                    "question_id": item_id,
                    "text": text,
                    "gt_ans": str(row["answer"]),
                    "model_id": "huatuo",
                    "metadata": {
                        "generated_token_count": token_count,
                        "generated_token_ids": bot.tokenizer(
                            text, add_special_tokens=False
                        ).input_ids,
                        "hit_max_new_tokens": token_count >= args.max_new_tokens,
                        "stop_reason": (
                            "length" if token_count >= args.max_new_tokens else "eos_or_template"
                        ),
                        "fingerprint": fingerprint,
                    },
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(f"[{index + 1}/{len(rows)}] {item_id}", flush=True)
    finally:
        del bot
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final = load_resume(answers_path, expected)
    if len(final) != len(rows):
        raise RuntimeError(f"Huatuo native OE run incomplete: {len(final)}/{len(rows)}")


if __name__ == "__main__":
    main()
