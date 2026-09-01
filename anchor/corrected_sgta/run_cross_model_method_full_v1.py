#!/usr/bin/env python3
"""Resumable full-manifest native/VCD/DoLa generation for medical VLMs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError

from anchor.medeval.hashing import sha256_file, sha256_json
from .cross_model_cve import generate_cve
from .cross_model_agla import generate_agla
from .cross_model_avisc import generate_avisc
from .cross_model_clearsight import generate_clearsight
from .cross_model_dola import generate_dola
from .cross_model_icd import generate_icd
from .cross_model_vcd import generate_vcd
from .models_oe import load_oe_adapter
from .protocol_v2 import build_prompt


ImageFile.LOAD_TRUNCATED_IMAGES = True


def qid(row: dict, index: int) -> str:
    return str(row.get("qid", row.get("question_id", row.get("id", index))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu", "llava", "llava16", "qwen"), required=True)
    parser.add_argument("--method", choices=("greedy", "vcd", "icd", "cve", "agla", "avisc", "clearsight", "dola"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text())
    expected = [qid(row, index) for index, row in enumerate(rows)]
    if len(expected) != len(set(expected)):
        raise ValueError("manifest qids are not unique")
    method_source = {
        "vcd": "cross_model_vcd.py", "icd": "cross_model_icd.py",
        "cve": "cross_model_cve.py", "agla": "cross_model_agla.py",
        "avisc": "cross_model_avisc.py", "clearsight": "cross_model_clearsight.py", "dola": "cross_model_dola.py",
        "greedy": "models_oe.py",
    }[args.method]
    method_source = Path(__file__).with_name(method_source)
    model_root = Path({
        "huatuo": "/home/dbw/models/HuatuoGPT-Vision-7B",
        "hulu": "/home/dbw/models/Hulu-Med-4B",
        "llava": "/home/dbw/models/LLaVA-Med-v1.5-mistral-7b",
        "llava16": "/home/dbw/models/llava-v1.6-vicuna-7b",
        "qwen": "/home/dbw/models/Qwen2.5-VL-7B-Instruct",
    }[args.model])
    config = {
        "protocol": "cross-model-method-full-v1",
        "model": args.model,
        "method": args.method,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "image_root": str(args.image_root.resolve()),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "generation": "deterministic greedy; native EOS only",
        "defaults": {
            "vcd": {"alpha": 1.0, "beta": 0.1, "noise_step": 500},
            "icd": {"alpha": 1.0, "beta": 0.1, "disturbance": "official ICD fuzzy detector instruction"},
            "cve": {"alpha": 2.0, "beta": 0.2, "mid_quantile": 0.60, "high_quantile": 0.95},
            "agla": {"alpha": 2.0, "beta": 0.5, "augmentation_strength": 1.35},
            "avisc": {"alpha": 2.5, "beta": 0.1, "lambda": 1.0, "mask_space": "image_proxy"},
            "clearsight": {"alpha": 0.15, "beta": 0.10, "attention_hook": False, "mask_space": "image_proxy"},
            "dola": {"dola_layers": "low"},
            "greedy": {"do_sample": False, "num_beams": 1},
        }[args.method],
        "model_config_sha256": sha256_file(model_root / "config.json"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "method_source_sha256": sha256_file(method_source),
    }
    fingerprint = sha256_json(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "generation_config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text()).get("fingerprint") != fingerprint:
            raise ValueError("refusing to resume an incompatible cross-model method run")
    else:
        config_path.write_text(json.dumps({**config, "fingerprint": fingerprint}, indent=2) + "\n")
    answers = args.output_dir / "answers.jsonl"
    done = [json.loads(line) for line in answers.read_text().splitlines() if line.strip()] if answers.is_file() else []
    if [str(row["question_id"]) for row in done] != expected[: len(done)]:
        raise ValueError("existing answers are not an exact manifest prefix")

    adapter = load_oe_adapter(args.model)
    try:
        with answers.open("a") as handle:
            for index, row in enumerate(rows[len(done):], start=len(done)):
                item_id = expected[index]
                image_path = args.image_root / str(row["img_name"])
                try:
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                    if args.method == "greedy":
                        generation = adapter.generate_control(
                            image,
                            build_prompt(row),
                            do_sample=False,
                            temperature=0.7,
                            top_p=0.9,
                            num_beams=1,
                            max_new_tokens=args.max_new_tokens,
                            seed=args.seed,
                        )
                        audit = {
                            "method": "greedy",
                            "architecture_port": type(adapter).__name__,
                            "decode_loop": "native_generate",
                        }
                    elif args.method == "vcd":
                        generation, audit = generate_vcd(
                            adapter, image, build_prompt(row),
                            max_new_tokens=args.max_new_tokens, seed=args.seed,
                        )
                    elif args.method == "icd":
                        generation, audit = generate_icd(
                            adapter, image, build_prompt(row),
                            max_new_tokens=args.max_new_tokens, seed=args.seed,
                        )
                    elif args.method == "cve":
                        generation, audit = generate_cve(
                            adapter, image, build_prompt(row),
                            max_new_tokens=args.max_new_tokens, seed=args.seed,
                        )
                    elif args.method == "agla":
                        agla_image = None
                        if row.get("agla_img_name"):
                            with Image.open(str(row["agla_img_name"])) as agla_source:
                                agla_image = agla_source.convert("RGB")
                        generation, audit = generate_agla(
                            adapter, image, build_prompt(row),
                            max_new_tokens=args.max_new_tokens, seed=args.seed,
                            augmented_image=agla_image,
                        )
                    elif args.method == "avisc":
                        generation, audit = generate_avisc(
                            adapter, image, build_prompt(row),
                            max_new_tokens=args.max_new_tokens, seed=args.seed,
                        )
                    elif args.method == "clearsight":
                        generation, audit = generate_clearsight(
                            adapter, image, build_prompt(row),
                            max_new_tokens=args.max_new_tokens, seed=args.seed,
                        )
                    else:
                        generation, audit = generate_dola(
                            adapter, image, build_prompt(row),
                            max_new_tokens=args.max_new_tokens, seed=args.seed,
                            dola_layers="low",
                        )
                    text = generation.text
                    token_ids = list(generation.token_ids)
                    stop_reason = "length" if generation.token_count >= args.max_new_tokens else "eos"
                    input_error = None
                except (UnidentifiedImageError, OSError) as exc:
                    text, token_ids, audit = "", [], {}
                    stop_reason = "input_unavailable"
                    input_error = f"{type(exc).__name__}: {exc}"
                record = {
                    "question_id": item_id,
                    "text": text,
                    "gt_ans": str(row["answer"]),
                    "model_id": args.model,
                    "method": args.method,
                    "metadata": {
                        "generated_token_ids": token_ids,
                        "generated_token_count": len(token_ids),
                        "hit_max_new_tokens": len(token_ids) >= args.max_new_tokens,
                        "stop_reason": stop_reason,
                        "input_error": input_error,
                        "method_audit": audit,
                        "fingerprint": fingerprint,
                    },
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                handle.flush()
                print(f"[{index + 1}/{len(rows)}] {item_id} tokens={len(token_ids)}", flush=True)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
