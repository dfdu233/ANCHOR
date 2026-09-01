#!/usr/bin/env python3
"""Manifest runner for the official HF LLaVA-1.5 VHR implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parents[2]
VHR = ROOT / "third_party/baselines/VHR"
sys.path.insert(0, str(VHR))
from generation import replace_generation  # noqa: E402
from utils import get_layers, get_model  # noqa: E402

VERSION = "vhr-official-manifest-runner-v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method", choices=("greedy", "vhr"), required=True)
    p.add_argument("--native-control", action="store_true", help="Do not install the official VHR generation monkeypatch; valid only with greedy.")
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def row_id(row: dict, index: int) -> str:
    return str(row.get("qid", row.get("question_id", row.get("id", index))))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.native_control and args.method != "greedy":
        raise ValueError("--native-control is valid only for greedy")
    if not args.native_control:
        replace_generation()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = json.loads(args.manifest.read_text())
    if args.limit:
        rows = rows[: args.limit]
    expected = [row_id(row, i) for i, row in enumerate(rows)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    answers = args.output_dir / "answers.jsonl"
    prior = [json.loads(line) for line in answers.read_text().splitlines() if line.strip()] if answers.is_file() else []
    if [str(row["question_id"]) for row in prior] != expected[: len(prior)]:
        raise ValueError("existing VHR answers are not an exact manifest prefix")
    config = {
        "version": VERSION,
        "method": args.method,
        "official_generation_patch_enabled": not args.native_control,
        "official_source_commit": "f0db54a7eae62b4b8d1d585636a446ed40799512",
        "official_generation_sha256": sha(VHR / "generation.py"),
        "official_vhr_sha256": sha(VHR / "vhr.py"),
        "official_main_sha256": sha(VHR / "main.py"),
        "model_path": str(args.model_path.resolve()),
        "model_index_sha256": sha(args.model_path / "model.safetensors.index.json"),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha(args.manifest),
        "image_root": str(args.image_root.resolve()),
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "do_sample": False,
        "num_beams": 1,
        "prompt": "USER: <image>\\n{question} ASSISTANT:",
        "vhr_aug_ratio": 2.0,
        "vhr_last_layers": 14,
        "vhr_layer1": True,
        "vhr_filter": True,
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config["fingerprint"] = fingerprint
    config_path = args.output_dir / "generation_config.json"
    if config_path.is_file() and json.loads(config_path.read_text()).get("fingerprint") != fingerprint:
        raise ValueError("refusing to resume incompatible VHR run")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    device = torch.device("cuda:0")
    processor, model, template, _, num_image_tokens, image_token_id = get_model(str(args.model_path), device, "sdpa")
    if template != "USER: <image>\n{question} ASSISTANT:" or num_image_tokens != 576:
        raise RuntimeError("official LLaVA-1.5 model dispatch contract changed")
    vhr_layers = [1] + list(range(len(get_layers(model))))[-14:]
    with answers.open("a") as handle:
        for index, row in enumerate(rows):
            identifier = expected[index]
            if index < len(prior):
                continue
            image_name = str(row.get("img_name", row.get("image", "")))
            try:
                with Image.open(args.image_root / image_name) as source:
                    image = source.convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                record = {"question_id": identifier, "text": "", "gt_ans": row.get("answer"), "model_id": f"official-{args.method}", "metadata": {"generated_token_ids": [], "generated_token_count": 0, "stop_reason": "input_unavailable", "input_error": f"{type(exc).__name__}: {exc}", "fingerprint": fingerprint}}
                handle.write(json.dumps(record) + "\n"); handle.flush(); continue
            question = str(row.get("question", row.get("text", ""))).strip()
            prompt = template.format(question=question)
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            positions = (inputs["input_ids"] == image_token_id).nonzero()
            if not len(positions):
                raise RuntimeError("official processor produced no image tokens")
            image_start = int(positions[0, 1])
            image_end = int(positions[-1, 1]) + 1
            kwargs = {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens}
            if args.method == "vhr":
                kwargs.update({"vhr_aug_ratio": 2.0, "vhr_layers": vhr_layers, "vhr_filter": True, "vhr_image_start": image_start, "vhr_image_end": image_end})
            output = model.generate(**inputs, **kwargs, output_scores=True, return_dict_in_generate=True)
            count = len(output.scores)
            token_ids = output.sequences[0, -count:].tolist() if count else []
            text = processor.batch_decode(output.sequences[:, -count:], skip_special_tokens=True)[0].strip() if count else ""
            record = {"question_id": identifier, "text": text, "gt_ans": row.get("answer"), "model_id": f"official-{args.method}", "metadata": {"generated_token_ids": token_ids, "generated_token_count": count, "hit_max_new_tokens": count >= args.max_new_tokens, "fingerprint": fingerprint, "vhr_layers": vhr_layers if args.method == "vhr" else None}}
            handle.write(json.dumps(record) + "\n"); handle.flush()
            print(f"[{index+1}/{len(rows)}] {identifier} tokens={count}", flush=True)


if __name__ == "__main__":
    main()
