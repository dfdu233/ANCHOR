#!/usr/bin/env python3
"""Paired inference-only DG validation against an existing native baseline.

The baseline answers are never regenerated.  Only transformed views are sent
through the model, so every output can be compared to the exact completed
native answer for the same question/image.  The script is resumable and is
intended to run behind the shared GPU lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from corrected_sgta.methods import feddg_frequency_interpolation


VERSION = "dg-paired-validation-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qid(row: dict[str, Any]) -> str:
    return str(row.get("qid", row.get("question_id", row.get("id"))))


def stable_seed(seed: int, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:4], "big") & 0x7FFFFFFF


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-answers", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="feddg", choices=("feddg", "gamma"))
    parser.add_argument("--alpha", type=float, default=0.01, help="low-frequency window ratio for feddg")
    parser.add_argument("--source-ratio", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "answers.jsonl"
    config = {
        "version": VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "baseline_answers": str(args.baseline_answers.resolve()),
        "baseline_sha256": sha256(args.baseline_answers),
        "image_root": str(args.image_root.resolve()),
        "source_bank": str(args.source_bank.resolve()),
        "source_bank_sha256": sha256(args.source_bank),
        "variant": args.variant,
        "alpha": args.alpha,
        "source_ratio": args.source_ratio,
        "gamma": args.gamma,
        "limit": args.limit,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = args.output_dir / "generation_config.json"
    if config_path.exists() and json.loads(config_path.read_text()).get("fingerprint") != config["fingerprint"]:
        raise RuntimeError("refusing to resume incompatible DG validation config")
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    manifest = load_jsonl(args.manifest)
    baseline = {qid(row): row for row in load_jsonl(args.baseline_answers)}
    rows = [row for row in manifest if qid(row) in baseline]
    if args.limit > 0:
        rows = rows[: args.limit]
    expected = [qid(row) for row in rows]
    done = load_jsonl(output) if output.exists() else []
    done_ids = [qid(row) for row in done]
    if done_ids != expected[: len(done_ids)] or len(done_ids) != len(set(done_ids)):
        raise RuntimeError("existing DG answers are not an exact manifest prefix")
    remaining = rows[len(done):]
    if not remaining:
        return

    import numpy as np
    from corrected_sgta.models_oe import load_oe_adapter

    bank = np.load(args.source_bank)
    adapter = load_oe_adapter("huatuo", llava_conv_mode="mistral_instruct")
    try:
        for row in remaining:
            item_id = qid(row)
            relative = str(row.get("img_name", row.get("image", "")))
            image_path = args.image_root / relative
            with Image.open(image_path) as source:
                original = source.convert("RGB")
            if args.variant == "feddg":
                transformed = feddg_frequency_interpolation(
                    original, bank, low_frequency_ratio=args.alpha, source_ratio=args.source_ratio
                )
                variant_name = f"feddg_l{args.alpha:g}_sr{args.source_ratio:g}"
            else:
                from corrected_sgta.methods import gamma_transform
                transformed = gamma_transform(original, args.gamma)
                variant_name = f"gamma_{args.gamma:g}"
            view_path = args.output_dir / "views" / variant_name / relative
            view_path.parent.mkdir(parents=True, exist_ok=True)
            transformed.save(view_path)
            result = adapter.generate_control(
                image=transformed,
                prompt=str(row["question"]),
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                seed=stable_seed(args.seed, item_id),
            )
            native = baseline[item_id]
            append_jsonl(output, {
                "question_id": item_id,
                "text": result.text,
                "gt_ans": str(row.get("answer", native.get("gt_ans", ""))),
                "model_id": "huatuo",
                "metadata": {
                    "variant": variant_name,
                    "view_path": str(view_path.resolve()),
                    "generated_token_count": result.token_count,
                    "generated_token_ids": list(result.token_ids),
                    "mean_token_nll": result.uncertainty,
                    "native_text": native.get("text", ""),
                    "native_token_ids": native.get("metadata", {}).get("generated_token_ids", []),
                    "fingerprint": config["fingerprint"],
                },
            })
    finally:
        adapter.close()
    if len(load_jsonl(output)) != len(rows):
        raise RuntimeError("DG validation output is incomplete")


if __name__ == "__main__":
    main()
