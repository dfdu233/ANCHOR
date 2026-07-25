#!/usr/bin/env python3
"""Resumable LLaVA-Med OE pilot for source-view KL-barycenter decoding."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from corrected_sgta.cache import load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.infer_ce import make_styles_with_metadata, resize_image
from corrected_sgta.models_oe import load_oe_adapter
from corrected_sgta.protocol_v2 import (
    CACHE_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    file_sha256,
    protocol_fingerprint,
    resolve_image,
)


VERSION = "oe-dg-kl-barycenter-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--feddg-l", type=float, default=0.03)
    parser.add_argument("--feddg-source-ratio", type=float, default=0.5)
    parser.add_argument("--min-style-psnr", type=float, default=20.0)
    parser.add_argument("--min-edge-correlation", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    config = {
        "version": VERSION,
        "model": "llava",
        "dataset_sha256": file_sha256(args.dataset),
        "seed": args.seed,
        "beta": args.beta,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "max_image_side": args.max_image_side,
        "center_policy": "matched",
        "feddg_l": args.feddg_l,
        "feddg_source_ratio": args.feddg_source_ratio,
        "min_style_psnr": args.min_style_psnr,
        "min_edge_correlation": args.min_edge_correlation,
        "decoder": "per-step_original-anchored_KL_barycenter_greedy",
    }
    fingerprint = protocol_fingerprint(config)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "config": config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        if json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError("existing metadata fingerprint mismatch")
    else:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    repair_truncated_jsonl_tail(args.output)
    saved = load_successful_qids(args.output, fingerprint)
    eligible = [
        row
        for row in rows
        if str(row.get("answer", "")).strip()
        and resolve_image(row.get("img_name", "")) is not None
    ]
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row.get('qid')}".encode()
        ).hexdigest()
    )
    if args.max_samples:
        eligible = eligible[: args.max_samples]
    eligible = [row for row in eligible if str(row.get("qid")) not in saved]
    print(
        f"fingerprint={fingerprint[:12]} eligible={len(eligible)} cached={len(saved)}",
        flush=True,
    )
    if not eligible:
        return
    adapter = load_oe_adapter("llava")
    style_args = argparse.Namespace(
        no_feddg=False,
        gammas=(),
        center_policy="matched",
        feddg_l=args.feddg_l,
        feddg_l_values=[args.feddg_l],
        feddg_source_ratio=args.feddg_source_ratio,
        feddg_source_ratios=[args.feddg_source_ratio],
        dataset=args.dataset,
        keep_unsafe_styles=True,
        min_style_psnr=0.0,
        min_edge_correlation=-1.0,
    )
    with args.output.open("a", encoding="utf-8") as stream:
        for sample in tqdm(eligible, desc="OE DG KL smoothing"):
            qid = str(sample.get("qid"))
            try:
                image_path = resolve_image(sample.get("img_name", ""))
                assert image_path is not None
                with Image.open(image_path) as source:
                    original = resize_image(source, args.max_image_side)
                names, images, style_metadata = make_styles_with_metadata(
                    original, sample, style_args
                )
                candidates = [
                    index
                    for index, item in enumerate(style_metadata)
                    if index > 0
                    and item.get("family") == "feddg"
                    and float((item.get("structure") or {}).get("psnr") or -1e9)
                    >= args.min_style_psnr
                    and float(
                        (item.get("structure") or {}).get("edge_correlation") or -1e9
                    )
                    >= args.min_edge_correlation
                ]
                if not candidates:
                    raise RuntimeError("no matched source view passed structure gate")
                style_index = candidates[0]
                baseline, baseline_diagnostics = adapter.generate_dg_smoothed(
                    [original],
                    str(sample["question"]).strip(),
                    beta=0.0,
                    max_new_tokens=args.max_new_tokens,
                )
                adapted, adapted_diagnostics = adapter.generate_dg_smoothed(
                    [original, images[style_index]],
                    str(sample["question"]).strip(),
                    beta=args.beta,
                    max_new_tokens=args.max_new_tokens,
                )
                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "fingerprint": fingerprint,
                    "status": "ok",
                    "qid": sample.get("qid"),
                    "img_name": sample.get("img_name"),
                    "question": sample.get("question"),
                    "answer": sample.get("answer"),
                    "baseline": baseline.__dict__,
                    "dg_smoothed": adapted.__dict__,
                    "baseline_diagnostics": baseline_diagnostics,
                    "dg_diagnostics": adapted_diagnostics,
                    "style_name": names[style_index],
                    "style_metadata": style_metadata[style_index],
                }
            except Exception as error:
                traceback.print_exc()
                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "fingerprint": fingerprint,
                    "status": "error",
                    "qid": sample.get("qid"),
                    "error": f"{type(error).__name__}: {error}"[:500],
                }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    adapter.close()


if __name__ == "__main__":
    main()
