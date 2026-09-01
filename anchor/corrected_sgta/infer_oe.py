#!/usr/bin/env python3
"""Resumable OE evidence generation for Hulu-Med and LLaVA-Med.

The cache contains a separate greedy baseline, an original-image sampled
stream, and (optionally) an equal-budget style-augmented sampled stream.  No
ground-truth-derived score is stored or used during generation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
import traceback
from itertools import zip_longest
from pathlib import Path

import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.infer_ce import make_styles_with_metadata, resize_image
from corrected_sgta.models_oe import Generation, load_oe_adapter
from corrected_sgta.report_protocol import (
    is_report_generation_row as protocol_is_report_generation_row,
    report_prompt as protocol_report_prompt,
)
from corrected_sgta.protocol_v2 import (
    CACHE_SCHEMA_VERSION,
    IMAGE_ROOT,
    PROTOCOL_VERSION,
    file_sha256,
    protocol_fingerprint,
    resolve_image,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
# ``expandable_segments`` is unavailable in the pinned PyTorch used by the
# local LLaVA-Med environment.  Keep an allocator hint that works across both
# old and current runtimes; it is operational metadata, not a method setting.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("hulu", "llava"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--candidate-batch", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--llava-conv-mode",
        default="mistral_instruct",
        help="Conversation template for LLaVA-Med OE generation; official LLaVA-Med v1.5 uses mistral_instruct.",
    )
    parser.add_argument(
        "--report-prompt-mode",
        choices=(
            "dataset", "official_zero_shot", "official_rag", "mmedrag", "structured", "impression", "abnormality_focused",
        ),
        default="official_zero_shot",
        help="Released RULE/MMed-RAG report wording without retrieval leakage.",
    )
    parser.add_argument("--style-augmentation", action="store_true")
    parser.add_argument("--gammas", type=float, nargs="*", default=(0.8, 1.2))
    parser.add_argument(
        "--center-policy", choices=("matched", "inferred", "all"), default="matched"
    )
    parser.add_argument("--max-style-views", type=int, default=2)
    parser.add_argument(
        "--selector", choices=("conservative", "consistency"), default="conservative"
    )
    parser.add_argument("--feddg-l", type=float, default=0.003)
    parser.add_argument("--feddg-l-values", type=float, nargs="*", default=None)
    parser.add_argument("--feddg-source-ratio", type=float, default=0.0)
    parser.add_argument("--feddg-source-ratios", type=float, nargs="*", default=None)
    parser.add_argument("--min-style-psnr", type=float, default=20.0)
    parser.add_argument("--min-edge-correlation", type=float, default=0.90)
    return parser.parse_args()


def qid_seed(seed: int, qid: object) -> int:
    digest = hashlib.sha256(f"{seed}:{qid}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def candidate_id(
    qid: object,
    stream: str,
    style: str,
    index: int,
    run_fingerprint: str,
    text_value: str,
) -> str:
    payload = json.dumps(
        {
            "fingerprint": run_fingerprint,
            "qid": str(qid),
            "stream": stream,
            "style": style,
            "index": int(index),
            "text_sha256": hashlib.sha256(text_value.strip().encode()).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def as_dict(
    value: Generation,
    style: str,
    position: int,
    *,
    qid: object | None = None,
    stream: str = "unknown",
    style_metadata: dict | None = None,
    run_fingerprint: str = "",
) -> dict:
    metadata = style_metadata or {
        "family": "original", "domain_id": "original", "parameters": {}
    }
    result = {
        "text": value.text,
        "uncertainty": value.uncertainty if math.isfinite(value.uncertainty) else None,
        "token_count": value.token_count,
        "style": style,
        "position": position,
        "acquisition_step": position,
        "domain_id": metadata.get("domain_id", "unknown"),
        "transform_family": metadata.get("family", "unknown"),
        "transform_parameters": metadata.get("parameters") or {},
        "center_file": metadata.get("center_file"),
        "center_distance": metadata.get("center_distance"),
        "structure": metadata.get("structure"),
    }
    if qid is not None:
        result["candidate_id"] = candidate_id(
            qid, stream, style, position, run_fingerprint, value.text
        )
    return result


def select_style_views(
    names: list[str],
    metadata: list[dict],
    max_style_views: int,
    selector: str,
    min_psnr: float = 20.0,
    min_edge_correlation: float = 0.90,
) -> list[int]:
    """Choose at most two modality-matched FedDG views without references or labels."""

    eligible = []
    for index, item in enumerate(metadata):
        structure = item.get("structure") or {}
        psnr = structure.get("psnr")
        edge = structure.get("edge_correlation")
        if (
            index > 0
            and item.get("family") == "feddg"
            and psnr is not None
            and float(psnr) >= min_psnr
            and edge is not None
            and float(edge) >= min_edge_correlation
        ):
            eligible.append(index)
    if max_style_views <= 0 or not eligible:
        return [0]
    def distortion(index: int) -> tuple[float, float, int]:
        structure = metadata[index].get("structure") or {}
        return (
            float(structure.get("pixel_mse") or 0.0),
            -float(structure.get("edge_correlation") or 1.0),
            index,
        )
    ordered = sorted(eligible, key=distortion)
    if selector == "conservative" or max_style_views == 1:
        chosen = ordered[:max_style_views]
    else:
        # Consistency mode spans the safe intensity range, then lets the v2
        # analyzer reward agreement. This order is fixed before seeing text.
        chosen = [ordered[0]]
        if len(ordered) > 1:
            chosen.append(ordered[-1])
        chosen.extend(index for index in ordered[1:-1] if index not in chosen)
        chosen = chosen[:max_style_views]
    return [0, *chosen]


def original_quota_rows(raw_original_rows: list[dict], quota: int) -> list[dict]:
    """Allocate SGTA's original quota before any confidence-based ordering."""

    return [dict(row) for row in raw_original_rows[:quota]]


def is_report_generation_row(sample: dict) -> bool:
    """Return whether the row should use report-generation protocol logic."""
    return protocol_is_report_generation_row(sample)


def report_prompt(sample: dict, mode: str) -> str:
    """Return a report prompt without using target references or labels."""
    return protocol_report_prompt(sample, mode)


def risk_order(candidates: list[dict]) -> list[dict]:
    """Reference-independent acquisition order: confident, stable candidates first."""

    ordered = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda candidate: (
            1e6 if candidate.get("uncertainty") is None else float(candidate["uncertainty"]),
            str(candidate.get("candidate_id", "")),
        ),
    )
    for step, candidate in enumerate(ordered):
        candidate["position"] = step
        candidate["acquisition_step"] = step
        candidate["acquisition_policy"] = "ascending_sequence_nll_then_candidate_id"
    return ordered


def sample_style(
    adapter, image, prompt: str, count: int, args, seed: int
) -> list[Generation]:
    """Sample one style, falling back to B=1 for remote-code constraints."""

    outputs: list[Generation] = []
    for start in range(0, count, max(1, args.candidate_batch)):
        chunk_size = min(max(1, args.candidate_batch), count - start)
        chunk_seed = seed + start
        adapter._seed(chunk_seed)
        try:
            chunk = adapter._generate_once(
                image,
                prompt,
                chunk_size,
                True,
                args.temperature,
                args.top_p,
                args.max_new_tokens,
                chunk_seed,
            )
        except (RuntimeError, AssertionError, ValueError):
            gc.collect()
            torch.cuda.empty_cache()
            chunk = []
            for offset in range(chunk_size):
                item_seed = chunk_seed + offset
                adapter._seed(item_seed)
                chunk.extend(
                    adapter._generate_once(
                        image,
                        prompt,
                        1,
                        True,
                        args.temperature,
                        args.top_p,
                        args.max_new_tokens,
                        item_seed,
                    )
                )
        outputs.extend(chunk)
    if len(outputs) != count:
        raise RuntimeError(f"expected {count} samples for style, got {len(outputs)}")
    return outputs


def round_robin(buckets: list[list[dict]]) -> list[dict]:
    output = []
    for group in zip_longest(*buckets):
        output.extend(item for item in group if item is not None)
    output = [dict(item) for item in output]
    for position, item in enumerate(output):
        item["position"] = position
        item["acquisition_step"] = position
    return output


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    config = {
        "model": args.model,
        "dataset_sha256": file_sha256(args.dataset),
        "image_root": str(IMAGE_ROOT.resolve()),
        "max_image_side": args.max_image_side,
        "candidates": args.candidates,
        "candidate_batch": args.candidate_batch,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "llava_conv_mode": args.llava_conv_mode,
        "report_prompt_mode": args.report_prompt_mode,
        "style_augmentation": args.style_augmentation,
        "style_family": (
            "domain_center_bank" if args.center_policy == "all" else "modality_matched_center"
        ),
        "max_style_views": args.max_style_views,
        "selector": args.selector,
        "structure_gate": {
            "min_psnr": args.min_style_psnr,
            "min_edge_correlation": args.min_edge_correlation,
        },
        "acquisition_policy": "ascending_sequence_nll_then_candidate_id",
        "gammas": list(args.gammas),
        "center_policy": args.center_policy,
        "feddg_l_values": list(args.feddg_l_values or [args.feddg_l]),
        "feddg_source_ratios": list(
            args.feddg_source_ratios or [args.feddg_source_ratio]
        ),
        "candidate_score": "mean_processed_sampling_token_nll",
        "greedy_is_separate": True,
        "generation_attention_mask": (
            "explicit_all_ones"
            if args.model.lower().replace("_", "-").startswith("llava")
            else "processor_supplied"
        ),
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "dataset_n": len(rows),
        "missing_images": sum(
            resolve_image(row.get("img_name", "")) is None for row in rows
        ),
    }
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"metadata mismatch; choose a new output path: {metadata_path}"
            )
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2))

    repair = repair_truncated_jsonl_tail(args.output)
    if repair["action"] != "none":
        print(f"cache tail repair: {repair}", flush=True)
    saved = load_successful_qids(args.output, fingerprint)
    target_rows = [
        row
        for row in rows
        if str(row.get("answer", "")).strip()
        and resolve_image(row.get("img_name", "")) is not None
    ]
    target_rows.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row.get('qid')}".encode()
        ).hexdigest()
    )
    if args.max_samples:
        target_rows = target_rows[: args.max_samples]
    eligible = [row for row in target_rows if str(row.get("qid")) not in saved]
    print(
        f"protocol={PROTOCOL_VERSION} fingerprint={fingerprint[:12]} "
        f"eligible={len(eligible)} cached={len(saved)}",
        flush=True,
    )
    if not eligible:
        return

    adapter = load_oe_adapter(args.model, llava_conv_mode=args.llava_conv_mode)
    print(f"Loaded {adapter.name}", flush=True)
    style_args = argparse.Namespace(
        no_feddg=False,
        gammas=args.gammas,
        center_policy=args.center_policy,
        feddg_l=args.feddg_l,
        feddg_l_values=args.feddg_l_values,
        feddg_source_ratio=args.feddg_source_ratio,
        feddg_source_ratios=args.feddg_source_ratios,
        dataset=args.dataset,
    )
    started = time.time()
    errors = 0
    with args.output.open("a") as handle:
        for sample in tqdm(eligible, desc=f"OE {args.model}"):
            try:
                image_path = resolve_image(sample.get("img_name", ""))
                assert image_path is not None
                with Image.open(image_path) as source:
                    image = resize_image(source, args.max_image_side)
                prompt = report_prompt(sample, args.report_prompt_mode)
                seed = qid_seed(args.seed, sample["qid"])
                greedy, original = adapter.generate_oe(
                    image,
                    prompt,
                    args.candidates,
                    args.temperature,
                    args.top_p,
                    args.max_new_tokens,
                    seed,
                    args.candidate_batch,
                )
                raw_original_rows = [
                    as_dict(
                        value,
                        "original",
                        i,
                        qid=sample["qid"],
                        stream="original",
                        run_fingerprint=fingerprint,
                    )
                    for i, value in enumerate(original)
                ]
                original_rows = risk_order(raw_original_rows)
                style_rows = None
                if args.style_augmentation:
                    style_names, style_images, style_metadata = make_styles_with_metadata(
                        image, sample, style_args
                    )
                    selected_indices = select_style_views(
                        style_names,
                        style_metadata,
                        args.max_style_views,
                        args.selector,
                        args.min_style_psnr,
                        args.min_edge_correlation,
                    )
                    quotas = [args.candidates // len(selected_indices)] * len(selected_indices)
                    for index in range(args.candidates % len(selected_indices)):
                        quotas[index] += 1
                    buckets = [[
                        {
                            **dict(value),
                            "candidate_id": candidate_id(
                                sample["qid"],
                                "style",
                                "original",
                                index,
                                fingerprint,
                                str(value.get("text", "")),
                            ),
                            "selection_reason": "original_anchor",
                        }
                        for index, value in enumerate(
                            original_quota_rows(raw_original_rows, quotas[0])
                        )
                    ]]
                    for quota_index, style_index in enumerate(selected_indices[1:], start=1):
                        values = sample_style(
                            adapter,
                            style_images[style_index],
                            prompt,
                            quotas[quota_index],
                            args,
                            seed + style_index * 100003,
                        )
                        buckets.append(
                            [
                                {
                                    **as_dict(
                                        value,
                                        style_names[style_index],
                                        i,
                                        qid=sample["qid"],
                                        stream="style",
                                        style_metadata=style_metadata[style_index],
                                        run_fingerprint=fingerprint,
                                    ),
                                    "selection_reason": f"{args.selector}_matched_center",
                                }
                                for i, value in enumerate(values)
                            ]
                        )
                    style_rows = risk_order(
                        [candidate for bucket in buckets for candidate in bucket]
                    )
                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "fingerprint": fingerprint,
                    "status": "ok",
                    "qid": sample["qid"],
                    "img_name": sample.get("img_name", ""),
                    "question": sample["question"],
                    "prompt_used": prompt,
                    "report_prompt_mode": args.report_prompt_mode,
                    "llava_conv_mode": args.llava_conv_mode if args.model.lower().replace("_", "-").startswith("llava") else None,
                    "answer": sample["answer"],
                    "structured_answer": sample.get("structured_answer"),
                    "greedy": as_dict(
                        greedy,
                        "original",
                        0,
                        qid=sample["qid"],
                        stream="greedy",
                        run_fingerprint=fingerprint,
                    ),
                    "sampled": original_rows,
                    "style_sampled": style_rows,
                }
            except Exception as exc:
                errors += 1
                traceback.print_exc()
                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "fingerprint": fingerprint,
                    "status": "error",
                    "qid": sample.get("qid"),
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    gc.collect()
                    torch.cuda.empty_cache()
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
    elapsed = time.time() - started
    print(
        f"Finished {len(eligible)} rows in {elapsed / 60:.2f} min; errors={errors}",
        flush=True,
    )
    adapter.close()


if __name__ == "__main__":
    main()
