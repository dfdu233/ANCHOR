#!/usr/bin/env python3
"""Run task-general Huatuo inference with one image-side DG calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corrected_sgta.evaluate_rule_vqa import evaluate_rule_rows
from corrected_sgta.local_source_projection import (
    LocalSourceIndex,
    load_local_source_index,
    local_source_projection,
    source_mean_std_projection,
)
from corrected_sgta.mosec import (
    gamma_style_shift,
    load_bank,
    model_visible_image,
    radial_mean_calibration,
    source_envelope_calibration,
    stable_sha256,
)
from corrected_sgta.oe_metrics import lexical_metrics


DEFAULT_MODEL = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO = Path("/home/dbw/HuatuoGPT-Vision")
DEFAULT_BANK = Path("/home/dbw/data/mosec_banks/huatuo_pubmedvision_cxr/cxr_radial_envelope.npz")
DEFAULT_LOCAL_INDEX = Path(
    "/home/dbw/data/mosec_banks/huatuo_pubmedvision_cxr_v2/"
    "cxr_local_source_index.npz"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("input JSON must contain a list")
    return payload


def row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("qid", row.get("question_id", row.get("id", index))))


def image_name(row: dict[str, Any]) -> str:
    value = row.get("img_name", row.get("image"))
    if isinstance(value, list):
        value = value[0]
    if not value:
        raise ValueError("row has no image/img_name")
    return str(value)


def select_rows(
    rows: list[dict[str, Any]],
    maximum: int,
    seed: int,
    offset: int = 0,
) -> list[dict[str, Any]]:
    scored = [
        (
            hashlib.sha256(f"{seed}:{row_id(row, index)}".encode()).hexdigest(),
            row,
        )
        for index, row in enumerate(rows)
    ]
    ordered = [row for _, row in sorted(scored, key=lambda item: item[0])]
    if maximum <= 0:
        return ordered[offset:]
    return ordered[offset : offset + maximum]


def import_huatuo(root: Path):
    sys.path.insert(0, str(root))
    from cli import HuatuoChatbot  # type: ignore

    return HuatuoChatbot


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    for method in methods:
        if method == "native":
            continue
        if method.startswith("mean_w"):
            float(method.removeprefix("mean_w"))
            continue
        match = re.fullmatch(r"meanlow(\d+)_w(.+)", method)
        if match:
            if int(match.group(1)) <= 0:
                raise ValueError(f"invalid active bins: {method}")
            float(match.group(2))
            continue
        if method.startswith("envelope_s"):
            float(method.removeprefix("envelope_s"))
            continue
        if method.startswith("local_l"):
            value = float(method.removeprefix("local_l"))
            if not 0.0 < value <= 0.5:
                raise ValueError(f"invalid local low-frequency ratio: {method}")
            continue
        if method.startswith("sourcestats_s"):
            value = float(method.removeprefix("sourcestats_s"))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"invalid source-stat strength: {method}")
            continue
        raise ValueError(f"unknown method: {method}")
    if "native" not in methods:
        methods.insert(0, "native")
    return methods


def calibrated_image(
    method: str,
    visible: Image.Image,
    bank: dict[str, np.ndarray],
    local_index: LocalSourceIndex | None,
) -> tuple[Image.Image, dict[str, Any]]:
    if method == "native":
        return visible.copy(), {
            "identity": True,
            "changed_band_count": 0,
            "structure": {"mse": 0.0, "psnr": None, "edge_correlation": 1.0},
        }
    if method.startswith("mean_w"):
        weight = float(method.removeprefix("mean_w"))
        image, metadata = radial_mean_calibration(
            visible, bank["mean"], source_weight=weight
        )
        metadata["source_weight"] = weight
        return image, metadata
    match = re.fullmatch(r"meanlow(\d+)_w(.+)", method)
    if match:
        active_bins = int(match.group(1))
        weight = float(match.group(2))
        image, metadata = radial_mean_calibration(
            visible,
            bank["mean"],
            source_weight=weight,
            active_bins=active_bins,
        )
        metadata["source_weight"] = weight
        return image, metadata
    if method.startswith("local_l"):
        if local_index is None:
            raise ValueError(f"{method} requires --local-index")
        low_frequency_ratio = float(method.removeprefix("local_l"))
        return local_source_projection(
            visible,
            local_index,
            low_frequency_ratio=low_frequency_ratio,
        )
    if method.startswith("sourcestats_s"):
        if local_index is None:
            raise ValueError(f"{method} requires --local-index")
        strength = float(method.removeprefix("sourcestats_s"))
        return source_mean_std_projection(
            visible, local_index, strength=strength
        )
    strength = float(method.removeprefix("envelope_s"))
    image, metadata = source_envelope_calibration(
        visible,
        bank["lower"],
        bank["upper"],
        bank["scale"],
        strength=strength,
    )
    metadata["strength"] = strength
    return image, metadata


def aggregate_oe(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        lexical_metrics(str(row["text"]), str(row["ground_truth"]))
        for row in records
        if str(row.get("text") or "").strip() and str(row.get("ground_truth") or "").strip()
    ]
    metrics = {
        key: float(np.mean([row[key] for row in values]))
        for key in values[0]
    } if values else {}
    texts = [str(row.get("text") or "").strip() for row in records]
    return {
        "n": len(records),
        "n_evaluated": len(values),
        "metrics": metrics,
        "diagnostics": {
            "empty_rate": sum(not text for text in texts) / max(len(texts), 1),
            "unique_prediction_rate": len(set(texts)) / max(len(texts), 1),
            "mean_output_words": float(np.mean([len(text.split()) for text in texts])) if texts else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["rule_ce", "open_vqa", "report_generation"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--local-index", type=Path, default=DEFAULT_LOCAL_INDEX)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=parse_methods(
            "native,mean_w0.05,mean_w0.1,mean_w0.2,"
            "envelope_s0.25,envelope_s0.5,envelope_s1.0"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--input-gamma", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--allow-truncated-images",
        action="store_true",
        help="Allow Pillow to decode otherwise readable JPEGs with truncated tails.",
    )
    args = parser.parse_args()
    if args.input_gamma <= 0.0:
        parser.error("--input-gamma must be positive")
    if args.sample_offset < 0:
        parser.error("--sample-offset must be non-negative")
    ImageFile.LOAD_TRUNCATED_IMAGES = args.allow_truncated_images

    rows = select_rows(
        load_rows(args.input),
        args.max_samples,
        args.seed,
        args.sample_offset,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_generations.jsonl"
    if raw_path.exists():
        raise SystemExit(f"refusing to overwrite existing run: {raw_path}")
    bank = load_bank(args.bank)
    needs_local_index = any(
        method.startswith(("local_l", "sourcestats_s"))
        for method in args.methods
    )
    local_index = (
        load_local_source_index(args.local_index) if needs_local_index else None
    )
    config = {
        "version": "huatuo-mosec-runner-v1",
        "created_at": now_iso(),
        "command": " ".join(sys.argv),
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "image_root": str(args.image_root),
        "dataset": args.dataset,
        "task": args.task,
        "n": len(rows),
        "selection": "stable_sha256",
        "sample_offset": args.sample_offset,
        "seed": args.seed,
        "methods": args.methods,
        "bank": str(args.bank),
        "bank_sha256": file_sha256(args.bank),
        "local_index": str(args.local_index) if needs_local_index else None,
        "local_index_sha256": (
            file_sha256(args.local_index) if needs_local_index else None
        ),
        "model": str(args.model_dir),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": 1,
            "repetition_penalty": 1.2,
        },
        "prompt_policy": "raw dataset question; Huatuo handles the image placeholder",
        "allow_truncated_images": args.allow_truncated_images,
        "input_style": {"type": "gamma", "gamma": args.input_gamma},
        "pid": os.getpid(),
    }
    write_json(args.output_dir / "config.json", config)
    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device=args.device)
    bot.gen_kwargs.update(config["generation"])
    bot.gen_kwargs.pop("temperature", None)

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        identifier = row_id(row, index)
        prompt = str(row.get("question") or "")
        path = args.image_root / image_name(row)
        record_base = {
            "item_id": identifier,
            "qid": identifier,
            "question_id": identifier,
            "dataset": args.dataset,
            "task": args.task,
            "image": str(path),
            "question": prompt,
            "prompt_sha256": stable_sha256(prompt),
            "ground_truth": str(row.get("answer") or row.get("report") or ""),
        }
        try:
            with Image.open(path) as source:
                visible = model_visible_image(source.convert("RGB"))
                visible = gamma_style_shift(visible, args.input_gamma)
        except Exception as exc:
            for method in args.methods:
                failed = {
                    **record_base,
                    "method": method,
                    "text": "",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(raw_path, failed)
                by_method[method].append(failed)
            continue

        for method in args.methods:
            started = time.time()
            current = {
                **record_base,
                "method": method,
                "text": "",
                "error": None,
            }
            try:
                image, calibration = calibrated_image(
                    method, visible, bank, local_index
                )
                torch.manual_seed(args.seed + index)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed + index)
                response = bot.inference(prompt, [image])
                text = str(response[0] if response else "").strip()
                generated_token_count = len(
                    bot.tokenizer(text, add_special_tokens=False).input_ids
                )
                current.update(
                    {
                        "text": text,
                        "generated_token_count": generated_token_count,
                        "hit_max_new_tokens": (
                            generated_token_count >= args.max_new_tokens
                        ),
                        "calibration": calibration,
                    }
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                current["error"] = f"CUDA OOM: {exc}"
                current["traceback"] = traceback.format_exc()
            except Exception as exc:
                current["error"] = repr(exc)
                current["traceback"] = traceback.format_exc()
            current["elapsed_sec"] = time.time() - started
            current["completed_at"] = now_iso()
            append_jsonl(raw_path, current)
            by_method[method].append(current)
        if (index + 1) == 1 or (index + 1) % args.progress_every == 0 or (index + 1) == len(rows):
            print(
                json.dumps(
                    {
                        "progress": f"{index + 1}/{len(rows)}",
                        "item_id": identifier,
                        "outputs": {
                            method: by_method[method][-1].get("text", "")[:120]
                            for method in args.methods
                        },
                        "errors": {
                            method: by_method[method][-1].get("error")
                            for method in args.methods
                            if by_method[method][-1].get("error")
                        },
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary: dict[str, Any] = {"dataset": args.dataset, "task": args.task, "n": len(rows), "methods": {}}
    for method, records in by_method.items():
        method_path = args.output_dir / method
        method_path.mkdir(parents=True, exist_ok=True)
        answers = [
            {
                "question_id": row["question_id"],
                "qid": row["qid"],
                "item_id": row["item_id"],
                "answer": row["text"],
                "model_answer": row["text"],
                "text": row["text"],
                "ground_truth": row["ground_truth"],
                "gt_answer": row["ground_truth"],
            }
            for row in records
        ]
        answers_path = method_path / "answers.jsonl"
        answers_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in answers))
        errors = sum(row.get("error") is not None for row in records)
        calibration_rows = [row.get("calibration") or {} for row in records]
        changed_bands = [
            float(value)
            for row in calibration_rows
            if isinstance(
                (value := row.get("changed_band_count")), (int, float)
            )
        ]
        method_summary: dict[str, Any] = {
            "n": len(records),
            "errors": errors,
            "error_rate": errors / max(len(records), 1),
            "identity_rate": sum(bool(row.get("identity")) for row in calibration_rows) / max(len(records), 1),
            "mean_elapsed_sec": float(np.mean([row.get("elapsed_sec", 0.0) for row in records])),
            "mean_changed_bands": (
                float(np.mean(changed_bands)) if changed_bands else None
            ),
            "hit_max_new_tokens_rate": (
                sum(bool(row.get("hit_max_new_tokens")) for row in records)
                / max(len(records), 1)
            ),
        }
        if args.task == "rule_ce":
            metrics, evaluation_records = evaluate_rule_rows(rows, answers)
            write_json(method_path / "metrics.json", metrics)
            (method_path / "records.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in evaluation_records)
            )
            method_summary["metrics"] = metrics
            method_summary["huatuo_primary"] = {
                "name": "decision_first.accuracy_invalid_as_error",
                "accuracy": metrics["decision_first"]["accuracy_invalid_as_error"],
                "parse_rate": metrics["decision_first"]["parse_rate"],
            }
        else:
            method_summary["metrics"] = aggregate_oe(records)
            write_json(method_path / "metrics.json", method_summary["metrics"])
        summary["methods"][method] = method_summary
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
