#!/usr/bin/env python3
"""Run a paired source-shift and source-repair causal probe with Huatuo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from corrected_sgta.mosec import (
    _reconstruct_with_radial_delta,
    load_bank,
    model_visible_image,
    radial_log_amplitude,
    source_envelope_calibration,
    stable_sha256,
    structure_metrics,
)
from corrected_sgta.oe_metrics import lexical_metrics
from corrected_sgta.run_huatuo_mosec import (
    file_sha256,
    image_name,
    import_huatuo,
    load_rows,
    row_id,
    select_rows,
    write_json,
)


DEFAULT_MODEL = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO = Path("/home/dbw/HuatuoGPT-Vision")
DEFAULT_SOURCE_BANK = Path(
    "/home/dbw/data/mosec_banks/huatuo_pubmedvision_cxr_v2/"
    "cxr_radial_envelope_c50.npz"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_strengths(value: str) -> tuple[float, ...]:
    strengths = tuple(float(item) for item in value.split(",") if item.strip())
    if not strengths or any(not 0.0 < item <= 1.0 for item in strengths):
        raise argparse.ArgumentTypeError("strengths must be comma-separated values in (0,1]")
    return strengths


def append_jsonl(path: Path, payload: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def bank_distance(image: Image.Image, bank: dict[str, np.ndarray]) -> dict[str, Any]:
    descriptor = radial_log_amplitude(image).astype(np.float64)
    scale = np.maximum(bank["scale"].astype(np.float64), 1e-6)
    outside = np.maximum(bank["lower"] - descriptor, 0.0)
    outside += np.maximum(descriptor - bank["upper"], 0.0)
    return {
        "outside_band_count": int(np.count_nonzero(outside > 1e-8)),
        "mean_normalized_exceedance": float(np.mean(outside / scale)),
        "mean_center_distance": float(
            np.mean(np.abs(descriptor - bank["median"]) / scale)
        ),
    }


def build_conditions(
    visible: Image.Image,
    source_bank: dict[str, np.ndarray],
    shift_bank: dict[str, np.ndarray],
    mismatch_bank: dict[str, np.ndarray],
    strengths: tuple[float, ...],
) -> dict[str, tuple[Image.Image, dict[str, Any]]]:
    conditions: dict[str, tuple[Image.Image, dict[str, Any]]] = {
        "native": (
            visible.copy(),
            {
                "family": "native",
                "identity": True,
                "structure": {
                    "mse": 0.0,
                    "psnr": None,
                    "edge_correlation": 1.0,
                },
            },
        )
    }
    domain_residual = (
        shift_bank["median"].astype(np.float64)
        - source_bank["median"].astype(np.float64)
    )
    for strength in strengths:
        label = str(strength).replace(".", "p")
        radial_delta = strength * domain_residual
        shifted = _reconstruct_with_radial_delta(visible, radial_delta)
        shift_meta = {
            "identity": bool(np.all(np.abs(radial_delta) <= 1e-8)),
            "changed_band_count": int(
                np.count_nonzero(np.abs(radial_delta) > 1e-8)
            ),
            "mean_abs_log_gain": float(np.mean(np.abs(radial_delta))),
            "max_abs_log_gain": float(np.max(np.abs(radial_delta))),
            "structure": structure_metrics(visible, shifted),
        }
        shift_name = f"shift_l{label}"
        shift_meta.update(
            {
                "family": "shift",
                "operator": "target_median_minus_model_source_median",
                "shift_strength": strength,
                "parent_shift": shift_name,
            }
        )
        conditions[shift_name] = (shifted, shift_meta)

        matched, matched_meta = source_envelope_calibration(
            shifted,
            source_bank["lower"],
            source_bank["upper"],
            source_bank["scale"],
            strength=1000.0,
        )
        matched_meta.update(
            {
                "family": "repair_matched",
                "shift_strength": strength,
                "parent_shift": shift_name,
                "projection": "exact_envelope_clip",
            }
        )
        conditions[f"{shift_name}_repair_matched"] = (matched, matched_meta)

        mismatched, mismatch_meta = source_envelope_calibration(
            shifted,
            mismatch_bank["lower"],
            mismatch_bank["upper"],
            mismatch_bank["scale"],
            strength=1000.0,
        )
        mismatch_meta.update(
            {
                "family": "repair_mismatched",
                "shift_strength": strength,
                "parent_shift": shift_name,
                "projection": "exact_envelope_clip",
            }
        )
        conditions[f"{shift_name}_repair_mismatched"] = (
            mismatched,
            mismatch_meta,
        )

    for image, metadata in conditions.values():
        metadata["structure_from_native"] = structure_metrics(visible, image)
        metadata["distance"] = {
            "source": bank_distance(image, source_bank),
            "shift": bank_distance(image, shift_bank),
            "mismatch": bank_distance(image, mismatch_bank),
        }
    return conditions


def aggregate_oe(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for row in records:
        metrics = lexical_metrics(row.get("text", ""), row["ground_truth"])
        scored.append({**row, **metrics})
    return {
        "n": len(scored),
        "rouge_l": float(np.mean([row["rouge_l"] for row in scored])),
        "token_f1": float(np.mean([row["token_f1"] for row in scored])),
        "mean_output_words": float(
            np.mean([len(str(row.get("text", "")).split()) for row in scored])
        ),
        "records": scored,
    }


def ce_pairing(
    evaluations: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, int]]:
    native = {
        str(row["question_id"]): row for row in evaluations.get("native", [])
    }
    output = {}
    for method, rows in evaluations.items():
        if method == "native":
            continue
        rescue = harm = same_correct = same_wrong = 0
        for row in rows:
            baseline = native[str(row["question_id"])]
            before = bool(baseline["decision_first_correct"])
            after = bool(row["decision_first_correct"])
            rescue += int(not before and after)
            harm += int(before and not after)
            same_correct += int(before and after)
            same_wrong += int(not before and not after)
        output[method] = {
            "rescue": rescue,
            "harm": harm,
            "same_correct": same_correct,
            "same_wrong": same_wrong,
        }
    return output


def oe_pairing(
    scored: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    native = {
        str(row["item_id"]): row for row in scored["native"]["records"]
    }
    output = {}
    for method, summary in scored.items():
        if method == "native":
            continue
        deltas_rouge = []
        deltas_f1 = []
        for row in summary["records"]:
            baseline = native[str(row["item_id"])]
            deltas_rouge.append(row["rouge_l"] - baseline["rouge_l"])
            deltas_f1.append(row["token_f1"] - baseline["token_f1"])
        output[method] = {
            "mean_delta_rouge_l_vs_native": float(np.mean(deltas_rouge)),
            "mean_delta_token_f1_vs_native": float(np.mean(deltas_f1)),
            "wins_rouge_l": int(sum(value > 1e-12 for value in deltas_rouge)),
            "losses_rouge_l": int(sum(value < -1e-12 for value in deltas_rouge)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=["rule_ce", "open_vqa", "report_generation"],
        required=True,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-bank", type=Path, default=DEFAULT_SOURCE_BANK)
    parser.add_argument("--shift-bank", type=Path, required=True)
    parser.add_argument("--mismatch-bank", type=Path, required=True)
    parser.add_argument("--shift-name", required=True)
    parser.add_argument("--mismatch-name", required=True)
    parser.add_argument("--strengths", type=parse_strengths, default=(0.25, 0.5))
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    rows = select_rows(
        load_rows(args.input), args.max_samples, args.seed, args.sample_offset
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_bank = load_bank(args.source_bank)
    shift_bank = load_bank(args.shift_bank)
    mismatch_bank = load_bank(args.mismatch_bank)
    config = {
        "version": "huatuo-ssrt-v1",
        "created_at": now_iso(),
        "command": " ".join(sys.argv),
        "pid": os.getpid(),
        "dataset": args.dataset,
        "task": args.task,
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "image_root": str(args.image_root),
        "n": len(rows),
        "selection": "stable_sha256",
        "seed": args.seed,
        "sample_offset": args.sample_offset,
        "source_bank": str(args.source_bank),
        "source_bank_sha256": file_sha256(args.source_bank),
        "shift_bank": str(args.shift_bank),
        "shift_bank_sha256": file_sha256(args.shift_bank),
        "shift_name": args.shift_name,
        "mismatch_bank": str(args.mismatch_bank),
        "mismatch_bank_sha256": file_sha256(args.mismatch_bank),
        "mismatch_name": args.mismatch_name,
        "strengths": args.strengths,
        "model": str(args.model_dir),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": 1,
            "repetition_penalty": 1.2,
        },
        "prompt_policy": "raw dataset question; no task-specific prompt change",
        "audit_only": args.audit_only,
    }
    write_json(args.output_dir / "config.json", config)
    raw_path = args.output_dir / (
        "image_audit.jsonl" if args.audit_only else "raw_generations.jsonl"
    )
    bot = None
    if not args.audit_only:
        HuatuoChatbot = import_huatuo(args.huatuo_root)
        bot = HuatuoChatbot(str(args.model_dir), device=args.device)
        bot.gen_kwargs.update(config["generation"])
        bot.gen_kwargs.pop("temperature", None)

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        identifier = row_id(row, index)
        path = args.image_root / image_name(row)
        prompt = str(row.get("question") or "")
        reference = str(row.get("answer") or row.get("report") or "")
        with Image.open(path) as source:
            visible = model_visible_image(source.convert("RGB"))
        conditions = build_conditions(
            visible, source_bank, shift_bank, mismatch_bank, args.strengths
        )
        if index < 4:
            image_dir = args.output_dir / "images" / str(identifier)
            image_dir.mkdir(parents=True, exist_ok=True)
            for method, (image, _) in conditions.items():
                image.save(image_dir / f"{method}.png")

        for method, (image, transform) in conditions.items():
            record = {
                "item_id": str(identifier),
                "qid": str(identifier),
                "question_id": str(identifier),
                "dataset": args.dataset,
                "task": args.task,
                "image": str(path),
                "question": prompt,
                "prompt_sha256": stable_sha256(prompt),
                "ground_truth": reference,
                "method": method,
                "transform": transform,
                "text": "",
                "error": None,
            }
            started = time.time()
            if not args.audit_only:
                try:
                    torch.manual_seed(args.seed + index)
                    torch.cuda.manual_seed_all(args.seed + index)
                    response = bot.inference(prompt, [image])
                    record["text"] = str(response[0] if response else "").strip()
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    record["error"] = f"CUDA OOM: {exc}"
                    record["traceback"] = traceback.format_exc()
                except Exception as exc:
                    record["error"] = repr(exc)
                    record["traceback"] = traceback.format_exc()
            record["elapsed_sec"] = time.time() - started
            record["completed_at"] = now_iso()
            append_jsonl(raw_path, record)
            by_method[method].append(record)
        print(
            json.dumps(
                {
                    "progress": f"{index + 1}/{len(rows)}",
                    "item_id": identifier,
                    "conditions": len(conditions),
                }
            ),
            flush=True,
        )

    if args.audit_only:
        summary = {
            "dataset": args.dataset,
            "task": args.task,
            "n": len(rows),
            "audit_only": True,
            "methods": {
                method: {
                    "identity_rate": float(
                        np.mean(
                            [
                                bool(row["transform"].get("identity"))
                                for row in records
                            ]
                        )
                    ),
                    "mean_psnr": float(
                        np.mean(
                            [
                                row["transform"]["structure_from_native"][
                                    "psnr"
                                ]
                                for row in records
                                if row["transform"]["structure_from_native"][
                                    "psnr"
                                ]
                                is not None
                            ]
                        )
                    )
                    if any(
                        row["transform"]["structure_from_native"]["psnr"]
                        is not None
                        for row in records
                    )
                    else None,
                    "mean_edge_correlation": float(
                        np.mean(
                            [
                                row["transform"]["structure_from_native"][
                                    "edge_correlation"
                                ]
                                for row in records
                            ]
                        )
                    ),
                    "mean_source_exceedance": float(
                        np.mean(
                            [
                                row["transform"]["distance"]["source"][
                                    "mean_normalized_exceedance"
                                ]
                                for row in records
                            ]
                        )
                    ),
                }
                for method, records in by_method.items()
            },
        }
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "task": args.task,
        "n": len(rows),
        "methods": {},
    }
    if args.task == "rule_ce":
        evaluations: dict[str, list[dict[str, Any]]] = {}
        for method, records in by_method.items():
            answers = [
                {"question_id": row["question_id"], "answer": row["text"]}
                for row in records
            ]
            metrics, evaluation = evaluate_rule_rows(rows, answers)
            evaluations[method] = evaluation
            summary["methods"][method] = {
                "rule_normalized_accuracy": metrics["rule_normalized"][
                    "accuracy"
                ],
                "decision_first_accuracy": metrics["decision_first"][
                    "accuracy_invalid_as_error"
                ],
                "decision_first_parse_rate": metrics["decision_first"][
                    "parse_rate"
                ],
                "errors": sum(row["error"] is not None for row in records),
            }
            method_dir = args.output_dir / method
            method_dir.mkdir()
            write_json(method_dir / "metrics.json", metrics)
            (method_dir / "records.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in evaluation)
            )
        summary["paired"] = ce_pairing(evaluations)
    else:
        scored = {}
        for method, records in by_method.items():
            scored[method] = aggregate_oe(records)
            summary["methods"][method] = {
                key: value
                for key, value in scored[method].items()
                if key != "records"
            }
            method_dir = args.output_dir / method
            method_dir.mkdir()
            write_json(method_dir / "metrics.json", summary["methods"][method])
        summary["paired"] = oe_pairing(scored)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
