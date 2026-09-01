#!/usr/bin/env python3
"""Breadth-first Huatuo + ANCHOR FedDG selector screening on RULE/MIMIC."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from corrected_sgta.evaluate_medheval_answers import parse_answer, rule_pope_prediction
from corrected_sgta.evaluate_rule_vqa import evaluate_rule_rows, load_jsonl, write_jsonl
from corrected_sgta.frequency_alignment_source_spectrum import source_spectrum_alignment
from corrected_sgta.methods import feddg_frequency_interpolation, gamma_transform
from corrected_sgta.run_huatuo_rule_feddg import (
    DEFAULT_BANK,
    DEFAULT_HUATUO_ROOT,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MODEL_DIR,
    DEFAULT_QUESTION_FILE,
    append_jsonl,
    build_prompt,
    candidate_decision_label,
    generate_with_nll,
    import_huatuo,
    now_iso,
    sha256_file,
    stage_summary,
    structure_metrics,
    style_is_safe,
    write_json,
)


RUN_VERSION = "huatuo-rule-feddg-bfs-v1"
DEFAULT_OUTPUT_ROOT = Path("/home/dbw/ANCHOR/corrected_runs/huatuo_rule_mimic_feddg_bfs")
METHODS = [
    "baseline_greedy",
    "feddg_min_nll",
    "feddg_strict_then_nll",
    "feddg_anchor_consistency",
    "feddg_switch_gate",
    "feddg_oe_risk_order",
    "flow_sgta_views",
    "mfcd_decoding",
]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_stage_sizes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def valid_candidates(candidates: list[dict[str, Any]], family: str | None = None) -> list[dict[str, Any]]:
    def family_of(item: dict[str, Any]) -> str | None:
        metadata = item.get("metadata") or {}
        value = item.get("view_family") or metadata.get("view_family") or metadata.get("family")
        if family == "feddg" and value == "feddg":
            return "feddg"
        if family == "flow" and str(value or "").startswith(("sgta_", "gamma", "flow")):
            return "flow"
        return str(value) if value is not None else None

    items = [
        item
        for item in candidates
        if not item.get("skipped")
        and item.get("error") is None
        and item.get("mean_token_nll") is not None
        and (family is None or family_of(item) == family)
    ]
    return items


def min_nll(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    return min(
        items,
        key=lambda item: (
            float(item["mean_token_nll"]),
            0 if item.get("style") == "original" else 1,
            str(item.get("style", "")),
        ),
    )


def candidate_label(candidate: dict[str, Any] | None) -> str | None:
    if candidate is None:
        return None
    return candidate_decision_label(candidate)


def is_parseable(candidate: dict[str, Any]) -> bool:
    return candidate_label(candidate) in {"yes", "no"}


def baseline_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((item for item in candidates if item.get("style") == "original"), None)


def select_method(method: str, candidates: list[dict[str, Any]], switch_margin: float) -> tuple[dict[str, Any] | None, str | None]:
    baseline = baseline_candidate(candidates)
    all_valid = valid_candidates(candidates)
    feddg = valid_candidates(candidates, "feddg")
    flow = valid_candidates(candidates, "flow")
    if method == "baseline_greedy":
        return baseline, None
    if method == "mfcd_decoding":
        return None, "blocked_runtime_huatuo_decoder_level_mfcd_not_integrated"
    if method == "flow_sgta_views":
        return min_nll(flow) or baseline, "baseline_fallback_no_flow_candidate" if not flow else None
    if method == "feddg_min_nll":
        return min_nll(feddg) or baseline, "baseline_fallback_no_feddg_candidate" if not feddg else None
    if method == "feddg_oe_risk_order":
        candidates_with_text = [
            item
            for item in all_valid
            if int(item.get("generated_token_count") or 0) >= 5 and str(item.get("text") or "").strip()
        ]
        return min_nll(candidates_with_text) or baseline, "baseline_fallback_no_risk_order_candidate" if not candidates_with_text else None
    parseable = [item for item in all_valid if is_parseable(item)]
    if method == "feddg_strict_then_nll":
        return min_nll(parseable) or baseline, "baseline_fallback_no_parseable_candidate" if not parseable else None
    if method in {"feddg_anchor_consistency", "feddg_switch_gate"}:
        original_label = candidate_label(baseline)
        if original_label is None:
            return min_nll(parseable) or baseline, "baseline_fallback_no_original_label"
        same = [item for item in parseable if candidate_label(item) == original_label]
        opposite = [
            item
            for item in parseable
            if item.get("style") != "original" and candidate_label(item) != original_label
        ]
        if method == "feddg_switch_gate" and same and len(opposite) >= 2:
            opposite_best = min_nll(opposite)
            same_best = min_nll(same)
            if (
                opposite_best is not None
                and same_best is not None
                and float(opposite_best["mean_token_nll"]) + switch_margin < float(same_best["mean_token_nll"])
            ):
                return opposite_best, None
        return min_nll(same) or baseline, "baseline_fallback_no_anchor_consistent_candidate" if not same else None
    raise ValueError(f"unknown method: {method}")


def make_candidate_views(
    image: Image.Image,
    center: np.ndarray,
    low_frequency_ratios: list[float],
    source_ratios: list[float],
    *,
    flow_spectrum_alpha: float,
    flow_low_frequency_ratio: float,
    flow_gamma: float,
) -> list[tuple[str, Image.Image, dict[str, Any]]]:
    views: list[tuple[str, Image.Image, dict[str, Any]]] = [
        (
            "original",
            image,
            {
                "view_family": "original",
                "family": "original",
                "parameters": {},
                "structure": {"pixel_mse": 0.0, "psnr": None, "edge_correlation": 1.0},
            },
        )
    ]
    for ratio in low_frequency_ratios:
        for source_ratio in source_ratios:
            transformed = feddg_frequency_interpolation(
                image,
                center,
                low_frequency_ratio=ratio,
                source_ratio=source_ratio,
            )
            views.append(
                (
                    f"feddg_l{ratio:g}_sr{source_ratio:g}",
                    transformed,
                    {
                        "view_family": "feddg",
                        "family": "feddg",
                        "parameters": {"low_frequency_ratio": ratio, "source_ratio": source_ratio},
                        "structure": structure_metrics(image, transformed),
                    },
                )
            )
    flow_specs = [
        (
            "flow_source_spectrum",
            source_spectrum_alignment(image, center, low_frequency_ratio=flow_spectrum_alpha, source_ratio=0.0),
            {"view_family": "flow", "family": "sgta_source_spectrum", "parameters": {"spectral_alpha": flow_spectrum_alpha}},
        ),
        (
            "flow_low_frequency",
            feddg_frequency_interpolation(
                image,
                center,
                low_frequency_ratio=flow_low_frequency_ratio,
                source_ratio=0.8,
            ),
            {"view_family": "flow", "family": "sgta_low_frequency", "parameters": {"low_frequency_ratio": flow_low_frequency_ratio, "source_ratio": 0.8}},
        ),
        (
            "flow_gamma",
            gamma_transform(image, flow_gamma),
            {"view_family": "flow", "family": "gamma", "parameters": {"gamma": flow_gamma}},
        ),
    ]
    for name, transformed, metadata in flow_specs:
        views.append((name, transformed, {**metadata, "structure": structure_metrics(image, transformed)}))
    return views


def annotate_candidate(candidate: dict[str, Any]) -> None:
    if candidate.get("skipped") or candidate.get("error") is not None:
        return
    parsed = parse_answer(candidate.get("text") or "", answer_type="binary")
    candidate["strict_prediction"] = parsed.labels[0] if parsed.labels else None
    candidate["strict_status"] = parsed.status
    candidate["pope_prediction"] = rule_pope_prediction(candidate.get("text") or "")
    candidate["selection_decision_label"] = candidate_label(candidate)


def generate_pool(args: argparse.Namespace, bot: Any, center: np.ndarray, rows: list[dict[str, Any]], pool_dir: Path) -> list[dict[str, Any]]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    raw_path = pool_dir / "candidate_pool.jsonl"
    if raw_path.exists() and args.reuse_pool:
        return load_jsonl(raw_path)
    if raw_path.exists():
        raw_path.unlink()
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        qid = row.get("question_id")
        prompt = build_prompt(str(row.get("question", "")))
        record = {
            "version": RUN_VERSION,
            "question_id": qid,
            "image": row.get("image"),
            "ground_truth": row.get("answer"),
            "prompt": prompt,
            "candidates": [],
            "error_count": 0,
        }
        try:
            image = Image.open(args.image_root / str(row["image"])).convert("RGB")
            views = make_candidate_views(
                image,
                center,
                args.low_frequency_ratio,
                args.source_ratio,
                flow_spectrum_alpha=args.flow_spectrum_alpha,
                flow_low_frequency_ratio=args.flow_low_frequency_ratio,
                flow_gamma=args.flow_gamma,
            )
        except Exception as exc:  # noqa: BLE001
            record["load_or_view_error"] = repr(exc)
            record["traceback"] = traceback.format_exc()
            record["error_count"] += 1
            views = []
        for style, view_image, metadata in views:
            safe = style_is_safe(metadata, args.min_style_psnr, args.min_edge_correlation)
            if not safe and not args.keep_unsafe_styles:
                record["candidates"].append(
                    {"style": style, **metadata, "metadata": metadata, "skipped": True, "skip_reason": "structure_gate"}
                )
                continue
            candidate = {"style": style, **metadata, "metadata": metadata, "skipped": False, "error": None}
            try:
                candidate.update(
                    generate_with_nll(
                        bot,
                        prompt,
                        view_image,
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=args.repetition_penalty,
                    )
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                candidate["error"] = f"CUDA OOM: {exc}"
                record["error_count"] += 1
            except Exception as exc:  # noqa: BLE001
                candidate["error"] = repr(exc)
                candidate["traceback"] = traceback.format_exc()
                record["error_count"] += 1
            annotate_candidate(candidate)
            record["candidates"].append(candidate)
        record["candidate_count"] = len(valid_candidates(record["candidates"]))
        record["completed_at"] = now_iso()
        records.append(record)
        append_jsonl(raw_path, record)
        if index == 1 or index % args.progress_every == 0 or index == len(rows):
            print(json.dumps({"pool": str(pool_dir), "progress": f"{index}/{len(rows)}", "qid": qid, "candidate_count": record["candidate_count"], "error_count": record["error_count"]}, ensure_ascii=False), flush=True)
    return records


def write_method_stage(
    method: str,
    questions: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    stage_dir: Path,
    switch_margin: float,
) -> dict[str, Any]:
    method_dir = stage_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    baseline_answers: list[dict[str, Any]] = []
    method_answers: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    blocked = method == "mfcd_decoding"
    for record in pool:
        selected, reason = select_method(method, record["candidates"], switch_margin)
        baseline = baseline_candidate(record["candidates"])
        baseline_text = "" if baseline is None else str(baseline.get("text") or "")
        selected_text = "" if selected is None else str(selected.get("text") or baseline_text)
        if blocked:
            selected_text = baseline_text
        raw = {
            "question_id": record["question_id"],
            "image": record.get("image"),
            "prompt": record.get("prompt"),
            "ground_truth": record.get("ground_truth"),
            "baseline_text": baseline_text,
            "selected_text": selected_text,
            "selected_style": None if selected is None else selected.get("style"),
            "selected_mean_token_nll": None if selected is None else selected.get("mean_token_nll"),
            "selected_prediction_pope": rule_pope_prediction(selected_text),
            "fallback_reason": reason,
            "candidate_count": record.get("candidate_count", 0),
            "error_count": record.get("error_count", 0),
            "candidates": record["candidates"],
        }
        raw_rows.append(raw)
        baseline_answers.append({"question_id": record["question_id"], "answer": baseline_text})
        method_answers.append({"question_id": record["question_id"], "answer": selected_text})
    write_jsonl(method_dir / "questions.jsonl", questions)
    write_jsonl(method_dir / "raw_generations.jsonl", raw_rows)
    write_jsonl(method_dir / "answers.baseline_greedy.jsonl", baseline_answers)
    write_jsonl(method_dir / f"answers.{method}.jsonl", method_answers)
    baseline_metrics, baseline_records = evaluate_rule_rows(questions, baseline_answers)
    method_metrics, method_records = evaluate_rule_rows(questions, method_answers)
    write_json(method_dir / "baseline_greedy.metrics.json", baseline_metrics)
    write_jsonl(method_dir / "baseline_greedy.records.jsonl", baseline_records)
    write_json(method_dir / f"{method}.metrics.json", method_metrics)
    write_jsonl(method_dir / f"{method}.records.jsonl", method_records)
    summary = stage_summary(questions, baseline_answers, method_answers, raw_rows, baseline_metrics, method_metrics)
    summary.update(
        {
            "method": method,
            "status": "blocked_runtime" if blocked else "done",
            "blocked_reason": "Huatuo decoder-level MFCD is not integrated in this BFS wrapper" if blocked else None,
            "ended_at": now_iso(),
        }
    )
    write_json(method_dir / "summary.json", summary)
    return summary


def bfs_decision(stage_size: int, summary: dict[str, Any]) -> str:
    if summary["status"] == "blocked_runtime":
        return "blocked_runtime"
    if summary["error_rate"] > 0.0 or summary["fallback_rate"] > 0.05:
        return "blocked_runtime"
    if summary["anchor_strict_parse_rate"] + 0.05 < summary["baseline_strict_parse_rate"]:
        return "stop_negative"
    delta = summary["anchor_accuracy"] - summary["baseline_accuracy"]
    rescue = int(summary["wrong_to_correct_flips"])
    harm = int(summary["correct_to_wrong_flips"])
    if stage_size == 16:
        return "advance_32" if delta >= 0.0 and harm <= rescue else "stop_negative"
    if stage_size == 32:
        if delta >= 0.01 or rescue > harm:
            return "advance_128"
        if delta >= 0.0 and harm == 0 and summary["anchor_strict_parse_rate"] >= summary["baseline_strict_parse_rate"]:
            return "stop_neutral"
        return "stop_negative"
    if stage_size == 128:
        if delta >= 0.01 or (delta >= 0.0 and rescue > harm and summary["anchor_strict_parse_rate"] >= summary["baseline_strict_parse_rate"]):
            return "advance_depth"
        return "stop_neutral" if delta >= 0.0 else "stop_negative"
    return "done"


def run_stage(args: argparse.Namespace, stage_size: int, methods: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    questions = load_jsonl(args.questions)
    rows = questions[:stage_size]
    center = np.load(args.bank)
    stage_dir = args.output_root / f"stage_n{stage_size}"
    write_jsonl(stage_dir / "questions.jsonl", rows)
    config = {
        "version": RUN_VERSION,
        "created_at": now_iso(),
        "command": " ".join(sys.argv),
        "stage_size": stage_size,
        "methods": methods,
        "questions": str(args.questions),
        "image_root": str(args.image_root),
        "model_dir": str(args.model_dir),
        "huatuo_root": str(args.huatuo_root),
        "bank": str(args.bank),
        "bank_sha256": sha256_file(args.bank),
        "low_frequency_ratio": args.low_frequency_ratio,
        "source_ratio": args.source_ratio,
        "min_style_psnr": args.min_style_psnr,
        "min_edge_correlation": args.min_edge_correlation,
        "flow_spectrum_alpha": args.flow_spectrum_alpha,
        "flow_low_frequency_ratio": args.flow_low_frequency_ratio,
        "flow_gamma": args.flow_gamma,
        "max_new_tokens": args.max_new_tokens,
        "single_gpu_single_process": True,
    }
    write_json(stage_dir / "config.json", config)
    if args.candidate_pool:
        pool = load_jsonl(args.candidate_pool)[:stage_size]
    else:
        HuatuoChatbot = import_huatuo(args.huatuo_root)
        bot = HuatuoChatbot(str(args.model_dir), device=args.device)
        pool = generate_pool(args, bot, center, rows, stage_dir / "candidate_pool")
    summaries = {method: write_method_stage(method, rows, pool, stage_dir, args.switch_margin) for method in methods}
    decisions = {method: bfs_decision(stage_size, summary) for method, summary in summaries.items()}
    payload = {"version": RUN_VERSION, "stage_size": stage_size, "summaries": summaries}
    write_json(stage_dir / "bfs_summary.json", payload)
    write_json(stage_dir / "bfs_decision.json", {"version": RUN_VERSION, "stage_size": stage_size, "decisions": decisions})
    return summaries, decisions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTION_FILE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO_ROOT)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage-size", type=int, default=16)
    parser.add_argument("--bfs", action="store_true", help="Run 16 then selected methods at 32.")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--candidate-pool", type=Path, help="Optional existing candidate_pool.jsonl or raw_generations.jsonl for no-GPU reselect.")
    parser.add_argument("--reuse-pool", action="store_true")
    parser.add_argument("--low-frequency-ratio", type=parse_float_list, default=parse_float_list("0.003,0.01,0.03"))
    parser.add_argument("--source-ratio", type=parse_float_list, default=parse_float_list("0.8"))
    parser.add_argument("--min-style-psnr", type=float, default=18.0)
    parser.add_argument("--min-edge-correlation", type=float, default=0.90)
    parser.add_argument("--keep-unsafe-styles", action="store_true")
    parser.add_argument("--flow-spectrum-alpha", type=float, default=0.01)
    parser.add_argument("--flow-low-frequency-ratio", type=float, default=0.01)
    parser.add_argument("--flow-gamma", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--switch-margin", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    methods = [method for method in args.methods if method in METHODS]
    manifest = args.output_root / "manifest.jsonl"
    append_jsonl(manifest, {"event": "run_start", "time": now_iso(), "command": " ".join(sys.argv), "methods": methods})
    summaries16, decisions16 = run_stage(args, args.stage_size, methods)
    append_jsonl(manifest, {"event": "stage_done", "time": now_iso(), "stage_size": args.stage_size, "decisions": decisions16})
    if args.bfs and args.stage_size == 16:
        advanced = [method for method, decision in decisions16.items() if decision == "advance_32"]
        if advanced:
            summaries32, decisions32 = run_stage(args, 32, advanced)
            append_jsonl(manifest, {"event": "stage_done", "time": now_iso(), "stage_size": 32, "decisions": decisions32})
        else:
            append_jsonl(manifest, {"event": "bfs_stop", "time": now_iso(), "reason": "no_method_advanced_from_16"})
    print(json.dumps({"stage_size": args.stage_size, "decisions": decisions16}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
