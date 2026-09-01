#!/usr/bin/env python3
"""Phased HuatuoGPT-Vision RULE/MIMIC evaluation with ANCHOR FedDG views."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from corrected_sgta.evaluate_medheval_answers import parse_answer, rule_pope_prediction
from corrected_sgta.evaluate_rule_vqa import evaluate_rule_rows, load_jsonl, write_jsonl
from corrected_sgta.methods import feddg_frequency_interpolation


DEFAULT_QUESTION_FILE = Path("/home/dbw/ANCHOR/data/rule/test/mimic_test.jsonl")
DEFAULT_IMAGE_ROOT = Path("/home/dbw/ANCHOR/data/medheval/images")
DEFAULT_MODEL_DIR = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO_ROOT = Path("/home/dbw/HuatuoGPT-Vision")
DEFAULT_BANK = Path("/home/dbw/data/modality_centers/PubMedVision/train/ct__chest.npy")
DEFAULT_OUTPUT_ROOT = Path("/home/dbw/ANCHOR/corrected_runs/huatuo_rule_mimic_feddg_bfs")
DEFAULT_STAGE_SIZES = [1, 8, 16, 32]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def import_huatuo(root: Path):
    sys.path.insert(0, str(root))
    from cli import HuatuoChatbot  # type: ignore

    return HuatuoChatbot


def build_prompt(question: str) -> str:
    clean = question.replace("<image>", "").strip()
    return (
        f"{clean}\n"
        "Answer based only on the chest X-ray image. "
        "Use a complete sentence and make the yes/no decision explicit."
    )


def structure_metrics(source: Image.Image, transformed: Image.Image) -> dict[str, float | None]:
    left = np.asarray(source.convert("L"), dtype=np.float64) / 255.0
    right = np.asarray(transformed.convert("L"), dtype=np.float64) / 255.0
    mse = float(np.mean((left - right) ** 2))
    source_edge = np.hypot(*np.gradient(left))
    target_edge = np.hypot(*np.gradient(right))
    source_centered = source_edge.ravel() - source_edge.mean()
    target_centered = target_edge.ravel() - target_edge.mean()
    denominator = float(np.linalg.norm(source_centered) * np.linalg.norm(target_centered))
    edge_correlation = (
        float(np.clip(source_centered @ target_centered / denominator, -1.0, 1.0))
        if denominator > 1e-12
        else 1.0
    )
    return {
        "pixel_mse": mse,
        "psnr": None if mse <= 1e-12 else float(-10.0 * math.log10(mse)),
        "edge_correlation": edge_correlation,
    }


def style_views(
    image: Image.Image,
    bank: np.ndarray,
    low_frequency_ratios: list[float],
    source_ratios: list[float],
) -> list[tuple[str, Image.Image, dict[str, Any]]]:
    views: list[tuple[str, Image.Image, dict[str, Any]]] = [
        (
            "original",
            image,
            {
                "family": "original",
                "parameters": {},
                "structure": {"pixel_mse": 0.0, "psnr": None, "edge_correlation": 1.0},
            },
        )
    ]
    for low_frequency_ratio in low_frequency_ratios:
        for source_ratio in source_ratios:
            name = f"feddg_l{low_frequency_ratio:g}_sr{source_ratio:g}"
            transformed = feddg_frequency_interpolation(
                image,
                bank,
                low_frequency_ratio=low_frequency_ratio,
                source_ratio=source_ratio,
            )
            views.append(
                (
                    name,
                    transformed,
                    {
                        "family": "feddg",
                        "parameters": {
                            "low_frequency_ratio": low_frequency_ratio,
                            "source_ratio": source_ratio,
                        },
                        "structure": structure_metrics(image, transformed),
                    },
                )
            )
    return views


def style_is_safe(metadata: dict[str, Any], min_psnr: float, min_edge_correlation: float) -> bool:
    if metadata.get("family") == "original":
        return True
    structure = metadata.get("structure") or {}
    psnr = structure.get("psnr")
    edge = structure.get("edge_correlation")
    return (
        psnr is not None
        and float(psnr) >= min_psnr
        and edge is not None
        and float(edge) >= min_edge_correlation
    )


def prepare_inputs(bot: Any, text: str, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    text = bot.input_moderation(text)
    text = bot.insert_image_placeholder(text, 1)
    conv = bot.get_conv_without_history(text)
    input_ids = bot.preprocess(conv, return_tensors="pt").unsqueeze(0).to(bot.device)
    image_tensors = torch.stack(bot.get_image_tensors([image])).to(dtype=torch.bfloat16).to(bot.device)
    return input_ids, image_tensors


def generate_with_nll(
    bot: Any,
    prompt: str,
    image: Image.Image,
    *,
    max_new_tokens: int,
    repetition_penalty: float,
) -> dict[str, Any]:
    input_ids, image_tensors = prepare_inputs(bot, prompt, image)
    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 1,
        "repetition_penalty": repetition_penalty,
        "eos_token_id": bot.tokenizer.eos_token_id,
        "pad_token_id": bot.tokenizer.pad_token_id or bot.tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
        "use_cache": True,
    }
    started = time.time()
    with torch.inference_mode():
        outputs = bot.model.generate(input_ids, images=image_tensors, **generation_kwargs)
    elapsed = time.time() - started
    sequence = outputs.sequences[0]
    prompt_len = int(input_ids.shape[1])
    generated_ids = sequence
    sequence_layout = "huatuo_generation_only"
    text = bot.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    scores = bot.model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        normalize_logits=True,
    )[0]
    valid_scores = scores[: len(generated_ids)].detach().float().cpu().numpy()
    finite = valid_scores[np.isfinite(valid_scores)]
    mean_nll = float(-finite.mean()) if len(finite) else None
    total_nll = float(-finite.sum()) if len(finite) else None
    return {
        "text": text,
        "mean_token_nll": mean_nll,
        "total_token_nll": total_nll,
        "generated_token_count": int(len(generated_ids)),
        "sequence_token_count": int(sequence.shape[0]),
        "prompt_token_count": prompt_len,
        "sequence_layout": sequence_layout,
        "elapsed_sec": elapsed,
    }


def candidate_decision_label(candidate: dict[str, Any]) -> str | None:
    strict = candidate.get("strict_prediction")
    if strict in {"yes", "no"}:
        return str(strict)
    pope = candidate.get("pope_prediction")
    if pope in {"yes", "no"}:
        return str(pope)
    parsed = parse_answer(candidate.get("text") or "", answer_type="binary")
    if parsed.labels:
        return parsed.labels[0]
    return rule_pope_prediction(candidate.get("text") or "")


def candidate_is_rule_parseable(candidate: dict[str, Any]) -> bool:
    parsed = parse_answer(candidate.get("text") or "", answer_type="binary")
    return bool(parsed.labels) or candidate_decision_label(candidate) in {"yes", "no"}


def min_nll(items: list[dict[str, Any]]) -> float:
    return min(float(item["mean_token_nll"]) for item in items)


def pick_candidate(
    candidates: list[dict[str, Any]], *, prefer_rule_parseable: bool, switch_margin: float = 0.05
) -> tuple[dict[str, Any] | None, str | None]:
    usable = [
        item
        for item in candidates
        if item.get("error") is None and item.get("mean_token_nll") is not None
    ]
    if not usable:
        baseline = next((item for item in candidates if item.get("style") == "original"), None)
        return baseline, "baseline_fallback_no_nll"
    if prefer_rule_parseable:
        parseable = [item for item in usable if candidate_is_rule_parseable(item)]
        if parseable:
            original = next((item for item in parseable if item.get("style") == "original"), None)
            original_label = candidate_decision_label(original) if original is not None else None
            if original is not None and original_label is not None:
                same = [
                    item
                    for item in parseable
                    if candidate_decision_label(item) == original_label
                ]
                opposite = [
                    item
                    for item in parseable
                    if item.get("style") != "original"
                    and candidate_decision_label(item) is not None
                    and candidate_decision_label(item) != original_label
                ]
                if same and len(opposite) >= 2 and min_nll(opposite) + switch_margin < min_nll(same):
                    usable = opposite
                elif same:
                    usable = same
            else:
                label_counts = Counter(candidate_decision_label(item) for item in parseable)
                label, _ = max(
                    label_counts.items(),
                    key=lambda item: (item[1], -min(
                        float(cand["mean_token_nll"])
                        for cand in parseable
                        if candidate_decision_label(cand) == item[0]
                    )),
                )
                usable = [item for item in parseable if candidate_decision_label(item) == label]
        else:
            baseline = next((item for item in usable if item.get("style") == "original"), None)
            if baseline is not None:
                return baseline, "conservative_original_no_rule_parseable_candidate"
    return min(
        usable,
        key=lambda item: (
            float(item["mean_token_nll"]),
            0 if item.get("style") == "original" else 1,
            item.get("style", ""),
        ),
    ), None


def evaluate_answers(
    questions: list[dict[str, Any]],
    answer_rows: list[dict[str, Any]],
    out_prefix: Path,
) -> dict[str, Any]:
    metrics, records = evaluate_rule_rows(questions, answer_rows)
    write_json(out_prefix.with_suffix(".metrics.json"), metrics)
    write_jsonl(out_prefix.with_suffix(".records.jsonl"), records)
    return metrics


def explicit_correct_map(records: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        str(row["question_id"]): bool(row["explicit_ground_truth_correct"])
        for row in records
        if row.get("explicit_ground_truth_correct") is not None
    }


def stage_summary(
    questions: list[dict[str, Any]],
    baseline_answers: list[dict[str, Any]],
    anchor_answers: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    anchor_metrics: dict[str, Any],
) -> dict[str, Any]:
    _, baseline_records = evaluate_rule_rows(questions, baseline_answers)
    _, anchor_records = evaluate_rule_rows(questions, anchor_answers)
    baseline_correct = explicit_correct_map(baseline_records)
    anchor_correct = explicit_correct_map(anchor_records)
    qids = sorted(set(baseline_correct) & set(anchor_correct), key=int)
    wrong_to_correct = sum((not baseline_correct[qid]) and anchor_correct[qid] for qid in qids)
    correct_to_wrong = sum(baseline_correct[qid] and (not anchor_correct[qid]) for qid in qids)
    selected_styles = Counter(row.get("selected_style", "unknown") for row in raw_rows)
    conservative_count = sum(
        1
        for row in raw_rows
        if row.get("fallback_reason")
        in {
            "conservative_original_no_rule_parseable_candidate",
            "baseline_fallback_no_rule_parseable_candidate",
        }
    )
    fallback_count = sum(
        1
        for row in raw_rows
        if row.get("fallback_reason")
        and row.get("fallback_reason")
        not in {
            "conservative_original_no_rule_parseable_candidate",
            "baseline_fallback_no_rule_parseable_candidate",
        }
    )
    error_count = sum(int(row.get("error_count", 0)) for row in raw_rows)
    candidate_counts = [int(row.get("candidate_count", 0)) for row in raw_rows]
    selected_nll = [
        float(row["selected_mean_token_nll"])
        for row in raw_rows
        if row.get("selected_mean_token_nll") is not None
    ]
    return {
        "n": len(questions),
        "explicit_n": baseline_metrics["explicit_ground_truth"]["n"],
        "baseline_accuracy": baseline_metrics["explicit_ground_truth"]["accuracy"],
        "anchor_accuracy": anchor_metrics["explicit_ground_truth"]["accuracy"],
        "baseline_strict_parse_rate": baseline_metrics["strict_explicit"]["parse_rate"],
        "anchor_strict_parse_rate": anchor_metrics["strict_explicit"]["parse_rate"],
        "baseline_pope_accuracy": baseline_metrics["pope_compatible"]["accuracy"],
        "anchor_pope_accuracy": anchor_metrics["pope_compatible"]["accuracy"],
        "wrong_to_correct_flips": wrong_to_correct,
        "correct_to_wrong_flips": correct_to_wrong,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(raw_rows) if raw_rows else 0.0,
        "conservative_original_count": conservative_count,
        "conservative_original_rate": conservative_count / len(raw_rows) if raw_rows else 0.0,
        "error_count": error_count,
        "error_rate": error_count / max(sum(candidate_counts), 1),
        "candidate_count_mean": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "candidate_count_min": int(min(candidate_counts)) if candidate_counts else 0,
        "candidate_count_max": int(max(candidate_counts)) if candidate_counts else 0,
        "selected_style_histogram": dict(sorted(selected_styles.items())),
        "selected_mean_token_nll_mean": float(np.mean(selected_nll)) if selected_nll else None,
        "selected_mean_token_nll_median": float(np.median(selected_nll)) if selected_nll else None,
        "baseline_invalid_as_error_accuracy": baseline_metrics["strict_explicit"][
            "accuracy_invalid_as_error"
        ],
        "anchor_invalid_as_error_accuracy": anchor_metrics["strict_explicit"][
            "accuracy_invalid_as_error"
        ],
    }


def gate_decision(stage_size: int, summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary["error_count"] > 0 and stage_size <= 1:
        reasons.append("stage0_has_errors")
    if summary["error_rate"] > 0.05:
        reasons.append(f"error_rate>{0.05:g}")
    if summary["fallback_rate"] > 0.20:
        reasons.append(f"fallback_rate>{0.20:g}")
    if summary["anchor_strict_parse_rate"] + 0.05 < summary["baseline_strict_parse_rate"]:
        reasons.append("anchor_parse_rate_drop_gt_5pp")
    if stage_size >= 32 and summary["anchor_accuracy"] + 0.10 < summary["baseline_accuracy"]:
        reasons.append("anchor_accuracy_drop_gt_10pp")
    return not reasons, reasons


def run_stage(args: argparse.Namespace, bot: Any, bank: np.ndarray, rows: list[dict[str, Any]], stage_dir: Path) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    questions_path = stage_dir / "questions.jsonl"
    raw_path = stage_dir / "raw_generations.jsonl"
    baseline_answers_path = stage_dir / "answers.baseline_greedy.jsonl"
    anchor_answers_path = stage_dir / "answers.anchor_feddg_style.jsonl"
    write_jsonl(questions_path, rows)
    if raw_path.exists():
        raw_path.unlink()

    baseline_answers: list[dict[str, Any]] = []
    anchor_answers: list[dict[str, Any]] = []
    start = now_iso()
    for index, row in enumerate(rows, start=1):
        qid = row.get("question_id")
        prompt = build_prompt(str(row.get("question", "")))
        image_path = args.image_root / str(row["image"])
        raw_record: dict[str, Any] = {
            "question_id": qid,
            "image": str(row.get("image")),
            "prompt": prompt,
            "ground_truth": row.get("answer"),
            "candidates": [],
            "error_count": 0,
        }
        try:
            image = Image.open(image_path).convert("RGB")
            views = style_views(image, bank, args.low_frequency_ratio, args.source_ratio)
        except Exception as exc:  # noqa: BLE001
            raw_record["load_or_style_error"] = repr(exc)
            raw_record["traceback"] = traceback.format_exc()
            raw_record["error_count"] += 1
            views = []

        for style_name, view_image, metadata in views:
            safe = style_is_safe(metadata, args.min_style_psnr, args.min_edge_correlation)
            if not safe and not args.keep_unsafe_styles:
                raw_record["candidates"].append(
                    {
                        "style": style_name,
                        "metadata": metadata,
                        "skipped": True,
                        "skip_reason": "structure_gate",
                    }
                )
                continue
            candidate = {"style": style_name, "metadata": metadata, "skipped": False, "error": None}
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
                raw_record["error_count"] += 1
            except Exception as exc:  # noqa: BLE001
                candidate["error"] = repr(exc)
                candidate["traceback"] = traceback.format_exc()
                raw_record["error_count"] += 1
            raw_record["candidates"].append(candidate)

        baseline = next(
            (item for item in raw_record["candidates"] if item.get("style") == "original"),
            None,
        )
        for candidate in raw_record["candidates"]:
            if not candidate.get("skipped") and candidate.get("error") is None:
                parsed = parse_answer(candidate.get("text") or "", answer_type="binary")
                candidate["strict_prediction"] = parsed.labels[0] if parsed.labels else None
                candidate["strict_status"] = parsed.status
                candidate["pope_prediction"] = rule_pope_prediction(candidate.get("text") or "")
                candidate["selection_decision_label"] = candidate_decision_label(candidate)
        selected, fallback_reason = pick_candidate(
            raw_record["candidates"],
            prefer_rule_parseable=args.prefer_rule_parseable,
            switch_margin=args.switch_margin,
        )
        if selected is None:
            selected = {"style": "missing", "text": "", "mean_token_nll": None}
            fallback_reason = fallback_reason or "no_candidate"
        baseline_text = "" if baseline is None else str(baseline.get("text") or "")
        anchor_text = str(selected.get("text") or baseline_text)
        if not anchor_text and baseline_text:
            anchor_text = baseline_text
            fallback_reason = fallback_reason or "empty_selected_text"

        raw_record.update(
            {
                "baseline_text": baseline_text,
                "baseline_prediction_pope": rule_pope_prediction(baseline_text),
                "selected_text": anchor_text,
                "selected_style": selected.get("style"),
                "selected_mean_token_nll": selected.get("mean_token_nll"),
                "selected_prediction_pope": rule_pope_prediction(anchor_text),
                "fallback_reason": fallback_reason,
                "candidate_count": sum(
                    1
                    for item in raw_record["candidates"]
                    if not item.get("skipped") and item.get("error") is None
                ),
                "completed_at": now_iso(),
            }
        )
        append_jsonl(raw_path, raw_record)
        baseline_answers.append({"question_id": qid, "answer": baseline_text})
        anchor_answers.append({"question_id": qid, "answer": anchor_text})
        if index == 1 or index % args.progress_every == 0 or index == len(rows):
            print(
                json.dumps(
                    {
                        "stage_dir": str(stage_dir),
                        "progress": f"{index}/{len(rows)}",
                        "qid": qid,
                        "baseline_pred": raw_record["baseline_prediction_pope"],
                        "selected_pred": raw_record["selected_prediction_pope"],
                        "selected_style": raw_record["selected_style"],
                        "candidate_count": raw_record["candidate_count"],
                        "error_count": raw_record["error_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    write_jsonl(baseline_answers_path, baseline_answers)
    write_jsonl(anchor_answers_path, anchor_answers)
    baseline_metrics = evaluate_answers(
        rows, baseline_answers, stage_dir / "baseline_greedy"
    )
    anchor_metrics = evaluate_answers(
        rows, anchor_answers, stage_dir / "anchor_feddg_style"
    )
    raw_rows = load_jsonl(raw_path)
    summary = stage_summary(
        rows,
        baseline_answers,
        anchor_answers,
        raw_rows,
        baseline_metrics,
        anchor_metrics,
    )
    passed, reasons = gate_decision(len(rows), summary)
    summary.update(
        {
            "started_at": start,
            "ended_at": now_iso(),
            "gate_passed": passed,
            "gate_reasons": reasons,
        }
    )
    write_json(stage_dir / "summary.json", summary)
    return summary


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_stage_sizes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTION_FILE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO_ROOT)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage-size", type=int, help="Run one prefix stage; 0 means full.")
    parser.add_argument("--phased", action="store_true", help="Run the short BFS stages 1,8,16,32 with gates.")
    parser.add_argument(
        "--depth-stage",
        action="store_true",
        help="Allow explicit 128/512/full stages after BFS has selected a promising method.",
    )
    parser.add_argument("--stage-sizes", type=parse_stage_sizes, default=DEFAULT_STAGE_SIZES)
    parser.add_argument("--low-frequency-ratio", type=parse_float_list, default=parse_float_list("0.003,0.01,0.03"))
    parser.add_argument("--source-ratio", type=parse_float_list, default=parse_float_list("0.8"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--min-style-psnr", type=float, default=18.0)
    parser.add_argument("--min-edge-correlation", type=float, default=0.90)
    parser.add_argument("--keep-unsafe-styles", action="store_true")
    parser.add_argument(
        "--no-prefer-rule-parseable",
        dest="prefer_rule_parseable",
        action="store_false",
        help="Ablation: select by NLL even when a candidate is not binary-parseable.",
    )
    parser.set_defaults(prefer_rule_parseable=True)
    parser.add_argument("--switch-margin", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.phased and args.stage_size is None:
        raise SystemExit("Specify --stage-size N or --phased.")
    if args.stage_size in {128, 512, 0} and not args.depth_stage:
        raise SystemExit("128/512/full are depth stages; pass --depth-stage explicitly.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.jsonl"
    questions = load_jsonl(args.questions)
    bank = np.load(args.bank)
    config = {
        "created_at": now_iso(),
        "command": " ".join(sys.argv),
        "questions": str(args.questions),
        "question_count": len(questions),
        "image_root": str(args.image_root),
        "model_dir": str(args.model_dir),
        "huatuo_root": str(args.huatuo_root),
        "bank": str(args.bank),
        "bank_sha256": sha256_file(args.bank),
        "low_frequency_ratio": args.low_frequency_ratio,
        "source_ratio": args.source_ratio,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "min_style_psnr": args.min_style_psnr,
        "min_edge_correlation": args.min_edge_correlation,
        "keep_unsafe_styles": args.keep_unsafe_styles,
        "prefer_rule_parseable": args.prefer_rule_parseable,
        "switch_margin": args.switch_margin,
        "device": args.device,
        "prompt_template": build_prompt("{question}"),
        "selection_rule": (
            "RULE CE anchored consistency over parseable candidates, then reference-free "
            "min mean generated-token NLL to keep a complete natural-language output"
        ),
        "pid": os.getpid(),
    }
    write_json(args.output_root / "config.json", config)
    append_jsonl(
        manifest_path,
        {
            "event": "run_start",
            "time": now_iso(),
            "config": config,
        },
    )
    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device=args.device)
    stage_sizes = args.stage_sizes if args.phased else [args.stage_size]
    for size in stage_sizes:
        assert size is not None
        selected_rows = questions if size == 0 else questions[:size]
        stage_name = "stage5_full" if size == 0 else f"stage_n{size}"
        stage_dir = args.output_root / stage_name
        append_jsonl(
            manifest_path,
            {
                "event": "stage_start",
                "time": now_iso(),
                "stage": stage_name,
                "n": len(selected_rows),
                "output_dir": str(stage_dir),
            },
        )
        summary = run_stage(args, bot, bank, selected_rows, stage_dir)
        append_jsonl(
            manifest_path,
            {
                "event": "stage_end",
                "time": now_iso(),
                "stage": stage_name,
                "summary": summary,
            },
        )
        print(json.dumps({"stage": stage_name, "summary": summary}, indent=2, ensure_ascii=False), flush=True)
        if not summary["gate_passed"]:
            append_jsonl(
                manifest_path,
                {
                    "event": "gate_blocked",
                    "time": now_iso(),
                    "stage": stage_name,
                    "reasons": summary["gate_reasons"],
                },
            )
            raise SystemExit(f"Gate blocked at {stage_name}: {summary['gate_reasons']}")
    append_jsonl(manifest_path, {"event": "run_done", "time": now_iso()})


if __name__ == "__main__":
    main()
