#!/usr/bin/env python3
"""Common-protocol confirmation of style-induced medical VLM decision flips."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from scipy.stats import binomtest

from corrected_sgta.methods import (
    feddg_frequency_interpolation,
    gamma_transform,
)
from corrected_sgta.run_huatuo_style_prior_probe import (
    build_prompt,
    explicit_label,
    import_huatuo,
    load_jsonl,
    load_style_bank,
    score_binary,
    sha256_file,
    structure_metrics,
    write_json,
)


VERSION = "huatuo-style-phenomenon-confirm-v1"
DEFAULT_QUESTIONS = Path("/home/dbw/ANCHOR/data/rule/test/mimic_test.jsonl")
DEFAULT_IMAGE_ROOT = Path("/home/dbw/ANCHOR/data/medheval/images")
DEFAULT_MODEL_DIR = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO_ROOT = Path("/home/dbw/HuatuoGPT-Vision")
DEFAULT_STYLE_BANK = Path(
    "/home/dbw/data/modality_centers/VQA-RAD/train/xray__chest.npy"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/dbw/ANCHOR/corrected_runs/style_phenomenon/"
    "huatuo_rule_mimic_n128_v1"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def patient_key(row: dict[str, Any]) -> str:
    matches = re.findall(r"(?:^|/)(p\d+)(?=/|$)", str(row["image"]))
    return max(matches, key=len) if matches else f"image:{row['image']}"


def stable_patient_balanced_rows(
    rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count % 2:
        raise ValueError("max_samples must be even")
    per_label = count // 2
    chosen: list[dict[str, Any]] = []
    used_patients: set[str] = set()
    for label in ("yes", "no"):
        candidates = [
            row for row in rows if explicit_label(row.get("answer")) == label
        ]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['question_id']}".encode()
            ).hexdigest()
        )
        selected = 0
        for row in candidates:
            patient = patient_key(row)
            if patient in used_patients:
                continue
            chosen.append(row)
            used_patients.add(patient)
            selected += 1
            if selected == per_label:
                break
        if selected != per_label:
            raise RuntimeError(
                f"only found {selected} independent-patient {label} rows"
            )
    return chosen


def binary_metrics(
    records: list[dict[str, Any]], view: str
) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    for row in records:
        truth = row["ground_truth"]
        prediction = row["scores"][view]["prediction"]
        tp += int(truth == "yes" and prediction == "yes")
        tn += int(truth == "no" and prediction == "no")
        fp += int(truth == "no" and prediction == "yes")
        fn += int(truth == "yes" and prediction == "no")
    n = len(records)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": (
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity
            else 0.0
        ),
        "predicted_positive_prevalence": (tp + fp) / n if n else 0.0,
    }


def wilson_interval(successes: int, total: int, z: float = 1.6448536269514722) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(
            rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def style_summary(
    records: list[dict[str, Any]], style: str
) -> dict[str, Any]:
    original = np.asarray(
        [row["scores"]["original"]["yes_minus_no"] for row in records],
        dtype=np.float64,
    )
    transformed = np.asarray(
        [row["scores"][style]["yes_minus_no"] for row in records],
        dtype=np.float64,
    )
    drift = transformed - original
    original_prediction = np.asarray(
        [row["scores"]["original"]["prediction"] for row in records]
    )
    transformed_prediction = np.asarray(
        [row["scores"][style]["prediction"] for row in records]
    )
    truth = np.asarray([row["ground_truth"] for row in records])
    flips = original_prediction != transformed_prediction
    original_correct = original_prediction == truth
    transformed_correct = transformed_prediction == truth
    rescue = int(np.sum(~original_correct & transformed_correct))
    harm = int(np.sum(original_correct & ~transformed_correct))
    discordant = rescue + harm
    mcnemar_p = (
        float(binomtest(rescue, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    abs_margin = np.abs(original)
    threshold = float(np.median(abs_margin))
    low = abs_margin <= threshold
    guards = [row["style_guards"][style]["passed"] for row in records]
    return {
        "metrics": binary_metrics(records, style),
        "paired_vs_original": {
            "decision_flips": int(np.sum(flips)),
            "decision_flip_rate": float(np.mean(flips)),
            "decision_flip_rate_wilson_90ci": wilson_interval(
                int(np.sum(flips)), len(records)
            ),
            "rescue": rescue,
            "harm": harm,
            "exact_mcnemar_p_two_sided": mcnemar_p,
            "mean_signed_yes_minus_no_drift": float(np.mean(drift)),
            "mean_absolute_margin_drift": float(np.mean(np.abs(drift))),
            "median_absolute_margin_drift": float(np.median(np.abs(drift))),
            "baseline_abs_margin_median": threshold,
            "low_margin_flip_rate": float(np.mean(flips[low])),
            "high_margin_flip_rate": float(np.mean(flips[~low])),
        },
        "style_guard_pass_rate": float(np.mean(guards)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO_ROOT)
    parser.add_argument("--style-bank", type=Path, default=DEFAULT_STYLE_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "raw.jsonl"
    rows = stable_patient_balanced_rows(
        load_jsonl(args.questions), args.max_samples, args.seed
    )
    bank = load_style_bank(args.style_bank)
    styles: dict[str, Callable[[Image.Image], Image.Image]] = {
        "lf_vqarad_l0.01_sr0.8": lambda image: feddg_frequency_interpolation(
            image, bank, low_frequency_ratio=0.01, source_ratio=0.8
        ),
        "gamma_0.9": lambda image: gamma_transform(image, 0.9),
        "gamma_1.1": lambda image: gamma_transform(image, 1.1),
    }
    config = {
        "version": VERSION,
        "evaluation_contract": "anchor-eval-contract-v1/CE-D",
        "created_at": now_iso(),
        "questions": str(args.questions.resolve()),
        "questions_sha256": sha256_file(args.questions),
        "selected_question_ids": [str(row["question_id"]) for row in rows],
        "selected_patient_ids": [patient_key(row) for row in rows],
        "selection": (
            "stable-hash seed; balanced explicit Yes/No; one question per patient"
        ),
        "model_dir": str(args.model_dir.resolve()),
        "model_config_sha256": sha256_file(args.model_dir / "config.json"),
        "model_index_sha256": sha256_file(
            args.model_dir / "model.safetensors.index.json"
        ),
        "style_bank": str(args.style_bank.resolve()),
        "style_bank_sha256": sha256_file(args.style_bank),
        "styles": {
            "lf_vqarad_l0.01_sr0.8": {
                "family": "low_frequency_amplitude_interpolation",
                "low_frequency_ratio": 0.01,
                "source_ratio": 0.8,
            },
            "gamma_0.9": {"family": "gamma", "gamma": 0.9},
            "gamma_1.1": {"family": "gamma", "gamma": 1.1},
        },
        "style_guard": {
            "minimum_psnr": 18.0,
            "minimum_edge_correlation": 0.90,
            "note": "pixel guard only; not a clinical equivalence certification",
        },
        "decision_track": {
            "prompt": build_prompt("{question}"),
            "verbalizers": ["Yes", "No"],
            "score": "first-answer-position FP32 Yes logit minus No logit",
            "generated_text_parser_used": False,
        },
        "primary_endpoint": "paired CE-D decision flip rate by frozen style",
        "minimum_effect": {
            "per_style_flip_rate": 0.05,
            "minimum_styles_meeting_effect": 2,
            "minimum_style_guard_pass_rate": 0.90,
        },
        "claim_boundary": (
            "Confirms or rejects style sensitivity of CE-D decisions. "
            "It does not establish that a transformation preserves all "
            "clinical evidence or that a flip is a hallucination."
        ),
        "seed": args.seed,
        "image_size": args.image_size,
        "device": args.device,
        "code_sha256_before_run": sha256_file(Path(__file__)),
    }
    write_json(args.output_dir / "config.json", config)

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device=args.device)
    yes_ids = bot.tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = bot.tokenizer.encode("No", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError(
            f"verbalizers are not single tokens: Yes={yes_ids}, No={no_ids}"
        )
    config["decision_track"]["verbalizer_token_ids"] = {
        "Yes": yes_ids[0],
        "No": no_ids[0],
    }
    write_json(args.output_dir / "config.json", config)

    for index, row in enumerate(rows):
        record: dict[str, Any] = {
            "version": VERSION,
            "question_id": str(row["question_id"]),
            "patient_id": patient_key(row),
            "question": str(row["question"]),
            "image": str(row["image"]),
            "ground_truth": explicit_label(row.get("answer")),
            "status": "error",
        }
        try:
            image_path = args.image_root / str(row["image"])
            record["image_sha256"] = sha256_file(image_path)
            with Image.open(image_path) as source:
                original = source.convert("RGB").resize(
                    (args.image_size, args.image_size),
                    Image.Resampling.BICUBIC,
                )
            transformed = {name: function(original) for name, function in styles.items()}
            prompt = build_prompt(str(row["question"]))
            scores = {
                "original": score_binary(
                    bot, prompt, original, yes_ids[0], no_ids[0]
                )
            }
            scores.update(
                {
                    name: score_binary(
                        bot, prompt, image, yes_ids[0], no_ids[0]
                    )
                    for name, image in transformed.items()
                }
            )
            guards = {}
            for name, image in transformed.items():
                metrics = structure_metrics(original, image)
                metrics["passed"] = bool(
                    metrics["psnr"] is not None
                    and metrics["psnr"] >= 18.0
                    and metrics["edge_correlation"] is not None
                    and metrics["edge_correlation"] >= 0.90
                )
                guards[name] = metrics
            record.update(
                {
                    "status": "ok",
                    "prompt": prompt,
                    "scores": scores,
                    "style_guards": guards,
                    "completed_at": now_iso(),
                }
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            record.update(
                {
                    "error": f"CUDA OOM: {error}",
                    "traceback": traceback.format_exc(),
                    "completed_at": now_iso(),
                }
            )
        except Exception as error:
            record.update(
                {
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "completed_at": now_iso(),
                }
            )
        append_jsonl(raw_path, record)
        if index == 0 or (index + 1) % 16 == 0 or index + 1 == len(rows):
            print(
                json.dumps(
                    {
                        "progress": f"{index + 1}/{len(rows)}",
                        "question_id": record["question_id"],
                        "status": record["status"],
                        "error": record.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    records = [row for row in load_jsonl(raw_path) if row.get("status") == "ok"]
    results = {
        "original": {"metrics": binary_metrics(records, "original")},
        **{name: style_summary(records, name) for name in styles},
    }
    meeting = [
        name
        for name in styles
        if results[name]["paired_vs_original"]["decision_flip_rate"] >= 0.05
        and results[name]["style_guard_pass_rate"] >= 0.90
    ]
    status = (
        "phenomenon_confirmed"
        if len(meeting) >= 2
        else "phenomenon_not_confirmed"
    )
    summary = {
        "version": VERSION,
        "n_requested": len(rows),
        "n_successful": len(records),
        "n_errors": len(rows) - len(records),
        "results": results,
        "styles_meeting_preregistered_effect": meeting,
        "phenomenon_status": status,
        "claim_boundary": config["claim_boundary"],
        "completed_at": now_iso(),
        "code_sha256_after_run": sha256_file(Path(__file__)),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
