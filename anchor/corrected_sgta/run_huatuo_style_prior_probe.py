#!/usr/bin/env python3
"""Probe whether appearance style switches Huatuo's clinical answer prior.

The probe deliberately factorizes a content-preserving style change from a
content-removed, style-preserved control. It scores frozen Yes/No verbalizers
at the first answer position, so no generated-text parser enters the primary
measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr

from corrected_sgta.methods import feddg_frequency_interpolation


VERSION = "huatuo-style-conditioned-prior-probe-v3"
IGNORE_INDEX = -100
DEFAULT_QUESTIONS = Path("/home/dbw/ANCHOR/data/rule/test/mimic_test.jsonl")
DEFAULT_IMAGE_ROOT = Path("/home/dbw/ANCHOR/data/medheval/images")
DEFAULT_MODEL_DIR = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO_ROOT = Path("/home/dbw/HuatuoGPT-Vision")
DEFAULT_STYLE_BANK = Path(
    "/home/dbw/data/modality_centers/VQA-RAD/train/xray__chest.npy"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/dbw/ANCHOR/corrected_runs/style_prior_probe/"
    "huatuo_rule_mimic_n16_v1"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def explicit_label(value: object) -> str | None:
    match = re.match(r"^\s*(yes|no)\b", str(value), flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def stable_balanced_rows(
    rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count % 2:
        raise ValueError("max_samples must be even for the frozen balanced probe")
    per_label = count // 2
    chosen: list[dict[str, Any]] = []
    used_images: set[str] = set()
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
            image = str(row["image"])
            if image in used_images:
                continue
            chosen.append(row)
            used_images.add(image)
            selected += 1
            if selected == per_label:
                break
        if selected != per_label:
            raise RuntimeError(f"only found {selected} unique-image {label} rows")
    return chosen


def build_prompt(question: str) -> str:
    clean = question.replace("<image>", "").strip()
    return (
        f"{clean}\n"
        "Answer based only on the chest X-ray. "
        "Begin the answer with exactly Yes or No."
    )


def structure_metrics(
    source: Image.Image, transformed: Image.Image
) -> dict[str, float | None]:
    left = np.asarray(source.convert("L"), dtype=np.float64) / 255.0
    right = np.asarray(transformed.convert("L"), dtype=np.float64) / 255.0
    mse = float(np.mean((left - right) ** 2))
    source_edge = np.hypot(*np.gradient(left))
    target_edge = np.hypot(*np.gradient(right))
    source_centered = source_edge.ravel() - source_edge.mean()
    target_centered = target_edge.ravel() - target_edge.mean()
    denominator = float(
        np.linalg.norm(source_centered) * np.linalg.norm(target_centered)
    )
    edge_correlation = (
        float(
            np.clip(
                source_centered @ target_centered / denominator,
                -1.0,
                1.0,
            )
        )
        if denominator > 1e-12
        else 1.0
    )
    return {
        "pixel_mse": mse,
        "psnr": None if mse <= 1e-12 else float(-10.0 * math.log10(mse)),
        "edge_correlation": edge_correlation,
    }


def load_style_bank(path: Path) -> np.ndarray:
    bank = np.load(path)
    if bank.ndim == 3 and bank.shape[-1] in (1, 3):
        bank = bank.transpose(2, 0, 1)
    if bank.ndim not in (2, 3):
        raise ValueError(f"unsupported style bank shape: {bank.shape}")
    return np.asarray(bank)


def random_phase(
    shape: tuple[int, int, int], seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(128.0, 32.0, size=shape)
    spectrum = np.fft.fft2(noise, axes=(-2, -1))
    phase = np.angle(spectrum)
    phase[:, 0, 0] = 0.0
    return phase


def reconstruct_with_phase(image: Image.Image, phase: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float64).transpose(2, 0, 1)
    amplitude = np.abs(np.fft.fft2(array, axes=(-2, -1)))
    reconstructed = np.fft.ifft2(
        amplitude * np.exp(1j * phase), axes=(-2, -1)
    ).real
    clipped = np.clip(np.rint(reconstructed), 0, 255)
    return Image.fromarray(
        clipped.transpose(1, 2, 0).astype(np.uint8), mode="RGB"
    )


def null_pair(
    original: Image.Image, styled: Image.Image, seed: int
) -> tuple[Image.Image, Image.Image]:
    shape = (3, original.height, original.width)
    phase = random_phase(shape, seed)
    return (
        reconstruct_with_phase(original, phase),
        reconstruct_with_phase(styled, phase),
    )


def import_huatuo(root: Path):
    sys.path.insert(0, str(root))
    from cli import HuatuoChatbot  # type: ignore

    return HuatuoChatbot


@torch.inference_mode()
def score_binary(
    bot: Any,
    prompt: str,
    image: Image.Image,
    yes_token_id: int,
    no_token_id: int,
    *,
    zero_visual: bool = False,
) -> dict[str, float | str]:
    prompt_with_image = bot.insert_image_placeholder(prompt, 1)
    prompt_ids = bot.preprocess(
        bot.get_conv_without_history(prompt_with_image),
        return_tensors="pt",
    ).to(bot.model.device)
    if int((prompt_ids < 0).sum()) != 1:
        raise RuntimeError("prompt must contain exactly one image placeholder")

    target = torch.tensor(
        [yes_token_id], dtype=prompt_ids.dtype, device=prompt_ids.device
    )
    full = torch.cat((prompt_ids, target), dim=0)
    labels = torch.full_like(full, IGNORE_INDEX)
    labels[-1] = yes_token_id
    attention = torch.ones_like(full, dtype=torch.bool)
    image_tensor = torch.stack(bot.get_image_tensors([image])).to(
        device=bot.model.device, dtype=torch.bfloat16
    )
    if zero_visual:
        image_tensor = torch.zeros_like(image_tensor)
    (
        _,
        position_ids,
        expanded_attention,
        _,
        embeddings,
        expanded_labels,
    ) = bot.model.prepare_inputs_labels_for_multimodal_new(
        [full],
        None,
        [attention],
        None,
        [labels],
        image_tensor,
    )
    if embeddings is None or expanded_labels is None:
        raise RuntimeError("multimodal expansion returned no embeddings/labels")
    output = bot.model.model(
        input_ids=None,
        attention_mask=expanded_attention,
        position_ids=position_ids,
        inputs_embeds=embeddings,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    shifted_labels = expanded_labels[:, 1:]
    mask = shifted_labels.ne(IGNORE_INDEX)
    hidden = output.last_hidden_state[:, :-1][mask]
    if hidden.shape[0] != 1:
        raise RuntimeError(
            f"expected one answer-position state, got {tuple(hidden.shape)}"
        )
    output_weight = bot.model.get_output_embeddings().weight
    selected_weight = output_weight[[yes_token_id, no_token_id]].float()
    selected_logits = hidden.float() @ selected_weight.T
    yes_logit = float(selected_logits[0, 0].cpu())
    no_logit = float(selected_logits[0, 1].cpu())
    pair = torch.tensor([yes_logit, no_logit], dtype=torch.float64)
    probabilities = torch.softmax(pair, dim=0)
    return {
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "yes_minus_no": yes_logit - no_logit,
        "yes_probability_binary": float(probabilities[0]),
        "prediction": "yes" if yes_logit >= no_logit else "no",
    }


def safe_statistic(
    function: Any, left: np.ndarray, right: np.ndarray
) -> tuple[float | None, float | None]:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None, None
    result = function(left, right)
    return float(result.statistic), float(result.pvalue)


def bootstrap_spearman(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> list[float | None]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(left), size=len(left))
        value, _ = safe_statistic(spearmanr, left[indices], right[indices])
        if value is not None and np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return [None, None]
    return [
        float(np.quantile(estimates, 0.05)),
        float(np.quantile(estimates, 0.95)),
    ]


def permutation_pvalue(
    left: np.ndarray,
    right: np.ndarray,
    observed: float | None,
    *,
    seed: int,
    samples: int,
) -> float | None:
    if observed is None:
        return None
    rng = np.random.default_rng(seed)
    exceed = 0
    valid = 0
    for _ in range(samples):
        shuffled = rng.permutation(right)
        value, _ = safe_statistic(spearmanr, left, shuffled)
        if value is None:
            continue
        valid += 1
        exceed += int(value >= observed)
    return (exceed + 1) / (valid + 1) if valid else None


def accuracy(records: list[dict[str, Any]], view: str) -> float:
    correct = sum(
        row["scores"][view]["prediction"] == row["ground_truth"]
        for row in records
    )
    return correct / len(records) if records else 0.0


def summarize(
    records: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    successful = [row for row in records if row.get("status") == "ok"]
    full_delta = np.asarray(
        [
            row["scores"]["styled_full"]["yes_minus_no"]
            - row["scores"]["original_full"]["yes_minus_no"]
            for row in successful
        ],
        dtype=np.float64,
    )
    null_names = sorted(
        key.removeprefix("original_null_")
        for key in successful[0]["scores"]
        if key.startswith("original_null_")
    ) if successful else []
    if len(null_names) < 4 or len(null_names) % 2:
        raise RuntimeError(
            "summary requires an even number of at least four null replicates"
        )
    null_matrix = np.asarray(
        [
            [
                row["scores"][f"styled_null_{name}"]["yes_minus_no"]
                - row["scores"][f"original_null_{name}"]["yes_minus_no"]
                for name in null_names
            ]
            for row in successful
        ],
        dtype=np.float64,
    )
    midpoint = len(null_names) // 2
    null_delta_a = null_matrix[:, :midpoint].mean(axis=1)
    null_delta_b = null_matrix[:, midpoint:].mean(axis=1)
    null_delta = null_matrix.mean(axis=1)
    baseline_margin = np.asarray(
        [
            abs(row["scores"]["original_full"]["yes_minus_no"])
            for row in successful
        ],
        dtype=np.float64,
    )
    label_sign = np.asarray(
        [1.0 if row["ground_truth"] == "yes" else -1.0 for row in successful],
        dtype=np.float64,
    )
    visual_support = np.asarray(
        [
            row["scores"]["original_full"]["yes_minus_no"]
            - row["scores"]["zero_visual"]["yes_minus_no"]
            for row in successful
        ],
        dtype=np.float64,
    )
    correct_oriented_support = label_sign * visual_support
    correct_oriented_style_effect = label_sign * full_delta

    pearson, pearson_p = safe_statistic(pearsonr, full_delta, null_delta)
    spearman, spearman_p = safe_statistic(spearmanr, full_delta, null_delta)
    null_reliability, null_reliability_p = safe_statistic(
        spearmanr, null_delta_a, null_delta_b
    )
    gain_spearman, gain_spearman_p = safe_statistic(
        spearmanr, correct_oriented_support, correct_oriented_style_effect
    )
    gain_pearson, gain_pearson_p = safe_statistic(
        pearsonr, correct_oriented_support, correct_oriented_style_effect
    )
    bootstrap_ci = bootstrap_spearman(
        full_delta, null_delta, seed=seed + 301, samples=5000
    )
    permutation_p = permutation_pvalue(
        full_delta,
        null_delta,
        spearman,
        seed=seed + 401,
        samples=10000,
    )
    nonzero = (np.abs(full_delta) > 1e-8) & (np.abs(null_delta) > 1e-8)
    sign_agreement = (
        float(np.mean(np.sign(full_delta[nonzero]) == np.sign(null_delta[nonzero])))
        if np.any(nonzero)
        else None
    )
    cosine = (
        float(
            full_delta @ null_delta
            / (np.linalg.norm(full_delta) * np.linalg.norm(null_delta))
        )
        if np.linalg.norm(full_delta) > 1e-12
        and np.linalg.norm(null_delta) > 1e-12
        else None
    )
    median_margin = float(np.median(baseline_margin)) if len(baseline_margin) else None
    low = baseline_margin <= median_margin if median_margin is not None else np.zeros(0)
    original_predictions = [
        row["scores"]["original_full"]["prediction"] for row in successful
    ]
    styled_predictions = [
        row["scores"]["styled_full"]["prediction"] for row in successful
    ]
    flips = np.asarray(
        [
            original != styled
            for original, styled in zip(original_predictions, styled_predictions)
        ],
        dtype=np.float64,
    )

    guard_pass = [
        row["style_guard"]["passed"] for row in successful
    ]
    minimum = {
        "spearman": 0.40,
        "sign_agreement": 0.625,
        "null_replicate_spearman": 0.30,
        "one_sided_permutation_p": 0.10,
    }
    if len(successful) < 12 or np.mean(guard_pass) < 0.90:
        status = "invalid_probe"
    elif null_reliability is None or null_reliability < minimum[
        "null_replicate_spearman"
    ]:
        status = "inconclusive_null_control_unreliable"
    elif (
        spearman is not None
        and spearman >= minimum["spearman"]
        and sign_agreement is not None
        and sign_agreement >= minimum["sign_agreement"]
        and permutation_p is not None
        and permutation_p < minimum["one_sided_permutation_p"]
    ):
        status = "mechanism_signal"
    elif (
        spearman is not None
        and spearman <= 0.0
        and bootstrap_ci[1] is not None
        and bootstrap_ci[1] < minimum["spearman"]
    ):
        status = "strong_mechanism_falsified"
    else:
        status = "inconclusive"

    return {
        "version": VERSION,
        "n_requested": len(records),
        "n_successful": len(successful),
        "n_errors": len(records) - len(successful),
        "primary_endpoint": (
            "Spearman correlation between full-view style-induced Yes-No "
            "margin drift and the mean matched style-only drift"
        ),
        "minimum_effect_preregistered": minimum,
        "primary": {
            "pearson_r": pearson,
            "pearson_p_two_sided": pearson_p,
            "spearman_rho": spearman,
            "spearman_p_two_sided": spearman_p,
            "spearman_bootstrap_90ci": bootstrap_ci,
            "spearman_permutation_p_one_sided": permutation_p,
            "uncentered_cosine": cosine,
            "sign_agreement": sign_agreement,
            "nonzero_pairs": int(np.sum(nonzero)),
        },
        "measurement_validity": {
            "null_split_half_spearman": null_reliability,
            "null_split_half_p_two_sided": null_reliability_p,
            "null_replicates": len(null_names),
            "style_guard_pass_rate": float(np.mean(guard_pass))
            if guard_pass
            else 0.0,
        },
        "decision_diagnostics": {
            "original_accuracy": accuracy(successful, "original_full"),
            "styled_accuracy": accuracy(successful, "styled_full"),
            "full_prediction_flips": int(flips.sum()),
            "full_prediction_flip_rate": float(flips.mean()) if len(flips) else 0.0,
            "baseline_abs_margin_median": median_margin,
            "low_margin_flip_rate": float(flips[low].mean())
            if np.any(low)
            else None,
            "high_margin_flip_rate": float(flips[~low].mean())
            if np.any(~low)
            else None,
        },
        "style_content_interaction_diagnostic": {
            "definition": (
                "correlate correct-oriented visual support "
                "(original minus zero-visual margin) with the correct-oriented "
                "style effect (styled minus original margin)"
            ),
            "spearman_rho": gain_spearman,
            "spearman_p_two_sided": gain_spearman_p,
            "pearson_r": gain_pearson,
            "pearson_p_two_sided": gain_pearson_p,
            "visual_support_values": correct_oriented_support.tolist(),
            "style_effect_values": correct_oriented_style_effect.tolist(),
            "interpretation": (
                "A stable nonzero association supports evidence-gain "
                "modulation; a stable style-only null shift would instead "
                "support an additive style prior."
            ),
        },
        "drift_descriptives": {
            "full_mean": float(full_delta.mean()) if len(full_delta) else None,
            "full_std": float(full_delta.std()) if len(full_delta) else None,
            "null_mean": float(null_delta.mean()) if len(null_delta) else None,
            "null_std": float(null_delta.std()) if len(null_delta) else None,
            "full_values": full_delta.tolist(),
            "null_values": null_delta.tolist(),
            "null_first_half_values": null_delta_a.tolist(),
            "null_second_half_values": null_delta_b.tolist(),
            "null_replicate_matrix": null_matrix.tolist(),
        },
        "probe_status": status,
        "claim_ceiling": (
            "A mechanism plausibility signal on one model, one binary CXR task, "
            "and one pre-registered style transform; not a mitigation claim."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO_ROOT)
    parser.add_argument("--style-bank", type=Path, default=DEFAULT_STYLE_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--low-frequency-ratio", type=float, default=0.01)
    parser.add_argument("--source-ratio", type=float, default=0.8)
    parser.add_argument("--min-psnr", type=float, default=18.0)
    parser.add_argument("--min-edge-correlation", type=float, default=0.90)
    parser.add_argument("--null-replicates", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.null_replicates < 4 or args.null_replicates % 2:
        raise ValueError("--null-replicates must be even and at least 4")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "raw.jsonl"
    rows = stable_balanced_rows(
        load_jsonl(args.questions), args.max_samples, args.seed
    )
    bank = load_style_bank(args.style_bank)
    model_index = args.model_dir / "model.safetensors.index.json"
    model_config = args.model_dir / "config.json"
    config = {
        "version": VERSION,
        "evaluation_contract": "anchor-eval-contract-v1/CE-D",
        "created_at": now_iso(),
        "questions": str(args.questions.resolve()),
        "questions_sha256": sha256_file(args.questions),
        "selected_question_ids": [str(row["question_id"]) for row in rows],
        "selection": (
            "stable-hash seed; balanced explicit Yes/No; one question per image"
        ),
        "model_dir": str(args.model_dir.resolve()),
        "model_config_sha256": sha256_file(model_config),
        "model_index_sha256": sha256_file(model_index),
        "huatuo_root": str(args.huatuo_root.resolve()),
        "style_bank": str(args.style_bank.resolve()),
        "style_bank_sha256": sha256_file(args.style_bank),
        "style_transform": {
            "name": "low-frequency amplitude interpolation",
            "low_frequency_ratio": args.low_frequency_ratio,
            "source_ratio": args.source_ratio,
            "image_size": args.image_size,
            "guard": {
                "minimum_psnr": args.min_psnr,
                "minimum_edge_correlation": args.min_edge_correlation,
            },
        },
        "null_control": {
            "construction": (
                "Monte Carlo shared-random-phase reconstruction; each pair "
                "keeps the original/styled amplitudes and shares its phase"
            ),
            "replicates": args.null_replicates,
            "reliability": "Spearman between first-half and second-half means",
        },
        "decision_track": {
            "prompt": build_prompt("{question}"),
            "verbalizers": ["Yes", "No"],
            "score": "first-answer-position Yes logit minus No logit",
            "output_dot_product_dtype": "float32",
            "generated_text_parser_used": False,
        },
        "primary_endpoint": (
            "Spearman(full style margin drift, mean matched null style margin drift)"
        ),
        "secondary_preregistered_endpoint": (
            "association between correct-oriented original-vs-zero visual "
            "support and correct-oriented style effect"
        ),
        "secondary_minimum_effect": {"absolute_spearman": 0.40},
        "minimum_effect": {
            "spearman": 0.40,
            "sign_agreement": 0.625,
            "null_replicate_spearman": 0.30,
            "one_sided_permutation_p": 0.10,
        },
        "simple_alternative": (
            "low decision margin alone causes arbitrary flips; it does not "
            "predict alignment with matched style-only drift"
        ),
        "seed": args.seed,
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
            "question": str(row["question"]),
            "image": str(row["image"]),
            "ground_truth": explicit_label(row.get("answer")),
            "status": "error",
        }
        try:
            image_path = args.image_root / str(row["image"])
            with Image.open(image_path) as source:
                original = source.convert("RGB").resize(
                    (args.image_size, args.image_size),
                    Image.Resampling.BICUBIC,
                )
            styled = feddg_frequency_interpolation(
                original,
                bank,
                low_frequency_ratio=args.low_frequency_ratio,
                source_ratio=args.source_ratio,
            )
            metrics = structure_metrics(original, styled)
            style_passed = bool(
                metrics["psnr"] is not None
                and metrics["psnr"] >= args.min_psnr
                and metrics["edge_correlation"] is not None
                and metrics["edge_correlation"] >= args.min_edge_correlation
            )
            prompt = build_prompt(str(row["question"]))
            views = {
                "original_full": original,
                "styled_full": styled,
            }
            null_views: list[tuple[Image.Image, Image.Image]] = []
            for replicate in range(args.null_replicates):
                pair = null_pair(
                    original,
                    styled,
                    args.seed + index * 1009 + replicate + 1,
                )
                null_views.append(pair)
                name = f"{replicate:02d}"
                views[f"original_null_{name}"] = pair[0]
                views[f"styled_null_{name}"] = pair[1]
            scores = {
                name: score_binary(
                    bot, prompt, view, yes_ids[0], no_ids[0]
                )
                for name, view in views.items()
            }
            scores["zero_visual"] = score_binary(
                bot,
                prompt,
                original,
                yes_ids[0],
                no_ids[0],
                zero_visual=True,
            )
            record.update(
                {
                    "status": "ok",
                    "prompt": prompt,
                    "style_guard": {
                        **metrics,
                        "passed": style_passed,
                    },
                    "null_content_destruction": {
                        f"original_vs_null_{replicate:02d}": structure_metrics(
                            original, pair[0]
                        )
                        for replicate, pair in enumerate(null_views[:2])
                    },
                    "scores": scores,
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

    records = load_jsonl(raw_path)
    summary = summarize(records, args.seed)
    summary["completed_at"] = now_iso()
    summary["code_sha256_after_run"] = sha256_file(Path(__file__))
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
