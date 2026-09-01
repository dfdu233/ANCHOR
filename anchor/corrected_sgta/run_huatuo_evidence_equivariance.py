#!/usr/bin/env python3
"""Falsification-first pilot for semantic evidence equivariance.

The probe asks the same atomic finding claim through two exact semantic
involutions:

1. present vs absent Yes/No questions;
2. present/absent statements with the A/B order swapped.

For a genuine signed evidence score, complementing the claim or swapping the
answer labels must flip the score.  We decompose each score pair into an
equivariant component (the sign-flipping part) and an invariant residual (the
surface-form/answer bias part).  This is a grade-C pilot only; report-derived
labels must not be presented as reader ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata

from corrected_sgta.run_huatuo_style_prior_probe import (
    IGNORE_INDEX,
    import_huatuo,
    load_jsonl,
    sha256_file,
    write_json,
)


VERSION = "huatuo-evidence-equivariance-v1"
DEFAULT_MANIFEST = Path(
    "/home/dbw/ANCHOR/corrected_runs/clinical_selectivity/manifest_v3.jsonl"
)
DEFAULT_MODEL_DIR = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO_ROOT = Path("/home/dbw/HuatuoGPT-Vision")
DEFAULT_OUTPUT_DIR = Path(
    "/home/dbw/ANCHOR/corrected_runs/evidence_equivariance/huatuo_v1"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def semantic_prompts(finding: str) -> dict[str, str]:
    name = finding.replace("_", " ")
    suffix = (
        "Answer based only on the chest X-ray. Begin the answer with exactly "
    )
    return {
        "present_yes_no": (
            f"Is {name} present on this chest X-ray?\n{suffix}Yes or No."
        ),
        "absent_yes_no": (
            f"Is {name} absent on this chest X-ray?\n{suffix}Yes or No."
        ),
        "present_a": (
            "Which statement is better supported by this chest X-ray?\n"
            f"A. {name} is present.\nB. {name} is absent.\n"
            f"{suffix}A or B."
        ),
        "present_b": (
            "Which statement is better supported by this chest X-ray?\n"
            f"A. {name} is absent.\nB. {name} is present.\n"
            f"{suffix}A or B."
        ),
    }


def stable_balanced_split(
    rows: list[dict[str, Any]], split: str, maximum: int, seed: int
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("experiment_split") == split
        and row.get("reader_state") in {"supported", "refuted"}
        and Path(str(row.get("image_path", ""))).is_file()
    ]
    by_label = {
        label: [row for row in candidates if row["reader_state"] == label]
        for label in ("supported", "refuted")
    }
    for label in by_label:
        by_label[label].sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{split}:{row['image_id']}".encode()
            ).hexdigest()
        )
    per_label = min(maximum // 2, *(len(part) for part in by_label.values()))
    if per_label < 2:
        raise RuntimeError(f"too few balanced {split} rows: {per_label} per label")
    selected = by_label["supported"][:per_label] + by_label["refuted"][:per_label]
    selected.sort(key=lambda row: str(row["image_id"]))
    return selected


@torch.inference_mode()
def score_pair(
    bot: Any,
    prompt: str,
    image: Image.Image,
    first_token_id: int,
    second_token_id: int,
) -> dict[str, float]:
    prompt_with_image = bot.insert_image_placeholder(prompt, 1)
    prompt_ids = bot.preprocess(
        bot.get_conv_without_history(prompt_with_image), return_tensors="pt"
    ).to(bot.model.device)
    if int((prompt_ids < 0).sum()) != 1:
        raise RuntimeError("prompt must contain exactly one image placeholder")
    target = torch.tensor(
        [first_token_id], dtype=prompt_ids.dtype, device=prompt_ids.device
    )
    full = torch.cat((prompt_ids, target), dim=0)
    labels = torch.full_like(full, IGNORE_INDEX)
    labels[-1] = first_token_id
    attention = torch.ones_like(full, dtype=torch.bool)
    image_tensor = torch.stack(bot.get_image_tensors([image])).to(
        device=bot.model.device, dtype=torch.bfloat16
    )
    _, position_ids, expanded_attention, _, embeddings, expanded_labels = (
        bot.model.prepare_inputs_labels_for_multimodal_new(
            [full], None, [attention], None, [labels], image_tensor
        )
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
    mask = expanded_labels[:, 1:].ne(IGNORE_INDEX)
    hidden = output.last_hidden_state[:, :-1][mask]
    if hidden.shape[0] != 1:
        raise RuntimeError(f"expected one answer state, got {tuple(hidden.shape)}")
    weights = bot.model.get_output_embeddings().weight[
        [first_token_id, second_token_id]
    ].float()
    logits = hidden.float() @ weights.T
    first = float(logits[0, 0].cpu())
    second = float(logits[0, 1].cpu())
    return {"first_logit": first, "second_logit": second, "margin": first - second}


def decompose(first_margin: float, transformed_margin: float) -> dict[str, float]:
    """Project a score pair onto sign and invariant representations."""
    equivariant = 0.5 * (first_margin - transformed_margin)
    invariant = 0.5 * (first_margin + transformed_margin)
    violation = abs(first_margin + transformed_margin) / (
        abs(first_margin) + abs(transformed_margin) + 1e-12
    )
    return {
        "equivariant": equivariant,
        "invariant_residual": invariant,
        "normalized_violation": violation,
    }


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(scores, method="average")
    value = (ranks[labels].sum() - positive * (positive + 1) / 2) / (
        positive * negative
    )
    return float(value)


def balanced_accuracy(y: np.ndarray, prediction: np.ndarray) -> float:
    positive = y == 1
    negative = ~positive
    sensitivity = float(np.mean(prediction[positive] == 1))
    specificity = float(np.mean(prediction[negative] == 0))
    return 0.5 * (sensitivity + specificity)


def fit_threshold(y: np.ndarray, score: np.ndarray) -> float:
    unique = np.unique(score)
    candidates = np.concatenate(
        ([unique[0] - 1e-6], 0.5 * (unique[:-1] + unique[1:]), [unique[-1] + 1e-6])
    )
    ranked = sorted(
        (
            (-balanced_accuracy(y, (score >= threshold).astype(int)), abs(threshold), threshold)
            for threshold in candidates
        )
    )
    return float(ranked[0][2])


def classification_metrics(
    y: np.ndarray, score: np.ndarray, threshold: float
) -> dict[str, float | int | None]:
    prediction = (score >= threshold).astype(int)
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    return {
        "n": len(y),
        "threshold_from_dev": threshold,
        "accuracy": float(np.mean(prediction == y)),
        "balanced_accuracy": balanced_accuracy(y, prediction),
        "positive_claim_precision": tp / (tp + fp) if tp + fp else None,
        "positive_claim_hallucination_rate": fp / (tp + fp) if tp + fp else None,
        "negative_case_false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "positive_case_omission_rate": fn / (fn + tp) if fn + tp else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support_auroc": auroc(y, score),
    }


def bootstrap_accuracy_delta(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    baseline_threshold: float,
    candidate_threshold: float,
    seed: int,
    draws: int = 2000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    base_prediction = baseline >= baseline_threshold
    candidate_prediction = candidate >= candidate_threshold
    for _ in range(draws):
        indices = rng.integers(0, len(y), size=len(y))
        values.append(
            float(
                np.mean(candidate_prediction[indices] == y[indices])
                - np.mean(base_prediction[indices] == y[indices])
            )
        )
    return {
        "estimate": float(
            np.mean(candidate_prediction == y) - np.mean(base_prediction == y)
        ),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    ok = [row for row in records if row.get("status") == "ok"]
    return {
        "y": np.asarray([row["label"] for row in ok], dtype=int),
        "direct": np.asarray([row["scores"]["present_yes_no"]["margin"] for row in ok]),
        "negation_projected": np.asarray(
            [row["decomposition"]["negation"]["equivariant"] for row in ok]
        ),
        "negation_violation": np.asarray(
            [row["decomposition"]["negation"]["normalized_violation"] for row in ok]
        ),
        "option_projected": np.asarray(
            [row["decomposition"]["option_order"]["equivariant"] for row in ok]
        ),
        "option_violation": np.asarray(
            [row["decomposition"]["option_order"]["normalized_violation"] for row in ok]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-per-split", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    manifest = load_jsonl(args.manifest)
    rows_by_split = {
        split: stable_balanced_split(
            manifest, split, args.max_per_split, args.seed
        )
        for split in ("dev", "test")
    }
    rows = [
        {**row, "probe_split": split}
        for split, selected in rows_by_split.items()
        for row in selected
    ]
    config = {
        "version": VERSION,
        "created_at": now_iso(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "model_dir": str(args.model_dir.resolve()),
        "model_config_sha256": sha256_file(args.model_dir / "config.json"),
        "n_by_split": {key: len(value) for key, value in rows_by_split.items()},
        "evidence_grade": "C",
        "formal_reference": False,
        "frozen_hypothesis": (
            "A genuine visual claim score changes sign under polarity complement "
            "and label permutation. Its equivariant projection predicts support "
            "better than the raw direct margin; the invariant residual is enriched "
            "in errors."
        ),
        "falsification": (
            "Reject as a method candidate if neither projection improves held-out "
            "accuracy/AUROC over a dev-calibrated direct margin, or if any gain is "
            "explained solely by syntactic negation failure."
        ),
        "claim_boundary": (
            "Atomic chest-X-ray finding polarity only. Report-derived binary QA "
            "labels are a screening proxy, not multi-reader clinical truth."
        ),
        "device": args.device,
        "seed": args.seed,
    }
    write_json(args.output_dir / "config.json", config)

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device=args.device)
    token_ids = {}
    for token in ("Yes", "No", "A", "B"):
        encoded = bot.tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"verbalizer is not one token: {token}={encoded}")
        token_ids[token] = encoded[0]
    config["verbalizer_token_ids"] = token_ids
    write_json(args.output_dir / "config.json", config)

    raw_path = args.output_dir / "raw.jsonl"
    for index, row in enumerate(rows):
        record: dict[str, Any] = {
            "version": VERSION,
            "image_id": row["image_id"],
            "finding": row["finding"],
            "probe_split": row["probe_split"],
            "reader_state": row["reader_state"],
            "label": int(row["reader_state"] == "supported"),
            "status": "error",
        }
        try:
            image_path = Path(row["image_path"])
            with Image.open(image_path) as source:
                image = source.convert("RGB").resize(
                    (args.image_size, args.image_size), Image.Resampling.BICUBIC
                )
            prompts = semantic_prompts(str(row["finding"]))
            scores = {
                "present_yes_no": score_pair(
                    bot, prompts["present_yes_no"], image, token_ids["Yes"], token_ids["No"]
                ),
                "absent_yes_no": score_pair(
                    bot, prompts["absent_yes_no"], image, token_ids["Yes"], token_ids["No"]
                ),
                "present_a": score_pair(
                    bot, prompts["present_a"], image, token_ids["A"], token_ids["B"]
                ),
                "present_b": score_pair(
                    bot, prompts["present_b"], image, token_ids["A"], token_ids["B"]
                ),
            }
            record.update(
                {
                    "status": "ok",
                    "image_path": str(image_path),
                    "image_sha256": sha256_file(image_path),
                    "prompts": prompts,
                    "scores": scores,
                    "decomposition": {
                        "negation": decompose(
                            scores["present_yes_no"]["margin"],
                            scores["absent_yes_no"]["margin"],
                        ),
                        "option_order": decompose(
                            scores["present_a"]["margin"],
                            scores["present_b"]["margin"],
                        ),
                    },
                    "completed_at": now_iso(),
                }
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            record.update({"error": f"CUDA OOM: {error}", "traceback": traceback.format_exc()})
        except Exception as error:
            record.update({"error": repr(error), "traceback": traceback.format_exc()})
        append_jsonl(raw_path, record)
        if index == 0 or (index + 1) % 8 == 0 or index + 1 == len(rows):
            print(json.dumps({"progress": f"{index + 1}/{len(rows)}", "status": record["status"]}), flush=True)

    records = load_jsonl(raw_path)
    split_records = {
        split: [row for row in records if row.get("probe_split") == split and row.get("status") == "ok"]
        for split in ("dev", "test")
    }
    dev = arrays(split_records["dev"])
    test = arrays(split_records["test"])
    methods = ("direct", "negation_projected", "option_projected")
    thresholds = {name: fit_threshold(dev["y"], dev[name]) for name in methods}
    metrics = {
        name: classification_metrics(test["y"], test[name], thresholds[name])
        for name in methods
    }
    direct_prediction = (test["direct"] >= thresholds["direct"]).astype(int)
    errors = (direct_prediction != test["y"]).astype(int)
    summary = {
        "version": VERSION,
        "status": "complete",
        "n_successful_by_split": {key: len(value) for key, value in split_records.items()},
        "n_errors": len(records) - sum(len(value) for value in split_records.values()),
        "test_metrics": metrics,
        "paired_accuracy_delta_vs_direct": {
            name: bootstrap_accuracy_delta(
                test["y"], test["direct"], test[name], thresholds["direct"], thresholds[name], args.seed + offset
            )
            for offset, name in enumerate(("negation_projected", "option_projected"), start=1)
        },
        "raw_error_detection_auroc": {
            "negation_equivariance_violation": auroc(errors, test["negation_violation"]),
            "option_order_equivariance_violation": auroc(errors, test["option_violation"]),
            "low_direct_confidence": auroc(errors, -np.abs(test["direct"] - thresholds["direct"])),
        },
        "mean_normalized_violation": {
            "negation": float(np.mean(test["negation_violation"])),
            "option_order": float(np.mean(test["option_violation"])),
        },
        "decision": (
            "retain_for_formal_validation"
            if any(
                metrics[name]["accuracy"] > metrics["direct"]["accuracy"]
                and metrics[name]["support_auroc"] > metrics["direct"]["support_auroc"]
                for name in ("negation_projected", "option_projected")
            )
            else "prune_as_mitigation_candidate"
        ),
        "claim_boundary": config["claim_boundary"],
        "completed_at": now_iso(),
        "code_sha256_after_run": sha256_file(Path(__file__)),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
