"""Exploratory CLEAR finding anchor on the fixed MedHEval CXR n=32 subset.

The finding-to-prompt mapping is a pre-registered, exact question-text lookup.
It never receives an answer or label.  CLEAR paired-prompt scores are diagnostic
cosine similarities, not calibrated clinical probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from corrected_sgta.protocol_v2 import (
    file_sha256,
    ground_truth_index,
    normalize_text,
    protocol_fingerprint,
    resolve_image,
)


VERSION = "clear-finding-anchor-probe-v1"
MAPPING_VERSION = "medheval-cxr-question-lexical-v1"
CLEAR_REPO_COMMIT = "57d2eae57ddabf8e56c8013ec675336d71b3d8e7"
FIXED_DATASET_SHA256 = (
    "a8c939b1519eca88e14bd93315b8456e87b22149076462605fc3669d613be7ac"
)
DEFAULT_DATASET = Path(
    "corrected_runs/medheval_mitigation_smoke_v1/"
    "cxr_vishal/subset_seed42_n32.json"
)
DEFAULT_CLEAR_REPO = Path("/root/autodl-tmp/CLEAR")
DEFAULT_TORCH_HOME = Path("/root/autodl-tmp/torch_cache")


@dataclass(frozen=True)
class PromptPair:
    finding_id: str
    positive: str
    negative: str
    tier: str = "strict"


def _question_key(question: str) -> str:
    return normalize_text(question)


def _pairs(
    questions: Iterable[str],
    finding_id: str,
    positive: str,
    negative: str,
    *,
    tier: str = "strict",
) -> dict[str, PromptPair]:
    pair = PromptPair(finding_id, positive, negative, tier)
    return {_question_key(question): pair for question in questions}


# Constructed only from question wording.  In particular, normality questions
# reverse the prompt polarity so index 0 always means the answer "Yes".
STRICT_QUESTION_MAP: dict[str, PromptPair] = {
    **_pairs(
        (
            "Is there any pleural effusion present in the image?",
            "Is there any evidence of pleural effusion in the image?",
            "Is there a pleural effusion present in the image?",
            "Is there a pleural effusion present?",
            "Can you identify any effusion in the image?",
            "Does the image show any signs of pleural effusion?",
            "Does the image show any pleural effusions?",
            "Is there any evidence of pleural effusion in the image?",
        ),
        "pleural_effusion",
        "pleural effusion",
        "no pleural effusion",
    ),
    **_pairs(
        (
            "Does the image show any signs of pneumothorax?",
            "Is there evidence of pneumothorax in the image?",
            "Is there any evidence of pneumothorax in the image?",
            "Can you identify any pneumothoraces in the image?",
        ),
        "pneumothorax",
        "pneumothorax",
        "no pneumothorax",
    ),
    **_pairs(
        (
            "Does the image show any acute bony abnormalities?",
            "Are there any acute bony findings?",
        ),
        "acute_osseous_abnormality",
        "acute osseous abnormality",
        "no acute osseous abnormality",
    ),
    **_pairs(
        ("Does the image show any noncalcified pulmonary nodules?",),
        "noncalcified_pulmonary_nodule",
        "noncalcified pulmonary nodule",
        "no noncalcified pulmonary nodule",
    ),
    **_pairs(
        ("Is there any evidence of pulmonary edema?",),
        "pulmonary_edema",
        "pulmonary edema",
        "no pulmonary edema",
    ),
    **_pairs(
        (
            "Does the image show any abnormalities in the cardiomediastinal silhouette?",
            "Is the cardio mediastinal silhouette remarkable?",
        ),
        "abnormal_cardiomediastinal_silhouette",
        "abnormal cardiomediastinal silhouette",
        "normal cardiomediastinal silhouette",
    ),
    **_pairs(
        (
            "Does the image show any abnormalities in the cardiac size?",
            "Is the heart size abnormal?",
        ),
        "abnormal_heart_size",
        "abnormal heart size",
        "normal heart size",
    ),
    **_pairs(
        (
            "Is the heart size within normal limits according to the image?",
            "Does the image show the heart size to be within normal limits?",
        ),
        "normal_heart_size",
        "normal heart size",
        "abnormal heart size",
    ),
    **_pairs(
        ("Can we see any suspicious pulmonary opacity in the image?",),
        "suspicious_pulmonary_opacity",
        "suspicious pulmonary opacity",
        "no suspicious pulmonary opacity",
    ),
    **_pairs(
        ("Can you locate any focal consolidation in the lungs?",),
        "focal_consolidation",
        "focal consolidation",
        "no focal consolidation",
    ),
}

EXTENDED_QUESTION_MAP: dict[str, PromptPair] = {
    **_pairs(
        ("Are the pulmonary vascularity levels normal according to the image?",),
        "normal_pulmonary_vascularity",
        "normal pulmonary vascularity",
        "abnormal pulmonary vascularity",
        tier="extended",
    ),
    **_pairs(
        ("Can you identify any degenerative changes in the image?",),
        "degenerative_change",
        "degenerative change",
        "no degenerative change",
        tier="extended",
    ),
    **_pairs(
        ("Are the lungs clear in the image?",),
        "clear_lungs",
        "clear lungs",
        "abnormal lung opacity",
        tier="extended",
    ),
}


def mapping_sha256() -> str:
    manifest = {
        "strict": {
            question: asdict(pair)
            for question, pair in sorted(STRICT_QUESTION_MAP.items())
        },
        "extended": {
            question: asdict(pair)
            for question, pair in sorted(EXTENDED_QUESTION_MAP.items())
        },
    }
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def map_question(question: str, *, include_extended: bool = True) -> PromptPair | None:
    """Return a pre-registered prompt pair using question text alone."""

    key = _question_key(question)
    pair = STRICT_QUESTION_MAP.get(key)
    if pair is None and include_extended:
        pair = EXTENDED_QUESTION_MAP.get(key)
    return pair


def prediction_from_vlm_cache(row: dict[str, Any]) -> int:
    """Use the original-view full-string NLL when available."""

    labels = [str(label).lower() for label in row.get("labels", [])]
    if labels and labels != ["yes", "no"]:
        raise ValueError(f"VLM cache is not in Yes/No order: {row.get('labels')}")
    sequence_nll = row.get("style_sequence_nll")
    if sequence_nll and sequence_nll[0] is not None:
        return int(np.argmin(np.asarray(sequence_nll[0], dtype=float)))
    logits = row.get("style_logits")
    if not logits:
        raise ValueError("VLM cache row has neither original NLL nor logits")
    return int(np.argmax(np.asarray(logits[0], dtype=float)))


def classification_metrics(
    details: list[dict[str, Any]], prediction_key: str
) -> dict[str, float | int | None]:
    """Binary metrics with index 0=Yes/positive and index 1=No/negative."""

    tp = sum(row["gt_index"] == 0 and row[prediction_key] == 0 for row in details)
    fn = sum(row["gt_index"] == 0 and row[prediction_key] == 1 for row in details)
    tn = sum(row["gt_index"] == 1 and row[prediction_key] == 1 for row in details)
    fp = sum(row["gt_index"] == 1 and row[prediction_key] == 0 for row in details)
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    overall = (tp + tn) / len(details) if details else None
    return {
        "n": len(details),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "overall_accuracy": overall,
    }


def summarize_subset(details: list[dict[str, Any]]) -> dict[str, Any]:
    clear_metrics = classification_metrics(details, "clear_prediction")
    shuffled_metrics = classification_metrics(details, "shuffled_clear_prediction")
    vlm_metrics = classification_metrics(details, "vlm_prediction")
    always_no = [dict(row, always_no_prediction=1) for row in details]
    rescues = sum(row["clear_rescues_vlm"] for row in details)
    harms = sum(row["clear_harms_vlm"] for row in details)
    return {
        "clear": clear_metrics,
        "cyclic_image_shuffle": shuffled_metrics,
        "vlm_original_full_string_nll": vlm_metrics,
        "always_no": classification_metrics(always_no, "always_no_prediction"),
        "clear_rescues_vlm": rescues,
        "clear_harms_vlm": harms,
        "rescue_minus_harm": rescues - harms,
        "clear_shuffle_prediction_agreement": (
            float(
                np.mean(
                    [
                        row["clear_prediction"] == row["shuffled_clear_prediction"]
                        for row in details
                    ]
                )
            )
            if details
            else None
        ),
    }


def summarize_full_fallback(
    rows: list[dict[str, Any]],
    clear_details: list[dict[str, Any]],
    cache_rows: dict[str, dict[str, Any]],
    *,
    allowed_tiers: set[str],
) -> dict[str, Any]:
    """Evaluate CLEAR where mapped and preserve the VLM answer otherwise."""

    clear_by_qid = {
        str(row["qid"]): row
        for row in clear_details
        if row["mapping"]["tier"] in allowed_tiers
    }
    fallback_rows = []
    for row in rows:
        qid = str(row["qid"])
        if qid not in cache_rows:
            raise RuntimeError(f"VLM cache lacks fallback qid: {qid}")
        gt_index = ground_truth_index(row)
        vlm_prediction = prediction_from_vlm_cache(cache_rows[qid])
        clear_row = clear_by_qid.get(qid)
        fallback_prediction = (
            int(clear_row["clear_prediction"])
            if clear_row is not None
            else vlm_prediction
        )
        fallback_rows.append(
            {
                "gt_index": gt_index,
                "vlm_prediction": vlm_prediction,
                "fallback_prediction": fallback_prediction,
                "used_clear": clear_row is not None,
            }
        )
    rescues = sum(
        row["vlm_prediction"] != row["gt_index"]
        and row["fallback_prediction"] == row["gt_index"]
        for row in fallback_rows
    )
    harms = sum(
        row["vlm_prediction"] == row["gt_index"]
        and row["fallback_prediction"] != row["gt_index"]
        for row in fallback_rows
    )
    return {
        "clear_or_vlm_fallback": classification_metrics(
            fallback_rows, "fallback_prediction"
        ),
        "vlm_original_full_string_nll": classification_metrics(
            fallback_rows, "vlm_prediction"
        ),
        "used_clear_n": sum(row["used_clear"] for row in fallback_rows),
        "used_vlm_fallback_n": sum(not row["used_clear"] for row in fallback_rows),
        "rescues_vlm": rescues,
        "harms_vlm": harms,
        "rescue_minus_harm": rescues - harms,
    }


def _repo_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _load_clear(checkpoint: Path, clear_repo: Path, device: torch.device):
    source = str((clear_repo / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from clear import load_pretrained, tokenize

    model, preprocess = load_pretrained(
        checkpoint_path=checkpoint,
        device=device,
        local_files_only=True,
        strict=True,
    )
    return model, preprocess, tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--vlm-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clear-repo", type=Path, default=DEFAULT_CLEAR_REPO)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    dataset_sha = file_sha256(args.dataset)
    if dataset_sha != FIXED_DATASET_SHA256:
        raise RuntimeError(
            "This exploratory probe is locked to subset_seed42_n32: "
            f"{dataset_sha} != {FIXED_DATASET_SHA256}"
        )
    repo_commit = _repo_commit(args.clear_repo)
    if repo_commit != CLEAR_REPO_COMMIT:
        raise RuntimeError(
            f"CLEAR checkout mismatch: {repo_commit} != {CLEAR_REPO_COMMIT}"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    os.environ["TORCH_HOME"] = str(args.torch_home.resolve())
    args.torch_home.mkdir(parents=True, exist_ok=True)

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 32:
        raise RuntimeError("Locked dataset must contain exactly 32 rows")
    mapped: list[tuple[dict[str, Any], PromptPair]] = []
    unmapped: list[dict[str, Any]] = []
    for row in rows:
        pair = map_question(str(row.get("question", "")), include_extended=True)
        if pair is None:
            unmapped.append(
                {"qid": row.get("qid"), "question": row.get("question", "")}
            )
        else:
            mapped.append((row, pair))
    strict_count = sum(pair.tier == "strict" for _, pair in mapped)
    if strict_count != 26 or len(mapped) != 29:
        raise RuntimeError(
            f"Pre-registered mapping mismatch: strict={strict_count}, all={len(mapped)}"
        )

    cache_rows = {
        str(row["qid"]): row
        for row in (
            json.loads(line)
            for line in args.vlm_cache.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("status") == "ok"
    }
    missing_cache = [row["qid"] for row, _ in mapped if str(row["qid"]) not in cache_rows]
    if missing_cache:
        raise RuntimeError(f"VLM cache lacks mapped qids: {missing_cache}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model, preprocess, tokenize = _load_clear(args.checkpoint, args.clear_repo, device)

    # Each mapped image is opened and deterministically preprocessed exactly once.
    image_tensors: list[torch.Tensor] = []
    for row, _ in mapped:
        image_path = resolve_image(str(row.get("img_name", "")))
        if image_path is None:
            raise FileNotFoundError(f"Cannot resolve image for qid={row.get('qid')}")
        with Image.open(image_path) as image:
            image_tensors.append(preprocess(image.convert("RGB")))

    image_feature_chunks = []
    with torch.inference_mode():
        for start in range(0, len(image_tensors), args.batch_size):
            batch = torch.stack(image_tensors[start : start + args.batch_size]).to(
                device
            )
            image_feature_chunks.append(
                F.normalize(model.encode_image(batch), dim=-1).cpu()
            )
        texts = [
            prompt
            for _, pair in mapped
            for prompt in (pair.positive, pair.negative)
        ]
        tokens = tokenize(texts).to(device)
        text_features = F.normalize(model.encode_text(tokens), dim=-1).cpu()

    image_features = torch.cat(image_feature_chunks)
    text_features = text_features.reshape(len(mapped), 2, -1)
    paired_scores = torch.einsum("nd,nkd->nk", image_features, text_features)
    shuffled_scores = torch.einsum(
        "nd,nkd->nk", image_features.roll(shifts=1, dims=0), text_features
    )
    paired_predictions = paired_scores.argmax(dim=-1).numpy()
    shuffled_predictions = shuffled_scores.argmax(dim=-1).numpy()

    details: list[dict[str, Any]] = []
    for index, (row, pair) in enumerate(mapped):
        gt_index = ground_truth_index(row)
        vlm_prediction = prediction_from_vlm_cache(cache_rows[str(row["qid"])])
        clear_prediction = int(paired_predictions[index])
        details.append(
            {
                "qid": row["qid"],
                "img_name": row.get("img_name", ""),
                "question": row["question"],
                "mapping": asdict(pair),
                "gt_index": gt_index,
                "vlm_prediction": vlm_prediction,
                "clear_prediction": clear_prediction,
                "shuffled_clear_prediction": int(shuffled_predictions[index]),
                "clear_pair_cosine": paired_scores[index].tolist(),
                "clear_cosine_margin_pos_minus_neg": float(
                    paired_scores[index, 0] - paired_scores[index, 1]
                ),
                "shuffled_pair_cosine": shuffled_scores[index].tolist(),
                "clear_rescues_vlm": vlm_prediction != gt_index
                and clear_prediction == gt_index,
                "clear_harms_vlm": vlm_prediction == gt_index
                and clear_prediction != gt_index,
            }
        )

    strict_details = [row for row in details if row["mapping"]["tier"] == "strict"]
    strict = summarize_subset(strict_details)
    extended = summarize_subset(details)
    fallback = {
        "strict_mapping": summarize_full_fallback(
            rows, details, cache_rows, allowed_tiers={"strict"}
        ),
        "strict_plus_extended_mapping": summarize_full_fallback(
            rows, details, cache_rows, allowed_tiers={"strict", "extended"}
        ),
    }
    strict_clear_balanced = strict["clear"]["balanced_accuracy"]
    strict_shuffle_balanced = strict["cyclic_image_shuffle"]["balanced_accuracy"]
    gate = {
        "strict_balanced_accuracy_gt_0_60": (
            strict_clear_balanced is not None and strict_clear_balanced > 0.60
        ),
        "strict_rescue_minus_harm_at_least_2": strict["rescue_minus_harm"] >= 2,
        "strict_shuffle_degradation": (
            strict_clear_balanced is not None
            and strict_shuffle_balanced is not None
            and strict_clear_balanced > strict_shuffle_balanced
        ),
    }
    gate["all_pass"] = all(gate.values())

    config = {
        "version": VERSION,
        "mapping_version": MAPPING_VERSION,
        "mapping_sha256": mapping_sha256(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_sha,
        "vlm_cache": str(args.vlm_cache.resolve()),
        "vlm_cache_sha256": file_sha256(args.vlm_cache),
        "clear_repo": str(args.clear_repo.resolve()),
        "clear_repo_commit": repo_commit,
        "clear_checkpoint": str(args.checkpoint.resolve()),
        "clear_checkpoint_sha256": file_sha256(args.checkpoint),
        "torch_home": str(args.torch_home.resolve()),
        "device": str(device),
        "batch_size": args.batch_size,
        "image_preprocessing": (
            "explicit PIL RGB conversion followed once by the official CLEAR "
            "build_cxr_preprocess transform"
        ),
        "pair_scoring": "cosine(argmax([question-positive, question-negative]))",
        "image_shuffle_control": "cyclic previous-image feature in locked dataset order",
        "mapping_uses_ground_truth": False,
    }
    result = {
        "version": VERSION,
        "fingerprint": protocol_fingerprint(config),
        "config": config,
        "mapping_audit": {
            "strict_mapped": len(strict_details),
            "extended_mapped": len(details) - len(strict_details),
            "unmapped": unmapped,
        },
        "strict_mapped_subset": strict,
        "extended_mapped_subset_exploratory": extended,
        "full_n32_vlm_fallback_exploratory": fallback,
        "gate": gate,
        "exploratory_only": True,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "details"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
