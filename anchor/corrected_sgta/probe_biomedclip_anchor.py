"""Diagnose whether a frozen medical CLIP supplies a useful answer direction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from corrected_sgta.protocol_v2 import (
    file_sha256,
    ground_truth_index,
    resolve_image,
)
from corrected_sgta.source_bank_v2 import sha256_file


BIOMEDCLIP_ROOT = Path("/root/autodl-tmp/BiomedCLIP")
VERSION = "biomedclip-semantic-anchor-probe-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--vlm-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model():
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.factory import _MODEL_CONFIGS

    config = json.loads((BIOMEDCLIP_ROOT / "open_clip_config.json").read_text())
    _MODEL_CONFIGS["biomedclip_local"] = config["model_cfg"]
    model, _, preprocess = create_model_and_transforms(
        "biomedclip_local",
        pretrained=str(BIOMEDCLIP_ROOT / "open_clip_pytorch_model.bin"),
        **{f"image_{key}": value for key, value in config["preprocess_cfg"].items()},
    )
    return model, preprocess, get_tokenizer("biomedclip_local")


def exact_prediction_from_cache(row: dict) -> int:
    sequence_nll = row.get("style_sequence_nll")
    if sequence_nll and sequence_nll[0] is not None:
        return int(np.argmin(sequence_nll[0]))
    return int(np.argmax(row["style_logits"][0]))


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    rows.sort(
        key=lambda row: hashlib.sha256(f"{args.seed}:{row['qid']}".encode()).hexdigest()
    )
    rows = rows[: args.max_samples]
    cache = {
        str(row["qid"]): row
        for row in (
            json.loads(line)
            for line in args.vlm_cache.read_text().splitlines()
            if line.strip()
        )
        if row.get("status") == "ok"
    }
    rows = [row for row in rows if str(row["qid"]) in cache]
    model, preprocess, tokenizer = load_model()
    device = torch.device("cuda")
    model = model.to(device).eval()
    image_tensors = []
    texts = []
    for row in rows:
        path = resolve_image(row.get("img_name", ""))
        with Image.open(path) as image:
            image_tensors.append(preprocess(image.convert("RGB")))
        question = str(row["question"]).strip()
        texts.extend(
            [
                f"Chest radiograph question: {question} Answer: yes.",
                f"Chest radiograph question: {question} Answer: no.",
            ]
        )
    with torch.inference_mode():
        image_features = model.encode_image(
            torch.stack(image_tensors).to(device), normalize=True
        )
        text_features = model.encode_text(
            tokenizer(texts, context_length=256).to(device), normalize=True
        )
    text_features = text_features.reshape(len(rows), 2, -1)
    paired_scores = torch.einsum("nd,nkd->nk", image_features, text_features)
    shifted_scores = torch.einsum(
        "nd,nkd->nk", image_features.roll(shifts=1, dims=0), text_features
    )
    paired_predictions = paired_scores.argmax(-1).cpu().numpy()
    shifted_predictions = shifted_scores.argmax(-1).cpu().numpy()
    details = []
    for index, row in enumerate(rows):
        gt = ground_truth_index(row)
        vlm_prediction = exact_prediction_from_cache(cache[str(row["qid"])])
        anchor_prediction = int(paired_predictions[index])
        details.append(
            {
                "qid": row["qid"],
                "gt_index": gt,
                "vlm_prediction": vlm_prediction,
                "anchor_prediction": anchor_prediction,
                "shifted_image_anchor_prediction": int(shifted_predictions[index]),
                "anchor_scores": paired_scores[index].float().cpu().tolist(),
                "shifted_image_anchor_scores": shifted_scores[index]
                .float()
                .cpu()
                .tolist(),
                "vlm_correct": vlm_prediction == gt,
                "anchor_correct": anchor_prediction == gt,
                "anchor_rescues_vlm": vlm_prediction != gt
                and anchor_prediction == gt,
                "anchor_harms_vlm": vlm_prediction == gt
                and anchor_prediction != gt,
            }
        )
    def accuracy(key: str) -> float:
        return float(np.mean([row[key] == row["gt_index"] for row in details]))
    result = {
        "version": VERSION,
        "config": {
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": file_sha256(args.dataset),
            "vlm_cache": str(args.vlm_cache.resolve()),
            "vlm_cache_sha256": sha256_file(args.vlm_cache),
            "max_samples": args.max_samples,
            "seed": args.seed,
            "text_interface": (
                "symmetric fixed QA strings: 'Chest radiograph question: {q} "
                "Answer: yes/no.'"
            ),
            "image_shuffle_control": "cyclic shift by one after fixed qid ordering",
            "biomedclip_config_sha256": sha256_file(
                BIOMEDCLIP_ROOT / "open_clip_config.json"
            ),
            "biomedclip_weights_sha256": sha256_file(
                BIOMEDCLIP_ROOT / "open_clip_pytorch_model.bin"
            ),
        },
        "n": len(details),
        "vlm_accuracy": accuracy("vlm_prediction"),
        "anchor_accuracy": accuracy("anchor_prediction"),
        "shifted_image_anchor_accuracy": accuracy(
            "shifted_image_anchor_prediction"
        ),
        "anchor_rescues_vlm": sum(row["anchor_rescues_vlm"] for row in details),
        "anchor_harms_vlm": sum(row["anchor_harms_vlm"] for row in details),
        "anchor_changed_vs_vlm": sum(
            row["anchor_prediction"] != row["vlm_prediction"] for row in details
        ),
        "label_counts": {
            "yes": sum(row["gt_index"] == 0 for row in details),
            "no": sum(row["gt_index"] == 1 for row in details),
        },
        "anchor_prediction_counts": {
            "yes": sum(row["anchor_prediction"] == 0 for row in details),
            "no": sum(row["anchor_prediction"] == 1 for row in details),
        },
        "shifted_prediction_counts": {
            "yes": sum(
                row["shifted_image_anchor_prediction"] == 0 for row in details
            ),
            "no": sum(
                row["shifted_image_anchor_prediction"] == 1 for row in details
            ),
        },
        "always_no_accuracy": float(
            np.mean([row["gt_index"] == 1 for row in details])
        ),
        "anchor_shift_prediction_agreement": float(
            np.mean(
                [
                    row["anchor_prediction"]
                    == row["shifted_image_anchor_prediction"]
                    for row in details
                ]
            )
        ),
        "details": details,
    }
    result["gate"] = {
        "at_least_two_potential_rescues": result["anchor_rescues_vlm"] >= 2,
        "rescues_exceed_harms": (
            result["anchor_rescues_vlm"] > result["anchor_harms_vlm"]
        ),
        "image_dependence_positive": (
            result["anchor_accuracy"]
            > result["shifted_image_anchor_accuracy"]
        ),
        "beats_always_no": (
            result["anchor_accuracy"] > result["always_no_accuracy"]
        ),
        "shuffle_changes_at_least_25pct": (
            result["anchor_shift_prediction_agreement"] < 0.75
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
