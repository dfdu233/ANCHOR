"""Measure whether CXR style views attenuate evidence or rotate clinical priors.

The probe uses complete positive/negative clinical sentences only as a
teacher-forced diagnostic.  It never uses their likelihoods as predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from anchor.corrected_sgta.anchor_models import NULL_RGB
from anchor.corrected_sgta.build_pubmed_style_prototypes import (
    smooth_low_frequency_mask,
)
from anchor.corrected_sgta.run_center_native_qwen import messages_for, pad_392


VERSION = "visual-evidence-chord-probe-v1"
CONDITIONS = {
    "pneumothorax": "pneumothorax",
    "effusion": "pleural effusion",
    "opacity": "pulmonary opacity or consolidation",
    "cardiomegaly": "cardiomegaly",
    "edema": "pulmonary edema",
    "device": "an indwelling medical device",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def unique_cases(
    questions: Path,
    image_manifest: Path,
    limit: int,
) -> list[dict]:
    images = {
        row["relative_path"]: row["path_in_repo"]
        for row in read_jsonl(image_manifest)
    }
    cases: list[dict] = []
    seen: set[str] = set()
    for row in read_jsonl(questions):
        relative = row["image"]
        if relative in seen or relative not in images:
            continue
        seen.add(relative)
        cases.append(
            {
                "case_id": f"mimic-{len(cases):03d}",
                "image_relative": relative,
                "image": images[relative],
                "source_question_id": str(row["question_id"]),
            }
        )
        if len(cases) >= limit:
            break
    if len(cases) < limit:
        raise RuntimeError(f"only found {len(cases)} unique MIMIC images")
    return cases


def grayscale(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def style_transfer(
    image: Image.Image,
    target_log_amplitude: np.ndarray,
    mask: np.ndarray,
    strength: float,
) -> Image.Image:
    """Transfer smooth low-frequency log amplitude while preserving phase/DC."""

    source = grayscale(image)
    spectrum = np.fft.fft2(source)
    source_log = np.log(np.abs(spectrum) + 1e-6)
    mixed = source_log + float(strength) * mask * (
        target_log_amplitude - source_log
    )
    mixed[0, 0] = source_log[0, 0]
    reconstructed = np.fft.ifft2(
        np.exp(mixed) * np.exp(1j * np.angle(spectrum))
    ).real
    output = np.uint8(np.clip(reconstructed, 0.0, 1.0) * 255)
    return Image.fromarray(output, mode="L").convert("RGB")


def image_metrics(reference: Image.Image, candidate: Image.Image) -> dict:
    first = grayscale(reference)
    second = grayscale(candidate)
    first_gradient = np.hypot(*np.gradient(first))
    second_gradient = np.hypot(*np.gradient(second))

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        if float(left.std()) < 1e-8 or float(right.std()) < 1e-8:
            return 0.0
        return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])

    return {
        "pixel_correlation": correlation(first, second),
        "edge_correlation": correlation(first_gradient, second_gradient),
        "mean_absolute_change": float(np.abs(first - second).mean()),
    }


def condition_prompt(condition: str) -> str:
    return (
        "Independently determine from this chest radiograph whether "
        f"{condition} is present or absent. Do not assume it is present merely "
        "because it is named. State the conclusion in one complete sentence."
    )


def candidate_sentences(condition: str) -> dict[str, str]:
    if condition.startswith("an "):
        return {
            "positive": f"The chest radiograph shows {condition}.",
            "negative": f"The chest radiograph does not show {condition}.",
        }
    return {
        "positive": f"The chest radiograph shows {condition}.",
        "negative": f"The chest radiograph does not show {condition}.",
    }


def supervised_labels(
    full: dict[str, torch.Tensor],
    prompts: dict[str, torch.Tensor],
) -> torch.Tensor:
    labels = full["input_ids"].clone()
    for index in range(labels.shape[0]):
        full_nonpad = int(full["attention_mask"][index].sum())
        prompt_nonpad = int(prompts["attention_mask"][index].sum())
        start = labels.shape[1] - full_nonpad
        labels[index, : start + prompt_nonpad] = -100
    labels[full["attention_mask"] == 0] = -100
    return labels


@torch.inference_mode()
def sequence_nll(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    batch: list[dict],
) -> list[tuple[float, int]]:
    full_texts = [
        processor.apply_chat_template(
            messages_for(row["question"], row["answer"]),
            tokenize=False,
            add_generation_prompt=False,
        )
        for row in batch
    ]
    prompt_texts = [
        processor.apply_chat_template(
            messages_for(row["question"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in batch
    ]
    images = [row["image"] for row in batch]
    full = processor(
        text=full_texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    prompts = processor(
        text=prompt_texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    labels = supervised_labels(full, prompts)
    inputs = {key: value.to("cuda") for key, value in full.items()}
    labels = labels.to("cuda")
    output = model(**inputs)
    shift_logits = output.logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shift_labels.shape)
    active = shift_labels.ne(-100)
    counts = active.sum(dim=1)
    means = (losses * active).sum(dim=1) / counts.clamp_min(1)
    return [
        (float(loss), int(count))
        for loss, count in zip(means.cpu(), counts.cpu(), strict=True)
    ]


def build_views(
    case: dict,
    prototypes: list[dict],
    radius: float,
    strength: float,
    output: Path,
) -> tuple[dict[str, Image.Image], dict[str, dict]]:
    original = pad_392(case["image"])
    mask = smooth_low_frequency_mask(392, radius)
    views = {
        "real": original,
        "null": Image.new("RGB", original.size, NULL_RGB),
    }
    metrics = {
        "real": {
            "pixel_correlation": 1.0,
            "edge_correlation": 1.0,
            "mean_absolute_change": 0.0,
        },
        "null": image_metrics(original, views["null"]),
    }
    for prototype in prototypes:
        style_id = f"style_{int(prototype['cluster'])}"
        target = pad_392(prototype["image"])
        target_log = np.log(np.abs(np.fft.fft2(grayscale(target))) + 1e-6)
        styled = style_transfer(original, target_log, mask, strength)
        views[style_id] = styled
        metrics[style_id] = image_metrics(original, styled)
    if case["case_id"] == "mimic-000":
        view_dir = output.parent / "views"
        view_dir.mkdir(parents=True, exist_ok=True)
        for name, image in views.items():
            image.save(view_dir / f"{case['case_id']}_{name}.png")
    return views, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--style-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--radius", type=float, default=0.12)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = unique_cases(args.questions, args.image_manifest, args.limit)
    prototypes = [
        row
        for row in read_jsonl(args.style_manifest)
        if int(row["replicate"]) == 0
    ]
    if len(prototypes) < 3:
        raise RuntimeError("style manifest must contain at least three clusters")

    config_sha = sha256(args.model / "config.json")
    pending: list[dict] = []
    metadata: dict[tuple[str, str], dict] = {}
    for case in cases:
        views, view_metrics = build_views(
            case,
            prototypes,
            args.radius,
            args.strength,
            args.output,
        )
        for view_name, image in views.items():
            metadata[(case["case_id"], view_name)] = view_metrics[view_name]
            for disease, condition in CONDITIONS.items():
                question = condition_prompt(condition)
                for polarity, answer in candidate_sentences(condition).items():
                    pending.append(
                        {
                            **case,
                            "view": view_name,
                            "image": image,
                            "disease": disease,
                            "polarity": polarity,
                            "question": question,
                            "answer": answer,
                        }
                    )

    existing_keys: set[tuple[str, str, str, str]] = set()
    if args.resume and args.output.exists():
        existing = read_jsonl(args.output)
        expected = {
            "model_config_sha256": config_sha,
            "questions_sha256": sha256(args.questions),
            "image_manifest_sha256": sha256(args.image_manifest),
            "style_manifest_sha256": sha256(args.style_manifest),
            "radius": args.radius,
            "strength": args.strength,
        }
        for record in existing:
            for field, value in expected.items():
                if record[field] != value:
                    raise RuntimeError(
                        f"resume fingerprint mismatch for {field}: "
                        f"{record[field]!r} != {value!r}"
                    )
            existing_keys.add(
                (
                    record["case_id"],
                    record["view"],
                    record["disease"],
                    record["polarity"],
                )
            )
        pending = [
            row
            for row in pending
            if (
                row["case_id"],
                row["view"],
                row["disease"],
                row["polarity"],
            )
            not in existing_keys
        ]
    if not pending:
        print(
            json.dumps(
                {
                    "completed": len(existing_keys),
                    "remaining": 0,
                    "output": str(args.output),
                }
            )
        )
        return

    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, use_fast=False
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).to("cuda").eval()
    mode = "a" if args.resume and existing_keys else "w"
    with args.output.open(mode) as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            scores = sequence_nll(model, processor, batch)
            for row, (nll, token_count) in zip(batch, scores, strict=True):
                record = {
                    "version": VERSION,
                    "case_id": row["case_id"],
                    "image_relative": row["image_relative"],
                    "source_question_id": row["source_question_id"],
                    "view": row["view"],
                    "disease": row["disease"],
                    "polarity": row["polarity"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "sequence_nll": nll,
                    "answer_token_count": token_count,
                    "image_metrics": metadata[
                        (row["case_id"], row["view"])
                    ],
                    "model": str(args.model.resolve()),
                    "model_config_sha256": config_sha,
                    "questions_sha256": sha256(args.questions),
                    "image_manifest_sha256": sha256(args.image_manifest),
                    "style_manifest_sha256": sha256(args.style_manifest),
                    "radius": args.radius,
                    "strength": args.strength,
                }
                handle.write(json.dumps(record) + "\n")
            print(
                json.dumps(
                    {
                        "completed": min(start + len(batch), len(pending)),
                        "remaining_total": len(pending),
                        "previously_completed": len(existing_keys),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
