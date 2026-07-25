"""Create a same-image, multi-source FedDG qualitative RULE example."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from corrected_sgta.evaluate_medheval_answers import (
    evaluate_rows,
    rule_pope_prediction,
)
from corrected_sgta.frequency_alignment_source_spectrum_release2 import (
    source_spectrum_alignment_release2,
)
from corrected_sgta.models_surface import LlavaMedSurfaceAdapter


VERSION = "rule-feddg-showcase-v1"
DEFAULT_RULE_TEST = Path("/root/autodl-tmp/RULE/data/test/iuxray_test.jsonl")
DEFAULT_IMAGE_ROOT = Path("/root/autodl-tmp/MedHEval/images/IU-Xray")
DEFAULT_OUTPUT = Path(
    "corrected_runs/rule_protocol_v1/iuxray/feddg_multisource_showcase_v1"
)
DEFAULT_CENTERS = {
    "iuxray_train": Path(
        "corrected_runs/rule_protocol_v1/source_bank/"
        "rule_iuxray_train_amplitude.npy"
    ),
    "mimic_cxr": Path(
        "corrected_runs/source_bank_v1/amplitudes/mimic_cxr_leaksafe.npy"
    ),
    "pubmedvision_xray": Path(
        "corrected_runs/source_bank_v2/amplitudes/pubmedvision_xray_formal.npy"
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rule_row(path: Path, qid: int) -> dict[str, Any]:
    matches = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip() and int(json.loads(line)["question_id"]) == qid
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one RULE row for qid={qid}, found {len(matches)}")
    return matches[0]


def first_sentence(text: object) -> str:
    value = " ".join(str(text or "").split())
    if not value:
        return ""
    match = re.match(r"^.*?[.!?](?:\s|$)", value)
    return match.group(0).strip() if match else value


def prompt_for(question: object) -> str:
    value = str(question).replace("<image>", "").strip()
    return (
        f"{value} Please answer the question based on the image and choose from "
        "the following two options: [yes, no]."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", type=int, default=540)
    parser.add_argument("--rule-test", type=Path, default=DEFAULT_RULE_TEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--low-frequency-ratio", type=float, default=0.02)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = load_rule_row(args.rule_test, args.qid)
    image_path = args.image_root / row["image"]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    missing = [str(path) for path in DEFAULT_CENTERS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source centers: {missing}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as handle:
        original = handle.convert("RGB")
    views: list[tuple[str, Image.Image, Path | None]] = [
        ("original", original, None)
    ]
    for source_id, center_path in DEFAULT_CENTERS.items():
        center = np.load(center_path)
        view = source_spectrum_alignment_release2(
            original,
            center,
            low_frequency_ratio=args.low_frequency_ratio,
            source_ratio=0.0,
        )
        views.append((source_id, view.convert("RGB"), center_path))

    image_dir = args.output_root / "images"
    image_dir.mkdir(exist_ok=True)
    for source_id, image, _ in views:
        image.save(image_dir / f"{source_id}.png")

    prompt = prompt_for(row["question"])
    adapter = LlavaMedSurfaceAdapter(conv_mode="vicuna_v1")
    try:
        raw_outputs = adapter.decode_ce(
            [image for _, image, _ in views],
            prompt,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        adapter.close()

    answer_rows = []
    for (source_id, _, center_path), raw_text in zip(views, raw_outputs):
        sentence = first_sentence(raw_text)
        answer_rows.append(
            {
                "question_id": row["question_id"],
                "variant_id": source_id,
                "image": str((image_dir / f"{source_id}.png").resolve()),
                "prompt": prompt,
                "question_type": "binary",
                "gt_answer": row["answer"],
                "text": sentence,
                "raw_text": raw_text,
                "source_center": None
                if center_path is None
                else {
                    "path": str(center_path.resolve()),
                    "sha256": file_sha256(center_path),
                },
            }
        )

    evaluation = evaluate_rows(answer_rows)
    gt_rule_label = rule_pope_prediction(row["answer"])
    for answer, detail in zip(answer_rows, evaluation["details"]):
        rule_prediction = rule_pope_prediction(answer["text"])
        answer["evaluation"] = {
            "primary_protocol": "RULE/LLaVA POPE first-sentence no/not convention",
            "rule_prediction": rule_prediction,
            "rule_correct": rule_prediction == gt_rule_label,
            "secondary_protocol_version": evaluation["protocol_version"],
            "strict_prediction": detail["prediction"],
            "strict_parse_status": detail["parse_status"],
            "strict_correct": detail["correct"],
        }

    correct_values = {
        bool(answer["evaluation"]["rule_correct"]) for answer in answer_rows
    }
    summary = {
        "version": VERSION,
        "qid": row["question_id"],
        "question": str(row["question"]).replace("<image>", "").strip(),
        "ground_truth": row["answer"],
        "report": row.get("report"),
        "model": "llava-med-v1.5-mistral-7b",
        "decoder": {
            "do_sample": False,
            "conv_mode": "vicuna_v1",
            "max_new_tokens": args.max_new_tokens,
            "output_policy": "first generated sentence under the official RULE no-reference prompt",
        },
        "feddg": {
            "low_frequency_ratio": args.low_frequency_ratio,
            "source_ratio": 0.0,
        },
        "evaluation": {
            "primary_protocol": "RULE/LLaVA POPE first-sentence no/not convention",
            "ground_truth_label": gt_rule_label,
            "primary_metrics": evaluation["rule_compatible_binary_diagnostic"],
            "secondary_protocol": evaluation["protocol_version"],
            "secondary_strict_metrics": evaluation["decoded_strict"],
        },
        "contains_both_correct_and_incorrect": correct_values == {False, True},
        "variants": answer_rows,
    }
    answers_path = args.output_root / "answers.jsonl"
    answers_path.write_text(
        "".join(json.dumps(answer) + "\n" for answer in answer_rows)
    )
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not summary["contains_both_correct_and_incorrect"]:
        raise SystemExit(
            "showcase did not produce both correct and incorrect outputs; "
            "select another predeclared RULE flip candidate"
        )


if __name__ == "__main__":
    main()
