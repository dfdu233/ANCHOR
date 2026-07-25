#!/usr/bin/env python3
"""Resumable RULE inference for a source-preference checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.evaluate_rule_source_preference_barycenter import _predict
from corrected_sgta.infer_rule_dg_adapter import atomic_json, load_rows, official_prompt
from corrected_sgta.rule_dg_adapter_fingerprint_v3 import (
    FINGERPRINT_VERSION,
    image_manifest_identity,
    tree_identity,
)
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_source_preference import (
    VERSION as TRAIN_VERSION,
    LinearLowRankResidual,
    SourceBarycenterResidual,
    file_sha256,
    stable_json_sha256,
)

VERSION = "rule-source-preference-barycenter-inference-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/root/autodl-tmp/LLaVA-Med/microsoft/llava-med-v1.5-mistral-7b"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("pooled", "barycenter"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("rule_source_preference.py"),
        Path(__file__).with_name("evaluate_rule_source_preference_barycenter.py"),
        Path(__file__).with_name("train_rule_dg_adapter.py"),
        Path(__file__).with_name("infer_rule_dg_adapter.py"),
        Path(__file__).with_name("models_alignment.py"),
        Path(__file__).with_name("rule_dg_adapter_fingerprint_v3.py"),
    ]
    return {str(path): file_sha256(path) for path in paths}


def inference_content_identity(
    rows: list[dict[str, Any]], image_root: Path, base_model: Path
) -> dict[str, Any]:
    """Bind inference to actual image bytes and the complete base-model tree."""
    return {
        "content_identity_schema": FINGERPRINT_VERSION,
        "base_model": tree_identity(base_model),
        "images": image_manifest_identity(rows, image_root),
    }


def load_resume_records(path: Path, fingerprint: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    qids = [str(record["question_id"]) for record in records]
    if len(qids) != len(set(qids)):
        raise ValueError("resume output contains duplicate question ids")
    if any(record.get("fingerprint") != fingerprint for record in records):
        raise RuntimeError("resume record fingerprint mismatch")
    return records


def rewrite_successes(path: Path, records: list[dict[str, Any]]) -> set[str]:
    """Remove failed rows before retry so one qid can never appear twice."""
    successes = [record for record in records if record.get("status") == "ok"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        for record in successes:
            handle.write(json.dumps(record) + "\n")
    temporary.replace(path)
    return {str(record["question_id"]) for record in successes}


def _load_module(
    checkpoint: dict[str, Any], variant: str, device: torch.device
) -> torch.nn.Module:
    maximum = float(checkpoint["max_relative_update"])
    if variant == "barycenter":
        states = checkpoint.get("per_source_state_dict")
        if not isinstance(states, dict) or len(states) < 2:
            raise ValueError("checkpoint lacks >=2 source-specific modules")
        module = SourceBarycenterResidual(states, maximum)
    else:
        state = checkpoint.get("pooled_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint lacks pooled module")
        module = LinearLowRankResidual(
            int(checkpoint["width"]), int(checkpoint["rank"]), maximum
        )
        module.load_state_dict(state)
    module.to(device)
    module.eval()
    return module


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    for path in (args.questions, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.image_root.is_dir() or not args.base_model.is_dir():
        raise FileNotFoundError(args.image_root)
    rows = load_rows(args.questions)
    rows = rows[: args.max_samples] if args.max_samples is not None else rows
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != TRAIN_VERSION:
        raise ValueError("unsupported preference checkpoint")
    if checkpoint.get("prompt_protocol") != "rule_mimic":
        raise ValueError("checkpoint prompt protocol is not rule_mimic")

    fingerprint_payload = {
        "version": VERSION,
        "questions_sha256": file_sha256(args.questions),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        **inference_content_identity(rows, args.image_root, args.base_model),
        "checkpoint_fingerprint": checkpoint.get("fingerprint"),
        "variant": args.variant,
        "selected": [
            {
                "question_id": str(row.get("question_id", row.get("qid"))),
                "image": str(row["image"]),
            }
            for row in rows
        ],
        "prompt_protocol": "rule_mimic",
        "prediction_interface": "argmax_complete_yes_no_sequence_log_probability",
        "single_view": True,
        "target_labels_used_for_prediction": False,
        "code_sha256": code_hashes(),
    }
    fingerprint = stable_json_sha256(fingerprint_payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta = {
        **fingerprint_payload,
        "fingerprint": fingerprint,
        "questions": str(args.questions.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "n_requested": len(rows),
    }
    if args.resume:
        if meta_path.is_file():
            if json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
                raise RuntimeError("existing inference fingerprint mismatch")
        elif args.output.exists() and args.output.stat().st_size:
            raise RuntimeError("resume output lacks metadata")
    elif args.output.exists() and args.output.stat().st_size:
        raise FileExistsError("output exists; use --resume for an identical run")
    existing = load_resume_records(args.output, fingerprint) if args.resume else []
    complete = rewrite_successes(args.output, existing)
    atomic_json(meta_path, meta)

    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    module = _load_module(checkpoint, args.variant, adapter.model.device)
    try:
        with args.output.open("a") as handle, torch.no_grad():
            for row in tqdm(rows, desc=f"RULE preference:{args.variant}"):
                qid = str(row.get("question_id", row.get("qid")))
                if qid in complete:
                    continue
                record: dict[str, Any] = {
                    "question_id": row.get("question_id", row.get("qid")),
                    "image": row["image"],
                    "gt_answer": row.get("answer"),
                    "fingerprint": fingerprint,
                }
                try:
                    with Image.open(args.image_root / row["image"]) as source:
                        image = source.convert("RGB")
                    prompt = official_prompt(row, "rule_mimic")
                    identity, identity_scores = _predict(adapter, image, prompt, None)
                    adapted, adapted_scores = _predict(adapter, image, prompt, module)
                    record.update(
                        {
                            "status": "ok",
                            "prompt": prompt,
                            "base_text": identity,
                            "adapted_text": adapted,
                            "text": adapted,
                            "answer": adapted,
                            "model_id": f"llava-med-source-preference-{args.variant}",
                            "metadata": {
                                "checkpoint_version": checkpoint["version"],
                                "checkpoint_fingerprint": checkpoint.get("fingerprint"),
                                "variant": args.variant,
                                "adapter_location": "post",
                                "single_view": True,
                                "identity_sequence_log_probabilities": identity_scores,
                                "adapted_sequence_log_probabilities": adapted_scores,
                            },
                        }
                    )
                except Exception as error:
                    record.update(
                        {
                            "status": "error",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                handle.write(json.dumps(record) + "\n")
                handle.flush()
    finally:
        adapter.close()

    final_records = load_resume_records(args.output, fingerprint)
    meta["n_complete"] = sum(
        record.get("status") == "ok" for record in final_records
    )
    meta["n_errors"] = sum(
        record.get("status") != "ok" for record in final_records
    )
    atomic_json(meta_path, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
