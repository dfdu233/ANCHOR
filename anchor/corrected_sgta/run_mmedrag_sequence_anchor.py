#!/usr/bin/env python3
"""Generate and sequence-energy-rerank unrestricted source-guided reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.cache import repair_truncated_jsonl_tail
from corrected_sgta.models_oe import LlavaMedOEAdapter
from corrected_sgta.rule_source_preference import file_sha256, stable_json_sha256
from corrected_sgta.train_rule_dg_adapter import (
    build_teacher_forcing,
    sequence_forward,
)

VERSION = "mmedrag-sequence-anchor-v1"


def prompts(dataset: str, reference: str | None = None) -> str:
    role = "ophthalmologist" if dataset == "harvard" else "radiologist"
    image_name = "fundus image" if dataset == "harvard" else "X-ray image"
    if reference is None:
        return (
            f"You are a professional {role}. You are provided with a {image_name}. "
            "Please generate a report based on the image. "
            "Please only include the content of the report in your response."
        )
    return (
        f"You are a professional {role}. You are provided with a {image_name} "
        f"and 1 reference report(s): 1. {reference} "
        "Please generate a report based on the image. It should be noted that "
        "the diagnostic information in the reference reports cannot be directly "
        "used as the basis for diagnosis, but should only be used for reference "
        "and comparison. Please only include the content of the report in your response."
    )


@torch.inference_mode()
def mean_sequence_nll(
    adapter: LlavaMedOEAdapter,
    image: Image.Image,
    prompt: str,
    answer: str,
) -> float:
    input_ids, labels = build_teacher_forcing(adapter, prompt, answer)
    loss, _ = sequence_forward(adapter, image, input_ids, labels, None, "post")
    return float(loss.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"]
    if args.max_samples:
        records = records[: args.max_samples]
    fingerprint_payload = {
        "version": VERSION,
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_fingerprint": manifest["fingerprint"],
        "max_new_tokens": args.max_new_tokens,
        "max_samples": args.max_samples,
        "model": "microsoft/llava-med-v1.5-mistral-7b",
        "conversation": "vicuna_v1",
        "decoding": "greedy",
        "candidate_budget": 2,
        "selection": "minimum mean complete-sequence NLL under the no-reference prompt",
        "code_sha256": file_sha256(Path(__file__)),
    }
    fingerprint = stable_json_sha256(fingerprint_payload)
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    if args.raw.exists():
        repair_truncated_jsonl_tail(args.raw)
    done = {}
    if args.raw.exists():
        for line in args.raw.read_text().splitlines():
            row = json.loads(line)
            if row["fingerprint"] != fingerprint:
                raise ValueError("raw-cache fingerprint mismatch")
            done[(row["dataset"], row["id"])] = row

    adapter = LlavaMedOEAdapter(conv_mode="vicuna_v1")
    try:
        with args.raw.open("a") as output:
            for row in tqdm(records, desc="sequence-anchor", initial=len(done), total=len(records)):
                key = (row["dataset"], row["id"])
                if key in done:
                    continue
                with Image.open(row["image"]) as handle:
                    image = handle.convert("RGB")
                base_prompt = prompts(row["dataset"])
                source_report = row["neighbors"][0]["report"]
                guided_prompt = prompts(row["dataset"], source_report)
                baseline = adapter._generate_once(
                    image, base_prompt, 1, False, 1.0, 1.0, args.max_new_tokens, 0
                )[0].text
                guided = adapter._generate_once(
                    image, guided_prompt, 1, False, 1.0, 1.0, args.max_new_tokens, 0
                )[0].text
                if not baseline or not guided:
                    raise RuntimeError(f"empty report for {key}")
                base_nll = mean_sequence_nll(adapter, image, base_prompt, baseline)
                guided_nll = mean_sequence_nll(adapter, image, base_prompt, guided)
                selected = "guided" if guided_nll < base_nll else "baseline"
                result = {
                    "version": VERSION,
                    "fingerprint": fingerprint,
                    "dataset": row["dataset"],
                    "id": row["id"],
                    "image": row["image"],
                    "ground_truth": row["reference"],
                    "source_neighbor": row["neighbors"][0],
                    "candidates": {
                        "baseline": baseline,
                        "guided": guided,
                    },
                    "mean_sequence_nll": {
                        "baseline": base_nll,
                        "guided": guided_nll,
                    },
                    "selected": selected,
                    "sequence_anchor": guided if selected == "guided" else baseline,
                    "ground_truth_used_for_generation_or_selection": False,
                }
                output.write(json.dumps(result) + "\n")
                output.flush()
                done[key] = result
    finally:
        adapter.close()

    ordered = [done[(row["dataset"], row["id"])] for row in records]
    payload = {
        "version": VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "status": "final",
        "n": len(ordered),
        "ground_truth_used_for_generation_or_selection": False,
        "records": ordered,
    }
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "n": len(ordered),
                "selected": {
                    name: sum(row["selected"] == name for row in ordered)
                    for name in ("baseline", "guided")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
