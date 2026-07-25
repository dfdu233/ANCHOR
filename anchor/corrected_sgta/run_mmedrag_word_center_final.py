#!/usr/bin/env python3
"""Resumable baseline and unrestricted source-word-center report generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from corrected_sgta.cache import repair_truncated_jsonl_tail
from corrected_sgta.models_oe import LlavaMedOEAdapter
from corrected_sgta.rule_source_preference import file_sha256, stable_json_sha256
from corrected_sgta.run_mmedrag_sequence_anchor import prompts
VERSION = "mmedrag-source-word-center-final-v3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-centers", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"]
    if args.max_samples:
        records = records[: args.max_samples]
    center_payload = json.loads(args.source_centers.read_text())
    word_centers = center_payload["centers"]
    config = {
        "version": VERSION,
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_fingerprint": manifest["fingerprint"],
        "source_centers_sha256": file_sha256(args.source_centers),
        "source_centers_fingerprint": center_payload["fingerprint"],
        "centers": word_centers,
        "max_new_tokens": args.max_new_tokens,
        "max_samples": args.max_samples,
        "model": "microsoft/llava-med-v1.5-mistral-7b",
        "conversation": "vicuna_v1",
        "decoding": "greedy",
        "code_sha256": file_sha256(Path(__file__)),
    }
    fingerprint = stable_json_sha256(config)
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    if args.raw.exists():
        repair_truncated_jsonl_tail(args.raw)
    done = {}
    if args.raw.exists():
        for line in args.raw.read_text().splitlines():
            record = json.loads(line)
            if record["fingerprint"] != fingerprint:
                raise ValueError("raw-cache fingerprint mismatch")
            done[(record["dataset"], record["id"])] = record

    adapter = LlavaMedOEAdapter(conv_mode="vicuna_v1")
    try:
        with args.raw.open("a") as output:
            for record in tqdm(
                records,
                desc="source-word-center-final",
                initial=len(done),
                total=len(records),
            ):
                key = (record["dataset"], record["id"])
                if key in done:
                    continue
                domain = (
                    "ophthalmology"
                    if record["dataset"] == "harvard"
                    else "radiology"
                )
                center = word_centers[domain]
                baseline_prompt = prompts(record["dataset"])
                center_prompt = (
                    baseline_prompt
                    + f" Provide a detailed report of approximately "
                    f"{center['median_words']} words. Use only findings supported "
                    "by the provided image and do not introduce external "
                    "diagnostic information."
                )
                with Image.open(record["image"]) as handle:
                    image = handle.convert("RGB")
                centered = adapter._generate_once(
                    image,
                    center_prompt,
                    1,
                    False,
                    1.0,
                    1.0,
                    args.max_new_tokens,
                    0,
                )[0].text
                baseline = adapter._generate_once(
                    image,
                    baseline_prompt,
                    1,
                    False,
                    1.0,
                    1.0,
                    args.max_new_tokens,
                    0,
                )[0].text
                if not baseline or not centered:
                    raise RuntimeError(f"empty report for {key}")
                result = {
                    "version": VERSION,
                    "fingerprint": fingerprint,
                    "dataset": record["dataset"],
                    "id": record["id"],
                    "image": record["image"],
                    "ground_truth": record["reference"],
                    "candidates": {
                        "baseline": baseline,
                        "source_word_center": centered,
                    },
                    "source_word_center": {
                        "domain": domain,
                        **center,
                        "source_report_content_used": False,
                    },
                    "ground_truth_used_for_generation_or_selection": False,
                }
                output.write(json.dumps(result) + "\n")
                output.flush()
                done[key] = result
    finally:
        adapter.close()

    ordered = [done[(row["dataset"], row["id"])] for row in records]
    result = {
        "version": VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": config,
        "status": "final",
        "n": len(ordered),
        "ground_truth_used_for_generation_or_selection": False,
        "records": ordered,
    }
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "n": len(ordered),
                "centers": word_centers,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
