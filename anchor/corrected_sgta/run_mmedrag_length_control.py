#!/usr/bin/env python3
"""Generate a source-length-only control for report prompting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from corrected_sgta.models_oe import LlavaMedOEAdapter
from corrected_sgta.rule_source_preference import file_sha256, stable_json_sha256
from corrected_sgta.run_mmedrag_sequence_anchor import prompts


VERSION = "mmedrag-source-length-control-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    payload = json.loads(args.input.read_text())
    config = {
        "version": VERSION,
        "input_sha256": file_sha256(args.input),
        "source_fingerprint": payload["fingerprint"],
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "code_sha256": file_sha256(Path(__file__)),
    }
    fingerprint = stable_json_sha256(config)
    output = []
    adapter = LlavaMedOEAdapter(conv_mode="vicuna_v1")
    try:
        for record in tqdm(payload["records"], desc="source-length-control"):
            word_count = len(record["source_neighbor"]["report"].split())
            prompt = (
                prompts(record["dataset"])
                + f" Provide a detailed report of approximately {word_count} words. "
                "Use only findings supported by the provided image and do not "
                "introduce external diagnostic information."
            )
            with Image.open(record["image"]) as handle:
                image = handle.convert("RGB")
            answer = adapter._generate_once(
                image,
                prompt,
                1,
                False,
                1.0,
                1.0,
                args.max_new_tokens,
                0,
            )[0].text
            if not answer:
                raise RuntimeError(
                    f"empty length-control report: {record['dataset']}:{record['id']}"
                )
            output.append(
                {
                    **record,
                    "length_control_fingerprint": fingerprint,
                    "candidates": {
                        **record["candidates"],
                        "source_length_control": answer,
                    },
                    "source_length_control": {
                        "requested_words": word_count,
                        "ground_truth_used": False,
                        "source_report_content_used": False,
                    },
                }
            )
    finally:
        adapter.close()

    result = {
        "version": VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": config,
        "n": len(output),
        "ground_truth_used_for_generation_or_selection": False,
        "records": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fingerprint": fingerprint, "n": len(output)}, indent=2))


if __name__ == "__main__":
    main()
