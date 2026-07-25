#!/usr/bin/env python3
"""Fuse conservative and source-guided reports with one image-grounded pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from corrected_sgta.models_oe import LlavaMedOEAdapter
from corrected_sgta.rule_source_preference import file_sha256, stable_json_sha256
from corrected_sgta.run_mmedrag_sequence_anchor import prompts


VERSION = "mmedrag-source-consensus-fusion-v1"


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
        "fusion": "one image-grounded consensus generation",
        "code_sha256": file_sha256(Path(__file__)),
    }
    fingerprint = stable_json_sha256(config)
    output = []
    adapter = LlavaMedOEAdapter(conv_mode="vicuna_v1")
    try:
        for record in tqdm(payload["records"], desc="source-consensus-fusion"):
            baseline = record["candidates"]["baseline"]
            guided = record["candidates"]["guided"]
            prompt = (
                prompts(record["dataset"])
                + "\nTwo candidate reports were produced from different source-domain "
                "views of this same image.\n"
                + f"Candidate A (conservative): {baseline}\n"
                + f"Candidate B (source-guided): {guided}\n"
                + "Write one final report grounded only in the provided image. "
                "Preserve clinically useful detail when it is supported by the image, "
                "but reject any finding that may have been copied from a candidate "
                "and is not visually supported. Output only the final report."
            )
            with Image.open(record["image"]) as handle:
                image = handle.convert("RGB")
            fused = adapter._generate_once(
                image,
                prompt,
                1,
                False,
                1.0,
                1.0,
                args.max_new_tokens,
                0,
            )[0].text
            if not fused:
                raise RuntimeError(
                    f"empty fused report: {record['dataset']}:{record['id']}"
                )
            output.append(
                {
                    **record,
                    "fusion_fingerprint": fingerprint,
                    "candidates": {
                        **record["candidates"],
                        "source_consensus_fusion": fused,
                    },
                    "ground_truth_used_for_fusion": False,
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
