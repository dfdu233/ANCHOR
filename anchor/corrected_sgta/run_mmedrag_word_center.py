#!/usr/bin/env python3
"""Generate reports aligned only to the robust source word-count center."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from corrected_sgta.models_oe import LlavaMedOEAdapter
from corrected_sgta.rule_source_preference import file_sha256, stable_json_sha256
from corrected_sgta.run_mmedrag_sequence_anchor import prompts
from corrected_sgta.run_mmedrag_structure_center import centers


VERSION = "mmedrag-source-word-center-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    payload = json.loads(args.input.read_text())
    full_centers = centers(args.source_bank)
    word_centers = {
        name: {
            "source_reports": value["source_reports"],
            "median_words": value["median_words"],
        }
        for name, value in full_centers.items()
    }
    config = {
        "version": VERSION,
        "input_sha256": file_sha256(args.input),
        "source_bank_sha256": file_sha256(args.source_bank),
        "source_fingerprint": payload["fingerprint"],
        "centers": word_centers,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "code_sha256": file_sha256(Path(__file__)),
    }
    fingerprint = stable_json_sha256(config)
    output = []
    adapter = LlavaMedOEAdapter(conv_mode="vicuna_v1")
    try:
        for record in tqdm(payload["records"], desc="source-word-center"):
            domain = "ophthalmology" if record["dataset"] == "harvard" else "radiology"
            center = word_centers[domain]
            prompt = (
                prompts(record["dataset"])
                + f" Provide a detailed report of approximately "
                f"{center['median_words']} words. Use only findings supported "
                "by the provided image and do not introduce external "
                "diagnostic information."
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
                    f"empty word-center report: {record['dataset']}:{record['id']}"
                )
            output.append(
                {
                    **record,
                    "word_center_fingerprint": fingerprint,
                    "candidates": {
                        **record["candidates"],
                        "source_word_center": answer,
                    },
                    "source_word_center": {
                        "domain": domain,
                        **center,
                        "source_report_content_used": False,
                        "ground_truth_used": False,
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
