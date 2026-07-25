#!/usr/bin/env python3
"""Generate reports aligned to a diagnosis-free source structure center."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.models_oe import LlavaMedOEAdapter
from corrected_sgta.rule_source_preference import file_sha256, stable_json_sha256
from corrected_sgta.run_mmedrag_sequence_anchor import prompts


VERSION = "mmedrag-source-structure-center-v1"


def sentence_count(text: str) -> int:
    return len(
        [
            value
            for value in re.split(r"(?<=[.!?])\s+", text.strip())
            if value.strip()
        ]
    )


def centers(source_bank: Path) -> dict[str, dict[str, int]]:
    payload = torch.load(source_bank, map_location="cpu", weights_only=False)
    groups = {
        "radiology": [
            row["report"]
            for row in payload["rows_data"]
            if row["domain"] == "radiology_iuxray"
        ],
        "ophthalmology": [
            row["report"]
            for row in payload["rows_data"]
            if row["domain"] == "ophthalmology_harvard"
        ],
    }
    result = {}
    for name, reports in groups.items():
        if not reports:
            raise ValueError(f"empty source domain: {name}")
        result[name] = {
            "source_reports": len(reports),
            "median_words": int(statistics.median(len(x.split()) for x in reports)),
            "median_sentences": int(
                statistics.median(sentence_count(x) for x in reports)
            ),
        }
    return result


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
    source_centers = centers(args.source_bank)
    config = {
        "version": VERSION,
        "input_sha256": file_sha256(args.input),
        "source_bank_sha256": file_sha256(args.source_bank),
        "source_fingerprint": payload["fingerprint"],
        "centers": source_centers,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "code_sha256": file_sha256(Path(__file__)),
    }
    fingerprint = stable_json_sha256(config)
    output = []
    adapter = LlavaMedOEAdapter(conv_mode="vicuna_v1")
    try:
        for record in tqdm(payload["records"], desc="source-structure-center"):
            domain = "ophthalmology" if record["dataset"] == "harvard" else "radiology"
            center = source_centers[domain]
            prompt = (
                prompts(record["dataset"])
                + f" Match the source-domain report structure: approximately "
                f"{center['median_sentences']} concise sentences and "
                f"{center['median_words']} words in total. Every sentence must "
                "state only evidence supported by the provided image; omit "
                "uncertain content rather than filling the target length."
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
                    f"empty structure-center report: {record['dataset']}:{record['id']}"
                )
            output.append(
                {
                    **record,
                    "structure_center_fingerprint": fingerprint,
                    "candidates": {
                        **record["candidates"],
                        "source_structure_center": answer,
                    },
                    "source_structure_center": {
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
