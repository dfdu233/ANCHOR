#!/usr/bin/env python3
"""Ground source-guided reports with a source-contrastive sequence test.

A statement copied from a retrieved source report should be at least as
compatible with the retrieved source image as with the test image.  For a
candidate statement ``s`` we therefore use

    evidence(s) = NLL(s | x_source, q) - NLL(s | x_test, q).

Positive evidence favors the test image.  This is label-free, sequence-level,
and applies to unrestricted answers rather than a fixed answer vocabulary.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from corrected_sgta.models_oe import LlavaMedOEAdapter
from corrected_sgta.rule_source_preference import file_sha256, stable_json_sha256
from corrected_sgta.run_mmedrag_sequence_anchor import mean_sequence_nll, prompts


VERSION = "mmedrag-source-contrast-v1"


def sentences(text: str) -> list[str]:
    return [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+", text.strip())
        if value.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    payload = json.loads(args.input.read_text())
    config = {
        "version": VERSION,
        "input_sha256": file_sha256(args.input),
        "source_fingerprint": payload["fingerprint"],
        "margin": args.margin,
        "score": "mean_NLL(source_image)-mean_NLL(test_image)",
        "code_sha256": file_sha256(Path(__file__)),
    }
    fingerprint = stable_json_sha256(config)
    output = []
    adapter = LlavaMedOEAdapter(conv_mode="vicuna_v1")
    try:
        for record in tqdm(payload["records"], desc="source-contrast"):
            prompt = prompts(record["dataset"])
            with Image.open(record["image"]) as handle:
                test_image = handle.convert("RGB")
            with Image.open(record["source_neighbor"]["image"]) as handle:
                source_image = handle.convert("RGB")
            guided = record["candidates"]["guided"]
            guided_test_nll = record["mean_sequence_nll"]["guided"]
            guided_source_nll = mean_sequence_nll(
                adapter, source_image, prompt, guided
            )
            whole_evidence = guided_source_nll - guided_test_nll
            whole = (
                guided
                if whole_evidence >= args.margin
                else record["candidates"]["baseline"]
            )

            sentence_records = []
            accepted = []
            for sentence in sentences(guided):
                test_nll = mean_sequence_nll(
                    adapter, test_image, prompt, sentence
                )
                source_nll = mean_sequence_nll(
                    adapter, source_image, prompt, sentence
                )
                evidence = source_nll - test_nll
                keep = evidence >= args.margin
                if keep:
                    accepted.append(sentence)
                sentence_records.append(
                    {
                        "text": sentence,
                        "test_nll": test_nll,
                        "source_nll": source_nll,
                        "evidence": evidence,
                        "accepted": keep,
                    }
                )
            sentence_candidate = (
                " ".join(accepted)
                if accepted
                else record["candidates"]["baseline"]
            )
            updated = {
                **record,
                "contrast_fingerprint": fingerprint,
                "candidates": {
                    **record["candidates"],
                    "whole_source_contrast": whole,
                    "sentence_source_contrast": sentence_candidate,
                },
                "source_contrast": {
                    "margin": args.margin,
                    "whole": {
                        "test_nll": guided_test_nll,
                        "source_nll": guided_source_nll,
                        "evidence": whole_evidence,
                        "accepted": whole_evidence >= args.margin,
                    },
                    "sentences": sentence_records,
                },
                "ground_truth_used_for_source_contrast": False,
            }
            output.append(updated)
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
    print(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "n": len(output),
                "whole_accepted": sum(
                    row["source_contrast"]["whole"]["accepted"]
                    for row in output
                ),
                "sentences_accepted": sum(
                    item["accepted"]
                    for row in output
                    for item in row["source_contrast"]["sentences"]
                ),
                "sentences_total": sum(
                    len(row["source_contrast"]["sentences"])
                    for row in output
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
