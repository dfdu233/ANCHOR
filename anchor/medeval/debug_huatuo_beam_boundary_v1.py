#!/usr/bin/env python3
"""Capture one raw Huatuo beam output to diagnose its generation boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from anchor.corrected_sgta.models_oe import HuatuoOEAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = json.loads(args.manifest.read_text())[0]
    adapter = HuatuoOEAdapter()
    with Image.open(args.image_root / row["img_name"]) as source:
        image = source.convert("RGB")
    input_ids, image_tensors = adapter._inputs(image, str(row["question"]))
    with torch.inference_mode():
        output = adapter.model.generate(
            input_ids,
            images=image_tensors,
            use_cache=True,
            max_new_tokens=256,
            min_new_tokens=1,
            repetition_penalty=1.2,
            do_sample=False,
            num_return_sequences=1,
            num_beams=4,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=adapter.tokenizer.eos_token_id,
            pad_token_id=adapter.tokenizer.pad_token_id or adapter.tokenizer.eos_token_id,
        )
    ids = output.sequences[0].tolist()
    payload = {
        "qid": row.get("qid"),
        "input_shape": list(input_ids.shape),
        "sequence_shape": list(output.sequences.shape),
        "score_steps": len(output.scores),
        "beam_indices_shape": list(output.beam_indices.shape) if output.beam_indices is not None else None,
        "tokenizer_bos": adapter.tokenizer.bos_token_id,
        "tokenizer_eos": adapter.tokenizer.eos_token_id,
        "tokenizer_pad": adapter.tokenizer.pad_token_id,
        "ids": ids,
        "decoded_full": adapter.tokenizer.decode(ids, skip_special_tokens=False),
        "decoded_last_score_steps": adapter.tokenizer.decode(ids[-len(output.scores):], skip_special_tokens=False),
        "decoded_after_input_length": adapter.tokenizer.decode(ids[input_ids.shape[1]:], skip_special_tokens=False),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    adapter.close()
    print(json.dumps({key: value for key, value in payload.items() if key != "ids"}, indent=2))


if __name__ == "__main__":
    main()
