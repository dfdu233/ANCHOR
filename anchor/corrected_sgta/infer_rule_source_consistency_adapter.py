"""Run RULE-style decoded inference with a trained source-consistency adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.train_rule_source_consistency_adapter import (
    ResidualBottleneck,
    attach_adapter,
    clean_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    width = int(adapter.model.config.hidden_size)
    rank = int(payload["config"]["rank"])
    module = ResidualBottleneck(width, rank).to(adapter.model.device)
    module.load_state_dict(payload["state_dict"])
    module.eval()
    rows = [json.loads(line) for line in args.questions.read_text().splitlines()]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as handle, attach_adapter(adapter.model, module):
        for row in tqdm(rows, desc="decoded inference"):
            with Image.open(args.image_root / row["image"]) as source:
                image = source.convert("RGB")
            prompt = clean_prompt(row.get("text", row["question"]))
            output = adapter.decode_ce([image], prompt, max_new_tokens=64)[0]
            handle.write(json.dumps({
                "question_id": row["question_id"],
                "prompt": prompt,
                "text": output,
                "answer_id": shortuuid.uuid(),
                "model_id": "llava-med-source-consistency-adapter",
                "metadata": {
                    "checkpoint": str(args.checkpoint),
                    "method": payload["version"],
                    "mode": payload["config"]["mode"],
                },
            }) + "\n")


if __name__ == "__main__":
    main()
