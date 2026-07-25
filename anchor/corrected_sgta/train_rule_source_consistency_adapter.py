"""Train a tiny DG adapter from paired original/source-guided image views.

The VLM is frozen.  A zero-initialized residual bottleneck after the visual
projector is optimized on RULE's test-disjoint IU-Xray alignment split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm

from corrected_sgta.frequency_alignment_source_spectrum_release2 import (
    source_spectrum_alignment_release2,
)
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter


VERSION = "source-view-consistency-adapter-v1"


class ResidualBottleneck(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.up(F.gelu(self.down(value.float())))
        return value + update.to(value.dtype)


@contextmanager
def attach_adapter(model, module: nn.Module):
    original = model.encode_images

    def encode(images):
        return module(original(images))

    model.encode_images = encode
    try:
        yield
    finally:
        model.encode_images = original


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rule_label(text: str) -> int:
    first = str(text).split(".")[0].lower()
    tokens = first.replace(",", " ").replace(";", " ").split()
    return 1 if any(token in {"no", "not"} for token in tokens) else 0


def clean_prompt(text: str) -> str:
    return str(text).replace("<image>", "").strip()


def class_logits(adapter, image: Image.Image, prompt: str) -> torch.Tensor:
    from llava.mm_utils import process_images

    input_ids = adapter._prompt_ids(prompt).to(adapter.model.device)
    pixels = process_images([image], adapter.image_processor, adapter.model.config)
    if isinstance(pixels, list):
        pixels = [
            item.to(adapter.model.device, dtype=adapter.model.dtype) for item in pixels
        ]
    else:
        pixels = pixels.to(adapter.model.device, dtype=adapter.model.dtype)
    _, positions, mask, _, embeds, _ = adapter.model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        None,
        None,
        None,
        pixels,
        image_sizes=[image.size],
    )
    output = adapter.model.model(
        input_ids=None,
        attention_mask=mask,
        position_ids=positions,
        inputs_embeds=embeds,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state[:, -1]
    weight = adapter.model.get_output_embeddings().weight
    columns = []
    for group in adapter.label_id_groups(("Yes", "No")):
        columns.append((hidden.to(weight.dtype) @ weight[group].T).max(-1).values)
    return torch.stack(columns, dim=-1).float()


def symmetric_kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_log = F.log_softmax(left, dim=-1)
    right_log = F.log_softmax(right, dim=-1)
    return 0.5 * (
        F.kl_div(left_log, right_log.exp(), reduction="batchmean")
        + F.kl_div(right_log, left_log.exp(), reduction="batchmean")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--center", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("erm", "consistency"), required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = json.loads(args.train_json.read_text())
    rows = sorted(rows, key=lambda row: hashlib.sha256(
        f"{args.seed}:{row['id']}".encode()
    ).hexdigest())[: args.max_samples]
    center = np.load(args.center)

    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    width = int(adapter.model.config.hidden_size)
    module = ResidualBottleneck(width, args.rank).to(adapter.model.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.learning_rate)
    history = []

    with attach_adapter(adapter.model, module):
        for epoch in range(args.epochs):
            random.Random(args.seed + epoch).shuffle(rows)
            progress = tqdm(rows, desc=f"{args.mode} epoch {epoch + 1}")
            for row in progress:
                image_path = args.image_root / row["image"]
                with Image.open(image_path) as handle:
                    original = handle.convert("RGB")
                prompt = clean_prompt(row["conversations"][0]["value"])
                target = torch.tensor(
                    [rule_label(row["conversations"][1]["value"])],
                    device=adapter.model.device,
                )
                optimizer.zero_grad(set_to_none=True)
                original_logits = class_logits(adapter, original, prompt)
                loss = F.cross_entropy(original_logits, target)
                consistency = torch.zeros((), device=loss.device)
                if args.mode == "consistency":
                    view = source_spectrum_alignment_release2(
                        original, center, low_frequency_ratio=args.alpha
                    )
                    view_logits = class_logits(adapter, view, prompt)
                    loss = 0.5 * (
                        F.cross_entropy(original_logits, target)
                        + F.cross_entropy(view_logits, target)
                    )
                    consistency = symmetric_kl(original_logits, view_logits)
                    loss = loss + args.consistency_weight * consistency
                loss.backward()
                optimizer.step()
                item = {
                    "loss": float(loss.detach()),
                    "consistency": float(consistency.detach()),
                }
                history.append(item)
                progress.set_postfix(loss=f"{item['loss']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VERSION,
        "config": vars(args) | {
            "train_json": str(args.train_json),
            "image_root": str(args.image_root),
            "center": str(args.center),
            "output": str(args.output),
        },
        "provenance": {
            "train_sha256": sha256(args.train_json),
            "center_sha256": sha256(args.center),
        },
        "state_dict": module.state_dict(),
        "history": history,
    }
    torch.save(payload, args.output)
    print(json.dumps({
        "output": str(args.output),
        "samples": len(rows),
        "final_loss": history[-1]["loss"],
    }, indent=2))


if __name__ == "__main__":
    main()
