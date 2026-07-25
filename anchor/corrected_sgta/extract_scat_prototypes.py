#!/usr/bin/env python3
"""Extract fixed Yes/No semantic LM-head prototypes for SCA-T analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch.nn.functional as F

from corrected_sgta.models_surface import load_adapter
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("hulu", "llava"))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = load_adapter(args.model)
    labels = ("Yes", "No")
    groups = adapter.label_id_groups(labels)
    weight = adapter.model.get_output_embeddings().weight.detach().float()
    prototypes = []
    for group in groups:
        rows = F.normalize(weight[group].cpu(), dim=-1)
        prototypes.append(F.normalize(rows.mean(0), dim=0).numpy())
    array = np.stack(prototypes).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, prototypes=array)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "model": args.model,
        "adapter_name": adapter.name,
        "labels": list(labels),
        "surface_token_ids": groups,
        "aggregation": "normalize each surface LM-head row, mean, renormalize",
        "feature_space": "last multimodal prompt hidden state / LM output embedding",
        "shape": list(array.shape),
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(json.dumps(metadata, indent=2))
    adapter.close()


if __name__ == "__main__":
    main()
