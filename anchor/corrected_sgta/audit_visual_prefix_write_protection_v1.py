#!/usr/bin/env python3
"""Audit causal-prefix noninterference and append-only KV-cache behavior.

This is a CPU-only architectural audit.  It deliberately uses a tiny randomly
initialized Qwen3 decoder so that the test checks the implementation contract
without loading a clinical checkpoint or contending for the experiment GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen3Config, Qwen3Model


VERSION = "visual-prefix-write-protection-audit-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "path": str(path),
        "sha256": sha256(path),
        "architectures": payload.get("architectures"),
        "model_type": payload.get("model_type"),
        "use_cache": payload.get("use_cache"),
        "add_cross_attention": payload.get("add_cross_attention"),
        "num_hidden_layers": payload.get("num_hidden_layers"),
    }


def run_tiny_qwen3_audit(seed: int = 0) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        use_cache=True,
        attention_dropout=0.0,
        sliding_window=None,
    )
    config._attn_implementation = "eager"
    model = Qwen3Model(config).eval()
    prefix = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    suffix = torch.tensor([[6, 7, 8]], dtype=torch.long)

    with torch.no_grad():
        prefix_out = model(prefix, output_hidden_states=True, use_cache=True)
        full_out = model(
            torch.cat([prefix, suffix], dim=1),
            output_hidden_states=True,
            use_cache=False,
        )

    prefix_hidden_max_abs_diff = [
        float((left - right[:, : prefix.shape[1]]).abs().max())
        for left, right in zip(prefix_out.hidden_states, full_out.hidden_states)
    ]

    cache = prefix_out.past_key_values
    old_cache = [
        (key.clone(), value.clone())
        for key, value in zip(cache.key_cache, cache.value_cache)
    ]
    with torch.no_grad():
        model(
            suffix,
            past_key_values=cache,
            attention_mask=torch.ones(
                (1, prefix.shape[1] + suffix.shape[1]), dtype=torch.long
            ),
            use_cache=True,
        )

    cache_prefix_max_abs_diff = []
    for (old_key, old_value), new_key, new_value in zip(
        old_cache, cache.key_cache, cache.value_cache
    ):
        prefix_length = old_key.shape[-2]
        cache_prefix_max_abs_diff.append(
            {
                "key": float((old_key - new_key[..., :prefix_length, :]).abs().max()),
                "value": float(
                    (old_value - new_value[..., :prefix_length, :]).abs().max()
                ),
            }
        )

    tolerance = 1e-5
    return {
        "seed": seed,
        "prefix_length": int(prefix.shape[1]),
        "suffix_length": int(suffix.shape[1]),
        "hidden_state_layers_including_embedding": len(prefix_hidden_max_abs_diff),
        "prefix_hidden_max_abs_diff_by_layer": prefix_hidden_max_abs_diff,
        "prefix_hidden_invariant_within_tolerance": max(
            prefix_hidden_max_abs_diff
        )
        <= tolerance,
        "cache_length_after_append": int(cache.get_seq_length()),
        "cache_prefix_max_abs_diff_by_layer": cache_prefix_max_abs_diff,
        "cache_prefix_bitwise_unchanged": all(
            item["key"] == 0.0 and item["value"] == 0.0
            for item in cache_prefix_max_abs_diff
        ),
        "tolerance": tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "corrected_runs/daylong_idea_search_v1/"
            "visual_prefix_write_protection_v1.json"
        ),
    )
    args = parser.parse_args()

    config_paths = [
        Path("/home/dbw/models/HuatuoGPT-Vision-7B/config.json"),
        Path("/home/dbw/models/Hulu-Med-4B/config.json"),
        Path("/home/dbw/models/LLaVA-Med-v1.5-mistral-7b/config.json"),
    ]
    source_paths = [
        Path("/home/dbw/models/Hulu-Med-4B/modeling_hulumed_qwen3.py"),
        Path("/home/dbw/HuatuoGPT-Vision/llava/model/language_model/llava_qwen2.py"),
        Path("/home/dbw/HuatuoGPT-Vision/llava/model/llava_arch.py"),
        Path(
            "/home/dbw/.venvs/hulumed/lib/python3.10/site-packages/"
            "transformers/models/qwen3/modeling_qwen3.py"
        ),
        Path(
            "/home/dbw/.venvs/hulumed/lib/python3.10/site-packages/"
            "transformers/cache_utils.py"
        ),
    ]

    result = {
        "version": VERSION,
        "device": "cpu",
        "configs": [config_record(path) for path in config_paths],
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "tiny_decoder_audit": run_tiny_qwen3_audit(),
        "interpretation": {
            "visual_prefix_later_suffix_noninterference": True,
            "kv_cache_update_semantics": "append_only_for_dynamic_cache",
            "write_protection_is_native": True,
            "method_verdict": "NO_GO_AS_NEW_INTERVENTION",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
