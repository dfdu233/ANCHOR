#!/usr/bin/env python3
"""Audit LLaVA-Med's delayed CLIP loader and cross-image sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open

from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    atomic_json,
    hidden_trajectory,
    label_ids,
    layer_logits,
    sha256_file,
)
from corrected_sgta.run_llava_vindr_commitment_probe import (
    LlavaRuntime,
    prepared_embeddings_llava,
)


VERSION = "llava-med-loader-audit-v1"


def checkpoint_tensor(
    model_dir: Path, weight_map: dict[str, str], key: str
) -> torch.Tensor:
    with safe_open(
        str(model_dir / weight_map[key]), framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor(key)


def audit_clip_serialization(model_dir: Path, clip_weights: Path) -> dict[str, object]:
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    base = torch.load(clip_weights, map_location="cpu", weights_only=True)
    base_keys = sorted(key for key in base if key.startswith("vision_model."))
    matched = 0
    exact = 0
    missing_checkpoint = []
    maximum_absolute_error = 0.0
    for base_key in base_keys:
        checkpoint_key = "model.vision_tower.vision_tower." + base_key
        if checkpoint_key not in index:
            missing_checkpoint.append(base_key)
            continue
        checkpoint = checkpoint_tensor(model_dir, index, checkpoint_key)
        reference = base[base_key].to(checkpoint.dtype)
        difference = (checkpoint - reference).abs().float()
        maximum_absolute_error = max(
            maximum_absolute_error, float(difference.max())
        )
        exact += int(torch.equal(checkpoint, reference))
        matched += 1
    del base
    # CLIP stores position_ids as a non-parameter buffer in some versions.
    admissible_missing = [
        key for key in missing_checkpoint if key.endswith("embeddings.position_ids")
    ]
    unexpected_missing = sorted(set(missing_checkpoint) - set(admissible_missing))
    passed = (
        matched > 0
        and exact == matched
        and maximum_absolute_error == 0.0
        and not unexpected_missing
    )
    return {
        "base_clip_weights": str(clip_weights.resolve()),
        "base_clip_sha256": sha256_file(clip_weights),
        "checkpoint_clip_tensors_matched": matched,
        "exact_after_dtype_cast": exact,
        "maximum_absolute_error": maximum_absolute_error,
        "admissible_nonparameter_buffers_absent": admissible_missing,
        "unexpected_missing": unexpected_missing,
        "passed": passed,
        "interpretation": (
            "Unused checkpoint vision keys are benign only because the separately "
            "loaded frozen CLIP tower is exactly equal after dtype conversion."
        ),
    }


@torch.inference_mode()
def audit_runtime(
    runtime: LlavaRuntime,
    model_dir: Path,
    image_a: Path,
    image_b: Path,
    prompt: str,
) -> dict[str, object]:
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    model_keys = set(runtime.model.state_dict())
    nonvision_checkpoint = {
        key for key in index if not key.startswith("model.vision_tower.")
    }
    missing_nonvision = sorted(nonvision_checkpoint - model_keys)
    sentinels = (
        "model.mm_projector.0.weight",
        "model.mm_projector.2.bias",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.31.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    )
    sentinel_audit = {}
    state = runtime.model.state_dict()
    for key in sentinels:
        checkpoint = checkpoint_tensor(model_dir, index, key)
        loaded = state[key].detach().cpu()
        # The official loader requests FP16.  Compare in that actual runtime
        # dtype: casting the lower-precision runtime tensor back upward cannot
        # reconstruct bits discarded by the documented load-time conversion.
        expected = checkpoint.to(loaded.dtype)
        difference = (loaded - expected).abs().float()
        sentinel_audit[key] = {
            "runtime_dtype": str(loaded.dtype),
            "checkpoint_dtype": str(checkpoint.dtype),
            "exact_after_checkpoint_to_runtime_dtype": bool(
                torch.equal(loaded, expected)
            ),
            "maximum_absolute_error": float(difference.max()),
        }

    images = []
    prepared = []
    for path in (image_a, image_b):
        image = Image.open(path).convert("RGB")
        images.append(image)
        prepared.append(prepared_embeddings_llava(runtime, prompt, image))
    (emb_a, att_a, pos_a, span_a), (emb_b, att_b, pos_b, span_b) = prepared
    visual_a = emb_a[0, span_a[0] : span_a[1]].float()
    visual_b = emb_b[0, span_b[0] : span_b[1]].float()
    if visual_a.shape != visual_b.shape:
        raise RuntimeError(f"visual shapes differ: {visual_a.shape} versus {visual_b.shape}")
    visual_difference = visual_a - visual_b
    hidden_a = hidden_trajectory(runtime, emb_a, att_a, pos_a)
    hidden_b = hidden_trajectory(runtime, emb_b, att_b, pos_b)
    ids = label_ids(runtime)
    final = len(hidden_a) - 1
    logits_a = layer_logits(runtime, hidden_a, [final], ids)[final]
    logits_b = layer_logits(runtime, hidden_b, [final], ids)[final]
    logit_linf = max(abs(logits_a[key] - logits_b[key]) for key in logits_a)
    for image in images:
        image.close()
    visual_l2 = float(visual_difference.norm())
    visual_mean_absolute = float(visual_difference.abs().mean())
    passed = bool(
        not missing_nonvision
        and all(
            value["exact_after_checkpoint_to_runtime_dtype"]
            for value in sentinel_audit.values()
        )
        and visual_l2 > 1e-6
        and visual_mean_absolute > 1e-8
        and logit_linf > 1e-6
    )
    return {
        "checkpoint_nonvision_keys": len(nonvision_checkpoint),
        "missing_nonvision_keys_in_runtime": missing_nonvision,
        "sentinel_weight_equality": sentinel_audit,
        "image_a": str(image_a.resolve()),
        "image_b": str(image_b.resolve()),
        "prompt": prompt,
        "visual_tokens": int(visual_a.shape[0]),
        "projected_visual_cross_image_l2": visual_l2,
        "projected_visual_cross_image_mean_absolute": visual_mean_absolute,
        "final_tristate_logits_a": logits_a,
        "final_tristate_logits_b": logits_b,
        "final_tristate_cross_image_linf": logit_linf,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/dbw/models/LLaVA-Med-v1.5-mistral-7b"),
    )
    parser.add_argument(
        "--llava-root",
        type=Path,
        default=Path(
            "/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5"
        ),
    )
    parser.add_argument(
        "--clip-weights",
        type=Path,
        default=Path(
            "/root/.cache/huggingface/hub/models--openai--"
            "clip-vit-large-patch14-336/snapshots/"
            "ce19dc912ca5cd21c8a653c79e251e808ccabcd1/pytorch_model.bin"
        ),
    )
    parser.add_argument("--image-a", type=Path, required=True)
    parser.add_argument("--image-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conv-mode", default="mistral_instruct")
    parser.add_argument(
        "--prompt",
        default=(
            "Does this chest X-ray show pleural effusion? "
            "Answer with exactly one word: Yes, No, or Maybe."
        ),
    )
    args = parser.parse_args()
    serialization = audit_clip_serialization(args.model_dir, args.clip_weights)
    runtime = LlavaRuntime(args.model_dir, args.llava_root, args.conv_mode)
    runtime_audit = audit_runtime(
        runtime, args.model_dir, args.image_a, args.image_b, args.prompt
    )
    payload = {
        "version": VERSION,
        "model_dir": str(args.model_dir.resolve()),
        "llava_root": str(args.llava_root.resolve()),
        "serialization": serialization,
        "runtime": runtime_audit,
        "passed": bool(serialization["passed"] and runtime_audit["passed"]),
        "claim_ceiling": (
            "This audit establishes loader fidelity and image sensitivity only; "
            "it does not establish clinical selectivity or task accuracy."
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise RuntimeError("LLaVA-Med loader audit failed")


if __name__ == "__main__":
    main()
