#!/usr/bin/env python3
"""Outcome-blind HALP-style representation compatibility preflight.

This module implements three representation *semantics* for Huatuo
(LlavaQwen2) and Hulu (custom Qwen3): globally pooled pre-projector visual
features, the last decoder vision token, and the last active context/query
token.  It does not train a probe, read an outcome, reproduce HALP's model
ports, or authorize a scientific comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


VERSION = "halp-three-plane-compatibility-preflight-v1"
AUDIT_SCHEMA = "halp-three-plane-source-audit-v1"
LAYER_SCHEMA = "halp-three-plane-layer-contract-v1"
CAPTURE_SCHEMA = "halp-three-plane-engineering-capture-v1"
PLANES = ("visual_only", "decoder_vision_token", "query_token")
ROOT = Path("/home/dbw/ANCHOR")
MODEL_DIRS = {
    "huatuo": Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
    "hulu": Path("/home/dbw/models/Hulu-Med-4B"),
}
HUATUO_ROOT = Path("/home/dbw/HuatuoGPT-Vision")
DEFAULT_DICOM = Path(
    "/workspace/vinbigdata/train/d925309691e7929d905eaa42f081833f.dicom"
)
DEFAULT_GPU_LOCK = ROOT / "corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock"
DEFAULT_PROMPT = (
    "Does this chest X-ray show aortic enlargement? "
    "Answer with exactly one word: Yes, No, or Maybe."
)


class HALPCompatibilityError(RuntimeError):
    """Fail-closed compatibility or capture error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise HALPCompatibilityError(f"required file is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _source_spec(family: str, model_dir: Path, huatuo_root: Path) -> dict[str, Any]:
    if family == "huatuo":
        return {
            "model_type": "llava_qwen2",
            "architecture": "LlavaQwen2ForCausalLM",
            "lineage": "LLaVA-style CLIP -> MLP projector -> Qwen2 decoder",
            "source_files": [
                huatuo_root / "llava/model/language_model/llava_qwen2.py",
                huatuo_root / "llava/model/llava_arch.py",
                huatuo_root / "llava/model/multimodal_encoder/clip_encoder.py",
                huatuo_root / "llava/model/multimodal_projector/builder.py",
            ],
            "markers": {
                "llava_qwen2.py": ("class LlavaQwen2ForCausalLM",),
                "llava_arch.py": (
                    "get_vision_tower()(images)",
                    "mm_projector(image_features)",
                    "cur_new_input_embeds.append(cur_image_features)",
                ),
                "clip_encoder.py": (
                    "hidden_states[self.select_layer]",
                    "image_features[:, 1:]",
                ),
            },
            "visual_only_semantics": (
                "global mean of model-native CLIP selected-layer patch tokens "
                "immediately before mm_projector"
            ),
            "visual_span_semantics": (
                "one negative image placeholder expanded in-place to projected patch tokens"
            ),
        }
    if family == "hulu":
        return {
            "model_type": "hulumed_qwen3",
            "architecture": "HulumedQwen3ForCausalLM",
            "lineage": "custom post-LN/interpolated vision encoder -> MLP projector -> Qwen3 decoder",
            "source_files": [
                model_dir / "modeling_hulumed_qwen3.py",
                model_dir / "modeling_hulumed_encoder.py",
                model_dir / "processing_hulumed.py",
                model_dir / "configuration_hulumed_qwen3.py",
            ],
            "markers": {
                "modeling_hulumed_qwen3.py": (
                    "class HulumedQwen3ForCausalLM",
                    "get_vision_encoder()(",
                    "mm_projector(mm_features)",
                    "inputs_embeds[image_selected]",
                ),
                "modeling_hulumed_encoder.py": (
                    "self.post_layernorm(hidden_states)",
                    "torch.nn.functional.interpolate(",
                ),
                "processing_hulumed.py": ("self.image_token_id",),
            },
            "visual_only_semantics": (
                "global mean of model-native custom vision output after post-LN and "
                "spatial interpolation, immediately before mm_projector"
            ),
            "visual_span_semantics": (
                "contiguous processor-expanded image-token IDs replaced in-place; "
                "token compression must be disabled"
            ),
        }
    raise HALPCompatibilityError(f"unsupported model family: {family}")


def selection_policy() -> dict[str, Any]:
    policy = {
        "selection_split": "dev_only",
        "group_unit": "global_image_id",
        "cross_validation": "stratified_group_kfold",
        "folds": 5,
        "selection_is_model_family_specific": True,
        "candidate_planes": list(PLANES),
        "candidate_layers": "all enumerated post-block layers for decoder planes",
        "primary_selection_metric": "group-CV AUROC",
        "tie_breakers": ["group-CV Brier", "shallower normalized depth"],
        "confirmation_mode": "apply_only",
        "confirmation_refit": False,
        "confirmation_layer_selection": False,
        "confirmation_plane_selection": False,
        "confirmation_threshold_tuning": False,
        "confirmation_outcome_read_during_capture": False,
    }
    policy["fingerprint"] = canonical_sha256(policy)
    return policy


def validate_selection_policy(policy: Mapping[str, Any]) -> None:
    declared = policy.get("fingerprint")
    body = {key: value for key, value in policy.items() if key != "fingerprint"}
    if declared != canonical_sha256(body):
        raise HALPCompatibilityError("selection policy fingerprint mismatch")
    if (
        policy.get("selection_split") != "dev_only"
        or policy.get("group_unit") != "global_image_id"
        or policy.get("cross_validation") != "stratified_group_kfold"
        or policy.get("confirmation_mode") != "apply_only"
        or any(
            policy.get(field) is not False
            for field in (
                "confirmation_refit",
                "confirmation_layer_selection",
                "confirmation_plane_selection",
                "confirmation_threshold_tuning",
                "confirmation_outcome_read_during_capture",
            )
        )
    ):
        raise HALPCompatibilityError("selection/confirmation split policy drift")


def cpu_source_audit(
    *, family: str, model_dir: Path, huatuo_root: Path = HUATUO_ROOT
) -> dict[str, Any]:
    """Audit architecture semantics without importing model code or touching CUDA."""

    if torch.cuda.is_initialized():
        raise HALPCompatibilityError("CPU/source audit entered after CUDA initialization")
    model_dir = model_dir.resolve()
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = _source_spec(family, model_dir, huatuo_root.resolve())
    layers = config.get("num_hidden_layers")
    if (
        config.get("model_type") != spec["model_type"]
        or spec["architecture"] not in config.get("architectures", [])
        or isinstance(layers, bool)
        or not isinstance(layers, int)
        or layers <= 0
    ):
        raise HALPCompatibilityError(f"{family} config/architecture identity drift")
    if family == "hulu" and config.get("use_token_compression") is not False:
        raise HALPCompatibilityError("Hulu token compression prevents exact visual-span identity")
    sources = []
    for path in spec["source_files"]:
        record = file_record(path)
        text = path.read_text(encoding="utf-8")
        for marker in spec["markers"].get(path.name, ()):
            if marker not in text:
                raise HALPCompatibilityError(f"{family} source marker missing: {path.name}:{marker}")
        sources.append(record)
    layer_rows = [
        {
            "layer_number": number,
            "zero_based_block_index": number - 1,
            "normalized_depth": number / layers,
            "module_path": f"model.layers.{number - 1}",
            "capture_location": "decoder block output; post-residual; pre-final-norm",
        }
        for number in range(1, layers + 1)
    ]
    policy = selection_policy()
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "status": "cpu_source_audit_passed_no_model_or_cuda",
        "model_family": family,
        "model_dir": str(model_dir),
        "model_type": spec["model_type"],
        "architecture": spec["architecture"],
        "lineage": spec["lineage"],
        "hidden_size": int(config["hidden_size"]),
        "decoder_layer_count": layers,
        "config": file_record(config_path),
        "checkpoint_index": file_record(model_dir / "model.safetensors.index.json"),
        "source_files": sources,
        "capture_implementation": file_record(Path(__file__)),
        "representation_contract": {
            "visual_only": spec["visual_only_semantics"],
            "decoder_vision_token": (
                "post-block state at exact final index of the contiguous visual-token span"
            ),
            "query_token": (
                "post-block state at the final active context token used to predict the first answer token"
            ),
        },
        "visual_span_semantics": spec["visual_span_semantics"],
        "layer_indexing": "one-based decoder block number; no embedding pseudo-layer",
        "layers": layer_rows,
        "layer_enumeration_sha256": canonical_sha256(layer_rows),
        "selection_policy": policy,
        "halp_conceptual_three_plane_compatibility": True,
        "official_halp_code_reproduction_claimed": False,
        "cross_architecture_latent_semantics_identical_claimed": False,
        "probe_training_authorized": False,
        "outcome_read": False,
        "model_loaded": False,
        "cuda_touched": False,
    }
    audit["fingerprint"] = canonical_sha256(audit)
    return audit


def _decoder_core(causal_lm: Any) -> Any:
    if hasattr(causal_lm, "layers"):
        return causal_lm
    core = getattr(causal_lm, "model", None)
    if core is not None and hasattr(core, "layers"):
        return core
    raise HALPCompatibilityError("cannot resolve decoder model.layers")


def runtime_layer_contract(causal_lm: Any, audit: Mapping[str, Any]) -> dict[str, Any]:
    core = _decoder_core(causal_lm)
    blocks = list(core.layers)
    if len(blocks) != audit["decoder_layer_count"]:
        raise HALPCompatibilityError("runtime decoder layer count differs from CPU audit")
    rows = []
    for number, block in enumerate(blocks, 1):
        parameter_schema = [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
            }
            for name, parameter in block.named_parameters(recurse=True)
        ]
        rows.append(
            {
                "layer_number": number,
                "zero_based_block_index": number - 1,
                "normalized_depth": number / len(blocks),
                "module_path": f"model.layers.{number - 1}",
                "module_class": f"{type(block).__module__}.{type(block).__qualname__}",
                "parameter_count": sum(parameter.numel() for parameter in block.parameters()),
                "parameter_schema_sha256": canonical_sha256(parameter_schema),
                "capture_location": "decoder block output; post-residual; pre-final-norm",
            }
        )
    contract = {
        "schema_version": LAYER_SCHEMA,
        "model_family": audit["model_family"],
        "source_audit_fingerprint": audit["fingerprint"],
        "layer_count": len(rows),
        "layers": rows,
        "layer_order_sha256": canonical_sha256(rows),
    }
    contract["fingerprint"] = canonical_sha256(contract)
    return contract


def _tensor_output(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(value):
        raise HALPCompatibilityError(f"hook output is not a tensor: {type(value).__name__}")
    return value


class PreProjectorVisionCapture:
    """Capture one model-native vision-encoder output without modifying it."""

    def __init__(self, vision_module: Any):
        self.vision_module = vision_module
        self.values: list[torch.Tensor] = []
        self.handle: Any = None

    def __enter__(self) -> "PreProjectorVisionCapture":
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self.values.append(_tensor_output(output).detach())

        self.handle = self.vision_module.register_forward_hook(hook)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def one(self) -> torch.Tensor:
        if len(self.values) != 1:
            raise HALPCompatibilityError(
                f"expected one pre-projector vision call, observed {len(self.values)}"
            )
        return self.values[0]


def pool_visual_only(output: torch.Tensor) -> tuple[np.ndarray, int]:
    if output.ndim == 3:
        if output.shape[0] != 1:
            raise HALPCompatibilityError("visual-only capture requires batch size one")
        tokens = output[0]
    elif output.ndim == 2:
        tokens = output
    else:
        raise HALPCompatibilityError(f"unexpected visual-only shape: {tuple(output.shape)}")
    if tokens.shape[0] <= 0 or tokens.shape[1] <= 0 or not bool(torch.isfinite(tokens).all()):
        raise HALPCompatibilityError("visual-only tokens are empty or nonfinite")
    return tokens.float().mean(dim=0).cpu().numpy(), int(tokens.shape[0])


@torch.inference_mode()
def capture_three_planes(
    *,
    decoder_model: Any,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    visual_span: tuple[int, int],
    visual_only_output: torch.Tensor,
    layers: Sequence[int],
) -> dict[str, Any]:
    """Capture exact HALP-style positions from post-block decoder states."""

    core = _decoder_core(decoder_model)
    blocks = list(core.layers)
    requested = sorted(set(int(layer) for layer in layers))
    if (
        not requested
        or requested[0] < 1
        or requested[-1] > len(blocks)
        or requested[-1] != len(blocks)
    ):
        raise HALPCompatibilityError("capture layers must be valid and include the final block")
    if embeddings.ndim != 3 or embeddings.shape[0] != 1:
        raise HALPCompatibilityError("decoder capture requires [1, sequence, hidden] embeddings")
    sequence = int(embeddings.shape[1])
    start, end = (int(visual_span[0]), int(visual_span[1]))
    if not 0 <= start < end <= sequence:
        raise HALPCompatibilityError("visual span is invalid")
    if attention_mask is None:
        active = torch.arange(sequence, device=embeddings.device)
        attention_mask = torch.ones((1, sequence), dtype=torch.bool, device=embeddings.device)
    else:
        if tuple(attention_mask.shape) != (1, sequence):
            raise HALPCompatibilityError("attention mask shape disagrees with embeddings")
        active = torch.nonzero(attention_mask[0].bool(), as_tuple=False).flatten()
    if active.numel() == 0 or not bool(
        torch.equal(active, torch.arange(int(active[0]), int(active[-1]) + 1, device=active.device))
    ):
        raise HALPCompatibilityError("active context tokens must form one contiguous span")
    query_index = int(active[-1])
    if start <= query_index < end or not bool(attention_mask[0, start:end].bool().all()):
        raise HALPCompatibilityError("query/visual position identity is ambiguous")
    vision_index = end - 1
    captured: dict[int, dict[str, torch.Tensor]] = {}
    final_full: list[torch.Tensor] = []
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = _tensor_output(output)
            if tuple(hidden.shape[:2]) != (1, sequence):
                raise HALPCompatibilityError(
                    f"layer {layer} hidden shape drift: {tuple(hidden.shape)}"
                )
            if not bool(torch.isfinite(hidden[:, (vision_index, query_index)]).all()):
                raise HALPCompatibilityError(f"layer {layer} captured nonfinite state")
            captured[layer] = {
                "decoder_vision_token": hidden[0, vision_index].detach().float().cpu(),
                "query_token": hidden[0, query_index].detach().float().cpu(),
            }
            if layer == len(blocks):
                final_full.append(hidden.detach().clone())

        return hook

    for layer in requested:
        handles.append(blocks[layer - 1].register_forward_hook(make_hook(layer)))
    try:
        output = core(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=embeddings,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(requested) or len(final_full) != 1:
        raise HALPCompatibilityError("decoder hooks did not close the requested layers")
    observed = output.last_hidden_state
    reconstructed = core.norm(final_full[0]) if hasattr(core, "norm") else final_full[0]
    selected = torch.tensor([vision_index, query_index], device=observed.device)
    left = reconstructed[0].index_select(0, selected).float()
    right = observed[0].index_select(0, selected).float()
    maximum_error = float((left - right).abs().max().cpu())
    cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=-1).min().cpu())
    if maximum_error > 0.03 or cosine < 0.999:
        raise HALPCompatibilityError(
            "final hook is not post-block/pre-final-norm: "
            f"max_abs={maximum_error:.6f}, cosine={cosine:.6f}"
        )
    visual_only, visual_token_count = pool_visual_only(visual_only_output)
    return {
        "visual_only": visual_only,
        "decoder_vision_token": {
            layer: captured[layer]["decoder_vision_token"].numpy() for layer in requested
        },
        "query_token": {
            layer: captured[layer]["query_token"].numpy() for layer in requested
        },
        "capture_audit": {
            "layers": requested,
            "visual_span": [start, end],
            "vision_token_index": vision_index,
            "query_token_index": query_index,
            "visual_only_token_count": visual_token_count,
            "expanded_sequence_length": sequence,
            "final_norm_max_abs_error": maximum_error,
            "final_norm_min_cosine": cosine,
            "one_forward_no_generation": True,
        },
    }


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_engineering_capture(
    *,
    output_dir: Path,
    family: str,
    audit: Mapping[str, Any],
    layer_contract: Mapping[str, Any],
    capture: Mapping[str, Any],
    input_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        raise HALPCompatibilityError(f"engineering capture output already exists: {output_dir}")
    arrays: dict[str, np.ndarray] = {
        "visual_only": np.asarray(capture["visual_only"], dtype=np.float32)
    }
    for plane in ("decoder_vision_token", "query_token"):
        for layer, value in capture[plane].items():
            arrays[f"{plane}__layer_{int(layer):03d}"] = np.asarray(value, dtype=np.float32)
    output_dir.mkdir(parents=True)
    arrays_path = output_dir / "representations.npz"
    _atomic_npz(arrays_path, arrays)
    metadata = {
        "schema_version": CAPTURE_SCHEMA,
        "status": "engineering_single_claim_capture_complete",
        "scientific_status": "plumbing_only_no_probe_or_outcome",
        "model_family": family,
        "source_audit_fingerprint": audit["fingerprint"],
        "layer_contract": layer_contract,
        "input_identity": dict(input_identity),
        "representations": file_record(arrays_path),
        "array_names": sorted(arrays),
        "capture_audit": capture["capture_audit"],
        "probe_trained": False,
        "outcome_read": False,
        "dev_selection_performed": False,
        "confirmation_applied": False,
        "official_halp_code_reproduction_claimed": False,
        "paper_claim_authorized": False,
    }
    metadata["fingerprint"] = canonical_sha256(metadata)
    atomic_json(output_dir / "metadata.json", metadata)
    return metadata


def real_single_claim_capture(
    *,
    family: str,
    model_dir: Path,
    dicom_path: Path,
    prompt: str,
    output_dir: Path,
    gpu_lock: Path,
    huatuo_root: Path = HUATUO_ROOT,
) -> dict[str, Any]:
    """Run one engineering-only capture; never generates or reads an answer."""

    from anchor.corrected_sgta.cecd_dual_semantics_ce_adapter_v1 import gpu_flock
    from anchor.corrected_sgta.run_cecd_factorial_v1 import HuatuoScorer, HuluScorer
    from anchor.corrected_sgta.run_huatuo_vindr_commitment_probe import load_image

    audit = cpu_source_audit(
        family=family, model_dir=model_dir, huatuo_root=huatuo_root
    )
    image = load_image(dicom_path)
    with gpu_flock(gpu_lock):
        if family == "huatuo":
            scorer = HuatuoScorer(model_dir, huatuo_root, "cuda:0")
            bot = scorer.bot
            image_tensor = torch.stack(bot.get_image_tensors([image])).to(
                bot.model.device, dtype=torch.bfloat16
            )
            with_image = bot.insert_image_placeholder(prompt, 1)
            input_ids = bot.preprocess(
                bot.get_conv_without_history(with_image), return_tensors="pt"
            ).to(bot.model.device)
            positions = torch.where(input_ids < 0)[0]
            if positions.numel() != 1:
                raise HALPCompatibilityError("Huatuo requires one image placeholder")
            attention = torch.ones_like(input_ids, dtype=torch.bool)
            labels = torch.full_like(input_ids, -100)
            with PreProjectorVisionCapture(bot.model.get_vision_tower()) as vision_capture:
                _, position_ids, attention, _, embeddings, _ = (
                    bot.model.prepare_inputs_labels_for_multimodal_new(
                        [input_ids], None, [attention], None, [labels], image_tensor
                    )
                )
            start = int(positions.item())
            count = int(embeddings.shape[1] - input_ids.numel() + 1)
            visual_span = (start, start + count)
            causal_lm, decoder = bot.model, bot.model.model
            raw_ids = [int(value) for value in input_ids.detach().cpu().tolist()]
        elif family == "hulu":
            scorer = HuluScorer(model_dir, 1024)
            runtime = scorer.runtime
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            inputs = runtime.processor(
                images=[image],
                conversation=conversation,
                add_system_prompt=False,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            for key, value in list(inputs.items()):
                if torch.is_tensor(value):
                    if key == "pixel_values":
                        value = value.to(dtype=runtime.model.dtype)
                    inputs[key] = value.to(runtime.model.device)
            input_ids = inputs["input_ids"]
            mask = input_ids[0].eq(runtime.model.config.image_token_index)
            positions = torch.nonzero(mask, as_tuple=False).flatten()
            if positions.numel() == 0:
                raise HALPCompatibilityError("Hulu prompt has no image token")
            start, end = int(positions.min()), int(positions.max()) + 1
            if not bool(mask[start:end].all()) or int(mask.sum()) != end - start:
                raise HALPCompatibilityError("Hulu visual tokens are not one contiguous span")
            with PreProjectorVisionCapture(runtime.model.get_vision_encoder()) as vision_capture:
                _, attention, position_ids, _, embeddings, _ = (
                    runtime.model.prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids,
                        attention_mask=inputs.get("attention_mask"),
                        position_ids=inputs.get("position_ids"),
                        pixel_values=inputs.get("pixel_values"),
                        grid_sizes=inputs.get("grid_sizes"),
                        merge_sizes=inputs.get("merge_sizes"),
                        modals=inputs.get("modals"),
                    )
                )
            if embeddings.shape[1] != input_ids.shape[1]:
                raise HALPCompatibilityError("Hulu multimodal preparation changed sequence length")
            if attention is None:
                attention = torch.ones_like(input_ids, dtype=torch.bool)
            visual_span = (start, end)
            causal_lm, decoder = runtime.model, runtime.model.model
            raw_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
        else:
            raise HALPCompatibilityError(f"unsupported family: {family}")
        if embeddings is None:
            raise HALPCompatibilityError("multimodal preparation returned no embeddings")
        layer_contract = runtime_layer_contract(causal_lm, audit)
        capture = capture_three_planes(
            decoder_model=decoder,
            embeddings=embeddings,
            attention_mask=attention,
            position_ids=position_ids,
            visual_span=visual_span,
            visual_only_output=vision_capture.one(),
            layers=range(1, audit["decoder_layer_count"] + 1),
        )
    identity = {
        "dicom": file_record(dicom_path),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "signed_input_ids_sha256": canonical_sha256(raw_ids),
        "record_key": "fixed_engineering_image_no_outcome_lookup",
    }
    return write_engineering_capture(
        output_dir=output_dir,
        family=family,
        audit=audit,
        layer_contract=layer_contract,
        capture=capture,
        input_identity=identity,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=("huatuo", "hulu"), required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--huatuo-root", type=Path, default=HUATUO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-source-audit", action="store_true")
    parser.add_argument("--engineering-capture-smoke", action="store_true")
    parser.add_argument("--dicom", type=Path, default=DEFAULT_DICOM)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    args = parser.parse_args()
    if args.cpu_source_audit == args.engineering_capture_smoke:
        raise HALPCompatibilityError("select exactly one execution mode")
    model_dir = args.model_dir or MODEL_DIRS[args.model_family]
    if args.cpu_source_audit:
        result = cpu_source_audit(
            family=args.model_family,
            model_dir=model_dir,
            huatuo_root=args.huatuo_root,
        )
        atomic_json(args.output_dir / "source_audit.json", result)
    else:
        result = real_single_claim_capture(
            family=args.model_family,
            model_dir=model_dir,
            dicom_path=args.dicom,
            prompt=args.prompt,
            output_dir=args.output_dir,
            gpu_lock=args.gpu_lock,
            huatuo_root=args.huatuo_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
