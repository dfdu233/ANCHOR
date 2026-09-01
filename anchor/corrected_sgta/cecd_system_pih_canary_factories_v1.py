#!/usr/bin/env python3
"""Adapter-backed frozen-input factories for CECD native/eager canaries.

Importing or describing this module never imports a model adapter, initializes
CUDA, or runs a forward pass.  The four public factory callables are consumed
only by the explicit ``canary`` command in
``cecd_system_pih_runtime_integration_v1``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .cecd_system_pih_runtime_integration_v1 import (
    ExpandedRoleTokenProvenance,
    RuntimeIntegrationError,
    build_huatuo_expanded_provenance,
    build_hulu_expanded_provenance,
    canonical_sha256,
    file_record,
)


VERSION = "cecd-system-pih-canary-factories-v1"
INPUT_IDENTITY_SCHEMA = "cecd-system-pih-frozen-canary-input-v1"
FROZEN_IMAGE = Path(
    "/workspace/vinbigdata/train/d925309691e7929d905eaa42f081833f.dicom"
)
FROZEN_IMAGE_SHA256 = "2893814198d6126656311fe55c08515a86eeb1917535ba273e11d5429a18bec5"
FROZEN_RECORD_KEY = "vindr_train_canary_d925309691e7929d"
FROZEN_FINDING = "aortic_enlargement"
FROZEN_PROMPT = (
    "Does this chest X-ray show aortic enlargement? "
    "Answer with exactly one word: Yes, No, or Maybe."
)
MODEL_PATHS = {
    "huatuo": Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
    "hulu": Path("/home/dbw/models/Hulu-Med-4B"),
}


def _load_frozen_image() -> Any:
    """Load the one frozen DICOM through the already-audited VinDr loader."""

    observed = file_record(FROZEN_IMAGE)
    if observed["sha256"] != FROZEN_IMAGE_SHA256:
        raise RuntimeIntegrationError("frozen canary DICOM identity changed")
    from .run_huatuo_vindr_commitment_probe import load_image

    return load_image(FROZEN_IMAGE)


def _load_huatuo_adapter() -> Any:
    # Deliberately local: module import and describe remain model/GPU free.
    from .models_oe import HuatuoOEAdapter

    return HuatuoOEAdapter(model_path=MODEL_PATHS["huatuo"])


def _load_hulu_adapter() -> Any:
    # Deliberately local: module import and describe remain model/GPU free.
    from .models import HuluAdapter

    return HuluAdapter(model_path=MODEL_PATHS["hulu"])


def _attach_adapter(model: torch.nn.Module, *, family: str, adapter: Any) -> torch.nn.Module:
    if not isinstance(model, torch.nn.Module):
        raise RuntimeIntegrationError(f"{family} adapter did not expose a torch model")
    model.eval()
    # The adapter retains the tokenizer/processor/native image pipeline.  It is
    # intentionally attached only to this process-local model instance.
    setattr(model, "_cecd_canary_adapter", adapter)
    setattr(model, "_cecd_canary_family", family)
    return model


def _adapter_for(model: torch.nn.Module, family: str) -> Any:
    if getattr(model, "_cecd_canary_family", None) != family:
        raise RuntimeIntegrationError(f"model is not bound to the {family} canary adapter")
    adapter = getattr(model, "_cecd_canary_adapter", None)
    if adapter is None or getattr(adapter, "model", None) is not model:
        raise RuntimeIntegrationError(f"{family} canary adapter/model identity is broken")
    return adapter


def huatuo_model_factory() -> torch.nn.Module:
    adapter = _load_huatuo_adapter()
    return _attach_adapter(adapter.model, family="huatuo", adapter=adapter)


def hulu_model_factory() -> torch.nn.Module:
    adapter = _load_hulu_adapter()
    return _attach_adapter(adapter.model, family="hulu", adapter=adapter)


def _as_batch_one(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        raise RuntimeIntegrationError(f"{name} must be batch-one rank two")
    return tensor


def _expanded_attention(
    attention: torch.Tensor | None, embeddings: torch.Tensor
) -> torch.Tensor:
    if attention is None:
        return torch.ones(
            embeddings.shape[:2], dtype=torch.bool, device=embeddings.device
        )
    attention = _as_batch_one(attention, name="expanded attention mask")
    if tuple(attention.shape) != tuple(embeddings.shape[:2]):
        raise RuntimeIntegrationError("expanded attention/embedding shape mismatch")
    return attention


def _input_identity(
    *,
    family: str,
    raw_input_ids: list[int],
    embeddings: torch.Tensor,
    provenance: ExpandedRoleTokenProvenance,
) -> dict[str, Any]:
    payload = {
        "schema_version": INPUT_IDENTITY_SCHEMA,
        "model_family": family,
        "record_key": FROZEN_RECORD_KEY,
        "finding": FROZEN_FINDING,
        "image": {
            "path": str(FROZEN_IMAGE),
            "sha256": FROZEN_IMAGE_SHA256,
        },
        "prompt_sha256": hashlib.sha256(FROZEN_PROMPT.encode("utf-8")).hexdigest(),
        "signed_input_ids_sha256": canonical_sha256(raw_input_ids),
        "raw_token_count": len(raw_input_ids),
        "expanded_token_count": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "provenance_fingerprint": provenance.fingerprint,
        "cache_free": True,
    }
    payload["fingerprint"] = canonical_sha256(payload)
    return payload


def _forward_bundle(
    *,
    family: str,
    raw_input_ids: list[int],
    embeddings: torch.Tensor,
    attention: torch.Tensor,
    position_ids: torch.Tensor | None,
    provenance: ExpandedRoleTokenProvenance,
) -> dict[str, Any]:
    if embeddings.ndim != 3 or embeddings.shape[0] != 1:
        raise RuntimeIntegrationError("multimodal embeddings must be batch-one rank three")
    if embeddings.shape[1] != len(provenance.role_provenance):
        raise RuntimeIntegrationError("expanded embeddings/provenance length mismatch")
    observed_mask = tuple(int(bool(value)) for value in attention[0].detach().cpu().tolist())
    if observed_mask != provenance.attention_mask:
        raise RuntimeIntegrationError("expanded attention differs from exact provenance")
    forward_kwargs: dict[str, Any] = {
        "input_ids": None,
        "inputs_embeds": embeddings,
        "attention_mask": attention,
        "use_cache": False,
        "return_dict": True,
    }
    if position_ids is not None:
        position_ids = _as_batch_one(position_ids, name="position IDs")
        if position_ids.shape[-1] != embeddings.shape[1]:
            raise RuntimeIntegrationError("position IDs/embedding length mismatch")
        forward_kwargs["position_ids"] = position_ids
    return {
        "provenance": provenance,
        "forward_kwargs": forward_kwargs,
        "input_identity": _input_identity(
            family=family,
            raw_input_ids=raw_input_ids,
            embeddings=embeddings,
            provenance=provenance,
        ),
    }


@torch.inference_mode()
def _huatuo_input_factory_impl(model: torch.nn.Module) -> Mapping[str, Any]:
    """Expand the native Huatuo prompt/image pair once, without KV cache."""

    adapter = _adapter_for(model, "huatuo")
    image = _load_frozen_image()
    input_ids, image_tensors = adapter._inputs(image, FROZEN_PROMPT)
    input_ids = _as_batch_one(input_ids, name="Huatuo input IDs")
    raw = input_ids[0]
    image_positions = torch.where(raw < 0)[0]
    if int(image_positions.numel()) != 1:
        raise RuntimeIntegrationError("Huatuo input must contain exactly one image placeholder")
    placeholder_index = int(image_positions[0].item())
    placeholder_id = int(raw[placeholder_index].item())
    raw_attention = torch.ones_like(raw, dtype=torch.bool)
    labels = torch.full_like(raw, -100)
    _, position_ids, attention, _, embeddings, _ = (
        model.prepare_inputs_labels_for_multimodal_new(
            [raw], None, [raw_attention], None, [labels], image_tensors
        )
    )
    if not torch.is_tensor(embeddings):
        raise RuntimeIntegrationError("Huatuo multimodal preparation returned no embeddings")
    attention = _expanded_attention(attention, embeddings)
    raw_ids = [int(value) for value in raw.detach().cpu().tolist()]
    projected_visual_tokens = int(embeddings.shape[1]) - len(raw_ids) + 1
    roles = ["user_text"] * len(raw_ids)
    roles[placeholder_index] = "image"
    provenance = build_huatuo_expanded_provenance(
        input_ids=raw_ids,
        token_roles=roles,
        image_placeholder_id=placeholder_id,
        projected_visual_token_count=projected_visual_tokens,
        attention_mask=[int(value) for value in raw_attention.detach().cpu().tolist()],
        frozen_prefix_length=int(embeddings.shape[1]),
    )
    return _forward_bundle(
        family="huatuo",
        raw_input_ids=raw_ids,
        embeddings=embeddings,
        attention=attention,
        position_ids=position_ids,
        provenance=provenance,
    )


def huatuo_input_factory(model: torch.nn.Module) -> Mapping[str, Any]:
    """Source-backed public wrapper around the inference-only implementation."""

    return _huatuo_input_factory_impl(model)


@torch.inference_mode()
def _hulu_input_factory_impl(model: torch.nn.Module) -> Mapping[str, Any]:
    """Materialize Hulu's native processor image-token run without KV cache."""

    adapter = _adapter_for(model, "hulu")
    image = _load_frozen_image()
    inputs = adapter._inputs(image, FROZEN_PROMPT)
    input_ids = _as_batch_one(inputs["input_ids"], name="Hulu input IDs")
    image_token_id = int(model.config.image_token_index)
    image_mask = input_ids[0].eq(image_token_id)
    image_positions = torch.where(image_mask)[0]
    if int(image_positions.numel()) <= 0:
        raise RuntimeIntegrationError("Hulu input contains no image tokens")
    observed_positions = tuple(int(value) for value in image_positions.detach().cpu().tolist())
    if observed_positions != tuple(range(observed_positions[0], observed_positions[-1] + 1)):
        raise RuntimeIntegrationError("Hulu image-token run is not contiguous")
    _, attention, position_ids, _, embeddings, _ = (
        model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            position_ids=inputs.get("position_ids"),
            pixel_values=inputs.get("pixel_values"),
            grid_sizes=inputs.get("grid_sizes"),
            merge_sizes=inputs.get("merge_sizes"),
            modals=inputs.get("modals"),
        )
    )
    if not torch.is_tensor(embeddings):
        raise RuntimeIntegrationError("Hulu multimodal preparation returned no embeddings")
    if int(embeddings.shape[1]) != int(input_ids.shape[1]):
        raise RuntimeIntegrationError(
            "Hulu token compression changed sequence length; exact provenance is unavailable"
        )
    attention = _expanded_attention(attention, embeddings)
    raw_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    roles = ["image" if value == image_token_id else "user_text" for value in raw_ids]
    provenance = build_hulu_expanded_provenance(
        expanded_input_ids=raw_ids,
        token_roles=roles,
        image_token_id=image_token_id,
        attention_mask=[int(bool(value)) for value in attention[0].detach().cpu().tolist()],
        frozen_prefix_length=int(embeddings.shape[1]),
    )
    return _forward_bundle(
        family="hulu",
        raw_input_ids=raw_ids,
        embeddings=embeddings,
        attention=attention,
        position_ids=position_ids,
        provenance=provenance,
    )


def hulu_input_factory(model: torch.nn.Module) -> Mapping[str, Any]:
    """Source-backed public wrapper around the inference-only implementation."""

    return _hulu_input_factory_impl(model)


def factory_description() -> dict[str, Any]:
    """Return source/sample bindings without importing adapters or touching CUDA."""

    image = file_record(FROZEN_IMAGE)
    if image["sha256"] != FROZEN_IMAGE_SHA256:
        raise RuntimeIntegrationError("frozen canary DICOM identity changed")
    payload = {
        "version": VERSION,
        "status": "source_ready_explicit_canary_pending",
        "model_loaded": False,
        "gpu_touched": False,
        "canary_run": False,
        "frozen_input": {
            "record_key": FROZEN_RECORD_KEY,
            "finding": FROZEN_FINDING,
            "image": image,
            "prompt": FROZEN_PROMPT,
            "prompt_sha256": hashlib.sha256(FROZEN_PROMPT.encode("utf-8")).hexdigest(),
        },
        "factories": {
            "huatuo": {
                "model": f"{__name__}:huatuo_model_factory",
                "input": f"{__name__}:huatuo_input_factory",
            },
            "hulu": {
                "model": f"{__name__}:hulu_model_factory",
                "input": f"{__name__}:hulu_input_factory",
            },
        },
        "input_contract": (
            "native adapter preprocessing; already-expanded batch-one inputs_embeds; "
            "exact role provenance; use_cache=false"
        ),
    }
    payload["fingerprint"] = canonical_sha256(payload)
    return payload


__all__ = [
    "factory_description",
    "huatuo_model_factory",
    "huatuo_input_factory",
    "hulu_model_factory",
    "hulu_input_factory",
]
