"""Trace acquisition-style orbits through a Qwen2.5-VL model lineage.

This is a mechanism diagnostic, not a prediction method.  It applies the same
content-preserving CXR style views to the exact same images and records pooled
representations at visual, merger, and language-model locations.  A model
lineage is evaluated by swapping only its visual-merger checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from anchor.corrected_sgta.run_center_native_qwen import (
    load_merger,
    merger_parameters,
    messages_for,
)
from anchor.corrected_sgta.run_visual_evidence_chord_probe import (
    build_views,
    read_jsonl,
    sha256,
    unique_cases,
)


VERSION = "layerwise-style-orbit-probe-v1"
DEFAULT_PROMPT = (
    "Analyze this chest radiograph and summarize the visible findings in one "
    "complete sentence."
)
VISION_BLOCKS = (0, 7, 15, 23, 31)
LANGUAGE_BLOCKS = (0, 7, 14, 21, 27)


def parse_variant(specification: str) -> tuple[str, Path | None]:
    """Parse ``name`` or ``name=/path/to/merger.pt``."""
    if "=" not in specification:
        name = specification.strip()
        checkpoint = None
    else:
        name, raw_path = specification.split("=", 1)
        name = name.strip()
        checkpoint = Path(raw_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if not name or any(character.isspace() for character in name):
        raise ValueError(f"invalid variant name: {name!r}")
    return name, checkpoint


def selected_cases(
    questions: Path,
    image_manifest: Path,
    view_audit: Path,
    limit: int,
) -> list[dict[str, Any]]:
    """Return frontal cases from the exact 64-case development lineage."""
    cases = unique_cases(questions, image_manifest, 64)
    category = {
        row["case_id"]: row["predicted_category"] for row in read_jsonl(view_audit)
    }
    frontal = [
        case
        for case in cases
        if category.get(case["case_id"]) == "a frontal chest radiograph"
    ]
    if len(frontal) < limit:
        raise RuntimeError(
            f"requested {limit} frontal cases but only {len(frontal)} are available"
        )
    return frontal[:limit]


def state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in merger_parameters(module)
    }


def restore_merger(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    current = dict(model.named_parameters())
    missing = sorted(set(state) - set(current))
    if missing:
        raise RuntimeError(f"base merger state has missing parameters: {missing}")
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(device=current[name].device))


def output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if not torch.is_tensor(output):
        raise TypeError(f"unsupported hook output: {type(output)!r}")
    return output


class LayerCapture:
    """Pool selected hook outputs without retaining full activation tensors."""

    def __init__(self, model: Qwen2_5_VLForConditionalGeneration) -> None:
        self.model = model
        self.batch_size = 0
        self.input_ids: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.features: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

        visual = model.model.visual
        for index in VISION_BLOCKS:
            self.handles.append(
                visual.blocks[index].register_forward_hook(
                    self._vision_hook(f"vision_block_{index}")
                )
            )
        self.handles.append(
            visual.merger.register_forward_hook(self._vision_hook("merger"))
        )
        for index in LANGUAGE_BLOCKS:
            self.handles.append(
                model.model.language_model.layers[index].register_forward_hook(
                    self._language_hook(index)
                )
            )

    def set_context(self, inputs: dict[str, torch.Tensor]) -> None:
        self.batch_size = int(inputs["input_ids"].shape[0])
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
        self.features = {}

    def _vision_hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output_tensor(output)
            if tensor.ndim != 2 or tensor.shape[0] % self.batch_size:
                raise RuntimeError(
                    f"{name}: cannot split shape {tuple(tensor.shape)} "
                    f"over batch={self.batch_size}"
                )
            pooled = tensor.reshape(
                self.batch_size, -1, tensor.shape[-1]
            ).float().mean(dim=1)
            self.features[name] = pooled.detach().cpu()

        return hook

    def _language_hook(self, index: int):
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            if self.input_ids is None or self.attention_mask is None:
                raise RuntimeError("language hook called without input context")
            hidden = output_tensor(output).float()
            image_mask = self.input_ids.eq(self.model.config.image_token_id)
            image_count = image_mask.sum(dim=1)
            if int(image_count.min()) == 0:
                raise RuntimeError("prompt contains no image token")
            image_pooled = (
                hidden * image_mask.unsqueeze(-1)
            ).sum(dim=1) / image_count.unsqueeze(-1)
            positions = torch.arange(
                hidden.shape[1], device=hidden.device
            ).expand(hidden.shape[0], -1)
            last = positions.masked_fill(
                self.attention_mask.eq(0), -1
            ).max(dim=1).values
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            prompt_state = hidden[batch, last]
            self.features[f"llm_{index}_image"] = image_pooled.detach().cpu()
            self.features[f"llm_{index}_prompt"] = prompt_state.detach().cpu()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def fingerprint_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--style-manifest", type=Path, required=True)
    parser.add_argument("--view-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--radius", type=float, default=0.12)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    variants = [parse_variant(specification) for specification in args.variant]
    if len({name for name, _ in variants}) != len(variants):
        raise ValueError("variant names must be unique")
    cases = selected_cases(
        args.questions, args.image_manifest, args.view_audit, args.limit
    )
    prototypes = [
        row
        for row in read_jsonl(args.style_manifest)
        if int(row["replicate"]) == 0
    ]
    if len(prototypes) < 3:
        raise RuntimeError("style manifest must contain at least three clusters")

    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, use_fast=False
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).to("cuda").eval()
    base_merger = state_dict_cpu(model)
    capture = LayerCapture(model)

    feature_rows: dict[str, list[np.ndarray]] = {}
    row_metadata: list[dict[str, Any]] = []
    variant_metadata = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for variant_index, (variant_name, checkpoint) in enumerate(variants):
            restore_merger(model, base_merger)
            if checkpoint is not None:
                load_merger(model, checkpoint)
            variant_metadata.append(
                {
                    "name": variant_name,
                    "checkpoint": str(checkpoint) if checkpoint else None,
                    "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
                }
            )
            for case_index, case in enumerate(cases):
                views, metrics = build_views(
                    case,
                    prototypes,
                    args.radius,
                    args.strength,
                    args.output,
                )
                names = ["real", "null"] + sorted(
                    name for name in views if name.startswith("style_")
                )
                images = [views[name] for name in names]
                text = processor.apply_chat_template(
                    messages_for(args.prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=[text] * len(images),
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
                capture.set_context(inputs)
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    model(**inputs, use_cache=False, return_dict=True)
                for layer, tensor in capture.features.items():
                    if tensor.shape[0] != len(names):
                        raise RuntimeError(
                            f"{layer}: expected {len(names)} rows, got "
                            f"{tensor.shape[0]}"
                        )
                    feature_rows.setdefault(layer, []).append(
                        tensor.numpy().astype(np.float16)
                    )
                if variant_index == 0:
                    patient_id = case["image_relative"].split("/")[1]
                    for name in names:
                        row_metadata.append(
                            {
                                "case_id": case["case_id"],
                                "patient_id": patient_id,
                                "image_relative": case["image_relative"],
                                "view": name,
                                "image_metrics": metrics[name],
                            }
                        )
                print(
                    json.dumps(
                        {
                            "variant": variant_name,
                            "completed_cases": case_index + 1,
                            "total_cases": len(cases),
                        }
                    ),
                    flush=True,
                )
            torch.cuda.empty_cache()
    finally:
        capture.close()

    arrays = {}
    view_count = 2 + len(prototypes)
    for layer, rows in feature_rows.items():
        array = np.stack(rows)
        arrays[layer] = array.reshape(
            len(variants), len(cases), view_count, array.shape[-1]
        )
    np.savez_compressed(args.output, **arrays)

    provenance = {
        "version": VERSION,
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256(args.model / "config.json"),
        "questions": str(args.questions.resolve()),
        "questions_sha256": sha256(args.questions),
        "image_manifest": str(args.image_manifest.resolve()),
        "image_manifest_sha256": sha256(args.image_manifest),
        "style_manifest": str(args.style_manifest.resolve()),
        "style_manifest_sha256": sha256(args.style_manifest),
        "view_audit": str(args.view_audit.resolve()),
        "view_audit_sha256": sha256(args.view_audit),
        "radius": args.radius,
        "strength": args.strength,
        "prompt": args.prompt,
        "variants": variant_metadata,
    }
    metadata = {
        **provenance,
        "fingerprint": fingerprint_payload(provenance),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "cases": len(cases),
        "views": ["real", "null"]
        + [f"style_{int(row['cluster'])}" for row in prototypes],
        "layers": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for name, array in arrays.items()
        },
        "rows": row_metadata,
        "claim_ceiling": (
            "paired representation diagnostic on exposed MIMIC development "
            "images; no generated-answer utility claim"
        ),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metadata": str(metadata_path),
                "fingerprint": metadata["fingerprint"],
                "layers": list(arrays),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
