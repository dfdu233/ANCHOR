"""Run OPA-DPO's released evaluator with fail-closed conformance evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

import torch


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def import_entry(path: Path):
    spec = importlib.util.spec_from_file_location("released_opa_model_vqa", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import OPA-DPO entry: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-entry", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--image-folder", required=True, type=Path)
    parser.add_argument("--question-file", required=True, type=Path)
    parser.add_argument("--answers-file", required=True, type=Path)
    parser.add_argument("--evidence-file", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import peft
    import transformers
    from transformers import AutoConfig, CLIPImageProcessor

    original_auto_config_register = AutoConfig.register
    if "exist_ok" not in inspect.signature(original_auto_config_register).parameters:
        def register_config_431(model_type, config, exist_ok=False):
            return original_auto_config_register(model_type, config)

        AutoConfig.register = register_config_431

    entry = import_entry(args.official_entry.resolve())
    prompt_rows: list[dict[str, Any]] = []
    pixel_rows: list[dict[str, Any]] = []
    peft_evidence: dict[str, Any] = {
        "called": False,
        "checkpoint": None,
        "lora_tensor_count": 0,
        "lora_numel": 0,
        "sampled_nonzero_delta_count": 0,
    }

    original_tokenizer_image_token = entry.tokenizer_image_token

    def audited_tokenizer_image_token(*values, **kwargs):
        result = original_tokenizer_image_token(*values, **kwargs)
        ids = result.tolist()
        prompt_rows.append(
            {
                "prompt_token_ids": ids,
                "prompt_token_ids_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
        return result

    entry.tokenizer_image_token = audited_tokenizer_image_token
    original_preprocess = CLIPImageProcessor.preprocess

    def audited_preprocess(processor, images, *values, **kwargs):
        result = original_preprocess(processor, images, *values, **kwargs)
        pixels = result.get("pixel_values")
        if isinstance(pixels, torch.Tensor) and pixels.ndim == 4 and pixels.shape[0] == 1:
            pixel_rows.append(
                {
                    "pixel_tensor_shape": list(pixels[0].shape),
                    "pixel_tensor_dtype": str(pixels[0].dtype),
                    "pixel_tensor_sha256": tensor_digest(pixels[0]),
                }
            )
        return result

    CLIPImageProcessor.preprocess = audited_preprocess

    original_peft_from_pretrained = entry.PeftModel.from_pretrained.__func__

    def audited_peft_from_pretrained(cls, model, model_id, *values, **kwargs):
        result = original_peft_from_pretrained(
            cls, model, model_id, *values, **kwargs
        )
        resolved = Path(model_id).resolve()
        if resolved != args.adapter_path.resolve():
            raise RuntimeError(f"OPA-DPO loaded {resolved}, expected {args.adapter_path.resolve()}")
        tensors = [
            parameter
            for name, parameter in result.named_parameters()
            if "lora_A" in name or "lora_B" in name
        ]
        nonzero = 0
        for _, module in result.named_modules():
            if not hasattr(module, "lora_A") or "default" not in module.lora_A:
                continue
            a = module.lora_A["default"].weight.detach().float()
            b = module.lora_B["default"].weight.detach().float()
            width = min(8, a.shape[1], b.shape[0])
            for index in range(width):
                nonzero += int(torch.dot(b[index], a[:, index]).item() != 0.0)
        peft_evidence.update(
            {
                "called": True,
                "checkpoint": str(resolved),
                "adapter_config_sha256": digest(resolved / "adapter_config.json"),
                "lora_tensor_count": len(tensors),
                "lora_numel": sum(parameter.numel() for parameter in tensors),
                "sampled_nonzero_delta_count": nonzero,
            }
        )
        if not tensors or nonzero == 0:
            raise RuntimeError("OPA-DPO adapter has no demonstrably nonzero LoRA delta")
        return result

    entry.PeftModel.from_pretrained = classmethod(audited_peft_from_pretrained)
    args.answers_file.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_file.parent.mkdir(parents=True, exist_ok=True)
    official_args = argparse.Namespace(
        model_path=str(args.model_path),
        model_base=None,
        image_folder=str(args.image_folder),
        question_file=str(args.question_file),
        answers_file=str(args.answers_file),
        conv_mode="llava_v1",
        num_chunks=1,
        chunk_idx=0,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        use_qlora=True,
        qlora_path=str(args.adapter_path),
        short_eval=False,
        max_new_tokens=args.max_new_tokens,
        image_aspect_ratio="pad",
        test_prompt="",
    )
    try:
        entry.eval_model(official_args)
    finally:
        entry.PeftModel.from_pretrained = classmethod(original_peft_from_pretrained)
        CLIPImageProcessor.preprocess = original_preprocess
    if not peft_evidence["called"]:
        raise RuntimeError("OPA-DPO evaluator silently fell back to the base model")
    rows = [json.loads(line) for line in args.answers_file.read_text().splitlines() if line]
    if not (len(rows) == len(prompt_rows) == len(pixel_rows)):
        raise RuntimeError(
            f"OPA evidence cardinality mismatch: answers={len(rows)}, "
            f"prompts={len(prompt_rows)}, pixels={len(pixel_rows)}"
        )
    inputs = []
    for row, prompt, pixels in zip(rows, prompt_rows, pixel_rows):
        inputs.append({"question_id": str(row["question_id"]), **prompt, **pixels})
    import llava

    loaded_model_module = __import__(
        entry.LlavaLlamaForCausalLM.__module__, fromlist=["__name__"]
    )
    loaded_model_source = Path(inspect.getfile(loaded_model_module)).resolve()
    loading_source = {
        "builder_source": str(args.official_entry.resolve()),
        "builder_source_sha256": digest(args.official_entry.resolve()),
        "loaded_model_source": str(loaded_model_source),
        "loaded_model_source_sha256": digest(loaded_model_source),
    }

    result = {
        "protocol": "trained-opa-official-evidence-v3",
        "variant": "opa-dpo",
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "transformers_path": str(Path(transformers.__file__).resolve()),
            "peft": peft.__version__,
            "llava_path": str(Path(llava.__file__).resolve()),
        },
        "official_entry": str(args.official_entry.resolve()),
        "official_entry_sha256": digest(args.official_entry.resolve()),
        "loading_ledger": {
            "variant": "opa-dpo",
            "model_path": str(args.model_path.resolve()),
            "adapter_path": str(args.adapter_path.resolve()),
            "compute_dtype": "bfloat16",
            "image_aspect_ratio": "pad",
            "peft_merged": False,
        },
        "loading_source": loading_source,
        "peft_evidence": peft_evidence,
        "inputs": inputs,
        "answers_sha256": digest(args.answers_file),
    }
    if transformers.__version__ != "4.31.0":
        raise RuntimeError(f"expected transformers 4.31.0, got {transformers.__version__}")
    args.evidence_file.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
