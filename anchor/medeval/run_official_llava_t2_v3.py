"""Run the released LLaVA evaluator with fail-closed loading evidence.

The released evaluator is imported from ``--official-entry`` and remains the
generation implementation.  This wrapper only (1) records the tensors handed
to it and (2) rejects an adapter load unless PEFT and non-LoRA weights can be
shown to be active.  It never substitutes a base-model answer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def normalize_non_lora(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {
        (key[11:] if key.startswith("base_model.") else key): value
        for key, value in state.items()
    }
    if any(key.startswith("model.model.") for key in normalized):
        normalized = {
            (key[6:] if key.startswith("model.") else key): value
            for key, value in normalized.items()
        }
    return normalized


def import_entry(path: Path):
    spec = importlib.util.spec_from_file_location("released_llava_model_vqa_loader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official entry: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--official-entry", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-base", type=Path)
    parser.add_argument("--image-folder", required=True, type=Path)
    parser.add_argument("--question-file", required=True, type=Path)
    parser.add_argument("--answers-file", required=True, type=Path)
    parser.add_argument("--evidence-file", required=True, type=Path)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import peft
    import transformers

    entry = import_entry(args.official_entry.resolve())
    original_loader = entry.load_pretrained_model
    evidence_rows: list[dict[str, Any]] = []
    peft_evidence: dict[str, Any] = {
        "called": False,
        "checkpoint": None,
        "lora_tensor_count": 0,
        "lora_numel": 0,
        "sampled_nonzero_delta_count": 0,
    }

    from peft import PeftModel

    original_peft_from_pretrained = PeftModel.from_pretrained.__func__

    def audited_peft_from_pretrained(cls, model, model_id, *values, **kwargs):
        result = original_peft_from_pretrained(
            cls, model, model_id, *values, **kwargs
        )
        resolved = Path(model_id).resolve()
        if resolved != args.model_path.resolve():
            raise RuntimeError(
                f"PEFT loaded {resolved}, expected {args.model_path.resolve()}"
            )
        tensors = [
            (name, parameter)
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
                delta = torch.dot(b[index], a[:, index]).item()
                nonzero += int(delta != 0.0)
        peft_evidence.update(
            {
                "called": True,
                "checkpoint": str(resolved),
                "adapter_config_sha256": digest(resolved / "adapter_config.json"),
                "lora_tensor_count": len(tensors),
                "lora_numel": sum(parameter.numel() for _, parameter in tensors),
                "sampled_nonzero_delta_count": nonzero,
            }
        )
        if not tensors or nonzero == 0:
            raise RuntimeError("adapter loaded without a demonstrably nonzero LoRA delta")
        return result

    PeftModel.from_pretrained = classmethod(audited_peft_from_pretrained)

    loading_ledger: dict[str, Any] = {}
    decoded_token_rows: list[list[int]] = []

    def audited_loader(model_path, model_base, model_name, *values, **kwargs):
        model_path_resolved = Path(model_path).resolve()
        if model_path_resolved != args.model_path.resolve():
            raise RuntimeError("official entry changed the requested checkpoint")
        if args.model_base is not None and "lora" not in model_name.lower():
            raise RuntimeError(
                "adapter basename does not select the released LoRA loading branch"
            )
        tokenizer, model, processor, context = original_loader(
            model_path, model_base, model_name, *values, **kwargs
        )
        non_lora_path = args.model_path.resolve() / "non_lora_trainables.bin"
        non_lora_record: dict[str, Any] = {
            "present": non_lora_path.is_file(),
            "checkpoint_keys": 0,
            "matched_keys": 0,
            "shape_mismatches": [],
            "value_mismatches": [],
        }
        if non_lora_path.is_file():
            state = normalize_non_lora(torch.load(non_lora_path, map_location="cpu"))
            current = model.state_dict()
            non_lora_record["checkpoint_keys"] = len(state)
            for key, value in state.items():
                if key not in current:
                    non_lora_record["shape_mismatches"].append([key, "missing"])
                    continue
                if tuple(current[key].shape) != tuple(value.shape):
                    non_lora_record["shape_mismatches"].append(
                        [key, list(value.shape), list(current[key].shape)]
                    )
                    continue
                non_lora_record["matched_keys"] += 1
                observed = current[key].detach().cpu()
                expected = value.to(dtype=observed.dtype)
                if not torch.equal(observed, expected):
                    non_lora_record["value_mismatches"].append(key)
            if (
                non_lora_record["matched_keys"] != len(state)
                or non_lora_record["shape_mismatches"]
                or non_lora_record["value_mismatches"]
            ):
                raise RuntimeError(f"non-LoRA checkpoint was not loaded exactly: {non_lora_record}")
        loading_ledger.update(
            {
                "variant": args.variant,
                "model_path": str(args.model_path.resolve()),
                "model_base": str(args.model_base.resolve()) if args.model_base else None,
                "model_name_seen_by_official_builder": model_name,
                "model_class": f"{type(model).__module__}.{type(model).__name__}",
                "config_model_type": model.config.model_type,
                "image_aspect_ratio": getattr(model.config, "image_aspect_ratio", None),
                "non_lora": non_lora_record,
            }
        )
        original_batch_decode = tokenizer.batch_decode

        def audited_batch_decode(sequences, *decode_args, **decode_kwargs):
            first = sequences[0]
            decoded_token_rows.append(
                first.detach().cpu().tolist()
                if isinstance(first, torch.Tensor)
                else list(first)
            )
            return original_batch_decode(sequences, *decode_args, **decode_kwargs)

        tokenizer.batch_decode = audited_batch_decode
        return tokenizer, model, processor, context

    entry.load_pretrained_model = audited_loader

    original_getitem = entry.CustomDataset.__getitem__

    def audited_getitem(dataset, index):
        item = original_getitem(dataset, index)
        input_ids, image_tensor = item[0], item[1]
        evidence_rows.append(
            {
                "index": index,
                "question_id": str(dataset.questions[index]["question_id"]),
                "prompt_token_ids": input_ids.tolist(),
                "prompt_token_ids_sha256": hashlib.sha256(
                    json.dumps(input_ids.tolist(), separators=(",", ":")).encode()
                ).hexdigest(),
                "pixel_tensor_shape": list(image_tensor.shape),
                "pixel_tensor_dtype": str(image_tensor.dtype),
                "pixel_tensor_sha256": tensor_digest(image_tensor),
            }
        )
        return item

    entry.CustomDataset.__getitem__ = audited_getitem

    def deterministic_loader(
        questions, image_folder, tokenizer, image_processor, model_config,
        batch_size=1, num_workers=0,
    ):
        if batch_size != 1:
            raise RuntimeError("official conformance requires batch size one")
        dataset = entry.CustomDataset(
            questions, image_folder, tokenizer, image_processor, model_config
        )
        collate_fn = getattr(entry, "collate_fn", None)
        return DataLoader(
            dataset, batch_size=1, num_workers=0, shuffle=False,
            collate_fn=collate_fn,
        )

    entry.create_data_loader = deterministic_loader
    args.answers_file.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_file.parent.mkdir(parents=True, exist_ok=True)
    official_args = argparse.Namespace(
        model_path=str(args.model_path),
        model_base=str(args.model_base) if args.model_base else None,
        image_folder=str(args.image_folder),
        question_file=str(args.question_file),
        answers_file=str(args.answers_file),
        conv_mode=args.conv_mode,
        num_chunks=1,
        chunk_idx=0,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
    )
    entry.args = official_args
    try:
        entry.eval_model(official_args)
    finally:
        PeftModel.from_pretrained = classmethod(original_peft_from_pretrained)

    adapter_expected = args.model_base is not None
    if adapter_expected and not peft_evidence["called"]:
        raise RuntimeError("official evaluator silently fell back to the base model")
    if len(decoded_token_rows) != len(evidence_rows):
        raise RuntimeError(
            f"decoded-token evidence mismatch: tokens={len(decoded_token_rows)}, "
            f"inputs={len(evidence_rows)}"
        )
    for item, token_ids in zip(evidence_rows, decoded_token_rows):
        item["generated_token_ids"] = token_ids
    import llava

    builder_module = __import__("llava.model.builder", fromlist=["__name__"])
    builder_source = Path(inspect.getfile(builder_module)).resolve()
    loaded_model_module = __import__(
        loading_ledger["model_class"].rsplit(".", 1)[0], fromlist=["__name__"]
    )
    loaded_model_source = Path(inspect.getfile(loaded_model_module)).resolve()
    source_ledger = {
        "builder_source": str(builder_source),
        "builder_source_sha256": digest(builder_source),
        "loaded_model_source": str(loaded_model_source),
        "loaded_model_source_sha256": digest(loaded_model_source),
    }
    release_loading_contract = None
    if args.variant == "sentinel":
        checkpoint = args.model_path.resolve()
        readme = checkpoint / "README.md"
        config_path = checkpoint / "config.json"
        adapter_config_path = checkpoint / "adapter_config.json"
        lines = readme.read_text().splitlines()
        required_phrases = (
            "library_name: transformers",
            "This model is a PEFT (LoRA) adapter.",
            "You first need to load the base model",
            "Please follow the official repo of [LLaVA]",
        )
        citations = []
        for phrase in required_phrases:
            matches = [
                {"line": number, "text": line.strip()}
                for number, line in enumerate(lines, start=1)
                if phrase in line
            ]
            if not matches:
                raise RuntimeError(f"SENTINEL release instruction missing: {phrase}")
            citations.extend(matches)
        released_config = json.loads(config_path.read_text())
        released_adapter = json.loads(adapter_config_path.read_text())
        release_loading_contract = {
            "reason": "released model card requires standard LLaVA PEFT base-plus-adapter loading",
            "readme": str(readme),
            "readme_sha256": digest(readme),
            "instruction_lines": citations,
            "config": str(config_path),
            "config_sha256": digest(config_path),
            "config_model_type": released_config.get("model_type"),
            "config_transformers_version": released_config.get("transformers_version"),
            "adapter_config": str(adapter_config_path),
            "adapter_config_sha256": digest(adapter_config_path),
            "adapter_base_model_name_or_path": released_adapter.get("base_model_name_or_path"),
            "adapter_peft_type": released_adapter.get("peft_type"),
            "selected_standard_llava_root": str(Path(llava.__file__).resolve().parents[1]),
            **source_ledger,
        }

    result = {
        "protocol": "trained-llava-official-evidence-v3",
        "variant": args.variant,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "transformers_path": str(Path(transformers.__file__).resolve()),
            "peft": peft.__version__,
            "llava_path": str(Path(llava.__file__).resolve()),
        },
        "official_entry": str(args.official_entry.resolve()),
        "official_entry_sha256": digest(args.official_entry.resolve()),
        "loading_ledger": loading_ledger,
        "loading_source": source_ledger,
        "release_loading_contract": release_loading_contract,
        "peft_evidence": peft_evidence,
        "inputs": evidence_rows,
        "answers_sha256": digest(args.answers_file),
    }
    if transformers.__version__ != "4.31.0":
        raise RuntimeError(f"expected transformers 4.31.0, got {transformers.__version__}")
    args.evidence_file.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
