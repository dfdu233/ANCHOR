"""Evaluate released LLaVA-1.5 training-based hallucination mitigators.

Every adapter is compared with the exact same LLaVA-1.5-7B base, prompt
serialization, image processor, and deterministic decoding contract.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
import transformers
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _install_legacy_transformers_mask_helpers() -> None:
    """Restore private mask helpers removed after the released LLaVA fork.

    HA-DPO's vendored MPT prefix-LM converter imports the private BLOOM/OPT
    helpers that existed in Transformers 4.28--4.31.  The current shared
    runtime (4.37+) moved the implementations to ``modeling_attn_mask_utils``
    and removed those module-level names.  Reintroduce only the exact legacy
    signatures/semantics needed by that converter; this does not alter any
    public Transformers class or model weights.
    """

    from transformers.models.bloom import modeling_bloom
    from transformers.models.opt import modeling_opt

    if not hasattr(modeling_bloom, "_make_causal_mask"):
        def _make_causal_mask_bloom(
            input_ids_shape: torch.Size,
            device: torch.device,
            past_key_values_length: int,
        ) -> torch.BoolTensor:
            batch_size, target_length = input_ids_shape
            mask = torch.empty(
                (target_length, target_length + past_key_values_length),
                dtype=torch.bool,
                device=device,
            )
            seq_ids = torch.arange(target_length, device=device)
            mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]
            if past_key_values_length > 0:
                mask[:, :past_key_values_length] = False
            return mask[None, None, :, :].expand(
                batch_size, 1, target_length, target_length + past_key_values_length
            )

        modeling_bloom._make_causal_mask = _make_causal_mask_bloom

    if not hasattr(modeling_bloom, "_expand_mask"):
        def _expand_mask_bloom(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
            batch_size, src_length = mask.shape
            expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
            return expanded_mask.expand(batch_size, 1, tgt_length, src_length)

        modeling_bloom._expand_mask = _expand_mask_bloom

    if not hasattr(modeling_opt, "_expand_mask"):
        def _expand_mask_opt(
            mask: torch.Tensor,
            dtype: torch.dtype,
            tgt_len: int | None = None,
        ) -> torch.Tensor:
            batch_size, src_length = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_length
            expanded_mask = mask[:, None, None, :].expand(
                batch_size, 1, tgt_len, src_length
            ).to(dtype)
            inverted_mask = 1.0 - expanded_mask
            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(dtype).min
            )

        modeling_opt._expand_mask = _expand_mask_opt

    if not hasattr(modeling_opt, "_make_causal_mask"):
        def _make_causal_mask_opt(
            input_ids_shape: torch.Size,
            dtype: torch.dtype,
            device: torch.device,
            past_key_values_length: int = 0,
        ) -> torch.Tensor:
            batch_size, target_length = input_ids_shape
            mask = torch.full(
                (target_length, target_length),
                torch.finfo(dtype).min,
                device=device,
            )
            mask_cond = torch.arange(mask.size(-1), device=device)
            mask.masked_fill_(
                mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0
            )
            mask = mask.to(dtype)
            if past_key_values_length > 0:
                mask = torch.cat(
                    [
                        torch.zeros(
                            target_length,
                            past_key_values_length,
                            dtype=dtype,
                            device=device,
                        ),
                        mask,
                    ],
                    dim=-1,
                )
            return mask[None, None, :, :].expand(
                batch_size,
                1,
                target_length,
                target_length + past_key_values_length,
            )

        modeling_opt._make_causal_mask = _make_causal_mask_opt


_install_legacy_transformers_mask_helpers()


def _requested_variant() -> str:
    try:
        return sys.argv[sys.argv.index("--variant") + 1]
    except (ValueError, IndexError):
        return "base"


_VARIANT = _requested_variant()
if _VARIANT == "da-dpo":
    LLAVA_ROOT = Path("/home/dbw/ANCHOR/third_party/training_baselines/DA-DPO")
elif _VARIANT == "opa-dpo":
    LLAVA_ROOT = Path(
        "/home/dbw/ANCHOR/third_party/training_baselines/OPA-DPO/llava_setup/LLaVA"
    )
elif _VARIANT == "sentinel":
    # SENTINEL's released checkpoint uses the current official LLaVA
    # ``llava_llama`` registry/config (not the older ``llava`` registry used
    # by HA-DPO).  The model card explicitly directs users to official LLaVA
    # PEFT inference, so preserve that checkpoint identity without rewriting
    # its config.
    LLAVA_ROOT = Path(
        "/home/dbw/ANCHOR/third_party/MedHEval/code/baselines/Med-LVLMs/llava_1.6/LLaVA"
    )
else:
    LLAVA_ROOT = Path(
        "/home/dbw/ANCHOR/third_party/training_baselines/HA-DPO/ha_dpo/models/llava-v1_5"
    )
sys.path.insert(0, str(LLAVA_ROOT))
if _VARIANT == "opa-dpo":
    sys.path.insert(
        1, "/home/dbw/ANCHOR/third_party/training_baselines/OPA-DPO"
    )

# Transformers >=4.36 ships its own class under the ``llava`` registry key.
# The released LLaVA-1.5 fork registers its checkpoint-compatible class under
# that same key at import time.  Permit the intentional local override; this
# affects only this isolated baseline process and avoids patching vendor code.
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

_auto_config_register = AutoConfig.register
_auto_model_register = AutoModelForCausalLM.register

# Transformers 4.31 (the released LLaVA-1.5 environment) has no built-in
# ``llava`` registry entry and must retain the original two-argument API.
# Newer Transformers versions need an explicit local override.  Do not make
# the 4.31 conformance path depend on a compatibility behavior it never used.
if "exist_ok" in inspect.signature(_auto_config_register).parameters:
    def _register_config(model_type, config, exist_ok=False):
        return _auto_config_register(model_type, config, exist_ok=True)

    def _register_model(config_class, model_class, exist_ok=False):
        return _auto_model_register(config_class, model_class, exist_ok=True)

    AutoConfig.register = _register_config
    AutoModelForCausalLM.register = _register_model
elif _VARIANT == "opa-dpo":
    # OPA-DPO's released LLaVA patch passes ``exist_ok=True`` even though its
    # pinned Transformers/LLaVA API is otherwise compatible with 4.31.  Make
    # that keyword a no-op in the isolated OPA process; the 4.31 registry has
    # no pre-existing ``llava`` entry, so no override or model substitution is
    # performed.
    def _register_config_431(model_type, config, exist_ok=False):
        return _auto_config_register(model_type, config)

    AutoConfig.register = _register_config_431

from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model import LlavaLlamaForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


VERSION = "trained-llava-baseline-v1"
BASE = Path("/home/dbw/models/llava-v1.5-7b")
VARIANTS = {
    "base": None,
    "factmm-rag-generator": Path("/home/dbw/models/factmm-rag-generator-v1"),
    "ha-dpo": Path("/home/dbw/models/hadpo-llava-1.5"),
    "opa-dpo": Path("/home/dbw/models/opadpo-lora-llava-v1.5-7b"),
    "da-dpo": Path("/home/dbw/models/da-dpo-llava-v1.5-7b"),
    "sentinel": Path("/home/dbw/models/sentinel-llava-v1.5-7b"),
    "less-is-more": Path("/home/dbw/models/less-is-more-llava-v1.5-7b"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(variant: str):
    full_model_path = VARIANTS[variant] if variant == "factmm-rag-generator" else BASE
    compute_dtype = torch.bfloat16 if variant == "opa-dpo" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    config_source = (
        VARIANTS[variant]
        if VARIANTS[variant] is not None and (VARIANTS[variant] / "config.json").is_file()
        else full_model_path
    )
    model_config = LlavaLlamaForCausalLM.config_class.from_pretrained(config_source)
    model = LlavaLlamaForCausalLM.from_pretrained(
        full_model_path,
        low_cpu_mem_usage=True,
        torch_dtype=compute_dtype,
        device_map="auto",
        config=model_config,
    )
    vision_tower = model.get_vision_tower()
    # The exact OpenAI CLIP weights are already materialized locally (and are
    # byte-identical to the cached Hub blob), but the shared offline Hub
    # index also contains an incomplete TensorFlow download.  Resolve the
    # canonical model id to the verified local snapshot so an offline run
    # cannot select the incomplete blob or attempt a network fetch.
    vision_tower_source = str(getattr(vision_tower, "vision_tower_name", ""))
    local_clip = Path(
        "/home/dbw/models/HuatuoGPT-Vision-7B/vit/clip_vit_large_patch14_336"
    )
    if (
        "clip-vit-large-patch14-336" in vision_tower_source
        and (local_clip / "config.json").is_file()
        and (local_clip / "pytorch_model.bin").is_file()
    ):
        vision_tower.vision_tower_name = str(local_clip)
        vision_tower_source = str(local_clip)
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=model.device, dtype=compute_dtype)
    image_processor = vision_tower.image_processor

    ledger = {
        "variant": variant,
        "llava_source_root": str(LLAVA_ROOT.resolve()),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "transformers_path": str(Path(transformers.__file__).resolve()),
        },
        "config_source": str(Path(config_source).resolve()),
        "config_model_type": model.config.model_type,
        "vision_tower_source": vision_tower_source,
        "vision_tower_sha256": sha256(local_clip / "pytorch_model.bin")
        if (local_clip / "pytorch_model.bin").is_file()
        else None,
        "image_aspect_ratio": getattr(model.config, "image_aspect_ratio", None),
        "non_lora_checkpoint_keys": 0,
        "non_lora_matched_keys": 0,
        "non_lora_shape_mismatches": [],
        "non_lora_unexpected_keys": [],
        "non_lora_value_mismatches": [],
        "adapter_tensor_count": 0,
        "adapter_parameter_abs_sum": 0.0,
        "adapter_sampled_nonzero_delta_count": 0,
        "adapter_active": variant in {"base", "factmm-rag-generator"},
    }
    adapter_path = None if variant == "factmm-rag-generator" else VARIANTS[variant]
    if adapter_path is not None:
        non_lora = adapter_path / "non_lora_trainables.bin"
        if non_lora.is_file():
            state = torch.load(non_lora, map_location="cpu")
            state = {(k[11:] if k.startswith("base_model.") else k): v for k, v in state.items()}
            if any(k.startswith("model.model.") for k in state):
                state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
            incompatible = model.load_state_dict(state, strict=False)
            ledger["non_lora_checkpoint_keys"] = len(state)
            ledger["non_lora_unexpected_keys"] = list(incompatible.unexpected_keys)
            current = model.state_dict()
            for key, value in state.items():
                if key not in current or tuple(current[key].shape) != tuple(value.shape):
                    ledger["non_lora_shape_mismatches"].append(key)
                    continue
                ledger["non_lora_matched_keys"] += 1
                observed = current[key].detach().cpu()
                if not torch.equal(observed, value.to(dtype=observed.dtype)):
                    ledger["non_lora_value_mismatches"].append(key)
            if (
                incompatible.unexpected_keys
                or ledger["non_lora_matched_keys"] != len(state)
                or ledger["non_lora_shape_mismatches"]
                or ledger["non_lora_value_mismatches"]
            ):
                raise RuntimeError(
                    f"{variant} did not load non-LoRA tensors exactly: "
                    f"unexpected={incompatible.unexpected_keys[:8]}, "
                    f"shape={ledger['non_lora_shape_mismatches'][:8]}, "
                    f"value={ledger['non_lora_value_mismatches'][:8]}"
                )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        adapter_parameters = [
            parameter.detach().float()
            for name, parameter in model.named_parameters()
            if "lora_" in name
        ]
        ledger["adapter_tensor_count"] = len(adapter_parameters)
        ledger["adapter_parameter_abs_sum"] = float(
            sum(parameter.abs().sum().item() for parameter in adapter_parameters)
        )
        sampled_nonzero = 0
        for _, module in model.named_modules():
            if not hasattr(module, "lora_A") or "default" not in module.lora_A:
                continue
            lora_a = module.lora_A["default"].weight.detach().float()
            lora_b = module.lora_B["default"].weight.detach().float()
            width = min(8, lora_a.shape[1], lora_b.shape[0])
            for index in range(width):
                sampled_nonzero += int(
                    torch.dot(lora_b[index], lora_a[:, index]).item() != 0.0
                )
        ledger["adapter_sampled_nonzero_delta_count"] = sampled_nonzero
        ledger["adapter_active"] = bool(
            ledger["adapter_tensor_count"] > 0
            and ledger["adapter_parameter_abs_sum"] > 0.0
            and sampled_nonzero > 0
        )
        if not ledger["adapter_active"]:
            raise RuntimeError(f"{variant} LoRA checkpoint did not alter the loaded model")
        # OPA-DPO's released evaluator keeps the PEFT wrapper and evaluates in
        # BF16 because its LoRA targets include the vision tower.  The other
        # released LLaVA loaders merge the language-side adapters.
        if variant != "opa-dpo":
            model = model.merge_and_unload()
    model.eval()
    return tokenizer, model, image_processor, ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = json.loads(args.manifest.read_text())
    if isinstance(rows, dict):
        rows = rows.get("questions", rows.get("data", []))
    rows = rows[: args.limit] if args.limit else rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = args.output_dir / "answers.jsonl"
    expected_qids = [
        str(row.get("qid", row.get("question_id", row.get("id", index))))
        for index, row in enumerate(rows, start=1)
    ]
    done = {}
    observed_qids = []
    if answers_path.is_file():
        for line in answers_path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                item_qid = str(item["qid"])
                if item_qid in done:
                    raise ValueError(f"duplicate resumed qid: {item_qid}")
                done[item_qid] = item
                observed_qids.append(item_qid)
    if observed_qids != expected_qids[: len(observed_qids)]:
        raise ValueError("existing answers are not an exact manifest prefix")
    t0_path = Path("corrected_runs/paper_baselines_v1/trained_llava_t0_v1.json")
    t0 = json.loads(t0_path.read_text())
    t0_row = next(row for row in t0["methods"] if row["method"] == args.variant)
    if t0_row.get("status") != "pass":
        raise RuntimeError(f"trained baseline T0 did not pass for {args.variant}")
    config = {
        "protocol_version": VERSION,
        "variant": args.variant,
        "base": str(BASE),
        "checkpoint_role": (
            "base"
            if args.variant == "base"
            else "full_generator_no_retrieval"
            if args.variant == "factmm-rag-generator"
            else "peft_adapter"
        ),
        "full_model_checkpoint": str(
            VARIANTS[args.variant]
            if args.variant == "factmm-rag-generator"
            else BASE
        ),
        "adapter": (
            str(VARIANTS[args.variant])
            if args.variant not in {"base", "factmm-rag-generator"}
            else None
        ),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "image_root": str(args.image_root.resolve()),
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "do_sample": False,
        "num_beams": 1,
        "conversation": "llava_v1",
        "image_preprocessing": "official llava.mm_utils.process_images; respects image_aspect_ratio",
        "compute_dtype": "bfloat16" if args.variant == "opa-dpo" else "float16",
        "peft_merge": args.variant not in {"base", "factmm-rag-generator", "opa-dpo"},
        "runner_sha256": sha256(Path(__file__).resolve()),
        "base_config_sha256": sha256(BASE / "config.json"),
        "checkpoint_fingerprint": t0_row["checkpoint_fingerprint"],
        "checkpoint_t0_audit": str(t0_path.resolve()),
        "checkpoint_t0_audit_sha256": sha256(t0_path),
        "llava_source_root": str(LLAVA_ROOT.resolve()),
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    config["fingerprint"] = fingerprint
    config_path = args.output_dir / "generation_config.json"
    if config_path.is_file():
        prior = json.loads(config_path.read_text())
        if prior.get("fingerprint") != fingerprint:
            raise ValueError("refusing to resume an incompatible trained-baseline run")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    tokenizer, model, image_processor, loading_ledger = load_model(args.variant)
    ledger_path = args.output_dir / "loading_ledger.json"
    ledger_path.write_text(json.dumps(loading_ledger, indent=2) + "\n")
    loading_ledger_sha256 = sha256(ledger_path)
    with answers_path.open("a") as handle:
        for index, row in enumerate(rows, start=1):
            qid = str(row.get("qid", row.get("question_id", row.get("id", index))))
            if qid in done:
                print(f"[{index}/{len(rows)}] reuse {qid}", flush=True)
                continue
            image_name = str(row.get("img_name", row.get("image", "")))
            image_path = args.image_root / image_name
            try:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                record = {
                    "qid": qid,
                    "question_id": qid,
                    "img_name": image_name,
                    "question": str(row.get("question", row.get("text", ""))).strip(),
                    "answer": row.get("answer", row.get("gt_ans")),
                    "text": "",
                    "model_id": args.variant,
                    "metadata": {
                        "generated_token_ids": [],
                        "generated_token_count": 0,
                        "max_new_tokens": args.max_new_tokens,
                        "stop_reason": "input_unavailable",
                        "input_error": f"{type(exc).__name__}: {exc}",
                        "fingerprint": fingerprint,
                        "loading_ledger_sha256": loading_ledger_sha256,
                    },
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                handle.flush()
                print(f"[{index}/{len(rows)}] {qid} input_unavailable", flush=True)
                continue
            question = str(row.get("question", row.get("text", ""))).strip()
            image_prefix = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                if model.config.mm_use_im_start_end
                else DEFAULT_IMAGE_TOKEN
            )
            conversation = conv_templates["llava_v1"].copy()
            conversation.append_message(conversation.roles[0], image_prefix + "\n" + question)
            conversation.append_message(conversation.roles[1], None)
            prompt = conversation.get_prompt()
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)
            # The official loader indexes the single image inside Dataset and
            # DataLoader restores the batch dimension.  This runner bypasses
            # DataLoader, so preserve/restore that exact [1, 3, H, W] shape.
            pixels = process_images([image], image_processor, model.config)
            if isinstance(pixels, list):
                pixels = pixels[0].unsqueeze(0)
            elif pixels.ndim == 3:
                pixels = pixels.unsqueeze(0)
            preprocessed_pixel_tensor_sha256 = hashlib.sha256(
                pixels.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            compute_dtype = torch.bfloat16 if args.variant == "opa-dpo" else torch.float16
            pixels = pixels.to(model.device, dtype=compute_dtype)
            prompt_token_sha256 = hashlib.sha256(
                json.dumps(input_ids[0].tolist(), separators=(",", ":")).encode()
            ).hexdigest()
            pixel_tensor_sha256 = hashlib.sha256(
                pixels.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            stop = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
            generation_inputs = {}
            if args.variant == "sentinel":
                generation_inputs["image_sizes"] = [image.size]
            output = model.generate(
                input_ids,
                images=pixels,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=False,
                **generation_inputs,
            )
            raw_suffix = (
                output.sequences[0].tolist()
                if args.variant == "sentinel"
                else output.sequences[0, input_ids.shape[1] :].tolist()
            )
            suffix = list(raw_suffix)
            eos = tokenizer.eos_token_id
            if eos in suffix:
                suffix = suffix[: suffix.index(eos)]
            text = tokenizer.decode(suffix, skip_special_tokens=True).strip()
            if text.endswith(stop):
                text = text[: -len(stop)].strip()
            record = {
                "qid": qid,
                "question_id": qid,
                "img_name": image_name,
                "question": question,
                "answer": row.get("answer", row.get("gt_ans")),
                "text": text,
                "model_id": args.variant,
                "metadata": {
                    "generated_token_ids": suffix,
                    "raw_generated_token_ids": raw_suffix,
                    "generated_token_count": len(suffix),
                    "max_new_tokens": args.max_new_tokens,
                    "fingerprint": fingerprint,
                    "loading_ledger_sha256": loading_ledger_sha256,
                    "prompt_token_ids_sha256": prompt_token_sha256,
                    "preprocessed_pixel_tensor_sha256": preprocessed_pixel_tensor_sha256,
                    "pixel_tensor_sha256": pixel_tensor_sha256,
                },
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {qid} tokens={len(suffix)}", flush=True)


if __name__ == "__main__":
    main()
