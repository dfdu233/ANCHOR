#!/usr/bin/env python3
"""Common CE-G mitigation inference using the patched LLaVA-Med backend.

This module intentionally supports only methods whose LLaVA-Med generation
hooks are locally available and activation-auditable.  Prompt
construction retains RULE's dataset-specific evidence wording and adds one
uniform leading-decision contract required by the corrected evaluator.
Every run emits a method-activation sidecar; callers must reject missing or
inconsistent counters rather than silently accepting ordinary decoding.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "rule-vqa-mitigation-backend-v8-leading-native-eos"
SUPPORTED_METHODS = (
    "AVISC",
    "DoLa",
    "M3ID",
    "OPERA",
    "PAI",
    "PAIControl",
    "VCD",
)
CONV_MODE = "vicuna_v1"
RULE_VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's "
    "questions."
)
RULE_VICUNA_STOP = "</s>"
LEADING_DECISION_SUFFIX = (
    " Begin your answer with exactly Yes or No, then give at most one concise sentence."
)


class RuleMitigationBackendError(RuntimeError):
    """Raised when protocol-preserving mitigation inference cannot proceed."""


def stable_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def render_rule_prompt(dataset: str, row: dict[str, Any]) -> str:
    """Retain RULE evidence wording under a uniform CE-G answer contract."""
    if dataset not in {"iuxray", "mimic", "harvard"}:
        raise RuleMitigationBackendError(f"unsupported RULE dataset: {dataset}")
    question = str(row["question"]).replace("<image>", "").strip()
    if "reference_report" not in row:
        if dataset == "iuxray":
            suffix = (
                "Please answer the question based on the image and choose from "
                "the following two options: [yes, no]."
            )
        else:
            suffix = (
                "Please answer the question based on the image and report and "
                "choose from the following two options: [yes, no]."
            )
        return question + " " + suffix + LEADING_DECISION_SUFFIX

    reports = row["reference_report"]
    if not isinstance(reports, list):
        reports = [reports]
    topk = len(reports)
    if dataset == "mimic":
        cleaned = [" ".join(str(report).replace("\n", " ").split()) for report in reports]
        if topk == 1:
            formatted = f"{cleaned[0]} "
        else:
            formatted = "".join(
                f"{index + 1}. {report} " for index, report in enumerate(cleaned)
            )
        appendix_1 = (
            "You are provided with a chest X-ray image, a image-related question: \n"
        )
        appendix_2 = (
            f"You are also provided with {topk} reference report(s).Please answer "
            "the question based on the image and report and answer the question "
            "based on the image and report and choose from the following two "
            "options: [yes, no]. It should be noted that the diagnostic "
            "information in the reference reports cannot be directly used as the "
            "basis for diagnosis, but should only be used for reference. "
            "\nReference reports:"
        )
        return (
            appendix_1 + question + "\n" + appendix_2 + "\n" + formatted
            + LEADING_DECISION_SUFFIX
        )

    if topk == 1:
        formatted = str(reports[0])
    else:
        formatted = "".join(
            f"{index + 1}. {report} " for index, report in enumerate(reports)
        )
    if dataset == "iuxray":
        appendix_1 = (
            f"You are provided with a chest X-ray image, a image-related question "
            f"and {topk} reference report(s): "
        )
        appendix_2 = (
            "Please answer the question based on the image and report and choose "
            "from the following two options: [yes, no]. It should be noted that "
            "the diagnostic information in the reference reports cannot be "
            "directly used as the basis for diagnosis, but should only be used "
            "for reference and comparison. Question: "
        )
    else:
        appendix_1 = (
            f"You are provided with a fundus image, a image-related question and "
            f"{topk} reference summary(s): "
        )
        appendix_2 = (
            "Please answer the question based on the image and the reference "
            "summaries and choose from the following two options: [yes, no]. It "
            "should be noted that the diagnostic information in the reference "
            "summaries cannot be directly used as the basis for diagnosis, but "
            "should only be used for reference and comparison. Question: "
        )
    return appendix_1 + formatted + "\n" + appendix_2 + question + LEADING_DECISION_SUFFIX


def render_rule_model_prompt(
    current_prompt: str,
    *,
    image_token: str,
    image_start_token: str,
    image_end_token: str,
    use_image_start_end: bool,
) -> str:
    """Render RULE's one-turn ``vicuna_v1`` prompt without import ambiguity."""
    if use_image_start_end:
        image_prefix = image_start_token + image_token + image_end_token
    else:
        image_prefix = image_token
    user_message = image_prefix + "\n" + current_prompt
    return f"{RULE_VICUNA_SYSTEM} USER: {user_message} ASSISTANT:"


def pai_cfg_model_prompt(
    model_prompt: str,
    *,
    image_token: str,
    image_start_token: str,
    image_end_token: str,
) -> str:
    """Return PAI's full conversation prompt with one removable image marker.

    The official PAI CFG branch removes the image placeholder from the same
    conversation template used by the conditional branch. Canonicalizing a
    wrapped LLaVA marker to one bare marker lets the upstream-compatible
    ``init_cfg_processor`` remove the whole visual marker while preserving the
    system message, user/assistant roles, and task text.
    """
    wrapped = image_start_token + image_token + image_end_token
    canonical = model_prompt.replace(wrapped, image_token)
    if canonical.count(image_token) != 1:
        raise RuleMitigationBackendError(
            "PAI CFG requires exactly one image marker in the full model prompt"
        )
    return canonical


def prompt_manifest(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = [
        {
            "question_id": str(row["question_id"]),
            "sha256": sha256_text(render_rule_prompt(dataset, row)),
        }
        for row in rows
    ]
    return {
        "dataset": dataset,
        "conv_mode": CONV_MODE,
        "count": len(prompts),
        "ordered_prompt_sha256": hashlib.sha256(
            stable_json(prompts).encode()
        ).hexdigest(),
        "prompts": prompts,
    }


def opera_key_position(
    input_ids: Any,
    *,
    image_token_index: int,
    num_image_tokens: int,
) -> dict[str, int]:
    """Map one LLaVA image placeholder to OPERA's expanded attention indices."""
    if num_image_tokens <= 0:
        raise RuleMitigationBackendError("OPERA image-token count must be positive")
    if getattr(input_ids, "ndim", None) != 2 or input_ids.shape[0] != 1:
        raise RuleMitigationBackendError("OPERA requires a single unpadded prompt")
    positions = (input_ids[0] == image_token_index).nonzero(as_tuple=False).flatten()
    if positions.numel() != 1:
        raise RuleMitigationBackendError(
            f"OPERA requires exactly one image placeholder, observed={positions.numel()}"
        )
    image_start = int(positions.item())
    image_end = image_start + num_image_tokens - 1
    response_start = int(input_ids.shape[1]) + num_image_tokens - 1
    if not (0 <= image_start <= image_end < response_start):
        raise RuleMitigationBackendError("invalid OPERA attention positions")
    return {
        "image_start": image_start,
        "image_end": image_end,
        "response_start": response_start,
    }


def generation_config(
    method: str, max_new_tokens: int = 1024
) -> dict[str, Any]:
    """Return the exact generation kwargs that materially define a port."""
    if method == "VCD":
        return {
            "do_sample": True,
            "temperature": 1.0,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "noise_step": 500,
            "cd_alpha": 1.0,
            "cd_beta": 0.1,
        }
    if method == "DoLa":
        return {
            "do_sample": False,
            "temperature_argument": 0.9,
            "top_p_argument": 0.95,
            "top_k_argument": 0,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "relative_top": 0.1,
            "early_exit_layers": [0, 2, 4, 6, 8, 10, 12, 14, 32],
        }
    if method == "OPERA":
        return {
            "do_sample": False,
            "num_beams": 5,
            "max_new_tokens": max_new_tokens,
            "output_attentions": True,
            "scale_factor": 50.0,
            "threshold": 15,
            "num_attn_candidates": 5,
            "penalty_weights": 1.0,
            "key_position": "dynamic_from_image_placeholder_and_vision_patches",
        }
    if method == "PAI":
        return {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "attention_alpha": 0.2,
            "cfg_gamma": 1.1,
            "start_layer": 2,
            "end_layer_exclusive": 32,
            "image_range": "dynamic_from_image_placeholder_and_vision_patches",
            "cfg_prompt": "full_vicuna_conversation_without_image_marker",
            "attention_kernel": "PAI_Mistral_compatibility_port",
        }
    if method == "PAIControl":
        return {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "attention_intervention": False,
            "cfg_intervention": False,
            "runtime_control_for": "PAI",
            "attention_kernel": "PAI_Mistral_compatibility_port",
        }
    if method == "M3ID":
        return {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "lambda": 0.02,
            "cd_beta": 0.1,
            "unconditioned_view": "dynamic_remove_image_placeholder",
            "implementation": "MedHEval_paper_based_reimplementation",
        }
    if method == "AVISC":
        return {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "cd_alpha": 1.0,
            "cd_beta": 0.1,
            "layer_gamma": 0.5,
            "masking_scheme": "zeros",
            "lamb": 1.0,
            "image_range": "dynamic_from_image_placeholder_and_vision_patches",
            "mistral_mask_offset": 34,
        }
    raise RuleMitigationBackendError(
        f"unsupported mitigation method: {method}"
    )


def expected_activation_counters(method: str, samples: int) -> dict[str, int]:
    if method == "VCD":
        return {
            "sampling_hook_installed": 1,
            "distorted_views_created": samples,
            "contrastive_generate_calls": samples,
        }
    if method == "DoLa":
        return {
            "early_exit_configuration_calls": samples,
            "dola_generate_calls": samples,
        }
    if method == "OPERA":
        return {
            "opera_branch_wrapped": 1,
            "dynamic_key_positions_validated": samples,
            "opera_beam_search_entries": samples,
            "opera_generate_calls": samples,
        }
    if method == "PAI":
        return {
            "dynamic_image_ranges_validated": samples,
            "attention_patch_calls": samples,
            "cfg_processors_created": samples,
            "pai_generate_calls": samples,
        }
    if method == "PAIControl":
        return {
            "dynamic_image_ranges_validated": samples,
            "identity_attention_patch_calls": samples,
            "identity_generate_calls": samples,
        }
    if method == "M3ID":
        return {
            "sampling_hook_installed": 1,
            "dynamic_image_ranges_validated": samples,
            "m3id_sample_entries": samples,
            "m3id_generate_calls": samples,
        }
    if method == "AVISC":
        return {
            "sampling_hook_installed": 1,
            "dynamic_image_ranges_validated": samples,
            "avisc_sample_entries": samples,
            "avisc_generate_calls": samples,
        }
    raise RuleMitigationBackendError(f"unsupported mitigation method: {method}")


def validate_activation_report(
    report: dict[str, Any],
    *,
    method: str,
    expected_samples: int,
    expected_prompt_sha256: str,
    expected_max_new_tokens: int = 1024,
) -> None:
    if report.get("protocol_version") != PROTOCOL_VERSION:
        raise RuleMitigationBackendError("activation protocol version differs")
    if report.get("method") != method:
        raise RuleMitigationBackendError("activation method differs")
    if report.get("conv_mode") != CONV_MODE:
        raise RuleMitigationBackendError("activation conversation mode differs")
    if report.get("processed_samples") != expected_samples:
        raise RuleMitigationBackendError("activation sample count differs")
    if report.get("ordered_prompt_sha256") != expected_prompt_sha256:
        raise RuleMitigationBackendError("activation prompt hash differs")
    expected_generation = generation_config(
        method, expected_max_new_tokens
    )
    if report.get("generation") != expected_generation:
        raise RuleMitigationBackendError(
            "activation generation configuration differs"
        )
    counters = report.get("counters")
    if counters != expected_activation_counters(method, expected_samples):
        raise RuleMitigationBackendError(
            f"activation counters differ: observed={counters}"
        )
    if method == "OPERA":
        key_positions = report.get("key_positions")
        if not isinstance(key_positions, list) or len(key_positions) != expected_samples:
            raise RuleMitigationBackendError("OPERA key-position audit is incomplete")
        for item in key_positions:
            start = item.get("image_start")
            end = item.get("image_end")
            response = item.get("response_start")
            patches = item.get("num_image_tokens")
            if not all(isinstance(value, int) for value in (start, end, response, patches)):
                raise RuleMitigationBackendError("OPERA key positions must be integers")
            if patches <= 0 or end - start + 1 != patches or not (0 <= start <= end < response):
                raise RuleMitigationBackendError("OPERA key-position ranges are invalid")
    if method == "PAI":
        audits = report.get("pai_audits")
        if not isinstance(audits, list) or len(audits) != expected_samples:
            raise RuleMitigationBackendError("PAI activation audit is incomplete")
        for item in audits:
            start = item.get("image_start")
            end = item.get("image_end_exclusive")
            patches = item.get("num_image_tokens")
            cfg_calls = item.get("cfg_logits_calls")
            attention_calls = item.get("attention_forward_calls")
            patched_layers = item.get("patched_layers")
            if not all(
                isinstance(value, int)
                for value in (
                    start,
                    end,
                    patches,
                    cfg_calls,
                    attention_calls,
                    patched_layers,
                )
            ):
                raise RuleMitigationBackendError("PAI audit fields must be integers")
            if (
                patches <= 0
                or end - start != patches
                or cfg_calls <= 0
                or attention_calls <= 0
                or patched_layers != 30
            ):
                raise RuleMitigationBackendError("PAI activation audit is invalid")
    if method == "M3ID":
        audits = report.get("m3id_audits")
        if not isinstance(audits, list) or len(audits) != expected_samples:
            raise RuleMitigationBackendError("M3ID activation audit is incomplete")
        for item in audits:
            values = (
                item.get("image_start"),
                item.get("image_end_exclusive"),
                item.get("num_image_tokens"),
                item.get("sample_entries"),
                item.get("unconditioned_preparations"),
                item.get("generated_tokens"),
            )
            if not all(isinstance(value, int) for value in values):
                raise RuleMitigationBackendError("M3ID audit fields must be integers")
            start, end, patches, sample_entries, preparations, generated = values
            if (
                patches <= 0
                or end - start != patches
                or sample_entries != 1
                or generated <= 0
                or preparations != generated
            ):
                raise RuleMitigationBackendError("M3ID activation audit is invalid")
            if item.get("image_removed_from_unconditioned_view") is not True:
                raise RuleMitigationBackendError(
                    "M3ID unconditioned view did not remove the image"
                )
    if method == "AVISC":
        audits = report.get("avisc_audits")
        if not isinstance(audits, list) or len(audits) != expected_samples:
            raise RuleMitigationBackendError("AVISC activation audit is incomplete")
        for item in audits:
            values = (
                item.get("image_start"),
                item.get("image_end_exclusive"),
                item.get("num_image_tokens"),
                item.get("sample_entries"),
                item.get("method_preparations"),
                item.get("masked_forward_calls"),
                item.get("masked_token_count"),
                item.get("generated_tokens"),
            )
            if not all(isinstance(value, int) for value in values):
                raise RuleMitigationBackendError("AVISC audit fields must be integers")
            (
                start,
                end,
                patches,
                entries,
                preparations,
                masked_calls,
                masked,
                generated,
            ) = values
            if (
                patches <= 0
                or end - start != patches
                or entries != 1
                or generated <= 0
                or preparations != generated
                or masked_calls <= 0
                or not (0 <= masked <= patches)
            ):
                raise RuleMitigationBackendError("AVISC activation audit is invalid")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def _generate(args: argparse.Namespace) -> None:
    # Heavy imports are delayed so protocol/unit tests remain CPU-only.
    import types

    import torch
    from PIL import Image
    from transformers import set_seed

    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    if args.method not in SUPPORTED_METHODS:
        raise RuleMitigationBackendError(
            f"{args.method} is not enabled by this backend"
        )
    if args.conv_mode != CONV_MODE:
        raise RuleMitigationBackendError(
            f"RULE protocol requires --conv-mode {CONV_MODE}"
        )
    set_seed(args.seed)
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        os.path.expanduser(str(args.model_path)),
        None,
        "mistral_llava_med_1.5",
        peft_path=None,
    )
    counters = expected_activation_counters(args.method, 0)
    key_position_audit: list[dict[str, Any]] = []
    pai_audits: list[dict[str, Any]] = []
    m3id_audits: list[dict[str, Any]] = []
    avisc_audits: list[dict[str, Any]] = []
    if args.method == "VCD":
        from llava.eval.VCD_files.vcd_add_noise import add_diffusion_noise
        from llava.eval.VCD_files.vcd_sample import evolve_vcd_sampling

        evolve_vcd_sampling()
        counters["sampling_hook_installed"] = 1
    elif args.method == "OPERA":
        original_opera_beam_search = getattr(model, "opera_beam_search", None)
        if not callable(original_opera_beam_search):
            raise RuleMitigationBackendError(
                "active transformers build has no OPERA beam-search implementation"
            )

        def counted_opera_beam_search(*call_args: Any, **call_kwargs: Any) -> Any:
            counters["opera_beam_search_entries"] += 1
            return original_opera_beam_search(*call_args, **call_kwargs)

        model.opera_beam_search = counted_opera_beam_search
        counters["opera_branch_wrapped"] = 1
    elif args.method in {"PAI", "PAIControl"}:
        from PAI_files.attention import llama_modify, llama_new_forward_caf
        from PAI_files.model_loader import init_cfg_processor
        from transformers.generation.logits_process import LogitsProcessorList
    elif args.method == "M3ID":
        from avisc_utils.avisc_sample import evolve_avisc_sampling

        evolve_avisc_sampling()
        counters["sampling_hook_installed"] = 1
        original_m3id_sample = model.sample
        active_m3id_counts: dict[str, dict[str, int] | None] = {"value": None}

        def counted_m3id_sample(
            self: Any, *call_args: Any, **call_kwargs: Any
        ) -> Any:
            if call_kwargs.get("use_m3id") is not True:
                raise RuleMitigationBackendError(
                    "M3ID sampling hook entered without use_m3id=True"
                )
            sample_counts = active_m3id_counts["value"]
            if sample_counts is None:
                raise RuleMitigationBackendError("M3ID sample has no active audit")
            sample_counts["sample_entries"] += 1
            sampler_input_ids = (
                call_args[0] if call_args else call_kwargs.get("input_ids")
            )
            if sampler_input_ids is None:
                raise RuleMitigationBackendError("M3ID sampler input ids are missing")
            result = original_m3id_sample(*call_args, **call_kwargs)
            sample_counts["generated_tokens"] = int(
                result.shape[-1] - sampler_input_ids.shape[-1]
            )
            return result

        model.sample = types.MethodType(counted_m3id_sample, model)
    elif args.method == "AVISC":
        from corrected_sgta.avisc_sample_dynamic import evolve_avisc_sampling

        evolve_avisc_sampling()
        counters["sampling_hook_installed"] = 1
        original_avisc_sample = model.sample
        original_method_prepare = model.prepare_inputs_for_generation_method
        original_model_forward = model.forward
        active_avisc: dict[str, Any] = {"counts": None, "image_start": None}

        def counted_avisc_sample(
            self: Any, *call_args: Any, **call_kwargs: Any
        ) -> Any:
            if call_kwargs.get("use_avisc") is not True:
                raise RuleMitigationBackendError(
                    "AVISC sampling hook entered without use_avisc=True"
                )
            counts = active_avisc["counts"]
            if counts is None:
                raise RuleMitigationBackendError("AVISC sample has no active audit")
            counts["sample_entries"] += 1
            sampler_input_ids = (
                call_args[0] if call_args else call_kwargs.get("input_ids")
            )
            if sampler_input_ids is None:
                raise RuleMitigationBackendError("AVISC sampler input ids are missing")
            result = original_avisc_sample(*call_args, **call_kwargs)
            counts["generated_tokens"] = int(
                result.shape[-1] - sampler_input_ids.shape[-1]
            )
            return result

        def counted_method_prepare(
            self: Any, *call_args: Any, **call_kwargs: Any
        ) -> Any:
            counts = active_avisc["counts"]
            if counts is not None:
                counts["method_preparations"] += 1
            return original_method_prepare(*call_args, **call_kwargs)

        def dynamic_mask_forward(
            self: Any, *call_args: Any, **call_kwargs: Any
        ) -> Any:
            mask_idx = call_kwargs.get("mask_idx")
            counts = active_avisc["counts"]
            if mask_idx is not None and counts is not None:
                image_start = active_avisc["image_start"]
                if not isinstance(image_start, int):
                    raise RuleMitigationBackendError("AVISC image start is unavailable")
                adjusted = mask_idx + (image_start - 34)
                masked_token_count = int(mask_idx.numel())
                if masked_token_count:
                    lower = int(adjusted.min().item()) + 34
                    upper = int(adjusted.max().item()) + 34
                    image_end = image_start + counts["num_image_tokens"]
                    if not (image_start <= lower <= upper < image_end):
                        raise RuleMitigationBackendError(
                            "AVISC mask coordinates are invalid"
                        )
                call_kwargs["mask_idx"] = adjusted
                counts["masked_forward_calls"] += 1
                counts["masked_token_count"] = masked_token_count
            return original_model_forward(*call_args, **call_kwargs)

        dynamic_mask_forward.__signature__ = inspect.signature(original_model_forward)
        model.sample = types.MethodType(counted_avisc_sample, model)
        model.prepare_inputs_for_generation_method = types.MethodType(
            counted_method_prepare, model
        )
        model.forward = types.MethodType(dynamic_mask_forward, model)

    rows = load_jsonl(args.question_file)
    manifest = prompt_manifest(args.dataset, rows)
    args.answers_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_answers = args.answers_file.with_suffix(args.answers_file.suffix + ".tmp")
    with temporary_answers.open("w") as output:
        for row, prompt_item in zip(rows, manifest["prompts"]):
            current_prompt = render_rule_prompt(args.dataset, row)
            model_prompt = render_rule_model_prompt(
                current_prompt,
                image_token=DEFAULT_IMAGE_TOKEN,
                image_start_token=DEFAULT_IM_START_TOKEN,
                image_end_token=DEFAULT_IM_END_TOKEN,
                use_image_start_end=model.config.mm_use_im_start_end,
            )
            input_ids = tokenizer_image_token(
                model_prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).cuda()
            attention_mask = torch.ones_like(input_ids)
            image = Image.open(args.image_folder / str(row["image"]))
            image_tensor = process_images([image], image_processor, model.config)[0]
            if args.method == "VCD":
                distorted = add_diffusion_noise(image_tensor, 500)
                counters["distorted_views_created"] += 1
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        images_cd=distorted.unsqueeze(0).half().cuda(),
                        cd_alpha=1,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=1,
                        max_new_tokens=args.max_new_tokens,
                    )
                counters["contrastive_generate_calls"] += 1
            elif args.method == "DoLa":
                early_exit_layers = [0, 2, 4, 6, 8, 10, 12, 14, 32]
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        max_new_tokens=args.max_new_tokens,
                        dola_decoding=True,
                        top_p=0.95,
                        top_k=0,
                        temperature=0.9,
                        relative_top=0.1,
                        mature_layer=early_exit_layers[-1],
                        premature_layer=None,
                        candidate_premature_layers=early_exit_layers[:-1],
                    )
                counters["early_exit_configuration_calls"] += 1
                counters["dola_generate_calls"] += 1
            elif args.method == "OPERA":
                vision_tower = model.get_vision_tower()
                num_image_tokens = int(getattr(vision_tower, "num_patches", 0))
                key_position = opera_key_position(
                    input_ids,
                    image_token_index=IMAGE_TOKEN_INDEX,
                    num_image_tokens=num_image_tokens,
                )
                key_position_audit.append(
                    {
                        "question_id": str(row["question_id"]),
                        **key_position,
                        "num_image_tokens": num_image_tokens,
                    }
                )
                counters["dynamic_key_positions_validated"] += 1
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        do_sample=False,
                        num_beams=5,
                        max_new_tokens=args.max_new_tokens,
                        output_attentions=True,
                        opera_decoding=True,
                        key_position=key_position,
                        scale_factor=50.0,
                        threshold=15,
                        num_attn_candidates=5,
                        penalty_weights=1.0,
                    )
                counters["opera_generate_calls"] += 1
            elif args.method == "PAI":
                vision_tower = model.get_vision_tower()
                num_image_tokens = int(getattr(vision_tower, "num_patches", 0))
                expanded = opera_key_position(
                    input_ids,
                    image_token_index=IMAGE_TOKEN_INDEX,
                    num_image_tokens=num_image_tokens,
                )
                image_start = expanded["image_start"]
                image_end_exclusive = expanded["image_end"] + 1
                counters["dynamic_image_ranges_validated"] += 1
                llama_modify(
                    model,
                    start_layer=2,
                    end_layer=32,
                    alpha=0.2,
                    use_attn=True,
                    use_cfg=True,
                    img_start_idx=image_start,
                    img_end_idx=image_end_exclusive,
                )
                patched_layers = sum(
                    getattr(model.model.layers[index].self_attn.forward, "__func__", None)
                    is llama_new_forward_caf
                    for index in range(2, 32)
                )
                if patched_layers != 30:
                    raise RuleMitigationBackendError(
                        f"PAI attention patch incomplete: {patched_layers}/30 layers"
                    )
                counters["attention_patch_calls"] += 1

                sample_counts = {"cfg": 0, "attention": 0}
                sentinel = model.model.layers[2].self_attn
                original_attention_forward = sentinel.forward

                def counted_attention_forward(
                    self: Any, *call_args: Any, **call_kwargs: Any
                ) -> Any:
                    sample_counts["attention"] += 1
                    return original_attention_forward(*call_args, **call_kwargs)

                sentinel.forward = types.MethodType(counted_attention_forward, sentinel)
                cfg_processor = init_cfg_processor(
                    tokenizer=tokenizer,
                    llm_model=model,
                    questions=[
                        pai_cfg_model_prompt(
                            model_prompt,
                            image_token=DEFAULT_IMAGE_TOKEN,
                            image_start_token=DEFAULT_IM_START_TOKEN,
                            image_end_token=DEFAULT_IM_END_TOKEN,
                        )
                    ],
                    gamma=1.1,
                    beam=1,
                    start_layer=2,
                    end_layer=32,
                )
                counters["cfg_processors_created"] += 1

                class CountingCFGProcessor:
                    def __call__(
                        self, processor_input_ids: Any, scores: Any
                    ) -> Any:
                        sample_counts["cfg"] += 1
                        return cfg_processor(processor_input_ids, scores)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        use_cache=True,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=args.max_new_tokens,
                        output_attentions=False,
                        output_hidden_states=False,
                        logits_processor=LogitsProcessorList(
                            [CountingCFGProcessor()]
                        ),
                    )
                counters["pai_generate_calls"] += 1
                pai_audits.append(
                    {
                        "question_id": str(row["question_id"]),
                        "image_start": image_start,
                        "image_end_exclusive": image_end_exclusive,
                        "num_image_tokens": num_image_tokens,
                        "patched_layers": patched_layers,
                        "cfg_logits_calls": sample_counts["cfg"],
                        "attention_forward_calls": sample_counts["attention"],
                    }
                )
            elif args.method == "PAIControl":
                vision_tower = model.get_vision_tower()
                num_image_tokens = int(getattr(vision_tower, "num_patches", 0))
                expanded = opera_key_position(
                    input_ids,
                    image_token_index=IMAGE_TOKEN_INDEX,
                    num_image_tokens=num_image_tokens,
                )
                counters["dynamic_image_ranges_validated"] += 1
                llama_modify(
                    model,
                    start_layer=2,
                    end_layer=32,
                    alpha=0.0,
                    use_attn=False,
                    use_cfg=False,
                    img_start_idx=expanded["image_start"],
                    img_end_idx=expanded["image_end"] + 1,
                )
                patched_layers = sum(
                    getattr(model.model.layers[index].self_attn.forward, "__func__", None)
                    is llama_new_forward_caf
                    for index in range(2, 32)
                )
                if patched_layers != 30:
                    raise RuleMitigationBackendError(
                        f"PAI control attention patch incomplete: {patched_layers}/30 layers"
                    )
                counters["identity_attention_patch_calls"] += 1
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        use_cache=True,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=args.max_new_tokens,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                counters["identity_generate_calls"] += 1

            elif args.method == "M3ID":
                vision_tower = model.get_vision_tower()
                num_image_tokens = int(getattr(vision_tower, "num_patches", 0))
                expanded = opera_key_position(
                    input_ids,
                    image_token_index=IMAGE_TOKEN_INDEX,
                    num_image_tokens=num_image_tokens,
                )
                image_start = expanded["image_start"]
                image_end_exclusive = expanded["image_end"] + 1
                counters["dynamic_image_ranges_validated"] += 1
                sample_counts = {
                    "sample_entries": 0,
                    "unconditioned_preparations": 0,
                    "generated_tokens": 0,
                }
                active_m3id_counts["value"] = sample_counts

                def dynamic_prepare_inputs_for_generation_m3id(
                    self: Any,
                    m3id_input_ids: Any,
                    past_key_values: Any = None,
                    attention_mask: Any = None,
                    inputs_embeds: Any = None,
                    **kwargs: Any,
                ) -> dict[str, Any]:
                    """Build M3ID's text-only branch without fixed 34/576 indices."""
                    if past_key_values is not None:
                        m3id_input_ids = m3id_input_ids[:, -1:]
                    if inputs_embeds is not None and past_key_values is None:
                        if inputs_embeds.shape[1] < image_end_exclusive:
                            raise RuleMitigationBackendError(
                                "expanded M3ID embeddings do not contain the image range"
                            )
                        text_embeds = torch.cat(
                            (
                                inputs_embeds[:, :image_start, :],
                                inputs_embeds[:, image_end_exclusive:, :],
                            ),
                            dim=1,
                        )
                        model_inputs = {"inputs_embeds": text_embeds}
                        text_attention_mask = torch.ones(
                            text_embeds.shape[:2],
                            dtype=torch.long,
                            device=text_embeds.device,
                        )
                    else:
                        placeholder_mask = m3id_input_ids.eq(IMAGE_TOKEN_INDEX)
                        placeholder_count = int(placeholder_mask.sum().item())
                        if placeholder_count == 1:
                            text_input_ids = m3id_input_ids[
                                ~placeholder_mask
                            ].reshape(m3id_input_ids.shape[0], -1)
                        elif placeholder_count == 0:
                            text_input_ids = m3id_input_ids
                        else:
                            raise RuleMitigationBackendError(
                                "M3ID requires at most one image placeholder"
                            )
                        model_inputs = {"input_ids": text_input_ids}
                        text_attention_mask = torch.ones_like(text_input_ids)
                    sample_counts["unconditioned_preparations"] += 1
                    model_inputs.update(
                        {
                            "past_key_values": past_key_values,
                            "use_cache": kwargs.get("use_cache"),
                            "attention_mask": text_attention_mask,
                            "images": None,
                        }
                    )
                    return model_inputs

                model.prepare_inputs_for_generation_m3id = types.MethodType(
                    dynamic_prepare_inputs_for_generation_m3id, model
                )
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        images_cd=None,
                        cd_alpha=1.0,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=None,
                        max_new_tokens=args.max_new_tokens,
                        use_avisc=False,
                        use_m3id=True,
                        layer_gamma=0.5,
                        lamb=1.0,
                    )
                counters["m3id_sample_entries"] += sample_counts["sample_entries"]
                counters["m3id_generate_calls"] += 1
                generated_tokens = sample_counts["generated_tokens"]
                m3id_audits.append(
                    {
                        "question_id": str(row["question_id"]),
                        "image_start": image_start,
                        "image_end_exclusive": image_end_exclusive,
                        "num_image_tokens": num_image_tokens,
                        "sample_entries": sample_counts["sample_entries"],
                        "unconditioned_preparations": sample_counts[
                            "unconditioned_preparations"
                        ],
                        "generated_tokens": generated_tokens,
                        "image_removed_from_unconditioned_view": True,
                    }
                )
                active_m3id_counts["value"] = None

            else:
                vision_tower = model.get_vision_tower()
                num_image_tokens = int(getattr(vision_tower, "num_patches", 0))
                expanded = opera_key_position(
                    input_ids,
                    image_token_index=IMAGE_TOKEN_INDEX,
                    num_image_tokens=num_image_tokens,
                )
                image_start = expanded["image_start"]
                image_end_exclusive = expanded["image_end"] + 1
                counters["dynamic_image_ranges_validated"] += 1
                sample_counts = {
                    "sample_entries": 0,
                    "method_preparations": 0,
                    "masked_forward_calls": 0,
                    "masked_token_count": 0,
                    "generated_tokens": 0,
                    "num_image_tokens": num_image_tokens,
                }
                active_avisc["counts"] = sample_counts
                active_avisc["image_start"] = image_start
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        images_cd=None,
                        cd_alpha=1.0,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=None,
                        max_new_tokens=args.max_new_tokens,
                        use_avisc=True,
                        layer_gamma=0.5,
                        masking_scheme="zeros",
                        lamb=1.0,
                        model_name="llava",
                        avisc_image_start=image_start,
                        avisc_num_image_tokens=num_image_tokens,
                    )
                counters["avisc_sample_entries"] += sample_counts["sample_entries"]
                counters["avisc_generate_calls"] += 1
                avisc_audits.append(
                    {
                        "question_id": str(row["question_id"]),
                        "image_start": image_start,
                        "image_end_exclusive": image_end_exclusive,
                        "num_image_tokens": num_image_tokens,
                        "sample_entries": sample_counts["sample_entries"],
                        "method_preparations": sample_counts["method_preparations"],
                        "masked_forward_calls": sample_counts["masked_forward_calls"],
                        "masked_token_count": sample_counts["masked_token_count"],
                        "generated_tokens": sample_counts["generated_tokens"],
                    }
                )
                active_avisc["counts"] = None
                active_avisc["image_start"] = None

            text = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()
            output.write(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "prompt": current_prompt,
                        "answer": text,
                        "gt_answer": row.get("answer"),
                        "image": row.get("image"),
                        "image_id": row.get("image"),
                        "model_id": "mistral_llava_med_1.5",
                        "metadata": {
                            "method": args.method,
                            "backend_protocol": PROTOCOL_VERSION,
                            "prompt_sha256": prompt_item["sha256"],
                            "conv_mode": args.conv_mode,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            output.flush()
    temporary_answers.replace(args.answers_file)
    activation = {
        "protocol_version": PROTOCOL_VERSION,
        "method": args.method,
        "dataset": args.dataset,
        "conv_mode": args.conv_mode,
        "processed_samples": len(rows),
        "ordered_prompt_sha256": manifest["ordered_prompt_sha256"],
        "generation": generation_config(args.method, args.max_new_tokens),
        "counters": counters,
    }
    if args.method == "OPERA":
        activation["key_positions"] = key_position_audit
    elif args.method == "PAI":
        activation["pai_audits"] = pai_audits
    elif args.method == "M3ID":
        activation["m3id_audits"] = m3id_audits
    elif args.method == "AVISC":
        activation["avisc_audits"] = avisc_audits
    validate_activation_report(
        activation,
        method=args.method,
        expected_samples=len(rows),
        expected_prompt_sha256=manifest["ordered_prompt_sha256"],
        expected_max_new_tokens=args.max_new_tokens,
    )
    atomic_json(args.activation_file, activation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iuxray", "mimic", "harvard"), required=True)
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--answers-file", type=Path, required=True)
    parser.add_argument("--activation-file", type=Path, required=True)
    parser.add_argument("--conv-mode", default=CONV_MODE)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise RuleMitigationBackendError("--max-new-tokens must be positive")
    _generate(args)


if __name__ == "__main__":
    main()
