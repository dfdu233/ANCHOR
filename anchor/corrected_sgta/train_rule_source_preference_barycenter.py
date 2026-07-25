#!/usr/bin/env python3
"""Train source-specific and pooled full-sequence preference adapters.

All VLM parameters remain frozen.  Each source-specific module is optimized
only on its own source; the pooled control consumes the identical balanced
schedule.  Both use the exact RULE MIMIC no-reference prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_source_preference import (
    VERSION,
    LinearLowRankResidual,
    canonical_binary_answer,
    experiment_fingerprint,
    file_sha256,
    opposite_binary_answer,
    preference_improvement_loss,
    pooled_preference_objective,
    rule_mimic_prompt,
    sequence_log_probability,
    target_ids_from_labels,
    validate_source_manifest,
)
from corrected_sgta.train_rule_dg_adapter import (
    atomic_torch_save,
    build_teacher_forcing,
    canonical_answer,
    projector_output_width,
    sequence_forward,
)
from corrected_sgta.train_rule_source_group_adapter import (
    balanced_schedule,
    normalize_source_rows,
    parse_named_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--source-json", action="append", required=True, metavar="DOMAIN=PATH"
    )
    parser.add_argument(
        "--source-image-root",
        action="append",
        required=True,
        metavar="DOMAIN=PATH",
    )
    parser.add_argument("--locked-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--training-mode",
        choices=("per_source", "pooled", "both"),
        default="both",
    )
    parser.add_argument(
        "--pooled-aggregation", choices=("mean", "worst"), default="mean"
    )
    parser.add_argument("--max-images-per-source", type=int, default=95)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="0 uses the smallest selected source exactly once",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-relative-update", type=float, default=0.02)
    parser.add_argument("--preference-beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {"output", "resume"}
    return {
        key: (
            [str(item) for item in value]
            if isinstance(value, list)
            else str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in vars(args).items()
        if key not in excluded
    }


def code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("rule_source_preference.py"),
        Path(__file__).with_name("train_rule_dg_adapter.py"),
        Path(__file__).with_name("train_rule_source_group_adapter.py"),
        Path(__file__).with_name("models_alignment.py"),
    ]
    return {str(path): file_sha256(path) for path in paths}


def prepare_candidates(
    adapter: LlavaMedAlignmentAdapter, prompt: str, answer: str
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    positive = canonical_binary_answer(canonical_answer(answer))
    negative = opposite_binary_answer(positive)
    return {
        "positive": build_teacher_forcing(adapter, prompt, positive),
        "negative": build_teacher_forcing(adapter, prompt, negative),
    }


def candidate_log_probability(
    adapter: LlavaMedAlignmentAdapter,
    image: Image.Image,
    candidate: tuple[torch.Tensor, torch.Tensor],
    module: LinearLowRankResidual | None,
) -> torch.Tensor:
    input_ids, labels = candidate
    _, logits = sequence_forward(
        adapter,
        image,
        input_ids,
        labels,
        module,
        adapter_location="post",
    )
    targets = target_ids_from_labels(labels)
    return sequence_log_probability(logits, targets)


def candidate_margin(
    adapter: LlavaMedAlignmentAdapter,
    image: Image.Image,
    candidates: dict[str, tuple[torch.Tensor, torch.Tensor]],
    module: LinearLowRankResidual | None,
) -> torch.Tensor:
    positive = candidate_log_probability(
        adapter, image, candidates["positive"], module
    )
    negative = candidate_log_probability(
        adapter, image, candidates["negative"], module
    )
    return positive - negative


def preference_loss_from_reference_margin(
    adapter: LlavaMedAlignmentAdapter,
    image: Image.Image,
    candidates: dict[str, tuple[torch.Tensor, torch.Tensor]],
    module: LinearLowRankResidual,
    reference_margin: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = candidate_log_probability(
        adapter, image, candidates["positive"], module
    )
    negative = candidate_log_probability(
        adapter, image, candidates["negative"], module
    )
    zero = reference_margin.new_zeros(())
    loss, margin, improvement = preference_improvement_loss(
        positive, negative, reference_margin, zero, beta
    )
    return loss, margin, improvement


def _optimizer(
    module: LinearLowRankResidual, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        module.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def main() -> None:
    args = parse_args()
    if args.rank <= 0 or args.max_relative_update <= 0:
        raise ValueError("rank and maximum relative update must be positive")
    if args.preference_beta <= 0 or args.learning_rate <= 0:
        raise ValueError("preference beta and learning rate must be positive")
    if args.weight_decay < 0 or args.gradient_clip <= 0:
        raise ValueError("weight decay must be nonnegative and clipping positive")
    if args.max_images_per_source <= 0 or args.steps < 0 or args.save_every <= 0:
        raise ValueError("sample count/save interval must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_jsons = parse_named_paths(args.source_json, "--source-json")
    source_roots = parse_named_paths(
        args.source_image_root, "--source-image-root"
    )
    if set(source_jsons) != set(source_roots) or len(source_jsons) < 2:
        raise ValueError("source JSON/root domains must match and contain >=2 domains")
    for path in [args.source_manifest, args.locked_test, *source_jsons.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in source_roots.values():
        if not path.is_dir():
            raise FileNotFoundError(path)

    manifest_contract = validate_source_manifest(
        args.source_manifest.resolve(),
        source_jsons,
        args.locked_test.resolve(),
    )
    groups = {
        name: normalize_source_rows(
            name,
            source_jsons[name],
            source_roots[name],
            args.max_images_per_source,
            args.seed,
        )
        for name in sorted(source_jsons)
    }
    schedule = balanced_schedule(groups, args.steps)
    selected = {
        name: [
            {
                "id": row["id"],
                "image": row["image"],
                "question": row["question"],
                "answer": canonical_binary_answer(canonical_answer(row["answer"])),
            }
            for row in rows
        ]
        for name, rows in sorted(groups.items())
    }
    fingerprint, fingerprint_payload = experiment_fingerprint(
        manifest_contract=manifest_contract,
        config=serializable_config(args),
        selected=selected,
        code_sha256=code_hashes(),
    )
    if args.output.exists() and not args.resume:
        raise FileExistsError("output exists; use --resume only for identical run")

    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    width = projector_output_width(adapter.model)
    names = sorted(groups)

    source_modules = torch.nn.ModuleDict()
    source_optimizers: dict[str, torch.optim.Optimizer] = {}
    if args.training_mode in {"per_source", "both"}:
        for name in names:
            module = LinearLowRankResidual(
                width, args.rank, args.max_relative_update
            ).to(adapter.model.device)
            source_modules[name] = module
            source_optimizers[name] = _optimizer(
                module, args.learning_rate, args.weight_decay
            )
    pooled_module = None
    pooled_optimizer = None
    if args.training_mode in {"pooled", "both"}:
        pooled_module = LinearLowRankResidual(
            width, args.rank, args.max_relative_update
        ).to(adapter.model.device)
        pooled_optimizer = _optimizer(
            pooled_module, args.learning_rate, args.weight_decay
        )

    history: list[dict[str, Any]] = []
    start = 0
    if args.resume and args.output.is_file():
        payload = torch.load(
            args.output, map_location="cpu", weights_only=False
        )
        if payload.get("version") != VERSION or payload.get("fingerprint") != fingerprint:
            raise RuntimeError("resume checkpoint fingerprint mismatch")
        if source_modules:
            stored = payload.get("per_source_state_dict")
            if set(stored or {}) != set(source_modules):
                raise RuntimeError("resume source-module set mismatch")
            for name in names:
                source_modules[name].load_state_dict(stored[name])
                source_optimizers[name].load_state_dict(
                    payload["per_source_optimizer"][name]
                )
        if pooled_module is not None:
            if payload.get("pooled_state_dict") is None:
                raise RuntimeError("resume checkpoint lacks pooled module")
            pooled_module.load_state_dict(payload["pooled_state_dict"])
            assert pooled_optimizer is not None
            pooled_optimizer.load_state_dict(payload["pooled_optimizer"])
        history = list(payload["history"])
        start = int(payload["next_step"])

    def save(next_step: int) -> None:
        atomic_torch_save(
            {
                "version": VERSION,
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "manifest_contract": manifest_contract,
                "width": width,
                "rank": args.rank,
                "max_relative_update": args.max_relative_update,
                "prompt_protocol": "rule_mimic",
                "training_mode": args.training_mode,
                "pooled_aggregation": args.pooled_aggregation,
                "per_source_state_dict": (
                    {
                        name: _cpu_state(source_modules[name])
                        for name in names
                    }
                    if source_modules
                    else None
                ),
                "pooled_state_dict": (
                    _cpu_state(pooled_module)
                    if pooled_module is not None
                    else None
                ),
                "per_source_optimizer": (
                    {
                        name: source_optimizers[name].state_dict()
                        for name in names
                    }
                    if source_optimizers
                    else None
                ),
                "pooled_optimizer": (
                    pooled_optimizer.state_dict()
                    if pooled_optimizer is not None
                    else None
                ),
                "history": history,
                "next_step": next_step,
            },
            args.output,
        )

    progress = tqdm(range(start, len(schedule)), desc="source-preference")
    try:
        for step in progress:
            examples: dict[
                str,
                tuple[
                    Image.Image,
                    dict[str, tuple[torch.Tensor, torch.Tensor]],
                    torch.Tensor,
                ],
            ] = {}
            for name in names:
                row = schedule[step][name]
                with Image.open(row["image"]) as handle:
                    image = handle.convert("RGB")
                prompt = rule_mimic_prompt(row["question"])
                candidates = prepare_candidates(
                    adapter, prompt, row["answer"]
                )
                with torch.no_grad():
                    reference_margin = candidate_margin(
                        adapter, image, candidates, None
                    ).detach()
                examples[name] = (image, candidates, reference_margin)

            source_metrics: dict[str, dict[str, float]] = {}
            if source_modules:
                for name in names:
                    module = source_modules[name]
                    optimizer = source_optimizers[name]
                    image, candidates, reference_margin = examples[name]
                    optimizer.zero_grad(set_to_none=True)
                    loss, margin, improvement = (
                        preference_loss_from_reference_margin(
                            adapter,
                            image,
                            candidates,
                            module,
                            reference_margin,
                            args.preference_beta,
                        )
                    )
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(
                        module.parameters(), args.gradient_clip
                    )
                    if not math.isfinite(float(gradient)):
                        raise FloatingPointError("non-finite source gradient")
                    optimizer.step()
                    source_metrics[name] = {
                        "loss": float(loss.detach()),
                        "reference_margin": float(reference_margin),
                        "adapted_margin": float(margin.detach()),
                        "margin_improvement": float(improvement.detach()),
                        "gradient_norm": float(gradient),
                        "mean_relative_update": module.last_mean_relative_norm,
                        "max_relative_update": module.last_max_relative_norm,
                    }

            pooled_metrics = None
            if pooled_module is not None:
                assert pooled_optimizer is not None
                pooled_optimizer.zero_grad(set_to_none=True)
                domain_losses = []
                domain_values = {}
                for name in names:
                    image, candidates, reference_margin = examples[name]
                    loss, margin, improvement = (
                        preference_loss_from_reference_margin(
                            adapter,
                            image,
                            candidates,
                            pooled_module,
                            reference_margin,
                            args.preference_beta,
                        )
                    )
                    domain_losses.append(loss)
                    if args.pooled_aggregation == "mean":
                        # Exact mean gradient, while freeing each full-LLM
                        # backward graph before processing the next domain.
                        (loss / len(names)).backward()
                    domain_values[name] = {
                        "loss": float(loss.detach()),
                        "reference_margin": float(reference_margin),
                        "adapted_margin": float(margin.detach()),
                        "margin_improvement": float(improvement.detach()),
                    }
                if args.pooled_aggregation == "mean":
                    pooled_loss = pooled_preference_objective(
                        [loss.detach() for loss in domain_losses],
                        "mean",
                    )
                else:
                    pooled_loss = pooled_preference_objective(domain_losses, "worst")
                    pooled_loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    pooled_module.parameters(), args.gradient_clip
                )
                if not math.isfinite(float(gradient)):
                    raise FloatingPointError("non-finite pooled gradient")
                pooled_optimizer.step()
                pooled_metrics = {
                    "loss": float(pooled_loss.detach()),
                    "gradient_norm": float(gradient),
                    "domains": domain_values,
                    "mean_relative_update": pooled_module.last_mean_relative_norm,
                    "max_relative_update": pooled_module.last_max_relative_norm,
                }

            item = {
                "step": step,
                "per_source": source_metrics,
                "pooled": pooled_metrics,
            }
            history.append(item)
            visible_losses = [
                value["loss"] for value in source_metrics.values()
            ]
            if pooled_metrics is not None:
                visible_losses.append(pooled_metrics["loss"])
            progress.set_postfix(
                loss=f"{float(np.mean(visible_losses)):.4f}"
            )
            next_step = step + 1
            if next_step % args.save_every == 0 or next_step == len(schedule):
                save(next_step)
            del examples
            torch.cuda.empty_cache()
    finally:
        adapter.close()

    print(
        json.dumps(
            {
                "output": str(args.output),
                "version": VERSION,
                "fingerprint": fingerprint,
                "manifest_fingerprint": manifest_contract[
                    "manifest_fingerprint"
                ],
                "training_mode": args.training_mode,
                "source_sizes": {
                    name: len(groups[name]) for name in names
                },
                "steps_complete": len(history),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
