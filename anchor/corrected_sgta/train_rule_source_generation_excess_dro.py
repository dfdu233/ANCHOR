#!/usr/bin/env python3
"""Train class-balanced generation-aligned excess-risk Source-DRO.

One rank-16 residual is capped at 2%. Domain selection uses adapted minus
frozen-identity autoregressive NLL, cancelling intrinsic domain difficulty.
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
from corrected_sgta.rule_source_generation_excess_dro import (
    DOMAIN_BATCH_SIZE, MAX_RELATIVE_UPDATE, RANK, SOURCE_DOMAINS,
    TRAIN_IMAGES_PER_SOURCE, VERSION, experiment_fingerprint,
    mean_domain_losses, select_worst_domain,
)
from corrected_sgta.rule_source_preference import (
    LinearLowRankResidual,
    file_sha256,
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


LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 1.0
SEED = 42


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
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("rule_source_generation_excess_dro.py"),
        Path(__file__).with_name("rule_source_preference.py"),
        Path(__file__).with_name("train_rule_dg_adapter.py"),
        Path(__file__).with_name("train_rule_source_group_adapter.py"),
        Path(__file__).with_name("models_alignment.py"),
    ]
    return {str(path): file_sha256(path) for path in paths}


def serializable_config(
    args: argparse.Namespace,
    source_jsons: dict[str, Path],
    source_roots: dict[str, Path],
) -> dict[str, Any]:
    return {
        "source_manifest": str(args.source_manifest),
        "source_json": {
            name: str(path) for name, path in sorted(source_jsons.items())
        },
        "source_image_root": {
            name: str(path) for name, path in sorted(source_roots.items())
        },
        "locked_test": str(args.locked_test),
        "objective": "max_domain_class_balanced_autoregressive_excess_nll",
        "domain_aggregation": "worst_class_balanced_excess_mean",
        "domain_batch_size": DOMAIN_BATCH_SIZE,
        "class_balance": "two_yes_two_no_per_domain",
        "domain_reference": "frozen_identity_loss_stop_gradient",
        "rank": RANK,
        "max_relative_update": MAX_RELATIVE_UPDATE,
        "images_per_source": TRAIN_IMAGES_PER_SOURCE,
        "total_source_examples": TRAIN_IMAGES_PER_SOURCE * SOURCE_DOMAINS,
        "steps": TRAIN_IMAGES_PER_SOURCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "seed": SEED,
    }


def example_loss(
    adapter: LlavaMedAlignmentAdapter,
    image: Image.Image,
    prompt: str,
    answer: str,
    module: LinearLowRankResidual | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, labels = build_teacher_forcing(
        adapter, prompt, canonical_answer(answer)
    )
    loss, logits = sequence_forward(
        adapter, image, input_ids, labels, module, adapter_location="post"
    )
    return loss, sequence_log_probability(
        logits, target_ids_from_labels(labels)
    )

def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def main() -> None:
    args = parse_args()
    if args.save_every <= 0:
        raise ValueError("save-every must be positive")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    source_jsons = parse_named_paths(args.source_json, "--source-json")
    source_roots = parse_named_paths(
        args.source_image_root, "--source-image-root"
    )
    if (
        set(source_jsons) != set(source_roots)
        or len(source_jsons) != SOURCE_DOMAINS
    ):
        raise ValueError("exactly three matching source JSON/root domains required")
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
            TRAIN_IMAGES_PER_SOURCE,
            SEED,
        )
        for name in sorted(source_jsons)
    }
    if any(len(rows) != TRAIN_IMAGES_PER_SOURCE for rows in groups.values()):
        raise ValueError("each source must provide exactly 95 selected train images")
    names = sorted(groups)
    class_groups = {
        name: {
            label: [row for row in groups[name] if canonical_answer(row["answer"]) == label]
            for label in ("Yes.", "No.")
        }
        for name in names
    }
    if any(not rows for values in class_groups.values() for rows in values.values()):
        raise ValueError("each source must contain both Yes and No examples")
    schedule = [
        {
            name: [
                class_groups[name][label][(step * 2 + offset) % len(class_groups[name][label])]
                for label in ("Yes.", "No.")
                for offset in range(2)
            ]
            for name in names
        }
        for step in range(TRAIN_IMAGES_PER_SOURCE)
    ]
    selected = {
        name: [
            {
                "id": row["id"],
                "image": row["image"],
                "question": row["question"],
                "answer": canonical_answer(row["answer"]),
                "image_sha256": file_sha256(Path(row["image"])),
            }
            for row in groups[name]
        ]
        for name in names
    }
    fingerprint, fingerprint_payload = experiment_fingerprint(
        manifest_contract=manifest_contract,
        config=serializable_config(args, source_jsons, source_roots),
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
    module = LinearLowRankResidual(
        width, RANK, MAX_RELATIVE_UPDATE
    ).to(adapter.model.device)
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history: list[dict[str, Any]] = []
    start = 0
    if args.resume and args.output.is_file():
        payload = torch.load(
            args.output, map_location="cpu", weights_only=False
        )
        if (
            payload.get("version") != VERSION
            or payload.get("fingerprint") != fingerprint
        ):
            raise RuntimeError("resume checkpoint fingerprint mismatch")
        module.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
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
                "rank": RANK,
                "max_relative_update": MAX_RELATIVE_UPDATE,
                "prompt_protocol": "rule_mimic",
                "objective": "class_balanced_autoregressive_excess_nll",
                "domain_aggregation": "worst_class_balanced_excess_mean",
                "state_dict": _cpu_state(module),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "next_step": next_step,
                "target_labels_accessed": False,
            },
            args.output,
        )

    progress = tqdm(range(start, len(schedule)), desc="source-generation-excess-dro")
    try:
        for step in progress:
            batch: dict[str, list[tuple[Image.Image, str, str]]] = {}
            domain_item_losses: dict[str, list[torch.Tensor]] = {}
            with torch.no_grad():
                for name in names:
                    items: list[tuple[Image.Image, str, str]] = []
                    losses: list[torch.Tensor] = []
                    for row in schedule[step][name]:
                        with Image.open(row["image"]) as handle:
                            image = handle.convert("RGB")
                        prompt = rule_mimic_prompt(row["question"])
                        answer = canonical_answer(row["answer"])
                        item_loss, _ = example_loss(
                            adapter, image, prompt, answer, module
                        )
                        identity_loss, _ = example_loss(
                            adapter, image, prompt, answer, None
                        )
                        items.append((image, prompt, answer))
                        losses.append((item_loss - identity_loss).detach())
                    batch[name] = items
                    domain_item_losses[name] = losses
            detached_losses = mean_domain_losses(domain_item_losses)

            # Accumulate the worst-domain mean gradient one item at a time to
            # preserve the exact minibatch gradient without four LLM graphs.
            worst = select_worst_domain(detached_losses)
            optimizer.zero_grad(set_to_none=True)
            active_values: list[tuple[torch.Tensor, torch.Tensor]] = []
            for image, prompt, answer in batch[worst]:
                values = example_loss(adapter, image, prompt, answer, module)
                (values[0] / DOMAIN_BATCH_SIZE).backward()
                active_values.append(tuple(value.detach() for value in values))
            loss = torch.stack([item[0] for item in active_values]).mean()
            answer_log_probability = torch.stack([item[1] for item in active_values]).mean()
            gradient = torch.nn.utils.clip_grad_norm_(
                module.parameters(), GRADIENT_CLIP
            )
            if not math.isfinite(float(gradient)):
                raise FloatingPointError("non-finite source-DRO gradient")
            optimizer.step()

            item = {
                "step": step,
                "worst_domain": worst,
                "loss": float(loss.detach()),
                "answer_log_probability": float(answer_log_probability.detach()),
                "target_labels": [answer for _, _, answer in batch[worst]],
                "gradient_norm": float(gradient),
                "mean_relative_update": module.last_mean_relative_norm,
                "maximum_relative_update": module.last_max_relative_norm,
                "detached_domain_excess_risks": {
                    name: float(value) for name, value in detached_losses.items()
                },
            }
            history.append(item)
            progress.set_postfix(loss=f"{item['loss']:.4f}", worst=worst)
            next_step = step + 1
            if next_step % args.save_every == 0 or next_step == len(schedule):
                save(next_step)
            del batch, domain_item_losses, detached_losses
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
                "objective": "class_balanced_autoregressive_excess_nll",
                "domain_aggregation": "worst_class_balanced_excess_mean",
                "source_sizes": {
                    name: len(groups[name]) for name in names
                },
                "total_source_examples": sum(map(len, groups.values())),
                "steps_complete": len(history),
                "target_labels_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
