"""Build centers and train paired Center-Native Qwen2.5-VL models."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from anchor.corrected_sgta.center_native_qwen import (
    StreamingAmplitudeCenter,
    install_qwen_patch_center,
    load_feature_center,
    save_feature_center,
    set_qwen_center_context,
)

VERSION = "center-native-qwen-train-v2"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pad_392(source: str | Path | bytes) -> Image.Image:
    handle_source = io.BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(handle_source) as handle:
        image = handle.convert("RGB")
    image.thumbnail((392, 392), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (392, 392), (0, 0, 0))
    canvas.paste(image, ((392 - image.width) // 2, (392 - image.height) // 2))
    return canvas


def question_answer(record: dict[str, Any]) -> tuple[str, str]:
    turns = record["conversations"]
    user = next(turn["content"] for turn in turns if turn["role"] in {"user", "human"})
    answer = next(turn["content"] for turn in reversed(turns) if turn["role"] in {"assistant", "gpt"})
    return str(user).replace("<image>", "").strip(), str(answer).strip()


def messages_for(question: str, answer: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


class ManifestDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows = read_jsonl(path)
        parquet_paths = {row.get("source_parquet") for row in self.rows if row.get("source_parquet")}
        self.image_bytes: dict[str, Any] = {}
        for parquet_path in parquet_paths:
            self.image_bytes[str(parquet_path)] = pq.read_table(
                parquet_path, columns=["image_bytes"]
            )["image_bytes"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        if row.get("source_parquet"):
            row["image_bytes"] = self.image_bytes[str(row["source_parquet"])][
                int(row["parquet_row_index"])
            ].as_py()
        return row


class QwenCollator:
    def __init__(self, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        images, full_texts, prompt_texts = [], [], []
        for record in records:
            question, answer = question_answer(record)
            images.append(pad_392(record.get("image_bytes", record.get("image"))))
            full_texts.append(
                self.processor.apply_chat_template(
                    messages_for(question, answer), tokenize=False, add_generation_prompt=False
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    messages_for(question), tokenize=False, add_generation_prompt=True
                )
            )
        full = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompts = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = full["input_ids"].clone()
        # Qwen left-pads by default.  Mask every token before the assistant
        # completion using the non-padding prompt length for each record.
        for index in range(len(records)):
            full_nonpad = int(full["attention_mask"][index].sum())
            prompt_nonpad = int(prompts["attention_mask"][index].sum())
            start = labels.shape[1] - full_nonpad
            labels[index, : start + prompt_nonpad] = -100
        labels[full["attention_mask"] == 0] = -100
        full["labels"] = labels
        full["anchor_is_cxr"] = torch.tensor(
            [bool(record.get("is_strict_cxr")) for record in records], dtype=torch.bool
        )
        full["anchor_ids"] = [record["id"] for record in records]
        return full


def load_model(model_path: Path, train: bool) -> Qwen2_5_VLForConditionalGeneration:
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "local_files_only": True,
    }
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    install_qwen_patch_center(model)
    model.config.use_cache = not train
    return model


def merger_parameters(model: torch.nn.Module):
    for name, parameter in model.named_parameters():
        if ".visual.merger." in name or name.startswith("visual.merger."):
            yield name, parameter


def freeze_all(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def enable_merger(model: torch.nn.Module) -> None:
    found = 0
    for _, parameter in merger_parameters(model):
        parameter.requires_grad = True
        found += parameter.numel()
    if not found:
        raise RuntimeError("Qwen visual merger parameters were not found")


def early_vision_parameters(model: torch.nn.Module, blocks: int):
    """Yield the patch embed and the first ``blocks`` visual transformer blocks."""
    prefixes = ("visual.patch_embed.",) + tuple(
        f"visual.blocks.{index}." for index in range(blocks)
    )
    for name, parameter in model.named_parameters():
        if any(prefix in name for prefix in prefixes):
            yield name, parameter


def enable_early_vision(model: torch.nn.Module, blocks: int) -> None:
    if blocks <= 0:
        return
    found = 0
    for _, parameter in early_vision_parameters(model, blocks):
        parameter.requires_grad = True
        found += parameter.numel()
    if not found:
        raise RuntimeError(
            f"Qwen patch_embed / first {blocks} visual blocks were not found"
        )


def save_early_vision(model: torch.nn.Module, path: Path, blocks: int) -> None:
    state = {
        name: parameter.detach().cpu()
        for name, parameter in early_vision_parameters(model, blocks)
    }
    if blocks > 0 and not state:
        raise RuntimeError("refusing to save an empty early-vision checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"blocks": blocks, "state": state}, path)


def load_early_vision(model: torch.nn.Module, path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    blocks = int(payload["blocks"])
    state = payload["state"]
    current = dict(model.named_parameters())
    missing = [name for name in state if name not in current]
    if missing:
        raise RuntimeError(f"early-vision checkpoint keys are missing: {missing[:5]}")
    for name, value in state.items():
        current[name].data.copy_(value.to(dtype=current[name].dtype))
    return blocks


def save_merger(model: torch.nn.Module, path: Path) -> None:
    state = {name: parameter.detach().cpu() for name, parameter in merger_parameters(model)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_merger(model: torch.nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(name in state for name in missing):
        raise RuntimeError(f"invalid merger checkpoint: missing={missing}, unexpected={unexpected}")


def make_loader(dataset: Dataset, collator: Any, batch: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        generator=generator,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collator,
    )


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trainable_state": {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "config": config,
    }


def train_loop(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, use_fast=False
    )
    dataset = ManifestDataset(args.manifest)
    collator = QwenCollator(processor, args.max_length)
    batches_per_epoch = int(np.ceil(len(dataset) / args.micro_batch))
    model = load_model(args.model, train=True)
    freeze_all(model)
    if args.stage == "alignment":
        enable_merger(model)
    else:
        if args.alignment_merger:
            load_merger(model, args.alignment_merger)
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora)
        install_qwen_patch_center(model.base_model.model)
        enable_merger(model)
        enable_early_vision(model, args.early_vision_blocks)
    center = load_feature_center(args.center) if args.center else None
    model.gradient_checkpointing_enable()
    model.to("cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    early_ids = {
        id(parameter)
        for _, parameter in early_vision_parameters(model, args.early_vision_blocks)
        if parameter.requires_grad
    }
    if args.early_vision_blocks:
        early = [parameter for parameter in trainable if id(parameter) in early_ids]
        remaining = [parameter for parameter in trainable if id(parameter) not in early_ids]
        optimizer = AdamW(
            [
                {"params": remaining, "lr": args.learning_rate},
                {
                    "params": early,
                    "lr": args.early_vision_learning_rate,
                },
            ],
            weight_decay=0.01,
        )
    else:
        optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    total_steps = args.max_steps or (batches_per_epoch * args.epochs // args.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * 0.03)), total_steps
    )
    config = {
        "version": VERSION,
        "stage": args.stage,
        "branch": args.branch,
        "model": str(args.model.resolve()),
        "model_contract": (
            str(args.model_contract.resolve()) if args.model_contract else None
        ),
        "model_contract_sha256": (
            file_sha256(args.model_contract) if args.model_contract else None
        ),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "seed": args.seed,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "center": str(args.center.resolve()) if args.center else None,
        "center_sha256": file_sha256(args.center) if args.center else None,
        "tau": args.tau,
        "center_probability": args.center_probability,
        "alignment_merger": (
            str(args.alignment_merger.resolve()) if args.alignment_merger else None
        ),
        "alignment_merger_sha256": (
            file_sha256(args.alignment_merger) if args.alignment_merger else None
        ),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "early_vision_blocks": args.early_vision_blocks,
        "early_vision_learning_rate": args.early_vision_learning_rate,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (output / "config.json").write_text(json.dumps(config, indent=2))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro = 0
    start_epoch = 0
    start_batch = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["config"]["fingerprint"] != config["fingerprint"]:
            raise RuntimeError("resume checkpoint fingerprint differs from current configuration")
        missing, unexpected = model.load_state_dict(state["trainable_state"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected trainable resume keys: {unexpected}")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        start_epoch = int(state["epoch"])
        start_batch = int(state["batch_in_epoch"]) + 1
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    started = time.time()
    log_path = output / "train_metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        loader = make_loader(dataset, collator, args.micro_batch, args.seed + epoch)
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            if step >= total_steps:
                break
            anchor_ids = batch.pop("anchor_ids")
            is_cxr = batch.pop("anchor_is_cxr")
            if args.branch == "center":
                choices = [
                    bool(flag) and random.Random(f"{args.seed}:{epoch}:{sample_id}").random() < args.center_probability
                    for flag, sample_id in zip(is_cxr.tolist(), anchor_ids)
                ]
                set_qwen_center_context(model, center, args.tau, torch.tensor(choices, device="cuda"))
            else:
                choices = [False] * len(anchor_ids)
                set_qwen_center_context(model, None)
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            supervised_tokens = [
                int((labels != -100).sum()) for labels in batch["labels"]
            ]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                raw_loss = model(**batch).loss
                if not torch.isfinite(raw_loss):
                    diagnostic = {
                        "error": "non_finite_loss",
                        "step_before_update": step,
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "sample_ids": anchor_ids,
                        "center_choices": choices,
                        "supervised_tokens": supervised_tokens,
                    }
                    with (output / "invalid_batches.jsonl").open("a") as handle:
                        handle.write(json.dumps(diagnostic) + "\n")
                    if sum(supervised_tokens) != 0:
                        raise FloatingPointError(json.dumps(diagnostic))
                    # Right truncation can exceptionally remove every answer
                    # token from a long PubMed prompt.  Preserve the paired
                    # optimizer/scheduler budget with an explicit zero-loss
                    # microbatch instead of allowing CrossEntropy's empty mean
                    # to poison the checkpoint with NaNs.
                    raw_loss = next(
                        parameter for parameter in trainable if parameter.requires_grad
                    ).sum() * 0.0
                loss = raw_loss / args.grad_accum
            loss.backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.detach()) * args.grad_accum,
                    "lr": scheduler.get_last_lr()[0],
                    "centered_fraction": float(np.mean(choices)),
                    "elapsed_seconds": time.time() - started,
                    "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if step % args.save_every == 0:
                if args.stage == "alignment":
                    save_merger(model, output / f"merger-step-{step}.pt")
                else:
                    model.save_pretrained(output / f"adapter-step-{step}")
                    save_merger(model, output / f"merger-step-{step}.pt")
                    if args.early_vision_blocks:
                        save_early_vision(
                            model,
                            output / f"early-vision-step-{step}.pt",
                            args.early_vision_blocks,
                        )
                torch.save(
                    checkpoint_payload(
                        model, optimizer, scheduler, step, epoch, batch_index, config
                    ),
                    # Checkpoints occur only at optimizer boundaries, so
                    # restarting at the next batch exactly preserves draws.
                    output / f"trainer-step-{step}.pt",
                )
        if step >= total_steps:
            break
    if args.stage == "alignment":
        save_merger(model, output / "merger-final.pt")
    else:
        model.save_pretrained(output / "adapter-final")
        save_merger(model, output / "merger-final.pt")
        if args.early_vision_blocks:
            save_early_vision(
                model, output / "early-vision-final.pt", args.early_vision_blocks
            )
    (output / "COMPLETED").write_text(json.dumps({"step": step, "fingerprint": config["fingerprint"]}))


@torch.inference_mode()
def build_center(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, use_fast=False
    )
    model = load_model(args.model, train=False).to("cuda").eval()
    rows = [row for row in read_jsonl(args.manifest) if row.get("is_strict_cxr")]
    if args.half in {"a", "b"}:
        parity = 0 if args.half == "a" else 1
        rows = [row for row in rows if int(row["image_sha256"], 16) % 2 == parity]
    rows = rows[: args.max_images] if args.max_images else rows
    parquet_paths = {row.get("source_parquet") for row in rows if row.get("source_parquet")}
    image_columns = {
        str(path): pq.read_table(path, columns=["image_bytes"])["image_bytes"]
        for path in parquet_paths
    }
    log_builder = arithmetic_builder = None
    observed_grid = None
    for index, row in enumerate(rows):
        if "source_parquet" in row:
            image = pad_392(
                image_columns[str(row["source_parquet"])][
                    int(row["parquet_row_index"])
                ].as_py()
            )
        else:
            image = pad_392(row["image"])
        text = processor.apply_chat_template(
            messages_for("Describe this chest radiograph."), tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        grid = tuple(int(value) for value in inputs["image_grid_thw"][0].tolist())
        # The center lives immediately after patch_embed.  Running the frozen
        # transformer blocks here would be mathematically irrelevant and made
        # large-bank construction needlessly expensive.
        tokens = model.visual.patch_embed(
            inputs["pixel_values"].to("cuda", torch.bfloat16)
        ).detach()
        if log_builder is None:
            observed_grid = grid
            channels = int(tokens.shape[-1])
            log_builder = StreamingAmplitudeCenter("log", grid, channels)
            arithmetic_builder = StreamingAmplitudeCenter("arithmetic", grid, channels)
        if grid != observed_grid:
            raise RuntimeError(f"fixed preprocessing produced multiple grids: {observed_grid} vs {grid}")
        log_builder.update(tokens)
        arithmetic_builder.update(tokens)
        if (index + 1) % 100 == 0:
            print(json.dumps({"center_images": index + 1, "grid": grid}), flush=True)
    metadata = {
        "version": VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "half": args.half,
        "preprocess": "aspect-preserving pad 392x392",
        "location": "post-patch_embed pre-window-reorder",
        "target_data_accessed": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_feature_center(log_builder.finalize(metadata), args.output / f"cxr_log_{args.half}.pt")
    save_feature_center(
        arithmetic_builder.finalize(metadata), args.output / f"cxr_arithmetic_{args.half}.pt"
    )
    (args.output / f"COMPLETED_{args.half}").write_text(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    center = sub.add_parser("build-center")
    center.add_argument("--model", type=Path, required=True)
    center.add_argument("--manifest", type=Path, required=True)
    center.add_argument("--output", type=Path, required=True)
    center.add_argument("--half", choices=["all", "a", "b"], default="all")
    center.add_argument("--max-images", type=int)
    center.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train")
    train.add_argument("--model", type=Path, required=True)
    train.add_argument("--model-contract", type=Path)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--stage", choices=["alignment", "sft"], required=True)
    train.add_argument("--branch", choices=["clean", "center"], default="clean")
    train.add_argument("--alignment-merger", type=Path)
    train.add_argument("--center", type=Path)
    train.add_argument("--tau", type=float, default=0.25)
    train.add_argument("--center-probability", type=float, default=0.5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--micro-batch", type=int, default=4)
    train.add_argument("--grad-accum", type=int, default=4)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--max-length", type=int, default=512)
    train.add_argument("--learning-rate", type=float, required=True)
    train.add_argument(
        "--early-vision-blocks",
        type=int,
        choices=[0, 2],
        default=0,
        help="Bounded fallback: also train patch_embed and visual blocks 0..N-1.",
    )
    train.add_argument(
        "--early-vision-learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for the bounded early-vision parameter group.",
    )
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--save-every", type=int, default=250)
    train.add_argument("--resume", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "build-center":
        build_center(args)
    else:
        train_loop(args)
