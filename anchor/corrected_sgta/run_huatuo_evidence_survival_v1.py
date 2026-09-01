#!/usr/bin/env python3
"""Measure reader-grounded lesion-evidence survival under progressive ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    VERBALIZERS,
    append_jsonl,
    atomic_json,
    import_huatuo,
    label_ids,
    load_image,
    load_jsonl,
    prepared_embeddings,
    prompt_for,
    resolve_image,
    sha256_file,
)


VERSION = "huatuo-reader-evidence-survival-v1"


def square_grid_roi_mask(
    width: int,
    height: int,
    boxes: list[dict[str, Any]],
    token_count: int,
) -> np.ndarray:
    side = int(round(math.sqrt(token_count)))
    if side * side != token_count:
        raise ValueError(f"visual token count {token_count} is not a square grid")
    square = max(width, height)
    # Huatuo's expand2square pastes with integer floor offsets (cli.py), so the
    # ROI transform must use the same convention when the size difference is
    # odd.  A half-pixel offset can otherwise flip a boundary patch.
    offset_x = float((square - width) // 2)
    offset_y = float((square - height) // 2)
    x0 = np.arange(side) * square / side
    x1 = (np.arange(side) + 1) * square / side
    y0 = np.arange(side) * square / side
    y1 = (np.arange(side) + 1) * square / side
    mask = np.zeros((side, side), dtype=bool)
    for box in boxes:
        bx0 = float(box["x_min"]) + offset_x
        bx1 = float(box["x_max"]) + offset_x
        by0 = float(box["y_min"]) + offset_y
        by1 = float(box["y_max"]) + offset_y
        x_overlap = (x1 > bx0) & (x0 < bx1)
        y_overlap = (y1 > by0) & (y0 < by1)
        mask |= y_overlap[:, None] & x_overlap[None, :]
    return mask.reshape(-1)


def deterministic_subset(
    indices: np.ndarray,
    count: int,
    key: str,
    replicate: int,
) -> np.ndarray:
    ranked = sorted(
        (int(index) for index in indices),
        key=lambda index: hashlib.sha256(
            f"{VERSION}:{key}:{replicate}:{index}".encode()
        ).hexdigest(),
    )
    return np.asarray(ranked[:count], dtype=int)


@torch.inference_mode()
def forward_with_capture(
    bot: Any,
    embeddings: torch.Tensor,
    attention: torch.Tensor,
    positions: torch.Tensor | None,
    visual_span: tuple[int, int],
    layer: int,
) -> tuple[dict[str, float], torch.Tensor]:
    blocks = bot.model.model.layers
    if not 1 <= layer <= len(blocks):
        raise ValueError(f"layer must lie in 1..{len(blocks)}")
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["visual"] = hidden[0, visual_span[0] : visual_span[1]].detach().clone()

    handle = blocks[layer - 1].register_forward_hook(hook)
    try:
        output = bot.model.model(
            input_ids=None,
            attention_mask=attention,
            position_ids=positions,
            inputs_embeds=embeddings,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        handle.remove()
    return final_logits(bot, output.last_hidden_state[0, -1]), captured["visual"]


@torch.inference_mode()
def final_logits(bot: Any, normalized_hidden: torch.Tensor) -> dict[str, float]:
    ids = label_ids(bot)
    weight = bot.model.get_output_embeddings().weight
    token_ids = torch.tensor([ids[state] for state in VERBALIZERS], device=weight.device)
    # BF16 logits around magnitude 20 have a 0.125 quantization step, which is
    # too coarse for paired causal deltas.  The activations remain those of the
    # native model; only the three-token readout is accumulated in FP32.
    logits = normalized_hidden.float() @ weight.index_select(0, token_ids).float().T
    return {
        state: float(logits[index].float().cpu())
        for index, state in enumerate(VERBALIZERS)
    }


def coordinates(logits: dict[str, float]) -> dict[str, float]:
    polarity = float(logits["supported"]) - float(logits["refuted"])
    commitment = max(float(logits["supported"]), float(logits["refuted"])) - float(logits["undetermined"])
    return {"polarity": polarity, "commitment": commitment}


@torch.inference_mode()
def ablated_logits(
    bot: Any,
    embeddings: torch.Tensor,
    attention: torch.Tensor,
    positions: torch.Tensor | None,
    visual_span: tuple[int, int],
    layer: int,
    selected: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    blocks = bot.model.model.layers
    audit: dict[str, float] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        visual = modified[0, visual_span[0] : visual_span[1]]
        before = visual.index_select(0, torch.as_tensor(selected, device=visual.device)).float()
        mean = visual.float().mean(dim=0)
        direction = mean / mean.norm().clamp_min(1e-12)
        replacement = direction.unsqueeze(0) * before.norm(dim=-1, keepdim=True)
        visual[torch.as_tensor(selected, device=visual.device)] = replacement.to(visual.dtype)
        after = visual.index_select(0, torch.as_tensor(selected, device=visual.device)).float()
        audit.update(
            {
                "selected_tokens": float(len(selected)),
                "maximum_token_norm_relative_error": float(
                    ((after.norm(dim=-1) - before.norm(dim=-1)).abs() / before.norm(dim=-1).clamp_min(1e-12)).max().cpu()
                ),
            }
        )
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    handle = blocks[layer - 1].register_forward_hook(hook)
    try:
        output = bot.model.model(
            input_ids=None,
            attention_mask=attention,
            position_ids=positions,
            inputs_embeds=embeddings,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        handle.remove()
    return final_logits(bot, output.last_hidden_state[0, -1]), audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bboxes", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--split", choices=("pilot", "dev", "confirmation"), default="pilot")
    parser.add_argument("--findings", nargs="+", default=("pleural_effusion", "nodule_mass"))
    parser.add_argument("--votes", nargs="+", type=int, default=(2, 3))
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--doses", nargs="+", type=float, default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if any(not 0 < dose <= 1 for dose in args.doses) or args.replicates <= 0:
        raise ValueError("doses must lie in (0,1] and replicates must be positive")

    rows = [
        row for row in load_jsonl(args.manifest)
        if row.get("experiment_split") == args.split
        and str(row["finding"]) in set(args.findings)
        and int(row["positive_votes"]) in set(args.votes)
    ]
    rows.sort(key=lambda row: hashlib.sha256(f"42:{row['finding']}:{row['image_id']}".encode()).hexdigest())
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    bbox = {
        f"{row['finding']}:{row['image_id']}": row["boxes"]
        for row in load_jsonl(args.bboxes)
    }
    if not rows:
        raise ValueError("no eligible rows")
    args.output_dir.mkdir(parents=True)
    config = {
        "version": VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "bboxes": str(args.bboxes.resolve()),
        "bboxes_sha256": sha256_file(args.bboxes),
        "model_dir": str(args.model_dir.resolve()),
        "split": args.split,
        "findings": list(args.findings),
        "votes": list(args.votes),
        "layer": args.layer,
        "doses": list(args.doses),
        "replicates": args.replicates,
        "roi_mapping": "DICOM coordinates -> integer-floor square padding -> sqrt(N) grid-cell overlap",
        "logit_readout": "FP32 accumulation from native-precision final hidden and lm_head weights",
        "ablation": "replace selected token direction by global visual mean direction, preserve each token norm",
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(args.output_dir / "config.json", config)
    klass = import_huatuo(args.huatuo_root)
    bot = klass(str(args.model_dir), device="cuda:0")
    raw_path = args.output_dir / "raw.jsonl"
    timings = []
    for index, row in enumerate(rows):
        started = time.perf_counter()
        key = f"{row['finding']}:{row['image_id']}"
        image = load_image(resolve_image(row, args.image_root))
        prompt = prompt_for(str(row["finding"]))
        tensor = torch.stack(bot.get_image_tensors([image])).to(bot.model.device, dtype=torch.bfloat16)
        embeddings, attention, positions, span = prepared_embeddings(bot, prompt, tensor)
        baseline_logits, visual = forward_with_capture(
            bot, embeddings, attention, positions, span, args.layer
        )
        positive_readers = {
            str(value["rad_id"]) for value in row["reader_votes"] if int(value["vote"]) == 1
        }
        boxes = [box for box in bbox[key] if str(box["rad_id"]) in positive_readers]
        roi = square_grid_roi_mask(image.width, image.height, boxes, visual.shape[0])
        roi_indices = np.flatnonzero(roi)
        background_indices = np.flatnonzero(~roi)
        if not len(roi_indices) or len(background_indices) < len(roi_indices):
            append_jsonl(
                raw_path,
                {
                    "version": VERSION,
                    "record_key": key,
                    "image_id": row["image_id"],
                    "finding": row["finding"],
                    "positive_votes": row["positive_votes"],
                    "status": "ineligible_background_match",
                    "roi_tokens": int(len(roi_indices)),
                    "background_tokens": int(len(background_indices)),
                },
            )
            print(json.dumps({"progress": f"{index + 1}/{len(rows)}", "record_key": key, "status": "ineligible_background_match"}), flush=True)
            continue
        interventions = []
        for dose in args.doses:
            count = max(1, int(round(float(dose) * len(roi_indices))))
            for replicate in range(args.replicates):
                for region, candidates in (("roi", roi_indices), ("background", background_indices)):
                    selected = deterministic_subset(candidates, count, key + region, replicate)
                    logits, audit = ablated_logits(
                        bot, embeddings, attention, positions, span, args.layer, selected
                    )
                    interventions.append(
                        {
                            "region": region,
                            "dose": float(dose),
                            "replicate": replicate,
                            "logits": logits,
                            "coordinates": coordinates(logits),
                            "audit": audit,
                        }
                    )
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        append_jsonl(
            raw_path,
            {
                "version": VERSION,
                "record_key": key,
                "image_id": row["image_id"],
                "finding": row["finding"],
                "positive_votes": row["positive_votes"],
                "reader_votes": row["reader_votes"],
                "status": "ok",
                "roi_tokens": int(len(roi_indices)),
                "background_tokens": int(len(background_indices)),
                "baseline_logits": baseline_logits,
                "baseline_coordinates": coordinates(baseline_logits),
                "interventions": interventions,
                "elapsed_seconds": elapsed,
            },
        )
        print(json.dumps({"progress": f"{index + 1}/{len(rows)}", "record_key": key, "seconds": elapsed}), flush=True)
    del bot
    torch.cuda.empty_cache()
    atomic_json(
        args.output_dir / "summary.json",
        {
            "status": "complete",
            "n_requested": len(rows),
            "n_completed": len(timings),
            "seconds_per_case_median": float(np.median(timings)),
            "seconds_per_case_p90": float(np.quantile(timings, 0.9)),
            "raw_sha256": sha256_file(raw_path),
        },
    )


if __name__ == "__main__":
    main()
