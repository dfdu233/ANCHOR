#!/usr/bin/env python3
"""Causal follow-up for the domain-orbit head probe.

The worst positive-margin render is selected only for mechanism diagnosis.  It
is not a deployable method evaluation.  Matched head-count arms test whether
domain-unstable, lesion-poor heads causally depress the supported claim margin.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from corrected_sgta.run_huatuo_dicom_render_pilot_v1 import build_render_views, read_dicom_pixels
from corrected_sgta.run_huatuo_domain_orbit_head_probe_v1 import VIEW_NAMES
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    VERBALIZERS,
    atomic_json,
    import_huatuo,
    label_ids,
    load_jsonl,
    prepared_embeddings,
    prompt_for,
    sha256_file,
)


VERSION = "huatuo-domain-orbit-head-intervention-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--bboxes", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    return parser.parse_args()


class HeadScaleContext(AbstractContextManager["HeadScaleContext"]):
    def __init__(self, model: Any, selected: Mapping[int, int], scale: float) -> None:
        self.model = model
        self.selected = {int(layer): int(head) for layer, head in selected.items()}
        self.scale = float(scale)
        self.handles = []

    def __enter__(self) -> "HeadScaleContext":
        for layer_index, head in self.selected.items():
            attention = self.model.layers[layer_index].self_attn
            if not 0 <= head < attention.num_heads:
                raise ValueError("selected head is out of range")

            def hook(
                _module: Any,
                inputs: tuple[torch.Tensor, ...],
                *,
                selected_head: int = head,
                head_dim: int = attention.head_dim,
            ) -> tuple[torch.Tensor, ...]:
                if len(inputs) != 1 or inputs[0].ndim != 3:
                    raise RuntimeError("unexpected o_proj input contract")
                output = inputs[0].clone()
                start = selected_head * head_dim
                output[..., start : start + head_dim] *= self.scale
                return (output,)

            self.handles.append(attention.o_proj.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def score(
    bot: Any,
    prompt: str,
    image: Any,
    ids: Mapping[str, int],
    selected: Mapping[int, int],
    scale: float,
) -> dict[str, Any]:
    tensor = torch.stack(bot.get_image_tensors([image])).to(bot.model.device, dtype=torch.bfloat16)
    embeddings, attention, positions, _span = prepared_embeddings(bot, prompt, tensor)
    with HeadScaleContext(bot.model.model, selected, scale):
        output = bot.model.model(
            input_ids=None,
            attention_mask=attention,
            position_ids=positions,
            inputs_embeds=embeddings,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    hidden = output.last_hidden_state[0, -1].float()
    weight = bot.model.get_output_embeddings().weight.float()
    token_ids = torch.tensor([ids[state] for state in VERBALIZERS], device=weight.device)
    values = hidden @ weight.index_select(0, token_ids).T
    logits = {state: float(values[index].cpu()) for index, state in enumerate(VERBALIZERS)}
    return {
        "logits": logits,
        "prediction": max(logits, key=logits.get),
        "polarity": logits["supported"] - logits["refuted"],
    }


def select_heads(record: Mapping[str, Any], seed: int) -> dict[str, dict[int, int]]:
    selections: dict[str, dict[int, int]] = {
        "domain_lesion": {},
        "domain_only": {},
        "spin_low_visual": {},
        "random": {},
    }
    rng = random.Random(seed)
    bbox_fraction = float(record["bbox_patch_fraction"])
    for layer, row in record["summary"]["heads"].items():
        layer_index = int(layer)
        instability = torch.tensor(row["signed_proxy_range"], dtype=torch.float32)
        lesion = torch.tensor(row["mean_lesion_attention_fraction"], dtype=torch.float32)
        visual = torch.tensor(row["mean_visual_attention_mass"], dtype=torch.float32)
        enrichment = lesion / max(bbox_fraction, 1e-12)
        selections["domain_lesion"][layer_index] = int(torch.argmax(instability / (1.0 + enrichment)))
        selections["domain_only"][layer_index] = int(torch.argmax(instability))
        selections["spin_low_visual"][layer_index] = int(torch.argmin(visual))
        selections["random"][layer_index] = rng.randrange(instability.numel())
    return selections


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0.0 <= args.scale <= 1.0:
        raise ValueError("scale must lie in [0,1]")
    probe = json.load(args.probe.open())
    bbox_rows = load_jsonl(args.bboxes)
    boxes_by_claim = {
        (str(row["image_id"]), str(row["finding"])): list(row.get("boxes", [])) for row in bbox_rows
    }
    klass = import_huatuo(args.huatuo_root)
    bot = klass(str(args.model_dir), device=args.device)
    ids = label_ids(bot)
    records = []
    for case_index, source in enumerate(probe["records"]):
        image_id, finding = str(source["image_id"]), str(source["finding"])
        boxes = boxes_by_claim[(image_id, finding)]
        path = args.image_root / f"{image_id}.dicom"
        pixels = read_dicom_pixels(path)
        render_views = build_render_views(pixels, boxes, boxes)
        by_name = {str(view["name"]): view for view in render_views}
        worst_view = min(VIEW_NAMES, key=lambda name: float(source["views"][name]["polarity"]))
        selections = select_heads(source, args.seed + 7919 * case_index)
        arms: dict[str, Any] = {}
        for arm, selected in selections.items():
            arms[arm] = {
                "selected_heads": {str(layer): head for layer, head in selected.items()},
                "baseline": score(bot, prompt_for(finding), by_name[VIEW_NAMES[0]]["image"], ids, selected, args.scale),
                "worst": score(bot, prompt_for(finding), by_name[worst_view]["image"], ids, selected, args.scale),
            }
        for view in render_views:
            view["image"].close()
        original_baseline = float(source["views"][VIEW_NAMES[0]]["polarity"])
        original_worst = float(source["views"][worst_view]["polarity"])
        for arm in arms.values():
            arm["delta_baseline"] = float(arm["baseline"]["polarity"] - original_baseline)
            arm["delta_worst"] = float(arm["worst"]["polarity"] - original_worst)
            arm["recovered_fraction"] = float(
                arm["delta_worst"] / max(original_baseline - original_worst, 1e-12)
            )
        records.append(
            {
                "image_id": image_id,
                "finding": finding,
                "worst_view": worst_view,
                "original_baseline_polarity": original_baseline,
                "original_worst_polarity": original_worst,
                "arms": arms,
            }
        )
        print(
            f"[{case_index + 1}/{len(probe['records'])}] {finding} {image_id} worst={worst_view} "
            + " ".join(f"{arm}={row['delta_worst']:+.3f}" for arm, row in arms.items()),
            flush=True,
        )
    atomic_json(
        args.output,
        {
            "version": VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scientific_role": "outcome-aware causal diagnosis on clear-positive cases; not deployable efficacy",
            "probe": str(args.probe.resolve()),
            "probe_sha256": sha256_file(args.probe),
            "bboxes": str(args.bboxes.resolve()),
            "bboxes_sha256": sha256_file(args.bboxes),
            "model_dir": str(args.model_dir.resolve()),
            "scale": args.scale,
            "seed": args.seed,
            "n": len(records),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
