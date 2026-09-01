#!/usr/bin/env python3
"""Minimal causal-substrate probe for domain-stable clinical attention heads.

For clear positive VinDr claims with reader boxes, this runner measures whether
the decoder heads used to score the claim remain stable across label-independent
DICOM display renders and whether their visual attention lands inside the
reader-marked evidence region.  It is a mechanism probe, not a mitigation run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from corrected_sgta.run_huatuo_dicom_render_pilot_v1 import (
    BASELINE_VIEW,
    balanced_rows,
    build_render_views,
    read_dicom_pixels,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    VERBALIZERS,
    atomic_json,
    import_huatuo,
    label_ids,
    load_jsonl,
    prepared_embeddings,
    prompt_for,
    resolve_image,
    sha256_file,
)


VERSION = "huatuo-domain-orbit-head-probe-v1"
VIEW_NAMES = (
    BASELINE_VIEW,
    "center_minus_0p05w",
    "center_plus_0p05w",
    "width_x0p8",
    "width_x1p25",
    "native_linear",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bboxes", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("pilot", "dev", "confirmation"), default="pilot")
    parser.add_argument(
        "--findings",
        nargs="+",
        default=["aortic_enlargement", "cardiomegaly", "pleural_effusion", "pulmonary_fibrosis"],
    )
    parser.add_argument("--per-bin", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 7, 14, 21, 27])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    return parser.parse_args()


def bbox_patch_mask(
    token_count: int,
    boxes: list[dict[str, Any]],
    image_height: int,
    image_width: int,
    device: torch.device,
) -> torch.Tensor:
    grid = int(round(token_count**0.5))
    if not boxes or grid * grid != token_count:
        raise ValueError("reader boxes and a square visual-token grid are required")
    side = float(max(image_height, image_width))
    x_pad = (side - image_width) / 2.0
    y_pad = (side - image_height) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(grid, device=device, dtype=torch.float32),
        torch.arange(grid, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x = (xx + 0.5) / grid * side - x_pad
    y = (yy + 0.5) / grid * side - y_pad
    mask = torch.zeros((grid, grid), dtype=torch.bool, device=device)
    for box in boxes:
        mask |= (
            (x >= float(box["x_min"]))
            & (x <= float(box["x_max"]))
            & (y >= float(box["y_min"]))
            & (y <= float(box["y_max"]))
        )
    if not mask.any() or mask.all():
        raise ValueError("reader box maps to an empty or full visual-token mask")
    return mask.reshape(-1)


class HeadOutputCapture:
    def __init__(self, model: Any, layers: tuple[int, ...]) -> None:
        self.model = model
        self.layers = layers
        self.values: dict[int, torch.Tensor] = {}
        self.handles = []

    def __enter__(self) -> "HeadOutputCapture":
        for layer_index in self.layers:
            module = self.model.layers[layer_index].self_attn.o_proj

            def hook(_module: Any, inputs: tuple[torch.Tensor, ...], *, index: int = layer_index) -> None:
                if len(inputs) != 1 or inputs[0].ndim != 3:
                    raise RuntimeError("unexpected o_proj input contract")
                self.values[index] = inputs[0][0, -1].detach().float().cpu()

            self.handles.append(module.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def score_view(
    bot: Any,
    prompt: str,
    image: Any,
    ids: Mapping[str, int],
    layers: tuple[int, ...],
    patch_mask: torch.Tensor,
) -> dict[str, Any]:
    tensor = torch.stack(bot.get_image_tensors([image])).to(bot.model.device, dtype=torch.bfloat16)
    embeddings, attention, positions, span = prepared_embeddings(bot, prompt, tensor)
    start, end = span
    with HeadOutputCapture(bot.model.model, layers) as capture:
        output = bot.model.model(
            input_ids=None,
            attention_mask=attention,
            position_ids=positions,
            inputs_embeds=embeddings,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )
    hidden = output.last_hidden_state[0, -1].float()
    lm_weight = bot.model.get_output_embeddings().weight.float()
    token_ids = torch.tensor([ids[state] for state in VERBALIZERS], device=lm_weight.device)
    values = hidden @ lm_weight.index_select(0, token_ids).T
    logits = {state: float(values[index].cpu()) for index, state in enumerate(VERBALIZERS)}
    claim_direction = lm_weight[ids["supported"]] - lm_weight[ids["refuted"]]
    layer_rows: dict[str, Any] = {}
    for layer_index in layers:
        module = bot.model.model.layers[layer_index].self_attn
        heads = capture.values[layer_index].to(module.o_proj.weight.device).reshape(module.num_heads, module.head_dim)
        coefficient = (claim_direction @ module.o_proj.weight.float()).reshape(module.num_heads, module.head_dim)
        proxy = (heads * coefficient).sum(dim=-1)
        weights = output.attentions[layer_index][0, :, -1, start:end].float()
        visual_mass = weights.sum(dim=-1)
        lesion_mass = weights[:, patch_mask.to(weights.device)].sum(dim=-1)
        layer_rows[str(layer_index)] = {
            "head_output": heads.cpu().tolist(),
            "signed_claim_proxy": proxy.cpu().tolist(),
            "visual_attention_mass": visual_mass.cpu().tolist(),
            "lesion_attention_fraction": (lesion_mass / visual_mass.clamp_min(1e-12)).cpu().tolist(),
        }
    del output, embeddings, tensor
    return {
        "span": [int(start), int(end)],
        "logits": logits,
        "prediction": max(logits, key=logits.get),
        "polarity": logits["supported"] - logits["refuted"],
        "layers": layer_rows,
    }


def summarize_case(views: dict[str, Any], layers: tuple[int, ...]) -> dict[str, Any]:
    baseline = views[BASELINE_VIEW]
    polarity = torch.tensor([views[name]["polarity"] for name in VIEW_NAMES], dtype=torch.float32)
    head_rows: dict[str, Any] = {}
    for layer_index in layers:
        layer = str(layer_index)
        outputs = torch.tensor(
            [views[name]["layers"][layer]["head_output"] for name in VIEW_NAMES], dtype=torch.float32
        )
        proxies = torch.tensor(
            [views[name]["layers"][layer]["signed_claim_proxy"] for name in VIEW_NAMES], dtype=torch.float32
        )
        visual = torch.tensor(
            [views[name]["layers"][layer]["visual_attention_mass"] for name in VIEW_NAMES], dtype=torch.float32
        )
        lesion = torch.tensor(
            [views[name]["layers"][layer]["lesion_attention_fraction"] for name in VIEW_NAMES], dtype=torch.float32
        )
        reference = outputs[0].unsqueeze(0).expand_as(outputs[1:])
        cosine_instability = 1.0 - F.cosine_similarity(outputs[1:], reference, dim=-1).mean(dim=0)
        head_rows[layer] = {
            "output_cosine_instability": cosine_instability.tolist(),
            "signed_proxy_std": proxies.std(dim=0, unbiased=False).tolist(),
            "signed_proxy_range": (proxies.max(dim=0).values - proxies.min(dim=0).values).tolist(),
            "mean_visual_attention_mass": visual.mean(dim=0).tolist(),
            "mean_lesion_attention_fraction": lesion.mean(dim=0).tolist(),
        }
    return {
        "baseline_prediction": baseline["prediction"],
        "any_prediction_flip": any(views[name]["prediction"] != baseline["prediction"] for name in VIEW_NAMES[1:]),
        "polarity_range": float((polarity.max() - polarity.min()).item()),
        "worst_supported_margin_drop": float((polarity[0] - polarity.min()).item()),
        "heads": head_rows,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    layers = tuple(sorted(set(args.layers)))
    if not layers or layers[0] < 0 or layers[-1] >= 28:
        raise ValueError("Huatuo layer indices must lie in [0,27]")
    rows = balanced_rows(args.manifest, args.split, args.findings, [3], args.per_bin, args.seed)
    rows = rows[: args.max_cases]
    bbox_rows = load_jsonl(args.bboxes)
    boxes_by_claim = {
        (str(row["image_id"]), str(row["finding"])): list(row.get("boxes", [])) for row in bbox_rows
    }
    rows = [row for row in rows if boxes_by_claim.get((str(row["image_id"]), str(row["finding"])))]
    if not rows:
        raise RuntimeError("no clear-positive reader-box cases selected")

    klass = import_huatuo(args.huatuo_root)
    bot = klass(str(args.model_dir), device=args.device)
    ids = label_ids(bot)
    records = []
    for case_index, row in enumerate(rows):
        image_id, finding = str(row["image_id"]), str(row["finding"])
        path = resolve_image(row, args.image_root)
        pixels = read_dicom_pixels(path)
        boxes = boxes_by_claim[(image_id, finding)]
        render_views = build_render_views(pixels, boxes, boxes)
        by_name = {str(view["name"]): view for view in render_views}
        missing = set(VIEW_NAMES) - set(by_name)
        if missing:
            raise RuntimeError(f"missing render views: {sorted(missing)}")
        prompt = prompt_for(finding)
        view_scores: dict[str, Any] = {}
        patch_mask = None
        for name in VIEW_NAMES:
            view = by_name[name]
            image = view["image"]
            if patch_mask is None:
                # Huatuo produces a fixed 24x24 projected-token grid.
                patch_mask = bbox_patch_mask(
                    576, boxes, int(pixels.modality.shape[0]), int(pixels.modality.shape[1]), bot.model.device
                )
            view_scores[name] = score_view(bot, prompt, image, ids, layers, patch_mask)
        for view in render_views:
            view["image"].close()
        summary = summarize_case(view_scores, layers)
        records.append(
            {
                "image_id": image_id,
                "finding": finding,
                "positive_votes": int(row["positive_votes"]),
                "bbox_count": len(boxes),
                "bbox_patch_fraction": float(patch_mask.float().mean().cpu()),
                "views": view_scores,
                "summary": summary,
            }
        )
        print(
            f"[{case_index + 1}/{len(rows)}] {finding} {image_id} "
            f"flip={summary['any_prediction_flip']} range={summary['polarity_range']:.4f}",
            flush=True,
        )
    payload = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "mechanism feasibility probe; clear-positive reader-box cases only",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "bboxes": str(args.bboxes.resolve()),
        "bboxes_sha256": sha256_file(args.bboxes),
        "image_root": str(args.image_root.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "split": args.split,
        "findings": args.findings,
        "layers": list(layers),
        "views": list(VIEW_NAMES),
        "seed": args.seed,
        "n": len(records),
        "records": records,
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
