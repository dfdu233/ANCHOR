#!/usr/bin/env python3
"""Fatal feasibility audit for single-image domain-orbit canonicalization.

The primary orbit uses label-independent VinDr DICOM display perturbations.  A
secondary source-radial spectrum path is available solely to test whether the
original FedDG-inspired instrument changes the negative conclusion.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from corrected_sgta.domain_orbit_diagnostic import (
    canonicalize,
    degeneration_ratio,
    fit_feature_basis,
    heldout_attenuation,
    random_basis,
    token_stability_gate,
)
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
from corrected_sgta.mosec import load_bank, model_visible_image, radial_mean_calibration


VERSION = "huatuo-domain-orbit-fatal-audit-v2"
RENDER_FIT_VIEWS = (BASELINE_VIEW, "center_minus_0p05w", "center_plus_0p05w", "width_x0p8")
RENDER_HELDOUT_VIEWS = ("native_linear", "width_x1p25")
SPECTRUM_FIT_WEIGHTS = (0.0, 0.1, 0.2, 0.3)
SPECTRUM_HELDOUT_WEIGHTS = (0.4, 0.5)
DEFAULT_SPECTRUM_BANK = Path(
    "/home/dbw/data/mosec_banks/huatuo_pubmedvision_cxr/cxr_radial_envelope.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bboxes", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orbit", choices=("render", "source_radial"), default="render")
    parser.add_argument("--spectrum-bank", type=Path, default=DEFAULT_SPECTRUM_BANK)
    parser.add_argument("--split", choices=("pilot", "dev", "confirmation"), default="pilot")
    parser.add_argument("--findings", nargs="+", default=["aortic_enlargement", "cardiomegaly", "pleural_effusion", "pulmonary_fibrosis"])
    parser.add_argument("--votes", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--per-bin", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--stability-fractions", nargs="+", type=float, default=[0.25])
    parser.add_argument("--stability-alphas", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    return parser.parse_args()


@torch.inference_mode()
def score_embeddings(
    bot: Any,
    embeddings: torch.Tensor,
    attention: torch.Tensor,
    positions: torch.Tensor | None,
    ids: Mapping[str, int],
) -> dict[str, Any]:
    output = bot.model.model(
        input_ids=None,
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=embeddings,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state[0, -1].float()
    weight = bot.model.get_output_embeddings().weight
    token_ids = torch.tensor([ids[state] for state in VERBALIZERS], device=weight.device)
    values = hidden @ weight.index_select(0, token_ids).float().T
    logits = {state: float(values[index].cpu()) for index, state in enumerate(VERBALIZERS)}
    return {
        "logits": logits,
        "prediction": max(logits, key=logits.get),
        "polarity": logits["supported"] - logits["refuted"],
        "commitment": max(logits["supported"], logits["refuted"]) - logits["undetermined"],
    }


@torch.inference_mode()
def encode_orbit(
    bot: Any,
    views: list[dict[str, Any]],
    prompt: str,
    fit_views: tuple[str, ...],
    heldout_views: tuple[str, ...],
    baseline_view: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tokens: dict[str, torch.Tensor] = {}
    baseline_context: dict[str, Any] | None = None
    for view in views:
        name = str(view["name"])
        if name not in set(fit_views) | set(heldout_views):
            view["image"].close()
            continue
        image = view["image"]
        tensor = torch.stack(bot.get_image_tensors([image])).to(bot.model.device, dtype=torch.bfloat16)
        embeddings, attention, positions, span = prepared_embeddings(bot, prompt, tensor)
        start, end = span
        tokens[name] = embeddings[0, start:end].float()
        if name == baseline_view:
            baseline_context = {
                "embeddings": embeddings,
                "attention": attention,
                "positions": positions,
                "span": span,
            }
        elif baseline_context is not None:
            if span != baseline_context["span"] or embeddings.shape != baseline_context["embeddings"].shape:
                raise RuntimeError("render views changed multimodal token layout")
            start0, end0 = baseline_context["span"]
            text_mask = torch.ones(embeddings.shape[1], dtype=torch.bool, device=embeddings.device)
            text_mask[start0:end0] = False
            if not torch.equal(embeddings[0, text_mask], baseline_context["embeddings"][0, text_mask]):
                raise RuntimeError("render views changed non-visual prompt embeddings")
        image.close()
    missing = (set(fit_views) | set(heldout_views)) - set(tokens)
    if missing or baseline_context is None:
        raise RuntimeError(f"missing required render views: {sorted(missing)}")
    return tokens, baseline_context


def spectrum_view_name(weight: float) -> str:
    return f"source_radial_w{weight:g}"


def build_source_radial_views(
    render_views: list[dict[str, Any]], bank_path: Path
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...], str]:
    baseline = next(view for view in render_views if view["name"] == BASELINE_VIEW)
    visible = model_visible_image(baseline["image"])
    for view in render_views:
        view["image"].close()
    bank = load_bank(bank_path)
    all_weights = SPECTRUM_FIT_WEIGHTS + SPECTRUM_HELDOUT_WEIGHTS
    views = []
    for weight in all_weights:
        if weight == 0.0:
            image = visible.copy()
            metadata = {"identity": True, "source_weight": weight}
        else:
            image, metadata = radial_mean_calibration(
                visible, bank["mean"], source_weight=weight
            )
            metadata["source_weight"] = weight
        views.append({"name": spectrum_view_name(weight), "image": image, "metadata": metadata})
    visible.close()
    fit = tuple(spectrum_view_name(weight) for weight in SPECTRUM_FIT_WEIGHTS)
    heldout = tuple(spectrum_view_name(weight) for weight in SPECTRUM_HELDOUT_WEIGHTS)
    return views, fit, heldout, spectrum_view_name(0.0)


def with_visual_tokens(context: dict[str, Any], visual: torch.Tensor) -> torch.Tensor:
    embeddings = context["embeddings"].clone()
    start, end = context["span"]
    if visual.shape != embeddings[0, start:end].shape:
        raise ValueError("replacement visual-token shape mismatch")
    embeddings[0, start:end] = visual.to(embeddings.dtype)
    return embeddings


def expected_state(votes: int) -> str:
    if votes == 0:
        return "refuted"
    if votes == 3:
        return "supported"
    return "undetermined"


def bbox_instability_summary(
    instability: torch.Tensor,
    boxes: list[dict[str, Any]],
    image_height: int,
    image_width: int,
) -> dict[str, Any] | None:
    """Compare 24x24 token instability inside versus outside reader boxes."""

    token_count = int(instability.numel())
    grid = int(round(token_count ** 0.5))
    if not boxes or grid * grid != token_count:
        return None
    side = float(max(image_height, image_width))
    x_pad = (side - image_width) / 2.0
    y_pad = (side - image_height) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(grid, device=instability.device, dtype=torch.float32),
        torch.arange(grid, device=instability.device, dtype=torch.float32),
        indexing="ij",
    )
    x = (xx + 0.5) / grid * side - x_pad
    y = (yy + 0.5) / grid * side - y_pad
    mask = torch.zeros((grid, grid), dtype=torch.bool, device=instability.device)
    for box in boxes:
        mask |= (
            (x >= float(box["x_min"]))
            & (x <= float(box["x_max"]))
            & (y >= float(box["y_min"]))
            & (y <= float(box["y_max"]))
        )
    flat_mask = mask.reshape(-1)
    if not flat_mask.any() or flat_mask.all():
        return None
    values = instability.float().reshape(-1)
    inside = values[flat_mask]
    outside = values[~flat_mask]
    threshold = torch.quantile(values, 0.75)
    top = values >= threshold
    return {
        "bbox_patch_fraction": float(flat_mask.float().mean().cpu()),
        "mean_inside": float(inside.mean().cpu()),
        "mean_outside": float(outside.mean().cpu()),
        "inside_outside_ratio": float((inside.mean() / outside.mean().clamp_min(1e-12)).cpu()),
        "top_quartile_bbox_precision": float((flat_mask & top).sum().float().div(top.sum().clamp_min(1)).cpu()),
        "top_quartile_bbox_recall": float((flat_mask & top).sum().float().div(flat_mask.sum()).cpu()),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(rank <= 0 for rank in args.ranks) or any(not 0.0 <= alpha <= 1.0 for alpha in args.alphas):
        raise ValueError("ranks must be positive and alphas must lie in [0,1]")
    if any(not 0.0 < fraction <= 1.0 for fraction in args.stability_fractions):
        raise ValueError("stability fractions must lie in (0,1]")
    if any(not 0.0 <= alpha <= 1.0 for alpha in args.stability_alphas):
        raise ValueError("stability alphas must lie in [0,1]")

    rows = balanced_rows(args.manifest, args.split, args.findings, args.votes, args.per_bin, args.seed)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    bbox_rows = load_jsonl(args.bboxes)
    boxes_by_claim = {(str(row["image_id"]), str(row["finding"])): list(row.get("boxes", [])) for row in bbox_rows}
    boxes_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in bbox_rows:
        boxes_by_image.setdefault(str(row["image_id"]), []).extend(row.get("boxes", []))

    klass = import_huatuo(args.huatuo_root)
    bot = klass(str(args.model_dir), device=args.device)
    ids = label_ids(bot)
    records = []
    for case_index, row in enumerate(rows):
        image_id, finding = str(row["image_id"]), str(row["finding"])
        path = resolve_image(row, args.image_root)
        pixels = read_dicom_pixels(path)
        render_views = build_render_views(
            pixels,
            boxes_by_claim.get((image_id, finding), []),
            boxes_by_image.get(image_id, []),
        )
        if args.orbit == "render":
            views = render_views
            fit_views = RENDER_FIT_VIEWS
            heldout_views = RENDER_HELDOUT_VIEWS
            baseline_view = BASELINE_VIEW
        else:
            views, fit_views, heldout_views, baseline_view = build_source_radial_views(
                render_views, args.spectrum_bank
            )
        prompt = prompt_for(finding)
        tokens, context = encode_orbit(
            bot, views, prompt, fit_views, heldout_views, baseline_view
        )
        fit_orbit = torch.stack([tokens[name] for name in fit_views])
        original = tokens[baseline_view]
        heldout = torch.stack([tokens[name] for name in heldout_views])
        token_instability = fit_orbit.float().var(dim=0, unbiased=False).mean(dim=-1)
        bbox_instability = bbox_instability_summary(
            token_instability,
            boxes_by_claim.get((image_id, finding), []),
            int(pixels.modality.shape[0]),
            int(pixels.modality.shape[1]),
        )
        max_rank = max(args.ranks)
        center, full_basis, explained = fit_feature_basis(fit_orbit, max_rank)
        original_score = score_embeddings(bot, context["embeddings"], context["attention"], context["positions"], ids)
        mean_score = score_embeddings(bot, with_visual_tokens(context, center), context["attention"], context["positions"], ids)
        methods: dict[str, Any] = {"original": original_score, "orbit_mean": mean_score}
        for fraction in args.stability_fractions:
            for alpha in args.stability_alphas:
                suffix = f"q{fraction:g}_a{alpha:g}"
                candidates = {}
                masks = {}
                for mode in ("unstable", "random", "stable"):
                    candidates[mode], masks[mode] = token_stability_gate(
                        original,
                        token_instability,
                        fraction,
                        alpha,
                        mode=mode,
                        seed=args.seed + 7919 * case_index,
                    )
                for mode, prefix in (
                    ("unstable", "stability"),
                    ("random", "random_token"),
                    ("stable", "inverse_stability"),
                ):
                    candidate = candidates[mode]
                    methods[f"{prefix}_{suffix}"] = {
                        **score_embeddings(
                            bot,
                            with_visual_tokens(context, candidate),
                            context["attention"],
                            context["positions"],
                            ids,
                        ),
                        "selected_token_count": int(masks[mode].sum().cpu()),
                        "global_norm_restored": True,
                        "visual_displacement_fro": float((candidate - original).norm().cpu()),
                    }
        for rank in args.ranks:
            basis = full_basis[:, :rank]
            random = random_basis(original.shape[1], rank, seed=args.seed + 1009 * case_index + rank, device=original.device)
            for alpha in args.alphas:
                suffix = f"r{rank}_a{alpha:g}"
                doc = canonicalize(original, center, basis, alpha)
                displacement = (doc - original).norm()
                mean_displacement = (center - original).norm().clamp_min(1e-12)
                beta = float((displacement / mean_displacement).cpu())
                mean_interp = original + beta * (center - original)
                random_raw = canonicalize(original, center, random, alpha)
                random_delta = random_raw - original
                random_norm = random_delta.norm().clamp_min(1e-12)
                random_doc = original + random_delta * (displacement / random_norm)
                methods[f"doc_{suffix}"] = {
                    **score_embeddings(bot, with_visual_tokens(context, doc), context["attention"], context["positions"], ids),
                    "degeneration_ratio": degeneration_ratio(doc, original, center),
                    "heldout_attenuation": heldout_attenuation(original, heldout, basis),
                    "visual_displacement_fro": float(displacement.cpu()),
                }
                methods[f"random_{suffix}"] = {
                    **score_embeddings(bot, with_visual_tokens(context, random_doc), context["attention"], context["positions"], ids),
                    "degeneration_ratio": degeneration_ratio(random_doc, original, center),
                    "heldout_attenuation": heldout_attenuation(original, heldout, random),
                    "visual_displacement_fro": float((random_doc - original).norm().cpu()),
                    "norm_matched_to_doc": True,
                }
                methods[f"mean_interp_{suffix}"] = {
                    **score_embeddings(bot, with_visual_tokens(context, mean_interp), context["attention"], context["positions"], ids),
                    "beta": beta,
                    "degeneration_ratio": degeneration_ratio(mean_interp, original, center),
                    "visual_displacement_fro": float((mean_interp - original).norm().cpu()),
                }
        records.append(
            {
                "image_id": image_id,
                "finding": finding,
                "positive_votes": int(row["positive_votes"]),
                "expected_state": expected_state(int(row["positive_votes"])),
                "visual_token_shape": list(original.shape),
                "orbit": args.orbit,
                "fit_views": list(fit_views),
                "heldout_views": list(heldout_views),
                "explained_fraction_by_component": [float(value.cpu()) for value in explained],
                "cumulative_explained_fraction": [float(value.cpu()) for value in explained.cumsum(0)],
                "bbox_instability": bbox_instability,
                "methods": methods,
            }
        )
        print(f"[{case_index + 1}/{len(rows)}] {image_id} {finding}", flush=True)

    payload = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "fatal feasibility audit; not a mitigation efficacy result",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "bboxes": str(args.bboxes.resolve()),
        "bboxes_sha256": sha256_file(args.bboxes),
        "model_dir": str(args.model_dir.resolve()),
        "orbit": args.orbit,
        "spectrum_bank": str(args.spectrum_bank.resolve()) if args.orbit == "source_radial" else None,
        "spectrum_bank_sha256": sha256_file(args.spectrum_bank) if args.orbit == "source_radial" else None,
        "split": args.split,
        "ranks": args.ranks,
        "alphas": args.alphas,
        "stability_fractions": args.stability_fractions,
        "stability_alphas": args.stability_alphas,
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "records": records,
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
