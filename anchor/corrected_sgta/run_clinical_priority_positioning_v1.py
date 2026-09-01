#!/usr/bin/env python3
"""Claim-margin canary for clinical-priority visual token positioning.

The intervention is a permutation: it preserves the visual-token multiset,
token count, norms, prompt, and decoder weights.  Only which projected visual
token occupies which decoder sequence address is changed.  High specialist-CAM
tokens are placed nearest the following question tokens in the priority arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from anchor.corrected_sgta.run_huatuo_vindr_commitment_probe import (
    VERBALIZERS,
    import_huatuo,
    label_ids,
    load_image,
    prepared_embeddings,
    prompt_for,
    sha256_file,
)
from anchor.corrected_sgta.run_hulu_vindr_commitment_probe import (
    HuluRuntime,
    prepared_embeddings_hulu,
)


CONDITIONS = ("native", "clinical_priority", "reverse_priority", "shuffled_priority")


def stable_seed(seed: int, *parts: str) -> int:
    payload = "\x1f".join((str(seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def load_panel(path: Path) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    pack = np.load(path, allow_pickle=False)
    rows = [json.loads(str(value)) for value in pack["panel"]]
    cams = np.asarray(pack["cams"], dtype=np.float32)
    provenance = json.loads(str(pack["provenance"]))
    if len(rows) != len(cams):
        raise ValueError("panel/CAM length mismatch")
    return rows, cams, provenance


def saliency_for_tokens(cam: np.ndarray, count: int) -> np.ndarray:
    side = int(round(count**0.5))
    if side * side != count:
        raise ValueError(f"visual span is not a square grid: {count}")
    value = torch.from_numpy(cam)[None, None]
    value = F.interpolate(value, size=(side, side), mode="bilinear", align_corners=False)
    return value.flatten().numpy().astype(np.float64)


def order_for(condition: str, saliency: np.ndarray, seed: int) -> np.ndarray:
    if condition == "native":
        return np.arange(len(saliency))
    if condition == "clinical_priority":
        return np.argsort(saliency, kind="stable")
    if condition == "reverse_priority":
        return np.argsort(-saliency, kind="stable")
    if condition == "shuffled_priority":
        shuffled = saliency.copy()
        np.random.default_rng(seed).shuffle(shuffled)
        return np.argsort(shuffled, kind="stable")
    raise KeyError(condition)


@torch.inference_mode()
def claim_logits(
    model: Any,
    embeddings: torch.Tensor,
    attention: torch.Tensor,
    positions: torch.Tensor | None,
    token_ids: dict[str, int],
) -> dict[str, float]:
    output = model.model(
        input_ids=None,
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=embeddings,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state[:, -1]
    weight = model.get_output_embeddings().weight
    ids = torch.tensor([token_ids[name] for name in VERBALIZERS], device=weight.device)
    values = hidden.to(weight.dtype) @ weight.index_select(0, ids).T
    return {name: float(values[0, index].float().cpu()) for index, name in enumerate(VERBALIZERS)}


def prepare_huatuo(bot: Any, prompt: str, image: Any):
    tensor = torch.stack(bot.get_image_tensors([image])).to(bot.model.device, dtype=torch.bfloat16)
    return prepared_embeddings(bot, prompt, tensor)


def evaluate_one(
    runtime: Any,
    model: Any,
    preparer: Callable,
    row: dict[str, Any],
    cam: np.ndarray,
    image_root: Path,
    seed: int,
) -> dict[str, Any]:
    image = load_image(image_root / f"{row['image_id']}.dicom")
    prompt = prompt_for(row["finding"])
    embeddings, attention, positions, (start, end) = preparer(runtime, prompt, image)
    visual = embeddings[:, start:end].clone()
    saliency = saliency_for_tokens(cam, end - start)
    ids = label_ids(runtime)
    measurements = {}
    original_multiset = torch.sort(torch.linalg.vector_norm(visual.float(), dim=-1), dim=-1).values
    for condition in CONDITIONS:
        order = order_for(
            condition,
            saliency,
            stable_seed(seed, row["image_id"], row["finding"], condition),
        )
        modified = embeddings.clone()
        modified[:, start:end] = visual.index_select(1, torch.as_tensor(order, device=visual.device))
        check = torch.sort(
            torch.linalg.vector_norm(modified[:, start:end].float(), dim=-1), dim=-1
        ).values
        if not torch.equal(original_multiset, check):
            raise RuntimeError("visual-token norm multiset changed under a permutation")
        logits = claim_logits(model, modified, attention, positions, ids)
        margin = logits["supported"] - logits["refuted"]
        measurements[condition] = {"logits": logits, "margin": margin}
    return {
        **row,
        "status": "ok",
        "visual_tokens": end - start,
        "cam_min": float(saliency.min()),
        "cam_max": float(saliency.max()),
        "cam_std": float(saliency.std()),
        "measurements": measurements,
    }


def signed_margin(row: dict[str, Any], condition: str) -> float:
    sign = 1.0 if int(row["label"]) == 1 else -1.0
    return sign * float(row["measurements"][condition]["margin"])


def summarize(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    comparisons = (
        ("clinical_priority", "native"),
        ("clinical_priority", "reverse_priority"),
        ("clinical_priority", "shuffled_priority"),
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(row)
    image_ids = sorted(groups)

    def point(batch: list[dict[str, Any]], left: str, right: str) -> float:
        return float(np.mean([signed_margin(row, left) - signed_margin(row, right) for row in batch]))

    rng = np.random.default_rng(seed)
    bootstrap = {f"{left}_minus_{right}": [] for left, right in comparisons}
    for _ in range(draws):
        sampled = rng.choice(image_ids, len(image_ids), replace=True)
        batch = [row for image_id in sampled for row in groups[image_id]]
        for left, right in comparisons:
            bootstrap[f"{left}_minus_{right}"].append(point(batch, left, right))
    effects = {}
    for left, right in comparisons:
        key = f"{left}_minus_{right}"
        effects[key] = {
            "mean_signed_correct_margin_delta": point(rows, left, right),
            "image_bootstrap_ci95": [
                float(np.quantile(bootstrap[key], 0.025)),
                float(np.quantile(bootstrap[key], 0.975)),
            ],
        }
    accuracy = {
        condition: float(np.mean([signed_margin(row, condition) > 0 for row in rows]))
        for condition in CONDITIONS
    }
    strata = {}
    for stratum in ("small", "large", "negative"):
        selected = [row for row in rows if row["size_stratum"] == stratum]
        strata[stratum] = {
            "n": len(selected),
            "priority_minus_native": point(selected, "clinical_priority", "native"),
            "priority_minus_reverse": point(selected, "clinical_priority", "reverse_priority"),
        }
    return {
        "n_claims": len(rows),
        "n_images": len(image_ids),
        "effects": effects,
        "signed_margin_accuracy": accuracy,
        "strata": strata,
        "gate": {
            "rule": "priority-minus-reverse and priority-minus-shuffled CI lower bounds exceed zero",
            "pass": effects["clinical_priority_minus_reverse_priority"]["image_bootstrap_ci95"][0] > 0
            and effects["clinical_priority_minus_shuffled_priority"]["image_bootstrap_ci95"][0] > 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu"), required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows, cams, panel_provenance = load_panel(args.panel)
    if args.limit is not None:
        rows, cams = rows[: args.limit], cams[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    completed = {
        (row["image_id"], row["finding"])
        for row in (json.loads(line) for line in raw_path.read_text().splitlines())
    } if raw_path.is_file() else set()
    config = {
        "protocol": "clinical-priority-positioning-v1",
        "model": args.model,
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256_file(args.panel),
        "panel_provenance": panel_provenance,
        "conditions": list(CONDITIONS),
        "operation": "stable sort projected visual embeddings by exact frozen-XRV CAM; highest CAM occupies the latest visual address",
        "invariants": "same projected-token multiset, count, per-token norm multiset, prompt, weights, and one VLM prefill",
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "source_sha256": sha256_file(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    config_path = args.output_dir / "config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != config:
        raise ValueError("refusing resume after configuration drift")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    if args.model == "huatuo":
        bot_class = import_huatuo(Path("/home/dbw/HuatuoGPT-Vision"))
        runtime = bot_class("/home/dbw/models/HuatuoGPT-Vision-7B", device="cuda:0")
        model, preparer = runtime.model, prepare_huatuo
    else:
        runtime = HuluRuntime(Path("/home/dbw/models/Hulu-Med-4B"), args.max_visual_tokens)
        model, preparer = runtime.model, prepared_embeddings_hulu
    with raw_path.open("a") as handle:
        for index, (row, cam) in enumerate(zip(rows, cams)):
            key = (row["image_id"], row["finding"])
            if key in completed:
                continue
            result = evaluate_one(runtime, model, preparer, row, cam, args.image_root, args.seed)
            handle.write(json.dumps(result, separators=(",", ":")) + "\n")
            handle.flush()
            print(f"[{index + 1}/{len(rows)}] {row['finding']}:{row['image_id']}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    completed_rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    analysis = summarize(completed_rows, args.bootstrap_draws, args.seed)
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
