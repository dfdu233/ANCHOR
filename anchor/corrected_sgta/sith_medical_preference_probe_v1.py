#!/usr/bin/env python3
"""Weight-only SITH-style medical concept probe for CLIP vision towers.

This is a first-signal audit, not a clinical shortcut claim.  It decomposes
per-head VO matrices from a HuggingFace CLIP vision encoder and names the top
singular directions with a frozen medical concept dictionary encoded by the
same CLIP text tower.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizerFast


PROTOCOL_VERSION = "sith-medical-preference-weight-probe-v2-sign-random-control"


CONCEPTS: tuple[tuple[str, str], ...] = (
    ("pathology", "pneumonia"),
    ("pathology", "pulmonary edema"),
    ("pathology", "pleural effusion"),
    ("pathology", "cardiomegaly"),
    ("pathology", "atelectasis"),
    ("pathology", "lung opacity"),
    ("pathology", "consolidation"),
    ("pathology", "pneumothorax"),
    ("pathology", "pulmonary fibrosis"),
    ("pathology", "aortic enlargement"),
    ("anatomy", "left lung"),
    ("anatomy", "right lung"),
    ("anatomy", "heart silhouette"),
    ("anatomy", "mediastinum"),
    ("anatomy", "pleura"),
    ("anatomy", "diaphragm"),
    ("acquisition_context", "portable chest x-ray"),
    ("acquisition_context", "AP view chest x-ray"),
    ("acquisition_context", "PA view chest x-ray"),
    ("acquisition_context", "lateral chest x-ray"),
    ("acquisition_context", "supine chest x-ray"),
    ("acquisition_context", "ICU chest x-ray"),
    ("acquisition_context", "inpatient chest x-ray"),
    ("acquisition_context", "outpatient chest x-ray"),
    ("device_artifact", "endotracheal tube"),
    ("device_artifact", "central venous catheter"),
    ("device_artifact", "chest tube"),
    ("device_artifact", "pacemaker"),
    ("device_artifact", "ECG lead"),
    ("device_artifact", "surgical clips"),
    ("device_artifact", "text marker"),
    ("device_artifact", "image border"),
    ("quality_render", "low contrast chest x-ray"),
    ("quality_render", "overexposed chest x-ray"),
    ("quality_render", "underexposed chest x-ray"),
    ("quality_render", "rotated chest x-ray"),
    ("quality_render", "cropped chest x-ray"),
    ("quality_render", "blurred chest x-ray"),
)


PROMPT_TEMPLATES: tuple[str, ...] = (
    "a chest x-ray showing {}",
    "a radiograph with {}",
    "medical imaging finding: {}",
)


@dataclass(frozen=True)
class HeadVector:
    layer: int
    head: int
    rank_index: int
    singular_value: float
    vector_type: str
    embedding: torch.Tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            layers.extend(range(int(left), int(right) + 1))
        else:
            layers.append(int(part))
    return layers


def deterministic_svd(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    u = u[..., :rank]
    s = s[..., :rank]
    vh = vh[..., :rank, :]
    max_abs = torch.argmax(torch.abs(u), dim=-2, keepdim=True)
    signs = torch.sign(torch.gather(u, -2, max_abs)).clamp(min=-1.0, max=1.0)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return u * signs, s, vh * signs.transpose(-2, -1)


def get_vo_matrix(state: dict[str, torch.Tensor], layer: int, num_heads: int) -> torch.Tensor:
    prefix = f"vision_model.encoder.layers.{layer}.self_attn"
    w_v = state[f"{prefix}.v_proj.weight"].float()
    w_o = state[f"{prefix}.out_proj.weight"].float()
    embed_dim = w_v.shape[0]
    head_dim = embed_dim // num_heads
    if embed_dim % num_heads != 0:
        raise ValueError(f"embed_dim={embed_dim} not divisible by num_heads={num_heads}")
    w_v_heads = w_v.reshape(num_heads, head_dim, embed_dim).transpose(1, 2)
    w_o_heads = w_o.reshape(embed_dim, num_heads, head_dim).permute(1, 2, 0)
    return torch.matmul(w_v_heads, w_o_heads)


def encode_concepts(
    model: CLIPModel,
    tokenizer: CLIPTokenizerFast,
    concepts: tuple[tuple[str, str], ...],
    device: torch.device,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for _group, concept in concepts:
            prompts = [template.format(concept) for template in PROMPT_TEMPLATES]
            tokens = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            encoded = model.get_text_features(**tokens)
            encoded = F.normalize(encoded.float(), dim=-1)
            pooled = F.normalize(encoded.mean(dim=0), dim=-1)
            rows.append(pooled.cpu())
    return torch.stack(rows, dim=0)


def project_hidden_vectors(model: CLIPModel, hidden: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        hidden = hidden.to(device)
        projected = model.vision_model.post_layernorm(hidden)
        projected = model.visual_projection(projected)
        return F.normalize(projected.float(), dim=-1).cpu()


def entropy_from_scores(scores: torch.Tensor) -> float:
    probs = torch.softmax(scores, dim=0)
    return float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    top1 = Counter(row["top_concepts"][0]["group"] for row in records if row["top_concepts"])
    top3 = Counter()
    rank_weighted = defaultdict(float)
    for row in records:
        for idx, concept in enumerate(row["top_concepts"][:3]):
            top3[concept["group"]] += 1
            rank_weighted[concept["group"]] += 1.0 / (idx + 1)
    denominator = max(len(records), 1)
    return {
        "n_vectors": len(records),
        "top1_group_counts": dict(sorted(top1.items())),
        "top1_group_rates": {
            key: value / denominator for key, value in sorted(top1.items())
        },
        "top3_group_counts": dict(sorted(top3.items())),
        "rank_weighted_group_mass": dict(sorted(rank_weighted.items())),
        "shortcut_top1_rate": (
            top1["acquisition_context"] + top1["device_artifact"] + top1["quality_render"]
        )
        / denominator,
        "pathology_top1_rate": top1["pathology"] / denominator,
    }


def match_concepts(
    embeddings: torch.Tensor,
    concept_embeddings: torch.Tensor,
    concept_names: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    sims_pos = embeddings @ concept_embeddings.T
    sims_neg = -embeddings @ concept_embeddings.T
    pos_best = sims_pos.max(dim=-1).values
    neg_best = sims_neg.max(dim=-1).values
    use_neg = neg_best > pos_best
    sims = torch.where(use_neg[:, None], sims_neg, sims_pos)
    values, indices = torch.topk(sims, k=top_k, dim=-1)
    rows: list[dict[str, Any]] = []
    for row_idx in range(embeddings.shape[0]):
        top_concepts = []
        for value, concept_idx in zip(values[row_idx], indices[row_idx], strict=True):
            item = concept_names[int(concept_idx)]
            top_concepts.append(
                {
                    "group": item["group"],
                    "name": item["name"],
                    "similarity": float(value.item()),
                }
            )
        margin = float((values[row_idx, 0] - values[row_idx, 1]).item()) if top_k > 1 else math.nan
        rows.append(
            {
                "matched_sign": "negative" if bool(use_neg[row_idx]) else "positive",
                "top_concepts": top_concepts,
                "top1_margin": margin,
                "topk_entropy": entropy_from_scores(values[row_idx]),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path)
    weights_path = model_path / "pytorch_model.bin"
    device = torch.device(args.device)
    model = CLIPModel.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval().requires_grad_(False)
    tokenizer = CLIPTokenizerFast.from_pretrained(str(model_path), local_files_only=True)
    state = torch.load(weights_path, map_location="cpu")

    layers = parse_layers(args.layers)
    num_heads = int(model.config.vision_config.num_attention_heads)
    rank = int(args.rank)
    concept_embeddings = encode_concepts(model, tokenizer, CONCEPTS, device)
    concept_names = [
        {"group": group, "name": name, "prompts": [template.format(name) for template in PROMPT_TEMPLATES]}
        for group, name in CONCEPTS
    ]

    records: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    for layer in layers:
        vo = get_vo_matrix(state, layer=layer, num_heads=num_heads)
        u, s, vh = deterministic_svd(vo, rank=rank)
        spectra.append(
            {
                "layer": layer,
                "mean_top_singular_value": float(s[:, 0].mean().item()),
                "mean_rank_energy_fraction": float(
                    (s.square().sum(dim=1) / torch.linalg.svdvals(vo.float()).square().sum(dim=1).clamp_min(1e-12)).mean().item()
                ),
            }
        )
        vectors = u.transpose(-2, -1) if args.vector_type == "left" else vh
        flat_vectors = vectors.reshape(num_heads * rank, vectors.shape[-1])
        projected = project_hidden_vectors(model, flat_vectors, device=device)
        matches = match_concepts(projected, concept_embeddings, concept_names, args.top_k)
        for flat_idx, match in enumerate(matches):
            head = flat_idx // rank
            rank_index = flat_idx % rank
            records.append(
                {
                    "layer": layer,
                    "head": head,
                    "rank_index": rank_index,
                    "singular_value": float(s[head, rank_index].item()),
                    "vector_type": args.vector_type,
                    **match,
                }
            )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_hidden = torch.randn(
        len(records),
        int(model.config.vision_config.hidden_size),
        generator=generator,
    )
    random_hidden = F.normalize(random_hidden, dim=-1)
    random_projected = project_hidden_vectors(model, random_hidden, device=device)
    random_records = match_concepts(
        random_projected,
        concept_embeddings,
        concept_names,
        args.top_k,
    )
    random_summary = summarize_records(random_records)
    observed_summary = summarize_records(records)
    enrichment = {
        group: observed_summary["top1_group_rates"].get(group, 0.0)
        - random_summary["top1_group_rates"].get(group, 0.0)
        for group in sorted({g for g, _ in CONCEPTS})
    }
    shortcut_enrichment = (
        observed_summary["shortcut_top1_rate"] - random_summary["shortcut_top1_rate"]
    )

    output = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "claim_boundary": (
            "Weight-space concept naming only. This artifact does not establish "
            "clinical shortcut use, hallucination mitigation, or data-distribution causality."
        ),
        "model": {
            "path": str(model_path),
            "weights_sha256": sha256_file(weights_path),
            "vision_layers": int(model.config.vision_config.num_hidden_layers),
            "vision_heads": num_heads,
            "hidden_size": int(model.config.vision_config.hidden_size),
            "projection_dim": int(model.config.projection_dim),
        },
        "settings": {
            "layers": layers,
            "rank": rank,
            "vector_type": args.vector_type,
            "top_k": args.top_k,
            "seed": args.seed,
            "sign_invariant_matching": True,
            "random_hidden_direction_count": len(records),
            "concept_dictionary_version": "medical-cxr-concepts-v1",
            "concept_count": len(CONCEPTS),
            "prompt_templates": list(PROMPT_TEMPLATES),
        },
        "spectra": spectra,
        "summary": observed_summary,
        "random_direction_control": {
            "summary": random_summary,
            "top1_group_rate_enrichment_observed_minus_random": enrichment,
            "shortcut_top1_rate_enrichment": shortcut_enrichment,
            "claim_boundary": (
                "Random hidden directions pass through the same post-layernorm and "
                "visual projection, then use the same sign-invariant concept matching."
            ),
        },
        "records": records,
    }
    output["fingerprint"] = canonical_json_sha256(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/home/dbw/models/HuatuoGPT-Vision-7B/vit/clip_vit_large_patch14_336",
    )
    parser.add_argument("--layers", default="20-23")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-type", choices=["left", "right"], default="right")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--output",
        default="corrected_runs/sith_medical_preference_probe_v1/huatuo_clip_vit_l20_23_rank16_right_v2.json",
    )
    args = parser.parse_args()
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
