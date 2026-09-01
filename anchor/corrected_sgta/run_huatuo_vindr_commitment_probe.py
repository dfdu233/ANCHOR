#!/usr/bin/env python3
"""Layerwise real/null commitment probe for VinDr reader-disagreement claims.

The controlled verbalizers are an instrument, not the research object.  Each
row represents an atomic finding claim with a continuous reader-support target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata, spearmanr

from corrected_sgta.clinical_claims import (
    VERSION as CLAIM_VERSION,
    epistemic_coordinates,
    evaluate_claim_rows,
    softmax_states,
    tristate_logits,
)


VERSION = "huatuo-vindr-commitment-probe-v8"
IGNORE_INDEX = -100
VERBALIZERS = {"supported": "Yes", "refuted": "No", "undetermined": "Maybe"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def freeze_or_validate_config(
    candidate: dict[str, object], path: Path, resume: bool
) -> dict[str, object]:
    """Keep one run fingerprint across crash-safe resume operations."""

    if not resume:
        atomic_json(path, candidate)
        return candidate
    if not path.is_file():
        raise FileNotFoundError("--resume requires the original config.json")
    existing = json.loads(path.read_text(encoding="utf-8"))
    ignored = {"created_at", "command", "fingerprint"}
    left = {key: value for key, value in existing.items() if key not in ignored}
    right = {key: value for key, value in candidate.items() if key not in ignored}
    if left != right:
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise ValueError(f"refusing resume after config drift: {changed}")
    if not existing.get("fingerprint"):
        raise ValueError("original resumed config has no fingerprint")
    return existing


def append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_global_null_sidecar(
    vector_path: Path, allow_plumbing: bool
) -> dict[str, object]:
    sidecar = vector_path.with_suffix(".json")
    if not vector_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(
            f"global null requires both vector and calibration sidecar: {vector_path}, {sidecar}"
        )
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("split_requirement") != "dev only":
        raise ValueError("global-null sidecar does not certify dev-only calibration")
    if str(metadata.get("vector_sha256")) != sha256_file(vector_path):
        raise ValueError("global-null vector hash disagrees with calibration sidecar")
    if bool(metadata.get("plumbing_only")) and not allow_plumbing:
        raise ValueError("plumbing-only global null is inadmissible for a formal probe")
    return metadata


def import_huatuo(root: Path):
    sys.path.insert(0, str(root))
    from cli import HuatuoChatbot  # type: ignore

    return HuatuoChatbot


def dicom_to_pil(path: Path) -> Image.Image:
    import pydicom

    dataset = pydicom.dcmread(str(path))
    pixels = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept
    finite = pixels[np.isfinite(pixels)]
    if finite.size == 0:
        raise ValueError(f"DICOM contains no finite pixels: {path}")
    lower, upper = np.percentile(finite, [0.5, 99.5])
    if upper <= lower:
        lower, upper = float(finite.min()), float(finite.max())
    if upper <= lower:
        raise ValueError(f"DICOM has constant pixels: {path}")
    pixels = np.clip((pixels - lower) / (upper - lower), 0.0, 1.0)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = 1.0 - pixels
    return Image.fromarray(np.round(pixels * 255).astype(np.uint8), mode="L").convert("RGB")


def load_image(path: Path) -> Image.Image:
    if path.suffix.lower() in {".dcm", ".dicom"}:
        return dicom_to_pil(path)
    return Image.open(path).convert("RGB")


def prompt_for(finding: str) -> str:
    readable = str(finding).replace("_", " ")
    return (
        f"Does this chest X-ray show {readable}? "
        "Answer with exactly one word: Yes, No, or Maybe."
    )


def label_ids(bot: Any) -> dict[str, int]:
    output = {}
    for state, text in VERBALIZERS.items():
        values = bot.tokenizer.encode(text, add_special_tokens=False)
        if len(values) != 1:
            raise ValueError(f"verbalizer must be one token: {text!r} -> {values}")
        output[state] = int(values[0])
    return output


def prepared_embeddings(
    bot: Any, prompt: str, image_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, int]]:
    with_image = bot.insert_image_placeholder(prompt, 1)
    input_ids = bot.preprocess(
        bot.get_conv_without_history(with_image), return_tensors="pt"
    ).to(bot.model.device)
    image_positions = torch.where(input_ids < 0)[0]
    if image_positions.numel() != 1:
        raise RuntimeError("prompt must contain exactly one image placeholder")
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    _, position_ids, expanded_attention, _, embeddings, _ = (
        bot.model.prepare_inputs_labels_for_multimodal_new(
            [input_ids], None, [attention], None, [labels], image_tensor
        )
    )
    if embeddings is None:
        raise RuntimeError("multimodal expansion returned no embeddings")
    start = int(image_positions.item())
    patch_count = int(embeddings.shape[1] - input_ids.numel() + 1)
    if patch_count <= 0 or start + patch_count > embeddings.shape[1]:
        raise RuntimeError(
            f"invalid visual span start={start}, count={patch_count}, shape={embeddings.shape}"
        )
    return embeddings, expanded_attention, position_ids, (start, start + patch_count)


@torch.inference_mode()
def hidden_trajectory(
    bot: Any,
    embeddings: torch.Tensor,
    attention: torch.Tensor,
    position_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    output = bot.model.model(
        input_ids=None,
        attention_mask=attention,
        position_ids=position_ids,
        inputs_embeds=embeddings,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    if output.hidden_states is None:
        raise RuntimeError("decoder returned no hidden states")
    return output.hidden_states


@torch.inference_mode()
def layer_logits(
    bot: Any,
    hidden_states: tuple[torch.Tensor, ...],
    layers: Sequence[int],
    ids: Mapping[str, int],
) -> dict[int, dict[str, float]]:
    output_weight = bot.model.get_output_embeddings().weight
    final = len(hidden_states) - 1
    values = {}
    token_ids = torch.tensor([ids[state] for state in VERBALIZERS], device=output_weight.device)
    for layer in sorted(set((*layers, final))):
        if not 0 <= layer <= final:
            raise ValueError(f"layer {layer} outside 0..{final}")
        hidden = hidden_states[layer][:, -1]
        normalized = hidden if layer == final else bot.model.model.norm(hidden)
        logits = normalized.to(output_weight.dtype) @ output_weight.index_select(0, token_ids).T
        values[layer] = {
            state: float(logits[0, index].float().cpu())
            for index, state in enumerate(VERBALIZERS)
        }
    return values


def entropy(probabilities: Mapping[str, float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in probabilities.values())


def norm_matched_direction_subtraction(
    hidden: torch.Tensor, direction: torch.Tensor, strength: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Subtract a unit direction by a relative L2 step, then restore norm."""

    if hidden.ndim != 1 or direction.shape != hidden.shape:
        raise ValueError("hidden and direction must be equal one-dimensional vectors")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("intervention strength must lie in [0,1]")
    original = hidden.float()
    unit = direction.float()
    unit_norm = unit.norm()
    original_norm = original.norm()
    if not torch.isfinite(unit_norm) or float(unit_norm) <= 0:
        raise ValueError("direction must have finite non-zero norm")
    if not torch.isfinite(original_norm) or float(original_norm) <= 0:
        raise ValueError("hidden vector must have finite non-zero norm")
    unit = unit / unit_norm
    raw = original - strength * original_norm * unit
    raw_norm = raw.norm()
    if not torch.isfinite(raw_norm) or float(raw_norm) <= 0:
        raise ValueError("intervention produced invalid norm")
    matched = raw * (original_norm / raw_norm)
    return matched.to(hidden.dtype), {
        "original_l2": float(original_norm),
        "raw_l2": float(raw_norm),
        "matched_l2": float(matched.float().norm()),
        "relative_step_l2": float((strength * original_norm) / original_norm),
        "direction_projection_before": float(torch.dot(original, unit)),
        "direction_projection_after": float(torch.dot(matched.float(), unit)),
    }


def orthogonalized_unit_direction(
    target_gradient: torch.Tensor, preserve_gradient: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """Remove the first-order preserve-gradient component from a target.

    For the Claim Plane, ``target_gradient`` is the commitment gradient and
    ``preserve_gradient`` is the polarity gradient.  Moving along the returned
    direction therefore changes commitment while leaving polarity unchanged to
    first order.  This is a geometric control, not evidence that the resulting
    direction is clinically meaningful.
    """

    if target_gradient.ndim != 1 or preserve_gradient.shape != target_gradient.shape:
        raise ValueError("target and preserve gradients must be equal 1-D vectors")
    target = target_gradient.detach().float()
    preserve = preserve_gradient.detach().float()
    target_norm = target.norm()
    preserve_norm = preserve.norm()
    if not torch.isfinite(target_norm) or float(target_norm) <= 0:
        raise ValueError("target gradient must have finite non-zero norm")
    if not torch.isfinite(preserve_norm) or float(preserve_norm) <= 0:
        raise ValueError("preserve gradient must have finite non-zero norm")
    preserve_unit = preserve / preserve_norm
    component = target - torch.dot(target, preserve_unit) * preserve_unit
    component_norm = component.norm()
    if not torch.isfinite(component_norm) or float(component_norm) <= 1e-8:
        raise RuntimeError("commitment gradient is degenerate after preserving polarity")
    direction = component / component_norm
    cosine_before = float(torch.dot(target / target_norm, preserve_unit))
    cosine_after = float(torch.dot(direction, preserve_unit))
    return direction, {
        "target_gradient_l2": float(target_norm),
        "preserve_gradient_l2": float(preserve_norm),
        "orthogonal_component_l2": float(component_norm),
        "fraction_target_gradient_retained": float(component_norm / target_norm),
        "target_preserve_cosine_before": cosine_before,
        "target_preserve_cosine_after": cosine_after,
    }


def deterministic_orthogonal_direction(
    direction: torch.Tensor,
    key: str,
    seed: int,
    additional_directions: Sequence[torch.Tensor] = (),
) -> torch.Tensor:
    """Generate a deterministic random unit vector outside locked directions."""

    constraints = [direction, *additional_directions]
    basis: list[torch.Tensor] = []
    for constraint in constraints:
        vector = constraint.detach().float().cpu().flatten()
        if vector.shape != direction.shape:
            raise ValueError("orthogonality constraints must have matching shapes")
        for unit in basis:
            vector = vector - torch.dot(vector, unit) * unit
        norm = vector.norm()
        if torch.isfinite(norm) and float(norm) > 1e-8:
            basis.append(vector / norm)
    if not basis:
        raise ValueError("at least one finite non-zero constraint is required")
    digest = hashlib.sha256(f"{VERSION}:{seed}:{key}:random-direction".encode()).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
    random = torch.randn(direction.shape, generator=generator)
    for unit in basis:
        random = random - torch.dot(random, unit) * unit
    norm = random.norm()
    if float(norm) <= 1e-8:
        raise RuntimeError("deterministic random direction degenerated")
    return random / norm


def null_commitment_direction(
    bot: Any,
    null_hidden: tuple[torch.Tensor, ...],
    layer: int,
    ids: Mapping[str, int],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Null Claim-Plane commitment gradient orthogonal to polarity.

    The exact coordinates are P=(Yes-No)/2 and
    C=(Yes+No)/2-Maybe.  The returned intervention direction is grad(C)
    projected off grad(P), so its first-order effect on P is zero.
    """

    if not 1 <= layer < len(null_hidden):
        raise ValueError(f"intervention layer must be in 1..{len(null_hidden) - 1}")
    # ``measure_one`` runs under inference_mode.  enable_grad alone does not
    # disable inference tensors, so rebuild the entire readout after explicitly
    # leaving inference mode here.
    with torch.inference_mode(False), torch.enable_grad():
        output_weight = bot.model.get_output_embeddings().weight
        token_ids = torch.tensor(
            [ids[state] for state in VERBALIZERS], device=output_weight.device
        )
        selected_weight = output_weight.index_select(0, token_ids).detach().clone()
        hidden = null_hidden[layer][0, -1].detach().clone().requires_grad_(True)
        normalized = (
            hidden
            if layer == len(null_hidden) - 1
            else bot.model.model.norm(hidden.unsqueeze(0))[0]
        )
        logits = normalized.to(selected_weight.dtype) @ selected_weight.T
        logits = logits.float()
        polarity = 0.5 * (logits[0] - logits[1])
        commitment = 0.5 * (logits[0] + logits[1]) - logits[2]
        commitment_gradient = torch.autograd.grad(
            commitment, hidden, retain_graph=True
        )[0]
        polarity_gradient = torch.autograd.grad(
            polarity, hidden, retain_graph=False
        )[0]
    direction, orthogonalization = orthogonalized_unit_direction(
        commitment_gradient, polarity_gradient
    )
    polarity_unit = polarity_gradient.detach().float()
    polarity_unit = polarity_unit / polarity_unit.norm().clamp_min(1e-12)
    return direction.detach(), polarity_unit.detach(), {
        "polarity": float(polarity.detach().cpu()),
        "commitment": float(commitment.detach().cpu()),
        **orthogonalization,
    }


@torch.inference_mode()
def intervened_final_logits(
    bot: Any,
    embeddings: torch.Tensor,
    attention: torch.Tensor,
    position_ids: torch.Tensor | None,
    layer: int,
    direction: torch.Tensor,
    strength: float,
    ids: Mapping[str, int],
) -> tuple[dict[str, float], dict[str, float]]:
    """Patch the prompt-boundary residual after one decoder layer."""

    decoder_layers = bot.model.model.layers
    if not 1 <= layer <= len(decoder_layers):
        raise ValueError(f"intervention layer must be in 1..{len(decoder_layers)}")
    audit: dict[str, float] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        vector, values = norm_matched_direction_subtraction(
            modified[0, -1], direction.to(modified.device), strength
        )
        modified[0, -1] = vector
        audit.update(values)
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    handle = decoder_layers[layer - 1].register_forward_hook(hook)
    try:
        hidden = hidden_trajectory(bot, embeddings, attention, position_ids)
    finally:
        handle.remove()
    logits = layer_logits(bot, hidden, [len(hidden) - 1], ids)[len(hidden) - 1]
    if not audit:
        raise RuntimeError("decoder intervention hook did not execute")
    return logits, audit


def measure_one(
    bot: Any,
    image: Image.Image,
    prompt: str,
    layers: Sequence[int],
    tau: float,
    intervention_layer: int,
    intervention_strength: float,
    temperature: float,
    random_key: str,
    seed: int,
    embedding_preparer: Callable[
        [Any, str, Image.Image],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, int]],
    ]
    | None = None,
    global_null_vector: torch.Tensor | None = None,
) -> dict[str, object]:
    if embedding_preparer is None:
        tensor = torch.stack(bot.get_image_tensors([image])).to(
            bot.model.device, dtype=torch.bfloat16
        )
        embeddings, attention, positions, (start, end) = prepared_embeddings(
            bot, prompt, tensor
        )
    else:
        embeddings, attention, positions, (start, end) = embedding_preparer(
            bot, prompt, image
        )
    mean_null = embeddings.clone()
    if global_null_vector is None:
        replacement = embeddings[:, start:end].mean(dim=1, keepdim=True)
        null_mode = "per_image_projected_token_mean"
    else:
        vector = global_null_vector.to(
            device=embeddings.device, dtype=embeddings.dtype
        ).flatten()
        if vector.numel() != embeddings.shape[-1]:
            raise ValueError(
                f"global null dimension {vector.numel()} != hidden size {embeddings.shape[-1]}"
            )
        replacement = vector.view(1, 1, -1)
        null_mode = "locked_dev_global_projected_mean"
    mean_null[:, start:end] = replacement
    # A matched control preserves each visual token's original L2 norm while
    # removing its direction.  This prevents a simple activation-scale cue
    # from making the image-null comparison look mechanistically specific.
    norm_matched_null = embeddings.clone()
    replacement_norm = torch.linalg.vector_norm(replacement.float(), dim=-1, keepdim=True)
    if float(replacement_norm.min()) <= 1e-12:
        raise ValueError("visual null replacement has degenerate norm")
    original_norms = torch.linalg.vector_norm(
        embeddings[:, start:end].float(), dim=-1, keepdim=True
    )
    norm_matched_null[:, start:end] = (
        replacement.float() / replacement_norm * original_norms
    ).to(embeddings.dtype)
    real_hidden = hidden_trajectory(bot, embeddings, attention, positions)
    null_hidden = hidden_trajectory(bot, mean_null, attention, positions)
    norm_null_hidden = hidden_trajectory(
        bot, norm_matched_null, attention, positions
    )
    ids = label_ids(bot)
    real = layer_logits(bot, real_hidden, layers, ids)
    null = layer_logits(bot, null_hidden, layers, ids)
    norm_null = layer_logits(bot, norm_null_hidden, layers, ids)
    commitment_direction, polarity_direction, null_claim_plane = null_commitment_direction(
        bot, null_hidden, intervention_layer, ids
    )
    random_direction = deterministic_orthogonal_direction(
        commitment_direction,
        random_key,
        seed,
        additional_directions=(polarity_direction,),
    )
    targeted_logits, targeted_audit = intervened_final_logits(
        bot,
        embeddings,
        attention,
        positions,
        intervention_layer,
        commitment_direction,
        intervention_strength,
        ids,
    )
    random_logits, random_audit = intervened_final_logits(
        bot,
        embeddings,
        attention,
        positions,
        intervention_layer,
        random_direction,
        intervention_strength,
        ids,
    )
    targeted_probs = softmax_states(targeted_logits)
    random_probs = softmax_states(random_logits)
    final_layer = len(real_hidden) - 1
    temperature_probs = softmax_states(
        {state: value / temperature for state, value in real[final_layer].items()}
    )
    trajectory = {}
    for layer in sorted(real):
        real_probs = softmax_states(real[layer])
        null_probs = softmax_states(null[layer])
        norm_null_probs = softmax_states(norm_null[layer])
        real_signed = real[layer]["supported"] - real[layer]["refuted"]
        null_signed = null[layer]["supported"] - null[layer]["refuted"]
        evidence = real_signed - null_signed
        coordinates = epistemic_coordinates(real[layer], null[layer])
        cbd_probs = softmax_states(tristate_logits(evidence, tau))
        trajectory[str(layer)] = {
            "real_logits": real[layer],
            "null_logits": null[layer],
            "norm_matched_null_logits": norm_null[layer],
            "real_probabilities": real_probs,
            "null_probabilities": null_probs,
            "norm_matched_null_probabilities": norm_null_probs,
            "signed_visual_evidence": evidence,
            "visual_epistemic_coordinates": coordinates,
            "legacy_simplex_constraint_residual": (
                coordinates["commitment"] - (abs(evidence) - tau)
            ),
            "visual_detail_ablated_signed_bias": null_signed,
            "null_commitment_bias": max(
                null[layer]["supported"], null[layer]["refuted"]
            )
            - null[layer]["undetermined"],
            "real_commitment": 1.0 - real_probs["undetermined"],
            "real_entropy": entropy(real_probs),
            "cbd_probabilities": cbd_probs,
            "baseline_state": max(real_probs, key=real_probs.get),
            "cbd_state": max(cbd_probs, key=cbd_probs.get),
        }
    return {
        "visual_span": {"start": start, "end": end, "tokens": end - start},
        "visual_null_mode": null_mode,
        "final_layer": final_layer,
        "trajectory": trajectory,
        "activation_intervention": {
            "layer": intervention_layer,
            "strength": intervention_strength,
            "null_claim_plane": null_claim_plane,
            "target_definition": (
                "visual-null grad(C) projected orthogonal to grad(P), where "
                "P=(Yes-No)/2 and C=(Yes+No)/2-Maybe"
            ),
            "targeted": {
                "logits": targeted_logits,
                "probabilities": targeted_probs,
                "state": max(targeted_probs, key=targeted_probs.get),
                "norm_audit": targeted_audit,
            },
            "random_orthogonal": {
                "logits": random_logits,
                "probabilities": random_probs,
                "state": max(random_probs, key=random_probs.get),
                "norm_audit": random_audit,
                "absolute_cosine_with_target": float(
                    abs(torch.dot(commitment_direction.cpu(), random_direction.cpu()))
                ),
                "absolute_cosine_with_polarity": float(
                    abs(torch.dot(polarity_direction.cpu(), random_direction.cpu()))
                ),
            },
            "temperature_control": {
                "temperature": temperature,
                "probabilities": temperature_probs,
                "state": max(temperature_probs, key=temperature_probs.get),
            },
        },
    }


@torch.inference_mode()
def calibrate_global_visual_null(
    bot: Any,
    rows: Sequence[Mapping[str, Any]],
    image_root: Path,
    embedding_preparer: Callable[
        [Any, str, Image.Image],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, int]],
    ]
    | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit an equal-image-weighted projected-token mean on a dev split."""

    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["image_id"]), row)
    means: list[torch.Tensor] = []
    token_counts: list[int] = []
    calibration_prompt = "Inspect this chest X-ray. Answer briefly."
    for image_id in sorted(unique):
        row = unique[image_id]
        path = resolve_image(row, image_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        image = load_image(path)
        if embedding_preparer is None:
            tensor = torch.stack(bot.get_image_tensors([image])).to(
                bot.model.device, dtype=torch.bfloat16
            )
            embeddings, _, _, (start, end) = prepared_embeddings(
                bot, calibration_prompt, tensor
            )
        else:
            embeddings, _, _, (start, end) = embedding_preparer(
                bot, calibration_prompt, image
            )
        means.append(embeddings[0, start:end].float().mean(dim=0).cpu())
        token_counts.append(end - start)
    if not means:
        raise ValueError("global-null calibration has no unique images")
    vector = torch.stack(means).mean(dim=0).numpy()
    audit = {
        "method": "equal-image-weighted mean of per-image projected visual-token means",
        "split_requirement": "dev only",
        "unique_images": len(means),
        "hidden_size": int(vector.shape[0]),
        "minimum_visual_tokens": min(token_counts),
        "maximum_visual_tokens": max(token_counts),
        "calibration_prompt": calibration_prompt,
    }
    return vector, audit


def auc_or_none(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(labels) != len(scores):
        raise ValueError("AUROC labels and scores must have equal length")
    positive = sum(int(label) == 1 for label in labels)
    negative = sum(int(label) == 0 for label in labels)
    if positive == 0 or negative == 0:
        return None
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all() or any(int(label) not in {0, 1} for label in labels):
        raise ValueError("AUROC requires finite scores and binary labels")
    ranks = rankdata(values, method="average")
    positive_rank_sum = sum(
        float(rank) for rank, label in zip(ranks, labels) if int(label) == 1
    )
    return float(
        (positive_rank_sum - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def bootstrap_delta(
    rows: Sequence[dict[str, Any]],
    metric,
    seed: int,
    draws: int,
) -> dict[str, float | int | None]:
    by_image: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_image.setdefault(str(row["image_id"]), []).append(row)
    ids = sorted(by_image)
    observed = metric(rows)
    if observed is None or len(ids) < 2:
        return {"estimate": observed, "ci_low": None, "ci_high": None, "valid_draws": 0}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        batch = [row for image_id in sampled for row in by_image[str(image_id)]]
        value = metric(batch)
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return {"estimate": observed, "ci_low": None, "ci_high": None, "valid_draws": 0}
    return {
        "estimate": observed,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "valid_draws": len(values),
    }


def intervention_coordinate_changes(row: Mapping[str, Any]) -> dict[str, float]:
    """Measure downstream Claim-Plane changes from matched interventions."""

    measurement = row["measurement"]
    final = int(measurement["final_layer"])
    baseline = epistemic_coordinates(
        measurement["trajectory"][str(final)]["real_logits"]
    )
    targeted = epistemic_coordinates(
        measurement["activation_intervention"]["targeted"]["logits"]
    )
    random = epistemic_coordinates(
        measurement["activation_intervention"]["random_orthogonal"]["logits"]
    )
    baseline_polarity = float(baseline["polarity"])
    targeted_polarity = float(targeted["polarity"])
    random_polarity = float(random["polarity"])
    return {
        "targeted_polarity_change": targeted_polarity - baseline_polarity,
        "targeted_commitment_change": float(targeted["commitment"])
        - float(baseline["commitment"]),
        "random_polarity_change": random_polarity - baseline_polarity,
        "random_commitment_change": float(random["commitment"])
        - float(baseline["commitment"]),
        "targeted_polarity_sign_flip": float(
            baseline_polarity * targeted_polarity < 0.0
        ),
        "random_polarity_sign_flip": float(baseline_polarity * random_polarity < 0.0),
    }


def analyze(
    rows: Sequence[dict[str, Any]], precollapse: int, tau: float, seed: int, draws: int
) -> dict[str, object]:
    valid = [row for row in rows if row.get("status") == "ok"]
    if not valid:
        return {"status": "no_valid_rows", "n": 0}
    final = int(valid[0]["measurement"]["final_layer"])
    layers = sorted(int(value) for value in valid[0]["measurement"]["trajectory"])
    if precollapse not in layers:
        raise ValueError(f"precollapse layer {precollapse} not found in {layers}")

    per_layer = {}
    ambiguity = [int(0.0 < float(row["reader_support"]) < 1.0) for row in valid]
    supports = [float(row["reader_support"]) for row in valid]
    for layer in layers:
        entries = [row["measurement"]["trajectory"][str(layer)] for row in valid]
        evidence = [float(value["signed_visual_evidence"]) for value in entries]
        relation = spearmanr(supports, evidence)
        per_layer[str(layer)] = {
            "ambiguity_auroc_from_weak_evidence": auc_or_none(
                ambiguity, [-abs(value) for value in evidence]
            ),
            "reader_support_spearman": (
                None if not math.isfinite(float(relation.statistic)) else float(relation.statistic)
            ),
            "mean_null_commitment_bias": float(
                np.mean([value["null_commitment_bias"] for value in entries])
            ),
            "mean_real_commitment": float(
                np.mean([value["real_commitment"] for value in entries])
            ),
        }

    def auc_delta(batch: Sequence[dict[str, Any]]) -> float | None:
        labels = [int(0.0 < float(row["reader_support"]) < 1.0) for row in batch]
        early = [
            -abs(float(row["measurement"]["trajectory"][str(precollapse)]["signed_visual_evidence"]))
            for row in batch
        ]
        late = [
            -abs(float(row["measurement"]["trajectory"][str(final)]["signed_visual_evidence"]))
            for row in batch
        ]
        early_auc, late_auc = auc_or_none(labels, early), auc_or_none(labels, late)
        return None if early_auc is None or late_auc is None else early_auc - late_auc

    def null_bias_growth(batch: Sequence[dict[str, Any]]) -> float:
        return float(
            np.mean(
                [
                    float(row["measurement"]["trajectory"][str(final)]["null_commitment_bias"])
                    - float(row["measurement"]["trajectory"][str(precollapse)]["null_commitment_bias"])
                    for row in batch
                ]
            )
        )

    def predicted_state(row: dict[str, Any], method: str) -> str:
        if method == "baseline":
            return str(row["measurement"]["trajectory"][str(final)]["baseline_state"])
        if method == "cbd":
            return str(row["measurement"]["trajectory"][str(final)]["cbd_state"])
        condition = {
            "targeted_activation": "targeted",
            "random_activation": "random_orthogonal",
            "temperature": "temperature_control",
        }[method]
        return str(row["measurement"]["activation_intervention"][condition]["state"])

    comparisons = {}
    methods = (
        "baseline",
        "cbd",
        "targeted_activation",
        "random_activation",
        "temperature",
    )
    for method in methods:
        claim_rows = [
            {
                "reader_support": row["reader_support"],
                "prediction_state": predicted_state(row, method),
            }
            for row in valid
        ]
        comparisons[method] = evaluate_claim_rows(claim_rows)
    clear = [row for row in valid if float(row["reader_support"]) in {0.0, 1.0}]

    def clear_accuracy(method: str) -> float | None:
        if not clear:
            return None
        return sum(
            predicted_state(row, method)
            == ("supported" if float(row["reader_support"]) == 1.0 else "refuted")
            for row in clear
        ) / len(clear)

    clear_accuracy_by_method = {method: clear_accuracy(method) for method in methods}
    baseline_clear = clear_accuracy_by_method["baseline"]

    def ambiguous_overcommitment(batch: Sequence[dict[str, Any]], method: str) -> float | None:
        selected = [
            row for row in batch if 0.0 < float(row["reader_support"]) < 1.0
        ]
        if not selected:
            return None
        return sum(predicted_state(row, method) != "undetermined" for row in selected) / len(selected)

    def target_minus_baseline(batch: Sequence[dict[str, Any]]) -> float | None:
        target = ambiguous_overcommitment(batch, "targeted_activation")
        baseline = ambiguous_overcommitment(batch, "baseline")
        return None if target is None or baseline is None else target - baseline

    def target_minus_random(batch: Sequence[dict[str, Any]]) -> float | None:
        target = ambiguous_overcommitment(batch, "targeted_activation")
        random = ambiguous_overcommitment(batch, "random_activation")
        return None if target is None or random is None else target - random

    auc_ci = bootstrap_delta(valid, auc_delta, seed, draws)
    bias_ci = bootstrap_delta(valid, null_bias_growth, seed + 1, draws)
    intervention_vs_baseline_ci = bootstrap_delta(
        valid, target_minus_baseline, seed + 2, draws
    )
    intervention_vs_random_ci = bootstrap_delta(
        valid, target_minus_random, seed + 3, draws
    )
    coordinate_changes = [intervention_coordinate_changes(row) for row in valid]

    def coordinate_mean(batch: Sequence[dict[str, Any]], field: str) -> float:
        return float(
            np.mean([intervention_coordinate_changes(row)[field] for row in batch])
        )

    targeted_commitment_change_ci = bootstrap_delta(
        valid,
        lambda batch: coordinate_mean(batch, "targeted_commitment_change"),
        seed + 4,
        draws,
    )
    targeted_minus_random_commitment_change_ci = bootstrap_delta(
        valid,
        lambda batch: coordinate_mean(batch, "targeted_commitment_change")
        - coordinate_mean(batch, "random_commitment_change"),
        seed + 5,
        draws,
    )
    targeted_abs_polarity_change_ci = bootstrap_delta(
        valid,
        lambda batch: float(
            np.mean(
                [
                    abs(intervention_coordinate_changes(row)["targeted_polarity_change"])
                    for row in batch
                ]
            )
        ),
        seed + 6,
        draws,
    )
    targeted_sign_flip_rate = float(
        np.mean([value["targeted_polarity_sign_flip"] for value in coordinate_changes])
    )
    random_sign_flip_rate = float(
        np.mean([value["random_polarity_sign_flip"] for value in coordinate_changes])
    )
    brier_base = float(comparisons["baseline"]["reader_distribution_brier"])
    brier_cbd = float(comparisons["cbd"]["reader_distribution_brier"])
    norm_relative_errors = []
    random_cosines = []
    random_polarity_cosines = []
    targeted_polarity_cosines = []
    for row in valid:
        intervention = row["measurement"]["activation_intervention"]
        for condition in ("targeted", "random_orthogonal"):
            norm = intervention[condition]["norm_audit"]
            norm_relative_errors.append(
                abs(float(norm["matched_l2"]) - float(norm["original_l2"]))
                / max(float(norm["original_l2"]), 1e-12)
            )
        random_cosines.append(
            float(intervention["random_orthogonal"]["absolute_cosine_with_target"])
        )
        if "absolute_cosine_with_polarity" in intervention["random_orthogonal"]:
            random_polarity_cosines.append(
                float(
                    intervention["random_orthogonal"][
                        "absolute_cosine_with_polarity"
                    ]
                )
            )
        claim_plane = intervention.get("null_claim_plane", {})
        if "target_preserve_cosine_after" in claim_plane:
            targeted_polarity_cosines.append(
                abs(float(claim_plane["target_preserve_cosine_after"]))
            )
    targeted_clear = clear_accuracy_by_method["targeted_activation"]
    targeted_gate = bool(
        intervention_vs_baseline_ci["ci_high"] is not None
        and float(intervention_vs_baseline_ci["ci_high"]) < 0
        and baseline_clear is not None
        and targeted_clear is not None
        and baseline_clear - targeted_clear <= 0.01
    )
    controls_gate = bool(
        targeted_gate
        and intervention_vs_random_ci["ci_high"] is not None
        and float(intervention_vs_random_ci["ci_high"]) < 0
        and max(norm_relative_errors, default=float("inf")) <= 1e-5
        and max(random_cosines, default=float("inf")) <= 1e-5
        and max(random_polarity_cosines, default=float("inf")) <= 1e-5
        and max(targeted_polarity_cosines, default=float("inf")) <= 1e-5
        and comparisons["temperature"]["coverage"] == comparisons["baseline"]["coverage"]
    )
    coordinate_actuation_gate = bool(
        targeted_commitment_change_ci["ci_high"] is not None
        and float(targeted_commitment_change_ci["ci_high"]) < 0.0
        and targeted_minus_random_commitment_change_ci["ci_high"] is not None
        and float(targeted_minus_random_commitment_change_ci["ci_high"]) < 0.0
        and targeted_sign_flip_rate <= 0.01
    )
    return {
        "status": "complete",
        "n": len(valid),
        "precollapse_layer": precollapse,
        "final_layer": final,
        "tau": tau,
        "per_layer": per_layer,
        "precollapse_minus_final_ambiguity_auroc": auc_ci,
        "final_minus_precollapse_null_commitment_bias": bias_ci,
        "targeted_minus_baseline_disagreement_overcommitment": intervention_vs_baseline_ci,
        "targeted_minus_random_disagreement_overcommitment": intervention_vs_random_ci,
        "continuous_intervention_effect": {
            "targeted_commitment_change": targeted_commitment_change_ci,
            "targeted_minus_random_commitment_change": (
                targeted_minus_random_commitment_change_ci
            ),
            "targeted_absolute_polarity_change": targeted_abs_polarity_change_ci,
            "targeted_polarity_sign_flip_rate": targeted_sign_flip_rate,
            "random_polarity_sign_flip_rate": random_sign_flip_rate,
            "interpretation": (
                "negative commitment change is the intended effect; polarity "
                "change and sign flips are leakage, not efficacy"
            ),
        },
        "claim_metrics": comparisons,
        "clear_case_accuracy": clear_accuracy_by_method,
        "intervention_control_audit": {
            "maximum_norm_relative_error": max(norm_relative_errors, default=None),
            "maximum_random_target_absolute_cosine": max(random_cosines, default=None),
            "maximum_random_polarity_absolute_cosine": max(
                random_polarity_cosines, default=None
            ),
            "maximum_targeted_polarity_absolute_cosine": max(
                targeted_polarity_cosines, default=None
            ),
            "temperature_preserves_argmax_coverage": (
                comparisons["temperature"]["coverage"]
                == comparisons["baseline"]["coverage"]
            ),
        },
        "legacy_cbd_relative_brier_improvement": (
            (brier_base - brier_cbd) / brier_base if brier_base > 0 else None
        ),
        "mechanism_gates": {
            "ambiguity_decodability_delta_ge_0.05_ci_above_zero": bool(
                auc_ci["estimate"] is not None
                and auc_ci["ci_low"] is not None
                and float(auc_ci["estimate"]) >= 0.05
                and float(auc_ci["ci_low"]) > 0
            ),
            "null_commitment_bias_growth_ci_above_zero": bool(
                bias_ci["ci_low"] is not None and float(bias_ci["ci_low"]) > 0
            ),
            "clear_case_drop_le_0.01": bool(
                baseline_clear is not None
                and targeted_clear is not None
                and baseline_clear - targeted_clear <= 0.01
            ),
            "activation_intervention_passed": targeted_gate,
            "continuous_coordinate_actuation_passed": coordinate_actuation_gate,
            "temperature_norm_random_controls_passed": controls_gate,
        },
        "claim_ceiling": (
            "causal polarity-preserving commitment claim is earned only if the "
            "reader-agreement gate and targeted activation, clear-case safety, "
            "and matched-control gates all pass on the locked formal test"
        ),
    }


def resolve_image(row: Mapping[str, Any], image_root: Path) -> Path:
    explicit = row.get("image_path")
    if explicit:
        path = Path(str(explicit))
        return path if path.is_absolute() else image_root / path
    return image_root / str(row["dicom_relpath"]).removeprefix("train/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21, 28])
    parser.add_argument("--precollapse-layer", type=int, default=14)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--intervention-layer", type=int, default=21)
    parser.add_argument("--intervention-strength", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--global-null-npy", type=Path)
    parser.add_argument("--calibrate-global-null-output", type=Path)
    parser.add_argument("--allow-plumbing-global-null", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--experiment-split", choices=("all", "dev", "test"), default="all"
    )
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="refresh summary.json from an existing raw.jsonl without loading a model",
    )
    args = parser.parse_args()
    if args.analyze_only:
        if args.resume or args.calibrate_global_null_output:
            raise ValueError("analyze-only is incompatible with resume or calibration")
        raw_path = args.output_dir / "raw.jsonl"
        config_path = args.output_dir / "config.json"
        if not raw_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                "analyze-only requires existing raw.jsonl and config.json in output-dir"
            )
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        all_rows = load_jsonl(raw_path)
        summary = analyze(
            all_rows,
            int(existing_config["precollapse_layer"]),
            float(existing_config["tau"]),
            args.seed,
            args.bootstrap_draws,
        )
        summary["config"] = existing_config
        summary["analysis_refresh"] = {
            "analyzer_version": VERSION,
            "analyzer_code_sha256": sha256_file(Path(__file__)),
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
            "raw_sha256": sha256_file(raw_path),
        }
        summary["errors"] = sum(row.get("status") != "ok" for row in all_rows)
        atomic_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        if summary.get("status") == "no_valid_rows":
            raise RuntimeError("probe contains no valid rows; inspect raw.jsonl")
        return
    if args.global_null_npy and args.calibrate_global_null_output:
        raise ValueError("global-null calibration and use are mutually exclusive")
    if args.calibrate_global_null_output:
        if args.experiment_split != "dev":
            raise ValueError("global null must be calibrated on the locked dev split")
        if args.calibrate_global_null_output.suffix != ".npy":
            raise ValueError("global-null output must have a .npy suffix")
    global_null_metadata = (
        validate_global_null_sidecar(
            args.global_null_npy, args.allow_plumbing_global_null
        )
        if args.global_null_npy
        else None
    )
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"{args.output_dir} exists; pass --resume or use a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    rows = load_jsonl(args.manifest)
    if args.experiment_split != "all":
        missing_split = [row for row in rows if "experiment_split" not in row]
        if missing_split:
            raise ValueError(
                "manifest lacks experiment_split; rebuild it before a locked dev/test run"
            )
        rows = [
            row
            for row in rows
            if str(row["experiment_split"]) == args.experiment_split
        ]
        if not rows:
            raise ValueError(f"manifest has no rows in {args.experiment_split} split")
    rows = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row['finding']}:{row['image_id']}".encode()
        ).hexdigest(),
    )
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    reference_sources = sorted(
        {str(row.get("reference_source", "unspecified")) for row in rows}
    )
    evidence_grades = sorted(
        {str(row.get("evidence_grade", "ungraded")) for row in rows}
    )
    formal_reference = bool(rows) and all(
        row.get("formal_reference") is True for row in rows
    )
    completed = {
        str(row["record_key"])
        for row in load_jsonl(raw_path)
        if row.get("status") == "ok"
    } if args.resume and raw_path.exists() else set()
    config = {
        "version": VERSION,
        "claim_contract_version": CLAIM_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": (
            "vindr-cxr-1.0.0-reader-votes"
            if reference_sources == ["vindr_reader_votes"] and formal_reference
            else "diagnostic-claim-manifest"
        ),
        "reference_sources": reference_sources,
        "evidence_grades": evidence_grades,
        "formal_reference": formal_reference,
        "model": str(args.model_dir.resolve()),
        "method": "claim-plane-polarity-orthogonal-commitment-probe",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "experiment_split": args.experiment_split,
        "image_root": str(args.image_root.resolve()),
        "layers": args.layers,
        "precollapse_layer": args.precollapse_layer,
        "tau": args.tau,
        "intervention_layer": args.intervention_layer,
        "intervention_strength": args.intervention_strength,
        "temperature_control": args.temperature,
        "verbalizers": VERBALIZERS,
        "prompt": (
            "Use row.question when present; otherwise ask whether the CXR shows "
            "<finding>. Always request exactly Yes, No, or Maybe."
        ),
        "mean_token_null": "replace every projected visual token by that image's projected-token mean",
        "norm_matched_null": (
            "replace visual-token directions by the image/global null direction "
            "while preserving every original token L2 norm"
        ),
        "null_claim_ceiling": (
            "per-image mean removes spatial detail but retains image-level mean; "
            "image-independent bias requires a locked dev-global or shuffled-image null control"
        ),
        "null_mode": (
            "locked_dev_global_projected_mean"
            if args.global_null_npy
            else "calibrate_locked_dev_global_projected_mean"
            if args.calibrate_global_null_output
            else "per_image_projected_token_mean"
        ),
        "global_null_npy": str(args.global_null_npy.resolve()) if args.global_null_npy else None,
        "global_null_sha256": sha256_file(args.global_null_npy) if args.global_null_npy else None,
        "global_null_calibration": global_null_metadata,
        "dicom_preprocess": "rescale slope/intercept; 0.5/99.5 percentile window; MONOCHROME1 inversion; RGB",
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config = freeze_or_validate_config(
        config, args.output_dir / "config.json", args.resume
    )
    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    if args.calibrate_global_null_output:
        vector, audit = calibrate_global_visual_null(bot, rows, args.image_root)
        args.calibrate_global_null_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.calibrate_global_null_output, vector, allow_pickle=False)
        calibration = {
            "version": VERSION,
            "config_fingerprint": config["fingerprint"],
            "vector": str(args.calibrate_global_null_output.resolve()),
            "vector_sha256": sha256_file(args.calibrate_global_null_output),
            "plumbing_only": args.max_samples is not None,
            **audit,
        }
        atomic_json(args.calibrate_global_null_output.with_suffix(".json"), calibration)
        print(json.dumps(calibration, indent=2))
        del bot
        torch.cuda.empty_cache()
        return
    global_null_vector = (
        torch.from_numpy(np.load(args.global_null_npy, allow_pickle=False))
        if args.global_null_npy
        else None
    )
    for index, row in enumerate(rows):
        key = f"{row['finding']}:{row['image_id']}"
        if key in completed:
            continue
        record: dict[str, Any] = {
            "version": VERSION,
            "fingerprint": config["fingerprint"],
            "record_key": key,
            "image_id": row["image_id"],
            "finding": row["finding"],
            "positive_votes": row["positive_votes"],
            "reader_count": row["reader_count"],
            "reader_support": row["reader_support"],
            "reader_state": row["reader_state"],
            "experiment_split": row.get("experiment_split"),
            "reference_source": row.get("reference_source"),
            "formal_reference": row.get("formal_reference"),
            "status": "error",
        }
        try:
            path = resolve_image(row, args.image_root)
            if not path.is_file():
                raise FileNotFoundError(path)
            image = load_image(path)
            question = str(row.get("question") or prompt_for(str(row["finding"])))
            record["question"] = question
            record["measurement"] = measure_one(
                bot,
                image,
                question,
                args.layers,
                args.tau,
                args.intervention_layer,
                args.intervention_strength,
                args.temperature,
                key,
                args.seed,
                global_null_vector=global_null_vector,
            )
            record["image_path"] = str(path.resolve())
            record["status"] = "ok"
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            record["error"] = f"CUDA OOM: {error}"
            record["traceback"] = traceback.format_exc()
        except Exception as error:
            record["error"] = repr(error)
            record["traceback"] = traceback.format_exc()
        append_jsonl(raw_path, record)
        print(json.dumps({"progress": f"{index + 1}/{len(rows)}", "record_key": key, "status": record["status"], "error": record.get("error")}), flush=True)
    del bot
    torch.cuda.empty_cache()
    all_rows = load_jsonl(raw_path)
    summary = analyze(
        all_rows,
        args.precollapse_layer,
        args.tau,
        args.seed,
        args.bootstrap_draws,
    )
    summary["config"] = config
    summary["errors"] = sum(row.get("status") != "ok" for row in all_rows)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary.get("status") == "no_valid_rows":
        raise RuntimeError("probe produced no valid rows; inspect raw.jsonl")


if __name__ == "__main__":
    main()
