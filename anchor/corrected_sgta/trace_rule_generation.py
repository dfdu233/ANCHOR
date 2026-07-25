"""Resumable mechanistic traces for an explicit RULE qid allow-list.

The script teacher-forces cached baseline text.  It never selects cases and
cannot run without explicit ``--qid`` arguments.  Attention is optional and
fail-open because support depends on the installed Transformers backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from corrected_sgta.infer_rule_dg_adapter import (
    load_rows,
    official_prompt,
    repair_jsonl_tail,
)
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.train_rule_dg_adapter import (
    build_teacher_forcing,
    process_image,
)


VERSION = "rule-teacher-forced-mechanistic-trace-v2"
IGNORE_INDEX = -100
SELECTED_HIDDEN = (0, 8, 16, 24, 32)
SELECTED_ATTENTION = (0, 7, 15, 23, 31)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def answer_text(row: dict[str, Any]) -> str:
    for key in ("base_text", "answer", "text"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("cached answer has no base_text/answer/text")


def token_statistics(
    logits: torch.Tensor, targets: torch.Tensor, tokenizer, top_k: int
) -> list[dict[str, Any]]:
    """Reduce [tokens,vocab] logits without persisting a full vocabulary."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    entropy = -(probabilities * log_probabilities).sum(-1)
    values, indices = torch.topk(logits.float(), k=min(top_k, logits.shape[-1]), dim=-1)
    result = []
    for index in range(logits.shape[0]):
        target = int(targets[index])
        result.append(
            {
                "target_id": target,
                "target_token": tokenizer.decode([target]),
                "target_log_probability": float(log_probabilities[index, target].cpu()),
                "entropy": float(entropy[index].cpu()),
                "top1_top2_margin": float((values[index, 0] - values[index, 1]).cpu()),
                "topk": [
                    {
                        "token_id": int(token_id),
                        "token": tokenizer.decode([int(token_id)]),
                        "logit": float(value),
                    }
                    for token_id, value in zip(
                        indices[index].cpu().tolist(), values[index].cpu().tolist()
                    )
                ],
            }
        )
    return result


def visual_span(input_ids: torch.Tensor, patch_count: int) -> tuple[int, int]:
    from llava.constants import IMAGE_TOKEN_INDEX

    positions = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
    if positions.numel() != 1:
        raise RuntimeError(f"expected one image token, found {positions.numel()}")
    start = int(positions.item())
    return start, start + patch_count


def surface_groups(adapter) -> dict[str, list[int]]:
    groups = {}
    for label in ("Yes", "No"):
        ids = []
        for form in (label, label.lower(), " " + label, " " + label.lower()):
            encoded = adapter.tokenizer.encode(form, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(encoded[0])
        if not ids:
            raise RuntimeError(f"no one-token surface form for {label}")
        groups[label] = ids
    return groups


def semantic_margin(
    logits: torch.Tensor, groups: dict[str, list[int]]
) -> torch.Tensor:
    """Yes-minus-No surface margin at the assistant prompt boundary."""
    return logits[groups["Yes"]].max() - logits[groups["No"]].max()


def polarity_positions(
    targets: torch.Tensor, tokenizer, groups: dict[str, list[int]]
) -> list[dict[str, Any]]:
    """Locate explicit yes/no/not tokens in the cached generated sequence."""
    not_ids: set[int] = set()
    for form in ("not", " not", "Not", " Not"):
        encoded = tokenizer.encode(form, add_special_tokens=False)
        if len(encoded) == 1:
            not_ids.add(int(encoded[0]))
    yes_ids, no_ids = set(groups["Yes"]), set(groups["No"])
    rows = []
    for position, token_id_tensor in enumerate(targets):
        token_id = int(token_id_tensor)
        polarity = (
            "yes" if token_id in yes_ids else
            "no" if token_id in no_ids else
            "not" if token_id in not_ids else None
        )
        if polarity is not None:
            rows.append({
                "answer_token_index": position, "token_id": token_id,
                "token": tokenizer.decode([token_id]), "polarity": polarity,
            })
    return rows


def logit_lens(adapter, hidden_states, answer_mask) -> dict[str, list[dict[str, float]]]:
    groups = surface_groups(adapter)
    output_weight = adapter.model.get_output_embeddings().weight
    rows = {}
    for layer in SELECTED_HIDDEN:
        hidden = hidden_states[layer][:, :-1][answer_mask]
        # Mistral's final RMSNorm is required before the unembedding.
        normalized = adapter.model.model.norm(hidden)
        layer_rows = []
        for vector in normalized:
            scores = {}
            for label, ids in groups.items():
                weight = output_weight[ids]
                scores[label] = float(
                    (vector.to(weight.dtype) @ weight.T).max().float().cpu()
                )
            layer_rows.append(
                {
                    "yes": scores["Yes"],
                    "no": scores["No"],
                    "yes_minus_no": scores["Yes"] - scores["No"],
                }
            )
        rows[str(layer)] = layer_rows
    return rows


def attention_summary(
    adapter,
    inputs_embeds,
    position_ids,
    attention_mask,
    answer_mask,
    start: int,
    end: int,
) -> tuple[dict[str, Any] | None, str | None]:
    original = getattr(adapter.model.config, "_attn_implementation", None)
    try:
        adapter.model.config._attn_implementation = "eager"
        adapter.model.model.config._attn_implementation = "eager"
        with torch.inference_mode():
            output = adapter.model.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds.detach(),
                use_cache=False,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )
        if output.attentions is None or any(value is None for value in output.attentions):
            return None, "backend returned no attention weights"
        query_positions = torch.where(answer_mask[0])[0]
        # answer_mask indexes shifted logits, so these are precisely the query
        # positions predicting each target token.
        result = {}
        for layer in SELECTED_ATTENTION:
            weights = output.attentions[layer][0, :, query_positions, start:end]
            result[str(layer + 1)] = {
                "visual_mass": weights.sum(-1).mean(0).float().cpu().tolist(),
                "patch_maps": weights.mean(0).float().cpu().tolist(),
            }
        del output
        return result, None
    except (RuntimeError, NotImplementedError, ValueError) as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, f"{type(error).__name__}: {error}"
    finally:
        adapter.model.config._attn_implementation = original
        adapter.model.model.config._attn_implementation = original


def trace_one(adapter, image: Image.Image, prompt: str, cached_text: str, top_k: int, with_attention: bool):
    input_ids, labels = build_teacher_forcing(adapter, prompt, cached_text)
    ids = input_ids.to(adapter.model.device)
    targets = labels.to(adapter.model.device)
    pixels = process_image(adapter, image)
    _, position_ids, attention_mask, _, embeds, expanded_labels = (
        adapter.model.prepare_inputs_labels_for_multimodal(
            ids, None, None, None, targets, pixels, image_sizes=[image.size]
        )
    )
    if expanded_labels is None:
        raise RuntimeError("multimodal expansion returned no labels")
    patch_count = int(adapter.model.get_vision_tower().num_patches)
    start, end = visual_span(ids, patch_count)
    if end > embeds.shape[1]:
        raise RuntimeError("visual span exceeds expanded sequence")

    adapter.model.requires_grad_(False)
    leaf = embeds.detach().requires_grad_(True)
    with torch.enable_grad():
        output = adapter.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=leaf,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        vocabulary_weight = adapter.model.get_output_embeddings().weight
        logits = output.last_hidden_state.to(vocabulary_weight.dtype) @ vocabulary_weight.T
        shifted_targets = expanded_labels[:, 1:]
        answer_mask = shifted_targets.ne(IGNORE_INDEX)
        selected_logits = logits[:, :-1][answer_mask].float()
        selected_targets = shifted_targets[answer_mask]
        if selected_targets.numel() == 0:
            raise RuntimeError("no answer targets after multimodal expansion")
        first_query = int(torch.where(answer_mask[0])[0][0])
        first_target = int(selected_targets[0])
        lexical_score = logits[0, first_query, first_target].float()
        lexical_gradient = torch.autograd.grad(
            lexical_score, leaf, retain_graph=True
        )[0]
        groups = surface_groups(adapter)
        boundary_margin = semantic_margin(logits[0, first_query].float(), groups)
        semantic_gradient = torch.autograd.grad(
            boundary_margin, leaf, retain_graph=False
        )[0]
        lexical_patch_gradient = lexical_gradient[0, start:end].float()
        semantic_patch_gradient = semantic_gradient[0, start:end].float()
        patch_embedding = leaf[0, start:end].float()
        lexical_signed = (lexical_patch_gradient * patch_embedding).sum(-1)
        semantic_signed = (semantic_patch_gradient * patch_embedding).sum(-1)

        stats = token_statistics(
            selected_logits.detach(), selected_targets.detach(), adapter.tokenizer, top_k
        )
        lenses = logit_lens(adapter, output.hidden_states, answer_mask)
        polarities = polarity_positions(
            selected_targets.detach().cpu(), adapter.tokenizer, groups
        )
    del output, logits, lexical_gradient, semantic_gradient

    attentions, unavailable = (None, "disabled")
    if with_attention:
        attentions, unavailable = attention_summary(
            adapter, leaf, position_ids, attention_mask, answer_mask, start, end
        )
    return {
        "answer_token_count": len(stats),
        "answer_tokens": stats,
        "logit_lens": lenses,
        "visual_span": {"start": start, "end": end, "patch_count": patch_count},
        "prompt_boundary_semantic_margin": {
            "definition": "max Yes surface logit - max No surface logit",
            "value": float(boundary_margin.detach().cpu()),
            "signed_gradient_x_activation": semantic_signed.detach().cpu().tolist(),
            "absolute_gradient_x_activation": semantic_signed.abs().detach().cpu().tolist(),
            "gradient_norm": semantic_patch_gradient.norm(dim=-1).detach().cpu().tolist(),
            "primary_causal_diagnostic": True,
        },
        "first_target_lexical_influence": {
            "target_id": first_target,
            "target_token": adapter.tokenizer.decode([first_target]),
            "target_logit": float(lexical_score.detach().cpu()),
            "signed_gradient_x_activation": lexical_signed.detach().cpu().tolist(),
            "absolute_gradient_x_activation": lexical_signed.abs().detach().cpu().tolist(),
            "gradient_norm": lexical_patch_gradient.norm(dim=-1).detach().cpu().tolist(),
            "primary_causal_diagnostic": False,
            "warning": "lexical path only; not the Yes/No decision when the first token is e.g. 'The'",
        },
        "polarity_token_positions": polarities,
        "attention": attentions,
        "attention_unavailable_reason": unavailable,
    }


def completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(json.loads(line)["question_id"])
        for line in path.read_text().splitlines()
        if line.strip() and json.loads(line).get("status") == "ok"
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--base-answers", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qid", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_metadata(args: argparse.Namespace, requested: set[str]) -> dict[str, Any]:
    code_path = Path(__file__).resolve()
    return {
        "version": VERSION, "code_sha256": file_sha256(code_path),
        "questions": str(args.questions.resolve()),
        "questions_sha256": file_sha256(args.questions),
        "base_answers": str(args.base_answers.resolve()),
        "base_answers_sha256": file_sha256(args.base_answers),
        "image_root": str(args.image_root.resolve()),
        "explicit_qids": sorted(requested, key=int),
        "attention_requested": args.attention, "top_k": args.top_k,
        "teacher_forcing": True,
    }


def protocol_fingerprint(metadata: dict[str, Any]) -> str:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_resume_metadata(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError(f"resume metadata is missing: {path}")
    if json.loads(path.read_text()) != expected:
        raise RuntimeError("resume metadata mismatch; refusing to mix trace protocols")


def main() -> None:
    args = parse_args()
    requested = set(str(value) for value in args.qid)
    questions = {
        str(row.get("question_id", row.get("qid"))): row
        for row in load_rows(args.questions)
    }
    answers = {
        str(row.get("question_id", row.get("qid"))): row
        for row in load_rows(args.base_answers)
    }
    missing = requested - (set(questions) & set(answers))
    if missing:
        raise ValueError(f"qids missing from questions/base answers: {sorted(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = run_metadata(args, requested)
    metadata["fingerprint"] = protocol_fingerprint(metadata)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if args.resume:
        verify_resume_metadata(meta_path, metadata)
        repair_jsonl_tail(args.output)
    else:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    done = completed(args.output) if args.resume else set()
    mode = "a" if args.resume else "w"
    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    try:
        with args.output.open(mode) as handle:
            for key in sorted(requested, key=int):
                if key in done:
                    continue
                row = questions[key]
                record = {
                    "version": VERSION,
                    "question_id": key,
                    "image": row["image"],
                    "fingerprint": metadata["fingerprint"],
                    "status": "error",
                }
                try:
                    image_path = args.image_root / row["image"]
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                    prompt = official_prompt(row)
                    cached = answer_text(answers[key])
                    record.update(
                        {
                            "status": "ok",
                            "prompt": prompt,
                            "cached_text": cached,
                            "trace": trace_one(
                                adapter, image, prompt, cached, args.top_k, args.attention
                            ),
                        }
                    )
                except Exception as error:
                    record.update(
                        {
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                handle.write(json.dumps(record) + "\n")
                handle.flush()
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
