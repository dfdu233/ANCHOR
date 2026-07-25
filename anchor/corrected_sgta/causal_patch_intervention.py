"""Bounded projected-token interventions for six frozen RULE cases.

The diagnostic replaces exactly ten percent of projected visual tokens with
the same image's projected-token mean.  Patch sets are selected from a frozen
semantic-boundary gradient-times-activation trace (top, bottom, or qid-hash
random).  The intervention is causal; the gradient ranking that chooses its
targets is only an observational attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from corrected_sgta.infer_rule_dg_adapter import (
    load_rows,
    official_prompt,
    repair_jsonl_tail,
)
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.trace_rule_generation import (
    file_sha256,
    semantic_margin,
    surface_groups,
    visual_span,
)
from corrected_sgta.train_rule_dg_adapter import process_image, rule_label


VERSION = "rule-frozen-six-patch-intervention-v2"
FROZEN_QIDS = frozenset({"20", "917", "1062", "2273", "748", "636"})
ERROR_QIDS = frozenset({"20", "917", "1062"})
CONTROL_QIDS = FROZEN_QIDS - ERROR_QIDS
TOP_FRACTION = 0.10
CONDITIONS = ("original", "top10pct", "bottom10pct", "random10pct")


def stable_random_indices(qid: str, patch_count: int, count: int) -> torch.Tensor:
    """Return a deterministic qid-hash random subset on CPU."""
    seed_bytes = hashlib.sha256(
        f"{VERSION}:random-patches:{qid}".encode()
    ).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big") % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(patch_count, generator=generator)[:count]


def patch_index_sets(
    influence: torch.Tensor, qid: str, fraction: float = TOP_FRACTION
) -> dict[str, torch.Tensor]:
    values = torch.as_tensor(influence, dtype=torch.float32).flatten().cpu()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("influence must be a non-empty finite vector")
    if torch.any(values < 0):
        raise ValueError("absolute influence cannot be negative")
    count = max(1, int(math.ceil(values.numel() * fraction)))
    order = torch.argsort(values, stable=True)
    return {
        "top10pct": order[-count:].flip(0),
        "bottom10pct": order[:count],
        "random10pct": stable_random_indices(qid, values.numel(), count),
    }


def mean_replace_visual_tokens(
    inputs_embeds: torch.Tensor, start: int, end: int, indices: torch.Tensor
) -> torch.Tensor:
    """Clone embeddings and mean-replace selected projected visual tokens."""
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError("expected embeddings shaped [1, sequence, width]")
    if not (0 <= start < end <= inputs_embeds.shape[1]):
        raise ValueError("invalid visual span")
    result = inputs_embeds.clone()
    visual = result[0, start:end]
    selected = indices.to(device=visual.device, dtype=torch.long)
    if selected.numel() == 0 or int(selected.min()) < 0 or int(selected.max()) >= len(visual):
        raise ValueError("patch indices are empty or out of range")
    visual[selected] = visual.mean(dim=0)
    return result


def oriented_margin(yes_minus_no: float, ground_truth: str) -> float:
    label = ground_truth.strip().lower()
    if "yes" in label:
        return float(yes_minus_no)
    if "no" in label:
        return -float(yes_minus_no)
    raise ValueError(f"ground truth is not binary: {ground_truth!r}")


def load_trace_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in load_rows(path):
        qid = str(row["question_id"])
        if qid in rows:
            raise ValueError(f"duplicate trace qid: {qid}")
        rows[qid] = row
    return rows


def evaluate_margin(
    adapter: LlavaMedAlignmentAdapter,
    embeds: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    first_query: int,
    groups: dict[str, list[int]],
) -> float:
    with torch.inference_mode():
        output = adapter.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=embeds,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        weight = adapter.model.get_output_embeddings().weight
        boundary_logits = (
            output.last_hidden_state[0, first_query].to(weight.dtype) @ weight.T
        ).float()
        margin = float(semantic_margin(boundary_logits, groups).cpu())
    del output, boundary_logits
    return margin


def greedy_sequence(
    adapter: LlavaMedAlignmentAdapter,
    embeds: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Greedy decode from already-expanded embeddings, bypassing LLaVA's wrapper."""
    with torch.inference_mode():
        generated = super(type(adapter.model), adapter.model).generate(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=adapter.tokenizer.eos_token_id,
        )
    text = adapter.tokenizer.batch_decode(
        generated, skip_special_tokens=True
    )[0].strip()
    return {"text": text, "rule_label": rule_label(text)}


def intervene_one(
    adapter: LlavaMedAlignmentAdapter,
    image: Image.Image,
    prompt: str,
    ground_truth: str,
    influence: torch.Tensor,
    qid: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    ids = adapter._prompt_ids(prompt).to(adapter.model.device)
    pixels = process_image(adapter, image)
    _, position_ids, attention_mask, _, embeds, expanded_labels = (
        adapter.model.prepare_inputs_labels_for_multimodal(
            ids, None, None, None, None, pixels, image_sizes=[image.size]
        )
    )
    if expanded_labels is not None:
        raise RuntimeError("prompt-only expansion unexpectedly returned labels")
    patch_count = int(adapter.model.get_vision_tower().num_patches)
    start, end = visual_span(ids, patch_count)
    first_query = int(embeds.shape[1] - 1)
    index_sets = patch_index_sets(influence, qid)
    groups = surface_groups(adapter)
    original = evaluate_margin(
        adapter, embeds, position_ids, attention_mask, first_query, groups
    )
    original_correct = oriented_margin(original, ground_truth)
    original_generation = greedy_sequence(
        adapter, embeds, position_ids, attention_mask, max_new_tokens
    )
    conditions: dict[str, Any] = {
        "original": {
            "patch_count": 0,
            "yes_minus_no_margin": original,
            "correct_oriented_margin": original_correct,
            "delta_yes_minus_no": 0.0,
            "delta_correct_oriented": 0.0,
            "greedy": original_generation,
        }
    }
    for condition in CONDITIONS[1:]:
        indices = index_sets[condition]
        modified = mean_replace_visual_tokens(embeds, start, end, indices)
        margin = evaluate_margin(
            adapter, modified, position_ids, attention_mask, first_query, groups
        )
        correct = oriented_margin(margin, ground_truth)
        generation = greedy_sequence(
            adapter, modified, position_ids, attention_mask, max_new_tokens
        )
        conditions[condition] = {
            "patch_count": int(indices.numel()),
            "patch_indices": indices.tolist(),
            "yes_minus_no_margin": margin,
            "correct_oriented_margin": correct,
            "delta_yes_minus_no": margin - original,
            "delta_correct_oriented": correct - original_correct,
            "greedy": generation,
        }
    return {
        "ground_truth": "Yes" if "yes" in ground_truth.lower() else "No",
        "case_role": "error" if qid in ERROR_QIDS else "control",
        "projected_token_shape": [patch_count, int(embeds.shape[-1])],
        "visual_span": {"start": start, "end": end},
        "selection_source": (
            "frozen absolute gradient-times-activation of prompt-boundary "
            "Yes-minus-No margin; attribution is observational"
        ),
        "intervention": (
            "causal mean replacement of selected projected visual tokens"
        ),
        "conditions": conditions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--base-answers", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qid", action="append", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def completed(path: Path, fingerprint: str) -> set[str]:
    if not path.is_file():
        return set()
    result = set()
    for row in load_rows(path):
        if row.get("status") == "ok" and row.get("fingerprint") == fingerprint:
            result.add(str(row["question_id"]))
    return result


def main() -> None:
    args = parse_args()
    requested = set(map(str, args.qid))
    outside = requested - FROZEN_QIDS
    if outside:
        raise ValueError(f"qids outside frozen six are forbidden: {sorted(outside)}")
    trace_rows = load_trace_rows(args.trace)
    questions = {
        str(row.get("question_id", row.get("qid"))): row
        for row in load_rows(args.questions)
    }
    answers = {
        str(row.get("question_id", row.get("qid"))): row
        for row in load_rows(args.base_answers)
    }
    missing = requested - (set(trace_rows) & set(questions) & set(answers))
    if missing:
        raise ValueError(f"requested qids missing from inputs: {sorted(missing)}")
    trace_meta = args.trace.with_suffix(args.trace.suffix + ".meta.json")
    metadata = {
        "version": VERSION,
        "code_sha256": file_sha256(Path(__file__).resolve()),
        "trace": str(args.trace.resolve()),
        "trace_sha256": file_sha256(args.trace),
        "trace_meta_sha256": file_sha256(trace_meta),
        "questions": str(args.questions.resolve()),
        "questions_sha256": file_sha256(args.questions),
        "base_answers": str(args.base_answers.resolve()),
        "base_answers_sha256": file_sha256(args.base_answers),
        "image_root": str(args.image_root.resolve()),
        "explicit_qids": sorted(requested, key=int),
        "frozen_qid_universe": sorted(FROZEN_QIDS, key=int),
        "error_qids": sorted(ERROR_QIDS, key=int),
        "control_qids": sorted(CONTROL_QIDS, key=int),
        "top_fraction": TOP_FRACTION,
        "conditions": list(CONDITIONS),
        "greedy_generation": True,
        "max_new_tokens": args.max_new_tokens,
        "parameter_tuning": False,
    }
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    metadata["fingerprint"] = hashlib.sha256(payload.encode()).hexdigest()
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not meta_path.is_file() or json.loads(meta_path.read_text()) != metadata:
            raise RuntimeError("resume metadata mismatch; refusing to mix protocols")
        repair_jsonl_tail(args.output)
    else:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    done = completed(args.output, metadata["fingerprint"]) if args.resume else set()
    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    try:
        mode = "a" if args.resume else "w"
        with args.output.open(mode) as handle:
            for qid in sorted(requested, key=int):
                if qid in done:
                    continue
                question = questions[qid]
                record: dict[str, Any] = {
                    "version": VERSION,
                    "question_id": qid,
                    "image": question["image"],
                    "fingerprint": metadata["fingerprint"],
                    "status": "error",
                }
                try:
                    trace = trace_rows[qid]
                    if trace.get("status") != "ok":
                        raise ValueError("trace row is not successful")
                    influence = torch.tensor(
                        trace["trace"]["prompt_boundary_semantic_margin"][
                            "absolute_gradient_x_activation"
                        ],
                        dtype=torch.float32,
                    )
                    with Image.open(args.image_root / question["image"]) as source:
                        image = source.convert("RGB")
                    record.update(
                        intervene_one(
                            adapter,
                            image,
                            official_prompt(question),
                            str(question["answer"]),
                            influence,
                            qid,
                            args.max_new_tokens,
                        )
                    )
                    record["status"] = "ok"
                except Exception as error:
                    record.update(
                        {"error_type": type(error).__name__, "error": str(error)}
                    )
                handle.write(json.dumps(record) + "\n")
                handle.flush()
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
