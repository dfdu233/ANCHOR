"""Single-case source-local causal test for frozen RULE qid 917.

This preregistered diagnostic replaces exactly the trace-selected top ten
percent of projected visual tokens by their cosine-nearest external-source
prototype.  Its shuffled control uses the identical prototype multiset at the
identical patch positions, so replacement count and prototype norms match
exactly.  The bottom condition applies the same nearest-prototype mechanism to
the trace-selected bottom ten percent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from corrected_sgta.causal_patch_intervention import (
    evaluate_margin,
    greedy_sequence,
    load_trace_rows,
    oriented_margin,
    patch_index_sets,
)
from corrected_sgta.diagnose_local_source_support import load_external_prototypes
from corrected_sgta.infer_rule_dg_adapter import (
    load_rows,
    official_prompt,
    repair_jsonl_tail,
)
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.trace_rule_generation import (
    file_sha256,
    surface_groups,
    visual_span,
)
from corrected_sgta.train_rule_dg_adapter import process_image


VERSION = "rule-q917-source-local-causal-v1"
FROZEN_QID = "917"
TOP_FRACTION = 0.10
CONDITIONS = ("original", "top_matched", "top_shuffled", "bottom_matched")


def nearest_prototypes(
    tokens: torch.Tensor, prototypes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cosine-nearest prototype rows and their indices."""
    if tokens.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("tokens and prototypes must both be matrices")
    if tokens.shape[1] != prototypes.shape[1]:
        raise ValueError("token/prototype width mismatch")
    token_unit = torch.nn.functional.normalize(tokens.float(), dim=-1)
    prototype_unit = torch.nn.functional.normalize(prototypes.float(), dim=-1)
    indices = (token_unit @ prototype_unit.T).argmax(dim=-1)
    return prototypes[indices].to(tokens), indices


def deterministic_permutation(qid: str, count: int) -> torch.Tensor:
    """Return a deterministic non-identity cyclic shuffle."""
    if count < 2:
        raise ValueError("at least two rows are required for a shuffled control")
    digest = hashlib.sha256(f"{VERSION}:shuffle:{qid}".encode()).digest()
    offset = 1 + int.from_bytes(digest[:8], "big") % (count - 1)
    return torch.roll(torch.arange(count), shifts=offset)


def replace_tokens(
    embeds: torch.Tensor,
    start: int,
    end: int,
    patch_indices: torch.Tensor,
    replacements: torch.Tensor,
) -> torch.Tensor:
    if embeds.ndim != 3 or embeds.shape[0] != 1:
        raise ValueError("expected embeddings shaped [1, sequence, width]")
    selected = patch_indices.to(device=embeds.device, dtype=torch.long)
    values = replacements.to(device=embeds.device, dtype=embeds.dtype)
    if values.shape != (selected.numel(), embeds.shape[-1]):
        raise ValueError("replacement shape mismatch")
    if int(selected.min()) < 0 or int(selected.max()) >= end - start:
        raise ValueError("patch index out of visual span")
    result = embeds.clone()
    result[0, start + selected] = values
    return result


def intervene_one(
    adapter: LlavaMedAlignmentAdapter,
    image: Image.Image,
    prompt: str,
    ground_truth: str,
    influence: torch.Tensor,
    prototype_union: torch.Tensor,
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
    visual = embeds[0, start:end]
    first_query = int(embeds.shape[1] - 1)
    groups = surface_groups(adapter)
    index_sets = patch_index_sets(influence, FROZEN_QID, TOP_FRACTION)
    top = index_sets["top10pct"]
    bottom = index_sets["bottom10pct"]

    top_nearest, top_prototype_indices = nearest_prototypes(
        visual[top.to(visual.device)], prototype_union
    )
    bottom_nearest, bottom_prototype_indices = nearest_prototypes(
        visual[bottom.to(visual.device)], prototype_union
    )
    permutation = deterministic_permutation(FROZEN_QID, len(top))
    top_shuffled = top_nearest[permutation.to(top_nearest.device)]

    condition_embeds = {
        "original": embeds,
        "top_matched": replace_tokens(
            embeds, start, end, top, top_nearest
        ),
        "top_shuffled": replace_tokens(
            embeds, start, end, top, top_shuffled
        ),
        "bottom_matched": replace_tokens(
            embeds, start, end, bottom, bottom_nearest
        ),
    }
    original_margin = evaluate_margin(
        adapter, embeds, position_ids, attention_mask, first_query, groups
    )
    original_correct = oriented_margin(original_margin, ground_truth)
    conditions: dict[str, Any] = {}
    for name in CONDITIONS:
        current = condition_embeds[name]
        margin = evaluate_margin(
            adapter, current, position_ids, attention_mask, first_query, groups
        )
        correct = oriented_margin(margin, ground_truth)
        conditions[name] = {
            "patch_count": 0 if name == "original" else int(len(top)),
            "yes_minus_no_margin": margin,
            "correct_oriented_margin": correct,
            "delta_correct_oriented": correct - original_correct,
            "greedy": greedy_sequence(
                adapter,
                current,
                position_ids,
                attention_mask,
                max_new_tokens,
            ),
        }
    conditions["top_matched"].update(
        {
            "patch_indices": top.tolist(),
            "prototype_indices": top_prototype_indices.cpu().tolist(),
        }
    )
    conditions["top_shuffled"].update(
        {
            "patch_indices": top.tolist(),
            "prototype_indices_before_shuffle": (
                top_prototype_indices.cpu().tolist()
            ),
            "permutation": permutation.tolist(),
            "prototype_norms_match_top_matched_exactly_as_multiset": True,
        }
    )
    conditions["bottom_matched"].update(
        {
            "patch_indices": bottom.tolist(),
            "prototype_indices": bottom_prototype_indices.cpu().tolist(),
        }
    )
    matched = conditions["top_matched"]
    shuffled = conditions["top_shuffled"]
    bottom_result = conditions["bottom_matched"]
    passed = (
        conditions["original"]["greedy"]["rule_label"] == "No"
        and matched["greedy"]["rule_label"] == "Yes"
        and matched["delta_correct_oriented"]
        > shuffled["delta_correct_oriented"]
        and matched["delta_correct_oriented"]
        > bottom_result["delta_correct_oriented"]
        and shuffled["greedy"]["rule_label"] != "Yes"
    )
    return {
        "ground_truth": "Yes",
        "projected_token_shape": [patch_count, int(embeds.shape[-1])],
        "visual_span": {"start": start, "end": end},
        "conditions": conditions,
        "preregistered_pass_rule": (
            "original greedy No; top_matched greedy Yes; matched correct-margin "
            "delta exceeds shuffled and bottom; top_shuffled does not rescue"
        ),
        "passed": passed,
        "decision": "proceed" if passed else "stop",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--base-answers", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--local-prototypes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qid", action="append", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = list(map(str, args.qid))
    if requested != [FROZEN_QID]:
        raise ValueError("this frozen diagnostic permits only one --qid 917")
    trace_rows = load_trace_rows(args.trace)
    questions = {
        str(row.get("question_id", row.get("qid"))): row
        for row in load_rows(args.questions)
    }
    answers = {
        str(row.get("question_id", row.get("qid"))): row
        for row in load_rows(args.base_answers)
    }
    if any(FROZEN_QID not in rows for rows in (trace_rows, questions, answers)):
        raise ValueError("qid 917 is missing from an input")
    prototypes, prototype_metadata = load_external_prototypes(
        args.local_prototypes
    )
    union = np.concatenate(list(prototypes.values()), axis=0).astype(np.float32)
    prototype_meta = args.local_prototypes.with_suffix(
        args.local_prototypes.suffix + ".meta.json"
    )
    metadata = {
        "version": VERSION,
        "code_sha256": file_sha256(Path(__file__).resolve()),
        "trace": str(args.trace.resolve()),
        "trace_sha256": file_sha256(args.trace),
        "questions": str(args.questions.resolve()),
        "questions_sha256": file_sha256(args.questions),
        "base_answers": str(args.base_answers.resolve()),
        "base_answers_sha256": file_sha256(args.base_answers),
        "image_root": str(args.image_root.resolve()),
        "local_prototypes": str(args.local_prototypes.resolve()),
        "local_prototypes_sha256": file_sha256(args.local_prototypes),
        "local_prototypes_meta_sha256": file_sha256(prototype_meta),
        "allowed_external_sources": sorted(prototypes),
        "external_union_shape": list(union.shape),
        "prototype_builder_version": prototype_metadata.get("version"),
        "explicit_qids": [FROZEN_QID],
        "top_fraction": TOP_FRACTION,
        "conditions": list(CONDITIONS),
        "alpha": 1.0,
        "nearest_k": 1,
        "parameter_tuning": False,
        "max_new_tokens": args.max_new_tokens,
    }
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    metadata["fingerprint"] = hashlib.sha256(payload.encode()).hexdigest()
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not meta_path.is_file() or json.loads(meta_path.read_text()) != metadata:
            raise RuntimeError("resume metadata mismatch; refusing protocol mixing")
        repair_jsonl_tail(args.output)
        if any(
            row.get("status") == "ok"
            and row.get("fingerprint") == metadata["fingerprint"]
            for row in load_rows(args.output)
        ):
            return
    else:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    record: dict[str, Any] = {
        "version": VERSION,
        "question_id": FROZEN_QID,
        "image": questions[FROZEN_QID]["image"],
        "fingerprint": metadata["fingerprint"],
        "status": "error",
    }
    try:
        trace = trace_rows[FROZEN_QID]
        if trace.get("status") != "ok":
            raise ValueError("trace row is not successful")
        influence = torch.tensor(
            trace["trace"]["prompt_boundary_semantic_margin"][
                "absolute_gradient_x_activation"
            ],
            dtype=torch.float32,
        )
        image_path = args.image_root / questions[FROZEN_QID]["image"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        record.update(
            intervene_one(
                adapter,
                image,
                official_prompt(questions[FROZEN_QID]),
                str(questions[FROZEN_QID]["answer"]),
                influence,
                torch.from_numpy(union).to(adapter.model.device),
                args.max_new_tokens,
            )
        )
        record["status"] = "ok"
    except Exception as error:
        record.update({"error_type": type(error).__name__, "error": str(error)})
    finally:
        adapter.close()
    with args.output.open("a" if args.resume else "w") as handle:
        handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
