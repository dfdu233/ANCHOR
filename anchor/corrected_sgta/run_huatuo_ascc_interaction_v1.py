#!/usr/bin/env python3
"""Crash-safe Huatuo scorer for Ambiguity-Selective Commitment Collapse.

Each job performs one ordinary multimodal forward pass for a frozen prompt and
fixed assistant prefix.  At the next-token position it records the logits of
three single-token states (unlikely/possible/present) at decoder quartiles.
No answer is generated and no model output can change claim admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .audit_diagnostic_completion_substrate_v1 import sha256_file
from .prepare_ascc_interaction_v1 import MARKERS, PROMPTS, VERSION as SUBSTRATE_VERSION
from .prepare_pragmatic_commitment_confirmatory_v1 import canonical_hash


VERSION = "huatuo-ascc-interaction-score-v3"
DEFAULT_SUBSTRATE = Path(
    "/home/dbw/ANCHOR/corrected_runs/ascc/confirmatory_substrate_v1"
)
DEFAULT_IMAGE_ROOT = Path("/workspace/vinbigdata/train")
DEFAULT_MODEL_DIR = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO_ROOT = Path("/home/dbw/HuatuoGPT-Vision")
PRIMARY_EDGE = "lung_opacity_to_pneumonia"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def record_key(item_id: str, prompt_name: str) -> str:
    return hashlib.sha256(f"{VERSION}\0{item_id}\0{prompt_name}".encode()).hexdigest()


def validate_substrate(
    substrate_dir: Path, image_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path = substrate_dir / "substrate_config.json"
    manifest_path = substrate_dir / "selected_manifest.jsonl"
    if not config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("ASCC substrate config and manifest are both required")
    config = json.loads(config_path.read_text())
    if config.get("version") != SUBSTRATE_VERSION:
        raise ValueError("wrong ASCC substrate version")
    if config.get("status") != "untouched_confirmatory_census_gpu_not_run":
        raise ValueError("ASCC substrate is not certified untouched")
    payload = {key: value for key, value in config.items() if key != "fingerprint"}
    if canonical_hash(payload) != config.get("fingerprint"):
        raise ValueError("ASCC substrate fingerprint mismatch")
    if sha256_file(manifest_path) != config.get("manifest_sha256"):
        raise ValueError("ASCC manifest hash mismatch")
    if config.get("selection_uses_model_output") is not False:
        raise ValueError("model output contaminated ASCC selection")
    if config.get("prompts") != list(PROMPTS) or config.get("markers") != list(MARKERS):
        raise ValueError("ASCC prompt/marker contract drifted")
    rows = load_jsonl(manifest_path)
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate ASCC item identity")
    pair_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pair_groups.setdefault(str(row["matched_pair_id"]), []).append(row)
        image_path = image_root / f"{row['image_id']}.dicom"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if str(row["fixed_prefix"]).endswith((" ", "\n", "\t")):
            raise ValueError("fixed prefix must end at a stable token boundary")
    for pair_id, pair in pair_groups.items():
        if len(pair) != 2:
            raise ValueError(f"matched pair {pair_id} does not have exactly two rows")
        left, right = pair
        nuisance_left = (left["edge_id"], left["parent_votes"], left["aspect_bucket"])
        nuisance_right = (right["edge_id"], right["parent_votes"], right["aspect_bucket"])
        if nuisance_left != nuisance_right:
            raise ValueError(f"matched nuisance identity drift: {pair_id}")
        expected = {0, 1} if left["comparison_family"] == "negative_boundary" else {2, 3}
        if {int(left["child_votes"]), int(right["child_votes"])} != expected:
            raise ValueError(f"matched reader boundary drift: {pair_id}")
    prompt_groups: dict[str, set[str]] = {}
    for prompt in PROMPTS:
        prompt_groups.setdefault(prompt["prompt_pair_id"], set()).add(prompt["framing"])
    if not prompt_groups or any(values != {"neutral", "existential"} for values in prompt_groups.values()):
        raise ValueError("every prompt pair must contain both framings")
    return config, rows


def marker_coordinates(logits: Mapping[str, float]) -> dict[str, float]:
    negative = float(logits[" unlikely"])
    uncertain = float(logits[" possible"])
    positive = float(logits[" present"])
    return {
        "commitment": 0.5 * (positive + negative) - uncertain,
        "polarity": positive - negative,
        "positive_overcommitment": positive - uncertain,
        "negative_overcommitment": negative - uncertain,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate-dir", type=Path, default=DEFAULT_SUBSTRATE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=19031)
    parser.add_argument("--stage", choices=("primary", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    substrate, rows = validate_substrate(args.substrate_dir, args.image_root)
    runner_path = Path(__file__).resolve()
    candidate = {
        "version": VERSION,
        "created_at": utc_now(),
        "command": sys.argv,
        "substrate_dir": str(args.substrate_dir.resolve()),
        "substrate_fingerprint": substrate["fingerprint"],
        "substrate_manifest_sha256": substrate["manifest_sha256"],
        "registered_rows": len(rows),
        "registered_jobs": len(rows) * len(PROMPTS),
        "primary_jobs": sum(row["edge_id"] == PRIMARY_EDGE for row in rows) * len(PROMPTS),
        "model_id": "huatuo",
        "model_dir": str(args.model_dir.resolve()),
        "huatuo_root": str(args.huatuo_root.resolve()),
        "image_root": str(args.image_root.resolve()),
        "device": args.device,
        "seed": args.seed,
        "prompts": list(PROMPTS),
        "markers": list(MARKERS),
        "forward_contract": (
            "one native multimodal teacher-forced prefix forward; selected next-token "
            "logits at decoder quartiles; no generation"
        ),
        "runner_sha256": sha256_file(runner_path),
        "reader_votes_visible_to_model": False,
        "clinical_outcomes_inspected_during_scoring": False,
    }
    immutable = {key: value for key, value in candidate.items() if key not in {"created_at", "command"}}
    candidate["fingerprint"] = canonical_hash(immutable)
    print(
        json.dumps(
            {
                "preflight_passed": True,
                "registered_rows": len(rows),
                "registered_jobs": candidate["registered_jobs"],
                "primary_jobs": candidate["primary_jobs"],
                "edge_rows": dict(Counter(row["edge_id"] for row in rows)),
                "cpu_fingerprint": candidate["fingerprint"],
            },
            indent=2,
        ),
        flush=True,
    )
    if args.preflight_only:
        return

    # Heavy imports and model construction occur only after all CPU gates pass.
    import torch
    import torch.nn.functional as F

    from .huatuo_lockin_adapter_v1 import (
        ASSISTANT_SUFFIX,
        IGNORE_INDEX,
        HuatuoLockinAdapter,
        partition_answer_tokens,
    )

    adapter = HuatuoLockinAdapter(
        model_dir=args.model_dir, huatuo_root=args.huatuo_root, device=args.device
    )
    candidate["adapter_fingerprint"] = adapter.fingerprint()
    immutable = {
        key: value
        for key, value in candidate.items()
        if key not in {"created_at", "command", "fingerprint"}
    }
    candidate["fingerprint"] = canonical_hash(immutable)
    fingerprint = str(candidate["fingerprint"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "score_config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("score output exists; use --resume")
        existing = json.loads(config_path.read_text())
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("refusing incompatible ASCC resume")
        candidate = existing
    else:
        if args.resume:
            raise FileNotFoundError("--resume requires score_config.json")
        atomic_json(config_path, candidate)

    # Certify that all three states are exactly one contextual token for every prefix.
    marker_ids_by_prefix: dict[str, dict[str, int]] = {}
    for prefix in sorted({str(row["fixed_prefix"]) for row in rows}):
        bare_prefix_ids = adapter.bot.tokenizer(prefix, add_special_tokens=False).input_ids
        ids: dict[str, int] = {}
        for marker in MARKERS:
            answer = prefix + marker + ASSISTANT_SUFFIX
            encoded = adapter.bot.tokenizer(
                answer, add_special_tokens=False, return_offsets_mapping=True
            )
            mapping = partition_answer_tokens(
                answer_text=answer,
                prefix=prefix,
                continuation=marker,
                token_ids=encoded.input_ids,
                offsets=encoded.offset_mapping,
            )
            if mapping["prefix_token_ids"] != bare_prefix_ids:
                raise ValueError("marker changes contextual prefix tokenization")
            if len(mapping["continuation_token_ids"]) != 1:
                raise ValueError(f"marker is not one contextual token: {marker!r}")
            ids[marker] = int(mapping["continuation_token_ids"][0])
        if len(set(ids.values())) != len(MARKERS):
            raise ValueError("certainty markers do not have distinct token IDs")
        marker_ids_by_prefix[prefix] = ids
    atomic_json(
        args.output_dir / "marker_token_contract.json",
        {
            "fingerprint": fingerprint,
            "assistant_suffix": repr(ASSISTANT_SUFFIX),
            "marker_ids_by_prefix": marker_ids_by_prefix,
        },
    )

    shards_dir = args.output_dir / "shards"
    shards_dir.mkdir(exist_ok=True)
    registered = list(rows)
    random.Random(args.seed).shuffle(registered)
    stage_rows = (
        [row for row in registered if row["edge_id"] == PRIMARY_EDGE]
        if args.stage == "primary"
        else registered
    )

    completed = 0
    total_stage_jobs = len(stage_rows) * len(PROMPTS)
    for row in stage_rows:
        for prompt in PROMPTS:
            path = shards_dir / f"{record_key(row['item_id'], prompt['name'])}.json"
            if path.exists():
                shard = json.loads(path.read_text())
                if shard.get("fingerprint") != fingerprint:
                    raise ValueError(f"incompatible shard: {path}")
                completed += 1
    print(f"strict resume: {completed}/{total_stage_jobs} stage shards", flush=True)

    output_weight = adapter.bot.model.get_output_embeddings().weight
    for row_index, row in enumerate(stage_rows, 1):
        image_path = args.image_root / f"{row['image_id']}.dicom"
        try:
            for prompt in PROMPTS:
                path = shards_dir / f"{record_key(row['item_id'], prompt['name'])}.json"
                if path.exists():
                    continue
                prompt_ids = adapter._prompt_ids(prompt["prompt"], "image")
                prefix_ids = torch.tensor(
                    adapter.bot.tokenizer(
                        row["fixed_prefix"], add_special_tokens=False
                    ).input_ids,
                    dtype=torch.long,
                    device=adapter.bot.model.device,
                )
                full_ids = torch.cat((prompt_ids, prefix_ids))
                labels = torch.full_like(full_ids, IGNORE_INDEX)
                image_tensor, image_sha = adapter._image_tensor(image_path, "image")
                embeddings, attention, positions, _ = adapter._expand(
                    full_ids, labels, image_tensor, "image"
                )
                output, captured = adapter._capture_forward(
                    embeddings=embeddings,
                    attention=attention,
                    positions=positions,
                    prediction_mask=None,
                )
                marker_ids = marker_ids_by_prefix[row["fixed_prefix"]]
                token_ids = torch.tensor(
                    [marker_ids[marker] for marker in MARKERS],
                    dtype=torch.long,
                    device=output_weight.device,
                )
                standard = output.logits[0, -1].float().index_select(0, token_ids)
                layer_scores: dict[str, Any] = {}
                for layer_id, layer_number in zip(adapter.layer_ids, adapter.layer_numbers):
                    if layer_number == len(adapter.blocks):
                        logits = standard
                    else:
                        hidden = adapter.bot.model.model.norm(captured[layer_number][0])
                        # Use the native full-vocabulary LM-head shape before
                        # gathering.  A three-row BF16 GEMM can select a different
                        # kernel and differed from the ordinary forward by 0.125 on
                        # one canary despite matching on the preceding 42 shards.
                        logits = F.linear(
                            hidden.to(output_weight.dtype), output_weight
                        ).float().index_select(0, token_ids)
                    values = {
                        marker: float(logits[index].detach().cpu())
                        for index, marker in enumerate(MARKERS)
                    }
                    layer_scores[layer_id] = {
                        "logits": values,
                        "coordinates": marker_coordinates(values),
                    }
                shard = {
                    "version": VERSION,
                    "fingerprint": fingerprint,
                    "item_id": row["item_id"],
                    "image_id": row["image_id"],
                    "edge_id": row["edge_id"],
                    "matched_pair_id": row["matched_pair_id"],
                    "comparison_family": row["comparison_family"],
                    "parent_votes": row["parent_votes"],
                    "child_votes": row["child_votes"],
                    "ambiguity_state": row["ambiguity_state"],
                    "prompt_name": prompt["name"],
                    "prompt_pair_id": prompt["prompt_pair_id"],
                    "framing": prompt["framing"],
                    "prompt": prompt["prompt"],
                    "fixed_prefix": row["fixed_prefix"],
                    "marker_token_ids": marker_ids,
                    "image_sha256": image_sha,
                    "layer_scores": layer_scores,
                    "final_selected_logit_source": (
                        "ordinary CausalLM forward output.logits[0,-1] gathered at "
                        "the three frozen contextual token IDs"
                    ),
                    "manual_final_reprojection_prohibited": (
                        "BF16 LM-head GEMM is shape-dependent; a last-token-only "
                        "reprojection is not a valid identity oracle for the native "
                        "full-sequence forward"
                    ),
                    "created_at": utc_now(),
                }
                atomic_json(path, shard)
                completed += 1
                if completed % 25 == 0:
                    print(
                        f"progress {completed}/{total_stage_jobs}; row {row_index}/{len(stage_rows)}",
                        flush=True,
                    )
        finally:
            # The base adapter caches GPU image tensors; formal scoring needs only one
            # image at a time and must not grow memory with the census size.
            adapter._image_tensor_cache.clear()
    print(f"stage complete: {completed}/{total_stage_jobs}", flush=True)


if __name__ == "__main__":
    main()
