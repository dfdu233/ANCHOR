#!/usr/bin/env python3
"""Native Huatuo scoring for the frozen ASCC-v2 symmetric 2x2 assay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .audit_diagnostic_completion_substrate_v1 import sha256_file
from .prepare_ascc_factorial_v2 import VERSION as SUBSTRATE_VERSION
from .prepare_pragmatic_commitment_confirmatory_v1 import canonical_hash


VERSION = "huatuo-ascc-symmetric-factorial-score-v2"
DEFAULT_SUBSTRATE = Path("/home/dbw/ANCHOR/corrected_runs/ascc/confirmatory_substrate_v2")
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


def tree_fingerprint(root: Path, suffixes: set[str]) -> dict[str, Any]:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
    )
    records = [
        {
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "root": str(root.resolve()),
        "files": records,
        "fingerprint": canonical_hash(records),
    }


def validate_substrate(
    substrate_dir: Path, image_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]], list[str]]:
    config_path = substrate_dir / "substrate_config.json"
    manifest_path = substrate_dir / "selected_manifest.jsonl"
    if not config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("ASCC-v2 config and manifest are required")
    config = json.loads(config_path.read_text())
    if config.get("version") != SUBSTRATE_VERSION:
        raise ValueError("wrong ASCC-v2 substrate version")
    if config.get("status") != "untouched_confirmatory_census_gpu_not_run":
        raise ValueError("ASCC-v2 substrate is not certified untouched")
    payload = {key: value for key, value in config.items() if key != "fingerprint"}
    if canonical_hash(payload) != config.get("fingerprint"):
        raise ValueError("ASCC-v2 substrate fingerprint mismatch")
    if sha256_file(manifest_path) != config.get("manifest_sha256"):
        raise ValueError("ASCC-v2 manifest hash mismatch")
    if config.get("selection_uses_model_output") is not False:
        raise ValueError("ASCC-v2 selection was contaminated by model output")
    rows = load_jsonl(manifest_path)
    if len(rows) != int(config["registered_rows"]):
        raise ValueError("ASCC-v2 registered row count mismatch")
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate ASCC-v2 item identity")
    for row in rows:
        if not (image_root / f"{row['image_id']}.dicom").is_file():
            raise FileNotFoundError(image_root / f"{row['image_id']}.dicom")
        if str(row["fixed_prefix"]).endswith((" ", "\t", "\n")):
            raise ValueError("ASCC-v2 fixed prefix lacks a stable token boundary")
    prompts = list(config["prompts"])
    cells = {(row["speech_act"], row["clinical_noun"]) for row in prompts}
    if cells != {
        (speech_act, noun)
        for speech_act in ("describe", "list")
        for noun in ("findings", "abnormalities")
    }:
        raise ValueError("ASCC-v2 prompt factorial drifted")
    markers = [str(value) for value in config["markers"]]
    if markers != [" absent", " uncertain", " present"]:
        raise ValueError("ASCC-v2 symmetric marker contract drifted")
    if int(config["registered_jobs_per_model"]) != len(rows) * len(prompts):
        raise ValueError("ASCC-v2 registered job count mismatch")
    return config, rows, prompts, markers


def marker_coordinates(logits: Mapping[str, float]) -> dict[str, float]:
    import numpy as np

    negative = float(logits[" absent"])
    uncertain = float(logits[" uncertain"])
    positive = float(logits[" present"])
    commitment = float(np.logaddexp(positive, negative) - uncertain)
    return {
        "commitment": commitment,
        "uncertainty_preference": -commitment,
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
    parser.add_argument("--seed", type=int, default=47291)
    parser.add_argument("--stage", choices=("primary", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    substrate, rows, prompts, markers = validate_substrate(
        args.substrate_dir, args.image_root
    )
    candidate = {
        "version": VERSION,
        "created_at": utc_now(),
        "command": sys.argv,
        "substrate_dir": str(args.substrate_dir.resolve()),
        "substrate_fingerprint": substrate["fingerprint"],
        "substrate_manifest_sha256": substrate["manifest_sha256"],
        "registered_rows": len(rows),
        "registered_jobs": len(rows) * len(prompts),
        "primary_jobs": sum(row["edge_id"] == PRIMARY_EDGE for row in rows)
        * len(prompts),
        "model_id": "huatuo",
        "model_dir": str(args.model_dir.resolve()),
        "huatuo_root": str(args.huatuo_root.resolve()),
        "image_root": str(args.image_root.resolve()),
        "device": args.device,
        "seed": args.seed,
        "prompts": prompts,
        "markers": markers,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "forward_contract": (
            "ordinary native multimodal prefix forward; complete full-vocabulary logits; "
            "symmetric three-state contextual token gather; no generation"
        ),
        "reader_votes_visible_to_model": False,
        "scientific_outcomes_inspected_during_scoring": False,
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

    import torch
    import torch.nn.functional as F
    import transformers

    from .huatuo_lockin_adapter_v1 import (
        ASSISTANT_SUFFIX,
        IGNORE_INDEX,
        HuatuoLockinAdapter,
        partition_answer_tokens,
    )

    candidate["model_tree"] = tree_fingerprint(
        args.model_dir, {".bin", ".safetensors", ".json", ".model"}
    )
    candidate["huatuo_source_tree"] = tree_fingerprint(args.huatuo_root, {".py"})
    candidate["runtime"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
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
            raise FileExistsError("ASCC-v2 output exists; use --resume")
        existing = json.loads(config_path.read_text())
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("refusing incompatible ASCC-v2 resume")
        candidate = existing
    else:
        if args.resume:
            raise FileNotFoundError("--resume requires score_config.json")
        atomic_json(config_path, candidate)

    marker_ids_by_prefix: dict[str, dict[str, int]] = {}
    for prefix in sorted({str(row["fixed_prefix"]) for row in rows}):
        bare_ids = adapter.bot.tokenizer(prefix, add_special_tokens=False).input_ids
        contextual: dict[str, int] = {}
        for marker in markers:
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
            if mapping["prefix_token_ids"] != bare_ids:
                raise ValueError("ASCC-v2 marker changes contextual prefix tokenization")
            if len(mapping["continuation_token_ids"]) != 1:
                raise ValueError(f"ASCC-v2 marker is not one contextual token: {marker!r}")
            contextual[marker] = int(mapping["continuation_token_ids"][0])
        if len(set(contextual.values())) != 3:
            raise ValueError("ASCC-v2 markers do not map to three distinct tokens")
        marker_ids_by_prefix[prefix] = contextual
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
    order = list(rows)
    random.Random(args.seed).shuffle(order)
    stage_rows = (
        [row for row in order if row["edge_id"] == PRIMARY_EDGE]
        if args.stage == "primary"
        else order
    )
    completed = 0
    total = len(stage_rows) * len(prompts)
    for row in stage_rows:
        for prompt in prompts:
            path = shards_dir / f"{record_key(row['item_id'], prompt['name'])}.json"
            if path.exists():
                shard = json.loads(path.read_text())
                if shard.get("fingerprint") != fingerprint:
                    raise ValueError(f"incompatible ASCC-v2 shard: {path}")
                completed += 1
    print(f"strict resume: {completed}/{total} stage shards", flush=True)

    output_weight = adapter.bot.model.get_output_embeddings().weight
    for row_index, row in enumerate(stage_rows, 1):
        image_path = args.image_root / f"{row['image_id']}.dicom"
        try:
            for prompt in prompts:
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
                    [marker_ids[marker] for marker in markers],
                    dtype=torch.long,
                    device=output_weight.device,
                )
                layer_scores: dict[str, Any] = {}
                for layer_id, layer_number in zip(adapter.layer_ids, adapter.layer_numbers):
                    if layer_number == len(adapter.blocks):
                        full_logits = output.logits[0, -1].float()
                        source = "ordinary_causallm_forward"
                    else:
                        hidden = adapter.bot.model.model.norm(captured[layer_number][0])
                        full_logits = F.linear(
                            hidden.to(output_weight.dtype), output_weight
                        ).float()
                        source = "quartile_hidden_native_norm_full_lm_head"
                    selected = full_logits.index_select(0, token_ids)
                    values = {
                        marker: float(selected[index].detach().cpu())
                        for index, marker in enumerate(markers)
                    }
                    selected_log_mass = float(
                        (
                            torch.logsumexp(selected, dim=0)
                            - torch.logsumexp(full_logits, dim=0)
                        )
                        .detach()
                        .cpu()
                    )
                    layer_scores[layer_id] = {
                        "source": source,
                        "logits": values,
                        "coordinates": marker_coordinates(values),
                        "restricted_log_probability_mass": selected_log_mass,
                        "restricted_top1": bool(
                            int(full_logits.argmax().detach().cpu())
                            in set(marker_ids.values())
                        ),
                        "full_logit_mean": float(full_logits.mean().detach().cpu()),
                        "full_logit_std": float(full_logits.std().detach().cpu()),
                    }
                atomic_json(
                    path,
                    {
                        "version": VERSION,
                        "fingerprint": fingerprint,
                        "substrate_fingerprint": substrate["fingerprint"],
                        "item_id": row["item_id"],
                        "image_id": row["image_id"],
                        "edge_id": row["edge_id"],
                        "parent_votes": row["parent_votes"],
                        "child_votes": row["child_votes"],
                        "parent_by_reader": row["parent_by_reader"],
                        "child_by_reader": row["child_by_reader"],
                        "aspect_bucket": row["aspect_bucket"],
                        "prompt_name": prompt["name"],
                        "speech_act": prompt["speech_act"],
                        "clinical_noun": prompt["clinical_noun"],
                        "prompt": prompt["prompt"],
                        "fixed_prefix": row["fixed_prefix"],
                        "marker_token_ids": marker_ids,
                        "image_sha256": image_sha,
                        "layer_scores": layer_scores,
                        "created_at": utc_now(),
                    },
                )
                completed += 1
                if completed % 25 == 0:
                    print(
                        f"progress {completed}/{total}; row {row_index}/{len(stage_rows)}",
                        flush=True,
                    )
        finally:
            adapter._image_tensor_cache.clear()
    print(f"stage complete: {completed}/{total}", flush=True)


if __name__ == "__main__":
    main()
