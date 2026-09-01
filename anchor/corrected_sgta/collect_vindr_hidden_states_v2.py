#!/usr/bin/env python3
"""Collect unified post-block raw hidden states for the VinDr v2 probe.

Unlike the legacy logit-lens probe, every selected layer is captured at the
same architectural location: decoder block output, before the final decoder
normalization.  Final-norm/unembedding scores are retained as diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    VERBALIZERS,
    atomic_json,
    import_huatuo,
    label_ids,
    load_image,
    load_jsonl,
    prepared_embeddings,
    prompt_for,
    resolve_image,
    sha256_file,
)
from corrected_sgta.run_hulu_vindr_commitment_probe import (
    HuluRuntime,
    model_file_inventory,
    prepared_embeddings_hulu,
)


VERSION = "vindr-unified-post-block-hidden-v3-resumable"


def select_rows(
    rows: Sequence[dict[str, Any]],
    split: str,
    findings: set[str],
    votes: set[int],
    seed: int,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    output = [
        row
        for row in rows
        if str(row.get("experiment_split")) == split
        and str(row["finding"]) in findings
        and int(row["positive_votes"]) in votes
    ]
    output.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['finding']}:{row['positive_votes']}:{row['image_id']}".encode()
        ).hexdigest()
    )
    if max_samples is not None:
        output = output[:max_samples]
    if not output:
        raise ValueError("filters selected no manifest rows")
    return output


@torch.inference_mode()
def capture_post_block(
    runtime: Any,
    embeddings: torch.Tensor,
    attention: torch.Tensor,
    position_ids: torch.Tensor | None,
    visual_span: tuple[int, int],
    layers: Sequence[int],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, float]]:
    """Capture block outputs without applying final norm to any layer."""

    blocks = runtime.model.model.layers
    requested = sorted(set(int(value) for value in layers))
    if not requested or requested[0] < 1 or requested[-1] > len(blocks):
        raise ValueError(f"layers must lie in 1..{len(blocks)}")
    captured: dict[int, dict[str, np.ndarray]] = {}
    conformance_tensors: dict[int, torch.Tensor] = {}
    handles = []
    start, end = visual_span

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(f"unexpected decoder hidden shape {tuple(hidden.shape)}")
            visual = hidden[0, start:end].float()
            claim = hidden[0, -1].float()
            # Keep the exact full device tensor at the final block for
            # architectural conformance. RMSNorm may select a shape-dependent
            # BF16 kernel: normalizing only the final token can differ by one
            # quantization step (0.03125) from the model's full-sequence norm
            # even when the hook location is exactly right.
            if layer == requested[-1]:
                conformance_tensors[layer] = hidden.detach().clone()
            visual_unit = torch.nn.functional.normalize(visual, dim=-1)
            claim_unit = torch.nn.functional.normalize(claim, dim=0)
            alignment = visual_unit @ claim_unit
            probability = torch.softmax(alignment / 0.1, dim=0)
            normalized_entropy = -(
                probability * probability.clamp_min(1e-12).log()
            ).sum() / np.log(max(int(alignment.numel()), 2))
            topk = min(16, int(alignment.numel()))
            captured[layer] = {
                "claim": claim.cpu().numpy(),
                "visual_mean": visual.mean(dim=0).cpu().numpy(),
                "visual_std": visual.std(dim=0, unbiased=False).cpu().numpy(),
                "routing_statistics": np.asarray(
                    [
                        float(alignment.mean().cpu()),
                        float(alignment.std(unbiased=False).cpu()),
                        float(alignment.max().cpu()),
                        float(torch.topk(alignment, topk).values.mean().cpu()),
                        float(normalized_entropy.cpu()),
                        float(visual.norm(dim=-1).mean().cpu()),
                        float(visual.norm(dim=-1).std(unbiased=False).cpu()),
                    ],
                    dtype=np.float32,
                ),
            }

        return hook

    for layer in requested:
        handles.append(blocks[layer - 1].register_forward_hook(make_hook(layer)))
    try:
        output = runtime.model.model(
            input_ids=None,
            attention_mask=attention,
            position_ids=position_ids,
            inputs_embeds=embeddings,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(requested):
        raise RuntimeError(f"missing hooked layers: {sorted(set(requested) - set(captured))}")
    if requested[-1] != len(blocks):
        raise ValueError("the last decoder block is required for final-norm conformance")
    final_raw = conformance_tensors[requested[-1]]
    reconstructed = runtime.model.model.norm(final_raw)[0, -1].float()
    observed = output.last_hidden_state[0, -1].float()
    maximum_error = float((reconstructed - observed).abs().max().cpu())
    cosine = float(torch.nn.functional.cosine_similarity(reconstructed, observed, dim=0).cpu())
    if maximum_error > 0.02 or cosine < 0.999:
        raise RuntimeError(
            "final hook is not post-block/pre-final-norm: "
            f"max_abs={maximum_error:.6f}, cosine={cosine:.6f}"
        )
    return captured, {
        "final_norm_max_abs_error": maximum_error,
        "final_norm_cosine": cosine,
    }


@torch.inference_mode()
def diagnostic_logits(
    runtime: Any,
    vectors: dict[int, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Legacy final-norm/unembedding readout; never used as mechanism proof."""

    ids = label_ids(runtime)
    output_weight = runtime.model.get_output_embeddings().weight
    token_ids = torch.tensor(
        [ids[state] for state in VERBALIZERS], device=output_weight.device
    )
    selected = output_weight.index_select(0, token_ids)
    output = {}
    for layer, fields in sorted(vectors.items()):
        hidden = torch.from_numpy(fields["claim"]).to(
            device=selected.device, dtype=selected.dtype
        )
        normalized = runtime.model.model.norm(hidden.unsqueeze(0))[0]
        # Preserve the native hidden trajectory but accumulate the three-token
        # diagnostic readout in FP32.  BF16 logits around magnitude 20 move in
        # 0.125 steps, which is too coarse for reader-threshold calibration.
        logits = normalized.float() @ selected.float().T
        output[str(layer)] = {
            state: float(logits[index].float().cpu())
            for index, state in enumerate(VERBALIZERS)
        }
    return output


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checkpoint_arrays(
    claim: Sequence[np.ndarray],
    visual: Sequence[np.ndarray],
    visual_std: Sequence[np.ndarray],
    routing: Sequence[np.ndarray],
    layers: Sequence[int],
) -> dict[str, np.ndarray]:
    return {
        "claim": np.asarray(claim, dtype=np.float16),
        "visual_mean": np.asarray(visual, dtype=np.float16),
        "visual_std": np.asarray(visual_std, dtype=np.float16),
        "routing_statistics": np.asarray(routing, dtype=np.float32),
        "routing_statistic_names": np.asarray(
            [
                "claim_visual_cosine_mean",
                "claim_visual_cosine_std",
                "claim_visual_cosine_max",
                "claim_visual_cosine_top16_mean",
                "claim_visual_alignment_entropy",
                "visual_token_norm_mean",
                "visual_token_norm_std",
            ]
        ),
        "layers": np.asarray(sorted(set(layers)), dtype=np.int16),
    }


def load_checkpoint(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    expected_contract_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]]]:
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("resume_contract_sha256") != expected_contract_sha256:
        raise ValueError("existing collector directory has a different resume contract")
    state_path = output_dir / "checkpoint_state.json"
    metadata_path = output_dir / "checkpoint_metadata.jsonl"
    arrays_path = output_dir / "checkpoint.npz"
    present = [path.is_file() for path in (state_path, metadata_path, arrays_path)]
    if not any(present):
        return [], {
            "claim": [],
            "visual_mean": [],
            "visual_std": [],
            "routing_statistics": [],
        }
    if not all(present):
        raise ValueError("existing collector directory lacks a complete atomic checkpoint")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("metadata_sha256") != sha256_file(metadata_path):
        raise ValueError("checkpoint metadata hash mismatch")
    if state.get("arrays_sha256") != sha256_file(arrays_path):
        raise ValueError("checkpoint array hash mismatch")
    metadata = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_keys = [f"{row['finding']}:{row['image_id']}" for row in rows[: len(metadata)]]
    observed_keys = [str(row.get("record_key")) for row in metadata]
    if observed_keys != expected_keys or int(state.get("completed", -1)) != len(metadata):
        raise ValueError("checkpoint rows are not the exact ordered manifest prefix")
    source = np.load(arrays_path, allow_pickle=False)
    names = ("claim", "visual_mean", "visual_std", "routing_statistics")
    if any(source[name].shape[0] != len(metadata) for name in names):
        raise ValueError("checkpoint feature count disagrees with metadata")
    features = {name: [value for value in source[name]] for name in names}
    return metadata, features


def build_runtime(args: argparse.Namespace):
    if args.model == "huatuo":
        klass = import_huatuo(args.huatuo_root)
        runtime = klass(str(args.model_dir), device="cuda:0")

        def prepare(obj, prompt, image):
            tensor = torch.stack(obj.get_image_tensors([image])).to(
                obj.model.device, dtype=torch.bfloat16
            )
            return prepared_embeddings(obj, prompt, tensor)

        return runtime, prepare
    runtime = HuluRuntime(args.model_dir, args.max_visual_tokens)
    return runtime, prepared_embeddings_hulu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("pilot", "dev", "confirmation"), required=True)
    parser.add_argument("--findings", nargs="+", required=True)
    parser.add_argument("--votes", nargs="+", type=int, choices=range(4), required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    rows = select_rows(
        load_jsonl(args.manifest),
        args.split,
        set(args.findings),
        set(args.votes),
        args.seed,
        args.max_samples,
    )
    inventory = model_file_inventory(args.model_dir)
    resume_contract = {
        "version": VERSION,
        "model_id": args.model,
        "model_dir": str(args.model_dir.resolve()),
        "model_inventory": inventory,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "image_root": str(args.image_root.resolve()),
        "split": args.split,
        "findings": sorted(set(args.findings)),
        "votes": sorted(set(args.votes)),
        "layers": sorted(set(args.layers)),
        "max_samples": args.max_samples,
        "max_visual_tokens": args.max_visual_tokens,
        "seed": args.seed,
        "ordered_record_keys_sha256": hashlib.sha256(
            "\n".join(f"{row['finding']}:{row['image_id']}" for row in rows).encode()
        ).hexdigest(),
        "code_sha256": sha256_file(Path(__file__)),
    }
    resume_contract_sha256 = hashlib.sha256(
        json.dumps(resume_contract, sort_keys=True).encode()
    ).hexdigest()
    if args.output_dir.exists() and (args.output_dir / "summary.json").is_file():
        summary = json.loads((args.output_dir / "summary.json").read_text())
        if summary.get("status") == "complete" and summary.get("n") == len(rows):
            print(json.dumps({"status": "already_complete", "n": len(rows)}))
            return
        raise FileExistsError(f"non-reusable completed directory: {args.output_dir}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"output directory already exists; pass --resume for a verified checkpoint: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model,
        "model_dir": str(args.model_dir.resolve()),
        "model_inventory": inventory,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "image_root": str(args.image_root.resolve()),
        "split": args.split,
        "findings": sorted(set(args.findings)),
        "votes": sorted(set(args.votes)),
        "layers": sorted(set(args.layers)),
        "representation_location": "post_decoder_block_pre_final_norm",
        "plain_logit_lens_role": "diagnostic_only",
        "seed": args.seed,
        "checkpoint_every": args.checkpoint_every,
        "resume_contract": resume_contract,
        "resume_contract_sha256": resume_contract_sha256,
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    config_path = args.output_dir / "config.json"
    if not config_path.is_file():
        atomic_json(config_path, config)
    else:
        persisted_config = json.loads(config_path.read_text(encoding="utf-8"))
        if persisted_config.get("resume_contract_sha256") != resume_contract_sha256:
            raise ValueError("persisted config does not match the requested resume contract")
        config = persisted_config

    if args.resume:
        metadata, restored = load_checkpoint(
            args.output_dir, rows, resume_contract_sha256
        )
        claim_features = restored["claim"]
        visual_features = restored["visual_mean"]
        visual_std_features = restored["visual_std"]
        routing_statistics = restored["routing_statistics"]
    else:
        metadata = []
        claim_features = []
        visual_features = []
        visual_std_features = []
        routing_statistics = []
    start_index = len(metadata)
    runtime = prepare = None
    if start_index < len(rows):
        runtime, prepare = build_runtime(args)
    started = time.perf_counter()
    for index, row in enumerate(rows[start_index:], start=start_index):
        case_started = time.perf_counter()
        path = resolve_image(row, args.image_root)
        image = load_image(path)
        prompt = str(row.get("question") or prompt_for(str(row["finding"])))
        embeddings, attention, positions, span = prepare(runtime, prompt, image)
        features, hook_conformance = capture_post_block(
            runtime, embeddings, attention, positions, span, args.layers
        )
        diagnostic = diagnostic_logits(runtime, features)
        layers = sorted(features)
        claim_features.append(np.stack([features[layer]["claim"] for layer in layers]))
        visual_features.append(
            np.stack([features[layer]["visual_mean"] for layer in layers])
        )
        visual_std_features.append(
            np.stack([features[layer]["visual_std"] for layer in layers])
        )
        routing_statistics.append(
            np.stack([features[layer]["routing_statistics"] for layer in layers])
        )
        metadata.append(
            {
                "record_key": f"{row['finding']}:{row['image_id']}",
                "image_id": row["image_id"],
                "finding": row["finding"],
                "positive_votes": row["positive_votes"],
                "reader_votes": row["reader_votes"],
                "reader_support": row["reader_support"],
                "reader_state": row["reader_state"],
                "experiment_split": row["experiment_split"],
                "visual_tokens": span[1] - span[0],
                "diagnostic_plain_logit_lens": diagnostic,
                "hook_conformance": hook_conformance,
                "elapsed_seconds": time.perf_counter() - case_started,
            }
        )
        print(json.dumps({"progress": f"{index + 1}/{len(rows)}", "record_key": metadata[-1]["record_key"]}), flush=True)
        if len(metadata) % args.checkpoint_every == 0 or len(metadata) == len(rows):
            arrays = checkpoint_arrays(
                claim_features,
                visual_features,
                visual_std_features,
                routing_statistics,
                args.layers,
            )
            checkpoint_arrays_path = args.output_dir / "checkpoint.npz"
            checkpoint_metadata_path = args.output_dir / "checkpoint_metadata.jsonl"
            atomic_npz(checkpoint_arrays_path, **arrays)
            atomic_jsonl(checkpoint_metadata_path, metadata)
            atomic_json(
                args.output_dir / "checkpoint_state.json",
                {
                    "version": VERSION,
                    "completed": len(metadata),
                    "total": len(rows),
                    "resume_contract_sha256": resume_contract_sha256,
                    "arrays_sha256": sha256_file(checkpoint_arrays_path),
                    "metadata_sha256": sha256_file(checkpoint_metadata_path),
                    "last_record_key": metadata[-1]["record_key"],
                },
            )
    if runtime is not None:
        del runtime
    torch.cuda.empty_cache()

    final_arrays = checkpoint_arrays(
        claim_features,
        visual_features,
        visual_std_features,
        routing_statistics,
        args.layers,
    )
    atomic_npz(args.output_dir / "hidden_states.npz", **final_arrays)
    metadata_path = args.output_dir / "metadata.jsonl"
    atomic_jsonl(metadata_path, metadata)
    elapsed = time.perf_counter() - started
    atomic_json(
        args.output_dir / "summary.json",
        {
            "status": "complete",
            "n": len(metadata),
            "elapsed_seconds": elapsed,
            "seconds_per_case_median": float(np.median([row["elapsed_seconds"] for row in metadata])),
            "seconds_per_case_p90": float(np.quantile([row["elapsed_seconds"] for row in metadata], 0.9)),
            "hidden_states_sha256": sha256_file(args.output_dir / "hidden_states.npz"),
            "metadata_sha256": sha256_file(metadata_path),
            "config_fingerprint": config["fingerprint"],
        },
    )


if __name__ == "__main__":
    main()
