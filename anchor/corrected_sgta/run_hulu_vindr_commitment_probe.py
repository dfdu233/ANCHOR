#!/usr/bin/env python3
"""Hulu-Med-4B replication of the frozen VinDr commitment probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from corrected_sgta.clinical_claims import VERSION as CLAIM_VERSION
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    VERBALIZERS,
    analyze,
    append_jsonl,
    atomic_json,
    calibrate_global_visual_null,
    load_image,
    load_jsonl,
    measure_one,
    prompt_for,
    resolve_image,
    sha256_file,
    validate_global_null_sidecar,
    freeze_or_validate_config,
)


VERSION = "hulu-vindr-commitment-probe-v8"


class HuluRuntime:
    def __init__(self, model_path: Path, max_visual_tokens: int):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        if max_visual_tokens <= 0:
            raise ValueError("max_visual_tokens must be positive")
        self.processor.image_processor.max_tokens = max_visual_tokens
        self.tokenizer = self.processor.tokenizer
        self.model.eval()


def prepared_embeddings_hulu(
    runtime: HuluRuntime, prompt: str, image: Image.Image
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, int]]:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = runtime.processor(
        images=[image],
        conversation=conversation,
        add_system_prompt=False,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    device = runtime.model.device
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            if key == "pixel_values":
                value = value.to(dtype=runtime.model.dtype)
            inputs[key] = value.to(device)
    input_ids = inputs["input_ids"]
    image_mask = input_ids[0].eq(runtime.model.config.image_token_index)
    positions = torch.nonzero(image_mask, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("Hulu prompt contains no expanded visual tokens")
    start, end = int(positions.min()), int(positions.max()) + 1
    if not bool(image_mask[start:end].all()) or int(image_mask.sum()) != end - start:
        raise RuntimeError("Hulu visual tokens are not one contiguous span")

    _, attention, position_ids, _, embeddings, _ = (
        runtime.model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            position_ids=inputs.get("position_ids"),
            pixel_values=inputs.get("pixel_values"),
            grid_sizes=inputs.get("grid_sizes"),
            merge_sizes=inputs.get("merge_sizes"),
            modals=inputs.get("modals"),
        )
    )
    if embeddings is None:
        raise RuntimeError("Hulu multimodal preparation returned no embeddings")
    if embeddings.shape[1] != input_ids.shape[1]:
        raise RuntimeError(
            f"Hulu visual expansion changed length {input_ids.shape[1]} -> {embeddings.shape[1]}"
        )
    if attention is None:
        attention = torch.ones(input_ids.shape, dtype=torch.bool, device=device)
    return embeddings, attention, position_ids, (start, end)


def model_file_inventory(model_path: Path) -> list[dict[str, object]]:
    names = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    )
    return [
        {
            "path": name,
            "bytes": (model_path / name).stat().st_size,
            "sha256": sha256_file(model_path / name),
        }
        for name in names
        if (model_path / name).is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/Hulu-Med-4B"))
    parser.add_argument("--layers", type=int, nargs="+", default=[9, 18, 27, 36])
    parser.add_argument("--precollapse-layer", type=int, default=18)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--intervention-layer", type=int, default=27)
    parser.add_argument("--intervention-strength", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--global-null-npy", type=Path)
    parser.add_argument("--calibrate-global-null-output", type=Path)
    parser.add_argument("--allow-plumbing-global-null", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--experiment-split", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="refresh summary.json from existing raw.jsonl without loading Hulu",
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
        if any("experiment_split" not in row for row in rows):
            raise ValueError("manifest lacks experiment_split; rebuild it before locked dev/test")
        rows = [row for row in rows if row["experiment_split"] == args.experiment_split]
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
    completed = (
        {
            str(row["record_key"])
            for row in load_jsonl(raw_path)
            if row.get("status") == "ok"
        }
        if args.resume and raw_path.exists()
        else set()
    )
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
        "model_inventory": model_file_inventory(args.model_dir),
        "method": "claim-plane-polarity-orthogonal-commitment-probe",
        "shared_analyzer": "run_huatuo_vindr_commitment_probe.analyze",
        "shared_analyzer_sha256": sha256_file(Path(analyze.__code__.co_filename)),
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
        "max_visual_tokens": args.max_visual_tokens,
        "verbalizers": VERBALIZERS,
        "prompt": (
            "Use row.question when present; otherwise ask whether the CXR shows "
            "<finding>. Always request exactly Yes, No, or Maybe."
        ),
        "mean_token_null": "replace every projected visual token by that image's projected-token mean",
        "norm_matched_null": (
            "shared v8 control preserving each visual token L2 norm while "
            "replacing its direction by the locked null direction"
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
        "dicom_preprocess": "shared deterministic loader from Huatuo probe",
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    config = freeze_or_validate_config(
        config, args.output_dir / "config.json", args.resume
    )

    runtime = HuluRuntime(args.model_dir, args.max_visual_tokens)
    if args.calibrate_global_null_output:
        vector, audit = calibrate_global_visual_null(
            runtime,
            rows,
            args.image_root,
            embedding_preparer=prepared_embeddings_hulu,
        )
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
        del runtime
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
            question = str(row.get("question") or prompt_for(str(row["finding"])))
            record["question"] = question
            record["measurement"] = measure_one(
                runtime,
                load_image(path),
                question,
                args.layers,
                args.tau,
                args.intervention_layer,
                args.intervention_strength,
                args.temperature,
                key,
                args.seed,
                embedding_preparer=prepared_embeddings_hulu,
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
        print(
            json.dumps(
                {
                    "progress": f"{index + 1}/{len(rows)}",
                    "record_key": key,
                    "status": record["status"],
                    "error": record.get("error"),
                }
            ),
            flush=True,
        )
    del runtime
    torch.cuda.empty_cache()
    all_rows = load_jsonl(raw_path)
    summary = analyze(
        all_rows, args.precollapse_layer, args.tau, args.seed, args.bootstrap_draws
    )
    summary["config"] = config
    summary["errors"] = sum(row.get("status") != "ok" for row in all_rows)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary.get("status") == "no_valid_rows":
        raise RuntimeError("probe produced no valid rows; inspect raw.jsonl")


if __name__ == "__main__":
    main()
