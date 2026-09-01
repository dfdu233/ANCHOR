#!/usr/bin/env python3
"""LLaVA-Med-v1.5 replication of the frozen clinical-claim probe.

The official model delays construction of its frozen CLIP tower.  Consequently
Transformers reports the serialized tower keys as unused and the official
loader then restores CLIP separately.  This runner records that loader choice;
``audit_llava_med_loader.py`` verifies checkpoint/base-tower equality and image
sensitivity before results are admitted.
"""

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
from corrected_sgta.models import center_pad_image
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


VERSION = "llava-med-vindr-commitment-probe-v6"


class LlavaRuntime:
    def __init__(self, model_path: Path, llava_root: Path, conv_mode: str):
        sys.path.insert(0, str(llava_root))

        # The local checkpoint is safetensors.  This compatibility shim only
        # bypasses a torch.load version gate reached by old tokenizer metadata.
        import transformers.modeling_utils as modeling_utils
        import transformers.utils.import_utils as import_utils

        import_utils.check_torch_load_is_safe = lambda: None
        modeling_utils.check_torch_load_is_safe = lambda: None

        from llava.model.builder import load_pretrained_model

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                str(model_path),
                None,
                "llava-med-v1.5-mistral-7b",
                device="cuda",
            )
        )
        self.model.eval()
        self.conv_mode = conv_mode


def prepared_embeddings_llava(
    runtime: LlavaRuntime, prompt: str, image: Image.Image
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, int]]:
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    image_token = DEFAULT_IMAGE_TOKEN
    if getattr(runtime.model.config, "mm_use_im_start_end", False):
        image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
    conversation = conv_templates[runtime.conv_mode].copy()
    conversation.append_message(conversation.roles[0], image_token + "\n" + prompt)
    conversation.append_message(conversation.roles[1], None)
    input_ids = tokenizer_image_token(
        conversation.get_prompt(),
        runtime.tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(runtime.model.device)
    positions = torch.nonzero(input_ids[0].eq(IMAGE_TOKEN_INDEX), as_tuple=False).flatten()
    if positions.numel() != 1:
        raise RuntimeError("LLaVA prompt must contain exactly one image token")

    prepared = image
    if getattr(runtime.model.config, "image_aspect_ratio", None) == "pad":
        mean = tuple(int(value * 255) for value in runtime.image_processor.image_mean)
        prepared = center_pad_image(image, mean)
    image_tensor = runtime.image_processor.preprocess(
        prepared, return_tensors="pt"
    )["pixel_values"].to(runtime.model.device, dtype=runtime.model.dtype)
    if prepared is not image:
        prepared.close()

    _, position_ids, attention, _, embeddings, _ = (
        runtime.model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            image_tensor,
            image_sizes=[image.size],
        )
    )
    if embeddings is None:
        raise RuntimeError("LLaVA multimodal preparation returned no embeddings")
    start = int(positions.item())
    patch_count = int(embeddings.shape[1] - input_ids.shape[1] + 1)
    end = start + patch_count
    if patch_count <= 0 or end > embeddings.shape[1]:
        raise RuntimeError(
            f"invalid LLaVA visual span {start}:{end} for {tuple(embeddings.shape)}"
        )
    if attention is None:
        attention = torch.ones(
            embeddings.shape[:2], dtype=torch.bool, device=embeddings.device
        )
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
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/dbw/models/LLaVA-Med-v1.5-mistral-7b"),
    )
    parser.add_argument(
        "--llava-root",
        type=Path,
        default=Path(
            "/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5"
        ),
    )
    parser.add_argument("--conv-mode", default="mistral_instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--precollapse-layer", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--intervention-layer", type=int, default=24)
    parser.add_argument("--intervention-strength", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.2)
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
        help="refresh summary.json from existing raw.jsonl without loading LLaVA-Med",
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
    null_metadata = (
        validate_global_null_sidecar(
            args.global_null_npy, args.allow_plumbing_global_null
        )
        if args.global_null_npy
        else None
    )
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"{args.output_dir} exists; pass --resume or choose another")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"

    rows = load_jsonl(args.manifest)
    if args.experiment_split != "all":
        if any("experiment_split" not in row for row in rows):
            raise ValueError("manifest lacks experiment_split")
        rows = [row for row in rows if row["experiment_split"] == args.experiment_split]
    rows = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row['finding']}:{row['image_id']}".encode()
        ).hexdigest(),
    )
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("no manifest rows selected")

    reference_sources = sorted(
        {str(row.get("reference_source", "unspecified")) for row in rows}
    )
    evidence_grades = sorted(
        {str(row.get("evidence_grade", "ungraded")) for row in rows}
    )
    formal_reference = all(row.get("formal_reference") is True for row in rows)
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
        "method": "claim-plane-polarity-orthogonal-commitment-probe",
        "shared_analyzer": "run_huatuo_vindr_commitment_probe.analyze",
        "shared_analyzer_sha256": sha256_file(Path(analyze.__code__.co_filename)),
        "model": str(args.model_dir.resolve()),
        "model_inventory": model_file_inventory(args.model_dir),
        "llava_root": str(args.llava_root.resolve()),
        "llava_builder_sha256": sha256_file(args.llava_root / "llava/model/builder.py"),
        "loader_audit_requirement": (
            "audit_llava_med_loader.py must show exact dtype-matched CLIP equality "
            "and nonzero cross-image visual/logit differences"
        ),
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
        "conv_mode": args.conv_mode,
        "verbalizers": VERBALIZERS,
        "mean_token_null": (
            "replace projected visual tokens by per-image mean or locked dev-global mean"
        ),
        "norm_matched_null": (
            "shared v8 control preserving each visual token L2 norm while "
            "replacing its direction by the locked null direction"
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
        "global_null_calibration": null_metadata,
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

    runtime = LlavaRuntime(args.model_dir, args.llava_root, args.conv_mode)
    if args.calibrate_global_null_output:
        vector, audit = calibrate_global_visual_null(
            runtime,
            rows,
            args.image_root,
            embedding_preparer=prepared_embeddings_llava,
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
        return

    global_null = (
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
            image = load_image(path)
            try:
                record["measurement"] = measure_one(
                    runtime,
                    image,
                    question,
                    args.layers,
                    args.tau,
                    args.intervention_layer,
                    args.intervention_strength,
                    args.temperature,
                    key,
                    args.seed,
                    embedding_preparer=prepared_embeddings_llava,
                    global_null_vector=global_null,
                )
            finally:
                image.close()
            record["question"] = question
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
