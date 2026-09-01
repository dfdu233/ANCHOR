#!/usr/bin/env python3
"""Crash-safe generation-only Clinical Presupposition Amplification probe.

The three prompts are deliberately *not* paraphrases.  They intervene on the
pragmatic task and answer space (unrestricted evidence summary, present-focused
listing, and absence-focused listing) while keeping the image, decoder, and
output budget fixed.  This runner records generation behavior only.  It never
extracts clinical claims and never assigns clinical correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from corrected_sgta.run_huatuo_dicom_render_pilot_v1 import (
    canonical_json_sha256,
    model_artifact_fingerprint,
)
from corrected_sgta.prepare_vindr_reader_manifest import (
    build_records,
    load_ontology_findings,
    read_votes,
    select_ontology_columns,
)
from corrected_sgta.clinical_claims import normalize_term
from corrected_sgta.prepare_vindr_reader_manifest_v2 import (
    fixed_panel_records,
    three_way_split,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    atomic_json,
    dicom_to_pil,
    import_huatuo,
    sha256_file,
)


VERSION = "clinical-presupposition-generation-only-v1"
SELECTION_VERSION = "label-blind-image-sha256-pilot-200-v1"
RENDER_CONTRACT = "dicom-percentile-0p5-99p5-monochrome-aware-v1"
DEFAULT_LABELS_CSV = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/"
    "annotations/image_labels_train.csv"
)
DEFAULT_ONTOLOGY = Path("/home/dbw/ANCHOR/configs/missing_third_state_vindr_ontology.json")
DEFAULT_IMAGE_ROOT = Path("/workspace/vinbigdata/train")
DEFAULT_MODEL_DIR = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO_ROOT = Path("/home/dbw/HuatuoGPT-Vision")

# A common response-form budget reduces trivial length differences without
# equating the task semantics.  The downstream screen still performs strict
# answer-length matching and may reject the candidate if overlap is too small.
COMMON_RESPONSE_FORM = (
    "Answer in one concise sentence of at most 30 words. "
    "State uncertainty explicitly rather than guessing."
)
PROMPT_CONDITIONS = (
    {
        "name": "neutral",
        "prompt": (
            "Summarize the image-grounded findings on this chest X-ray, including "
            "clinically relevant present, absent, or uncertain observations. "
            + COMMON_RESPONSE_FORM
        ),
        "pragmatic_task": "unrestricted_image_grounded_summary",
        "answer_space_focus": "present_absent_or_uncertain_findings",
    },
    {
        "name": "existential",
        "prompt": (
            "What abnormalities are present on this chest X-ray? Mention the "
            "present or uncertain abnormalities that are clinically relevant. "
            + COMMON_RESPONSE_FORM
        ),
        "pragmatic_task": "existential_abnormality_listing",
        "answer_space_focus": "present_or_uncertain_abnormalities",
    },
    {
        "name": "negative_obligation",
        "prompt": (
            "What common abnormalities are absent on this chest X-ray? Mention "
            "clinically relevant absent findings and any uncertainty. "
            + COMMON_RESPONSE_FORM
        ),
        "pragmatic_task": "negative_obligation_listing",
        "answer_space_focus": "absent_or_uncertain_abnormalities",
    },
)

# This is a conservative surface behavior diagnostic, not a clinical judge.
REFUSAL_SURFACE_PHRASES = (
    "i cannot interpret",
    "i can't interpret",
    "cannot diagnose",
    "can't diagnose",
    "unable to interpret",
    "consult a radiologist",
    "consult your doctor",
    "not a medical professional",
)
FROZEN_READER_PANEL = ("R8", "R9", "R10")
FROZEN_FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "other_lesion",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def prompt_contract() -> list[dict[str, str]]:
    rows = [dict(row) for row in PROMPT_CONDITIONS]
    if [row["name"] for row in rows] != [
        "neutral",
        "existential",
        "negative_obligation",
    ]:
        raise RuntimeError("the frozen prompt-condition order changed")
    if len({row["prompt"] for row in rows}) != 3:
        raise RuntimeError("prompt texts must be distinct")
    if any(COMMON_RESPONSE_FORM not in row["prompt"] for row in rows):
        raise RuntimeError("all prompts must share the frozen response form")
    return rows


def claim_universe_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "finding": str(row["finding"]),
            "positive_votes": int(row["positive_votes"]),
            "reader_count": int(row["reader_count"]),
            "reader_state": str(row["reader_state"]),
            "reference_contract_version": str(row["reference_contract_version"]),
        }
        for row in sorted(rows, key=lambda item: str(item["finding"]))
    ]
    return canonical_json_sha256(payload)


def full_fixed_panel_universe(labels_csv: Path, ontology: Path, seed: int) -> list[dict[str, Any]]:
    """Build every eligible claim for every exact-panel image before sampling."""

    votes, source_findings, _, _ = read_votes(labels_csv)
    selected_findings, _ = select_ontology_columns(
        source_findings, load_ontology_findings(ontology)
    )
    selected_findings = [
        finding
        for finding in selected_findings
        if normalize_term(finding) in set(FROZEN_FINDINGS)
    ]
    if {normalize_term(finding) for finding in selected_findings} != set(FROZEN_FINDINGS):
        raise ValueError("source/ontology does not contain the frozen eight-finding universe")
    all_records, _ = build_records(votes, selected_findings, "local-only")
    rows = fixed_panel_records(all_records, FROZEN_READER_PANEL)
    for row in rows:
        image_id = str(row["image_id"])
        row["experiment_split"] = three_way_split(image_id, seed)
        row["split_assignment"] = "global_image_sha256_20_20_60"
        row["dicom_relpath"] = f"train/{image_id}.dicom"
        row["reference_contract_version"] = "presupposition-full-fixed-panel-v1"
        row.pop("dicom_url", None)
    return rows


def select_label_blind_images(
    rows: Sequence[Mapping[str, Any]], split: str, limit: int, seed: int
) -> list[dict[str, Any]]:
    """Select by image ID only; reader outcomes cannot affect membership."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("experiment_split")) == split:
            grouped[str(row["image_id"])].append(row)
    ordered_ids = sorted(
        grouped,
        key=lambda image_id: hashlib.sha256(
            f"{SELECTION_VERSION}:{seed}:{split}:{image_id}".encode()
        ).hexdigest(),
    )
    if len(ordered_ids) < limit:
        raise ValueError(f"split {split!r} has {len(ordered_ids)} images, needs {limit}")
    selected = []
    for image_id in ordered_ids[:limit]:
        item_rows = grouped[image_id]
        dicom_paths = {str(row["dicom_relpath"]) for row in item_rows}
        finding_names = sorted(str(row["finding"]) for row in item_rows)
        if len(dicom_paths) != 1:
            raise ValueError(f"inconsistent DICOM path for image {image_id}")
        if len(finding_names) != len(set(finding_names)):
            raise ValueError(f"duplicate finding reference for image {image_id}")
        selected.append(
            {
                "item_id": image_id,
                "image_id": image_id,
                "dicom_relpath": next(iter(dicom_paths)),
                "experiment_split": split,
                "claim_names": finding_names,
                "claim_universe_sha256": claim_universe_sha256(item_rows),
                "reference_row_count": len(item_rows),
                "selection_uses_reader_labels": False,
            }
        )
    return selected


def surface_refusal(text: str) -> dict[str, Any]:
    normalized = " ".join(text.lower().split())
    matches = [phrase for phrase in REFUSAL_SURFACE_PHRASES if phrase in normalized]
    return {
        "surface_refusal_match": bool(matches),
        "surface_refusal_phrases": matches,
        "interpretation": (
            "conservative literal-phrase diagnostic only; formal refusal status "
            "is assigned by the shared audited evaluator"
        ),
    }


def record_key(item_id: str, condition: str) -> str:
    return hashlib.sha256(f"{item_id}\0{condition}".encode()).hexdigest()


def validate_shard(
    payload: Mapping[str, Any], item_id: str, condition: str, fingerprint: str
) -> None:
    required = {
        "version",
        "item_id",
        "image_id",
        "prompt_condition",
        "prompt",
        "text",
        "generated_token_count",
        "generated_token_ids",
        "hit_max_new_tokens",
        "fingerprint",
        "claim_universe_sha256",
        "clinical_claim_evaluation_status",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"incomplete shard {item_id}/{condition}: {sorted(missing)}")
    if payload["version"] != VERSION:
        raise ValueError(f"wrong shard version: {payload['version']}")
    if payload["item_id"] != item_id or payload["prompt_condition"] != condition:
        raise ValueError(f"shard identity mismatch: {item_id}/{condition}")
    if payload["fingerprint"] != fingerprint:
        raise ValueError(f"shard fingerprint mismatch: {item_id}/{condition}")
    if payload["clinical_claim_evaluation_status"] != "pending_shared_audit":
        raise ValueError("generation runner must not assign clinical truth")
    token_ids = payload["generated_token_ids"]
    if not isinstance(token_ids, list) or len(token_ids) != int(
        payload["generated_token_count"]
    ):
        raise ValueError(f"invalid token accounting: {item_id}/{condition}")
    if not str(payload["text"]).strip():
        raise ValueError(f"empty generation: {item_id}/{condition}")


def generation_order(items: Sequence[Mapping[str, Any]], seed: int) -> list[tuple[dict[str, Any], dict[str, str]]]:
    """Interleave conditions so a partial run is not a condition-prefix sample."""

    work = [(dict(item), dict(condition)) for item in items for condition in prompt_contract()]
    rng = random.Random(seed)
    rng.shuffle(work)
    return work


def prepare_inputs(bot: Any, prompt: str, image: Any) -> tuple[torch.Tensor, torch.Tensor]:
    moderated = bot.input_moderation(prompt)
    with_image = bot.insert_image_placeholder(moderated, 1)
    conversation = bot.get_conv_without_history(with_image)
    input_ids = bot.preprocess(conversation, return_tensors="pt").unsqueeze(0).to(bot.device)
    image_tensors = torch.stack(bot.get_image_tensors([image])).to(
        dtype=torch.bfloat16, device=bot.device
    )
    return input_ids, image_tensors


def exact_generate(
    bot: Any,
    prompt: str,
    image: Any,
    *,
    max_new_tokens: int,
    repetition_penalty: float,
) -> dict[str, Any]:
    """Return Huatuo's actual generation-only token sequence, without retokenizing."""

    input_ids, image_tensors = prepare_inputs(bot, prompt, image)
    kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 1,
        "repetition_penalty": repetition_penalty,
        "eos_token_id": bot.tokenizer.eos_token_id,
        "pad_token_id": bot.tokenizer.pad_token_id or bot.tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": False,
        "use_cache": True,
    }
    started = time.monotonic()
    with torch.inference_mode():
        outputs = bot.model.generate(input_ids, images=image_tensors, **kwargs)
    # Huatuo's inputs_embeds generation path returns only newly generated IDs;
    # this is the same contract exercised by run_huatuo_rule_feddg.
    generated_ids = [int(value) for value in outputs.sequences[0].detach().cpu().tolist()]
    text = bot.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "generated_token_ids": generated_ids,
        "generated_token_count": len(generated_ids),
        "prompt_token_count": int(input_ids.shape[1]),
        "sequence_layout": "huatuo_generation_only",
        "elapsed_seconds": time.monotonic() - started,
    }


def freeze_config(candidate: dict[str, Any], path: Path, resume: bool) -> dict[str, Any]:
    immutable = {key: value for key, value in candidate.items() if key not in {"created_at", "command"}}
    candidate["fingerprint"] = canonical_json_sha256(immutable)
    if not resume:
        if path.exists():
            raise FileExistsError(f"output already configured; use --resume: {path}")
        atomic_json(path, candidate)
        return candidate
    if not path.is_file():
        raise FileNotFoundError("--resume requires the original generation_config.json")
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("fingerprint") != candidate["fingerprint"]:
        raise ValueError("refusing to resume an incompatible generation run")
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    reference_rows = full_fixed_panel_universe(args.labels_csv, args.ontology, args.seed)
    items = select_label_blind_images(reference_rows, args.split, args.limit, args.seed)
    selected_manifest_text = "".join(
        json.dumps(item, sort_keys=True) + "\n" for item in items
    )
    selected_manifest_sha256 = hashlib.sha256(selected_manifest_text.encode()).hexdigest()
    renderer_source = Path(sys.modules[dicom_to_pil.__module__].__file__).resolve()
    runner_path = Path(__file__).resolve()
    generation = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": 1,
        "repetition_penalty": 1.2,
    }
    candidate_config = {
        "version": VERSION,
        "created_at": utc_now(),
        "command": sys.argv,
        "evidence_scope": "generation-only candidate; clinical truth pending shared audited evaluator",
        "model_id": "huatuo",
        "model_dir": str(args.model_dir.resolve()),
        "model_artifact_fingerprint": model_artifact_fingerprint(args.model_dir),
        "huatuo_root": str(args.huatuo_root.resolve()),
        "labels_csv": str(args.labels_csv.resolve()),
        "labels_csv_sha256": sha256_file(args.labels_csv),
        "ontology": str(args.ontology.resolve()),
        "ontology_sha256": sha256_file(args.ontology),
        "reader_panel": list(FROZEN_READER_PANEL),
        "reference_universe": "all exact-panel images before outcome-independent hash selection",
        "image_root": str(args.image_root.resolve()),
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed,
        "selection_version": SELECTION_VERSION,
        "selection_uses_reader_labels": False,
        "selected_manifest_sha256": selected_manifest_sha256,
        "claim_universe": "eight fixed VinDr multi-reader image-grounded finding claims per selected image",
        "prompts_are_paraphrases": False,
        "intervention": "pragmatic_task_and_answer_space",
        "prompt_conditions": prompt_contract(),
        "decode_mode": "greedy",
        "generation": generation,
        "renderer_contract": RENDER_CONTRACT,
        "renderer_source": str(renderer_source),
        "renderer_source_sha256": sha256_file(renderer_source),
        "runner_sha256": sha256_file(runner_path),
        "formal_clinical_claim_evaluation": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = freeze_config(
        candidate_config, args.output_dir / "generation_config.json", args.resume
    )
    fingerprint = str(config["fingerprint"])
    manifest_path = args.output_dir / "selected_manifest.jsonl"
    if manifest_path.exists():
        if sha256_file(manifest_path) != selected_manifest_sha256:
            raise ValueError("selected manifest differs from frozen config")
    else:
        atomic_write_text(manifest_path, selected_manifest_text)

    shards_dir = args.output_dir / "shards"
    errors_dir = args.output_dir / "errors"
    shards_dir.mkdir(exist_ok=True)
    errors_dir.mkdir(exist_ok=True)
    work = generation_order(items, args.seed)
    completed = 0
    for item, condition in work:
        shard = shards_dir / f"{record_key(item['item_id'], condition['name'])}.json"
        if shard.exists():
            validate_shard(json.loads(shard.read_text()), item["item_id"], condition["name"], fingerprint)
            completed += 1
    print(f"strict resume: {completed}/{len(work)} valid atomic shards", flush=True)

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device=args.device)
    bot.gen_kwargs.update(generation)
    bot.gen_kwargs["eos_token_id"] = bot.tokenizer.eos_token_id
    bot.gen_kwargs["pad_token_id"] = bot.tokenizer.pad_token_id
    try:
        conformance_path = args.output_dir / "generation_conformance.json"
        if conformance_path.exists():
            conformance = json.loads(conformance_path.read_text())
            if conformance.get("fingerprint") != fingerprint or conformance.get("passed") is not True:
                raise ValueError("invalid or failed direct/standard generation conformance")
        else:
            conformance_item = items[0]
            conformance_condition = prompt_contract()[0]
            conformance_image_path = args.image_root / str(
                conformance_item["dicom_relpath"]
            ).removeprefix("train/")
            conformance_image = dicom_to_pil(conformance_image_path)
            conformance_seed = int(
                hashlib.sha256(
                    f"{args.seed}:{conformance_item['item_id']}:conformance".encode()
                ).hexdigest()[:16],
                16,
            ) % (2**31)
            torch.manual_seed(conformance_seed)
            torch.cuda.manual_seed_all(conformance_seed)
            direct = exact_generate(
                bot,
                conformance_condition["prompt"],
                conformance_image,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=1.2,
            )
            torch.manual_seed(conformance_seed)
            torch.cuda.manual_seed_all(conformance_seed)
            standard_response = bot.inference(
                conformance_condition["prompt"], [conformance_image]
            )
            standard_text = str(standard_response[0] if standard_response else "").strip()
            conformance = {
                "version": "huatuo-direct-standard-generation-conformance-v1",
                "fingerprint": fingerprint,
                "item_id": conformance_item["item_id"],
                "prompt_condition": conformance_condition["name"],
                "direct_text": direct["text"],
                "direct_generated_token_ids": direct["generated_token_ids"],
                "direct_generated_token_count": direct["generated_token_count"],
                "standard_inference_text": standard_text,
                "standard_text_retokenized_ids_diagnostic_only": [
                    int(value)
                    for value in bot.tokenizer(
                        standard_text, add_special_tokens=False
                    ).input_ids
                ],
                "passed": bool(direct["text"] and direct["text"] == standard_text),
                "criterion": "nonempty exact decoded-text equality under greedy decoding",
                "created_at": utc_now(),
            }
            atomic_json(conformance_path, conformance)
            if not conformance["passed"]:
                raise RuntimeError("direct generation differs from standard bot.inference")
        for index, (item, condition) in enumerate(work, 1):
            key = record_key(item["item_id"], condition["name"])
            shard = shards_dir / f"{key}.json"
            if shard.exists():
                continue
            started = time.monotonic()
            try:
                image_path = args.image_root / str(item["dicom_relpath"]).removeprefix("train/")
                image = dicom_to_pil(image_path)
                sample_seed = int(
                    hashlib.sha256(
                        f"{args.seed}:{item['item_id']}:{condition['name']}".encode()
                    ).hexdigest()[:16],
                    16,
                ) % (2**31)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                direct = exact_generate(
                    bot,
                    condition["prompt"],
                    image,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=1.2,
                )
                text = str(direct["text"])
                token_ids = list(direct["generated_token_ids"])
                if not text:
                    raise RuntimeError("model returned an empty response")
                record = {
                    "version": VERSION,
                    "item_id": item["item_id"],
                    "image_id": item["image_id"],
                    "dicom_relpath": item["dicom_relpath"],
                    "prompt_condition": condition["name"],
                    "prompt": condition["prompt"],
                    "pragmatic_task": condition["pragmatic_task"],
                    "answer_space_focus": condition["answer_space_focus"],
                    "prompts_are_paraphrases": False,
                    "text": text,
                    "generated_token_count": len(token_ids),
                    "generated_token_ids": token_ids,
                    "visible_answer_token_count": len(
                        bot.tokenizer(text, add_special_tokens=False).input_ids
                    ),
                    "prompt_token_count": direct["prompt_token_count"],
                    "sequence_layout": direct["sequence_layout"],
                    "max_new_tokens": args.max_new_tokens,
                    "hit_max_new_tokens": len(token_ids) >= args.max_new_tokens,
                    "stop_reason": "length" if len(token_ids) >= args.max_new_tokens else "eos_or_template",
                    "empty_response": False,
                    **surface_refusal(text),
                    "claim_names": item["claim_names"],
                    "claim_universe_sha256": item["claim_universe_sha256"],
                    "clinical_claim_evaluation_status": "pending_shared_audit",
                    "ground_truth_used_for_generation_or_selection": False,
                    "automatic_labeler_used": False,
                    "sample_seed": sample_seed,
                    "elapsed_seconds": direct["elapsed_seconds"],
                    "fingerprint": fingerprint,
                    "created_at": utc_now(),
                }
                validate_shard(record, item["item_id"], condition["name"], fingerprint)
                atomic_json(shard, record)
                error_path = errors_dir / f"{key}.json"
                if error_path.exists():
                    error_path.unlink()
                completed += 1
                print(
                    f"[{completed}/{len(work)}] {item['item_id']} {condition['name']} "
                    f"tokens={len(token_ids)} seconds={record['elapsed_seconds']:.2f}",
                    flush=True,
                )
            except Exception as error:
                atomic_json(
                    errors_dir / f"{key}.json",
                    {
                        "version": VERSION,
                        "item_id": item["item_id"],
                        "prompt_condition": condition["name"],
                        "fingerprint": fingerprint,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "created_at": utc_now(),
                    },
                )
                raise
    finally:
        del bot
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records = []
    for item in items:
        for condition in prompt_contract():
            path = shards_dir / f"{record_key(item['item_id'], condition['name'])}.json"
            payload = json.loads(path.read_text())
            validate_shard(payload, item["item_id"], condition["name"], fingerprint)
            records.append(payload)
    records.sort(key=lambda row: (str(row["item_id"]), str(row["prompt_condition"])))
    atomic_write_text(
        args.output_dir / "generations.jsonl",
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    )
    summary = {
        "version": VERSION,
        "status": "generation_complete_clinical_audit_pending",
        "items": len(items),
        "prompt_conditions": 3,
        "generations": len(records),
        "generated_token_count": {
            condition["name"]: {
                "mean": sum(
                    row["generated_token_count"]
                    for row in records
                    if row["prompt_condition"] == condition["name"]
                )
                / len(items),
                "cap_hits": sum(
                    bool(row["hit_max_new_tokens"])
                    for row in records
                    if row["prompt_condition"] == condition["name"]
                ),
                "surface_refusal_matches": sum(
                    bool(row["surface_refusal_match"])
                    for row in records
                    if row["prompt_condition"] == condition["name"]
                ),
            }
            for condition in prompt_contract()
        },
        "clinical_claim_evaluation": "pending_shared_audit",
        "generations_sha256": sha256_file(args.output_dir / "generations.jsonl"),
        "fingerprint": fingerprint,
    }
    atomic_json(args.output_dir / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
