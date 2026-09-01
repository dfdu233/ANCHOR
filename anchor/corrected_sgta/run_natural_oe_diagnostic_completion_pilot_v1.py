#!/usr/bin/env python3
"""Crash-safe Huatuo generation for the bounded natural-OE completion pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from corrected_sgta.compile_natural_oe_diagnostic_completion_pilot_v1 import PROMPT


VERSION = "natural-oe-diagnostic-completion-generation-v1"
DEFAULT_MODEL = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO = Path("/home/dbw/HuatuoGPT-Vision")
FORBIDDEN_GENERATION_FIELDS = {
    "edge_id",
    "parent_label",
    "child_label",
    "parent_votes",
    "child_votes",
    "design_stratum",
}
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def surface_refusal(text: str) -> dict[str, Any]:
    normalized = " ".join(text.lower().split())
    matches = [phrase for phrase in REFUSAL_SURFACE_PHRASES if phrase in normalized]
    return {
        "surface_refusal_match": bool(matches),
        "surface_refusal_phrases": matches,
        "interpretation": "conservative literal-phrase diagnostic only",
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def record_key(item_id: str) -> str:
    return hashlib.sha256(f"{VERSION}\0{item_id}".encode()).hexdigest()


def validate_inputs(
    contract_path: Path,
    authorization_path: Path,
    *,
    limit: int | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if contract.get("version") != "natural-oe-diagnostic-completion-pilot-v1":
        raise ValueError("wrong pilot contract version")
    if contract.get("prompt") != PROMPT:
        raise ValueError("natural-OE prompt drift")
    if contract.get("generation_receives_reader_labels") is not False:
        raise ValueError("generation must not receive reader labels")
    if contract.get("generation_receives_target_edge") is not False:
        raise ValueError("generation must not receive target edges")
    if authorization.get("version") != "natural-oe-pilot-launch-authorization-v1":
        raise ValueError("wrong launch authorization version")
    if authorization.get("generation_authorized") is not True:
        raise PermissionError("generation is not authorized")
    if authorization.get("pilot_contract_sha256") != sha256_file(contract_path):
        raise ValueError("authorization does not bind this pilot contract")
    runner_path = Path(__file__).resolve()
    if authorization.get("runner_sha256") != sha256_file(runner_path):
        raise ValueError("authorization does not bind this runner source")
    progression_path = Path(str(authorization.get("progression_gate", "")))
    if not progression_path.is_file():
        raise FileNotFoundError("launch authorization progression gate is missing")
    if authorization.get("progression_gate_sha256") != sha256_file(progression_path):
        raise ValueError("launch authorization progression-gate hash drift")
    progression = json.loads(progression_path.read_text(encoding="utf-8"))
    if progression.get("allowed_next_stage", {}).get("name") != (
        "natural_oe_bounded_construct_pilot"
    ):
        raise PermissionError("progression gate does not permit this pilot class")

    manifest_path = Path(str(contract["generation_manifest"]))
    if sha256_file(manifest_path) != contract.get("generation_manifest_sha256"):
        raise ValueError("generation manifest hash drift")
    rows = load_jsonl(manifest_path)
    if len(rows) != int(contract["images"]):
        raise ValueError("manifest row count differs from pilot contract")
    for row in rows:
        leaked = FORBIDDEN_GENERATION_FIELDS & set(row)
        if leaked:
            raise ValueError(f"reader design leaked into generation manifest: {sorted(leaked)}")
        if row.get("generation_receives_reader_labels") is not False:
            raise ValueError("row does not attest reader-label blindness at generation")
        if row.get("generation_receives_target_edge") is not False:
            raise ValueError("row does not attest target-edge blindness at generation")
        if not Path(str(row["dicom_path"])).is_file():
            raise FileNotFoundError(row["dicom_path"])
    maximum = int(authorization["maximum_images"])
    if maximum > int(progression["allowed_next_stage"]["maximum_images"]):
        raise PermissionError("authorization exceeds the progression-gate ceiling")
    if authorization.get("full_pilot_authorized") is True:
        canary_path = Path(str(authorization.get("canary_audit", "")))
        if not canary_path.is_file():
            raise FileNotFoundError("full pilot requires a canary audit")
        if authorization.get("canary_audit_sha256") != sha256_file(canary_path):
            raise ValueError("full-pilot authorization canary-audit hash drift")
        canary = json.loads(canary_path.read_text(encoding="utf-8"))
        if canary.get("promotion_gate_passed") is not True:
            raise PermissionError("canary did not pass the frozen promotion gate")
        if canary.get("generations_sha256") != authorization.get(
            "canary_generations_sha256"
        ):
            raise ValueError("canary generation hash mismatch")
    requested = len(rows) if limit is None else limit
    if requested <= 0 or requested > len(rows):
        raise ValueError("limit must select between 1 and all manifest rows")
    if requested > maximum:
        raise PermissionError(
            f"requested {requested} images exceeds authorization maximum {maximum}"
        )
    return contract, authorization, rows[:requested]


def validate_shard(
    row: Mapping[str, Any], *, item_id: str, fingerprint: str
) -> None:
    if row.get("version") != VERSION or row.get("item_id") != item_id:
        raise ValueError(f"shard identity/version mismatch for {item_id}")
    if row.get("fingerprint") != fingerprint:
        raise ValueError(f"shard fingerprint mismatch for {item_id}")
    if row.get("clinical_claim_evaluation_status") != "pending_physician_construct_audit":
        raise ValueError("generation runner cannot assign clinical truth")
    if row.get("reader_labels_available_to_generation") is not False:
        raise ValueError("generation shard lacks reader-label isolation attestation")
    token_ids = row.get("generated_token_ids")
    if not isinstance(token_ids, list) or len(token_ids) != row.get("generated_token_count"):
        raise ValueError(f"invalid generated-token accounting for {item_id}")
    if not str(row.get("text", "")).strip():
        raise ValueError(f"empty generation for {item_id}")


def freeze_config(path: Path, candidate: dict[str, Any], resume: bool) -> dict[str, Any]:
    immutable = {key: value for key, value in candidate.items() if key not in {"created_at", "command"}}
    candidate["fingerprint"] = canonical_json_sha256(immutable)
    if not resume:
        if path.exists():
            raise FileExistsError(f"output already configured; use --resume: {path}")
        atomic_json(path, candidate)
        return candidate
    if not path.is_file():
        raise FileNotFoundError("--resume requires generation_config.json")
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("fingerprint") != candidate["fingerprint"]:
        raise ValueError("refusing incompatible resume")
    return existing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-contract", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    import torch

    from corrected_sgta.run_clinical_presupposition_generation_v1 import exact_generate
    from corrected_sgta.run_huatuo_dicom_render_pilot_v1 import (
        model_artifact_fingerprint,
    )
    from corrected_sgta.run_huatuo_vindr_commitment_probe import (
        dicom_to_pil,
        import_huatuo,
    )

    args = parse_args()
    contract, authorization, items = validate_inputs(
        args.pilot_contract, args.authorization, limit=args.limit
    )
    generation = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": 1,
        "repetition_penalty": 1.2,
    }
    runner = Path(__file__).resolve()
    renderer = Path(sys.modules[dicom_to_pil.__module__].__file__).resolve()
    candidate = {
        "version": VERSION,
        "created_at": utc_now(),
        "command": sys.argv,
        "model_id": "huatuo",
        "model_dir": str(args.model_dir.resolve()),
        "model_artifact_fingerprint": model_artifact_fingerprint(args.model_dir),
        "huatuo_root": str(args.huatuo_root.resolve()),
        "prompt": PROMPT,
        "pilot_contract": str(args.pilot_contract.resolve()),
        "pilot_contract_sha256": sha256_file(args.pilot_contract),
        "authorization": str(args.authorization.resolve()),
        "authorization_sha256": sha256_file(args.authorization),
        "manifest_sha256": contract["generation_manifest_sha256"],
        "selected_item_ids": [row["item_id"] for row in items],
        "selection_uses_reader_labels": True,
        "reader_labels_available_to_generation": False,
        "target_edge_available_to_generation": False,
        "clinical_truth_available_to_generation": False,
        "decode_mode": "greedy",
        "generation": generation,
        "runner_sha256": sha256_file(runner),
        "renderer_source_sha256": sha256_file(renderer),
        "formal_clinical_claim_evaluation": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = freeze_config(
        args.output_dir / "generation_config.json", candidate, args.resume
    )
    fingerprint = str(config["fingerprint"])
    selected_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in items)
    selected_path = args.output_dir / "selected_generation_manifest.jsonl"
    if selected_path.exists():
        if selected_path.read_text(encoding="utf-8") != selected_text:
            raise ValueError("selected manifest drift")
    else:
        atomic_write_text(selected_path, selected_text)

    shards = args.output_dir / "shards"
    errors = args.output_dir / "errors"
    shards.mkdir(exist_ok=True)
    errors.mkdir(exist_ok=True)
    completed = 0
    for item in items:
        path = shards / f"{record_key(str(item['item_id']))}.json"
        if path.exists():
            validate_shard(
                json.loads(path.read_text()),
                item_id=str(item["item_id"]),
                fingerprint=fingerprint,
            )
            completed += 1
    print(f"strict resume: {completed}/{len(items)} valid shards", flush=True)

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device=args.device)
    bot.gen_kwargs.update(generation)
    bot.gen_kwargs["eos_token_id"] = bot.tokenizer.eos_token_id
    bot.gen_kwargs["pad_token_id"] = bot.tokenizer.pad_token_id
    try:
        for item in items:
            item_id = str(item["item_id"])
            key = record_key(item_id)
            shard = shards / f"{key}.json"
            if shard.exists():
                continue
            try:
                image = dicom_to_pil(Path(str(item["dicom_path"])))
                sample_seed = int(
                    hashlib.sha256(f"{args.seed}:{item_id}".encode()).hexdigest()[:16],
                    16,
                ) % (2**31)
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                generated = exact_generate(
                    bot,
                    PROMPT,
                    image,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=1.2,
                )
                text = str(generated["text"]).strip()
                token_ids = list(generated["generated_token_ids"])
                record = {
                    "version": VERSION,
                    "item_id": item_id,
                    "image_id": item["image_id"],
                    "prompt_id": item["prompt_id"],
                    "prompt": PROMPT,
                    "text": text,
                    "generated_token_ids": token_ids,
                    "generated_token_count": len(token_ids),
                    "visible_answer_token_count": len(
                        bot.tokenizer(text, add_special_tokens=False).input_ids
                    ),
                    "prompt_token_count": generated["prompt_token_count"],
                    "max_new_tokens": args.max_new_tokens,
                    "hit_max_new_tokens": len(token_ids) >= args.max_new_tokens,
                    "stop_reason": "length" if len(token_ids) >= args.max_new_tokens else "eos_or_template",
                    **surface_refusal(text),
                    "reader_labels_available_to_generation": False,
                    "target_edge_available_to_generation": False,
                    "clinical_claim_evaluation_status": "pending_physician_construct_audit",
                    "automatic_clinical_labeler_used": False,
                    "sample_seed": sample_seed,
                    "elapsed_seconds": generated["elapsed_seconds"],
                    "fingerprint": fingerprint,
                    "created_at": utc_now(),
                }
                validate_shard(record, item_id=item_id, fingerprint=fingerprint)
                atomic_json(shard, record)
                error_path = errors / f"{key}.json"
                if error_path.exists():
                    error_path.unlink()
                completed += 1
                print(
                    f"[{completed}/{len(items)}] {item_id} tokens={len(token_ids)} "
                    f"seconds={record['elapsed_seconds']:.2f}",
                    flush=True,
                )
            except Exception as error:
                atomic_json(
                    errors / f"{key}.json",
                    {
                        "version": VERSION,
                        "item_id": item_id,
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
        torch.cuda.empty_cache()

    records = []
    for item in items:
        payload = json.loads(
            (shards / f"{record_key(str(item['item_id']))}.json").read_text()
        )
        validate_shard(
            payload, item_id=str(item["item_id"]), fingerprint=fingerprint
        )
        records.append(payload)
    records.sort(key=lambda row: str(row["item_id"]))
    aggregate = args.output_dir / "generations.jsonl"
    atomic_write_text(
        aggregate,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
    )
    summary = {
        "version": VERSION,
        "status": "generation_complete_construct_audit_pending",
        "images": len(records),
        "nonempty": sum(bool(str(row["text"]).strip()) for row in records),
        "cap_hits": sum(bool(row["hit_max_new_tokens"]) for row in records),
        "surface_refusal_matches": sum(
            bool(row["surface_refusal_match"]) for row in records
        ),
        "generations_sha256": sha256_file(aggregate),
        "fingerprint": fingerprint,
        "clinical_claim_evaluation": "pending_physician_construct_audit",
    }
    atomic_json(args.output_dir / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
