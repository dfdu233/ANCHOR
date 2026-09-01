"""Fail-closed token-exact audit for trained LLaVA T2 v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VARIANTS = (
    "base",
    "ha-dpo",
    "opa-dpo",
    "da-dpo",
    "sentinel",
    "less-is-more",
    "factmm-rag-generator",
)
ADAPTERS = set(VARIANTS) - {"base", "factmm-rag-generator"}
EXPECTED_CHECKPOINTS = {
    "base": Path("/home/dbw/models/llava-v1.5-7b"),
    "ha-dpo": Path("/home/dbw/models/hadpo-llava-1.5"),
    "opa-dpo": Path("/home/dbw/models/opadpo-lora-llava-v1.5-7b"),
    "da-dpo": Path("/home/dbw/models/da-dpo-llava-v1.5-7b"),
    "sentinel": Path("/home/dbw/models/sentinel-llava-v1.5-7b"),
    "less-is-more": Path("/home/dbw/models/less-is-more-llava-v1.5-7b"),
    "factmm-rag-generator": Path("/home/dbw/models/factmm-rag-generator-v1"),
}
HA_LLAVA = Path(
    "/home/dbw/ANCHOR/third_party/training_baselines/HA-DPO/ha_dpo/models/llava-v1_5"
)
EXPECTED_LLAVA_ROOTS = {
    "base": HA_LLAVA,
    "ha-dpo": HA_LLAVA,
    "opa-dpo": Path(
        "/home/dbw/ANCHOR/third_party/training_baselines/OPA-DPO/llava_setup/LLaVA"
    ),
    "da-dpo": Path("/home/dbw/ANCHOR/third_party/training_baselines/DA-DPO"),
    "sentinel": Path(
        "/home/dbw/ANCHOR/third_party/MedHEval/code/baselines/Med-LVLMs/llava_1.6/LLaVA"
    ),
    "less-is-more": HA_LLAVA,
    "factmm-rag-generator": HA_LLAVA,
}
EXPECTED_TRANSFORMERS_PATH = Path(
    "/home/dbw/.venvs/llava15-official-431/lib/python3.10/site-packages/transformers/__init__.py"
).resolve()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical(ids: list[int]) -> list[int]:
    result = list(ids)
    while result and result[-1] == 2:
        result.pop()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("n1", "n32"))
    parser.add_argument("--expected", required=True, type=int)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prerequisite-audit", type=Path)
    return parser.parse_args()


def audit_variant(
    root: Path, stage: str, expected: int, variant: str,
    t0_row: dict, t0: dict, t0_path: Path,
):
    unified_dir = root / stage / "unified" / variant
    official_dir = root / stage / "official" / variant
    paths = {
        "unified_answers": unified_dir / "answers.jsonl",
        "unified_config": unified_dir / "generation_config.json",
        "unified_ledger": unified_dir / "loading_ledger.json",
        "official_answers": official_dir / "answers.jsonl",
        "official_evidence": official_dir / "evidence.json",
    }
    checks: dict[str, bool] = {f"exists_{name}": path.is_file() for name, path in paths.items()}
    row = {"variant": variant, "checks": checks, "paths": {k: str(v) for k, v in paths.items()}}
    if not all(checks.values()):
        row["passed"] = False
        return row
    unified = load_jsonl(paths["unified_answers"])
    official = load_jsonl(paths["official_answers"])
    config = load_json(paths["unified_config"])
    ledger = load_json(paths["unified_ledger"])
    evidence = load_json(paths["official_evidence"])
    evidence_inputs = {str(item["question_id"]): item for item in evidence.get("inputs", [])}
    runner_path = Path("anchor/corrected_sgta/run_trained_llava_baseline_v1.py")
    official_entry = Path(evidence.get("official_entry", "__missing__"))
    loading_source = evidence.get("loading_source", {})
    builder_source = Path(loading_source.get("builder_source", "__missing__"))
    loaded_model_source = Path(
        loading_source.get("loaded_model_source", "__missing__")
    )
    expected_llava_root = EXPECTED_LLAVA_ROOTS[variant].resolve()
    official_llava_path = Path(
        evidence.get("environment", {}).get("llava_path", "__missing__")
    ).resolve()
    fingerprint_payload = dict(config)
    recorded_fingerprint = fingerprint_payload.pop("fingerprint", None)
    recomputed_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    ledger_sha256 = digest(paths["unified_ledger"])
    unified_qids = [str(item["question_id"]) for item in unified]
    official_qids = [str(item["question_id"]) for item in official]
    checks.update(
        {
            "expected_cardinality": len(unified) == len(official) == expected,
            "qid_order_exact": unified_qids == official_qids,
            "evidence_cardinality": len(evidence_inputs) == expected,
            "unified_transformers_431": config.get("runtime", {}).get("transformers") == "4.31.0",
            "official_transformers_431": evidence.get("environment", {}).get("transformers") == "4.31.0",
            "unified_transformers_path_exact": Path(config.get("runtime", {}).get("transformers_path", "__missing__")).resolve() == EXPECTED_TRANSFORMERS_PATH,
            "official_transformers_path_exact": Path(evidence.get("environment", {}).get("transformers_path", "__missing__")).resolve() == EXPECTED_TRANSFORMERS_PATH,
            "variant_bound": config.get("variant") == variant and ledger.get("variant") == variant,
            "generation_fingerprint_valid": recorded_fingerprint == recomputed_fingerprint,
            "runner_hash_current": config.get("runner_sha256") == digest(runner_path),
            "unified_llava_source_exact": Path(config.get("llava_source_root", "__missing__")).resolve() == expected_llava_root,
            "official_llava_source_exact": official_llava_path.is_relative_to(expected_llava_root),
            "t0_runner_hash_current": t0.get("runner_sha256") == digest(runner_path),
            "t0_audit_hash_bound": config.get("checkpoint_t0_audit_sha256") == digest(t0_path),
            "checkpoint_fingerprint_bound": config.get("checkpoint_fingerprint") == t0_row.get("checkpoint_fingerprint"),
            "checkpoint_path_exact": Path(
                evidence.get("loading_ledger", {}).get(
                    "adapter_path" if variant == "opa-dpo" else "model_path", "__missing__"
                )
            ).resolve() == EXPECTED_CHECKPOINTS[variant].resolve(),
            "official_answers_hash_bound": evidence.get("answers_sha256") == digest(paths["official_answers"]),
            "official_entry_hash_bound": official_entry.is_file() and evidence.get("official_entry_sha256") == digest(official_entry),
            "official_entry_independent_from_unified": official_entry.resolve() != runner_path.resolve(),
            "official_builder_source_hash_bound": builder_source.is_file() and loading_source.get("builder_source_sha256") == digest(builder_source),
            "official_model_source_hash_bound": loaded_model_source.is_file() and loading_source.get("loaded_model_source_sha256") == digest(loaded_model_source),
            "loading_ledger_hash_bound": all(
                item.get("metadata", {}).get("loading_ledger_sha256") == ledger_sha256
                for item in unified
            ),
            "generation_fingerprint_bound": all(
                item.get("metadata", {}).get("fingerprint") == recorded_fingerprint
                for item in unified
            ),
            "no_empty_unified": all(item.get("text", "").strip() for item in unified),
            "no_input_unavailable": all(item.get("metadata", {}).get("stop_reason") != "input_unavailable" for item in unified),
        }
    )
    prompt_exact = True
    pixels_exact = True
    raw_tokens_exact = True
    content_tokens_exact = True
    text_exact = True
    for left, right in zip(unified, official):
        qid = str(left["question_id"])
        observed_input = evidence_inputs.get(qid, {})
        prompt_exact &= left.get("metadata", {}).get("prompt_token_ids_sha256") == observed_input.get("prompt_token_ids_sha256")
        pixels_exact &= left.get("metadata", {}).get("preprocessed_pixel_tensor_sha256") == observed_input.get("pixel_tensor_sha256")
        left_raw = left.get("metadata", {}).get("raw_generated_token_ids", [])
        right_raw = observed_input.get(
            "generated_token_ids",
            right.get("metadata", {}).get("generated_token_ids", []),
        )
        raw_tokens_exact &= left_raw == right_raw
        content_tokens_exact &= canonical(left_raw) == canonical(right_raw)
        text_exact &= left.get("text", "").strip() == right.get("text", "").strip()
    checks.update(
        {
            "prompt_token_ids_exact": prompt_exact,
            "preprocessed_pixels_exact": pixels_exact,
            "raw_generated_tokens_exact": raw_tokens_exact,
            "content_generated_tokens_exact": content_tokens_exact,
            "decoded_text_exact": text_exact,
        }
    )
    peft = evidence.get("peft_evidence", {})
    if variant in ADAPTERS:
        checks.update(
            {
                "unified_adapter_active": ledger.get("adapter_active") is True,
                "unified_adapter_delta_nonzero": ledger.get("adapter_sampled_nonzero_delta_count", 0) > 0,
                "official_adapter_loaded": peft.get("called") is True,
                "official_adapter_delta_nonzero": peft.get("sampled_nonzero_delta_count", 0) > 0,
                "official_adapter_path_exact": Path(peft.get("checkpoint", "__missing__")).resolve() == EXPECTED_CHECKPOINTS[variant].resolve(),
            }
        )
        if variant != "opa-dpo":
            unified_non_lora = ledger
            official_non_lora = evidence.get("loading_ledger", {}).get("non_lora", {})
            checks.update(
                {
                    "unified_non_lora_exact": (
                        unified_non_lora.get("non_lora_checkpoint_keys", 0) > 0
                        and unified_non_lora.get("non_lora_matched_keys") == unified_non_lora.get("non_lora_checkpoint_keys")
                        and not unified_non_lora.get("non_lora_shape_mismatches")
                        and not unified_non_lora.get("non_lora_unexpected_keys")
                        and not unified_non_lora.get("non_lora_value_mismatches")
                    ),
                    "official_non_lora_exact": (
                        official_non_lora.get("checkpoint_keys", 0) > 0
                        and official_non_lora.get("matched_keys") == official_non_lora.get("checkpoint_keys")
                        and not official_non_lora.get("shape_mismatches")
                        and not official_non_lora.get("value_mismatches")
                    ),
                }
            )
        if variant == "sentinel":
            contract = evidence.get("release_loading_contract") or {}
            readme = Path(contract.get("readme", "__missing__"))
            released_config = Path(contract.get("config", "__missing__"))
            released_adapter = Path(contract.get("adapter_config", "__missing__"))
            citations = contract.get("instruction_lines", [])
            required_release_phrases = (
                "library_name: transformers",
                "This model is a PEFT (LoRA) adapter.",
                "You first need to load the base model",
                "Please follow the official repo of [LLaVA]",
            )
            current_lines = readme.read_text().splitlines() if readme.is_file() else []
            citations_exact = bool(citations) and all(
                isinstance(item.get("line"), int)
                and 1 <= item["line"] <= len(current_lines)
                and current_lines[item["line"] - 1].strip() == item.get("text")
                for item in citations
            ) and all(
                any(phrase in item.get("text", "") for item in citations)
                for phrase in required_release_phrases
            )
            checks.update(
                {
                    "sentinel_release_readme_hash_bound": readme.is_file() and contract.get("readme_sha256") == digest(readme),
                    "sentinel_release_instruction_lines_bound": citations_exact,
                    "sentinel_release_config_hash_bound": released_config.is_file() and contract.get("config_sha256") == digest(released_config),
                    "sentinel_release_adapter_config_hash_bound": released_adapter.is_file() and contract.get("adapter_config_sha256") == digest(released_adapter),
                    "sentinel_release_fields_exact": (
                        contract.get("config_model_type") == "llava_llama"
                        and contract.get("config_transformers_version") == "4.48.0"
                        and contract.get("adapter_peft_type") == "LORA"
                        and contract.get("adapter_base_model_name_or_path") == "liuhaotian/llava-v1.5-7b"
                    ),
                    "sentinel_selected_loader_root_exact": Path(contract.get("selected_standard_llava_root", "__missing__")).resolve() == EXPECTED_LLAVA_ROOTS["sentinel"].resolve(),
                }
            )
    else:
        checks["official_no_adapter"] = peft.get("called") is False
        checks["checkpoint_role_exact"] = config.get("checkpoint_role") == (
            "base" if variant == "base" else "full_generator_no_retrieval"
        )
    row.update(
        {
            "n_unified": len(unified),
            "n_official": len(official),
            "unified_sha256": digest(paths["unified_answers"]),
            "official_sha256": digest(paths["official_answers"]),
            "release_loading_contract": evidence.get("release_loading_contract"),
            "failed_checks": sorted(key for key, value in checks.items() if not value),
        }
    )
    row["passed"] = all(checks.values())
    return row


def main() -> None:
    args = parse_args()
    invalid = sorted(set(args.variants) - set(VARIANTS))
    if invalid:
        raise ValueError(f"unknown variants: {invalid}")
    prerequisite = None
    if args.prerequisite_audit:
        prerequisite = load_json(args.prerequisite_audit)
        admitted = set(prerequisite.get("passed_variants", []))
        missing = sorted(set(args.variants) - admitted)
        if missing:
            raise RuntimeError(f"n32 variants did not pass n1: {missing}")
    t0_path = Path("corrected_runs/paper_baselines_v1/trained_llava_t0_v1.json")
    t0 = load_json(t0_path)
    t0_rows = {row["method"]: row for row in t0["methods"]}
    rows = [
        audit_variant(
            args.root, args.stage, args.expected, variant,
            t0_rows[variant], t0, t0_path,
        )
        for variant in args.variants
    ]
    result = {
        "protocol": "trained-llava-token-exact-t2-v3",
        "stage": args.stage,
        "expected": args.expected,
        "t0_audit": str(t0_path.resolve()),
        "t0_audit_sha256": digest(t0_path),
        "prerequisite_audit": str(args.prerequisite_audit.resolve()) if args.prerequisite_audit else None,
        "prerequisite_audit_sha256": digest(args.prerequisite_audit) if args.prerequisite_audit else None,
        "rows": rows,
        "passed_variants": [row["variant"] for row in rows if row["passed"]],
        "failed_variants": [row["variant"] for row in rows if not row["passed"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["failed_variants"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
