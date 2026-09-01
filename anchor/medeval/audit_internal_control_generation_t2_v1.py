#!/usr/bin/env python3
"""Audit common-protocol T2 generation arms before clinical scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.prepare_vqa_rad_internal_control_dev_v1 import atomic_json
from anchor.medeval.run_native_oe_vqa import stable_seed


VERSION = "internal-control-generation-t2-audit-v1"
MODELS = ("huatuo", "hulu")
K_SEEDS = (42, 1042, 2042, 3042, 4042)
FIXED_ARMS = (
    "greedy128",
    "greedy256",
    "sample_t02_p09_seed42",
    "sample_t10_p09_seed42",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _qid(row: dict[str, Any]) -> str:
    return str(row.get("question_id", row.get("qid", row.get("id", ""))))


def _audit_arm(path: Path, expected_qids: list[str], model: str) -> dict[str, Any]:
    config_path = path / "generation_config.json"
    answers_path = path / "answers.jsonl"
    errors: list[str] = []
    if not config_path.exists() or not answers_path.exists():
        return {"path": str(path.resolve()), "passed": False, "errors": ["missing config or answers"]}
    config = json.loads(config_path.read_text())
    rows = _load_jsonl(answers_path)
    observed = [_qid(row) for row in rows]
    if observed != expected_qids:
        errors.append("answers are not the exact frozen qid sequence")
    if config.get("model") != model:
        errors.append("model mismatch")
    if config.get("manifest_sha256") is None:
        errors.append("manifest hash missing")
    if len(observed) != len(set(observed)):
        errors.append("duplicate qids")
    finite_nll = 0
    token_trace = 0
    stop_cap_trace = 0
    nonempty = 0
    for row in rows:
        if str(row.get("text", "")).strip():
            nonempty += 1
        metadata = row.get("metadata") or {}
        ids = metadata.get("generated_token_ids")
        if isinstance(ids, list) and ids and all(isinstance(value, int) for value in ids):
            token_trace += 1
        nll = metadata.get("mean_token_nll")
        if isinstance(nll, (int, float)) and math.isfinite(float(nll)):
            finite_nll += 1
        expected_seed = stable_seed(int(config.get("seed", -1)), _qid(row))
        if metadata.get("base_seed") != config.get("seed"):
            errors.append(f"base seed missing or wrong for {_qid(row)}")
        if metadata.get("sample_seed") != expected_seed:
            errors.append(f"sample seed missing or wrong for {_qid(row)}")
        hit_cap = metadata.get("hit_max_new_tokens")
        stop_reason = metadata.get("stop_reason")
        token_count = metadata.get("generated_token_count")
        if (
            isinstance(hit_cap, bool)
            and stop_reason in {"length", "eos_or_template"}
            and isinstance(token_count, int)
            and isinstance(ids, list)
            and token_count == len(ids)
            and (stop_reason == "length") == hit_cap
        ):
            stop_cap_trace += 1
    if nonempty != len(expected_qids):
        errors.append("empty generation")
    if token_trace != len(expected_qids):
        errors.append("actual generated-token trace incomplete")
    if finite_nll != len(expected_qids):
        errors.append("processed-token NLL incomplete")
    if stop_cap_trace != len(expected_qids):
        errors.append("stop/cap/token-count provenance incomplete or inconsistent")
    return {
        "path": str(path.resolve()),
        "generation_config_sha256": sha256_file(config_path),
        "answers_sha256": sha256_file(answers_path),
        "seed": config.get("seed"),
        "generation": config.get("generation"),
        "rows": len(rows),
        "nonempty_rows": nonempty,
        "token_trace_rows": token_trace,
        "finite_nll_rows": finite_nll,
        "stop_cap_trace_rows": stop_cap_trace,
        "passed": not errors,
        "errors": errors,
    }


def audit(
    *,
    run_root: Path,
    pilot_manifest: Path,
    freeze_provenance: Path,
    execution_contract: Path,
    limit: int,
    stage: str = "T2",
) -> dict[str, Any]:
    manifest_rows = json.loads(pilot_manifest.read_text())
    selected = manifest_rows[:limit] if limit else manifest_rows
    expected_qids = [str(row["qid"]) for row in selected]
    freeze = json.loads(freeze_provenance.read_text())
    contract = json.loads(execution_contract.read_text())
    errors: list[str] = []
    if freeze.get("pilot_manifest_sha256") != sha256_file(pilot_manifest):
        errors.append("pilot manifest differs from frozen provenance")
    if freeze.get("execution_contract_sha256") != sha256_file(execution_contract):
        errors.append("execution contract differs from frozen provenance")
    if freeze.get("source_test_image_overlap") != 0:
        errors.append("development/test image leakage")
    if len({row["image_sha256"] for row in selected}) != len(selected):
        errors.append("pilot rows are not image independent")

    model_results: dict[str, Any] = {}
    for model in MODELS:
        arms: dict[str, Any] = {}
        for arm in FIXED_ARMS:
            arms[arm] = _audit_arm(run_root / model / arm, expected_qids, model)
        for seed in K_SEEDS:
            arm = f"sample_t07_p09_seed{seed}"
            arms[arm] = _audit_arm(run_root / model / arm, expected_qids, model)
        replay_name = "replay_t07_p09_seed42"
        arms[replay_name] = _audit_arm(run_root / model / replay_name, expected_qids, model)

        k_rows = {
            seed: _load_jsonl(run_root / model / f"sample_t07_p09_seed{seed}" / "answers.jsonl")
            if arms[f"sample_t07_p09_seed{seed}"]["passed"]
            else []
            for seed in K_SEEDS
        }
        diverse_qids = 0
        if all(len(rows) == len(expected_qids) for rows in k_rows.values()):
            for index in range(len(expected_qids)):
                traces = {
                    tuple(k_rows[seed][index]["metadata"]["generated_token_ids"])
                    for seed in K_SEEDS
                }
                diverse_qids += int(len(traces) > 1)
        original = k_rows[42]
        replay = (
            _load_jsonl(run_root / model / replay_name / "answers.jsonl")
            if arms[replay_name]["passed"]
            else []
        )
        replay_exact = bool(original) and len(original) == len(replay) and all(
            left["text"] == right["text"]
            and left["metadata"]["generated_token_ids"]
            == right["metadata"]["generated_token_ids"]
            for left, right in zip(original, replay)
        )
        model_errors = [
            f"{name}: {message}"
            for name, result in arms.items()
            for message in result["errors"]
        ]
        if diverse_qids == 0:
            model_errors.append("all five sampling streams are degenerate")
        if not replay_exact:
            model_errors.append("same-seed deterministic replay differs")
        model_results[model] = {
            "arms": arms,
            "sampling_k": len(K_SEEDS),
            "sampling_seed_ledger": list(K_SEEDS),
            "diverse_qids": diverse_qids,
            "sampling_non_degenerate": diverse_qids > 0,
            "deterministic_replay_exact": replay_exact,
            "passed": not model_errors,
            "errors": model_errors,
        }
        errors.extend(f"{model}: {message}" for message in model_errors)

    if stage not in {"T2", "T3"}:
        raise ValueError("stage must be T2 or T3")
    result = {
        "protocol_version": VERSION if stage == "T2" else "internal-control-generation-t3-audit-v1",
        "stage": stage,
        "evidence_scope": (
            "development-only generation and trace qualification; no efficacy claim"
            if stage == "T2"
            else "held-out generation and trace qualification; clinical efficacy pending"
        ),
        "run_root": str(run_root.resolve()),
        "pilot_manifest_sha256": sha256_file(pilot_manifest),
        "freeze_provenance_sha256": sha256_file(freeze_provenance),
        "execution_contract_sha256": sha256_file(execution_contract),
        "held_out_manifest_sha256": freeze["held_out_manifest_sha256"],
        "development_and_held_out_hashes_distinct": (
            freeze["development_manifest_sha256"] != freeze["held_out_manifest_sha256"]
        ),
        "test_labels_used_for_selection": False,
        "limit": limit,
        "expected_qids": expected_qids,
        "models": model_results,
        "contract_models": contract["models"],
        "passed": not errors,
        "errors": errors,
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--pilot-manifest", required=True, type=Path)
    parser.add_argument("--freeze-provenance", required=True, type=Path)
    parser.add_argument("--execution-contract", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=("T2", "T3"), default="T2")
    args = parser.parse_args()
    result = audit(
        run_root=args.run_root,
        pilot_manifest=args.pilot_manifest,
        freeze_provenance=args.freeze_provenance,
        execution_contract=args.execution_contract,
        limit=args.limit,
        stage=args.stage,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
