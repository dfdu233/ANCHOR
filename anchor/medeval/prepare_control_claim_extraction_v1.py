#!/usr/bin/env python3
"""Prepare a hash-bound RadGraph input from the frozen K-sample matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.prepare_vqa_rad_internal_control_dev_v1 import atomic_json


VERSION = "control-claim-extraction-input-v1"


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(path)


def prepare(
    *,
    run_root: Path,
    generation_audit: Path,
    aggregation_contract: Path,
    output: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    audit = json.loads(generation_audit.read_text())
    contract = json.loads(aggregation_contract.read_text())
    if not audit.get("passed"):
        raise ValueError("generation audit must pass before claim extraction")
    qids = [str(value) for value in audit["expected_qids"]]
    seeds = [int(value) for value in contract["aggregation"]["seeds"]]
    baseline_arm = str(audit.get("baseline_arm", "greedy256"))
    rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    for model in contract.get("models", audit["models"]):
        if model not in audit["models"] or not audit["models"][model]["passed"]:
            raise ValueError(f"model generation audit did not pass: {model}")
        for seed in seeds:
            source_path = run_root / model / f"sample_t07_p09_seed{seed}" / "answers.jsonl"
            answers = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
            if [str(row["question_id"]) for row in answers] != qids:
                raise ValueError(f"qid mismatch: {source_path}")
            source_files.append({"path": str(source_path.resolve()), "sha256": sha256_file(source_path)})
            for answer in answers:
                metadata = answer.get("metadata") or {}
                nll = metadata.get("mean_token_nll")
                if not isinstance(nll, (int, float)) or not math.isfinite(float(nll)):
                    raise ValueError(f"non-finite NLL: {model}/{seed}/{answer['question_id']}")
                item_id = f"{model}:{answer['question_id']}:{seed}"
                rows.append(
                    {
                        "id": item_id,
                        "report": str(answer["text"]),
                        "source": {
                            "stream": "self_consistency",
                            "model": model,
                            "qid": str(answer["question_id"]),
                            "seed": seed,
                            "mean_token_nll": float(nll),
                            "generated_token_ids": metadata["generated_token_ids"],
                            "stop_reason": metadata.get("stop_reason"),
                            "hit_max_new_tokens": bool(metadata.get("hit_max_new_tokens")),
                        },
                    }
                )
        greedy_path = run_root / model / baseline_arm / "answers.jsonl"
        greedy_answers = [
            json.loads(line) for line in greedy_path.read_text().splitlines() if line.strip()
        ]
        if [str(row["question_id"]) for row in greedy_answers] != qids:
            raise ValueError(f"qid mismatch: {greedy_path}")
        source_files.append({"path": str(greedy_path.resolve()), "sha256": sha256_file(greedy_path)})
        for answer in greedy_answers:
            metadata = answer.get("metadata") or {}
            nll = metadata.get("mean_token_nll")
            if not isinstance(nll, (int, float)) or not math.isfinite(float(nll)):
                raise ValueError(
                    f"non-finite baseline NLL: {model}/{answer['question_id']}"
                )
            rows.append(
                {
                    "id": f"{model}:{answer['question_id']}:{baseline_arm}",
                    "report": str(answer["text"]),
                    "source": {
                        "stream": baseline_arm,
                        "model": model,
                        "qid": str(answer["question_id"]),
                        "seed": int(metadata.get("base_seed", 42)),
                        "mean_token_nll": float(nll),
                        "generated_token_ids": metadata["generated_token_ids"],
                        "stop_reason": metadata.get("stop_reason"),
                        "hit_max_new_tokens": bool(metadata.get("hit_max_new_tokens")),
                    },
                }
            )
    _atomic_jsonl(output, rows)
    result = {
        "protocol_version": VERSION,
        "evidence_scope": "prediction-side structure only; no reference truth",
        "run_root": str(run_root.resolve()),
        "generation_audit": str(generation_audit.resolve()),
        "generation_audit_sha256": sha256_file(generation_audit),
        "aggregation_contract": str(aggregation_contract.resolve()),
        "aggregation_contract_sha256": sha256_file(aggregation_contract),
        "models": sorted(audit["models"]),
        "qids": qids,
        "seeds": seeds,
        "baseline_arm": baseline_arm,
        "expected_reports": len(audit["models"]) * len(qids) * (len(seeds) + 1),
        "reports": len(rows),
        "source_files": source_files,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "test_labels_used": False,
    }
    result["fingerprint"] = sha256_json(result)
    atomic_json(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--generation-audit", required=True, type=Path)
    parser.add_argument("--aggregation-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(
        run_root=args.run_root,
        generation_audit=args.generation_audit,
        aggregation_contract=args.aggregation_contract,
        output=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
