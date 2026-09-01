"""Extend the evaluation supplement with role-correct FactMM-RAG evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor.medeval.build_evaluation_progress_supplement_v2 import build as build_v2
from anchor.medeval.hashing import sha256_json
from anchor.medeval.store import atomic_write_json


def build(**kwargs) -> dict:
    rag = json.loads(Path(kwargs["rag_audit"]).read_text())
    if rag.get("protocol_version") != "rag-dual-track-qualification-v2":
        raise ValueError("role-correct RAG dual-track v2 is required")
    if rag.get("factmm_role_confusion_closed") is not True:
        raise ValueError("FactMM-RAG asset-role confusion remains open")
    result = build_v2(**kwargs)
    result["version"] = "iclr-evaluation-progress-supplement-v3"
    result["evaluation_state"]["factmm_released_asset_role"] = rag.get(
        "factmm_released_asset_role"
    )
    result["evaluation_state"]["factmm_role_confusion_closed"] = True
    result["interpretation"] = (
        "Evaluation engineering advanced and the official FactMM-RAG archive is now correctly "
        "classified as a generator checkpoint, not a retriever. Generation qualification and "
        "asset availability still do not establish clinical efficacy; the paper remains fail-closed."
    )
    result.pop("fingerprint", None)
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--rag-audit", type=Path, required=True)
    parser.add_argument("--v1-failure", type=Path, required=True)
    parser.add_argument("--v2-execution", type=Path, required=True)
    parser.add_argument("--v2-provenance", type=Path, required=True)
    parser.add_argument("--v1-job-state", type=Path, required=True)
    parser.add_argument("--v2-job-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        base_audit=args.base_audit,
        rag_audit=args.rag_audit,
        v1_failure=args.v1_failure,
        v2_execution=args.v2_execution,
        v2_provenance=args.v2_provenance,
        v1_job_state=args.v1_job_state,
        v2_job_state=args.v2_job_state,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
