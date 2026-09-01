"""Bind the role-correct FactMM-RAG qualification into the two RAG tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor.medeval.audit_rag_dual_track_v1 import audit_dual_track as audit_v1
from anchor.medeval.hashing import sha256_json
from anchor.medeval.store import atomic_write_json


def audit_dual_track(evidence_path: Path, factmm_path: Path) -> dict:
    factmm = json.loads(factmm_path.read_text())
    if factmm.get("protocol_version") != "factmm-rag-t0-qualification-v2":
        raise ValueError("role-correct FactMM-RAG T0-v2 is required")
    result = audit_v1(evidence_path, factmm_path)
    result["protocol_version"] = "rag-dual-track-qualification-v2"
    result["factmm_released_asset_role"] = factmm.get("released_asset_role")
    result["factmm_role_confusion_closed"] = (
        factmm.get("released_asset_role") == "generator"
        and factmm.get("common_protocol_retriever_asset_candidate") is False
    )
    result.pop("fingerprint", None)
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-evidence", type=Path, required=True)
    parser.add_argument("--factmm-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_dual_track(args.method_evidence, args.factmm_qualification)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
