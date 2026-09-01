"""Freeze the FactMM-RAG T0 decision after official archive inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json


def qualify(
    source_audit_path: Path,
    archive_inventory_path: Path,
    semantic_audit_path: Path,
    repository: Path,
    mimic_root: Path,
    chexpert_root: Path,
) -> dict:
    source_audit = json.loads(source_audit_path.read_text())
    inventory = json.loads(archive_inventory_path.read_text())
    semantic = json.loads(semantic_audit_path.read_text())
    rows = [row for row in source_audit.get("methods", []) if row.get("name") == "FactMM-RAG"]
    if len(rows) != 1:
        raise ValueError("source audit must contain exactly one FactMM-RAG row")
    source = rows[0]
    retriever_tensor = bool(semantic.get("official_archive_tensor_asset_admissible"))
    required = {
        "official_source_fixed": source.get("source_fingerprint", {}).get("sha256") is not None,
        "license_present": bool(source.get("license_sha256")),
        "archive_safe_inventory": bool(inventory.get("safe_to_inventory")),
        "retriever_tensor_asset": retriever_tensor,
        "retriever_role_identity_verified": bool(
            semantic.get("paper_native_retriever_identity_verified")
        ),
        "split_complete_mimic_cxr_present": mimic_root.is_dir(),
        "split_complete_chexpert_present": chexpert_root.is_dir(),
        "paper_native_generator_checkpoint_present": False,
        "paper_native_projector_checkpoint_present": False,
    }
    missing = [name for name, present in required.items() if not present]
    t0_pass = not missing
    result = {
        "protocol_version": "factmm-rag-t0-qualification-v1",
        "method": "FactMM-RAG",
        "official_repository": str(repository.resolve()),
        "official_commit": "bae1f530fa8be246ca3d8a769a88eb2d0d6e5dfe",
        "source_audit": str(source_audit_path.resolve()),
        "source_audit_sha256": sha256_file(source_audit_path),
        "archive_inventory": str(archive_inventory_path.resolve()),
        "archive_inventory_sha256": sha256_file(archive_inventory_path),
        "semantic_audit": str(semantic_audit_path.resolve()),
        "semantic_audit_sha256": sha256_file(semantic_audit_path),
        "requirements": required,
        "missing_requirements": missing,
        "paper_native_t0_status": "pass" if t0_pass else "not_admissible",
        "paper_native_t1_authorized": t0_pass,
        "common_protocol_retriever_asset_candidate": retriever_tensor,
        "common_protocol_use_authorized": False,
        "paper_native_end_to_end_efficacy_authorized": False,
        "claim_boundary": (
            "A valid archive/tensor asset, if present, does not reproduce the native RAG system; "
            "generator/projector assets and split-complete corpora remain mandatory."
        ),
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--archive-inventory", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--chexpert-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(
        args.source_audit,
        args.archive_inventory,
        args.semantic_audit,
        args.repository,
        args.mimic_root,
        args.chexpert_root,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
