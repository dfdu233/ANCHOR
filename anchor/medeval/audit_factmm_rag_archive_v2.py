"""Safely inventory the official FactMM-RAG ZIP without extracting or executing it."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path, PurePosixPath

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json


def _unsafe_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return path.is_absolute() or ".." in path.parts


def audit_archive(path: Path) -> dict:
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ValueError(f"not a valid ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        members = [
            {
                "name": info.filename,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "encrypted": bool(info.flag_bits & 0x1),
                "is_directory": info.is_dir(),
            }
            for info in infos
        ]
    names = [item["name"] for item in members]
    unsafe = [name for name in names if _unsafe_member(name)]
    encrypted = [item["name"] for item in members if item["encrypted"]]
    lower_names = [name.lower() for name in names]
    candidate_checkpoints = [
        name
        for name, lower in zip(names, lower_names)
        if lower.endswith((".pt", ".pth", ".ckpt", ".safetensors"))
        or lower.endswith("pytorch_model.bin")
    ]
    candidate_embeddings = [
        name for name, lower in zip(names, lower_names) if lower.endswith((".pkl", ".npy", ".faiss"))
    ]
    safe_inventory = not unsafe and not encrypted and bool(members)
    result = {
        "protocol_version": "factmm-rag-official-archive-audit-v2",
        "archive_path": str(path.resolve()),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "member_count": len(members),
        "total_uncompressed_bytes": sum(item["uncompressed_bytes"] for item in members),
        "unsafe_members": unsafe,
        "encrypted_members": encrypted,
        "candidate_checkpoint_members": candidate_checkpoints,
        "candidate_embedding_members": candidate_embeddings,
        "members": members,
        "safe_to_inventory": safe_inventory,
        "archive_semantics_verified": False,
        "retriever_checkpoint_identity_verified": False,
        "paper_native_generator_released": False,
        "paper_native_efficacy_authorized": False,
        "decision": "inventory_only" if safe_inventory else "reject_archive",
        "decision_reason": (
            "member names require a separate semantic and tensor-schema audit"
            if safe_inventory
            else "archive contains unsafe/encrypted members or is empty"
        ),
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_archive(args.archive)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["safe_to_inventory"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
