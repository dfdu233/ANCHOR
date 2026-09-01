"""Inventory FactMM-RAG archives, including Hugging Face sharded checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from anchor.medeval.audit_factmm_rag_archive_v2 import audit_archive as audit_archive_v2
from anchor.medeval.hashing import sha256_json
from anchor.medeval.store import atomic_write_json


VERSION = "factmm-rag-official-archive-audit-v3"
_SHARD_RE = re.compile(r"^pytorch_model-(\d{5})-of-(\d{5})\.bin$")


def _read_small_json(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    info = archive.getinfo(member)
    if info.file_size > 10_000_000:
        raise ValueError(f"metadata JSON exceeds 10MB: {member}")
    payload = json.loads(archive.read(member))
    if not isinstance(payload, dict):
        raise ValueError(f"metadata JSON is not an object: {member}")
    return payload


def _sharded_groups(archive: zipfile.ZipFile) -> list[dict[str, Any]]:
    names = {info.filename for info in archive.infolist() if not info.is_dir()}
    by_parent: dict[str, list[str]] = defaultdict(list)
    for name in names:
        path = PurePosixPath(name)
        if _SHARD_RE.fullmatch(path.name):
            by_parent[str(path.parent)].append(name)

    groups: list[dict[str, Any]] = []
    for parent, members in sorted(by_parent.items()):
        index_member = str(PurePosixPath(parent) / "pytorch_model.bin.index.json")
        errors: list[str] = []
        indexed_shards: list[str] = []
        tensor_entries = 0
        total_size = None
        if index_member not in names:
            errors.append("missing pytorch_model.bin.index.json")
        else:
            try:
                index = _read_small_json(archive, index_member)
                weight_map = index.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    errors.append("index weight_map is empty or invalid")
                else:
                    tensor_entries = len(weight_map)
                    indexed_shards = sorted(
                        {str(PurePosixPath(parent) / str(value)) for value in weight_map.values()}
                    )
                    missing = sorted(set(indexed_shards) - names)
                    extra = sorted(set(members) - set(indexed_shards))
                    if missing:
                        errors.append(f"index references missing shards: {missing}")
                    if extra:
                        errors.append(f"unindexed checkpoint shards: {extra}")
                metadata = index.get("metadata")
                if isinstance(metadata, dict):
                    total_size = metadata.get("total_size")
            except Exception as exc:
                errors.append(f"invalid shard index: {type(exc).__name__}: {exc}")

        parsed = [_SHARD_RE.fullmatch(PurePosixPath(name).name) for name in members]
        totals = {int(match.group(2)) for match in parsed if match is not None}
        ordinals = {int(match.group(1)) for match in parsed if match is not None}
        if len(totals) != 1:
            errors.append("inconsistent shard totals")
        elif ordinals != set(range(1, next(iter(totals)) + 1)):
            errors.append("non-contiguous shard ordinals")

        groups.append(
            {
                "format": "huggingface_pytorch_sharded",
                "directory": parent,
                "index_member": index_member if index_member in names else None,
                "shard_members": sorted(members),
                "indexed_shard_members": indexed_shards,
                "tensor_entries_in_index": tensor_entries,
                "index_total_tensor_bytes": total_size,
                "complete": not errors,
                "errors": errors,
            }
        )
    return groups


def audit_archive(path: Path) -> dict[str, Any]:
    result = audit_archive_v2(path)
    with zipfile.ZipFile(path) as archive:
        groups = _sharded_groups(archive)
    shard_members = [member for group in groups for member in group["shard_members"]]
    standalone = list(result.get("candidate_checkpoint_members", []))
    result.update(
        {
            "protocol_version": VERSION,
            "candidate_checkpoint_members": sorted(set(standalone + shard_members)),
            "standalone_checkpoint_members": standalone,
            "sharded_checkpoint_groups": groups,
            "complete_checkpoint_group_count": sum(bool(group["complete"]) for group in groups),
            "archive_semantics_verified": False,
            "decision_reason": (
                "checkpoint groups are structurally inventoried; role and tensor schema require semantic audit"
                if result["safe_to_inventory"]
                else result["decision_reason"]
            ),
        }
    )
    result.pop("fingerprint", None)
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
