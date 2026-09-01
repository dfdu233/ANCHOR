"""Safely verify a complete HF sharded model and classify its released role."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json


VERSION = "factmm-rag-official-archive-semantic-audit-v2"


def _load_object(archive: zipfile.ZipFile, member: str, ceiling: int = 10_000_000) -> dict:
    info = archive.getinfo(member)
    if info.file_size > ceiling:
        raise ValueError(f"JSON metadata exceeds safety ceiling: {member}")
    payload = json.loads(archive.read(member))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON metadata is not an object: {member}")
    return payload


def _materialize_member(archive: zipfile.ZipFile, member: str, target: Path) -> dict[str, Any]:
    info = archive.getinfo(member)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with archive.open(info, "r") as source, temporary.open("wb") as sink:
        shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
    if temporary.stat().st_size != info.file_size:
        raise ValueError(f"materialized size mismatch: {member}")
    temporary.replace(target)
    return {
        "member": member,
        "path": str(target.resolve()),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "crc32_from_zip": f"{info.CRC:08x}",
    }


def _safe_shard_schema(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or not payload:
        return [], ["checkpoint shard root is empty or not a mapping"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for name, value in payload.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            errors.append("checkpoint shard contains a non-tensor or non-string entry")
            continue
        rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": int(value.numel()),
            }
        )
    del payload
    gc.collect()
    return rows, errors


def audit_semantics(
    archive_path: Path,
    inventory_path: Path,
    download_provenance_path: Path,
    materialize_dir: Path,
) -> dict[str, Any]:
    inventory = json.loads(inventory_path.read_text())
    provenance = json.loads(download_provenance_path.read_text())
    errors: list[str] = []
    archive_hash = sha256_file(archive_path)
    if inventory.get("archive_sha256") != archive_hash:
        errors.append("inventory/archive hash mismatch")
    if provenance.get("sha256") != archive_hash:
        errors.append("download provenance/archive hash mismatch")
    if not inventory.get("safe_to_inventory"):
        errors.append("archive failed safe inventory")
    groups = [group for group in inventory.get("sharded_checkpoint_groups", []) if group.get("complete")]
    if len(groups) != 1:
        errors.append(f"expected exactly one complete sharded checkpoint group, found {len(groups)}")

    config: dict[str, Any] = {}
    index: dict[str, Any] = {}
    materialized: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    shard_key_map: dict[str, str] = {}
    bad_crc_member: str | None = None
    if not errors:
        group = groups[0]
        parent = PurePosixPath(group["directory"])
        config_member = str(parent / "config.json")
        index_member = group["index_member"]
        with zipfile.ZipFile(archive_path) as archive:
            try:
                config = _load_object(archive, config_member)
                index = _load_object(archive, index_member)
                bad_crc_member = archive.testzip()
                if bad_crc_member is not None:
                    errors.append(f"ZIP CRC failure: {bad_crc_member}")
                weight_map = index.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    errors.append("invalid checkpoint weight_map")
                else:
                    shard_key_map = {str(key): str(value) for key, value in weight_map.items()}
                if not errors:
                    for member in group["indexed_shard_members"]:
                        target = materialize_dir / PurePosixPath(member).name
                        materialized.append(_materialize_member(archive, member, target))
                        rows, shard_errors = _safe_shard_schema(target)
                        errors.extend(f"{PurePosixPath(member).name}: {item}" for item in shard_errors)
                        expected_shard = PurePosixPath(member).name
                        for row in rows:
                            row["shard"] = expected_shard
                        tensor_rows.extend(rows)
            except Exception as exc:
                errors.append(f"safe sharded audit failed: {type(exc).__name__}: {exc}")

    actual_names = {row["name"] for row in tensor_rows}
    indexed_names = set(shard_key_map)
    missing_tensor_keys = sorted(indexed_names - actual_names)
    unindexed_tensor_keys = sorted(actual_names - indexed_names)
    wrong_shard_keys = sorted(
        row["name"]
        for row in tensor_rows
        if shard_key_map.get(row["name"]) not in {None, row["shard"]}
    )
    if missing_tensor_keys:
        errors.append(f"{len(missing_tensor_keys)} indexed tensors absent from shards")
    if unindexed_tensor_keys:
        errors.append(f"{len(unindexed_tensor_keys)} tensors absent from index")
    if wrong_shard_keys:
        errors.append(f"{len(wrong_shard_keys)} tensors stored in the wrong shard")

    architectures = config.get("architectures") if isinstance(config.get("architectures"), list) else []
    model_type = config.get("model_type")
    names = indexed_names or actual_names
    key_family_counts = {
        "language_layers": sum(name.startswith("model.layers.") for name in names),
        "lm_head": sum(name.startswith("lm_head.") for name in names),
        "vision_tower": sum("vision_tower" in name for name in names),
        "mm_projector": sum("mm_projector" in name for name in names),
        "retriever_named": sum(
            any(token in name.lower() for token in ("retriever", "query_encoder", "ctx_encoder"))
            for name in names
        ),
    }
    generator_role = bool(
        any("CausalLM" in str(item) for item in architectures)
        and key_family_counts["language_layers"]
        and key_family_counts["lm_head"]
    )
    projector_present = key_family_counts["mm_projector"] > 0
    vision_tower_present = key_family_counts["vision_tower"] > 0
    retriever_role = key_family_counts["retriever_named"] > 0 and not generator_role
    schema_valid = bool(tensor_rows) and not errors
    prefix_counts = Counter(".".join(row["name"].split(".")[:2]) for row in tensor_rows)

    result = {
        "protocol_version": VERSION,
        "archive": str(archive_path.resolve()),
        "archive_sha256": archive_hash,
        "inventory": str(inventory_path.resolve()),
        "inventory_sha256": sha256_file(inventory_path),
        "download_provenance": str(download_provenance_path.resolve()),
        "download_provenance_sha256": sha256_file(download_provenance_path),
        "model_identity": {
            "architectures": architectures,
            "model_type": model_type,
            "base_name_or_path": config.get("_name_or_path"),
            "vision_tower_config": config.get("mm_vision_tower"),
        },
        "checkpoint_group_count": len(groups),
        "materialized_shards": materialized,
        "zip_crc_all_members_valid": bad_crc_member is None and bool(groups),
        "tensor_entries_in_index": len(indexed_names),
        "tensor_entries_loaded": len(tensor_rows),
        "parameter_numel": sum(row["numel"] for row in tensor_rows),
        "top_level_prefix_counts": dict(sorted(prefix_counts.items())),
        "key_family_counts": key_family_counts,
        "tensor_schema_sha256": sha256_json(sorted(tensor_rows, key=lambda row: row["name"])) if tensor_rows else None,
        "missing_tensor_keys": missing_tensor_keys,
        "unindexed_tensor_keys": unindexed_tensor_keys,
        "wrong_shard_keys": wrong_shard_keys,
        "safe_weights_only_load": schema_valid,
        "official_archive_tensor_asset_admissible": schema_valid,
        "paper_native_generator_identity_verified": schema_valid and generator_role,
        "paper_native_projector_present": schema_valid and projector_present,
        "paper_native_vision_tower_present": schema_valid and vision_tower_present,
        "paper_native_retriever_identity_verified": schema_valid and retriever_role,
        "paper_native_end_to_end_efficacy_authorized": False,
        "decision": "generator_tensor_asset_only" if schema_valid and generator_role else "not_admissible",
        "decision_reason": (
            "official archive is a safely verified LLaVA generator checkpoint with vision tower and projector; no retriever asset or corpus is supplied"
            if schema_valid and generator_role and projector_present and vision_tower_present
            else "archive did not yield a complete, safely loadable and role-identifiable sharded checkpoint"
        ),
        "errors": errors,
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--download-provenance", type=Path, required=True)
    parser.add_argument("--materialize-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_semantics(args.archive, args.inventory, args.download_provenance, args.materialize_dir)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
