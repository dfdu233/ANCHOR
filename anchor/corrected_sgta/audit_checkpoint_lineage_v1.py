#!/usr/bin/env python3
"""Strict CPU-only integrity and tensor-lineage audit for Huatuo/Qwen checkpoints.

The audit deliberately separates three statements which are often conflated:

* repository integrity (the downloaded bytes match the Git/LFS checkout),
* checkpoint relationship (current tensors are equal or different), and
* historical ancestry (which cannot be recovered from tensor proximity alone).

No model is instantiated and no GPU operation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open


VERSION = "checkpoint-lineage-audit-v1"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
EXPECTED_HEADS = {
    "medical": "451ac32400e36cfd07b41b62cbe63e6894895b38",
    "raw_vl": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "text": "a09a35458c702b33eeacc393d103063234e8bc28",
}
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def run(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_value(root: Path, *args: str) -> str | None:
    result = run(["git", "-C", str(root), *args])
    return result["stdout"] if result["returncode"] == 0 else None


def component(key: str) -> str:
    if key.startswith("visual.merger."):
        return "projector_merger"
    if key.startswith("visual."):
        return "vision_encoder"
    if key == "lm_head.weight" or key.startswith("model."):
        return "language_model"
    return "other"


def find_lfs_pointers(root: Path) -> list[str]:
    pointers = []
    for path in root.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        with path.open("rb") as handle:
            if handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                pointers.append(path.name)
    return sorted(pointers)


def read_schema(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    by_shard: dict[str, set[str]] = defaultdict(set)
    for key, shard in weight_map.items():
        by_shard[shard].add(key)

    schema: dict[str, dict[str, Any]] = {}
    shard_rows = []
    duplicate_header_keys: list[str] = []
    unexpected_header_keys: list[str] = []
    mapped_to_wrong_shard: list[str] = []
    for shard_name, mapped_keys in sorted(by_shard.items()):
        shard_path = root / shard_name
        if not shard_path.is_file():
            continue
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            header_keys = set(handle.keys())
            for key in sorted(header_keys):
                if key in schema:
                    duplicate_header_keys.append(key)
                tensor_slice = handle.get_slice(key)
                dtype = str(tensor_slice.get_dtype()).replace("torch.", "").upper()
                dtype = {"FLOAT16": "F16", "BFLOAT16": "BF16", "FLOAT32": "F32"}.get(dtype, dtype)
                shape = list(tensor_slice.get_shape())
                nbytes = math.prod(shape) * DTYPE_BYTES[dtype]
                schema[key] = {"shape": shape, "dtype": dtype, "nbytes": nbytes, "shard": shard_name}
                if key not in weight_map:
                    unexpected_header_keys.append(key)
                elif weight_map[key] != shard_name:
                    mapped_to_wrong_shard.append(key)
            shard_rows.append(
                {
                    "name": shard_name,
                    "size_on_disk": shard_path.stat().st_size,
                    "header_key_count": len(header_keys),
                    "missing_mapped_keys": sorted(mapped_keys - header_keys),
                    "unexpected_header_keys": sorted(header_keys - mapped_keys),
                }
            )

    declared_shards = sorted(set(weight_map.values()))
    present_shards = sorted(path.name for path in root.glob("*.safetensors"))
    declared_total = index.get("metadata", {}).get("total_size")
    computed_total = sum(row["nbytes"] for row in schema.values())
    validation = {
        "index_key_count": len(weight_map),
        "header_key_count": len(schema),
        "declared_shards": declared_shards,
        "present_shards": present_shards,
        "missing_shards": sorted(set(declared_shards) - set(present_shards)),
        "unexpected_shards": sorted(set(present_shards) - set(declared_shards)),
        "index_keys_missing_from_headers": sorted(set(weight_map) - set(schema)),
        "unexpected_header_keys": sorted(set(unexpected_header_keys)),
        "duplicate_header_keys": sorted(set(duplicate_header_keys)),
        "mapped_to_wrong_shard": sorted(set(mapped_to_wrong_shard)),
        "declared_tensor_bytes": declared_total,
        "computed_tensor_bytes": computed_total,
        "declared_tensor_bytes_match": declared_total == computed_total,
        "shards": shard_rows,
    }
    validation["pass"] = not any(
        validation[name]
        for name in (
            "missing_shards",
            "unexpected_shards",
            "index_keys_missing_from_headers",
            "unexpected_header_keys",
            "duplicate_header_keys",
            "mapped_to_wrong_shard",
        )
    ) and validation["declared_tensor_bytes_match"]
    return schema, validation


def repository_audit(root: Path, expected_head: str, state_path: Path | None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    head = git_value(root, "rev-parse", "HEAD")
    lfs_fsck = run(["git", "lfs", "fsck"], cwd=root)
    state = json.loads(state_path.read_text()) if state_path and state_path.is_file() else None
    required = ["README.md", "config.json", "model.safetensors.index.json"]
    missing_required = [name for name in required if not (root / name).is_file()]
    incomplete = sorted(str(path.relative_to(root)) for path in root.rglob("*.incomplete"))
    pointers = find_lfs_pointers(root)
    schema, index_validation = read_schema(root) if not missing_required else ({}, {"pass": False})
    state_pass = bool(state and state.get("status") == "done" and state.get("exit_code") == 0)
    result = {
        "root": str(root.resolve()),
        "detached_state": state,
        "detached_exit_pass": state_pass,
        "git_head": head,
        "expected_git_head": expected_head,
        "git_head_match": head == expected_head,
        "git_remote": git_value(root, "remote", "get-url", "origin"),
        "git_status_porcelain": git_value(root, "status", "--porcelain"),
        "lfs_fsck": lfs_fsck,
        "lfs_fsck_pass": lfs_fsck["returncode"] == 0,
        "lfs_pointer_files": pointers,
        "missing_required_files": missing_required,
        "incomplete_files": incomplete,
        "index_validation": index_validation,
    }
    result["integrity_pass"] = bool(
        state_pass
        and result["git_head_match"]
        and result["lfs_fsck_pass"]
        and not pointers
        and not missing_required
        and not incomplete
        and index_validation["pass"]
    )
    return result, schema


def tensor_digest(root: Path, key: str, weight_map: dict[str, str]) -> str:
    with safe_open(root / weight_map[key], framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(key).contiguous()
        byte_view = tensor.view(torch.uint8).numpy()
        digest = hashlib.blake2b(memoryview(byte_view), digest_size=16).hexdigest()
        del tensor, byte_view
        return digest


def all_tensor_digests(root: Path, keys: Iterable[str]) -> dict[str, str]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    return {key: tensor_digest(root, key, weight_map) for key in keys}


def schema_signature(row: dict[str, Any]) -> tuple[tuple[int, ...], str]:
    return tuple(row["shape"]), row["dtype"]


def summarize_equality(
    left_name: str,
    right_name: str,
    left_schema: dict[str, dict[str, Any]],
    right_schema: dict[str, dict[str, Any]],
    left_digests: dict[str, str],
    right_digests: dict[str, str],
) -> dict[str, Any]:
    shared = sorted(set(left_schema) & set(right_schema))
    comparable = [key for key in shared if schema_signature(left_schema[key]) == schema_signature(right_schema[key])]
    rows = []
    for group in ("vision_encoder", "projector_merger", "language_model", "other"):
        keys = [key for key in comparable if component(key) == group]
        exact = [key for key in keys if left_digests[key] == right_digests[key]]
        rows.append(
            {
                "component": group,
                "comparable_tensors": len(keys),
                "exact_tensors": len(exact),
                "changed_tensors": len(keys) - len(exact),
                "exact_tensor_rate": len(exact) / len(keys) if keys else None,
                "exact_tensor_bytes": sum(left_schema[key]["nbytes"] for key in exact),
                "comparable_tensor_bytes": sum(left_schema[key]["nbytes"] for key in keys),
            }
        )
    return {
        "left": left_name,
        "right": right_name,
        "left_key_count": len(left_schema),
        "right_key_count": len(right_schema),
        "shared_keys": len(shared),
        "same_shape_dtype_keys": len(comparable),
        "left_only_keys": sorted(set(left_schema) - set(right_schema)),
        "right_only_keys": sorted(set(right_schema) - set(left_schema)),
        "schema_mismatch_keys": sorted(set(shared) - set(comparable)),
        "component_equality": rows,
    }


def load_float_tensor(root: Path, key: str) -> torch.Tensor:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    with safe_open(root / index["weight_map"][key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).float()


def sampled_tensor_distances(
    left_root: Path,
    right_root: Path,
    left_schema: dict[str, dict[str, Any]],
    right_schema: dict[str, dict[str, Any]],
    left_digests: dict[str, str],
    right_digests: dict[str, str],
    per_component: int = 4,
) -> list[dict[str, Any]]:
    """Measure a frozen stratified sample after complete exact-byte comparison."""
    shared = set(left_schema) & set(right_schema)
    comparable_changed = [
        key
        for key in shared
        if schema_signature(left_schema[key]) == schema_signature(right_schema[key])
        and left_digests[key] != right_digests[key]
        and left_schema[key]["nbytes"] <= 40_000_000
    ]
    selected = []
    for group in ("vision_encoder", "projector_merger", "language_model", "other"):
        candidates = [key for key in comparable_changed if component(key) == group]
        candidates.sort(key=lambda key: hashlib.sha256(f"{VERSION}:{group}:{key}".encode()).hexdigest())
        selected.extend(candidates[:per_component])
    rows = []
    for key in selected:
        left = load_float_tensor(left_root, key)
        right = load_float_tensor(right_root, key)
        delta = left - right
        denominator = float(right.norm().item())
        rows.append(
            {
                "key": key,
                "component": component(key),
                "shape": list(left.shape),
                "max_abs_delta": float(delta.abs().max().item()),
                "relative_l2": float(delta.norm().item() / denominator) if denominator else None,
            }
        )
        del left, right, delta
    return rows


def readme_evidence(medical_root: Path, raw_root: Path, text_root: Path) -> dict[str, Any]:
    text = (medical_root / "README.md").read_text(errors="replace")
    frontmatter_parent = re.search(r"base_model:\s*\n\s*-\s*([^\n]+)", text)
    claims_raw_parent = "Qwen/Qwen2.5-VL-7B-Instruct" in text
    stale_qwen2_claim = bool(re.search(r"trained based on Qwen2-7B", text, re.I))
    recipe_terms = bool(re.search(r"(?:conversion|convert|merge|replace).{0,100}(?:checkpoint|weight|model)", text, re.I | re.S))
    return {
        "medical_readme_base_model": frontmatter_parent.group(1).strip() if frontmatter_parent else None,
        "declares_qwen25_vl_parent": claims_raw_parent,
        "contains_stale_qwen2_llava_sentence": stale_qwen2_claim,
        "contains_reproducible_conversion_recipe": recipe_terms,
        "raw_vl_head_examined": git_value(raw_root, "rev-parse", "HEAD"),
        "text_head_examined": git_value(text_root, "rev-parse", "HEAD"),
        "interpretation": (
            "README metadata names Qwen2.5-VL-7B-Instruct as base_model, but it gives no pinned "
            "base revision or reproducible conversion/training recipe; a conflicting legacy Qwen2/LLaVA "
            "sentence remains. README evidence therefore supports family-level provenance only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medical", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B-Qwen2.5VL"))
    parser.add_argument("--raw-vl", type=Path, default=Path("/home/dbw/models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--text", type=Path, default=Path("/home/dbw/models/Qwen2.5-7B-Instruct"))
    parser.add_argument("--states", type=Path, default=Path("corrected_runs/detached_jobs"))
    parser.add_argument("--output", type=Path, default=Path("corrected_runs/checkpoint_lineage_audit_v1/audit.json"))
    args = parser.parse_args()

    roots = {"medical": args.medical, "raw_vl": args.raw_vl, "text": args.text}
    state_names = {
        "medical": "download-huatuo-qwen25vl-medical-v1.json",
        "raw_vl": "download-qwen25vl-7b-parent-v1.json",
        "text": "download-qwen25-7b-text-parent-v1.json",
    }
    repositories: dict[str, Any] = {}
    schemas: dict[str, dict[str, dict[str, Any]]] = {}
    for name, root in roots.items():
        repositories[name], schemas[name] = repository_audit(
            root, EXPECTED_HEADS[name], args.states / state_names[name]
        )
    if not all(row["integrity_pass"] for row in repositories.values()):
        raise RuntimeError("at least one checkpoint failed strict repository integrity")

    all_keys = {name: sorted(schema) for name, schema in schemas.items()}
    digests = {name: all_tensor_digests(roots[name], all_keys[name]) for name in roots}
    comparisons = {
        "medical_vs_raw_vl": summarize_equality(
            "medical", "raw_vl", schemas["medical"], schemas["raw_vl"], digests["medical"], digests["raw_vl"]
        ),
        "medical_vs_text": summarize_equality(
            "medical", "text", schemas["medical"], schemas["text"], digests["medical"], digests["text"]
        ),
        "raw_vl_vs_text": summarize_equality(
            "raw_vl", "text", schemas["raw_vl"], schemas["text"], digests["raw_vl"], digests["text"]
        ),
    }
    comparisons["medical_vs_raw_vl"]["stratified_changed_tensor_distances"] = sampled_tensor_distances(
        roots["medical"], roots["raw_vl"], schemas["medical"], schemas["raw_vl"], digests["medical"], digests["raw_vl"]
    )
    comparisons["medical_vs_text"]["stratified_changed_tensor_distances"] = sampled_tensor_distances(
        roots["medical"], roots["text"], schemas["medical"], schemas["text"], digests["medical"], digests["text"]
    )
    comparisons["raw_vl_vs_text"]["stratified_changed_tensor_distances"] = sampled_tensor_distances(
        roots["raw_vl"], roots["text"], schemas["raw_vl"], schemas["text"], digests["raw_vl"], digests["text"]
    )
    readme = readme_evidence(args.medical, args.raw_vl, args.text)
    exact_parent_authorized = False
    result = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "method": "strict Git/LFS/index/header validation plus complete per-tensor BLAKE2b equality",
        "seed": None,
        "scope": "CPU-only Git/LFS/header/index/full-tensor-equality lineage audit",
        "repositories": repositories,
        "comparisons": comparisons,
        "readme_evidence": readme,
        "exact_parent_claim_authorized": exact_parent_authorized,
        "decision": "FAMILY_LEVEL_LINEAGE_ONLY",
        "reason": (
            "Current tensor equality can identify retained or changed components, but neither equality nor "
            "proximity proves initialization history. The release does not pin a base revision or provide a "
            "reproducible training/conversion manifest."
        ),
        "gpu_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "output": str(args.output), "gpu_used": False}, indent=2))


if __name__ == "__main__":
    main()
