#!/usr/bin/env python3
"""CPU-only, outcome-blind lineage audit for Huatuo Qwen2.5-VL candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


VERSION = "huatuo-qwen25vl-lineage-audit-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    completed = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def repository_contract(root: Path) -> dict[str, Any]:
    required = ["README.md", "config.json", "model.safetensors.index.json"]
    missing = [name for name in required if not (root / name).is_file()]
    incomplete = sorted(str(path.relative_to(root)) for path in root.rglob("*.incomplete"))
    index = json.loads((root / "model.safetensors.index.json").read_text()) if not missing else {}
    shards = sorted(set(index.get("weight_map", {}).values()))
    missing_shards = [name for name in shards if not (root / name).is_file()]
    return {
        "root": str(root.resolve()),
        "git_head": git_value(root, "rev-parse", "HEAD"),
        "git_remote": git_value(root, "remote", "get-url", "origin"),
        "required_missing": missing,
        "incomplete_files": incomplete,
        "declared_shards": shards,
        "missing_shards": missing_shards,
        "complete": not missing and not incomplete and not missing_shards,
        "metadata_hashes": {
            name: sha256(root / name)
            for name in ("README.md", "config.json", "model.safetensors.index.json", "tokenizer.json", "vocab.json", "merges.txt")
            if (root / name).is_file()
        },
    }


def tensor_schema(root: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for key, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(key)
    schema = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                tensor = handle.get_slice(key)
                schema[key] = (tuple(tensor.get_shape()), str(tensor.get_dtype()))
    return schema


def deterministic_tensor_sample(shared: list[str], schema: dict[str, tuple[tuple[int, ...], str]], size: int = 64) -> list[str]:
    candidates = []
    for key in shared:
        shape = schema[key][0]
        count = int(np.prod(shape)) if shape else 1
        if count <= 20_000_000:
            candidates.append(key)
    return sorted(candidates, key=lambda key: hashlib.sha256((VERSION + ":" + key).encode()).hexdigest())[:size]


def load_tensor(root: Path, key: str):
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][key]
    with safe_open(root / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).float()


def compare(child: Path, parent: Path) -> dict[str, Any]:
    child_schema, parent_schema = tensor_schema(child), tensor_schema(parent)
    shared = sorted(set(child_schema) & set(parent_schema))
    same_schema = [key for key in shared if child_schema[key] == parent_schema[key]]
    sample = deterministic_tensor_sample(same_schema, child_schema)
    rows = []
    for key in sample:
        left, right = load_tensor(child, key), load_tensor(parent, key)
        delta = left - right
        denominator = float(right.norm().item())
        rows.append(
            {
                "key": key,
                "shape": list(left.shape),
                "exact": bool(left.equal(right)),
                "max_abs_delta": float(delta.abs().max().item()),
                "relative_l2": float(delta.norm().item() / denominator) if denominator else None,
            }
        )
    return {
        "child_keys": len(child_schema),
        "parent_keys": len(parent_schema),
        "shared_keys": len(shared),
        "same_shape_dtype_keys": len(same_schema),
        "child_only_keys": sorted(set(child_schema) - set(parent_schema)),
        "parent_only_keys": sorted(set(parent_schema) - set(child_schema)),
        "sample_rule": f"lowest SHA256({VERSION}:key), numel <= 20M, n={len(sample)}",
        "sampled_tensor_distances": rows,
        "sample_exact_rate": float(np.mean([row["exact"] for row in rows])) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B-Qwen2.5VL"))
    parser.add_argument("--vl-parent", type=Path, default=Path("/home/dbw/models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--text-control", type=Path, default=Path("/home/dbw/models/Qwen2.5-7B-Instruct"))
    parser.add_argument("--output", type=Path, default=Path("corrected_runs/ppi_model_lineage_v1/audit.json"))
    args = parser.parse_args()
    contracts = {
        "child": repository_contract(args.child),
        "vl_parent": repository_contract(args.vl_parent),
        "text_control": repository_contract(args.text_control),
    }
    if not all(value["complete"] for value in contracts.values()):
        raise ValueError("model repositories are not content-complete")
    child_readme = (args.child / "README.md").read_text(errors="replace")
    declared_qwen25vl = bool(re.search(r"Qwen2[.]5[- ]VL", child_readme, re.I))
    conversion_recipe_present = bool(re.search(r"(?:training|fine[- ]tun|convert).{0,80}(?:script|command|recipe)", child_readme, re.I | re.S))
    comparison = compare(args.child, args.vl_parent)
    # Tensor distance shows relationship/degree of adaptation, not historical
    # initialization.  Exact continuation needs an explicit base revision and
    # reproducible recipe, neither of which may be inferred from compatibility.
    exact_parent_authorized = bool(
        declared_qwen25vl
        and conversion_recipe_present
        and contracts["vl_parent"]["git_head"] is not None
        and re.search(re.escape(contracts["vl_parent"]["git_head"]), child_readme, re.I)
    )
    result = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU metadata, schema and deterministic tensor-distance audit",
        "repositories": contracts,
        "child_readme_declares_qwen25vl_family": declared_qwen25vl,
        "child_readme_contains_reproducible_conversion_or_training_recipe": conversion_recipe_present,
        "child_vs_vl_parent": comparison,
        "exact_parent_claim_authorized": exact_parent_authorized,
        "decision": "EXACT_PARENT_ADMITTED" if exact_parent_authorized else "LINEAGE_CANDIDATE_ONLY",
        "gpu_authorized": False,
        "prohibited_inference": "architecture and tensor proximity cannot recover undocumented training ancestry",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "exact_parent_claim_authorized": exact_parent_authorized, "gpu_authorized": False}, indent=2))


if __name__ == "__main__":
    main()
