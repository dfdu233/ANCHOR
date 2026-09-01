"""Materialize and safely inspect one unambiguous checkpoint from the official archive."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json


VERSION = "factmm-rag-official-archive-semantic-audit-v1"


def _candidate_score(name: str) -> int:
    lower = name.lower()
    base = PurePosixPath(lower).name
    score = 0
    if base == "dpr.best.pt":
        score += 100
    elif base == "model.best.pt":
        score += 70
    elif base == "pytorch_model.bin":
        score += 40
    if "retriever" in lower or "/dpr/" in lower:
        score += 50
    if "optimizer" in lower or "scheduler" in lower:
        score -= 100
    return score


def _select_candidate(candidates: list[str]) -> tuple[str | None, str]:
    if not candidates:
        return None, "no checkpoint-like member"
    scored = sorted(((_candidate_score(name), name) for name in candidates), reverse=True)
    if len(scored) == 1:
        return scored[0][1], "single checkpoint-like member"
    if scored[0][0] > scored[1][0]:
        return scored[0][1], "unique deterministic filename-role score"
    return None, "multiple equally ranked checkpoint-like members"


def _tensor_schema(checkpoint: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    suffix = checkpoint.suffix.lower()
    if suffix == ".safetensors":
        from safetensors import safe_open

        tensor_rows = []
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                tensor_rows.append(
                    {
                        "name": name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "numel": int(tensor.numel()),
                    }
                )
        root_keys = ["safetensors"]
    else:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(payload, dict):
            return {}, ["checkpoint root is not a mapping"]
        root_keys = sorted(map(str, payload))
        if isinstance(payload.get("model"), dict):
            state = payload["model"]
        elif isinstance(payload.get("state_dict"), dict):
            state = payload["state_dict"]
        elif payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            state = payload
        else:
            state = {}
            errors.append("no unambiguous tensor state mapping")
        tensor_rows = []
        for name, value in state.items():
            if isinstance(name, str) and isinstance(value, torch.Tensor):
                tensor_rows.append(
                    {
                        "name": name,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "numel": int(value.numel()),
                    }
                )
        if len(tensor_rows) != len(state):
            errors.append("state mapping contains non-tensor or non-string entries")
    if not tensor_rows:
        errors.append("checkpoint contains no tensors")
    prefixes = Counter(row["name"].split(".", 1)[0] for row in tensor_rows)
    return {
        "root_keys": root_keys,
        "tensor_entries": len(tensor_rows),
        "parameter_numel": sum(row["numel"] for row in tensor_rows),
        "top_level_prefix_counts": dict(sorted(prefixes.items())),
        "tensor_schema_sha256": sha256_json(tensor_rows),
    }, errors


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
    candidates = list(inventory.get("candidate_checkpoint_members", []))
    selected, selection_reason = _select_candidate(candidates)
    if selected is None:
        errors.append(selection_reason)

    checkpoint_path: Path | None = None
    checkpoint_hash: str | None = None
    schema: dict[str, Any] = {}
    if not errors and selected is not None:
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(selected)
            if info.file_size < 1_000_000:
                errors.append("selected checkpoint is implausibly small")
            elif info.file_size > 30_000_000_000:
                errors.append("selected checkpoint exceeds the 30GB safety ceiling")
            else:
                materialize_dir.mkdir(parents=True, exist_ok=True)
                suffixes = "".join(PurePosixPath(selected).suffixes) or ".bin"
                checkpoint_path = materialize_dir / f"selected_retriever{suffixes}"
                temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".partial")
                with archive.open(info, "r") as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                if temporary.stat().st_size != info.file_size:
                    errors.append("materialized checkpoint size mismatch")
                else:
                    temporary.replace(checkpoint_path)
                    checkpoint_hash = sha256_file(checkpoint_path)
                    try:
                        schema, schema_errors = _tensor_schema(checkpoint_path)
                        errors.extend(schema_errors)
                    except Exception as exc:  # fail closed and retain exact diagnostic
                        errors.append(f"safe tensor load failed: {type(exc).__name__}: {exc}")

    result = {
        "protocol_version": VERSION,
        "archive": str(archive_path.resolve()),
        "archive_sha256": archive_hash,
        "inventory": str(inventory_path.resolve()),
        "inventory_sha256": sha256_file(inventory_path),
        "download_provenance": str(download_provenance_path.resolve()),
        "download_provenance_sha256": sha256_file(download_provenance_path),
        "candidate_checkpoint_members": candidates,
        "selected_member": selected,
        "selection_reason": selection_reason,
        "materialized_checkpoint": str(checkpoint_path.resolve()) if checkpoint_path else None,
        "materialized_checkpoint_sha256": checkpoint_hash,
        **schema,
        "safe_weights_only_or_safetensors_load": bool(schema) and not errors,
        "official_archive_tensor_asset_admissible": bool(schema) and not errors,
        "paper_native_retriever_identity_verified": False,
        "paper_native_generator_released": False,
        "paper_native_end_to_end_efficacy_authorized": False,
        "decision": "tensor_asset_only" if bool(schema) and not errors else "not_admissible",
        "decision_reason": (
            "tensor schema is valid, but model role and end-to-end generator remain unverified"
            if bool(schema) and not errors
            else "archive did not yield one safely loadable, unambiguous checkpoint"
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
    result = audit_semantics(
        args.archive, args.inventory, args.download_provenance, args.materialize_dir
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
