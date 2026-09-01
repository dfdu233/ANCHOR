#!/usr/bin/env python3
"""One-case native Huatuo generation-ID identity canary for full replay.

The command loads no model until a physician-admitted full-replay manifest has
passed all CPU checks.  It regenerates one deterministic dev case under the
exact frozen source decode contract, directly captures ``output.sequences``,
and authorizes replay only when the decoded text exactly equals the frozen
visible answer.  A failed canary is written truthfully and blocks scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol

try:
    from anchor.corrected_sgta.specificity_full_replay_runtime_v1 import (
        IDENTITY_PROTOCOL_ID,
        ContractError,
        _resolve_image,
        load_full_replay_manifest,
    )
except ModuleNotFoundError:
    from specificity_full_replay_runtime_v1 import (  # type: ignore[no-redef]
        IDENTITY_PROTOCOL_ID,
        ContractError,
        _resolve_image,
        load_full_replay_manifest,
    )


class NativeIdentityAdapter(Protocol):
    def fingerprint(self) -> dict[str, Any]: ...

    def generate_native_identity(
        self,
        *,
        image_path: Path,
        question: str,
        seed: int,
        max_new_tokens: int,
    ) -> dict[str, Any]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    content = {key: value for key, value in payload.items() if key != "fingerprint"}
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def stable_seed(seed: int, item_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite identity canary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_identity_canary(
    *,
    manifest: Path,
    metadata: Path,
    image_root: Path,
    source_generation_config: Path,
    output: Path,
    adapter: NativeIdentityAdapter,
    base_seed: int = 42,
    command: list[str] | None = None,
) -> dict[str, Any]:
    rows, meta, _ = load_full_replay_manifest(
        manifest, metadata, identity_canary=None, require_identity=False
    )
    generation = json.loads(source_generation_config.read_text())
    fingerprint = str(generation.get("fingerprint", ""))
    if not fingerprint or fingerprint != meta.get("source_generation_fingerprint"):
        raise ContractError("source generation config fingerprint mismatch")
    if _canonical_fingerprint(generation) != fingerprint:
        raise ContractError("source generation config fingerprint is not self-consistent")
    expected_config_hash = meta.get("provenance", {}).get(
        "source_generation_config_sha256"
    )
    if not expected_config_hash or expected_config_hash != _sha256(
        source_generation_config
    ):
        raise ContractError("source generation config hash differs from manifest provenance")
    if generation.get("model") != "huatuo":
        raise ContractError("source generation config is not Huatuo")
    if generation.get("seed") != base_seed:
        raise ContractError("identity canary seed differs from frozen source generation")
    expected_contract = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 512,
        "min_new_tokens": 1,
        "repetition_penalty": 1.2,
    }
    observed_contract = {
        "do_sample": generation.get("generation", {}).get("do_sample"),
        "num_beams": generation.get("generation", {}).get("num_beams"),
        "max_new_tokens": generation.get("max_new_tokens"),
        "min_new_tokens": generation.get("generation", {}).get("min_new_tokens"),
        "repetition_penalty": generation.get("generation", {}).get("repetition_penalty"),
    }
    if observed_contract != expected_contract:
        raise ContractError(
            f"source generation contract is not the frozen greedy-512 path: {observed_contract}"
        )
    dev = sorted(
        (row for row in rows if row["split"] == "dev"),
        key=lambda row: row["sample_id"],
    )
    if not dev:
        raise ContractError("identity canary requires one admitted dev row")
    row = dev[0]
    if row.get("source_generation_fingerprint") != fingerprint:
        raise ContractError("canary row generation fingerprint drift")
    image_path = _resolve_image(image_root, row["image_relpath"])
    seed = stable_seed(base_seed, row["source_question_id"])
    adapter_fingerprint = adapter.fingerprint()
    if not isinstance(adapter_fingerprint, dict) or not adapter_fingerprint:
        raise ContractError("identity adapter fingerprint is absent")
    generated = adapter.generate_native_identity(
        image_path=image_path,
        question=row["question"],
        seed=seed,
        max_new_tokens=512,
    )
    ids = generated.get("direct_output_sequence_ids")
    if (
        generated.get("directly_captured_output_sequences") is not True
        or not isinstance(ids, list)
        or not ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in ids)
    ):
        raise ContractError("adapter did not directly return native output.sequences IDs")
    if generated.get("decode_contract") != expected_contract:
        raise ContractError("adapter native decode contract differs from frozen source")
    if generated.get("image_sha256") != _sha256(image_path):
        raise ContractError("identity adapter image hash mismatch")
    expected_text = row["full_visible_answer"]
    generated_text = str(generated.get("text", ""))
    identity = generated_text == expected_text
    payload = {
        "protocol": IDENTITY_PROTOCOL_ID,
        "status": "passed" if identity else "failed",
        "source_model": "huatuo",
        "sample_id": row["sample_id"],
        "case_id": row["case_id"],
        "edge_id": row["edge_id"],
        "source_question_id": row["source_question_id"],
        "manifest_sha256": _sha256(manifest),
        "metadata_sha256": _sha256(metadata),
        "source_generation_config_sha256": _sha256(source_generation_config),
        "source_generation_fingerprint": fingerprint,
        "adapter_fingerprint": adapter_fingerprint,
        "image_sha256": _sha256(image_path),
        "base_seed": base_seed,
        "sample_seed": seed,
        "directly_captured_output_sequences": True,
        "output_sequence_ids_sha256": _sha256_bytes(
            ",".join(str(value) for value in ids).encode()
        ),
        "output_sequence_token_count": len(ids),
        "expected_visible_text_sha256": _sha256_bytes(expected_text.encode()),
        "generated_visible_text_sha256": _sha256_bytes(generated_text.encode()),
        "decoded_visible_text_identity": identity,
        "hit_max_new_tokens": bool(generated.get("hit_max_new_tokens")),
        "decode_contract": expected_contract,
        "command": list(command or []),
        "gpu_scoring_authorized": identity,
        "failure_action": None if identity else "freeze failure; do not replay or substitute a new answer",
    }
    _write_once(output, payload)
    return payload


def _load_factory(specification: str, config: dict[str, Any]) -> NativeIdentityAdapter:
    if ":" not in specification:
        raise ContractError("--adapter-factory must be module:function")
    module_name, function_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--source-generation-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-factory", required=True)
    parser.add_argument("--adapter-config", type=Path)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()
    try:
        # Validate every CPU gate before constructing the GPU adapter.
        load_full_replay_manifest(
            args.manifest, args.metadata, identity_canary=None, require_identity=False
        )
        config = json.loads(args.adapter_config.read_text()) if args.adapter_config else {}
        adapter = _load_factory(args.adapter_factory, config)
        result = run_identity_canary(
            manifest=args.manifest,
            metadata=args.metadata,
            image_root=args.image_root,
            source_generation_config=args.source_generation_config,
            output=args.output,
            adapter=adapter,
            base_seed=args.base_seed,
            command=[shlex.join(sys.argv)],
        )
    except (ContractError, OSError, ValueError, ImportError, AttributeError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
