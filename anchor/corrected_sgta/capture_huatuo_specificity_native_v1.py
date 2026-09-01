#!/usr/bin/env python3
"""Capture direct Huatuo generation IDs for visible-answer replay.

The capture is case-complete and fail-closed.  It stores raw
``output.sequences`` IDs, removes only terminal EOS/PAD IDs for the visible
answer identity check, compares those IDs with the exact contextual
teacher-forcing target IDs, and verifies that the own image and both frozen
swaps expand to the same number of native visual tokens.
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
from typing import Any, Protocol, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corrected_sgta.specificity_ratchet_teacher_forcing_v1 import (
    ContractError,
    _canonical,
    _resolve_image,
    _safe_name,
    _sha256_bytes,
    _sha256_file,
)
from corrected_sgta.specificity_ratchet_visible_replay_v1 import (
    CAPTURE_PROTOCOL_ID,
    load_replay_manifest,
)


CAPTURE_RUNTIME_ID = "huatuo-specificity-native-capture-runtime-v1"


class NativeCaptureAdapter(Protocol):
    def fingerprint(self) -> dict[str, Any]: ...

    def generate_native_identity(
        self, *, image_path: Path, question: str, seed: int, max_new_tokens: int
    ) -> dict[str, Any]: ...

    def contextual_target_ids(self, *, target: str) -> list[int]: ...

    def visual_token_count(self, *, image_path: Path, question: str) -> int: ...


def _capture_status(*, is_canary: bool, failures: Sequence[str]) -> str:
    if is_canary:
        return "canary_failed" if failures else "canary_passed"
    return "complete_with_identity_failures" if failures else "complete_passed"


def stable_seed(base_seed: int, item_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_once_or_equal(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ContractError(f"native capture resume drift at {path}")
        return
    _atomic_write(path, payload)


def _trim_terminal_special_ids(
    raw_ids: Sequence[int], terminal_special_ids: Sequence[int]
) -> list[int]:
    ids = [int(value) for value in raw_ids]
    special = {int(value) for value in terminal_special_ids}
    if not ids:
        raise ContractError("direct output.sequences is empty")
    if any(value < 0 for value in ids) or any(value < 0 for value in special):
        raise ContractError("native generation IDs must be non-negative")
    while ids and ids[-1] in special:
        ids.pop()
    if not ids:
        raise ContractError("native generation contains only terminal special IDs")
    return ids


def _validate_existing_shard(
    path: Path, *, config_fingerprint: str, case_contract_sha256: str
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("capture_runtime_id") != CAPTURE_RUNTIME_ID:
        raise ContractError(f"wrong native capture shard protocol: {path}")
    if (
        payload.get("config_fingerprint") != config_fingerprint
        or payload.get("case_contract_sha256") != case_contract_sha256
    ):
        raise ContractError(f"native capture shard fingerprint drift: {path}")
    case = payload.get("case")
    if payload.get("case_sha256") != _sha256_bytes(_canonical(case)):
        raise ContractError(f"native capture shard checksum mismatch: {path}")
    return case


def run_capture(
    *,
    manifest: Path,
    metadata: Path,
    image_root: Path,
    output_dir: Path,
    adapter: NativeCaptureAdapter,
    split: str = "dev",
    base_seed: int = 42,
    limit_cases: int = 0,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows, meta = load_replay_manifest(manifest, metadata)
    if split not in {"dev", "test", "all"}:
        raise ContractError("capture split must be dev, test, or all")
    selected_rows = [row for row in rows if split == "all" or row["split"] == split]
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        by_case.setdefault(row["case_id"], []).append(row)
    case_ids = sorted(by_case)
    if not case_ids:
        raise ContractError(f"native capture has no cases for split={split}")
    if limit_cases < 0:
        raise ContractError("limit_cases must be non-negative")
    selected_case_ids = case_ids[:limit_cases] if limit_cases else case_ids

    adapter_fingerprint = adapter.fingerprint()
    if adapter_fingerprint.get("model_family") != meta["target_model_family"]:
        raise ContractError("native capture adapter targets another model family")
    config = {
        "capture_runtime_id": CAPTURE_RUNTIME_ID,
        "dataset": "VQA-RAD public image subset",
        "model": meta["target_model_family"],
        "method": "Specificity Ratchet full-visible-answer native identity capture",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": meta["manifest_sha256"],
        "metadata_sha256": _sha256_file(metadata),
        "image_root": str(image_root.resolve()),
        "adapter_fingerprint": adapter_fingerprint,
        "split": split,
        "base_seed": base_seed,
        "limit_cases": limit_cases,
        "command": list(command or []),
        "capture_source_sha256": _sha256_file(Path(__file__).resolve()),
        "terminal_normalization": "remove trailing EOS/PAD IDs only",
    }
    config_fingerprint = _sha256_bytes(_canonical(config))
    config["config_fingerprint"] = config_fingerprint
    _write_once_or_equal(
        output_dir / "config.json",
        (json.dumps(config, indent=2, sort_keys=True) + "\n").encode(),
    )

    captured_cases: list[dict[str, Any]] = []
    resumed = 0
    for case_id in selected_case_ids:
        case_rows = sorted(by_case[case_id], key=lambda row: row["sample_id"])
        exemplar = case_rows[0]
        case_contract = {
            "case_id": case_id,
            "source_question_id": exemplar["source_question_id"],
            "question": exemplar["question"],
            "image_relpath": exemplar["image_relpath"],
            "full_visible_answer_sha256": exemplar["full_visible_answer_sha256"],
            "matched_image_swaps": exemplar["matched_image_swaps"],
        }
        case_contract_sha = _sha256_bytes(_canonical(case_contract))
        shard_path = output_dir / "shards" / f"{_safe_name(case_id)}.json"
        if shard_path.exists():
            captured_cases.append(
                _validate_existing_shard(
                    shard_path,
                    config_fingerprint=config_fingerprint,
                    case_contract_sha256=case_contract_sha,
                )
            )
            resumed += 1
            continue

        own_path = _resolve_image(image_root, exemplar["image_relpath"])
        swap_paths = [
            _resolve_image(image_root, swap["image_relpath"])
            for swap in exemplar["matched_image_swaps"]
        ]
        sample_seed = stable_seed(base_seed, exemplar["source_question_id"])
        generated = adapter.generate_native_identity(
            image_path=own_path,
            question=exemplar["question"],
            seed=sample_seed,
            max_new_tokens=512,
        )
        raw_ids = generated.get("direct_output_sequence_ids")
        terminal_ids = generated.get("terminal_special_token_ids")
        if (
            generated.get("directly_captured_output_sequences") is not True
            or not isinstance(raw_ids, list)
            or not isinstance(terminal_ids, list)
        ):
            raise ContractError(f"{case_id}: adapter did not expose direct native IDs")
        visible_ids = _trim_terminal_special_ids(raw_ids, terminal_ids)
        contextual_ids = adapter.contextual_target_ids(
            target=exemplar["full_visible_answer"]
        )
        visual_counts = [
            adapter.visual_token_count(
                image_path=path, question=exemplar["question"]
            )
            for path in [own_path, *swap_paths]
        ]
        exact_text = generated.get("text") == exemplar["full_visible_answer"]
        exact_ids = visible_ids == contextual_ids
        equal_visual = len(set(visual_counts)) == 1
        case = {
            "case_id": case_id,
            "source_question_id": exemplar["source_question_id"],
            "frozen_visible_answer_sha256": exemplar["full_visible_answer_sha256"],
            "sample_seed": sample_seed,
            "directly_captured_output_sequences": True,
            "decoded_text_exact_frozen_match": exact_text,
            "native_ids_equal_contextual_target_ids": exact_ids,
            "raw_output_sequence_token_ids": [int(value) for value in raw_ids],
            "raw_output_sequence_token_ids_sha256": _sha256_bytes(_canonical(raw_ids)),
            "terminal_special_token_ids": [int(value) for value in terminal_ids],
            "native_generation_token_ids": visible_ids,
            "native_generation_token_ids_sha256": _sha256_bytes(_canonical(visible_ids)),
            "contextual_target_token_ids_sha256": _sha256_bytes(_canonical(contextual_ids)),
            "visual_token_counts_own_swap1_swap2": visual_counts,
            "visual_token_count_equal_across_own_swaps": equal_visual,
            "own_image_sha256": _sha256_file(own_path),
            "swap_image_sha256": [_sha256_file(path) for path in swap_paths],
            "decode_contract": generated.get("decode_contract"),
            "hit_max_new_tokens": bool(generated.get("hit_max_new_tokens")),
            "identity_passed": exact_text and exact_ids and equal_visual,
        }
        shard = {
            "capture_runtime_id": CAPTURE_RUNTIME_ID,
            "config_fingerprint": config_fingerprint,
            "case_contract_sha256": case_contract_sha,
            "case_sha256": _sha256_bytes(_canonical(case)),
            "case": case,
        }
        _atomic_write(
            shard_path,
            (json.dumps(shard, indent=2, sort_keys=True) + "\n").encode(),
        )
        captured_cases.append(case)

    captured_cases.sort(key=lambda row: row["case_id"])
    is_canary = bool(limit_cases and limit_cases < len(case_ids))
    failures = [row["case_id"] for row in captured_cases if not row["identity_passed"]]
    status = _capture_status(is_canary=is_canary, failures=failures)
    capture = {
        "capture_protocol_id": CAPTURE_PROTOCOL_ID,
        "status": status,
        "manifest_sha256": meta["manifest_sha256"],
        "metadata_sha256": config["metadata_sha256"],
        "target_model_family": meta["target_model_family"],
        "adapter_fingerprint": adapter_fingerprint,
        "split": split,
        "base_seed": base_seed,
        "config_fingerprint": config_fingerprint,
        "direct_output_sequences_captured_for_every_selected_case": True,
        "n_manifest_cases_in_split": len(case_ids),
        "n_captured_cases": len(captured_cases),
        "n_identity_failures": len(failures),
        "identity_failure_case_ids": failures,
        "cases": captured_cases,
    }
    filename = "CANARY.json" if is_canary else "native_capture.json"
    _write_once_or_equal(
        output_dir / filename,
        (json.dumps(capture, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {**capture, "resumed_cases": resumed, "output": str(output_dir / filename)}


def _load_factory(specification: str, config: dict[str, Any]) -> NativeCaptureAdapter:
    if ":" not in specification:
        raise ContractError("--adapter-factory must be module:function")
    module_name, function_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--adapter-factory", required=True)
    parser.add_argument("--adapter-config", type=Path)
    args = parser.parse_args()
    try:
        # CPU manifest admission happens before model construction.
        load_replay_manifest(args.manifest, args.metadata)
        adapter_config = (
            json.loads(args.adapter_config.read_text()) if args.adapter_config else {}
        )
        adapter = _load_factory(args.adapter_factory, adapter_config)
        result = run_capture(
            manifest=args.manifest,
            metadata=args.metadata,
            image_root=args.image_root,
            output_dir=args.output_dir,
            adapter=adapter,
            split=args.split,
            base_seed=args.base_seed,
            limit_cases=args.limit_cases,
            command=[shlex.join(sys.argv)],
        )
    except (ContractError, OSError, ValueError, ImportError, AttributeError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] in {"canary_failed", "complete_with_identity_failures"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
