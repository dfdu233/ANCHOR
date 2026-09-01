#!/usr/bin/env python3
"""Fail-closed full-visible-answer Specificity Ratchet replay runtime.

The runtime never constructs isolated parent/child answers.  It scores the
complete frozen Huatuo answer under its own image, two frozen same-split
modality/anatomy swaps with exactly matching native visual-token length, and a
secondary text-only condition.  GPU scoring requires a separate native
generation-ID identity canary produced after physician admission.
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

import numpy as np

try:
    from anchor.corrected_sgta.specificity_ratchet_teacher_forcing_v1 import (
        ContractError,
        RowExclusion,
        TeacherForcedTrace,
        _matched_indices,
        _mean_by_layer,
        _validate_trace,
        map_constraint_spans,
    )
except ModuleNotFoundError:
    from specificity_ratchet_teacher_forcing_v1 import (  # type: ignore[no-redef]
        ContractError,
        RowExclusion,
        TeacherForcedTrace,
        _matched_indices,
        _mean_by_layer,
        _validate_trace,
        map_constraint_spans,
    )


RUNTIME_PROTOCOL_ID = "specificity-ratchet-full-visible-replay-runtime-v1"
MANIFEST_PROTOCOL_ID = "specificity-ratchet-full-visible-replay-v1"
IDENTITY_PROTOCOL_ID = "specificity-ratchet-native-generation-identity-v1"


class FullReplayAdapter(Protocol):
    def fingerprint(self) -> dict[str, Any]: ...

    def visual_token_count(self, *, image_path: Path, question: str) -> int: ...

    def score(
        self,
        *,
        image_path: Path | None,
        question: str,
        target: str,
        condition: str,
    ) -> TeacherForcedTrace: ...


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
            raise ContractError(f"resume fingerprint drift at {path}")
        return
    _atomic_write(path, payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _resolve_image(image_root: Path, relative: str) -> Path:
    path = (image_root / relative).resolve()
    if not path.is_relative_to(image_root.resolve()):
        raise ContractError(f"image path escapes root: {relative}")
    if not path.is_file():
        raise ContractError(f"replay image is missing: {path}")
    return path


def load_full_replay_manifest(
    manifest: Path,
    metadata: Path,
    *,
    identity_canary: Path | None = None,
    require_identity: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    rows = _read_jsonl(manifest)
    meta = json.loads(metadata.read_text())
    if meta.get("manifest_protocol_id") != MANIFEST_PROTOCOL_ID:
        raise ContractError("full-replay metadata protocol mismatch")
    if meta.get("status") != "physician_admitted_full_visible_replay":
        raise ContractError("full-replay manifest is not physician-admitted")
    if meta.get("manifest_sha256") != _sha256(manifest) or meta.get("rows") != len(rows):
        raise ContractError("full-replay manifest/metadata identity mismatch")
    if meta.get("source_model") != "huatuo":
        raise ContractError("current full-replay protocol requires Huatuo-native outputs")
    if meta.get("native_generation_sequence_certified") is not False:
        raise ContractError("manifest must not pre-certify native generation identity")
    if meta.get("isolated_parent_child_runtime_prohibited") is not True:
        raise ContractError("manifest omitted the F6 isolated-target prohibition")
    required = {
        "sample_id",
        "case_id",
        "edge_id",
        "image_relpath",
        "question",
        "full_visible_answer",
        "full_visible_answer_sha256",
        "constraint_char_spans_in_visible_answer",
        "scientific_role",
        "split",
        "swap_candidates",
        "source_generation_fingerprint",
    }
    sample_ids: set[str] = set()
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ContractError(f"full-replay row missing fields: {sorted(missing)}")
        if "parent_target" in row or "child_target" in row:
            raise ContractError("F6-rejected isolated targets leaked into full replay")
        if row["sample_id"] in sample_ids:
            raise ContractError("full-replay sample IDs are duplicated")
        sample_ids.add(row["sample_id"])
        target = str(row["full_visible_answer"])
        if _sha256_bytes(target.encode()) != row["full_visible_answer_sha256"]:
            raise ContractError(f"{row['sample_id']}: visible-answer hash mismatch")
        if row["split"] not in {"dev", "test"}:
            raise ContractError(f"{row['sample_id']}: invalid split")
        if row["source_generation_fingerprint"] != meta.get(
            "source_generation_fingerprint"
        ):
            raise ContractError(f"{row['sample_id']}: source generation fingerprint drift")
        swaps = row["swap_candidates"]
        if not isinstance(swaps, list) or len(swaps) < 2:
            raise ContractError(f"{row['sample_id']}: fewer than two frozen swap candidates")
        for swap in swaps:
            if swap.get("case_id") == row["case_id"] or swap.get("split") != row["split"]:
                raise ContractError(f"{row['sample_id']}: invalid/leaking swap candidate")

    canary = None
    if identity_canary is not None:
        canary = json.loads(identity_canary.read_text())
        if canary.get("protocol") != IDENTITY_PROTOCOL_ID or canary.get("status") != "passed":
            raise ContractError("native-generation identity canary did not pass")
        if canary.get("manifest_sha256") != _sha256(manifest):
            raise ContractError("identity canary belongs to a different manifest")
        if canary.get("metadata_sha256") != _sha256(metadata):
            raise ContractError("identity canary belongs to different metadata")
        if canary.get("source_model") != "huatuo":
            raise ContractError("identity canary source model mismatch")
        if canary.get("source_generation_fingerprint") != meta.get(
            "source_generation_fingerprint"
        ):
            raise ContractError("identity canary source generation fingerprint mismatch")
        if canary.get("directly_captured_output_sequences") is not True:
            raise ContractError("identity canary did not capture output.sequences directly")
        if canary.get("decoded_visible_text_identity") is not True:
            raise ContractError("identity canary did not reproduce the frozen visible answer")
        if canary.get("gpu_scoring_authorized") is not True:
            raise ContractError("identity canary did not authorize GPU scoring")
        if canary.get("sample_id") not in sample_ids:
            raise ContractError("identity canary sample is absent from manifest")
    elif require_identity:
        raise ContractError("native-generation identity canary is required before model loading")
    return rows, meta, canary


def _trace_matrix(
    trace: TeacherForcedTrace,
    *,
    row: dict[str, Any],
    condition: str,
    expected_image_sha256: str | None,
) -> np.ndarray:
    return _validate_trace(
        trace,
        target=row["full_visible_answer"],
        question=row["question"],
        condition=condition,
        expected_image_sha256=expected_image_sha256,
    )


def compute_full_replay_signals(
    *,
    row: dict[str, Any],
    own_trace: TeacherForcedTrace,
    swap_traces: Sequence[TeacherForcedTrace],
    text_trace: TeacherForcedTrace,
    own_image_sha256: str,
    swap_image_sha256: Sequence[str],
) -> dict[str, Any]:
    if len(swap_traces) < 2 or len(swap_traces) != len(swap_image_sha256):
        raise ContractError("full replay requires at least two aligned swap traces")
    own = _trace_matrix(
        own_trace,
        row=row,
        condition="image",
        expected_image_sha256=own_image_sha256,
    )
    swaps = [
        _trace_matrix(
            trace,
            row=row,
            condition="image",
            expected_image_sha256=image_hash,
        )
        for trace, image_hash in zip(swap_traces, swap_image_sha256)
    ]
    text = _trace_matrix(
        text_trace,
        row=row,
        condition="text_only",
        expected_image_sha256=None,
    )
    traces = [own_trace, *swap_traces, text_trace]
    for trace in traces[1:]:
        if (
            trace.token_ids != own_trace.token_ids
            or trace.token_offsets != own_trace.token_offsets
            or trace.layer_ids != own_trace.layer_ids
            or trace.template_id != own_trace.template_id
        ):
            raise ContractError("own/swap/text replay token, layer, offset, or template drift")
    constraint = map_constraint_spans(
        row["full_visible_answer"],
        row["constraint_char_spans_in_visible_answer"],
        own_trace.token_offsets,
        own_trace.offset_unit,
    )
    constraint_set = set(constraint)
    nonconstraint = [
        index for index in range(len(own_trace.token_ids)) if index not in constraint_set
    ]
    matched = _matched_indices(
        nonconstraint,
        constraint,
        len(own_trace.token_ids),
        len(own_trace.token_ids),
    )

    def signals(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        constraint_logp = np.asarray(_mean_by_layer(values, constraint))
        matched_logp = np.asarray(_mean_by_layer(values, matched))
        return constraint_logp, matched_logp, constraint_logp - matched_logp

    own_constraint, own_matched, own_contrast = signals(own)
    swap_parts = [signals(values) for values in swaps]
    swap_contrasts = np.stack([part[2] for part in swap_parts], axis=0)
    text_constraint, text_matched, text_contrast = signals(text)
    swap_mean = swap_contrasts.mean(axis=0)
    return {
        "layer_ids": own_trace.layer_ids,
        "token_counts": {
            "full_visible_answer": len(own_trace.token_ids),
            "constraint": len(constraint),
            "matched_nonconstraint": len(matched),
        },
        "token_indices": {
            "constraint": constraint,
            "matched_nonconstraint": matched,
        },
        "own_image": {
            "constraint_logp": own_constraint.tolist(),
            "matched_nonconstraint_logp": own_matched.tolist(),
            "constraint_minus_matched": own_contrast.tolist(),
        },
        "swap_images": {
            "count": len(swaps),
            "constraint_minus_matched_by_swap": swap_contrasts.tolist(),
            "mean_constraint_minus_matched": swap_mean.tolist(),
        },
        "primary_own_minus_swap_difference_in_differences": (
            own_contrast - swap_mean
        ).tolist(),
        "text_only_secondary": {
            "constraint_logp": text_constraint.tolist(),
            "matched_nonconstraint_logp": text_matched.tolist(),
            "constraint_minus_matched": text_contrast.tolist(),
            "lexical_sensitivity_only": True,
        },
    }


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _validate_shard(path: Path, config_fingerprint: str, row_sha256: str) -> dict[str, Any]:
    shard = json.loads(path.read_text())
    if shard.get("config_fingerprint") != config_fingerprint or shard.get("row_sha256") != row_sha256:
        raise ContractError(f"resume shard identity drift: {path}")
    payload = shard.get("payload")
    if shard.get("payload_sha256") != _sha256_bytes(_canonical(payload)):
        raise ContractError(f"resume shard checksum mismatch: {path}")
    return payload


def run_full_replay(
    *,
    manifest: Path,
    metadata: Path,
    identity_canary: Path,
    image_root: Path,
    output_dir: Path,
    adapter: FullReplayAdapter,
    split: str = "dev",
    swaps_per_row: int = 2,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows, _, canary = load_full_replay_manifest(
        manifest, metadata, identity_canary=identity_canary, require_identity=True
    )
    assert canary is not None
    if split not in {"dev", "test"} or swaps_per_row < 2:
        raise ContractError("full replay requires split=dev/test and at least two swaps")
    rows = sorted(
        (row for row in rows if row["split"] == split),
        key=lambda row: row["sample_id"],
    )
    if not rows:
        raise ContractError(f"full-replay manifest has no rows for split={split}")
    adapter_fingerprint = adapter.fingerprint()
    if not isinstance(adapter_fingerprint, dict) or not adapter_fingerprint:
        raise ContractError("full-replay adapter fingerprint is absent")
    if canary.get("adapter_fingerprint") != adapter_fingerprint:
        raise ContractError("scoring adapter differs from native identity canary adapter")
    config = {
        "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
        "manifest_sha256": _sha256(manifest),
        "metadata_sha256": _sha256(metadata),
        "identity_canary_sha256": _sha256(identity_canary),
        "source_sha256": _sha256(Path(__file__)),
        "adapter_fingerprint": adapter_fingerprint,
        "image_root": str(image_root.resolve()),
        "split": split,
        "swaps_per_row": swaps_per_row,
        "command": list(command or []),
    }
    config_fingerprint = _sha256_bytes(_canonical(config))
    config["config_fingerprint"] = config_fingerprint
    _write_once_or_equal(
        output_dir / "config.json",
        (json.dumps(config, indent=2, sort_keys=True) + "\n").encode(),
    )
    resumed = 0
    payloads: list[dict[str, Any]] = []
    for row in rows:
        row_sha = _sha256_bytes(_canonical(row))
        shard_path = output_dir / "shards" / f"{_safe_name(row['sample_id'])}.json"
        if shard_path.exists():
            payloads.append(_validate_shard(shard_path, config_fingerprint, row_sha))
            resumed += 1
            continue
        own_path = _resolve_image(image_root, row["image_relpath"])
        own_count = int(
            adapter.visual_token_count(image_path=own_path, question=row["question"])
        )
        if own_count <= 0:
            raise ContractError("adapter returned a non-positive own visual-token count")
        selected_swaps: list[tuple[dict[str, Any], Path]] = []
        rejected_counts: list[dict[str, Any]] = []
        for candidate in row["swap_candidates"]:
            swap_path = _resolve_image(image_root, candidate["image_relpath"])
            count = int(
                adapter.visual_token_count(
                    image_path=swap_path, question=row["question"]
                )
            )
            if count == own_count:
                selected_swaps.append((candidate, swap_path))
                if len(selected_swaps) == swaps_per_row:
                    break
            else:
                rejected_counts.append(
                    {"case_id": candidate["case_id"], "visual_token_count": count}
                )
        base = {
            "sample_id": row["sample_id"],
            "case_id": row["case_id"],
            "edge_id": row["edge_id"],
            "split": row["split"],
            "scientific_role": row["scientific_role"],
            "edge_type": row["edge_type"],
            "modality_stratum": row["modality_stratum"],
            "anatomy_stratum": row["anatomy_stratum"],
            "prompt_requested_increment": row["prompt_requested_increment"],
            "full_visible_answer_sha256": row["full_visible_answer_sha256"],
            "own_visual_token_count": own_count,
            "visual_length_mismatch_candidates": rejected_counts,
        }
        if len(selected_swaps) < swaps_per_row:
            payload = {
                **base,
                "status": "excluded",
                "exclusion_reason": "fewer_than_required_exact_visual_length_swaps",
                "required_swaps": swaps_per_row,
                "available_swaps": len(selected_swaps),
                "no_length_caliper_relaxation": True,
            }
        else:
            target = row["full_visible_answer"]
            own_trace = adapter.score(
                image_path=own_path,
                question=row["question"],
                target=target,
                condition="image",
            )
            swap_traces = [
                adapter.score(
                    image_path=path,
                    question=row["question"],
                    target=target,
                    condition="image",
                )
                for _, path in selected_swaps
            ]
            text_trace = adapter.score(
                image_path=None,
                question=row["question"],
                target=target,
                condition="text_only",
            )
            try:
                signals = compute_full_replay_signals(
                    row=row,
                    own_trace=own_trace,
                    swap_traces=swap_traces,
                    text_trace=text_trace,
                    own_image_sha256=_sha256(own_path),
                    swap_image_sha256=[_sha256(path) for _, path in selected_swaps],
                )
            except RowExclusion as exc:
                payload = {
                    **base,
                    "status": "excluded",
                    "exclusion_reason": str(exc),
                    "no_fallback_used": True,
                }
            else:
                payload = {
                    **base,
                    "status": "ok",
                    "selected_swaps": [
                        {
                            "case_id": candidate["case_id"],
                            "image_sha256": _sha256(path),
                            "visual_token_count": own_count,
                        }
                        for candidate, path in selected_swaps
                    ],
                    "signals": signals,
                }
        shard = {
            "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
            "config_fingerprint": config_fingerprint,
            "row_sha256": row_sha,
            "payload_sha256": _sha256_bytes(_canonical(payload)),
            "payload": payload,
        }
        _atomic_write(
            shard_path, (json.dumps(shard, indent=2, sort_keys=True) + "\n").encode()
        )
        payloads.append(payload)
    analyzable = sum(row.get("status") == "ok" for row in payloads)
    completion = {
        "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
        "status": "complete" if analyzable else "no_analyzable_rows",
        "config_fingerprint": config_fingerprint,
        "rows": len(payloads),
        "analyzable_rows": analyzable,
        "excluded_rows": len(payloads) - analyzable,
        "resumed_rows": resumed,
        "native_identity_canary_bound": True,
    }
    if not analyzable:
        raise ContractError("full replay produced no analyzable rows; COMPLETE not written")
    _write_once_or_equal(
        output_dir / "COMPLETE.json",
        (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode(),
    )
    return completion


def _load_factory(specification: str, config: dict[str, Any]) -> FullReplayAdapter:
    if ":" not in specification:
        raise ContractError("--adapter-factory must be module:function")
    module_name, function_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--identity-canary", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--swaps-per-row", type=int, default=2)
    parser.add_argument("--adapter-factory")
    parser.add_argument("--adapter-config", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        rows, _, _ = load_full_replay_manifest(
            args.manifest,
            args.metadata,
            identity_canary=args.identity_canary,
            require_identity=not args.preflight_only,
        )
        selected = [row for row in rows if row["split"] == args.split]
        for row in selected:
            _resolve_image(args.image_root, row["image_relpath"])
            for swap in row["swap_candidates"]:
                _resolve_image(args.image_root, swap["image_relpath"])
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "status": "preflight_passed",
                        "rows": len(selected),
                        "gpu_started": False,
                        "identity_canary_bound": args.identity_canary is not None,
                    },
                    indent=2,
                )
            )
            return
        if not args.identity_canary or not args.adapter_factory:
            raise ContractError(
                "scientific run requires --identity-canary and audited --adapter-factory"
            )
        adapter_config = (
            json.loads(args.adapter_config.read_text()) if args.adapter_config else {}
        )
        adapter = _load_factory(args.adapter_factory, adapter_config)
        result = run_full_replay(
            manifest=args.manifest,
            metadata=args.metadata,
            identity_canary=args.identity_canary,
            image_root=args.image_root,
            output_dir=args.output_dir,
            adapter=adapter,
            split=args.split,
            swaps_per_row=args.swaps_per_row,
            command=[shlex.join(sys.argv)],
        )
    except (ContractError, OSError, ValueError, ImportError, AttributeError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
