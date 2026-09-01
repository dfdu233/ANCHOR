#!/usr/bin/env python3
"""Crash-safe VinDr hidden-state collection with deterministic aggregation.

The model-facing path deliberately reuses v2's already-conformed post-block
hook and FP32 verbalizer readout.  V3 changes only artifact durability: every
case is committed as one atomic NPZ containing both tensors and metadata, and
the legacy aggregate files are rebuilt only after every shard validates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import transformers

from corrected_sgta.collect_vindr_hidden_states_v2 import (
    build_runtime,
    capture_post_block,
    diagnostic_logits,
    select_rows,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    atomic_json,
    load_image,
    load_jsonl,
    prompt_for,
    resolve_image,
    sha256_file,
)
from corrected_sgta.run_hulu_vindr_commitment_probe import model_file_inventory


VERSION = "vindr-unified-post-block-hidden-v3"
SHARD_SCHEMA = "vindr-hidden-case-shard-v1"
ROUTING_STATISTIC_NAMES = (
    "claim_visual_cosine_mean",
    "claim_visual_cosine_std",
    "claim_visual_cosine_max",
    "claim_visual_cosine_top16_mean",
    "claim_visual_alignment_entropy",
    "visual_token_norm_mean",
    "visual_token_norm_std",
)
EPHEMERAL_CONFIG_FIELDS = {"created_at", "command", "fingerprint"}
SHARD_FIELDS = {
    "claim",
    "visual_mean",
    "visual_std",
    "routing_statistics",
    "layers",
    "metadata_json",
    "schema_version",
    "record_key",
    "config_fingerprint",
    "ordered_index",
    "payload_sha256",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_ordered_keys(keys: Sequence[str]) -> list[str]:
    output = [str(key) for key in keys]
    if not output:
        raise ValueError("selected record-key order is empty")
    seen: set[str] = set()
    duplicates: list[str] = []
    for key in output:
        if not key:
            raise ValueError("record keys must be non-empty strings")
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"duplicate record keys: {duplicates[:5]}")
    return output


def ordered_keys_payload(keys: Sequence[str]) -> dict[str, object]:
    checked = validate_ordered_keys(keys)
    return {
        "version": VERSION,
        "n": len(checked),
        "record_keys": checked,
        "record_keys_sha256": object_sha256(checked),
    }


def static_config(config: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in config.items() if key not in EPHEMERAL_CONFIG_FIELDS}


def freeze_or_validate_run(
    output_dir: Path,
    candidate_static: Mapping[str, object],
    ordered_keys: Sequence[str],
    *,
    resume: bool,
    command: str,
) -> dict[str, object]:
    """Freeze or strictly validate one resumable run identity.

    Created time and the literal command are provenance only.  Every semantic
    field (including manifest/model/code hashes) and the full ordered key list
    must match exactly on resume.
    """

    candidate = dict(candidate_static)
    if EPHEMERAL_CONFIG_FIELDS.intersection(candidate):
        raise ValueError("candidate_static contains ephemeral config fields")
    candidate_fingerprint = object_sha256(candidate)
    expected_order = ordered_keys_payload(ordered_keys)
    config_path = output_dir / "config.json"
    order_path = output_dir / "ordered_keys.json"

    if not resume:
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        config: dict[str, object] = {
            **candidate,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "fingerprint": candidate_fingerprint,
        }
        atomic_json(config_path, config)
        atomic_json(order_path, expected_order)
        (output_dir / "shards").mkdir()
        return config

    if not output_dir.is_dir():
        raise FileNotFoundError("--resume requires the original output directory")
    if not config_path.is_file() or not order_path.is_file():
        raise FileNotFoundError("--resume requires config.json and ordered_keys.json")
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    existing_static = static_config(existing)
    stored_fingerprint = str(existing.get("fingerprint", ""))
    if not stored_fingerprint or stored_fingerprint != object_sha256(existing_static):
        raise ValueError("stored config fingerprint is absent or internally inconsistent")
    if existing_static != candidate:
        changed = sorted(
            key
            for key in set(existing_static) | set(candidate)
            if existing_static.get(key) != candidate.get(key)
        )
        raise ValueError(f"refusing resume after config drift: {changed}")
    if stored_fingerprint != candidate_fingerprint:
        raise ValueError("refusing resume after config fingerprint drift")

    existing_order = json.loads(order_path.read_text(encoding="utf-8"))
    stored_keys = existing_order.get("record_keys")
    if not isinstance(stored_keys, list):
        raise ValueError("ordered_keys.json has no ordered record-key list")
    validate_ordered_keys(stored_keys)
    if existing_order.get("record_keys_sha256") != object_sha256(stored_keys):
        raise ValueError("ordered_keys.json hash is internally inconsistent")
    if existing_order != expected_order:
        raise ValueError("refusing resume after ordered record-key drift")
    (output_dir / "shards").mkdir(exist_ok=True)
    return existing


def shard_path(shard_dir: Path, index: int, record_key: str) -> Path:
    suffix = hashlib.sha256(record_key.encode("utf-8")).hexdigest()[:16]
    return shard_dir / f"{index:06d}-{suffix}.npz"


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _array_digest(digest: Any, name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(name.encode("utf-8"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(canonical_json(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))


def shard_payload_sha256(
    *,
    claim: np.ndarray,
    visual_mean: np.ndarray,
    visual_std: np.ndarray,
    routing_statistics: np.ndarray,
    layers: np.ndarray,
    metadata: Mapping[str, object],
    record_key: str,
    config_fingerprint: str,
    ordered_index: int,
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("claim", claim),
        ("visual_mean", visual_mean),
        ("visual_std", visual_std),
        ("routing_statistics", routing_statistics),
        ("layers", layers),
    ):
        _array_digest(digest, name, value)
    digest.update(canonical_json(dict(metadata)).encode("utf-8"))
    digest.update(record_key.encode("utf-8"))
    digest.update(config_fingerprint.encode("ascii"))
    digest.update(str(int(ordered_index)).encode("ascii"))
    digest.update(SHARD_SCHEMA.encode("ascii"))
    return digest.hexdigest()


def write_case_shard(
    path: Path,
    *,
    index: int,
    record_key: str,
    config_fingerprint: str,
    layers: Sequence[int],
    features: Mapping[int, Mapping[str, np.ndarray]],
    metadata: Mapping[str, object],
) -> None:
    ordered_layers = sorted(set(int(layer) for layer in layers))
    if sorted(features) != ordered_layers:
        raise ValueError("feature layer set disagrees with configured layers")
    claim = np.asarray(
        np.stack([features[layer]["claim"] for layer in ordered_layers]), dtype=np.float16
    )
    visual_mean = np.asarray(
        np.stack([features[layer]["visual_mean"] for layer in ordered_layers]),
        dtype=np.float16,
    )
    visual_std = np.asarray(
        np.stack([features[layer]["visual_std"] for layer in ordered_layers]),
        dtype=np.float16,
    )
    routing = np.asarray(
        np.stack([features[layer]["routing_statistics"] for layer in ordered_layers]),
        dtype=np.float32,
    )
    layer_array = np.asarray(ordered_layers, dtype=np.int16)
    metadata_dict = dict(metadata)
    if metadata_dict.get("record_key") != record_key:
        raise ValueError("metadata record_key disagrees with shard record_key")
    payload_hash = shard_payload_sha256(
        claim=claim,
        visual_mean=visual_mean,
        visual_std=visual_std,
        routing_statistics=routing,
        layers=layer_array,
        metadata=metadata_dict,
        record_key=record_key,
        config_fingerprint=config_fingerprint,
        ordered_index=index,
    )
    atomic_npz(
        path,
        claim=claim,
        visual_mean=visual_mean,
        visual_std=visual_std,
        routing_statistics=routing,
        layers=layer_array,
        metadata_json=np.asarray(canonical_json(metadata_dict)),
        schema_version=np.asarray(SHARD_SCHEMA),
        record_key=np.asarray(record_key),
        config_fingerprint=np.asarray(config_fingerprint),
        ordered_index=np.asarray(index, dtype=np.int64),
        payload_sha256=np.asarray(payload_hash),
    )


def _scalar(archive: Any, field: str) -> object:
    value = archive[field]
    if value.shape != ():
        raise ValueError(f"shard field {field} must be scalar")
    return value.item()


def load_and_validate_shard(
    path: Path,
    *,
    expected_index: int,
    expected_key: str,
    expected_fingerprint: str,
    expected_layers: Sequence[int],
) -> dict[str, object]:
    """Load one shard, rejecting structural, identity, dtype, or hash damage."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != SHARD_FIELDS:
                raise ValueError(
                    f"field set mismatch: missing={sorted(SHARD_FIELDS-set(archive.files))}, "
                    f"extra={sorted(set(archive.files)-SHARD_FIELDS)}"
                )
            schema = str(_scalar(archive, "schema_version"))
            record_key = str(_scalar(archive, "record_key"))
            fingerprint = str(_scalar(archive, "config_fingerprint"))
            ordered_index = int(_scalar(archive, "ordered_index"))
            stored_payload_hash = str(_scalar(archive, "payload_sha256"))
            metadata = json.loads(str(_scalar(archive, "metadata_json")))
            arrays = {
                "claim": np.array(archive["claim"], copy=True),
                "visual_mean": np.array(archive["visual_mean"], copy=True),
                "visual_std": np.array(archive["visual_std"], copy=True),
                "routing_statistics": np.array(archive["routing_statistics"], copy=True),
                "layers": np.array(archive["layers"], copy=True),
            }
    except Exception as error:
        if isinstance(error, ValueError) and str(error).startswith("invalid shard"):
            raise
        raise ValueError(f"invalid shard {path}: {error}") from error

    expected_layer_array = np.asarray(sorted(set(expected_layers)), dtype=np.int16)
    failures = []
    if schema != SHARD_SCHEMA:
        failures.append("schema")
    if record_key != expected_key:
        failures.append("record_key")
    if fingerprint != expected_fingerprint:
        failures.append("config_fingerprint")
    if ordered_index != expected_index:
        failures.append("ordered_index")
    if not isinstance(metadata, dict) or metadata.get("record_key") != expected_key:
        failures.append("metadata_record_key")
    if arrays["layers"].dtype != np.int16 or not np.array_equal(
        arrays["layers"], expected_layer_array
    ):
        failures.append("layers")
    layer_count = len(expected_layer_array)
    claim = arrays["claim"]
    if claim.dtype != np.float16 or claim.ndim != 2 or claim.shape[0] != layer_count:
        failures.append("claim_shape_or_dtype")
    for name in ("visual_mean", "visual_std"):
        value = arrays[name]
        if value.dtype != np.float16 or value.shape != claim.shape:
            failures.append(f"{name}_shape_or_dtype")
    routing = arrays["routing_statistics"]
    if routing.dtype != np.float32 or routing.shape != (
        layer_count,
        len(ROUTING_STATISTIC_NAMES),
    ):
        failures.append("routing_shape_or_dtype")
    if not failures:
        computed_hash = shard_payload_sha256(
            claim=claim,
            visual_mean=arrays["visual_mean"],
            visual_std=arrays["visual_std"],
            routing_statistics=routing,
            layers=arrays["layers"],
            metadata=metadata,
            record_key=record_key,
            config_fingerprint=fingerprint,
            ordered_index=ordered_index,
        )
        if computed_hash != stored_payload_hash:
            failures.append("payload_sha256")
    if failures:
        raise ValueError(f"invalid shard {path}: {', '.join(failures)}")
    return {**arrays, "metadata": metadata}


def validate_shard_set(
    output_dir: Path,
    ordered_keys: Sequence[str],
    config_fingerprint: str,
    layers: Sequence[int],
) -> list[int]:
    """Validate every present shard and return only genuinely missing indices."""

    checked_keys = validate_ordered_keys(ordered_keys)
    shard_dir = output_dir / "shards"
    expected = {
        shard_path(shard_dir, index, key): (index, key)
        for index, key in enumerate(checked_keys)
    }
    actual = set(shard_dir.glob("*.npz")) if shard_dir.is_dir() else set()
    unexpected = sorted(str(path) for path in actual - set(expected))
    if unexpected:
        raise ValueError(f"unexpected shard files (possible key drift): {unexpected[:5]}")
    missing: list[int] = []
    for path, (index, key) in expected.items():
        if not path.is_file():
            missing.append(index)
            continue
        load_and_validate_shard(
            path,
            expected_index=index,
            expected_key=key,
            expected_fingerprint=config_fingerprint,
            expected_layers=layers,
        )
    return missing


def aggregate_shards(
    output_dir: Path,
    ordered_keys: Sequence[str],
    config_fingerprint: str,
    layers: Sequence[int],
) -> list[dict[str, object]]:
    """Deterministically rebuild the v2-compatible aggregate artifacts."""

    missing = validate_shard_set(output_dir, ordered_keys, config_fingerprint, layers)
    if missing:
        raise FileNotFoundError(
            f"refusing partial aggregation: {len(missing)} shards are missing; "
            f"first indices={missing[:5]}"
        )
    loaded = []
    shard_dir = output_dir / "shards"
    for index, key in enumerate(ordered_keys):
        loaded.append(
            load_and_validate_shard(
                shard_path(shard_dir, index, key),
                expected_index=index,
                expected_key=key,
                expected_fingerprint=config_fingerprint,
                expected_layers=layers,
            )
        )
    atomic_npz(
        output_dir / "hidden_states.npz",
        claim=np.stack([row["claim"] for row in loaded]),
        visual_mean=np.stack([row["visual_mean"] for row in loaded]),
        visual_std=np.stack([row["visual_std"] for row in loaded]),
        routing_statistics=np.stack([row["routing_statistics"] for row in loaded]),
        routing_statistic_names=np.asarray(ROUTING_STATISTIC_NAMES),
        layers=np.asarray(sorted(set(layers)), dtype=np.int16),
    )
    metadata = [row["metadata"] for row in loaded]
    atomic_text(
        output_dir / "metadata.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata),
    )
    return metadata


def record_key(row: Mapping[str, object]) -> str:
    return f"{row['finding']}:{row['image_id']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("pilot", "dev", "confirmation"), required=True)
    parser.add_argument("--findings", nargs="+", required=True)
    parser.add_argument("--votes", nargs="+", type=int, choices=range(4), required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = select_rows(
        load_jsonl(args.manifest),
        args.split,
        set(args.findings),
        set(args.votes),
        args.seed,
        args.max_samples,
    )
    ordered_keys = validate_ordered_keys([record_key(row) for row in rows])
    v2_source = Path(__file__).with_name("collect_vindr_hidden_states_v2.py")
    candidate_config: dict[str, object] = {
        "version": VERSION,
        "model_id": args.model,
        "model_dir": str(args.model_dir.resolve()),
        "model_inventory": model_file_inventory(args.model_dir),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "image_root": str(args.image_root.resolve()),
        "split": args.split,
        "findings": sorted(set(args.findings)),
        "votes": sorted(set(args.votes)),
        "layers": sorted(set(args.layers)),
        "max_visual_tokens": args.max_visual_tokens,
        "max_samples": args.max_samples,
        "representation_location": "post_decoder_block_pre_final_norm",
        "plain_logit_lens_role": "diagnostic_only",
        "verbalizer_readout_accumulation": "float32",
        "seed": args.seed,
        "record_keys_sha256": object_sha256(ordered_keys),
        "code_sha256": sha256_file(Path(__file__)),
        "v2_hook_and_readout_code_sha256": sha256_file(v2_source),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
    }
    config = freeze_or_validate_run(
        args.output_dir,
        candidate_config,
        ordered_keys,
        resume=args.resume,
        command=" ".join(sys.argv),
    )
    fingerprint = str(config["fingerprint"])
    pending = validate_shard_set(
        args.output_dir, ordered_keys, fingerprint, args.layers
    )
    started = time.perf_counter()
    if pending:
        runtime, prepare = build_runtime(args)
        try:
            for progress, index in enumerate(pending, start=1):
                row = rows[index]
                key = ordered_keys[index]
                case_started = time.perf_counter()
                path = resolve_image(row, args.image_root)
                image = load_image(path)
                prompt = str(row.get("question") or prompt_for(str(row["finding"])))
                embeddings, attention, positions, span = prepare(runtime, prompt, image)
                features, hook_conformance = capture_post_block(
                    runtime, embeddings, attention, positions, span, args.layers
                )
                # diagnostic_logits keeps native hidden trajectories but performs
                # final-norm/unembedding accumulation in FP32 (inherited from v2).
                diagnostic = diagnostic_logits(runtime, features)
                metadata = {
                    "record_key": key,
                    "image_id": row["image_id"],
                    "finding": row["finding"],
                    "positive_votes": row["positive_votes"],
                    "reader_votes": row["reader_votes"],
                    "reader_support": row["reader_support"],
                    "reader_state": row["reader_state"],
                    "experiment_split": row["experiment_split"],
                    "visual_tokens": span[1] - span[0],
                    "diagnostic_plain_logit_lens": diagnostic,
                    "hook_conformance": hook_conformance,
                    "elapsed_seconds": time.perf_counter() - case_started,
                }
                destination = shard_path(args.output_dir / "shards", index, key)
                write_case_shard(
                    destination,
                    index=index,
                    record_key=key,
                    config_fingerprint=fingerprint,
                    layers=args.layers,
                    features=features,
                    metadata=metadata,
                )
                # Read-after-write catches storage corruption while the case is
                # still attributable, instead of discovering it at aggregation.
                load_and_validate_shard(
                    destination,
                    expected_index=index,
                    expected_key=key,
                    expected_fingerprint=fingerprint,
                    expected_layers=args.layers,
                )
                print(
                    json.dumps(
                        {
                            "progress": f"{progress}/{len(pending)} pending "
                            f"({index + 1}/{len(rows)} ordered)",
                            "record_key": key,
                        }
                    ),
                    flush=True,
                )
        finally:
            del runtime
            torch.cuda.empty_cache()

    metadata = aggregate_shards(
        args.output_dir, ordered_keys, fingerprint, args.layers
    )
    metadata_path = args.output_dir / "metadata.jsonl"
    hidden_path = args.output_dir / "hidden_states.npz"
    elapsed = time.perf_counter() - started
    durations = [float(row["elapsed_seconds"]) for row in metadata]
    atomic_json(
        args.output_dir / "summary.json",
        {
            "status": "complete",
            "n": len(metadata),
            "elapsed_seconds": float(sum(durations)),
            "elapsed_seconds_this_invocation": elapsed,
            "seconds_per_case_median": float(np.median(durations)),
            "seconds_per_case_p90": float(np.quantile(durations, 0.9)),
            "hidden_states_sha256": sha256_file(hidden_path),
            "metadata_sha256": sha256_file(metadata_path),
            "config_fingerprint": fingerprint,
            "record_keys_sha256": object_sha256(ordered_keys),
            "case_shards_validated": len(metadata),
        },
    )


if __name__ == "__main__":
    main()
