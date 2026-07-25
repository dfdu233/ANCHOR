"""Diagnose local external-source support at influential LLaVA image patches.

This command is deliberately read-only with respect to model predictions: it
extracts the frozen model's projected visual tokens and compares them with
precomputed spherical prototypes.  Distances are descriptive diagnostics, not
an anomaly score or a causal effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from corrected_sgta.infer_rule_dg_adapter import repair_jsonl_tail
from corrected_sgta.models_local_source import LlavaLocalSourceAdapter


VERSION = "local-external-source-support-v1"
ALLOWED_SOURCES = ("mimic_cxr_leaksafe", "pubmedvision_xray_formal")
INFLUENCE_FIELD = (
    "trace",
    "prompt_boundary_semantic_margin",
    "absolute_gradient_x_activation",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def nearest_cosine_distances(
    tokens: np.ndarray, prototypes: np.ndarray
) -> np.ndarray:
    """Return one-minus the maximum prototype cosine similarity per patch."""
    token_unit = unit_rows(tokens)
    prototype_unit = unit_rows(prototypes)
    return (1.0 - np.max(token_unit @ prototype_unit.T, axis=1)).astype(np.float64)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with average rank for ties, using zero-based ranks."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    cursor = 0
    while cursor < len(array):
        end = cursor + 1
        while end < len(array) and array[order[end]] == array[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0
        cursor = end
    return ranks


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    a = average_ranks(np.asarray(values_a, dtype=np.float64))
    b = average_ranks(np.asarray(values_b, dtype=np.float64))
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize_distances(
    distances: np.ndarray, influence: np.ndarray, top_fraction: float = 0.10
) -> dict[str, float | int | None]:
    distances = np.asarray(distances, dtype=np.float64)
    influence = np.asarray(influence, dtype=np.float64)
    if distances.ndim != 1 or influence.ndim != 1 or len(distances) != len(influence):
        raise ValueError("distance and influence must be aligned one-dimensional arrays")
    if len(distances) == 0:
        raise ValueError("cannot summarize empty patch arrays")
    if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(influence)):
        raise ValueError("distance and influence must be finite")
    if np.any(influence < 0):
        raise ValueError("absolute influence must be non-negative")
    weight_sum = float(influence.sum())
    weighted = (
        float(np.dot(distances, influence) / weight_sum)
        if weight_sum > 0.0
        else None
    )
    top_count = max(1, int(np.ceil(len(influence) * top_fraction)))
    top_indices = np.argsort(-influence, kind="mergesort")[:top_count]
    return {
        "patch_count": int(len(distances)),
        "unweighted_mean_distance": float(distances.mean()),
        "influence_weighted_distance": weighted,
        "top10pct_influence_patch_count": int(top_count),
        "top10pct_influence_patch_distance": float(distances[top_indices].mean()),
        "spearman_distance_influence": spearman(distances, influence),
    }


def load_trace_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id"))
            if qid in rows:
                raise ValueError(f"duplicate trace qid: {qid}")
            rows[qid] = row
    return rows


def load_external_prototypes(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(meta_path.read_text())
    payload = np.load(path, allow_pickle=False)
    entries = {entry["source_id"]: entry for entry in metadata.get("entries", [])}
    missing = set(ALLOWED_SOURCES) - set(entries)
    if missing:
        raise ValueError(f"required external prototypes are missing: {sorted(missing)}")
    prototypes = {
        source_id: payload[entries[source_id]["array_key"]].astype(np.float32)
        for source_id in ALLOWED_SOURCES
    }
    # Selecting only this explicit allow-list is a protocol constraint.  In
    # particular, an IU-Xray prototype present in the archive is never read.
    return prototypes, metadata


def completed(path: Path, fingerprint: str) -> set[str]:
    if not path.is_file():
        return set()
    result = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("fingerprint") == fingerprint and row.get("status") == "ok":
                result.add(str(row["question_id"]))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--local-prototypes", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qid", action="append", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = sorted(set(map(str, args.qid)), key=int)
    trace_rows = load_trace_rows(args.trace)
    missing = set(requested) - set(trace_rows)
    if missing:
        raise ValueError(f"qids missing from trace: {sorted(missing, key=int)}")
    prototypes, prototype_metadata = load_external_prototypes(args.local_prototypes)
    prototype_meta_path = args.local_prototypes.with_suffix(
        args.local_prototypes.suffix + ".meta.json"
    )
    metadata = {
        "version": VERSION,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "trace": str(args.trace.resolve()),
        "trace_sha256": sha256_file(args.trace),
        "trace_meta_sha256": sha256_file(
            args.trace.with_suffix(args.trace.suffix + ".meta.json")
        ),
        "local_prototypes": str(args.local_prototypes.resolve()),
        "local_prototypes_sha256": sha256_file(args.local_prototypes),
        "local_prototypes_meta_sha256": sha256_file(prototype_meta_path),
        "image_root": str(args.image_root.resolve()),
        "explicit_qids": requested,
        "source_ids": list(ALLOWED_SOURCES),
        "source_array_keys": {
            entry["source_id"]: entry["array_key"]
            for entry in prototype_metadata["entries"]
            if entry["source_id"] in ALLOWED_SOURCES
        },
        "distance": "per-patch nearest spherical-prototype cosine distance",
        "influence": ".".join(INFLUENCE_FIELD),
        "top_fraction": 0.10,
        "intervention": False,
        "interpretation": (
            "descriptive local support only; distance is neither an anomaly "
            "label nor a causal quantity"
        ),
    }
    fingerprint_payload = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode()
    metadata["fingerprint"] = hashlib.sha256(fingerprint_payload).hexdigest()
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not meta_path.is_file() or json.loads(meta_path.read_text()) != metadata:
            raise RuntimeError("resume metadata mismatch; refusing to mix protocols")
        repair_jsonl_tail(args.output)
    else:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    done = completed(args.output, metadata["fingerprint"]) if args.resume else set()
    mode = "a" if args.resume else "w"
    adapter = LlavaLocalSourceAdapter()
    try:
        with args.output.open(mode) as handle:
            for qid in requested:
                if qid in done:
                    continue
                trace = trace_rows[qid]
                record: dict[str, Any] = {
                    "version": VERSION,
                    "question_id": qid,
                    "image": trace.get("image"),
                    "fingerprint": metadata["fingerprint"],
                    "status": "error",
                }
                try:
                    if trace.get("status") != "ok":
                        raise ValueError("trace row is not successful")
                    influence = np.asarray(
                        trace["trace"]["prompt_boundary_semantic_margin"][
                            "absolute_gradient_x_activation"
                        ],
                        dtype=np.float64,
                    )
                    with Image.open(args.image_root / trace["image"]) as source:
                        image = source.convert("RGB")
                    tokens = adapter.visual_tokens([image])[0].astype(np.float32)
                    if tokens.shape != (576, 4096):
                        raise ValueError(
                            f"expected projected tokens [576,4096], got {tokens.shape}"
                        )
                    if len(influence) != len(tokens):
                        raise ValueError(
                            f"influence/token mismatch: {len(influence)} vs {len(tokens)}"
                        )
                    per_source_distance = {
                        source_id: nearest_cosine_distances(tokens, proto)
                        for source_id, proto in prototypes.items()
                    }
                    external_distance = np.minimum.reduce(
                        list(per_source_distance.values())
                    )
                    record.update(
                        {
                            "status": "ok",
                            "projected_token_shape": list(tokens.shape),
                            "source_statistics": {
                                source_id: summarize_distances(distance, influence)
                                for source_id, distance in per_source_distance.items()
                            },
                            "external_union_statistics": summarize_distances(
                                external_distance, influence
                            ),
                            "influence_summary": {
                                "sum": float(influence.sum()),
                                "mean": float(influence.mean()),
                                "max": float(influence.max()),
                                "nonzero_patch_count": int(np.count_nonzero(influence)),
                            },
                            "interpretation": metadata["interpretation"],
                        }
                    )
                except Exception as error:
                    record.update(
                        {"error_type": type(error).__name__, "error": str(error)}
                    )
                handle.write(json.dumps(record) + "\n")
                handle.flush()
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
