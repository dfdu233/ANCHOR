#!/usr/bin/env python3
"""Crash-safe capture of development-frozen finding scores for every visual patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from corrected_sgta.collect_vindr_hidden_states_v2 import build_runtime
from corrected_sgta.halp_three_plane_preflight_v1 import PreProjectorVisionCapture
from corrected_sgta.run_huatuo_vindr_commitment_probe import load_image, prompt_for, sha256_file
from corrected_sgta.run_hulu_vindr_commitment_probe import model_file_inventory


VERSION = "sparse-patch-score-collector-v1"


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def infer_grid(tokens: int) -> tuple[int, int]:
    candidates = []
    for groups in range(1, 9):
        side = int(round((tokens / groups) ** 0.5))
        if groups * side * side == tokens:
            candidates.append((groups, side))
    if not candidates:
        raise ValueError(f"cannot express {tokens} patches as <=8 square grids")
    return min(candidates, key=lambda value: value[0])


def shard_path(root: Path, index: int, image_id: str) -> Path:
    suffix = hashlib.sha256(image_id.encode()).hexdigest()[:16]
    return root / "shards" / f"{index:06d}-{suffix}.npz"


def load_ordered_images(raw_visual: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (raw_visual / "metadata.jsonl").read_text().splitlines()]
    rows.sort(key=lambda row: int(row["ordered_index"]))
    if [int(row["ordered_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("raw visual metadata is not a complete ordered sequence")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("raw visual metadata repeats image ids")
    return rows


def aggregate(args: argparse.Namespace, rows: list[dict[str, Any]], fingerprint: str) -> None:
    scores, metadata = [], []
    shape = None
    for index, row in enumerate(rows):
        path = shard_path(args.output_dir, index, row["image_id"])
        with np.load(path, allow_pickle=False) as archive:
            if (
                int(archive["ordered_index"].item()) != index
                or str(archive["image_id"].item()) != row["image_id"]
                or str(archive["fingerprint"].item()) != fingerprint
            ):
                raise ValueError(f"shard identity drift: {path}")
            value = np.asarray(archive["patch_scores"], dtype=np.float32)
            item = json.loads(str(archive["metadata_json"].item()))
        if shape is None:
            shape = value.shape
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"patch score shape/nonfinite drift: {path}")
        scores.append(value)
        metadata.append(item)
    atomic_npz(args.output_dir / "patch_scores.npz", patch_scores=np.stack(scores))
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata)
    temporary = args.output_dir / f".metadata.jsonl.{os.getpid()}.partial"
    temporary.write_text(text)
    os.replace(temporary, args.output_dir / "metadata.jsonl")
    atomic_json(
        args.output_dir / "summary.json",
        {
            "status": "complete",
            "version": VERSION,
            "model": args.model,
            "n_images": len(rows),
            "shape": list(np.stack(scores).shape),
            "fingerprint": fingerprint,
            "patch_scores_sha256": sha256_file(args.output_dir / "patch_scores.npz"),
            "metadata_sha256": sha256_file(args.output_dir / "metadata.jsonl"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu"), required=True)
    parser.add_argument("--raw-visual", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = load_ordered_images(args.raw_visual)
    directions_archive = np.load(args.directions)
    directions = np.asarray(directions_archive["directions"], dtype=np.float32)
    findings = [str(value) for value in directions_archive["findings"].tolist()]
    static = {
        "version": VERSION,
        "model": args.model,
        "model_dir": str(args.model_dir.resolve()),
        "model_inventory": model_file_inventory(args.model_dir),
        "raw_visual": str(args.raw_visual.resolve()),
        "raw_metadata_sha256": sha256_file(args.raw_visual / "metadata.jsonl"),
        "raw_prompt_invariance_canary_sha256": sha256_file(args.raw_visual / "prompt_invariance_canary.json"),
        "directions": str(args.directions.resolve()),
        "directions_sha256": sha256_file(args.directions),
        "findings": findings,
        "direction_shape": list(directions.shape),
        "image_root": str(args.image_root.resolve()),
        "ordered_images_sha256": canonical_hash([(row["image_id"], row["split"], row["dicom_sha256"]) for row in rows]),
        "max_visual_tokens": args.max_visual_tokens,
        "source_sha256": sha256_file(Path(__file__)),
    }
    fingerprint = canonical_hash(static)
    config_path = args.output_dir / "config.json"
    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError(args.output_dir)
        existing = json.loads(config_path.read_text())
        if existing.get("fingerprint") != fingerprint or existing.get("static") != static:
            raise ValueError("resume configuration drift")
    else:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "shards").mkdir()
        atomic_json(
            config_path,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
                "fingerprint": fingerprint,
                "static": static,
            },
        )
    completed = 0
    for index, row in enumerate(rows):
        path = shard_path(args.output_dir, index, row["image_id"])
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["fingerprint"].item()) != fingerprint:
                    raise ValueError(f"stale shard {path}")
            completed += 1
    if completed == len(rows):
        aggregate(args, rows, fingerprint)
        print(json.dumps({"status": "already_complete", "n": len(rows)}))
        return

    runtime, prepare = build_runtime(args)
    vision = runtime.model.get_vision_tower() if args.model == "huatuo" else runtime.model.get_vision_encoder()
    started = time.perf_counter()
    for index, row in enumerate(rows):
        path = shard_path(args.output_dir, index, row["image_id"])
        if path.is_file():
            continue
        image_path = args.image_root / f"{row['image_id']}.dicom"
        if sha256_file(image_path) != row["dicom_sha256"]:
            raise ValueError(f"DICOM content drift: {image_path}")
        image = load_image(image_path)
        with PreProjectorVisionCapture(vision) as capture:
            prepare(runtime, prompt_for(findings[0]), image)
        tokens = capture.one()
        if tokens.ndim == 3:
            tokens = tokens[0]
        values = tokens.detach().float().cpu().numpy()
        if values.ndim != 2 or values.shape[1] != directions.shape[1]:
            raise ValueError(f"visual token shape mismatch: {values.shape}")
        groups, side = infer_grid(values.shape[0])
        patch_scores = values @ directions.T
        metadata = {
            "ordered_index": index,
            "image_id": row["image_id"],
            "split": row["split"],
            "findings": row["findings"],
            "patch_tokens": int(values.shape[0]),
            "grid_groups": groups,
            "grid_side": side,
            "dicom_sha256": row["dicom_sha256"],
        }
        atomic_npz(
            path,
            patch_scores=patch_scores.astype(np.float32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            ordered_index=np.asarray(index),
            image_id=np.asarray(row["image_id"]),
            fingerprint=np.asarray(fingerprint),
        )
        completed += 1
        if completed % args.progress_every == 0 or completed == len(rows):
            print(json.dumps({"model": args.model, "completed": completed, "total": len(rows), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    aggregate(args, rows, fingerprint)
    print(json.dumps({"status": "complete", "n": len(rows)}))


if __name__ == "__main__":
    main()
