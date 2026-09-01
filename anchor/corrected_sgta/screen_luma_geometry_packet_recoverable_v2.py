#!/usr/bin/env python3
"""Recoverable CPU execution for the luma geometry packet fatal screen.

This preserves the scientific definition and analysis of
``screen_luma_geometry_packet_v1.py``.  The only changes are operational:

* image features are written atomically after every image batch;
* completed batches are validated and reused on restart;
* all views in a batch are packed into larger tower calls to reduce CPU
  dispatcher overhead;
* the final result is also written atomically.

No CUDA device is used or inspected by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anchor.corrected_sgta.screen_external_visual_increment_v1 import (
    load_claims,
    sha256_file,
)
from anchor.corrected_sgta.screen_luma_geometry_packet_v1 import (
    CHROMA_BASIS,
    DISPLAY,
    FINDINGS,
    LUMA,
    VIEWS,
    BiomedTower,
    balanced_cap,
    bootstrap,
    choose_c,
    construct_views,
    float_window,
    haar_packet,
    macro_auc,
    probe,
    riesz_packet,
)


VERSION = "luma-geometry-packet-biomedclip-recoverable-v2"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_components_one(image_id: str, image_root: Path, side: int):
    x = float_window(image_root / f"{image_id}.dicom", side)
    low, horizontal, vertical, diagonal = haar_packet(x)
    riesz_x, riesz_y = riesz_packet(low)
    return {
        "low": low,
        "haar_h": horizontal,
        "haar_v": vertical,
        "haar_d": diagonal,
        "riesz_x": riesz_x,
        "riesz_y": riesz_y,
    }


def load_valid_chunk(
    path: Path,
    expected_ids: list[str],
    expected_fingerprint: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            stored_fingerprint = str(cache["fingerprint"].item())
            stored_ids = [str(value) for value in cache["image_ids"].tolist()]
            if stored_fingerprint != expected_fingerprint or stored_ids != expected_ids:
                raise ValueError("cache identity mismatch")
            features = {
                view: np.asarray(cache[f"feature_{view}"], dtype=np.float32)
                for view in VIEWS
            }
            expected_rows = len(expected_ids)
            widths = {matrix.shape[1] for matrix in features.values()}
            if any(matrix.ndim != 2 or matrix.shape[0] != expected_rows for matrix in features.values()):
                raise ValueError("cache feature shape mismatch")
            if len(widths) != 1:
                raise ValueError("cache feature width mismatch")
            diagnostics = json.loads(str(cache["diagnostics_json"].item()))
            if len(diagnostics) != expected_rows:
                raise ValueError("cache diagnostics length mismatch")
            return features, diagnostics
    except Exception as error:
        quarantine = path.with_suffix(path.suffix + f".invalid.{int(time.time())}")
        os.replace(path, quarantine)
        print(
            f"[cache] quarantined invalid {path.name}: {type(error).__name__}: {error}",
            flush=True,
        )
        return None


def embed_chunk(
    tower: BiomedTower,
    current_ids: list[str],
    cross: dict[str, str],
    image_root: Path,
    side: int,
    amplitude: float,
    seed: int,
    tower_batch_size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    # Only keep source components needed by this image batch.  The cross-image
    # placebo can refer outside it, so compute those partners lazily as well.
    needed = set(current_ids) | {cross[image_id] for image_id in current_ids}
    components = {
        image_id: make_components_one(image_id, image_root, side)
        for image_id in sorted(needed)
    }
    images = {view: [] for view in VIEWS}
    diagnostics = []
    for image_id in current_ids:
        views, record = construct_views(
            image_id,
            cross[image_id],
            components,
            amplitude,
            seed,
        )
        diagnostics.append(record)
        for view in VIEWS:
            images[view].append(views[view])

    # View-major packing preserves the exact row order while amortizing tower
    # dispatch.  Split only for bounded peak memory.
    packed = [image for view in VIEWS for image in images[view]]
    pieces = []
    for start in range(0, len(packed), tower_batch_size):
        pieces.append(tower.image(packed[start : start + tower_batch_size]))
    matrix = np.concatenate(pieces, axis=0)
    count = len(current_ids)
    return {
        view: matrix[index * count : (index + 1) * count]
        for index, view in enumerate(VIEWS)
    }, diagnostics


def summarize_diagnostics(records: list[dict[str, dict[str, float]]]):
    output = {}
    for view in records[0]:
        output[view] = {}
        for key in records[0][view]:
            values = [record[view][key] for record in records]
            output[view][key] = float(max(values) if key == "max_abs_luma_error" else np.mean(values))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--biomedclip-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--side", type=int, default=224)
    parser.add_argument("--amplitude", type=float, default=0.06)
    parser.add_argument("--per-finding-dev", type=int, default=40)
    parser.add_argument("--per-finding-confirmation", type=int, default=40)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--tower-batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("", "-1"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' or '-1'; this screen must not use GPU")
    if args.image_batch_size <= 0 or args.tower_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = balanced_cap(
        [row for row in load_claims(args.dev, "dev", "label") if row["finding"] in FINDINGS],
        args.per_finding_dev,
        args.seed,
    )
    confirmation = balanced_cap(
        [
            row
            for row in load_claims(args.confirmation, "confirmation", "label")
            if row["finding"] in FINDINGS
        ],
        args.per_finding_confirmation,
        args.seed + 1,
    )
    dev_images = {row["image_id"] for row in dev}
    confirmation_images = {row["image_id"] for row in confirmation}
    overlap = dev_images & confirmation_images
    if overlap:
        raise ValueError(f"Development/confirmation image leakage: {len(overlap)} images")
    image_ids = sorted(dev_images | confirmation_images)
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    for row in dev + confirmation:
        row["image_index"] = image_index[row["image_id"]]
    shifted = image_ids[1:] + image_ids[:1]
    cross = dict(zip(image_ids, shifted))

    run_identity = {
        "version": VERSION,
        "dev_sha256": sha256_file(args.dev),
        "confirmation_sha256": sha256_file(args.confirmation),
        "weights_sha256": sha256_file(args.biomedclip_root / "open_clip_pytorch_model.bin"),
        "image_ids": image_ids,
        "side": args.side,
        "amplitude": args.amplitude,
        "seed": args.seed,
        "views": list(VIEWS),
    }
    run_fingerprint = fingerprint(run_identity)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_dir / "manifest.json"
    manifest = {
        **run_identity,
        "fingerprint": run_fingerprint,
        "image_batch_size": args.image_batch_size,
        "tower_batch_size": args.tower_batch_size,
        "threads": args.threads,
    }
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior["fingerprint"] != run_fingerprint:
            raise ValueError("cache directory belongs to a different scientific run")
    else:
        atomic_json(manifest_path, manifest)

    tower = BiomedTower(args.biomedclip_root, args.threads)
    text = tower.text_directions()
    feature_parts = {view: [] for view in VIEWS}
    diagnostic_records: list[dict[str, Any]] = []
    started = time.time()
    total_batches = (len(image_ids) + args.image_batch_size - 1) // args.image_batch_size
    for batch_number, start in enumerate(range(0, len(image_ids), args.image_batch_size)):
        current_ids = image_ids[start : start + args.image_batch_size]
        chunk_path = args.cache_dir / f"batch_{batch_number:05d}.npz"
        cached = load_valid_chunk(chunk_path, current_ids, run_fingerprint)
        if cached is None:
            chunk_features, chunk_diagnostics = embed_chunk(
                tower=tower,
                current_ids=current_ids,
                cross=cross,
                image_root=args.image_root,
                side=args.side,
                amplitude=args.amplitude,
                seed=args.seed,
                tower_batch_size=args.tower_batch_size,
            )
            atomic_npz(
                chunk_path,
                fingerprint=np.asarray(run_fingerprint),
                image_ids=np.asarray(current_ids),
                diagnostics_json=np.asarray(json.dumps(chunk_diagnostics, sort_keys=True)),
                **{f"feature_{view}": matrix for view, matrix in chunk_features.items()},
            )
            state = "computed"
        else:
            chunk_features, chunk_diagnostics = cached
            state = "reused"
        for view in VIEWS:
            feature_parts[view].append(chunk_features[view])
        diagnostic_records.extend(chunk_diagnostics)
        elapsed = time.time() - started
        print(
            f"[features] {batch_number + 1}/{total_batches} {state}; "
            f"images={min(start + len(current_ids), len(image_ids))}/{len(image_ids)} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    features = {view: np.concatenate(parts) for view, parts in feature_parts.items()}
    if any(matrix.shape[0] != len(image_ids) for matrix in features.values()):
        raise AssertionError("concatenated cache row count mismatch")

    base_c, cv = choose_c(dev, features["base_float"], args.seed)
    probe_scores = {
        view: probe(dev, confirmation, matrix, base_c, args.seed)
        for view, matrix in features.items()
    }
    for view in ("haar_true", "riesz_true"):
        combined = np.concatenate([features["base_float"], features[view]], axis=1)
        probe_scores[f"base_plus_{view}"] = probe(
            dev,
            confirmation,
            combined,
            base_c / 2.0,
            args.seed,
        )

    finding_index = {finding: index for index, finding in enumerate(FINDINGS)}
    zero_shot_scores = {}
    for view, matrix in features.items():
        all_finding = matrix @ text.T
        zero_shot_scores[view] = np.asarray(
            [
                all_finding[row["image_index"], finding_index[row["finding"]]]
                for row in confirmation
            ]
        )
    probe_auc = {view: macro_auc(confirmation, score) for view, score in probe_scores.items()}
    zero_shot_auc = {
        view: macro_auc(confirmation, score) for view, score in zero_shot_scores.items()
    }
    comparisons = {
        "base_float_minus_uint8": ("base_float", "base_uint8"),
        "haar_true_minus_base_float": ("haar_true", "base_float"),
        "haar_true_minus_shuffle": ("haar_true", "haar_shuffle"),
        "haar_true_minus_cross": ("haar_true", "haar_cross"),
        "riesz_true_minus_base_float": ("riesz_true", "base_float"),
        "riesz_true_minus_shuffle": ("riesz_true", "riesz_shuffle"),
        "riesz_true_minus_rotated": ("riesz_true", "riesz_rotated"),
        "haar_true_minus_noise": ("haar_true", "equal_energy_noise"),
        "riesz_true_minus_noise": ("riesz_true", "equal_energy_noise"),
        "base_plus_haar_minus_base": ("base_plus_haar_true", "base_float"),
        "base_plus_riesz_minus_base": ("base_plus_riesz_true", "base_float"),
    }
    boot = bootstrap(confirmation, probe_scores, comparisons, args.bootstrap_draws, args.seed)

    def gate(family: str) -> bool:
        required = [
            f"{family}_true_minus_base_float",
            f"{family}_true_minus_shuffle",
            f"{family}_true_minus_noise",
            f"base_plus_{family}_minus_base",
        ]
        required.append("haar_true_minus_cross" if family == "haar" else "riesz_true_minus_rotated")
        return bool(
            probe_auc[f"{family}_true"] - probe_auc["base_float"] >= 0.02
            and probe_auc[f"base_plus_{family}_true"] - probe_auc["base_float"] >= 0.02
            and all(boot[name]["ci95"][0] > 0 for name in required)
        )

    family_gate = {family: gate(family) for family in ("haar", "riesz")}
    result = {
        "version": VERSION,
        "status": "complete_cpu_fatal_screen",
        "decision": "PASS_L0" if any(family_gate.values()) else "NO_GO_L0",
        "family_gate": family_gate,
        "scope": (
            "Frozen BiomedCLIP accessibility/information screen only. A pass does not "
            "establish VLM generation mitigation, non-target preservation, or novelty."
        ),
        "command": shlex.join(sys.argv),
        "run_fingerprint": run_fingerprint,
        "cache_dir": str(args.cache_dir.resolve()),
        "n_dev_claims": len(dev),
        "n_confirmation_claims": len(confirmation),
        "n_unique_images": len(image_ids),
        "n_dev_images": len(dev_images),
        "n_confirmation_images": len(confirmation_images),
        "findings": list(FINDINGS),
        "views": list(VIEWS),
        "biomedclip": tower.provenance,
        "input_sha256": {
            "dev": sha256_file(args.dev),
            "confirmation": sha256_file(args.confirmation),
        },
        "encoding": {
            "source_side": 2 * args.side,
            "target_side": args.side,
            "amplitude": args.amplitude,
            "luma": LUMA.tolist(),
            "chroma_basis": CHROMA_BASIS.tolist(),
            "basis_orthogonality_error": float(
                np.max(np.abs(CHROMA_BASIS.T @ CHROMA_BASIS - np.eye(2)))
            ),
            "basis_luma_null_error": float(np.max(np.abs(LUMA @ CHROMA_BASIS))),
            "haar_identity": (
                "For each 2x2 block, sum pixel squared error after LL-only reconstruction "
                "is 4(H^2+V^2+D^2); keeping LL,H,V leaves exactly 4D^2."
            ),
            "riesz_identity": (
                "Before finite-grid/roundoff effects, ||R1 f||^2+||R2 f||^2="
                "||f-mean(f)||^2 and the pair is rotation-covariant."
            ),
            "important_boundary": (
                "Riesz(lowpass) is deterministic and contains no source information absent "
                "from the low-pass image. Only Haar H/V carries pre-downsample information."
            ),
        },
        "encoding_diagnostics": summarize_diagnostics(diagnostic_records),
        "probe": {
            "regularization_c_frozen_from_base_dev": base_c,
            "base_dev_cv_macro_auroc_by_c": cv,
            "confirmation_macro_auroc": probe_auc,
            "paired_image_bootstrap": boot,
        },
        "zero_shot_confirmation_macro_auroc": zero_shot_auc,
        "preregistered_gate": {
            "candidate_over_float_base": ">=0.02 macro AUROC and CI lower >0",
            "base_plus_candidate_over_base": ">=0.02 macro AUROC and CI lower >0",
            "specificity": (
                "true geometry must beat spatial shuffle, equal-energy noise, and its "
                "family-specific cross-image/orientation placebo with CI lower >0"
            ),
            "pass_rule": "at least one family passes every condition",
            "failure_action": "do not run a VLM generation canary for this candidate",
        },
        "runtime_seconds": time.time() - started,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
