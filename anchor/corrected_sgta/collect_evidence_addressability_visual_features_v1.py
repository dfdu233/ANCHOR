#!/usr/bin/env python3
"""Crash-safe raw-vision and post-projector capture for Addressability Stage 2.

The collector reads the exact image set already frozen by the development and
confirmation hidden-state artifacts.  It performs multimodal preparation only:
there is no decoder forward, answer generation, probe fitting, or outcome read.
Every image is committed as an atomic shard so a baseline-preempted run resumes
without repeating completed work.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import transformers

from corrected_sgta.collect_vindr_hidden_states_v2 import build_runtime
from corrected_sgta.halp_three_plane_preflight_v1 import (
    PreProjectorVisionCapture,
    cpu_source_audit,
    file_record,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    atomic_json,
    load_image,
    prompt_for,
    sha256_file,
)
from corrected_sgta.run_hulu_vindr_commitment_probe import model_file_inventory


VERSION = "evidence-addressability-visual-feature-collector-v1"
SHARD_SCHEMA = "evidence-addressability-visual-feature-shard-v1"
ARRAY_NAMES = ("pre_mean", "pre_std", "post_mean", "post_std")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def full_model_inventory(model_dir: Path) -> list[dict[str, Any]]:
    """Hash every checkpoint/source file before any resumable shard is written."""

    records = []
    for path in sorted(candidate for candidate in model_dir.rglob("*") if candidate.is_file()):
        records.append(
            {
                "path": str(path.relative_to(model_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise ValueError(f"empty model directory: {model_dir}")
    return records


def hidden_contract(directory: Path, model: str) -> dict[str, Any]:
    config_path = directory / "config.json"
    metadata_path = directory / "metadata.jsonl"
    summary_path = directory / "summary.json"
    for path in (config_path, metadata_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if config.get("model_id") != model or summary.get("status") != "complete":
        raise ValueError(f"hidden artifact model/status mismatch: {directory}")
    return {
        "directory": str(directory.resolve()),
        "model_id": model,
        "config_sha256": sha256_file(config_path),
        "metadata_sha256": sha256_file(metadata_path),
        "summary_sha256": sha256_file(summary_path),
        "model_inventory": config.get("model_inventory"),
        "collector_code_sha256": config.get("code_sha256"),
    }


def load_metadata(directory: Path, findings: set[str]) -> list[dict[str, Any]]:
    path = directory / "metadata.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = [row for row in rows if str(row["finding"]) in findings]
    if not rows:
        raise ValueError(f"empty metadata: {path}")
    return rows


def frozen_images(
    dev: Path, confirmation: Path, findings: set[str], image_root: Path
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for split, directory in (("development", dev), ("confirmation", confirmation)):
        grouped: dict[str, set[str]] = {}
        for row in load_metadata(directory, findings):
            image_id = str(row["image_id"])
            grouped.setdefault(image_id, set()).add(str(row["finding"]))
        for image_id in sorted(grouped):
            if image_id in seen:
                raise ValueError(
                    f"image {image_id} occurs in both {seen[image_id]} and {split}"
                )
            seen[image_id] = split
            image_path = image_root / f"{image_id}.dicom"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            stat = image_path.stat()
            output.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "findings": sorted(grouped[image_id]),
                    "dicom_bytes": stat.st_size,
                    "dicom_mtime_ns": stat.st_mtime_ns,
                }
            )
    return output


def feature_stats(tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray, int]:
    if tokens.ndim == 3:
        if tokens.shape[0] != 1:
            raise ValueError(f"batch size is not one: {tuple(tokens.shape)}")
        tokens = tokens[0]
    if tokens.ndim != 2 or min(tokens.shape) <= 0:
        raise ValueError(f"invalid token matrix: {tuple(tokens.shape)}")
    values = tokens.detach().float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("captured nonfinite visual feature")
    return (
        values.mean(dim=0).cpu().numpy().astype(np.float32),
        values.std(dim=0, correction=0).cpu().numpy().astype(np.float32),
        int(values.shape[0]),
    )


def simple_image_metadata(image: Any, image_path: Path) -> dict[str, Any]:
    """Low-capacity nuisance controls frozen independently of clinical labels."""

    import pydicom

    grayscale = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
    header = pydicom.dcmread(
        str(image_path),
        stop_before_pixels=True,
        specific_tags=[
            "ViewPosition",
            "PatientPosition",
            "PhotometricInterpretation",
            "Rows",
            "Columns",
        ],
    )
    return {
        "image_width": int(image.width),
        "image_height": int(image.height),
        "brightness_mean": float(grayscale.mean()),
        "brightness_std": float(grayscale.std()),
        "brightness_p05": float(np.quantile(grayscale, 0.05)),
        "brightness_p95": float(np.quantile(grayscale, 0.95)),
        "view_position": str(getattr(header, "ViewPosition", "UNKNOWN") or "UNKNOWN"),
        "patient_position": str(
            getattr(header, "PatientPosition", "UNKNOWN") or "UNKNOWN"
        ),
        "photometric_interpretation": str(
            getattr(header, "PhotometricInterpretation", "UNKNOWN") or "UNKNOWN"
        ),
        "dicom_rows": int(getattr(header, "Rows", image.height) or image.height),
        "dicom_columns": int(getattr(header, "Columns", image.width) or image.width),
    }


def capture_features(
    runtime: Any, prepare: Any, vision: Any, image: Any, prompt: str
) -> dict[str, Any]:
    with PreProjectorVisionCapture(vision) as capture:
        embeddings, _attention, _positions, (start, end) = prepare(
            runtime, prompt, image
        )
    pre_mean, pre_std, pre_tokens = feature_stats(capture.one())
    post_mean, post_std, post_tokens = feature_stats(embeddings[:, start:end])
    visual_span_tokens = int(end - start)
    if post_tokens != visual_span_tokens:
        raise ValueError(
            f"post-projector token count {post_tokens} differs from decoder visual span {visual_span_tokens}"
        )
    return {
        "pre_mean": pre_mean,
        "pre_std": pre_std,
        "post_mean": post_mean,
        "post_std": post_std,
        "pre_tokens": pre_tokens,
        "post_tokens": post_tokens,
        "visual_span_tokens": visual_span_tokens,
    }


def shard_path(directory: Path, index: int, image_id: str) -> Path:
    suffix = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:16]
    return directory / "shards" / f"{index:06d}-{suffix}.npz"


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


@contextlib.contextmanager
def gpu_lock(path: Path, wait_for_lock: bool):
    """Acquire the shared GPU lock without bypassing baseline ownership."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        if wait_for_lock:
            # A real blocking waiter is queued by the kernel.  Polling here can
            # lose every sub-second unlock/relock boundary to the active queue.
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        else:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise TimeoutError(f"GPU lock is busy: {path}")
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_shard(
    path: Path,
    index: int,
    image_id: str,
    fingerprint: str,
    audit_fingerprint: str,
    image_path: Path,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = set(ARRAY_NAMES) | {
            "metadata_json",
            "ordered_index",
            "image_id",
            "config_fingerprint",
            "schema_version",
        }
        if set(archive.files) != required:
            raise ValueError(f"{path}: shard field drift")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if (
            int(archive["ordered_index"].item()) != index
            or str(archive["image_id"].item()) != image_id
            or str(archive["config_fingerprint"].item()) != fingerprint
            or str(archive["schema_version"].item()) != SHARD_SCHEMA
        ):
            raise ValueError(f"{path}: shard identity drift")
        if metadata.get("source_audit_fingerprint") != audit_fingerprint:
            raise ValueError(f"{path}: source-audit drift")
        if metadata.get("dicom_sha256") != sha256_file(image_path):
            raise ValueError(f"{path}: DICOM content drift")
        for name in ARRAY_NAMES:
            value = np.asarray(archive[name], dtype=np.float32)
            if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
                raise ValueError(f"{path}: invalid {name}")
    return metadata


def freeze_config(
    args: argparse.Namespace,
    images: list[dict[str, Any]],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    code = Path(__file__).resolve()
    current_model_inventory = model_file_inventory(args.model_dir)
    development_hidden = hidden_contract(args.dev, args.model)
    confirmation_hidden = hidden_contract(args.confirmation, args.model)
    if (
        development_hidden["model_inventory"] != current_model_inventory
        or confirmation_hidden["model_inventory"] != current_model_inventory
    ):
        raise ValueError("hidden artifacts and current checkpoint inventory differ")
    helper_files = [
        code,
        Path(__file__).with_name("collect_vindr_hidden_states_v2.py"),
        Path(__file__).with_name("halp_three_plane_preflight_v1.py"),
        Path(__file__).with_name("run_huatuo_vindr_commitment_probe.py"),
        Path(__file__).with_name("run_hulu_vindr_commitment_probe.py"),
    ]
    static = {
        "version": VERSION,
        "model": args.model,
        "model_dir": str(args.model_dir.resolve()),
        "model_inventory": current_model_inventory,
        "full_model_inventory": full_model_inventory(args.model_dir),
        "source_audit_fingerprint": audit["fingerprint"],
        "source_audit": audit,
        "helper_files": [file_record(path) for path in helper_files],
        "huatuo_root": str(args.huatuo_root.resolve()),
        "development_hidden_contract": development_hidden,
        "confirmation_hidden_contract": confirmation_hidden,
        "findings": sorted(set(args.findings)),
        "image_root": str(args.image_root.resolve()),
        "max_visual_tokens": int(args.max_visual_tokens),
        "ordered_images_sha256": object_sha256(images),
        "n_images": len(images),
        "representation_contract": {
            "pre_mean_std": "model-native vision output immediately before projector",
            "post_mean_std": "projected visual token span immediately before decoder",
            "no_decoder_forward": True,
            "no_generation": True,
            "outcome_read": False,
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "code_sha256": sha256_file(code),
    }
    fingerprint = object_sha256(static)
    config_path = args.output_dir / "config.json"
    order_path = args.output_dir / "ordered_images.json"
    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError(f"output exists; use --resume: {args.output_dir}")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        existing_static = {
            key: value
            for key, value in existing.items()
            if key not in {"created_at", "command", "fingerprint"}
        }
        if existing.get("fingerprint") != object_sha256(existing_static):
            raise ValueError("stored config fingerprint is inconsistent")
        if existing_static != static or existing.get("fingerprint") != fingerprint:
            raise ValueError("refusing resume after semantic config drift")
        if json.loads(order_path.read_text(encoding="utf-8")) != images:
            raise ValueError("refusing resume after ordered-image drift")
        return existing
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "shards").mkdir()
    config = {
        **static,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "fingerprint": fingerprint,
    }
    atomic_json(config_path, config)
    atomic_json(order_path, images)
    return config


def aggregate(
    args: argparse.Namespace,
    images: list[dict[str, Any]],
    fingerprint: str,
    audit_fingerprint: str,
) -> None:
    canary_path = args.output_dir / "prompt_invariance_canary.json"
    if not canary_path.is_file():
        raise FileNotFoundError(canary_path)
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    if (
        canary.get("status") != "pass"
        or canary.get("config_fingerprint") != fingerprint
        or canary.get("source_audit_fingerprint") != audit_fingerprint
    ):
        raise ValueError("prompt-invariance canary is stale or failed")
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in ARRAY_NAMES}
    metadata: list[dict[str, Any]] = []
    for index, row in enumerate(images):
        path = shard_path(args.output_dir, index, row["image_id"])
        image_path = args.image_root / f"{row['image_id']}.dicom"
        metadata.append(
            validate_shard(
                path,
                index,
                row["image_id"],
                fingerprint,
                audit_fingerprint,
                image_path,
            )
        )
        with np.load(path, allow_pickle=False) as archive:
            for name in ARRAY_NAMES:
                arrays[name].append(np.asarray(archive[name], dtype=np.float32))
    shapes = {name: {value.shape for value in values} for name, values in arrays.items()}
    if any(len(values) != 1 for values in shapes.values()):
        raise ValueError(f"feature shape drift: {shapes}")
    atomic_npz(
        args.output_dir / "features.npz",
        **{name: np.stack(values) for name, values in arrays.items()},
    )
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata)
    metadata_path = args.output_dir / "metadata.jsonl"
    temporary = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, metadata_path)
    atomic_json(
        args.output_dir / "summary.json",
        {
            "status": "complete",
            "version": VERSION,
            "model": args.model,
            "n_images": len(images),
            "development_images": sum(row["split"] == "development" for row in images),
            "confirmation_images": sum(row["split"] == "confirmation" for row in images),
            "config_fingerprint": fingerprint,
            "features_sha256": sha256_file(args.output_dir / "features.npz"),
            "metadata_sha256": sha256_file(metadata_path),
            "prompt_invariance_canary_sha256": sha256_file(canary_path),
            "scientific_claim_authorized": False,
        },
    )


def run(args: argparse.Namespace) -> None:
    findings = set(args.findings)
    if len(findings) != len(args.findings) or len(findings) < 2:
        raise ValueError("--findings must contain at least two unique values")
    images = frozen_images(args.dev, args.confirmation, findings, args.image_root)
    audit = cpu_source_audit(
        family=args.model, model_dir=args.model_dir, huatuo_root=args.huatuo_root
    )
    config = freeze_config(args, images, audit)
    fingerprint = str(config["fingerprint"])
    audit_fingerprint = str(audit["fingerprint"])
    existing = 0
    for index, row in enumerate(images):
        path = shard_path(args.output_dir, index, row["image_id"])
        if path.is_file():
            image_path = args.image_root / f"{row['image_id']}.dicom"
            validate_shard(
                path,
                index,
                row["image_id"],
                fingerprint,
                audit_fingerprint,
                image_path,
            )
            existing += 1
    if existing == len(images):
        aggregate(args, images, fingerprint, audit_fingerprint)
        print(json.dumps({"status": "already_complete", "n": len(images)}))
        return
    started = time.perf_counter()
    with gpu_lock(args.gpu_lock, args.wait_for_gpu_lock):
        runtime, prepare = build_runtime(args)
        vision = (
            runtime.model.get_vision_tower()
            if args.model == "huatuo"
            else runtime.model.get_vision_encoder()
        )
        canary_path = args.output_dir / "prompt_invariance_canary.json"
        if not canary_path.is_file():
            canary_image_path = args.image_root / f"{images[0]['image_id']}.dicom"
            canary_image = load_image(canary_image_path)
            left = capture_features(
                runtime, prepare, vision, canary_image, prompt_for(args.findings[0])
            )
            right = capture_features(
                runtime, prepare, vision, canary_image, prompt_for(args.findings[1])
            )
            differences = {
                name: float(np.max(np.abs(left[name] - right[name])))
                for name in ARRAY_NAMES
            }
            passed = bool(max(differences.values()) <= 1e-6)
            atomic_json(
                canary_path,
                {
                    "status": "pass" if passed else "fail",
                    "config_fingerprint": fingerprint,
                    "source_audit_fingerprint": audit_fingerprint,
                    "image_id": images[0]["image_id"],
                    "prompt_findings": args.findings[:2],
                    "maximum_absolute_differences": differences,
                    "threshold": 1e-6,
                },
            )
            if not passed:
                raise ValueError("pre/post visual features vary with finding prompt")
        else:
            canary = json.loads(canary_path.read_text(encoding="utf-8"))
            if (
                canary.get("status") != "pass"
                or canary.get("config_fingerprint") != fingerprint
                or canary.get("source_audit_fingerprint") != audit_fingerprint
            ):
                raise ValueError("prompt-invariance canary is absent, failed, or stale")
        for index, row in enumerate(images):
            path = shard_path(args.output_dir, index, row["image_id"])
            if path.is_file():
                continue
            image_path = args.image_root / f"{row['image_id']}.dicom"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image = load_image(image_path)
            dicom_sha256 = sha256_file(image_path)
            nuisance = simple_image_metadata(image, image_path)
            prompt = prompt_for(row["findings"][0])
            case_started = time.perf_counter()
            captured = capture_features(runtime, prepare, vision, image, prompt)
            metadata = {
                **row,
                "ordered_index": index,
                "dicom_relpath": image_path.name,
                "dicom_sha256": dicom_sha256,
                "pre_tokens": captured["pre_tokens"],
                "post_tokens": captured["post_tokens"],
                "visual_span_tokens": captured["visual_span_tokens"],
                "pre_dimension": int(captured["pre_mean"].size),
                "post_dimension": int(captured["post_mean"].size),
                "simple_image_metadata": nuisance,
                "elapsed_seconds": time.perf_counter() - case_started,
                "source_audit_fingerprint": audit["fingerprint"],
            }
            atomic_npz(
                path,
                pre_mean=captured["pre_mean"],
                pre_std=captured["pre_std"],
                post_mean=captured["post_mean"],
                post_std=captured["post_std"],
                metadata_json=np.asarray(canonical_json(metadata)),
                ordered_index=np.asarray(index, dtype=np.int64),
                image_id=np.asarray(row["image_id"]),
                config_fingerprint=np.asarray(fingerprint),
                schema_version=np.asarray(SHARD_SCHEMA),
            )
            completed = index + 1
            if completed % args.progress_every == 0 or completed == len(images):
                print(
                    json.dumps(
                        {
                            "model": args.model,
                            "completed_prefix_index": completed,
                            "total": len(images),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    aggregate(args, images, fingerprint, audit_fingerprint)
    print(json.dumps({"status": "complete", "n": len(images)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("huatuo", "hulu"), required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--findings", nargs="+", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision")
    )
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument(
        "--gpu-lock",
        type=Path,
        default=Path(
            "/home/dbw/ANCHOR/corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument("--wait-for-gpu-lock", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    run(args)


if __name__ == "__main__":
    main()
