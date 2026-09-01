#!/usr/bin/env python3
"""Real single-token CE adapter for seven CECD factorial controls.

This adapter evaluates a two-render x two-prompt orbit with Huatuo or Hulu and
applies the architecture-neutral controls to *centered Yes/No/Maybe logits*.
The current CLI is engineering-only.  It cannot emit a formal scientific arm,
cannot implement CECD hidden intervention or either Treble variant, and cannot
stand in for aligned OE evaluation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np

from anchor.corrected_sgta.cecd_dual_semantics_kernels_v1 import (
    IMPLEMENTED_KERNEL_METHODS,
    apply_factorial_control,
    factorial_components,
)
from anchor.corrected_sgta.cecd_dual_semantics_worker_v1 import (
    compute_model_fingerprint,
    file_record,
    sha256_file,
)


VERSION = "cecd-dual-semantics-real-ce-adapter-v1"
FORMAL_CE_CLAIM_CONTRACT_SCHEMA = "cecd-dual-semantics-formal-ce-claim-contract-v1"
FORMAL_CE_CACHE_SCHEMA = "cecd-dual-semantics-formal-ce-raw-cache-v1"
FORMAL_CE_SHARD_SCHEMA = "cecd-dual-semantics-formal-ce-shard-v1"
ROOT = Path("/home/dbw/ANCHOR")
DEFAULT_MANIFEST = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/"
    "manifests_v2/reader_vote_manifest_v2.jsonl"
)
DEFAULT_IMAGE_ROOT = Path("/workspace/vinbigdata/train")
DEFAULT_GPU_LOCK = ROOT / "corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock"
MODEL_DIRS = {
    "huatuo": Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
    "hulu": Path("/home/dbw/models/Hulu-Med-4B"),
}
HUATUO_SOURCE_ROOT = Path("/home/dbw/HuatuoGPT-Vision")
STATES = ("supported", "refuted", "undetermined")
CELL_ORDER = ("h00", "h10", "h01", "h11")
PROMPTS = {
    "existential": "Is there {finding} on this chest X-ray? Answer with exactly one word: Yes, No, or Maybe.",
    "radiograph_subject": "Does this chest X-ray show {finding}? Answer with exactly one word: Yes, No, or Maybe.",
    "visibility": "Can {finding} be seen on this chest X-ray? Answer with exactly one word: Yes, No, or Maybe.",
}


class CEAdapterError(RuntimeError):
    """Fail-closed CE adapter error."""


class CEScorer(Protocol):
    model_family: str

    def score(self, image: Any, prompt: str) -> dict[str, Any]: ...

    def standard_next_token(self, image: Any, prompt: str) -> dict[str, Any]: ...


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            raise CEAdapterError(f"write-once artifact collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def gpu_flock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CEAdapterError(f"GPU lock is busy: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CEAdapterError(f"invalid JSONL line {line_number}: {path}") from error
        if not isinstance(row, dict):
            raise CEAdapterError(f"JSONL line {line_number} is not an object")
        rows.append(row)
    return rows


def readable_finding(finding: str) -> str:
    return finding.replace("_", " ")


def prompt_text(name: str, finding: str) -> str:
    if name not in PROMPTS:
        raise CEAdapterError(f"unknown prompt: {name}")
    text = PROMPTS[name].format(finding=readable_finding(finding))
    if text.count(readable_finding(finding)) != 1:
        raise CEAdapterError("prompt must contain the finding exactly once")
    return text


def resolve_dicom(row: Mapping[str, Any], image_root: Path) -> Path:
    explicit = row.get("image_path")
    path = (
        Path(str(explicit))
        if explicit
        else image_root / str(row.get("dicom_relpath", "")).removeprefix("train/")
    )
    if explicit and not path.is_absolute():
        path = image_root / path
    path = path.resolve()
    if not path.is_file():
        raise CEAdapterError(f"DICOM is missing: {path}")
    return path


def _record_key(row: Mapping[str, Any]) -> str:
    finding, image_id = str(row["finding"]), str(row["image_id"])
    suffix = hashlib.sha256(f"{finding}:{image_id}".encode()).hexdigest()[:12]
    return f"{finding}__{image_id}__{suffix}"


def choose_one_row(
    *, manifest: Path, image_root: Path, record_key: str | None = None
) -> tuple[dict[str, Any], Path]:
    candidates = load_jsonl(manifest)
    candidates.sort(
        key=lambda row: hashlib.sha256(
            f"{VERSION}:{row.get('finding')}:{row.get('image_id')}".encode()
        ).hexdigest()
    )
    if record_key is not None:
        candidates = [row for row in candidates if _record_key(row) == record_key]
    for row in candidates:
        if not row.get("finding") or not row.get("image_id"):
            continue
        try:
            return row, resolve_dicom(row, image_root)
        except CEAdapterError:
            continue
    raise CEAdapterError("no eligible local VinDr DICOM matches the requested record")


def prepare_two_by_two_orbit(
    *,
    row: Mapping[str, Any],
    dicom_path: Path,
    render_names: Sequence[str],
    prompt_names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    if len(render_names) != 2 or len(set(render_names)) != 2:
        raise CEAdapterError("CE orbit requires exactly two distinct renders")
    if len(prompt_names) != 2 or len(set(prompt_names)) != 2:
        raise CEAdapterError("CE orbit requires exactly two distinct prompts")
    # Importing the audited DICOM renderer is CPU-only; model code remains
    # outside this preflight function.
    from anchor.corrected_sgta.run_huatuo_dicom_render_pilot_v1 import (
        build_render_views,
        read_dicom_pixels,
    )

    views = build_render_views(read_dicom_pixels(dicom_path), [], [])
    by_name = {str(view["name"]): view for view in views}
    if not set(render_names) <= set(by_name):
        raise CEAdapterError(f"requested render is unavailable: {render_names}")
    selected = {name: by_name[name] for name in render_names}
    if not all(view["audit"].get("clinical_guard_pass") is True for view in selected.values()):
        raise CEAdapterError("engineering render guard failed for selected orbit")
    finding = str(row["finding"])
    prompts = {name: prompt_text(name, finding) for name in prompt_names}
    images = {
        "h00": selected[render_names[0]]["image"],
        "h10": selected[render_names[1]]["image"],
        "h01": selected[render_names[0]]["image"],
        "h11": selected[render_names[1]]["image"],
    }
    texts = {
        "h00": prompts[prompt_names[0]],
        "h10": prompts[prompt_names[0]],
        "h01": prompts[prompt_names[1]],
        "h11": prompts[prompt_names[1]],
    }
    audit = {
        "record_key": _record_key(row),
        "image_id": str(row["image_id"]),
        "finding": finding,
        "positive_votes": row.get("positive_votes"),
        "dicom": file_record(dicom_path),
        "render_names": list(render_names),
        "prompt_names": list(prompt_names),
        "renders": {
            name: {
                "pixel_sha256": selected[name]["audit"]["pixel_sha256"],
                "clinical_guard_pass": selected[name]["audit"]["clinical_guard_pass"],
                "track": selected[name]["track"],
            }
            for name in render_names
        },
        "prompt_sha256": {
            name: hashlib.sha256(prompts[name].encode()).hexdigest()
            for name in prompt_names
        },
        "clinical_equivalence_established": False,
        "engineering_only": True,
    }
    return images, texts, audit


def centered_logit_vector(scores: Mapping[str, Any]) -> np.ndarray:
    logits = scores.get("logits")
    if not isinstance(logits, Mapping) or set(logits) != set(STATES):
        raise CEAdapterError("scorer must return exactly three tri-state logits")
    values = np.asarray([logits[state] for state in STATES], dtype=np.float64)
    if not np.isfinite(values).all():
        raise CEAdapterError("tri-state logits must be finite")
    return values - values.mean()


def summarize_control_logits(cell_rows: Mapping[str, Mapping[str, Any]], seed: int) -> dict[str, Any]:
    if set(cell_rows) != set(CELL_ORDER):
        raise CEAdapterError("four completed CE cells are required")
    orbit = {
        cell: centered_logit_vector(cell_rows[cell]["scores"])
        for cell in CELL_ORDER
    }
    output: dict[str, Any] = {}
    for method in IMPLEMENTED_KERNEL_METHODS:
        if method == "random_norm":
            parts = factorial_components(orbit)
            generator = np.random.default_rng(int(seed))
            random = generator.standard_normal(parts.interaction.shape)
            random -= random.mean(axis=-1, keepdims=True)
            squared = np.sum(parts.interaction**2, axis=-1, keepdims=True)
            projection = np.divide(
                np.sum(random * parts.interaction, axis=-1, keepdims=True),
                squared,
                out=np.zeros_like(squared),
                where=squared > 0,
            )
            random -= projection * parts.interaction
            random_norm = np.linalg.norm(random, axis=-1, keepdims=True)
            target_norm = np.linalg.norm(parts.interaction, axis=-1, keepdims=True)
            if np.any((target_norm > 0) & (random_norm <= 1e-12)):
                raise CEAdapterError("centered random_norm control is degenerate")
            replacement = np.divide(
                random,
                random_norm,
                out=np.zeros_like(random),
                where=random_norm > 0,
            ) * target_norm
            logits = parts.grand + parts.render + parts.prompt + replacement
        else:
            logits = apply_factorial_control(orbit, method, seed=seed)
        # All method comparisons live in the softmax quotient space. Preserve
        # the zero-mean gauge after randomized arithmetic as well.
        logits = logits - logits.mean(axis=-1, keepdims=True)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        output[method] = {
            "centered_logits": {
                state: float(logits[index]) for index, state in enumerate(STATES)
            },
            "probabilities": {
                state: float(probabilities[index]) for index, state in enumerate(STATES)
            },
            "prediction": STATES[int(np.argmax(logits))],
            "entropy_nats": float(
                -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
            ),
            "operation_space": "centered FP32-equivalent tri-state next-token logits",
        }
    return output


def _valid_cell(path: Path, fingerprint: str, cell: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(row, dict)
        or row.get("version") != VERSION
        or row.get("status") != "complete"
        or row.get("config_fingerprint") != fingerprint
        or row.get("cell") != cell
    ):
        return None
    try:
        centered_logit_vector(row["scores"])
    except (CEAdapterError, KeyError):
        return None
    return row


def freeze_or_resume_config(
    *, output_dir: Path, candidate: Mapping[str, Any], resume: bool
) -> dict[str, Any]:
    path = output_dir / "config.json"
    immutable = {
        key: value for key, value in candidate.items() if key not in {"created_at", "command"}
    }
    fingerprint = canonical_sha256(immutable)
    frozen = {**candidate, "fingerprint": fingerprint}
    if not resume:
        if path.exists():
            raise CEAdapterError(f"output already exists; use --resume: {path}")
        atomic_json(path, frozen)
        return frozen
    if not path.is_file():
        raise CEAdapterError("--resume requires config.json")
    existing = json.loads(path.read_text(encoding="utf-8"))
    existing_immutable = {
        key: value
        for key, value in existing.items()
        if key not in {"created_at", "command", "fingerprint"}
    }
    if canonical_sha256(existing_immutable) != existing.get("fingerprint"):
        raise CEAdapterError("stored config fingerprint mismatch")
    if existing_immutable != immutable:
        changed = sorted(
            key
            for key in set(existing_immutable) | set(immutable)
            if existing_immutable.get(key) != immutable.get(key)
        )
        raise CEAdapterError(f"refusing resume after config drift: {changed}")
    return existing


def score_atomic_cells(
    *,
    scorer: CEScorer,
    images: Mapping[str, Any],
    prompts: Mapping[str, str],
    output_dir: Path,
    config_fingerprint: str,
    scientific_status: str = "engineering_only_no_scientific_authorization",
) -> tuple[dict[str, dict[str, Any]], int]:
    if set(images) != set(CELL_ORDER) or set(prompts) != set(CELL_ORDER):
        raise CEAdapterError("images/prompts must close the four-cell orbit")
    rows: dict[str, dict[str, Any]] = {}
    newly_scored = 0
    for cell in CELL_ORDER:
        path = output_dir / "cells" / f"{cell}.json"
        existing = _valid_cell(path, config_fingerprint, cell)
        if existing is not None:
            rows[cell] = existing
            continue
        scores = scorer.score(images[cell], prompts[cell])
        centered_logit_vector(scores)
        row = {
            "version": VERSION,
            "status": "complete",
            "scientific_status": scientific_status,
            "config_fingerprint": config_fingerprint,
            "model_family": scorer.model_family,
            "cell": cell,
            "prompt_sha256": hashlib.sha256(prompts[cell].encode()).hexdigest(),
            "scores": scores,
        }
        row["fingerprint"] = canonical_sha256(row)
        atomic_json(path, row)
        rows[cell] = row
        newly_scored += 1
    return rows, newly_scored


def source_closure(family: str, model_dir: Path) -> list[dict[str, Any]]:
    local = Path(__file__).resolve().parent
    paths = [
        Path(__file__).resolve(),
        local / "cecd_dual_semantics_kernels_v1.py",
        local / "cecd_dual_semantics_worker_v1.py",
        local / "run_cecd_factorial_v1.py",
        local / "run_huatuo_dicom_render_pilot_v1.py",
    ]
    if family == "huatuo":
        paths.extend(path for path in HUATUO_SOURCE_ROOT.rglob("*.py") if path.is_file())
    else:
        paths.extend(path for path in model_dir.glob("*.py") if path.is_file())
    return [file_record(path) for path in sorted({path.resolve() for path in paths})]


def cpu_preflight(
    *,
    family: str,
    manifest: Path,
    image_root: Path,
    model_dir: Path,
    record_key: str | None,
    render_names: Sequence[str],
    prompt_names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    row, dicom = choose_one_row(
        manifest=manifest, image_root=image_root, record_key=record_key
    )
    images, prompts, orbit_audit = prepare_two_by_two_orbit(
        row=row,
        dicom_path=dicom,
        render_names=render_names,
        prompt_names=prompt_names,
    )
    preflight = {
        "version": VERSION,
        "status": "cpu_preflight_passed_no_model_or_cuda",
        "scientific_status": "engineering_only_no_scientific_authorization",
        "family": family,
        "manifest": file_record(manifest),
        "image_root": str(image_root.resolve()),
        "model_dir": str(model_dir.resolve()),
        "model_fingerprint": compute_model_fingerprint(family, model_dir),
        "source_files": source_closure(family, model_dir),
        "orbit": orbit_audit,
        "methods": list(IMPLEMENTED_KERNEL_METHODS),
        "operation_space": "centered tri-state next-token logits",
        "cecd_hidden_intervention_implemented": False,
        "treble_variants_implemented": False,
        "oe_adapter_implemented": False,
        "model_loaded": False,
        "cuda_initialized_by_adapter": False,
    }
    preflight["fingerprint"] = canonical_sha256(preflight)
    return preflight, images, prompts


def build_real_scorer(family: str, model_dir: Path) -> CEScorer:
    # This is the first point at which model/Torch runtime code is imported.
    from anchor.corrected_sgta.run_cecd_factorial_v1 import HuatuoScorer, HuluScorer

    if family == "huatuo":
        return HuatuoScorer(model_dir, HUATUO_SOURCE_ROOT, "cuda:0")
    return HuluScorer(model_dir, 1024)


def run_engineering_smoke(
    *,
    family: str,
    manifest: Path,
    image_root: Path,
    model_dir: Path,
    output_dir: Path,
    record_key: str | None,
    render_names: Sequence[str],
    prompt_names: Sequence[str],
    seed: int,
    gpu_lock_path: Path,
    resume: bool,
) -> dict[str, Any]:
    preflight, images, prompts = cpu_preflight(
        family=family,
        manifest=manifest,
        image_root=image_root,
        model_dir=model_dir,
        record_key=record_key,
        render_names=render_names,
        prompt_names=prompt_names,
    )
    candidate = {
        **preflight,
        "created_at": utc_now(),
        "command": list(os.sys.argv),
        "seed": int(seed),
        "gpu_lock": str(gpu_lock_path.resolve()),
        "execution_mode": "explicit_engineering_single_sample_smoke",
        "formal_method_output_authorized": False,
        "paper_claim_authorized": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config = freeze_or_resume_config(
        output_dir=output_dir, candidate=candidate, resume=resume
    )
    existing = {
        cell: _valid_cell(output_dir / "cells" / f"{cell}.json", config["fingerprint"], cell)
        for cell in CELL_ORDER
    }
    if all(row is not None for row in existing.values()):
        rows = {cell: row for cell, row in existing.items() if row is not None}
        newly_scored = 0
    else:
        with gpu_flock(gpu_lock_path):
            scorer = build_real_scorer(family, model_dir)
            rows, newly_scored = score_atomic_cells(
                scorer=scorer,
                images=images,
                prompts=prompts,
                output_dir=output_dir,
                config_fingerprint=config["fingerprint"],
            )
    controls = summarize_control_logits(rows, seed)
    summary = {
        "version": VERSION,
        "status": "engineering_smoke_complete",
        "scientific_status": "engineering_only_no_scientific_authorization",
        "config_fingerprint": config["fingerprint"],
        "model_family": family,
        "record_key": preflight["orbit"]["record_key"],
        "completed_cells": list(CELL_ORDER),
        "newly_scored_cells": newly_scored,
        "controls": controls,
        "kernel_methods": list(IMPLEMENTED_KERNEL_METHODS),
        "formal_method_output_authorized": False,
        "oe_adapter_implemented": False,
        "cecd_hidden_intervention_implemented": False,
        "treble_variants_implemented": False,
        "paper_claim_authorized": False,
    }
    summary["fingerprint"] = canonical_sha256(summary)
    atomic_json(output_dir / "summary.json", summary)
    return summary


def _formal_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "task",
        "render_names",
        "prompt_names",
        "seed",
        "image_root",
        "minimum_clusters",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise CEAdapterError("formal CE claim-contract fields are not closed")
    if (
        payload["schema_version"] != FORMAL_CE_CLAIM_CONTRACT_SCHEMA
        or payload["task"] != "fixed_claim_single_token_ce"
        or payload["minimum_clusters"] != 30
        or not isinstance(payload["seed"], int)
        or len(payload["render_names"]) != 2
        or len(set(payload["render_names"])) != 2
        or len(payload["prompt_names"]) != 2
        or len(set(payload["prompt_names"])) != 2
    ):
        raise CEAdapterError("formal CE claim contract drift")
    return payload


def _ordered_record_keys(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "record_keys"}:
        raise CEAdapterError("record-key manifest fields are not closed")
    keys = payload["record_keys"]
    if (
        payload["schema_version"] != "cecd-dual-semantics-record-keys-v1"
        or not isinstance(keys, list)
        or len(keys) < 30
        or any(not isinstance(key, str) or not key for key in keys)
        or len(keys) != len(set(keys))
    ):
        raise CEAdapterError("record-key manifest is invalid or underpowered")
    return keys


def _formal_rows(evaluation_manifest: Path, record_keys: Sequence[str]) -> list[dict[str, Any]]:
    rows = load_jsonl(evaluation_manifest)
    if len(rows) != len(record_keys):
        raise CEAdapterError("evaluation manifest and record-key count disagree")
    observed = []
    clusters = set()
    for row in rows:
        required = {"record_key", "cluster_id", "image_id", "finding"}
        if not required <= set(row):
            raise CEAdapterError("formal CE evaluation row lacks identity/cluster fields")
        if row["record_key"] != _record_key(row):
            raise CEAdapterError("formal CE record_key is not canonical")
        observed.append(str(row["record_key"]))
        clusters.add(str(row["cluster_id"]))
    if observed != list(record_keys):
        raise CEAdapterError("evaluation rows do not match frozen record-key order")
    if len(clusters) < 30:
        raise CEAdapterError("formal CE evaluation requires at least 30 clusters")
    return rows


def run_formal_ce_method(
    *,
    family: str,
    method: str,
    model_dir: Path,
    evaluation_manifest: Path,
    record_keys_path: Path,
    claim_contract_path: Path,
    run_contract: Mapping[str, Any],
    shared_cache_root: Path,
    output_dir: Path,
    scorer_factory: Any = build_real_scorer,
    orbit_provider: Any = prepare_two_by_two_orbit,
) -> dict[str, Any]:
    """Build/reuse one raw four-cell cache and derive one formal CE control."""

    if method not in IMPLEMENTED_KERNEL_METHODS:
        raise CEAdapterError(f"formal centered-logit CE method is not implemented: {method}")
    formal = _formal_contract(claim_contract_path)
    record_keys = _ordered_record_keys(record_keys_path)
    rows = _formal_rows(evaluation_manifest, record_keys)
    run_fingerprint = str(run_contract.get("fingerprint", ""))
    if len(run_fingerprint) != 64 or family not in run_contract.get("models", {}):
        raise CEAdapterError("formal CE run contract is incomplete")
    descriptor = run_contract["runtime_descriptors"][family]
    descriptor_sha = canonical_sha256(descriptor)
    cache_candidate = {
        "version": VERSION,
        "schema_version": FORMAL_CE_CACHE_SCHEMA,
        "scientific_status": "formal_ce_only_oe_blocked",
        "run_fingerprint": run_fingerprint,
        "runtime_descriptor_sha256": descriptor_sha,
        "model_family": family,
        "model_id": run_contract["models"][family]["model_id"],
        "model_dir": str(model_dir.resolve()),
        "evaluation_manifest": file_record(evaluation_manifest),
        "record_keys": file_record(record_keys_path),
        "claim_contract": file_record(claim_contract_path),
        "claim_contract_payload": formal,
        "ordered_record_keys_sha256": canonical_sha256(record_keys),
        "record_count": len(rows),
        "cluster_count": len({str(row["cluster_id"]) for row in rows}),
        "source_files": source_closure(family, model_dir),
        "operation_space": "raw tri-state next-token logits; centering occurs only at method derivation",
        "oe_implemented": False,
        "hidden_intervention_implemented": False,
        "paper_native_treble_claimed": False,
    }
    shared_cache_root.mkdir(parents=True, exist_ok=True)
    cache_config = freeze_or_resume_config(
        output_dir=shared_cache_root,
        candidate=cache_candidate,
        resume=(shared_cache_root / "config.json").is_file(),
    )
    missing = []
    for row in rows:
        record_dir = shared_cache_root / "records" / str(row["record_key"])
        for cell in CELL_ORDER:
            if _valid_cell(
                record_dir / "cells" / f"{cell}.json", cache_config["fingerprint"], cell
            ) is None:
                missing.append(str(row["record_key"]))
                break
    scorer = None
    if missing:
        scorer = scorer_factory(family, model_dir)
        image_root = Path(str(formal["image_root"]))
        missing_set = set(missing)
        for row in rows:
            if str(row["record_key"]) not in missing_set:
                continue
            dicom = resolve_dicom(row, image_root)
            images, prompts, _ = orbit_provider(
                row=row,
                dicom_path=dicom,
                render_names=formal["render_names"],
                prompt_names=formal["prompt_names"],
            )
            score_atomic_cells(
                scorer=scorer,
                images=images,
                prompts=prompts,
                output_dir=shared_cache_root / "records" / str(row["record_key"]),
                config_fingerprint=cache_config["fingerprint"],
                scientific_status="formal_ce_raw_logit_cache_only",
            )

    cache_cells = []
    method_rows = []
    for row in rows:
        record_dir = shared_cache_root / "records" / str(row["record_key"])
        cell_rows = {}
        for cell in CELL_ORDER:
            path = record_dir / "cells" / f"{cell}.json"
            completed = _valid_cell(path, cache_config["fingerprint"], cell)
            if completed is None:
                raise CEAdapterError(f"formal CE cache remains incomplete: {row['record_key']}/{cell}")
            cell_rows[cell] = completed
            cache_cells.append(
                {"record_key": row["record_key"], "cell": cell, "file": file_record(path)}
            )
        control = summarize_control_logits(cell_rows, int(formal["seed"]))[method]
        method_rows.append(
            {
                "record_key": row["record_key"],
                "cluster_id": row["cluster_id"],
                "image_id": row["image_id"],
                "finding": row["finding"],
                "method": method,
                **control,
            }
        )
    cache_manifest = {
        "schema_version": FORMAL_CE_CACHE_SCHEMA,
        "status": "complete",
        "config_fingerprint": cache_config["fingerprint"],
        "run_fingerprint": run_fingerprint,
        "model_family": family,
        "records": len(rows),
        "clusters": len({str(row["cluster_id"]) for row in rows}),
        "cells": len(cache_cells),
        "cell_files": cache_cells,
        "shared_across_methods": True,
    }
    cache_manifest["fingerprint"] = canonical_sha256(cache_manifest)
    write_once_json(shared_cache_root / "raw_cache_manifest.json", cache_manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    ce_output = output_dir / "ce_rows.jsonl"
    atomic_jsonl(ce_output, method_rows)
    worker_record = run_contract["source_files"]["worker"]
    completion = {
        "schema_version": FORMAL_CE_SHARD_SCHEMA,
        "status": "formal_ce_complete_oe_blocked",
        "run_fingerprint": run_fingerprint,
        "model_family": family,
        "model_id": run_contract["models"][family]["model_id"],
        "method": method,
        "task": "ce",
        "raw_cache_manifest": file_record(shared_cache_root / "raw_cache_manifest.json"),
        "ce_output": {
            "path": ce_output.name,
            "sha256": sha256_file(ce_output),
            "bytes": ce_output.stat().st_size,
        },
        "rows": len(method_rows),
        "clusters": len({str(row["cluster_id"]) for row in rows}),
        "worker_sha256": worker_record["sha256"],
        "runtime_descriptor_sha256": descriptor_sha,
        "oe_implemented": False,
        "hidden_intervention_implemented": False,
        "paper_native_treble_claimed": False,
    }
    completion["completion_fingerprint"] = canonical_sha256(completion)
    write_once_json(output_dir / "ce_completion.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=("huatuo", "hulu"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--record-key")
    parser.add_argument("--render-names", nargs=2, default=["baseline_percentile", "native_linear"])
    parser.add_argument("--prompt-names", nargs=2, default=["existential", "radiograph_subject"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cpu-preflight-only", action="store_true")
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.cpu_preflight_only == args.engineering_smoke:
        raise CEAdapterError("select exactly one of --cpu-preflight-only or --engineering-smoke")
    model_dir = args.model_dir or MODEL_DIRS[args.model_family]
    if args.cpu_preflight_only:
        preflight, _, _ = cpu_preflight(
            family=args.model_family,
            manifest=args.manifest,
            image_root=args.image_root,
            model_dir=model_dir,
            record_key=args.record_key,
            render_names=args.render_names,
            prompt_names=args.prompt_names,
        )
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        raise CEAdapterError("--engineering-smoke requires --output-dir")
    result = run_engineering_smoke(
        family=args.model_family,
        manifest=args.manifest,
        image_root=args.image_root,
        model_dir=model_dir,
        output_dir=args.output_dir,
        record_key=args.record_key,
        render_names=args.render_names,
        prompt_names=args.prompt_names,
        seed=args.seed,
        gpu_lock_path=args.gpu_lock,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
