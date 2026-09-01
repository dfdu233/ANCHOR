#!/usr/bin/env python3
"""Fail-closed runtime for VinDr 14-label CECD listing experiments.

The runtime is intentionally inert until an externally pinned, independently
produced admission receipt passes exact hash closure.  It performs one native
greedy generation per image/cell, preserves malformed answers verbatim, and
writes immutable atomic cell shards that can be resumed without replaying
completed generations.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from PIL import Image

from corrected_sgta.prepare_vindr_reader_manifest import sha256_file
from corrected_sgta.validate_vindr_cecd_listing_scientific_admission_v1 import (
    ScientificAdmissionError,
    validate_scientific_admission,
)


VERSION = "vindr-cecd-listing-runtime-v1"
ADMISSION_VERSION = "vindr-cecd-listing-scientific-admission-v1"
PACK_VERSION = "vindr-cecd-listing-blinded-admission-pack-v1"
MANIFEST_VERSION = "vindr-cecd-ontology-listing-pack-v1"
NONE_TOKEN = "None of the listed findings"
ONTOLOGY = (
    ("aortic_enlargement", "Aortic enlargement"),
    ("atelectasis", "Atelectasis"),
    ("calcification", "Calcification"),
    ("cardiomegaly", "Cardiomegaly"),
    ("consolidation", "Consolidation"),
    ("ild", "ILD"),
    ("infiltration", "Infiltration"),
    ("lung_opacity", "Lung Opacity"),
    ("nodule_mass", "Nodule/Mass"),
    ("other_lesion", "Other lesion"),
    ("pleural_effusion", "Pleural effusion"),
    ("pleural_thickening", "Pleural thickening"),
    ("pneumothorax", "Pneumothorax"),
    ("pulmonary_fibrosis", "Pulmonary fibrosis"),
)
ID_BY_LABEL = {label: finding_id for finding_id, label in ONTOLOGY}
LABEL_BY_ID = dict(ONTOLOGY)
SCIENCE_RENDERS = (
    "baseline_percentile",
    "native_linear",
    "center_minus_0p05w",
    "center_plus_0p05w",
    "width_x1p25",
)
SCIENCE_PROMPTS = (
    "inspect_and_list",
    "which_are_visible",
    "report_all_from_ontology",
)
IDENTITY_RENDER = "identity_lossless_duplicate"
DUPLICATE_PROMPT = "inspect_and_list_exact_duplicate"
ALLOWED_MODELS = ("huatuo", "hulu")
DEFAULT_GPU_LOCK = Path(
    "/home/dbw/ANCHOR/corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock"
)


class RuntimeContractError(RuntimeError):
    pass


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_once_json(path: Path, value: Any) -> None:
    expected = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise RuntimeContractError(f"write-once artifact collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(expected)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def gpu_flock(path: Path):
    """Share the singleton GPU0 lock with the binary CE runtime."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeContractError(f"GPU lock is busy: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _require_canonical_gpu_lock(model_id: str, gpu_lock_path: Path) -> None:
    """Prevent direct real-model invocations from creating a split lock."""

    if model_id in ALLOWED_MODELS and gpu_lock_path.resolve() != DEFAULT_GPU_LOCK.resolve():
        raise RuntimeContractError(
            f"GPU lock drift: all VinDr real runners must use {DEFAULT_GPU_LOCK.resolve()}"
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _require_sha(value: object, name: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise RuntimeContractError(f"{name} is not a lowercase SHA-256")
    return text


def validate_admission_gate(
    *,
    admission_path: Path,
    expected_admission_sha256: str,
    adjudication_handoff_path: Path,
    expected_adjudication_handoff_sha256: str,
    upstream_binary_ce_gate_path: Path,
    expected_upstream_binary_ce_gate_sha256: str,
    pack_manifest_path: Path,
    experiment_manifest_path: Path,
) -> dict[str, Any]:
    """Validate authorization before importing any model/runtime module."""

    expected = _require_sha(expected_admission_sha256, "expected admission hash")
    if not admission_path.is_file():
        raise RuntimeContractError("listing admission receipt is absent")
    actual = sha256_file(admission_path)
    if actual != expected:
        raise RuntimeContractError("listing admission receipt does not match externally pinned hash")
    try:
        strict = validate_scientific_admission(
            receipt_path=admission_path,
            expected_receipt_sha256=expected,
            handoff_path=adjudication_handoff_path,
            expected_handoff_sha256=expected_adjudication_handoff_sha256,
            upstream_gate_path=upstream_binary_ce_gate_path,
            expected_upstream_gate_sha256=expected_upstream_binary_ce_gate_sha256,
            pack_manifest_path=pack_manifest_path,
            experiment_manifest_path=experiment_manifest_path,
        )
    except ScientificAdmissionError as error:
        raise RuntimeContractError(f"strict listing admission failed: {error}") from error
    admission = strict["receipt"]
    if admission.get("schema_version") != ADMISSION_VERSION:
        raise RuntimeContractError("listing admission schema/version mismatch")
    if admission.get("status") != "independently_admitted_for_model_scoring":
        raise RuntimeContractError("listing admission is not in the frozen admitted state")
    required_true = (
        "four_independent_human_returns_validated",
        "listing_render_equivalence_admitted",
        "listing_prompt_equivalence_admitted",
        "adjudication_complete",
        "upstream_binary_ce_gate_authorized",
        "model_scoring_authorized",
        "gpu_authorized",
    )
    if any(admission.get(field) is not True for field in required_true):
        raise RuntimeContractError("listing admission lacks a required true authorization field")
    if admission.get("model_outputs_read_for_admission") is not False:
        raise RuntimeContractError("admission must be outcome-blind to model outputs")
    if admission.get("authorized_model_ids") != list(ALLOWED_MODELS):
        raise RuntimeContractError("admission must authorize exactly Huatuo and Hulu")
    if not pack_manifest_path.is_file() or not experiment_manifest_path.is_file():
        raise RuntimeContractError("frozen pack or experiment manifest is absent")
    expected_hashes = {
        "pack_manifest_sha256": sha256_file(pack_manifest_path),
        "experiment_manifest_sha256": sha256_file(experiment_manifest_path),
    }
    for field, observed in expected_hashes.items():
        if admission.get(field) != observed:
            raise RuntimeContractError(f"admission {field} does not match the frozen source")
    pack = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    experiment = json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
    if pack.get("version") != PACK_VERSION or experiment.get("schema_version") != MANIFEST_VERSION:
        raise RuntimeContractError("source protocol version drift")
    if admission.get("reference_file_sha256") != experiment.get("reference_contract", {}).get(
        "reference_file_sha256"
    ):
        raise RuntimeContractError("admission reference hash drift")
    failure_hash = pack.get("clinical_review", {}).get(
        "computational_guard_failure_pair_ids_sha256"
    )
    if admission.get("computational_guard_failure_pair_ids_sha256") != failure_hash:
        raise RuntimeContractError("admission computational-failure set drift")
    return {
        "sha256": actual,
        "receipt": admission,
        "pack": pack,
        "experiment": experiment,
        "adjudication_handoff_sha256": expected_adjudication_handoff_sha256,
        "upstream_binary_ce_gate_sha256": expected_upstream_binary_ce_gate_sha256,
    }


def guard_failed_images(pack_dir: Path, admitted_failure_set_sha256: str) -> list[dict[str, str]]:
    """Open the sealed engineering mapping only after admission succeeds."""

    mapping_path = pack_dir / "sealed_mapping.json"
    pack = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(mapping_path) != pack.get("artifact_sha256", {}).get("sealed_mapping.json"):
        raise RuntimeContractError("sealed mapping hash differs from the admitted pack manifest")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    failures = [
        {
            "pair_id": str(row["pair_id"]),
            "image_id": str(row["image_id"]),
            "render_id": str(row["transform"]),
            "reason": "pre_frozen_computational_guard_failure",
        }
        for row in mapping.get("clinical_pairs", [])
        if row.get("transform_guard", {}).get("clinical_guard_pass") is False
    ]
    failure_ids = sorted(item["pair_id"] for item in failures)
    if canonical_json_sha256(failure_ids) != admitted_failure_set_sha256:
        raise RuntimeContractError("sealed computational-failure identities do not match admission")
    if len(failures) != 1 or len({row["image_id"] for row in failures}) != 1:
        raise RuntimeContractError("frozen listing pack must contain exactly one invalid image")
    return failures


@dataclass(frozen=True)
class ParsedListing:
    raw_text: str
    valid: bool
    finding_ids: tuple[str, ...]
    raw_segments: tuple[str, ...]
    out_of_ontology: tuple[str, ...]
    duplicate_finding_ids: tuple[str, ...]
    violations: tuple[str, ...]
    is_empty: bool
    refusal: bool
    hedge: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "valid": self.valid,
            "finding_ids": list(self.finding_ids),
            "raw_segments": list(self.raw_segments),
            "out_of_ontology": list(self.out_of_ontology),
            "duplicate_finding_ids": list(self.duplicate_finding_ids),
            "violations": list(self.violations),
            "is_empty": self.is_empty,
            "refusal": self.refusal,
            "hedge": self.hedge,
        }


def parse_listing(text: str) -> ParsedListing:
    """Strict exact-label parser; no normalization or silent deletion."""

    raw = str(text)
    lower = raw.lower()
    refusal = any(phrase in lower for phrase in ("cannot determine", "can't determine", "unable to", "as an ai", "sorry"))
    hedge = any(phrase in lower for phrase in ("maybe", "possibly", "possible ", "uncertain", "cannot exclude", "may represent"))
    violations: list[str] = []
    if raw != raw.strip():
        violations.append("outer_whitespace")
    stripped = raw.strip()
    if stripped == NONE_TOKEN:
        return ParsedListing(raw, not violations, (), (stripped,), (), (), tuple(violations), True, refusal, hedge)
    if not stripped:
        violations.append("blank_output")
    segments = tuple(part.strip() for part in stripped.split(",")) if stripped else ()
    if any(not part for part in segments):
        violations.append("empty_segment")
    if NONE_TOKEN in segments:
        violations.append("mixed_empty_set_token")
    out = tuple(part for part in segments if part and part not in ID_BY_LABEL and part != NONE_TOKEN)
    if out:
        violations.append("out_of_ontology")
    ids = tuple(ID_BY_LABEL[part] for part in segments if part in ID_BY_LABEL)
    duplicates = tuple(sorted({finding for finding in ids if ids.count(finding) > 1}))
    if duplicates:
        violations.append("duplicate_label")
    if refusal:
        violations.append("refusal_surface")
    if hedge:
        violations.append("hedge_surface")
    return ParsedListing(
        raw, not violations, ids, segments, out, duplicates, tuple(dict.fromkeys(violations)), False, refusal, hedge
    )


@dataclass(frozen=True)
class Cell:
    cell_id: str
    render_id: str
    prompt_id: str
    prompt_text: str
    role: str


def cells(experiment: Mapping[str, Any]) -> tuple[Cell, ...]:
    output = tuple(
        Cell(
            str(row["cell_id"]), str(row["render_id"]), str(row["prompt_id"]),
            str(row["prompt_text"]), str(row["role"]),
        )
        for row in experiment["orbit_contract"]["cells"]
    )
    if len(output) != 19 or len({row.cell_id for row in output}) != 19:
        raise RuntimeContractError("listing orbit must contain exactly 19 unique cells")
    science = [row for row in output if row.role == "science_factorial"]
    if {(row.render_id, row.prompt_id) for row in science} != {
        (render, prompt) for render in SCIENCE_RENDERS for prompt in SCIENCE_PROMPTS
    }:
        raise RuntimeContractError("listing science product orbit drift")
    return output


class ListingAdapter(Protocol):
    model_id: str

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int, seed: int) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class NativeListingAdapter:
    """Lazy Huatuo/Hulu adapter preserving each checkpoint's native wrapper."""

    def __init__(self, model_id: str):
        if model_id not in ALLOWED_MODELS:
            raise RuntimeContractError(f"unsupported native listing model: {model_id}")
        # Deliberately imported only after admission validation.
        from corrected_sgta.models_oe import load_oe_adapter

        self.model_id = model_id
        self.backend = load_oe_adapter(model_id)

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int, seed: int) -> Mapping[str, Any]:
        generation = self.backend.generate_control(
            image, prompt, do_sample=False, temperature=1.0, top_p=1.0,
            num_beams=1, max_new_tokens=max_new_tokens, seed=seed,
        )
        return {
            "text": generation.text,
            "token_count": generation.token_count,
            "uncertainty": generation.uncertainty,
            "token_ids_sha256": canonical_json_sha256(list(generation.token_ids)),
            "decoding": "native_greedy",
        }

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


class FakeListingAdapter:
    """CPU-only deterministic test backend; never accepted as a scientific model."""

    model_id = "fake"

    def __init__(self, answers: Mapping[tuple[str, str], str] | None = None):
        self.answers = dict(answers or {})
        self.calls: list[tuple[str, str]] = []

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int, seed: int) -> Mapping[str, Any]:
        image_id = str(image.info.get("image_id", ""))
        prompt_id = str(image.info.get("prompt_id", ""))
        self.calls.append((image_id, prompt_id))
        text = self.answers.get((image_id, prompt_id), NONE_TOKEN)
        return {"text": text, "token_count": len(text.split()), "uncertainty": 0.0, "token_ids_sha256": canonical_json_sha256(text), "decoding": "fake_cpu_test_only"}

    def close(self) -> None:
        return None


def _boxes(bbox_csv: Path, image_ids: set[str]) -> dict[str, list[dict[str, float]]]:
    output: dict[str, list[dict[str, float]]] = defaultdict(list)
    with bbox_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image_id = str(row.get("image_id", ""))
            if image_id not in image_ids:
                continue
            raw = [str(row.get(name, "")).strip() for name in ("x_min", "y_min", "x_max", "y_max")]
            if all(raw):
                x0, y0, x1, y1 = map(float, raw)
                if x1 > x0 and y1 > y0:
                    output[image_id].append({"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1})
    return dict(output)


def native_render_provider(rows: Sequence[Mapping[str, Any]], experiment: Mapping[str, Any]) -> Callable[[Mapping[str, Any], Cell], Image.Image]:
    from corrected_sgta.run_huatuo_dicom_render_pilot_v1 import build_render_views, read_dicom_pixels

    image_root = Path(experiment["source"]["image_root"])
    boxes = _boxes(Path(experiment["source"]["bbox"]["path"]), {str(row["image_id"]) for row in rows})
    cache: dict[str, dict[str, Image.Image]] = {}

    def provide(row: Mapping[str, Any], cell: Cell) -> Image.Image:
        image_id = str(row["image_id"])
        if image_id not in cache:
            dicom = image_root / f"{image_id}.dicom"
            views = build_render_views(read_dicom_pixels(dicom), [], boxes.get(image_id, []))
            cache[image_id] = {str(view["name"]): view["image"] for view in views}
        return cache[image_id][cell.render_id].copy()

    return provide


def stable_seed(base: int, image_id: str, cell_id: str) -> int:
    return int(hashlib.sha256(f"{base}:{image_id}:{cell_id}".encode()).hexdigest()[:8], 16)


def _valid_shard(
    path: Path, fingerprint: str, image_id: str, cell: Cell, model_id: str | None = None
) -> bool:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    recorded_fingerprint = row.pop("fingerprint", None)
    generation = row.get("generation", {})
    exact_parsed = parse_listing(str(generation.get("text", ""))).as_dict()
    valid = bool(
        row.get("version") == VERSION
        and row.get("config_fingerprint") == fingerprint
        and row.get("model_id") in (*ALLOWED_MODELS, "fake")
        and (model_id is None or row.get("model_id") == model_id)
        and row.get("image_id") == image_id
        and row.get("cell_id") == cell.cell_id
        and row.get("render_id") == cell.render_id
        and row.get("prompt_id") == cell.prompt_id
        and row.get("cell_role") == cell.role
        and row.get("prompt_text_sha256") == hashlib.sha256(cell.prompt_text.encode()).hexdigest()
        and isinstance(generation.get("text"), str)
        and row.get("parsed") == exact_parsed
        and recorded_fingerprint == canonical_json_sha256(row)
    )
    return valid


def run_runtime(
    *,
    experiment_manifest_path: Path,
    pack_dir: Path,
    admission_path: Path,
    expected_admission_sha256: str,
    reference_path: Path,
    output_dir: Path,
    model_id: str,
    split: str,
    adjudication_handoff_path: Path | None = None,
    expected_adjudication_handoff_sha256: str | None = None,
    upstream_binary_ce_gate_path: Path | None = None,
    expected_upstream_binary_ce_gate_sha256: str | None = None,
    max_new_tokens: int = 96,
    seed: int = 42,
    adapter_factory: Callable[[str], ListingAdapter] | None = None,
    render_provider: Callable[[Mapping[str, Any], Cell], Image.Image] | None = None,
    gpu_lock_path: Path = DEFAULT_GPU_LOCK,
    lock_factory: Callable[[Path], Any] = gpu_flock,
) -> dict[str, Any]:
    _require_canonical_gpu_lock(model_id, gpu_lock_path)
    gate = validate_admission_gate(
        admission_path=admission_path,
        expected_admission_sha256=expected_admission_sha256,
        adjudication_handoff_path=adjudication_handoff_path or admission_path,
        expected_adjudication_handoff_sha256=(
            expected_adjudication_handoff_sha256 or expected_admission_sha256
        ),
        upstream_binary_ce_gate_path=upstream_binary_ce_gate_path or admission_path,
        expected_upstream_binary_ce_gate_sha256=(
            expected_upstream_binary_ce_gate_sha256 or expected_admission_sha256
        ),
        pack_manifest_path=pack_dir / "manifest.json",
        experiment_manifest_path=experiment_manifest_path,
    )
    experiment = gate["experiment"]
    if sha256_file(reference_path) != experiment["reference_contract"]["reference_file_sha256"]:
        raise RuntimeContractError("reference JSONL hash drift")
    failed = guard_failed_images(
        pack_dir,
        str(gate["receipt"]["computational_guard_failure_pair_ids_sha256"]),
    )
    excluded_ids = {row["image_id"] for row in failed}
    all_rows = load_jsonl(reference_path)
    selected = [row for row in all_rows if row.get("experiment_split") == split and row.get("image_id") not in excluded_ids]
    excluded_selected = [row for row in all_rows if row.get("experiment_split") == split and row.get("image_id") in excluded_ids]
    if split not in {"pilot", "dev", "confirmation"} or not selected:
        raise RuntimeContractError("unknown or empty split")
    if model_id not in (*ALLOWED_MODELS, "fake"):
        raise RuntimeContractError("unknown model ID")
    if model_id == "fake" and adapter_factory is None:
        raise RuntimeContractError("fake model requires an explicit test adapter factory")
    cell_specs = cells(experiment)
    config = {
        "version": VERSION,
        "model_id": model_id,
        "split": split,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "experiment_manifest_sha256": sha256_file(experiment_manifest_path),
        "reference_sha256": sha256_file(reference_path),
        "pack_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
        "admission_sha256": gate["sha256"],
        "adjudication_handoff_sha256": gate["adjudication_handoff_sha256"],
        "upstream_binary_ce_gate_sha256": gate["upstream_binary_ce_gate_sha256"],
        "runtime_source_sha256": sha256_file(Path(__file__)),
        "excluded_guard_failures": failed,
        "gpu_lock": str(gpu_lock_path.resolve()) if model_id in ALLOWED_MODELS else None,
    }
    fingerprint = canonical_json_sha256(config)
    config["fingerprint"] = fingerprint
    run_manifest = output_dir / "run_manifest.json"
    if run_manifest.exists():
        if json.loads(run_manifest.read_text()) != config:
            raise RuntimeContractError("existing run manifest differs; use a new output directory")
    else:
        atomic_json(run_manifest, config)

    # No factory call occurs before every admission/source/exclusion check above.
    completed = 0
    resumed = 0
    ordered_selected = sorted(selected, key=lambda value: str(value["image_id"]))
    planned = [
        (row, cell, output_dir / "cell_shards" / str(row["image_id"]) / f"{cell.cell_id}.json")
        for row in ordered_selected for cell in cell_specs
    ]
    existing = []
    for row, cell, path in planned:
        if path.exists():
            if not _valid_shard(path, fingerprint, str(row["image_id"]), cell, model_id):
                raise RuntimeContractError(f"invalid existing shard: {path}")
            existing.append(path)
    resumed = len(existing)
    if len(existing) == len(planned):
        execution_context = nullcontext()
    else:
        execution_context = lock_factory(gpu_lock_path) if model_id in ALLOWED_MODELS else nullcontext()
    with execution_context:
        # Native adapter construction is inside the singleton GPU lock.
        adapter = None if len(existing) == len(planned) else (adapter_factory or (lambda name: NativeListingAdapter(name)))(model_id)
        provider = None if adapter is None else (render_provider or native_render_provider(selected, experiment))
        try:
            for row in ordered_selected:
                image_id = str(row["image_id"])
                for cell in cell_specs:
                    path = output_dir / "cell_shards" / image_id / f"{cell.cell_id}.json"
                    if path.exists():
                        continue
                    assert adapter is not None and provider is not None
                    image = provider(row, cell)
                    image.info["image_id"] = image_id
                    image.info["prompt_id"] = cell.prompt_id
                    generation = dict(adapter.generate(image, cell.prompt_text, max_new_tokens, stable_seed(seed, image_id, cell.cell_id)))
                    parsed = parse_listing(str(generation.get("text", "")))
                    shard = {
                        "version": VERSION,
                        "config_fingerprint": fingerprint,
                        "model_id": model_id,
                        "split": split,
                        "image_id": image_id,
                        "cell_id": cell.cell_id,
                        "render_id": cell.render_id,
                        "prompt_id": cell.prompt_id,
                        "cell_role": cell.role,
                        "prompt_text_sha256": hashlib.sha256(cell.prompt_text.encode()).hexdigest(),
                        "generation": generation,
                        "parsed": parsed.as_dict(),
                    }
                    shard["fingerprint"] = canonical_json_sha256(shard)
                    atomic_json(path, shard)
                    completed += 1
        finally:
            if adapter is not None:
                adapter.close()
    shard_paths = sorted((output_dir / "cell_shards").glob("*/*.json"))
    expected_cells = len(selected) * len(cell_specs)
    if len(shard_paths) != expected_cells:
        raise RuntimeContractError("cell shard count does not match the complete eligible orbit plan")
    summary = {
        "version": VERSION,
        "status": "complete_eligible_orbits_only",
        "config_fingerprint": fingerprint,
        "eligible_images": len(selected),
        "excluded_guard_invalid_images": len(excluded_selected),
        "excluded_image_ids": sorted(excluded_ids & {str(row["image_id"]) for row in all_rows if row.get("experiment_split") == split}),
        "cells_per_orbit": len(cell_specs),
        "cell_shards": len(shard_paths),
        "shard_inventory": [{"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path)} for path in shard_paths],
        "guard_invalid_images_entering_complete_orbit": 0,
        "scientific_model": model_id in ALLOWED_MODELS,
    }
    summary["fingerprint"] = canonical_json_sha256(summary)
    completion = output_dir / "completion.json"
    if completion.exists() and json.loads(completion.read_text()) != summary:
        raise RuntimeContractError("existing completion inventory differs")
    if not completion.exists():
        atomic_json(completion, summary)
    return {**summary, "invocation": {"new_shards": completed, "resumed_shards": resumed}}


def _weighted_mean(values: Iterable[tuple[float, float]]) -> float:
    pairs = [(float(value), float(weight)) for value, weight in values]
    total = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total if total else math.nan


def _top_k(
    scores: Mapping[str, float], k: int, random_tie_key: str | None = None
) -> tuple[str, ...]:
    order = {finding_id: index for index, (finding_id, _) in enumerate(ONTOLOGY)}
    def tie(finding: str) -> object:
        if random_tie_key is None:
            return order[finding]
        return hashlib.sha256(f"{random_tie_key}:{finding}".encode()).hexdigest()

    return tuple(sorted(scores, key=lambda finding: (-float(scores[finding]), tie(finding)))[:k])


def _matched_coverage(
    scores: Sequence[tuple[str, Mapping[str, float]]], target: int, random_ties: bool = False
) -> dict[str, tuple[str, ...]]:
    order = {finding_id: index for index, (finding_id, _) in enumerate(ONTOLOGY)}
    ranked = sorted(
        (
            (
                float(value), image_id,
                hashlib.sha256(f"20260803:{image_id}:{finding}".encode()).hexdigest()
                if random_ties else str(order[finding]).zfill(3),
                finding,
            )
            for image_id, row in scores for finding, value in row.items()
        ),
        key=lambda item: (-item[0], item[2], item[1]),
    )
    selected: dict[str, list[str]] = defaultdict(list)
    for _, image_id, _, finding in ranked[:target]:
        selected[image_id].append(finding)
    return {image_id: tuple(values) for image_id, values in selected.items()}


def evaluate_outputs(
    *, reference_rows: Sequence[Mapping[str, Any]], shard_rows: Sequence[Mapping[str, Any]], mode: str = "fixed_k"
) -> dict[str, Any]:
    """Evaluate preserved direct outputs and matched-content projections."""

    if mode not in {"fixed_k", "matched_coverage"}:
        raise ValueError("mode must be fixed_k or matched_coverage")
    refs = {str(row["image_id"]): row for row in reference_rows}
    by_image: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = defaultdict(dict)
    format_rows = []
    for shard in shard_rows:
        image_id = str(shard["image_id"])
        by_image[image_id][(str(shard["render_id"]), str(shard["prompt_id"]))] = shard
        format_rows.append(shard["parsed"])
    # Intention-to-evaluate: parser failure is an outcome, never an exclusion.
    # A clinical orbit requires all 15 scheduled science shards, but recognized
    # exact ontology atoms inside a malformed surface remain visible to claim
    # metrics while its parser failure is reported on a separate axis.
    complete: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    required_cells = {(render, prompt) for render in SCIENCE_RENDERS for prompt in SCIENCE_PROMPTS}
    for image_id, rows in by_image.items():
        if required_cells <= set(rows):
            complete[image_id] = rows

    scores_by_method: dict[str, list[tuple[str, dict[str, float]]]] = defaultdict(list)
    canonical_selected: dict[str, tuple[str, ...]] = {}
    canonical_surface: dict[str, Mapping[str, Any]] = {}
    canonical_generation: dict[str, Mapping[str, Any]] = {}
    for image_id, rows in complete.items():
        memberships = {
            key: {finding: float(finding in set(row["parsed"]["finding_ids"])) for finding, _ in ONTOLOGY}
            for key, row in rows.items() if key in required_cells
        }
        canonical_shard = rows[(SCIENCE_RENDERS[0], SCIENCE_PROMPTS[0])]
        canonical = canonical_shard["parsed"]
        canonical_selected[image_id] = tuple(canonical["finding_ids"])
        canonical_surface[image_id] = canonical
        canonical_generation[image_id] = canonical_shard.get("generation", {})
        grand = {f: sum(m[f] for m in memberships.values()) / 15.0 for f, _ in ONTOLOGY}
        render_marginal = {f: sum(memberships[(r, SCIENCE_PROMPTS[0])][f] for r in SCIENCE_RENDERS) / 5.0 for f, _ in ONTOLOGY}
        prompt_marginal = {f: sum(memberships[(SCIENCE_RENDERS[0], p)][f] for p in SCIENCE_PROMPTS) / 3.0 for f, _ in ONTOLOGY}
        additive = {f: prompt_marginal[f] + render_marginal[f] - grand[f] for f, _ in ONTOLOGY}
        for method, score in (
            ("canonical", memberships[(SCIENCE_RENDERS[0], SCIENCE_PROMPTS[0])]),
            ("render_marginal", render_marginal),
            ("prompt_marginal", prompt_marginal),
            ("full_orbit_mean", grand),
            ("full_orbit_mean_random_ties", grand),
            ("additive_projection", additive),
        ):
            scores_by_method[method].append((image_id, score))
    selections: dict[str, dict[str, tuple[str, ...]]] = {}
    total_k = sum(len(values) for values in canonical_selected.values())
    for method, rows in scores_by_method.items():
        if method == "canonical":
            selections[method] = canonical_selected
        elif mode == "fixed_k":
            selections[method] = {
                image_id: _top_k(
                    score, len(canonical_selected[image_id]),
                    f"20260803:{image_id}" if method == "full_orbit_mean_random_ties" else None,
                )
                for image_id, score in rows
            }
        else:
            selections[method] = _matched_coverage(
                rows, total_k, random_ties=method == "full_orbit_mean_random_ties"
            )
    metrics: dict[str, Any] = {}
    for method, chosen in selections.items():
        counts = defaultdict(float)
        weighted = defaultdict(float)
        image_weights = 0.0
        brier_values = []
        for image_id in complete:
            ref = refs[image_id]
            selected = set(chosen.get(image_id, ()))
            claims = {str(row["finding_id"]): row for row in ref["claims"]}
            weight = float(ref.get("inverse_sampling_weight", 1.0))
            image_weights += weight
            counts["selected"] += len(selected)
            weighted["selected"] += weight * len(selected)
            for finding, claim in claims.items():
                present = finding in selected
                votes = int(claim["positive_votes"])
                if present and votes == 0:
                    counts["fabricated"] += 1; weighted["fabricated"] += weight
                if present and votes in (1, 2):
                    counts["overcommitted"] += 1; weighted["overcommitted"] += weight
                if votes == 3:
                    counts["required"] += 1; weighted["required"] += weight
                    if present:
                        counts["required_hit"] += 1; weighted["required_hit"] += weight
                brier_values.append(((float(present) - float(claim["reader_support"])) ** 2, weight))
            if not selected:
                counts["negative_images"] += 1; weighted["negative_images"] += weight
        selected_n = counts["selected"]
        required_n = counts["required"]
        metrics[method] = {
            "images": len(complete),
            "selected_claims": int(selected_n),
            "mean_claim_count": selected_n / len(complete) if complete else math.nan,
            "positive_hallucination_rate": counts["fabricated"] / selected_n if selected_n else 0.0,
            "disagreement_overcommitment_rate": counts["overcommitted"] / selected_n if selected_n else 0.0,
            "required_omission_rate": 1.0 - counts["required_hit"] / required_n if required_n else 0.0,
            "supported_precision": counts["required_hit"] / selected_n if selected_n else 1.0,
            "reader_distribution_brier": sum(value for value, _ in brier_values) / len(brier_values) if brier_values else math.nan,
            "weighted_reader_distribution_brier": _weighted_mean(brier_values),
            "negative_output_rate": counts["negative_images"] / len(complete) if complete else math.nan,
            "weighted_positive_hallucination_rate": weighted["fabricated"] / weighted["selected"] if weighted["selected"] else 0.0,
            "weighted_required_omission_rate": 1.0 - weighted["required_hit"] / weighted["required"] if weighted["required"] else 0.0,
            "weighted_mean_claim_count": weighted["selected"] / image_weights if image_weights else math.nan,
            "claim_budget_delta_from_canonical": int(selected_n - total_k),
            "content_budget_preserved": int(selected_n) == total_k,
            # Content projections do not get to erase malformed canonical
            # surfaces; these jointly reported risks remain bound to every arm.
            "preserved_canonical_format_violation_rate": sum(
                not bool(row["valid"]) for row in canonical_surface.values()
            ) / len(canonical_surface) if canonical_surface else math.nan,
            "preserved_canonical_out_of_ontology_rate": sum(
                bool(row["out_of_ontology"]) for row in canonical_surface.values()
            ) / len(canonical_surface) if canonical_surface else math.nan,
            "preserved_canonical_refusal_rate": sum(
                bool(row["refusal"]) for row in canonical_surface.values()
            ) / len(canonical_surface) if canonical_surface else math.nan,
            "preserved_canonical_hedge_rate": sum(
                bool(row["hedge"]) for row in canonical_surface.values()
            ) / len(canonical_surface) if canonical_surface else math.nan,
            "preserved_canonical_mean_character_length": sum(
                len(str(row["raw_text"])) for row in canonical_surface.values()
            ) / len(canonical_surface) if canonical_surface else math.nan,
            "preserved_canonical_mean_generated_token_count": sum(
                int(canonical_generation[image_id].get(
                    "token_count", len(str(canonical_surface[image_id]["raw_text"]).split())
                )) for image_id in canonical_surface
            ) / len(canonical_surface) if canonical_surface else math.nan,
        }
    science_parser_rows = [
        rows[key]["parsed"] for rows in complete.values() for key in required_cells
    ]
    result = {
        "version": "vindr-cecd-listing-evaluation-v1",
        "mode": mode,
        "all_images_with_any_shard": len(by_image),
        "intention_to_evaluate_science_orbits": len(complete),
        "missing_science_shard_orbits": len(by_image) - len(complete),
        "orbits_with_any_parser_failure": sum(
            any(not bool(rows[key]["parsed"]["valid"]) for key in required_cells)
            for rows in complete.values()
        ),
        "all_cell_format_violation_rate": sum(not bool(row["valid"]) for row in format_rows) / len(format_rows) if format_rows else math.nan,
        "all_cell_out_of_ontology_rate": sum(bool(row["out_of_ontology"]) for row in format_rows) / len(format_rows) if format_rows else math.nan,
        "all_cell_refusal_rate": sum(bool(row["refusal"]) for row in format_rows) / len(format_rows) if format_rows else math.nan,
        "all_cell_hedge_rate": sum(bool(row["hedge"]) for row in format_rows) / len(format_rows) if format_rows else math.nan,
        "science_cell_exact_parse_rate": sum(bool(row["valid"]) for row in science_parser_rows) / len(science_parser_rows) if science_parser_rows else math.nan,
        "science_cell_out_of_ontology_rate": sum(bool(row["out_of_ontology"]) for row in science_parser_rows) / len(science_parser_rows) if science_parser_rows else math.nan,
        "science_cell_refusal_rate": sum(bool(row["refusal"]) for row in science_parser_rows) / len(science_parser_rows) if science_parser_rows else math.nan,
        "science_cell_hedge_rate": sum(bool(row["hedge"]) for row in science_parser_rows) / len(science_parser_rows) if science_parser_rows else math.nan,
        "canonical_mean_character_length": sum(len(str(row["raw_text"])) for row in canonical_surface.values()) / len(canonical_surface) if canonical_surface else math.nan,
        "canonical_claim_budget": total_k,
        "tie_breaker_policy": {
            "primary": "frozen ontology ID order",
            "random_control": "SHA256(20260803,image_id,finding_id) within exact score ties",
        },
        "methods": metrics,
    }
    result["fingerprint"] = canonical_json_sha256(result)
    return result


def evaluate_run(
    *,
    experiment_manifest_path: Path,
    reference_path: Path,
    run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Verify the completion-bound shard inventory, then emit both protocols."""

    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config_fingerprint = str(run_manifest.get("fingerprint", ""))
    config_body = dict(run_manifest)
    config_body.pop("fingerprint", None)
    if config_fingerprint != canonical_json_sha256(config_body):
        raise RuntimeContractError("run manifest fingerprint mismatch")
    if sha256_file(experiment_manifest_path) != run_manifest.get("experiment_manifest_sha256"):
        raise RuntimeContractError("evaluator experiment manifest differs from run")
    if sha256_file(reference_path) != run_manifest.get("reference_sha256"):
        raise RuntimeContractError("evaluator reference differs from run")
    experiment = json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
    specs = {cell.cell_id: cell for cell in cells(experiment)}
    completion = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
    completion_fingerprint = str(completion.get("fingerprint", ""))
    completion_body = dict(completion)
    completion_body.pop("fingerprint", None)
    if completion_fingerprint != canonical_json_sha256(completion_body):
        raise RuntimeContractError("completion fingerprint mismatch")
    if completion.get("config_fingerprint") != config_fingerprint:
        raise RuntimeContractError("completion is not bound to run config")
    inventory = completion.get("shard_inventory")
    if not isinstance(inventory, list) or len(inventory) != int(completion.get("cell_shards", -1)):
        raise RuntimeContractError("completion shard inventory malformed")
    listed_paths = [run_dir / str(row["path"]) for row in inventory]
    actual_paths = sorted((run_dir / "cell_shards").glob("*/*.json"))
    if sorted(path.resolve() for path in listed_paths) != sorted(path.resolve() for path in actual_paths):
        raise RuntimeContractError("completion inventory does not exactly cover cell shards")
    shard_rows = []
    for item, path in zip(inventory, listed_paths):
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise RuntimeContractError(f"completion-bound shard hash mismatch: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        cell_id = str(row.get("cell_id", ""))
        if cell_id not in specs or not _valid_shard(
            path, config_fingerprint, str(row.get("image_id", "")), specs[cell_id], str(run_manifest["model_id"])
        ):
            raise RuntimeContractError(f"invalid completion-bound shard: {path}")
        shard_rows.append(row)
    reference_rows = load_jsonl(reference_path)
    source_closure = {
        "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "completion_sha256": sha256_file(run_dir / "completion.json"),
        "config_fingerprint": config_fingerprint,
        "shard_inventory_fingerprint": canonical_json_sha256(inventory),
        "model_id": run_manifest["model_id"],
        "split": run_manifest["split"],
    }
    outputs = {}
    for mode in ("fixed_k", "matched_coverage"):
        result = evaluate_outputs(reference_rows=reference_rows, shard_rows=shard_rows, mode=mode)
        result["source_closure"] = source_closure
        result["fingerprint"] = canonical_json_sha256({key: value for key, value in result.items() if key != "fingerprint"})
        path = output_dir / f"{mode}.json"
        write_once_json(path, result)
        outputs[mode] = {"path": str(path), "sha256": sha256_file(path), "fingerprint": result["fingerprint"]}
    index = {
        "version": "vindr-cecd-listing-evaluator-index-v1",
        "status": "both_content_budget_protocols_complete",
        "source_closure": source_closure,
        "outputs": outputs,
    }
    index["fingerprint"] = canonical_json_sha256(index)
    write_once_json(output_dir / "evaluation_index.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--experiment-manifest", type=Path, required=True)
    run.add_argument("--pack-dir", type=Path, required=True)
    run.add_argument("--admission", type=Path, required=True)
    run.add_argument("--expected-admission-sha256", required=True)
    run.add_argument("--adjudication-handoff", type=Path, required=True)
    run.add_argument("--expected-adjudication-handoff-sha256", required=True)
    run.add_argument("--upstream-binary-ce-gate", type=Path, required=True)
    run.add_argument("--expected-upstream-binary-ce-gate-sha256", required=True)
    run.add_argument("--reference", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--model", choices=ALLOWED_MODELS, required=True)
    run.add_argument("--split", choices=("pilot", "dev", "confirmation"), required=True)
    run.add_argument("--max-new-tokens", type=int, default=96)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--experiment-manifest", type=Path, required=True)
    evaluate.add_argument("--reference", type=Path, required=True)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_runtime(
            experiment_manifest_path=args.experiment_manifest,
            pack_dir=args.pack_dir,
            admission_path=args.admission,
            expected_admission_sha256=args.expected_admission_sha256,
            adjudication_handoff_path=args.adjudication_handoff,
            expected_adjudication_handoff_sha256=args.expected_adjudication_handoff_sha256,
            upstream_binary_ce_gate_path=args.upstream_binary_ce_gate,
            expected_upstream_binary_ce_gate_sha256=args.expected_upstream_binary_ce_gate_sha256,
            reference_path=args.reference,
            output_dir=args.output_dir,
            model_id=args.model,
            split=args.split,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            gpu_lock_path=args.gpu_lock,
        )
    else:
        result = evaluate_run(
            experiment_manifest_path=args.experiment_manifest,
            reference_path=args.reference,
            run_dir=args.run_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
