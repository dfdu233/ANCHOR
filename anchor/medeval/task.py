"""YAML-backed task adapter with deterministic, interleaved messages."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import yaml

from .schema import EvalSample, TaskKind, TaskSpec


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("samples", "records", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"unsupported record container in {path}")


def load_task_spec(path: Path) -> TaskSpec:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"task config must be a mapping: {path}")
    allowed = {item.name for item in fields(TaskSpec)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown task keys in {path}: {unknown}")
    payload["kind"] = TaskKind(payload["kind"])
    return TaskSpec(**payload)


def _image_values(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, list) else [value]
    paths = tuple(str(item) for item in values if str(item).strip())
    if not paths:
        raise ValueError("empty image field")
    return paths


def materialize_samples(spec: TaskSpec) -> list[EvalSample]:
    records = load_records(Path(spec.dataset_path))
    samples: list[EvalSample] = []
    seen: set[str] = set()
    for index, row in enumerate(records):
        raw_id = row.get(spec.sample_id_field, index)
        sample_id = str(raw_id)
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        images = _image_values(row.get(spec.image_field))
        question = str(row.get(spec.question_field, "")).strip()
        prompt = spec.prompt_template.format(question=question)
        cluster = row.get(spec.cluster_field) if spec.cluster_field else sample_id
        samples.append(EvalSample(
            sample_id=sample_id,
            image_paths=images,
            question=prompt,
            reference=row.get(spec.reference_field),
            cluster_id=str(cluster),
            metadata={"source_index": index},
        ))
    return samples


def doc_to_messages(sample: EvalSample) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [
        {"type": "image", "path": image_path} for image_path in sample.image_paths
    ]
    content.append({"type": "text", "text": sample.question})
    return [{"role": "user", "content": content}]


def validate_images(samples: Iterable[EvalSample], image_root: Path) -> list[Path]:
    resolved: list[Path] = []
    for sample in samples:
        for raw in sample.image_paths:
            path = Path(raw)
            candidate = path if path.is_absolute() else image_root / path
            if not candidate.is_file():
                raise FileNotFoundError(f"missing image for {sample.sample_id}: {candidate}")
            resolved.append(candidate.resolve())
    return resolved
