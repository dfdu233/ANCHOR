"""Run manifests and strict resumable prediction storage."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .hashing import canonical_json, sha256_json
from .schema import PredictionRecord


PROTOCOL_ID = "anchor-eval-contract-v1"


@dataclass(frozen=True)
class RunManifest:
    protocol_id: str
    track: str
    task: dict[str, Any]
    model: dict[str, Any]
    method: dict[str, Any]
    generation: dict[str, Any]
    evaluator: dict[str, Any]
    ordered_samples_sha256: str
    ordered_images_sha256: str
    code: dict[str, Any]
    runtime: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def compare_manifests(
    left: RunManifest, right: RunManifest, allowed_differences: tuple[str, ...] = ("method",)
) -> dict[str, dict[str, Any]]:
    left_payload, right_payload = asdict(left), asdict(right)
    return {
        key: {"left": left_payload[key], "right": right_payload[key]}
        for key in left_payload
        if key not in allowed_differences and left_payload[key] != right_payload[key]
    }


class PredictionStore:
    def __init__(self, root: Path, manifest: RunManifest) -> None:
        self.root = root
        self.manifest = manifest
        self.manifest_path = root / "run_manifest.json"
        self.predictions_path = root / "predictions.jsonl"
        root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            prior = json.loads(self.manifest_path.read_text())
            if sha256_json(prior) != sha256_json(asdict(manifest)):
                raise ValueError("refusing to resume: run manifest fingerprint changed")
        else:
            atomic_write_json(self.manifest_path, asdict(manifest))
        self._records = self._read_index()

    def _read_index(self) -> dict[str, PredictionRecord]:
        records: dict[str, PredictionRecord] = {}
        if not self.predictions_path.exists():
            return records
        for number, line in enumerate(self.predictions_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = PredictionRecord(**json.loads(line))
            if record.run_fingerprint != self.manifest.fingerprint:
                raise ValueError(f"foreign run fingerprint at prediction line {number}")
            if record.sample_id in records:
                raise ValueError(f"duplicate prediction sample_id {record.sample_id!r}")
            records[record.sample_id] = record
        return records

    def cached(self, sample_id: str, sample_fingerprint: str) -> PredictionRecord | None:
        record = self._records.get(sample_id)
        if record is None:
            return None
        if record.sample_fingerprint != sample_fingerprint:
            raise ValueError(f"sample fingerprint changed for {sample_id}")
        return record

    def append(self, record: PredictionRecord) -> None:
        if record.run_fingerprint != self.manifest.fingerprint:
            raise ValueError("prediction run fingerprint mismatch")
        if record.sample_id in self._records:
            raise ValueError(f"duplicate prediction sample_id {record.sample_id!r}")
        with self.predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(asdict(record)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records[record.sample_id] = record

    def validate_complete(self, expected: Iterable[tuple[str, str]]) -> dict[str, Any]:
        expected_map = dict(expected)
        missing = sorted(set(expected_map) - set(self._records))
        extra = sorted(set(self._records) - set(expected_map))
        changed = sorted(
            sample_id for sample_id in set(expected_map) & set(self._records)
            if expected_map[sample_id] != self._records[sample_id].sample_fingerprint
        )
        errors = sorted(
            sample_id for sample_id, record in self._records.items()
            if record.status != "ok"
        )
        return {
            "complete": not (missing or extra or changed or errors),
            "expected": len(expected_map),
            "observed": len(self._records),
            "missing": missing,
            "extra": extra,
            "changed": changed,
            "errors": errors,
        }
