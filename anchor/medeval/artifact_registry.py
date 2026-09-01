#!/usr/bin/env python3
"""Append-only qualification registry for historical evaluation artifacts."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import canonical_json, sha256_file, sha256_json


VERSION = "artifact-provenance-registry-v1"
STATUSES = {
    "admissible",
    "rescore_only",
    "identity_only",
    "regenerate",
    "not_admissible",
    "failed_cutoff",
}


@dataclass(frozen=True)
class ArtifactQualification:
    artifact_path: str
    artifact_sha256: str
    status: str
    evaluator_version: str
    evidence_scope: str
    reason: str
    qualification_path: str | None = None
    qualification_sha256: str | None = None
    supersedes: str | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unsupported artifact status: {self.status}")
        if not all((self.artifact_path, self.artifact_sha256, self.evaluator_version)):
            raise ValueError("artifact path/hash and evaluator version are required")

    @property
    def event_id(self) -> str:
        return sha256_json(asdict(self))


def qualification_for(
    artifact: Path,
    *,
    status: str,
    evaluator_version: str,
    evidence_scope: str,
    reason: str,
    qualification: Path | None = None,
    supersedes: str | None = None,
) -> ArtifactQualification:
    artifact = artifact.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    qualification = qualification.resolve() if qualification is not None else None
    if qualification is not None and not qualification.is_file():
        raise FileNotFoundError(qualification)
    return ArtifactQualification(
        artifact_path=str(artifact),
        artifact_sha256=sha256_file(artifact),
        status=status,
        evaluator_version=evaluator_version,
        evidence_scope=evidence_scope,
        reason=reason,
        qualification_path=None if qualification is None else str(qualification),
        qualification_sha256=None if qualification is None else sha256_file(qualification),
        supersedes=supersedes,
    )


def append_qualification(registry: Path, value: ArtifactQualification) -> dict[str, Any]:
    """Append one immutable event; exact retries are idempotent."""

    registry = registry.resolve()
    registry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry.with_suffix(registry.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = []
        if registry.exists():
            for number, line in enumerate(registry.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("registry_version") != VERSION:
                    raise ValueError(f"foreign registry version at line {number}")
                rows.append(row)
        for row in rows:
            if row["event_id"] == value.event_id:
                return row
        payload = {
            "registry_version": VERSION,
            "event_id": value.event_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **asdict(value),
        }
        with registry.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return payload


def latest_by_artifact(registry: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not registry.exists():
        return latest
    for line in registry.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["artifact_path"]] = row
    return latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--status", choices=sorted(STATUSES), required=True)
    parser.add_argument("--evaluator-version", required=True)
    parser.add_argument("--evidence-scope", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--supersedes")
    args = parser.parse_args()
    value = qualification_for(
        args.artifact,
        status=args.status,
        evaluator_version=args.evaluator_version,
        evidence_scope=args.evidence_scope,
        reason=args.reason,
        qualification=args.qualification,
        supersedes=args.supersedes,
    )
    print(json.dumps(append_qualification(args.registry, value), indent=2))


if __name__ == "__main__":
    main()
