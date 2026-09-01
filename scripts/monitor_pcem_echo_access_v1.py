#!/usr/bin/env python3
"""Persistently wait for MIMIC-IV-ECHO and run the PCEM CPU data gate.

The monitor never authenticates, downloads protected data, downloads images,
or launches a model.  It only observes explicitly allowed local mount paths and
runs the outcome-blind temporal-join audit after the file is stable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path("/home/dbw/ANCHOR")
DATA_ROOT = Path("/home/dbw/datasets/physionet")
DEFAULT_CANDIDATES = (
    DATA_ROOT / "mimic-iv-echo/1.0/structured_measurement.csv.gz",
    DATA_ROOT / "mimic-iv-echo/1.0/structured_measurement.csv",
    DATA_ROOT / "physionet.org/files/mimic-iv-echo/1.0/structured_measurement.csv.gz",
    DATA_ROOT / "physionet.org/files/mimic-iv-echo/1.0/structured_measurement.csv",
)
DEFAULT_METADATA = DATA_ROOT / "pcem_metadata_v1/mimic-cxr-2.0.0-metadata.csv.gz"
DEFAULT_SPLIT = DATA_ROOT / "pcem_metadata_v1/mimic-cxr-2.0.0-split.csv.gz"
DEFAULT_OUTPUT = ROOT / "corrected_runs/pcem_echo_temporal_join_v1/audit.json"
DEFAULT_HEARTBEAT = ROOT / "corrected_runs/pcem_echo_temporal_join_v1/monitor.heartbeat.json"
DEFAULT_TERMINAL = ROOT / "corrected_runs/pcem_echo_temporal_join_v1/monitor.terminal.json"
PROTOCOL_ID = "pcem-echo-temporal-join-audit-v1"
MONITOR_VERSION = "pcem-echo-access-monitor-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def within_root(path: Path, root: Path = DATA_ROOT) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def resolve_echo_candidate(
    candidates: Sequence[Path], *, root: Path = DATA_ROOT
) -> Path | None:
    existing: dict[Path, Path] = {}
    for candidate in candidates:
        if not within_root(candidate, root):
            raise ValueError(f"echo candidate escapes protected data root: {candidate}")
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        existing.setdefault(resolved, candidate)
    if not existing:
        return None
    if len(existing) != 1:
        raise ValueError(
            "multiple MIMIC-IV-ECHO structured files found; select exactly one "
            f"with --echo-candidate: {[str(path) for path in sorted(existing)]}"
        )
    return next(iter(existing))


def file_observation(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_existing_audit(path: Path, echo_path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        row = load_json(path)
        return (
            row.get("protocol_id") == PROTOCOL_ID
            and Path(row["inputs"]["echo_structured"]).resolve() == echo_path.resolve()
            and row["echo_schema"]["sha256"]
            == _sha256(echo_path)
            and row["admission"]["gpu_authorized"] is False
            and row["admission"]["image_download_authorized"] is False
            and row["admission"]["independent_heart_size_truth_identified"] is False
            and isinstance(row.get("fingerprint"), str)
            and len(row["fingerprint"]) == 64
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def heartbeat_payload(stage: str, **extra: Any) -> dict[str, Any]:
    return {
        "version": MONITOR_VERSION,
        "time": utc_now(),
        "pid": os.getpid(),
        "stage": stage,
        "authentication_attempted": False,
        "protected_data_downloaded_by_monitor": False,
        "image_download_authorized": False,
        "gpu_authorized": False,
        **extra,
    }


def run_audit(
    *, metadata: Path, split: Path, echo: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "anchor/corrected_sgta/audit_pcem_echo_join_v1.py"),
        "--cxr-metadata",
        str(metadata),
        "--cxr-split",
        str(split),
        "--echo-structured",
        str(echo),
        "--output",
        str(output),
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--echo-candidate", type=Path, action="append")
    parser.add_argument("--cxr-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cxr-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--heartbeat", type=Path, default=DEFAULT_HEARTBEAT)
    parser.add_argument("--terminal", type=Path, default=DEFAULT_TERMINAL)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--stability-seconds", type=float, default=120.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError("interval must be at least one second")
    if args.stability_seconds < 0:
        raise ValueError("stability-seconds must be nonnegative")
    for path in (args.cxr_metadata, args.cxr_split):
        if not path.is_file():
            raise FileNotFoundError(path)
        if not within_root(path):
            raise ValueError(f"input escapes protected data root: {path}")

    candidates = tuple(args.echo_candidate or DEFAULT_CANDIDATES)
    last_observation: tuple[Path, tuple[int, int]] | None = None
    stable_since = 0.0
    while True:
        try:
            echo = resolve_echo_candidate(candidates)
        except ValueError as error:
            payload = heartbeat_payload(
                "ambiguous_echo_mount",
                error=str(error),
                candidates=[str(path) for path in candidates],
            )
            atomic_json(args.heartbeat, payload)
            atomic_json(args.terminal, {**payload, "terminal": True, "exit_code": 2})
            raise
        if echo is None:
            atomic_json(
                args.heartbeat,
                heartbeat_payload(
                    "waiting_for_authorized_echo_mount",
                    candidates=[str(path) for path in candidates],
                    output=str(args.output),
                ),
            )
            if args.once:
                return
            time.sleep(args.interval)
            continue

        if valid_existing_audit(args.output, echo):
            audit = load_json(args.output)
            payload = heartbeat_payload(
                "completed_cpu_data_gate",
                echo_structured=str(echo),
                output=str(args.output),
                fingerprint=audit["fingerprint"],
                decision=audit["admission"]["decision"],
                construct_review_required=audit["admission"]["construct_review_required"],
            )
            atomic_json(args.heartbeat, payload)
            atomic_json(args.terminal, {**payload, "terminal": True, "exit_code": 0})
            return

        observation = file_observation(echo)
        now = time.monotonic()
        if last_observation != (echo, observation):
            last_observation = (echo, observation)
            stable_since = now
        stable_for = now - stable_since
        if stable_for < args.stability_seconds:
            atomic_json(
                args.heartbeat,
                heartbeat_payload(
                    "waiting_for_stable_echo_file",
                    echo_structured=str(echo),
                    bytes=observation[0],
                    mtime_ns=observation[1],
                    stable_seconds=stable_for,
                    required_stability_seconds=args.stability_seconds,
                ),
            )
            if args.once:
                return
            time.sleep(args.interval)
            continue

        atomic_json(
            args.heartbeat,
            heartbeat_payload(
                "running_cpu_schema_and_temporal_join",
                echo_structured=str(echo),
                bytes=observation[0],
                output=str(args.output),
            ),
        )
        result = run_audit(
            metadata=args.cxr_metadata,
            split=args.cxr_split,
            echo=echo,
            output=args.output,
        )
        if result.returncode != 0 or not valid_existing_audit(args.output, echo):
            payload = heartbeat_payload(
                "cpu_data_gate_failed",
                echo_structured=str(echo),
                output=str(args.output),
                returncode=result.returncode,
                stdout=result.stdout[-4000:],
                stderr=result.stderr[-4000:],
            )
            atomic_json(args.heartbeat, payload)
            atomic_json(
                args.terminal,
                {**payload, "terminal": True, "exit_code": result.returncode or 3},
            )
            raise RuntimeError("PCEM echo CPU data gate failed; see monitor terminal")

        # The next loop validates and records the terminal decision from the
        # immutable audit rather than trusting subprocess stdout.


if __name__ == "__main__":
    main()
