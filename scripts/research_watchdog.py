#!/usr/bin/env python3
"""Recover the explicitly active detached research pipeline after shell loss.

The watchdog deliberately does not scan every historical state file.  Only
jobs named in the active manifest are eligible for recovery, and failed jobs
remain failed for inspection instead of being retried blindly.
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
from typing import Any


ROOT = Path("/home/dbw/ANCHOR")
RESTART_MARKER = ROOT / "corrected_runs/detached_jobs/container-restart-requested.json"
REQUIRED_MANIFEST = ROOT / "configs/research_required_jobs.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid: object) -> bool:
    try:
        value = int(pid)
        state = Path(f"/proc/{value}/stat").read_text().split()[2]
        return state != "Z"
    except (ValueError, OSError, IndexError, TypeError):
        return False


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merged_job_names(
    manifest: dict[str, Any], required: dict[str, Any] | None = None
) -> list[str]:
    names = manifest.get("active_jobs")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("active_jobs must be a list of job names")
    protected = [] if required is None else required.get("required_jobs")
    if not isinstance(protected, list) or not all(
        isinstance(name, str) for name in protected
    ):
        raise ValueError("required_jobs must be a list of job names")
    return list(dict.fromkeys([*names, *protected]))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def recovery_decision(row: dict[str, Any]) -> str:
    """Return one of alive, recover, terminal, failed, or invalid."""
    status = str(row.get("status", "missing"))
    process = row.get("child_pid", row.get("pid"))
    if status in {"done"}:
        return "terminal"
    if status == "failed":
        return "failed"
    if status not in {"starting", "running"}:
        return "invalid"
    if pid_alive(process):
        return "alive"
    command = row.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(token, str) for token in command
    ):
        return "invalid"
    return "recover"


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def restart_job(state_path: Path, row: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    cwd = Path(str(row.get("cwd", ROOT)))
    log = Path(str(row["log"]))
    command = [
        sys.executable,
        str(ROOT / "scripts/start_detached_job.py"),
        "--name",
        str(row["name"]),
        "--log",
        str(log),
        "--state",
        str(state_path),
        "--",
        *row["command"],
    ]
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/research_active_jobs.json",
    )
    parser.add_argument(
        "--heartbeat",
        type=Path,
        default=ROOT / "corrected_runs/detached_jobs/watchdog-heartbeat.json",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=ROOT / "corrected_runs/detached_jobs/watchdog-events.jsonl",
    )
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError("interval must be at least one second")

    while True:
        manifest = load_json(args.manifest)
        required = (
            load_json(REQUIRED_MANIFEST)
            if REQUIRED_MANIFEST.is_file()
            else {"required_jobs": []}
        )
        names = merged_job_names(manifest, required)
        decisions: dict[str, str] = {}
        if RESTART_MARKER.exists():
            decisions = {name: "paused-for-container-restart" for name in names}
            heartbeat = {
                "time": utc_now(),
                "watchdog_pid": os.getpid(),
                "manifest": str(args.manifest.resolve()),
                "restart_marker": str(RESTART_MARKER),
                "decisions": decisions,
            }
            atomic_json(args.heartbeat, heartbeat)
            if args.once:
                return
            time.sleep(args.interval)
            continue
        for name in names:
            state_path = ROOT / "corrected_runs/detached_jobs" / f"{name}.json"
            if not state_path.exists():
                decisions[name] = "missing"
                continue
            row = load_json(state_path)
            if row.get("name") != name:
                decisions[name] = "invalid-name"
                continue
            decision = recovery_decision(row)
            decisions[name] = decision
            if decision != "recover":
                continue
            result = restart_job(state_path, row)
            event = {
                "time": utc_now(),
                "event": "recover",
                "name": name,
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
            }
            append_event(args.events, event)
            decisions[name] = "restarted" if result.returncode == 0 else "restart-failed"

        heartbeat = {
            "time": utc_now(),
            "watchdog_pid": os.getpid(),
            "manifest": str(args.manifest.resolve()),
            "required_manifest": str(REQUIRED_MANIFEST.resolve()),
            "decisions": decisions,
        }
        atomic_json(args.heartbeat, heartbeat)
        policy = manifest.get("policy", {})
        exit_when_all_done = bool(policy.get("exit_when_all_done", True))
        if args.once or (
            exit_when_all_done
            and decisions
            and all(value == "terminal" for value in decisions.values())
        ):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
