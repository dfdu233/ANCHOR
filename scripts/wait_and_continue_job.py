#!/usr/bin/env python3
"""Persistently wait for a detached job and immediately run its continuation."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-state", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("continuation command required after --")

    while True:
        upstream = json.loads(args.upstream_state.read_text()) if args.upstream_state.exists() else {}
        status = str(upstream.get("status", "missing"))
        atomic_json(
            args.heartbeat,
            {
                "version": "wait-and-continue-job-v1",
                "time": datetime.now(timezone.utc).isoformat(),
                "stage": "waiting_for_upstream" if status not in {"done", "failed"} else "upstream_terminal",
                "upstream_state": str(args.upstream_state.resolve()),
                "upstream_status": status,
                "continuation": command,
            },
        )
        if status == "done":
            break
        if status == "failed":
            raise RuntimeError(f"upstream failed: {args.upstream_state}")
        time.sleep(args.interval)

    atomic_json(
        args.heartbeat,
        {
            "version": "wait-and-continue-job-v1",
            "time": datetime.now(timezone.utc).isoformat(),
            "stage": "running_continuation",
            "upstream_state": str(args.upstream_state.resolve()),
            "upstream_status": "done",
            "continuation": command,
        },
    )
    completed = subprocess.run(command, check=False)
    atomic_json(
        args.heartbeat,
        {
            "version": "wait-and-continue-job-v1",
            "time": datetime.now(timezone.utc).isoformat(),
            "stage": "complete" if completed.returncode == 0 else "continuation_failed",
            "upstream_state": str(args.upstream_state.resolve()),
            "upstream_status": "done",
            "continuation": command,
            "continuation_exit_code": completed.returncode,
        },
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
