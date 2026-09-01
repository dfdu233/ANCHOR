#!/usr/bin/env python3
"""Wait for multiple detached-job state files, then run one continuation."""

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
    parser.add_argument("--upstream-state", type=Path, action="append", required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("continuation command required after --")

    while True:
        statuses = {}
        for path in args.upstream_state:
            payload = json.loads(path.read_text()) if path.exists() else {}
            statuses[str(path.resolve())] = str(payload.get("status", "missing"))
        failed = [path for path, status in statuses.items() if status == "failed"]
        all_done = all(status == "done" for status in statuses.values())
        atomic_json(
            args.heartbeat,
            {
                "version": "wait-for-jobs-and-continue-v1",
                "time": datetime.now(timezone.utc).isoformat(),
                "stage": "upstream_failed" if failed else "running_continuation" if all_done else "waiting_for_upstreams",
                "upstreams": statuses,
                "continuation": command,
            },
        )
        if failed:
            raise RuntimeError(f"upstream jobs failed: {failed}")
        if all_done:
            break
        time.sleep(args.interval)

    completed = subprocess.run(command, check=False)
    atomic_json(
        args.heartbeat,
        {
            "version": "wait-for-jobs-and-continue-v1",
            "time": datetime.now(timezone.utc).isoformat(),
            "stage": "complete" if completed.returncode == 0 else "continuation_failed",
            "upstreams": statuses,
            "continuation": command,
            "continuation_exit_code": completed.returncode,
        },
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
