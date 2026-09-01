#!/usr/bin/env python3
"""Start a named experiment outside the terminal/VS Code process group."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def pid_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        try:
            if stat_path.read_text().split()[2] == "Z":
                return False
        except (OSError, IndexError):
            pass
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--supervisor", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("a command is required after --")
    if args.supervisor:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        time.sleep(0.2)
        payload = {
            "name": args.name,
            "pid": os.getpid(),
            "child_pid": process.pid,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "cwd": str(Path.cwd().resolve()),
            "command": command,
            "log": str(args.log.resolve()),
        }
        atomic_json(args.state, payload)
        exit_code = process.wait()
        payload.update({
            "status": "done" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(args.state, payload)
        return
    if args.state.exists():
        prior = json.loads(args.state.read_text())
        pid = int(prior.get("pid", -1))
        if pid_alive(pid):
            raise RuntimeError(f"{args.name} is already alive as PID {pid}")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    supervisor_command = [
        sys.executable, str(Path(__file__).resolve()),
        "--name", args.name,
        "--log", str(args.log.resolve()),
        "--state", str(args.state.resolve()),
        "--supervisor", "--", *command,
    ]
    process = subprocess.Popen(
        supervisor_command,
        cwd=Path.cwd(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    payload = {
        "name": args.name,
        "pid": process.pid,
        "status": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd().resolve()),
        "command": command,
        "log": str(args.log.resolve()),
    }
    atomic_json(args.state, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
