#!/usr/bin/env python3
"""Build a machine-readable manifest from Huatuo MoSEC run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_completion(raw_path: Path) -> str | None:
    completed = None
    for line in raw_path.read_text().splitlines():
        if line.strip():
            completed = json.loads(line).get("completed_at") or completed
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    entries = []
    for config_path in sorted(args.root.rglob("config.json")):
        run_dir = config_path.parent
        summary_path = run_dir / "summary.json"
        raw_path = run_dir / "raw_generations.jsonl"
        if not summary_path.exists() or not raw_path.exists():
            entries.append(
                {
                    "source": None,
                    "dataset": None,
                    "task": None,
                    "method": None,
                    "status": "failed",
                    "start_time": None,
                    "end_time": None,
                    "command": None,
                    "commit": commit,
                    "config_sha256": None,
                    "output_dir": str(run_dir),
                    "failure_reason": "missing summary or raw generations",
                }
            )
            continue
        config = load_json(config_path)
        summary = load_json(summary_path)
        for method, result in summary.get("methods", {}).items():
            errors = int(result.get("errors", 0))
            entries.append(
                {
                    "source": "MIMIC-CXR",
                    "dataset": config.get("dataset"),
                    "task": config.get("task"),
                    "method": method,
                    "status": "done" if errors == 0 else "failed",
                    "start_time": config.get("created_at"),
                    "end_time": last_completion(raw_path),
                    "command": config.get("command"),
                    "commit": commit,
                    "config_sha256": sha256(config_path),
                    "input_sha256": config.get("input_sha256"),
                    "bank_sha256": config.get("bank_sha256"),
                    "local_index_sha256": config.get(
                        "local_index_sha256"
                    ),
                    "output_dir": str(run_dir / method),
                    "failure_reason": (
                        None if errors == 0 else f"{errors} sample errors"
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in entries)
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "entries": len(entries),
                "status_counts": {
                    status: sum(row["status"] == status for row in entries)
                    for status in ("done", "failed", "blocked")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
