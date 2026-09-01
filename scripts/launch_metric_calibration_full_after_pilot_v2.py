#!/usr/bin/env python3
"""Launch the detached n=97 metric side-probe after strict runtime admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/dbw/ANCHOR")
VERSION = "metric-calibration-full-launch-gate-v2"


def qualified(path: Path) -> tuple[bool, dict]:
    value = json.loads(path.read_text())
    runtime = value["runtime"]
    checks = {
        "structured_json_valid_rate": runtime["structured_json_valid_rate"] >= 0.95,
        "structured_nonempty_rate": runtime["structured_nonempty_rate"] >= 0.95,
        "structured_cap_hit_rate": runtime["structured_cap_hit_rate"] <= 0.05,
        "runtime_admissible": value["runtime_admissible"] is True,
    }
    return all(checks.values()), checks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--huatuo", type=Path, required=True)
    parser.add_argument("--pilot-decision", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args()

    models = {}
    admitted = True
    for name, path in (("qwen_parent", args.qwen), ("huatuo_medical", args.huatuo)):
        passed, checks = qualified(path)
        models[name] = {"analysis": str(path.resolve()), "checks": checks, "passed": passed}
        admitted &= passed
    free_bytes = shutil.disk_usage(ROOT).free
    disk_pass = free_bytes >= 100 * 1024**3
    admitted &= disk_pass
    pilot = json.loads(args.pilot_decision.read_text())
    pilot_checks = {
        "decision_expands": pilot.get("decision") == "EXPAND_TO_N97_DIAGNOSTIC_ONLY",
        "n97_authorized": pilot.get("n97_authorized") is True,
        "gpu_authorized": pilot.get("gpu_authorized") is True,
        "qwen_analysis_hash_matches": pilot.get("models", {})
        .get("qwen_parent", {})
        .get("analysis_sha256")
        == sha256(args.qwen),
        "huatuo_analysis_hash_matches": pilot.get("models", {})
        .get("huatuo_medical", {})
        .get("analysis_sha256")
        == sha256(args.huatuo),
    }
    pilot_pass = all(pilot_checks.values())
    admitted &= pilot_pass
    result = {
        "version": VERSION,
        "time": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "disk_free_bytes": free_bytes,
        "disk_minimum_bytes": 100 * 1024**3,
        "disk_pass": disk_pass,
        "pilot_decision": str(args.pilot_decision.resolve()),
        "pilot_checks": pilot_checks,
        "pilot_pass": pilot_pass,
        "launch_admitted": admitted,
        "job": "metric-calibration-full-v2",
    }
    if not admitted:
        result["decision"] = "STOP_FAIL_CLOSED_GATE"
        atomic_json(args.decision, result)
        raise SystemExit(2)

    state = ROOT / "corrected_runs/detached_jobs/metric-calibration-full-v2.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/start_detached_job.py"),
        "--name",
        "metric-calibration-full-v2",
        "--log",
        str(ROOT / "corrected_runs/detached_jobs/metric-calibration-full-v2.log"),
        "--state",
        str(state),
        "--",
        "bash",
        "scripts/run_metric_calibration_full_v2.sh",
    ]
    launched = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    result.update(
        {
            "decision": "FULL_N97_LAUNCHED" if launched.returncode == 0 else "LAUNCH_FAILED",
            "launcher_returncode": launched.returncode,
            "launcher_stdout": launched.stdout,
            "launcher_stderr": launched.stderr,
        }
    )
    atomic_json(args.decision, result)
    if launched.returncode != 0:
        raise SystemExit(launched.returncode)


if __name__ == "__main__":
    main()
