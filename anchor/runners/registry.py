"""Lightweight registry runner for ANCHOR packaged experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def method_tasks(methods_cfg: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for group in ("baseline", "anchor", "ce_methods", "mitigation", "judges"):
        for name, spec in methods_cfg.get(group, {}).items():
            out[name] = set(spec.get("tasks", []))
    return out


def validate_dataset(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    root = ROOT / spec["data_root"]
    if not root.exists():
        return {"dataset": name, "ok": False, "reason": f"missing {root}"}
    result: dict[str, Any] = {"dataset": name, "ok": True, "task": spec["task"]}
    if "questions" in spec:
        questions = ROOT / spec["questions"]
        result["questions_exists"] = questions.is_file()
        if questions.is_file():
            result["questions_sha256"] = sha256_file(questions)
            result["records"] = sum(1 for line in questions.read_text().splitlines() if line.strip())
    if "manifest" in spec:
        manifest = ROOT / spec["manifest"]
        result["manifest_exists"] = manifest.is_file()
        if manifest.is_file():
            result["manifest_sha256"] = sha256_file(manifest)
    if "image_manifest" in spec:
        image_manifest = ROOT / spec["image_manifest"]
        result["image_manifest_exists"] = image_manifest.is_file()
        if image_manifest.is_file():
            rows = [json.loads(line) for line in image_manifest.read_text().splitlines() if line.strip()]
            result["images"] = len(rows)
            result["image_bytes"] = sum(int(row.get("bytes", 0)) for row in rows)
    return result


def run_opencode_mock(output: Path) -> None:
    output.write_text(json.dumps({"backend": "opencode", "mode": "mock", "ok": True}, indent=2) + "\n")


def write_run_record(dataset: str, method: str, task: str, judges: list[str], smoke: bool) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / timestamp / dataset / "llava_med" / method
    run_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.argv)
    config = {
        "dataset": dataset,
        "method": method,
        "task": task,
        "judges": judges,
        "smoke": smoke,
        "command": command,
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "command.txt").write_text(command + "\n")
    (run_dir / "fingerprint.json").write_text(json.dumps({"sha256": fingerprint}, indent=2) + "\n")
    (run_dir / "raw.jsonl").write_text(json.dumps({"status": "registry_smoke", **config}) + "\n")
    (run_dir / "records.jsonl").write_text(json.dumps({"dataset": dataset, "method": method, "task": task, "status": "not_run_heavy_in_registry"}) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps({"status": "registry_only", "heavy_inference_required": True}, indent=2) + "\n")
    (run_dir / "summary.json").write_text(json.dumps({"dataset": dataset, "method": method, "ok": True, "fingerprint": fingerprint}, indent=2) + "\n")
    if "opencode" in judges:
        if os.environ.get("OPENCODE_MODE", "mock") == "mock":
            run_opencode_mock(run_dir / "opencode_judge.json")
        else:
            subprocess.run(["opencode", "--version"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets")
    parser.add_argument("--methods")
    parser.add_argument("--judges", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    datasets_cfg = load_yaml(ROOT / "configs/datasets.yaml")
    methods_cfg = load_yaml(ROOT / "configs/methods.yaml")
    known_methods = method_tasks(methods_cfg)
    selected_datasets = split_csv(args.datasets, datasets_cfg["default"])
    selected_methods = split_csv(args.methods, ["greedy", "source_margin", "source_word_center"])
    selected_judges = split_csv(args.judges, ["rule_parser", "rouge"])

    summaries = []
    for dataset in selected_datasets:
        spec = datasets_cfg["datasets"][dataset]
        validation = validate_dataset(dataset, spec)
        summaries.append(validation)
        if not validation["ok"]:
            continue
        if args.check_only:
            continue
        for method in selected_methods:
            tasks = known_methods.get(method)
            if not tasks:
                summaries.append({"dataset": dataset, "method": method, "status": "skipped", "reason": "unknown method"})
                continue
            if spec["task"] not in tasks:
                summaries.append({"dataset": dataset, "method": method, "status": "skipped", "reason": f"unsupported task {spec['task']}"})
                continue
            run_dir = write_run_record(dataset, method, spec["task"], selected_judges, args.smoke)
            summaries.append({"dataset": dataset, "method": method, "status": "registry_ok", "run_dir": str(run_dir.relative_to(ROOT))})
    print(json.dumps({"status": "ok", "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
