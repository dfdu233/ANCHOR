#!/usr/bin/env python3
"""Freeze source, license, and checkpoint provenance for trained baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "trained-baseline-t0-audit-v1"
ROOT = Path(__file__).resolve().parents[2]

BASE = Path("/home/dbw/models/llava-v1.5-7b")
METHODS = {
    "base": {
        "checkpoint": BASE,
        "checkpoint_repo": "liuhaotian/llava-v1.5-7b",
        "source": ROOT / "third_party/baselines/VCD/experiments",
        "license": "Llama-2 Community License declared by checkpoint card",
    },
    "ha-dpo": {
        "checkpoint": Path("/home/dbw/models/hadpo-llava-1.5"),
        "checkpoint_repo": "juliozhao/hadpo-llava-1.5",
        "source": ROOT / "third_party/training_baselines/HA-DPO",
        "license": "not declared at official repository or checkpoint root",
    },
    "opa-dpo": {
        "checkpoint": Path("/home/dbw/models/opadpo-lora-llava-v1.5-7b"),
        "checkpoint_repo": "zhyang2226/opadpo-lora_llava-v1.5-7b",
        "source": ROOT / "third_party/training_baselines/OPA-DPO",
        "license": "MIT",
    },
    "da-dpo": {
        "checkpoint": Path("/home/dbw/models/da-dpo-llava-v1.5-7b"),
        "checkpoint_repo": "Artanic30/DA-DPO_llava_v1.5_7B",
        "source": ROOT / "third_party/training_baselines/DA-DPO",
        "license": "Apache-2.0",
    },
    "sentinel": {
        "checkpoint": Path("/home/dbw/models/sentinel-llava-v1.5-7b"),
        "checkpoint_repo": "psp-dada/LLaVA-v1.5-7B-SENTINEL",
        "source": None,
        "license": "Apache-2.0 declared by checkpoint card",
    },
    "less-is-more": {
        "checkpoint": Path("/home/dbw/models/less-is-more-llava-v1.5-7b"),
        "checkpoint_repo": "yuezih/llava-v1.5-7b-selective-23k-lora",
        "source": None,
        "license": "not declared by checkpoint card",
    },
    "factmm-rag-generator": {
        "checkpoint": Path("/home/dbw/models/factmm-rag-generator-v1"),
        "checkpoint_repo": "official FactMM-RAG Google Drive model.zip (archive SHA256 49ce0f31a57a39f8fd25420c2fbc32638167d6ab8ee0bbac68049d4e93db4abf)",
        "source": ROOT / "third_party/FactMM-RAG",
        "license": "MIT",
    },
}

WEIGHT_SUFFIXES = {".bin", ".safetensors"}
METADATA_NAMES = {"README.md", "adapter_config.json", "config.json", "pytorch_model.bin.index.json"}


def git_value(path: Path, *arguments: str) -> str | None:
    if not path.exists():
        return None
    try:
        return subprocess.check_output(["git", "-C", str(path), *arguments], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None


def checkpoint_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and (item.suffix in WEIGHT_SUFFIXES or item.name in METADATA_NAMES)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_files = checkpoint_files(BASE)
    base_hashes = {item.name: sha256_file(item) for item in base_files}
    rows = []
    for name, spec in METHODS.items():
        checkpoint = spec["checkpoint"]
        files = base_files if checkpoint == BASE else checkpoint_files(checkpoint)
        hashes = base_hashes if checkpoint == BASE else {item.name: sha256_file(item) for item in files}
        source = spec["source"]
        has_weight = any(Path(filename).suffix in WEIGHT_SUFFIXES for filename in hashes)
        license_declared = not str(spec["license"]).startswith("not declared")
        status = "pass"
        reasons = []
        if not has_weight:
            status = "N/A"
            reasons.append("no released checkpoint weight artifact")
        if not license_declared and status == "pass":
            reasons.append("license is not declared; report result separately and do not redistribute code/weights")
        combined = hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest() if hashes else None
        rows.append({
            "method": name,
            "status": status,
            "reasons": reasons,
            "checkpoint": str(checkpoint),
            "checkpoint_repo": spec["checkpoint_repo"],
            "checkpoint_files": {filename: {"sha256": digest, "bytes": (checkpoint / filename).stat().st_size} for filename, digest in hashes.items()},
            "checkpoint_fingerprint": combined,
            "source": str(source) if source else "standard released LLaVA-1.5 PEFT inference; method repository not required by runner",
            "source_commit": git_value(source, "rev-parse", "HEAD") if source else None,
            "source_remote": git_value(source, "remote", "get-url", "origin") if source else None,
            "license": spec["license"],
            "license_declared": license_declared,
            "base_checkpoint_fingerprint": hashlib.sha256(json.dumps(base_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        })
    result = {
        "version": VERSION,
        "runner": str((ROOT / "anchor/corrected_sgta/run_trained_llava_baseline_v1.py").resolve()),
        "runner_sha256": sha256_file(ROOT / "anchor/corrected_sgta/run_trained_llava_baseline_v1.py"),
        "methods": rows,
        "summary": {
            "methods": len(rows),
            "checkpoint_present": sum(row["status"] == "pass" for row in rows),
            "N/A": sum(row["status"] == "N/A" for row in rows),
            "license_not_declared": sum(not row["license_declared"] for row in rows),
        },
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
