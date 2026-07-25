#!/usr/bin/env python3
"""Unified corrected_sgta pipeline for official LLaVA-Med mitigation methods.

This module integrates VCD/DoLa/OPERA/AVISC/M3ID/DAMRO/PAI into the
corrected_sgta experiment lifecycle while preserving the official MedHEval
LLaVA-Med generation hooks. It provides deterministic input filtering,
chunk-level fingerprints, resumable execution, and unified summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path("/root/autodl-tmp/Hulu-Med/MedUniEval")
MED = Path("/root/autodl-tmp/MedHEval")
IMAGE_ROOT = MED / "images"
RUNNER = MED / "code/baselines/Mitigation/llava-med-1.5/llava/eval/model_vqa.py"
WORKDIR = MED / "code/baselines/Mitigation/llava-med-1.5"
MODEL_PATH = Path("/root/autodl-tmp/LLaVA-Med/microsoft/llava-med-v1.5-mistral-7b")
PYTHON = Path("/root/autodl-tmp/envs/medheval-mitigation/bin/python")
EVALUATOR = ROOT / "corrected_sgta/evaluate_medheval_answers.py"
PYTHONPATH_VALUE = (
    f"{WORKDIR}:{MED}/code/baselines/Med-LVLMs/llava-med-1.5/"
    f"transformers-4.37.2/src:{ROOT}"
)

PROTOCOL_VERSION = "corrected-sgta-official-mitigation-v2"
METHODS = ("greedy", "DoLa", "PAI", "opera", "avisc", "m3id", "VCD", "damro")
DATASETS = {
    "context": MED / "benchmark_data/Context_Misalignment_Hallucination/MIMIC-CXR_pairs.json",
    "knowledge_ce": MED / "benchmark_data/Knowledge_Deficiency_Hallucination/close-ended/MIMIC-CXR_sampled.json",
    "cxr_vishal": MED / "benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json",
    "mm_vishal": MED / "benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/MM-VisHal.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(config: dict[str, Any]) -> str:
    payload = {"protocol_version": PROTOCOL_VERSION, **config}
    return hashlib.sha256(stable_json(payload).encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "corrected_runs/aaai_medheval_mitigation_full_v1")
    parser.add_argument("--datasets", nargs="*", default=["knowledge_ce", "cxr_vishal", "mm_vishal"])
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=("root", "iu_xray", "slake", "vqa_rad"),
        help="Optional image-source filter used only for efficient resumable scheduling.",
    )
    parser.add_argument("--max-samples-per-source", type=int, help="Optional smoke-test cap applied before chunking.")
    parser.add_argument(
        "--question-types",
        nargs="*",
        choices=("binary", "multi-choice"),
        default=("binary", "multi-choice"),
        help="Filter CE interfaces so failed multi-choice outputs can be rerun separately.",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-reuse-legacy", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--allow-legacy-fingerprint-mismatch",
        action="store_true",
        help="Unsafe compatibility escape hatch; never use for paper results.",
    )
    return parser.parse_args()


def image_name(row: dict[str, Any]) -> str:
    return str(row.get("img_name") or row.get("image") or row.get("img_id") or "")


def source_for_image(name: str) -> tuple[str, Path, str] | None:
    candidates = [
        ("root", IMAGE_ROOT, name),
        ("iu_xray", IMAGE_ROOT / "IU-Xray", name),
        ("slake", IMAGE_ROOT / "Slake", name),
        ("vqa_rad", IMAGE_ROOT / "VQA-RAD", name),
    ]
    for source, folder, rel in candidates:
        if (folder / rel).exists():
            return source, folder, rel
    return None


def image_loads(path: Path) -> bool:
    try:
        with Image.open(path) as handle:
            handle.load()
        return True
    except Exception:
        return False


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def write_inputs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    input_root = args.out / "_inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "runner": str(RUNNER),
        "runner_sha256": sha256_file(RUNNER),
        "evaluator_sha256": sha256_file(EVALUATOR),
        "model_path": str(MODEL_PATH),
        "datasets": {},
        "methods": list(args.methods),
        "sources": args.sources,
        "max_samples_per_source": args.max_samples_per_source,
        "question_types": list(args.question_types),
        "chunk_size": args.chunk_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "filtering": "require existing image and PIL load success",
    }
    prepared: dict[str, dict[str, Any]] = {}
    for dataset in args.datasets:
        rows = load_rows(DATASETS[dataset])
        buckets: dict[str, list[dict[str, Any]]] = {}
        missing = 0
        invalid = 0
        for row in rows:
            row_type = str(row.get("question_type") or row.get("ground_truth_type") or "binary")
            if row_type not in args.question_types:
                continue
            resolved = source_for_image(image_name(row))
            if resolved is None:
                missing += 1
                continue
            source, folder, rel = resolved
            if not image_loads(folder / rel):
                invalid += 1
                continue
            item = dict(row)
            item["img_name"] = rel
            buckets.setdefault(source, []).append(item)
        for source, bucket in buckets.items():
            if args.sources is not None and source not in args.sources:
                continue
            if args.max_samples_per_source is not None:
                bucket = bucket[: args.max_samples_per_source]
            key = f"{dataset}.{source}"
            path = input_root / f"{key}.json"
            path.write_text(json.dumps(bucket, indent=2))
            folder = source_for_image(bucket[0]["img_name"])[1]
            info = {
                "dataset": dataset,
                "source": source,
                "input": str(DATASETS[dataset]),
                "input_sha256": sha256_file(DATASETS[dataset]),
                "question_file": str(path),
                "question_file_sha256": sha256_file(path),
                "image_folder": str(folder),
                "n": len(bucket),
                "missing": missing,
                "invalid_image": invalid,
            }
            manifest["datasets"][key] = info
            prepared[key] = info
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return prepared


def chunks(rows: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]


def eval_valid(path: Path, expected_n: int) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text())
    except Exception:
        return False
    return int(report.get("n", -1)) == int(expected_n)


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open() if line.strip())


def run_command(cmd: list[str], env: dict[str, str], log: Path, dry_run: bool) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write("\n$ " + " ".join(cmd) + "\n")
        handle.flush()
        if dry_run:
            return 0
        proc = subprocess.run(cmd, cwd=str(WORKDIR), env=env, stdout=handle, stderr=subprocess.STDOUT)
        return proc.returncode


def chunk_meta_config(args: argparse.Namespace, dataset: str, source: str, method: str, chunk_id: int, chunk_file: Path, image_folder: Path, expected_n: int) -> dict[str, Any]:
    return {
        "model": "llava_med",
        "dataset": dataset,
        "source": source,
        "method": method,
        "question_types": list(args.question_types),
        "chunk_id": chunk_id,
        "expected_n": expected_n,
        "chunk_size": args.chunk_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "runner_sha256": sha256_file(RUNNER),
        "evaluator_sha256": sha256_file(EVALUATOR),
        "question_sha256": sha256_file(chunk_file),
        "image_folder": str(image_folder),
        "official_runner": str(RUNNER),
    }


def main() -> int:
    args = parse_args()
    args.out = args.out.resolve()
    if not RUNNER.exists() or not PYTHON.exists() or not MODEL_PATH.exists():
        raise SystemExit("required official runner, Python env, or model path is missing")
    prepared = write_inputs(args)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONPATH": PYTHONPATH_VALUE})
    state: list[dict[str, Any]] = []
    for info in prepared.values():
        dataset = info["dataset"]
        source = info["source"]
        image_folder = Path(info["image_folder"])
        rows = load_rows(Path(info["question_file"]))
        for method in args.methods:
            for chunk_id, chunk in enumerate(chunks(rows, args.chunk_size)):
                job_dir = args.out / "llava_med" / dataset / method / source
                job_dir.mkdir(parents=True, exist_ok=True)
                chunk_file = job_dir / f"chunk_{chunk_id:04d}.questions.json"
                answers = job_dir / f"chunk_{chunk_id:04d}.answers.jsonl"
                eval_file = job_dir / f"chunk_{chunk_id:04d}.eval.json"
                log = job_dir / f"chunk_{chunk_id:04d}.log"
                meta_file = job_dir / f"chunk_{chunk_id:04d}.meta.json"
                if not eval_valid(eval_file, len(chunk)):
                    chunk_file.write_text(json.dumps(chunk, indent=2))
                config = chunk_meta_config(args, dataset, source, method, chunk_id, chunk_file, image_folder, len(chunk))
                fp = fingerprint(config)
                existing_meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
                if eval_valid(eval_file, len(chunk)) and existing_meta.get("fingerprint") == fp:
                    state.append({"dataset": dataset, "source": source, "method": method, "chunk": chunk_id, "status": "skipped", "fingerprint": fp})
                    continue
                if eval_valid(eval_file, len(chunk)) and args.allow_legacy_fingerprint_mismatch and not args.no_reuse_legacy:
                    meta_file.write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, "fingerprint": fp, "config": config, "status": "reused_legacy_completed", "answers": str(answers), "eval": str(eval_file)}, indent=2))
                    state.append({"dataset": dataset, "source": source, "method": method, "chunk": chunk_id, "status": "reused_legacy_completed", "fingerprint": fp})
                    continue
                cmd = [
                    str(PYTHON), str(RUNNER),
                    "--model-path", str(MODEL_PATH),
                    "--image-folder", str(image_folder),
                    "--question-file", str(chunk_file),
                    "--answers-file", str(answers),
                    "--conv-mode", "mistral_instruct",
                    "--temperature", "0",
                    "--top_p", "1",
                    "--num_beams", "1",
                    "--baseline", method,
                    "--max-new-tokens", str(args.max_new_tokens),
                    "--seed", str(args.seed),
                ]
                status = run_command(cmd, env, log, args.dry_run)
                if status == 0:
                    status = run_command([str(PYTHON), "-m", "corrected_sgta.evaluate_medheval_answers", "--answers", str(answers), "--questions", str(chunk_file), "--output", str(eval_file)], env, log, args.dry_run)
                label = "dry-run" if args.dry_run else ("ok" if status == 0 else f"failed:{status}")
                if status == 0 and not args.dry_run:
                    meta_file.write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, "fingerprint": fp, "config": config, "status": "ok", "answers": str(answers), "eval": str(eval_file), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
                state.append({"dataset": dataset, "source": source, "method": method, "chunk": chunk_id, "status": label, "fingerprint": fp, "answers_lines": line_count(answers)})
                (args.out / "queue_state.json").write_text(json.dumps(state, indent=2))
                if status != 0 and not args.continue_on_error:
                    return status
    (args.out / "queue_state.json").write_text(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
