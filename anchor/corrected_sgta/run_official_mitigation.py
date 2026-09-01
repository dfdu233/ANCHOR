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

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("ANCHOR_LEGACY_ROOT", REPO_ROOT))
MED = Path(os.environ.get("ANCHOR_MEDHEVAL_ROOT", REPO_ROOT / "data/medheval"))
IMAGE_ROOT = MED / "images"
RUNNER = MED / "code/baselines/Mitigation/llava-med-1.5/llava/eval/model_vqa.py"
WORKDIR = MED / "code/baselines/Mitigation/llava-med-1.5"
MODEL_PATH = Path(
    os.environ.get(
        "ANCHOR_MODEL_PATH", REPO_ROOT / "hf_cache/llava-med-v1.5-mistral-7b"
    )
)
PYTHON = Path(os.environ.get("ANCHOR_PYTHON", REPO_ROOT / ".venv-full/bin/python"))
EVALUATOR = ROOT / "anchor/corrected_sgta/evaluate_medheval_answers.py"
PYTHONPATH_VALUE = (
    f"{WORKDIR}:{WORKDIR / 'llava/eval'}:{MED}/code/baselines/Med-LVLMs/"
    f"llava-med-1.5/transformers-4.37.2/src:{ROOT}"
)

PROTOCOL_VERSION = "corrected-sgta-official-mitigation-v3"
METHODS = ("greedy", "DoLa", "PAI", "opera", "avisc", "m3id", "VCD", "damro")
DATASETS = {
    "context": MED / "benchmark_data/Context_Misalignment_Hallucination/MIMIC-CXR_pairs.json",
    "knowledge_ce": MED / "benchmark_data/Knowledge_Deficiency_Hallucination/close-ended/MIMIC-CXR_sampled.json",
    "cxr_vishal": MED / "benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json",
    "mm_vishal": MED / "benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/MM-VisHal.json",
    "mimic_fine_grained_ce": MED / "benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/mimic_cxr_closed_pairs.json",
    "iuxray_fine_grained_ce": MED / "benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/xray_closed_pairs.json",
    "slake_ce": MED / "benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/slake_qa_pairs.json",
    "vqa_rad_ce": MED / "benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/rad_vqa_pairs.json",
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
    parser.add_argument("--datasets", nargs="*", choices=sorted(DATASETS), default=["knowledge_ce", "cxr_vishal", "mm_vishal"])
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=("root", "iu_xray", "slake", "vqa_rad"),
        help="Optional image-source filter used only for efficient resumable scheduling.",
    )
    parser.add_argument("--max-samples-per-source", type=int, help="Optional smoke-test cap applied before chunking.")
    parser.add_argument(
        "--sampling-policy",
        choices=("first", "balanced_binary_image", "claim_universe_images"),
        default="first",
        help=(
            "Sampling before chunking. balanced_binary_image keeps exact Yes/No "
            "labels, balances them, and uses at most one question per image; "
            "claim_universe_images keeps every exact Yes/No claim for a sampled "
            "set of images."
        ),
    )
    parser.add_argument(
        "--max-images-per-source",
        type=int,
        help="Image cap required by claim_universe_images.",
    )
    parser.add_argument(
        "--exclude-images-from",
        nargs="*",
        type=Path,
        help="Question JSON files whose image names must be excluded before sampling.",
    )
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


def _stable_rank(row: dict[str, Any], seed: int) -> str:
    identity = str(row.get("qid", row.get("question_id", row.get("img_name", ""))))
    return hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest()


def balanced_binary_image_sample(
    rows: list[dict[str, Any]], cap: int, seed: int
) -> list[dict[str, Any]]:
    """Select balanced exact Yes/No rows with one independently sampled image each."""
    if cap <= 0 or cap % 2:
        raise ValueError("balanced_binary_image requires a positive even sample cap")
    groups: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {
        "yes": {}, "no": {},
    }
    for row in rows:
        truth = str(row.get("answer", "")).strip().lower()
        if truth not in groups:
            continue
        stratum = (
            str(row.get("modality") or "unknown"),
            str(row.get("hallucination_type") or "unknown"),
        )
        groups[truth].setdefault(stratum, []).append(row)
    for truth_groups in groups.values():
        for bucket in truth_groups.values():
            bucket.sort(key=lambda row: _stable_rank(row, seed))
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    quotas = {"yes": cap // 2, "no": cap // 2}
    positions = {truth: 0 for truth in groups}
    strata = {truth: sorted(group) for truth, group in groups.items()}
    while any(quotas.values()):
        progressed = False
        for truth in ("yes", "no"):
            if quotas[truth] == 0:
                continue
            keys = strata[truth]
            if not keys:
                continue
            for offset in range(len(keys)):
                stratum_index = (positions[truth] + offset) % len(keys)
                bucket = groups[truth][keys[stratum_index]]
                while bucket and image_name(bucket[0]) in used_images:
                    bucket.pop(0)
                if bucket:
                    row = bucket.pop(0)
                    selected.append(row)
                    used_images.add(image_name(row))
                    quotas[truth] -= 1
                    positions[truth] = (stratum_index + 1) % len(keys)
                    progressed = True
                    break
        if not progressed:
            break
    if any(quotas.values()):
        raise ValueError(
            f"insufficient image-disjoint exact Yes/No rows for cap={cap}; "
            f"unfilled quotas={quotas}"
        )
    return selected


def claim_universe_image_sample(
    rows: list[dict[str, Any]], image_cap: int, seed: int
) -> list[dict[str, Any]]:
    """Keep all exact Yes/No claims for a deterministic image-disjoint sample."""
    if image_cap <= 0:
        raise ValueError("claim_universe_images requires a positive image cap")
    eligible = [
        row for row in rows
        if str(row.get("answer", "")).strip().lower() in {"yes", "no"}
    ]
    by_image: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_image.setdefault(image_name(row), []).append(row)
    ranked_images = sorted(
        by_image,
        key=lambda name: hashlib.sha256(f"{seed}|{name}".encode()).hexdigest(),
    )
    if len(ranked_images) < image_cap:
        raise ValueError(
            f"requested {image_cap} images but only {len(ranked_images)} are eligible"
        )
    selected = []
    for name in ranked_images[:image_cap]:
        selected.extend(sorted(by_image[name], key=lambda row: _stable_rank(row, seed)))
    return selected


def write_inputs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    input_root = args.out / "_inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    excluded_images: set[str] = set()
    exclusion_inputs = []
    for path in args.exclude_images_from or []:
        exclusion_rows = load_rows(path)
        excluded_images.update(image_name(row) for row in exclusion_rows)
        exclusion_inputs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
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
        "sampling_policy": args.sampling_policy,
        "max_images_per_source": args.max_images_per_source,
        "exclusion_inputs": exclusion_inputs,
        "n_excluded_image_names": len(excluded_images),
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
            if image_name(row) in excluded_images:
                continue
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
            if args.sampling_policy == "balanced_binary_image":
                if args.max_samples_per_source is None:
                    raise ValueError("balanced_binary_image requires --max-samples-per-source")
                bucket = balanced_binary_image_sample(
                    bucket, args.max_samples_per_source, args.seed
                )
            elif args.sampling_policy == "claim_universe_images":
                if args.max_images_per_source is None:
                    raise ValueError("claim_universe_images requires --max-images-per-source")
                bucket = claim_universe_image_sample(
                    bucket, args.max_images_per_source, args.seed
                )
            elif args.max_samples_per_source is not None:
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
        raise SystemExit(
            "required official runner, Python env, or model path is missing: "
            f"runner={RUNNER} python={PYTHON} model={MODEL_PATH}"
        )
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
