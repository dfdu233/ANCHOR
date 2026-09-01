#!/usr/bin/env python3
"""Run source-separated OE/report generation through MedHEval mitigation hooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import (
    runtime_fingerprint,
    sha256_file,
    sha256_json,
    source_tree_fingerprint,
)
from anchor.medeval.legacy import audit_legacy_answers


REPO_ROOT = Path(__file__).resolve().parents[2]
MEDHEVAL_ROOT = Path(os.environ.get("ANCHOR_MEDHEVAL_ROOT", REPO_ROOT / "data/medheval"))
MITIGATION_ROOT = Path(os.environ.get("ANCHOR_MITIGATION_ROOT", MEDHEVAL_ROOT / "code/baselines/Mitigation/llava-med-1.5"))
TRANSFORMERS_SRC = Path(os.environ.get("ANCHOR_MITIGATION_TRANSFORMERS", MEDHEVAL_ROOT / "code/baselines/Med-LVLMs/llava-med-1.5/transformers-4.37.2/src"))
RUNNER = MITIGATION_ROOT / "llava/eval/model_vqa.py"
# Common-protocol identity requires the same numerical runtime as the canonical
# LLaVA-Med adapter. The previous .venv-full path used a different Torch/CUDA
# stack and diverged after otherwise identical prompt/image preprocessing.
PYTHON = Path(
    os.environ.get(
        "ANCHOR_PYTHON", "/opt/miniconda3/envs/huatuo/bin/python"
    )
)
MODEL_PATH = Path(os.environ.get("ANCHOR_MODEL_PATH", REPO_ROOT / "hf_cache/llava-med-v1.5-mistral-7b"))
VISTA_ADAPTER = REPO_ROOT / "anchor/corrected_sgta/vista_adapter.py"
VISTA_SOURCE = REPO_ROOT / "third_party/baselines/VISTA"
METHODS = (
    "greedy", "beam", "DoLa", "PAI", "opera", "avisc", "m3id", "VCD",
    "damro", "VISTA_off", "VISTA_VSV", "VISTA_SLA", "VISTA",
)
PROTOCOL_ID = "anchor-eval-contract-v1"


def generation_runtime() -> dict[str, str]:
    result = subprocess.run(
        [str(PYTHON), "-c", "import platform,sys; print(sys.version); print(platform.platform())"],
        text=True,
        capture_output=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    return {
        "executable": str(PYTHON.resolve()),
        "python": lines[0] if lines else "unknown",
        "platform": lines[1] if len(lines) > 1 else "unknown",
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], log: Path, dry_run: bool) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write("\n$ " + " ".join(cmd) + "\n")
        handle.flush()
        if dry_run:
            return 0
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        return proc.returncode


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", required=True, type=Path)
    parser.add_argument("--image-folder", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--task",
        choices=("open_vqa", "report_generation", "close_vqa"),
        required=True,
    )
    parser.add_argument("--methods", nargs="+", default=["greedy"])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--disable-keyword-stopping",
        action="store_true",
        help=(
            "Do not install the legacy conversation-separator keyword stopper. "
            "This is required for the audited Mistral port, where the stopper "
            "prematurely terminates generation after a leading function word."
        ),
    )
    parser.add_argument(
        "--qualification-run",
        action="store_true",
        help=(
            "Treat output-collapse diagnostics as a smoke-test failure. Full "
            "evaluation must leave this off so poor method behavior is scored, "
            "not censored or rerun."
        ),
    )
    args = parser.parse_args()

    if not RUNNER.exists():
        raise SystemExit(f"missing MedHEval mitigation runner: {RUNNER}")
    if not PYTHON.exists():
        raise SystemExit(f"missing Python executable: {PYTHON}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"missing model path: {MODEL_PATH}")

    args.question_file = args.question_file.resolve()
    args.image_folder = args.image_folder.resolve()
    args.out = args.out.resolve()
    rows = load_rows(args.question_file)
    if args.limit:
        rows = rows[: args.limit]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Every required artifact is already local.  Network fallback makes an
    # otherwise identical run depend on transient SSL/mirror state and can
    # silently mix model revisions, so common-protocol evaluation is offline.
    env["HF_HOME"] = str(REPO_ROOT / "hf_cache")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(MITIGATION_ROOT), str(RUNNER.parent), str(TRANSFORMERS_SRC), str(REPO_ROOT)])
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "queue_state.jsonl"
    root_config = {
        "protocol_id": PROTOCOL_ID,
        "track": "common_protocol",
        "question_file": str(args.question_file),
        "question_file_sha256": sha256_file(args.question_file),
        "image_folder": str(args.image_folder),
        "model_path": str(MODEL_PATH.resolve()),
        "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "model_index_sha256": (
            sha256_file(MODEL_PATH / "model.safetensors.index.json")
            if (MODEL_PATH / "model.safetensors.index.json").exists() else None
        ),
        "runner": str(RUNNER.resolve()),
        "runner_sha256": sha256_file(RUNNER),
        "llava_backend_tree": source_tree_fingerprint(MITIGATION_ROOT / "llava"),
        "transformers_backend_tree": source_tree_fingerprint(
            TRANSFORMERS_SRC / "transformers"
        ),
        "matrix_runner_sha256": sha256_file(Path(__file__)),
        "vista_runtime": {
            "adapter_sha256": sha256_file(VISTA_ADAPTER),
            "official_steering_sha256": sha256_file(
                VISTA_SOURCE / "steering_vector.py"
            ),
            "official_layers_sha256": sha256_file(VISTA_SOURCE / "llm_layers.py"),
            "official_utils_sha256": sha256_file(VISTA_SOURCE / "myutils.py"),
            "method_off_key": "VISTA_off",
            "functional_key": "VISTA",
            "ablation_keys": ["VISTA_VSV", "VISTA_SLA"],
        },
        "conv_mode": args.conv_mode,
        "max_new_tokens": args.max_new_tokens,
        "keyword_stopping_enabled": not args.disable_keyword_stopping,
        "seed": args.seed,
        "orchestrator_runtime": runtime_fingerprint(),
        "generation_runtime": generation_runtime(),
        "execution_environment": {
            "HF_HOME": env["HF_HOME"],
            "HF_HUB_OFFLINE": env["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": env["TRANSFORMERS_OFFLINE"],
            "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
        },
    }
    (args.out / "generation_contract.json").write_text(
        json.dumps(root_config, indent=2) + "\n"
    )
    any_failed = False
    for method in args.methods:
        if method not in METHODS:
            raise SystemExit(f"unsupported method: {method}")
        method_dir = args.out / args.source / args.dataset / args.task / method
        for chunk_id, chunk in enumerate(chunks(rows, args.chunk_size)):
            chunk_file = method_dir / f"chunk_{chunk_id:04d}.questions.json"
            answers = method_dir / f"chunk_{chunk_id:04d}.answers.jsonl"
            metrics = method_dir / f"chunk_{chunk_id:04d}.metrics.json"
            log = method_dir / f"chunk_{chunk_id:04d}.log"
            meta = method_dir / f"chunk_{chunk_id:04d}.meta.json"
            chunk_file.parent.mkdir(parents=True, exist_ok=True)
            chunk_payload_sha = sha256_json(chunk)
            run_config = {
                **root_config,
                "source": args.source,
                "dataset": args.dataset,
                "task": args.task,
                "method": method,
                "chunk": chunk_id,
                "chunk_sha256": chunk_payload_sha,
                "chunk_n": len(chunk),
            }
            run_fingerprint = sha256_json(run_config)
            prior_meta = json.loads(meta.read_text()) if meta.exists() else None
            if prior_meta and prior_meta.get("run_fingerprint") != run_fingerprint:
                raise RuntimeError(
                    f"refusing to overwrite incompatible run at {method_dir}; "
                    "choose a new output directory"
                )
            expected_ids = [str(row.get("qid", row.get("id", index)))
                            for index, row in enumerate(chunk)]
            prior_audit = (
                audit_legacy_answers(
                    answers,
                    expected_ids,
                    allow_short_answers=args.task == "open_vqa",
                    enforce_behavioral_quality=args.qualification_run,
                )
                if answers.exists() else None
            )
            if (
                prior_meta
                and prior_meta.get("status") in {"done", "skipped"}
                and metrics.exists()
                and prior_audit
                and prior_audit["aligned"]
                and not prior_audit["degenerate_reasons"]
            ):
                record = {
                    "source": args.source,
                    "dataset": args.dataset,
                    "task": args.task,
                    "method": method,
                    "chunk": chunk_id,
                    "status": "skipped",
                    "started_at": None,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "command": None,
                    "output_dir": str(method_dir),
                    "answers": str(answers),
                    "answers_lines": count_jsonl(answers),
                    "metrics": str(metrics),
                    "failure_reason": None,
                    "protocol_id": PROTOCOL_ID,
                    "run_fingerprint": run_fingerprint,
                    "evidence_grade": "A",
                    "output_audit": prior_audit,
                }
                meta.write_text(json.dumps(record, indent=2))
                with state_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                continue
            chunk_file.write_text(json.dumps(chunk, indent=2))
            command = [
                str(PYTHON),
                str(RUNNER),
                "--model-path",
                str(MODEL_PATH),
                "--image-folder",
                str(args.image_folder),
                "--question-file",
                str(chunk_file),
                "--answers-file",
                str(answers),
                "--conv-mode",
                args.conv_mode,
                "--temperature",
                "0",
                "--top_p",
                "1",
                "--num_beams",
                "1",
                "--baseline",
                method,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--seed",
                str(args.seed),
            ]
            if args.disable_keyword_stopping:
                command.append("--disable-keyword-stopping")
            started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            status = run_command(command, MITIGATION_ROOT, env, log, args.dry_run)
            output_audit = None
            if status == 0 and not args.dry_run:
                output_audit = audit_legacy_answers(
                    answers,
                    expected_ids,
                    allow_short_answers=args.task == "open_vqa",
                    enforce_behavioral_quality=args.qualification_run,
                )
                if not output_audit["aligned"] or output_audit["degenerate_reasons"]:
                    status = 86
                    with log.open("a") as handle:
                        handle.write(
                            "\nOUTPUT QUALITY GATE FAILED\n"
                            + json.dumps(output_audit, indent=2) + "\n"
                        )
            if status == 0:
                eval_command = [
                    str(PYTHON),
                    "-m",
                    "corrected_sgta.evaluate_generation_text",
                    "--answers",
                    str(answers),
                    "--output",
                    str(metrics),
                ]
                if args.task == "close_vqa":
                    eval_command = [
                        str(PYTHON),
                        "-m",
                        "corrected_sgta.evaluate_medheval_answers",
                        "--answers",
                        str(answers),
                        "--questions",
                        str(chunk_file),
                        "--output",
                        str(metrics),
                    ]
                status = run_command(
                    eval_command,
                    REPO_ROOT,
                    env,
                    log,
                    args.dry_run,
                )
            finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            record = {
                "source": args.source,
                "dataset": args.dataset,
                "task": args.task,
                "method": method,
                "chunk": chunk_id,
                "status": "dry-run" if args.dry_run else ("done" if status == 0 else "failed"),
                "started_at": started,
                "finished_at": finished,
                "command": command,
                "output_dir": str(method_dir),
                "answers": str(answers),
                "answers_lines": count_jsonl(answers),
                "metrics": str(metrics),
                "failure_reason": None if status == 0 else f"exit status {status}",
                "protocol_id": PROTOCOL_ID,
                "run_fingerprint": run_fingerprint,
                "run_config": run_config,
                "evidence_grade": "A" if status == 0 and not args.dry_run else "C",
                "output_audit": output_audit,
            }
            meta.write_text(json.dumps(record, indent=2))
            with state_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if status != 0 and not args.continue_on_error:
                return status
            if status != 0:
                any_failed = True
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
