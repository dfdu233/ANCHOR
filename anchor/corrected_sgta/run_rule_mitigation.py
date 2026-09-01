#!/usr/bin/env python3
"""Resumable, fingerprint-closed RULE VQA baseline runner.

Frozen greedy/beam use RULE's native entrypoints. Enabled mitigation methods use
a RULE-prompt-preserving LLaVA-Med backend and require validated activation
sidecars. Other methods remain fail-closed.

This runner targets RULE binary VQA only.  It does not reconstruct or claim
exact reproduction of RULE report-generation experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from corrected_sgta.evaluate_rule_vqa import (
    RuleEvaluationError,
    evaluate_rule_rows,
    load_jsonl,
    write_jsonl,
)
from corrected_sgta.rule_mitigation_backend import (
    CONV_MODE,
    PROTOCOL_VERSION as MITIGATION_BACKEND_PROTOCOL,
    RuleMitigationBackendError,
    generation_config,
    prompt_manifest,
    validate_activation_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("ANCHOR_LEGACY_ROOT", REPO_ROOT))
RULE_ROOT = Path(os.environ.get("ANCHOR_RULE_ROOT", REPO_ROOT / "third_party/RULE"))
RULE_DATA_ROOT = Path(os.environ.get("ANCHOR_RULE_DATA_ROOT", REPO_ROOT / "data/rule"))
MEDHEVAL_ROOT = Path(os.environ.get("ANCHOR_MEDHEVAL_ROOT", REPO_ROOT / "data/medheval"))
MODEL_PATH = Path(
    os.environ.get(
        "ANCHOR_MODEL_PATH",
        "/root/autodl-tmp/LLaVA-Med/microsoft/llava-med-v1.5-mistral-7b",
    )
)
PYTHON = Path(
    os.environ.get("ANCHOR_PYTHON", "/root/autodl-tmp/envs/medheval-mitigation/bin/python")
)
MEDALIGN_ROOT = REPO_ROOT / "third_party/baselines/MedAlign"
MEDALIGN_BACKEND = MEDALIGN_ROOT / "llava-med-1.5/llava/eval/model_vqa.py"
MEDALIGN_COMMIT = "fbefd179018a495407caa89ecbe64eb575da7663"
MITIGATION_ROOT = Path(
    os.environ.get(
        "ANCHOR_MITIGATION_ROOT",
        MEDHEVAL_ROOT / "code/baselines/Mitigation/llava-med-1.5",
    )
)
MITIGATION_EVAL_ROOT = MITIGATION_ROOT / "llava/eval"
MITIGATION_TRANSFORMERS = Path(
    os.environ.get(
        "ANCHOR_MITIGATION_TRANSFORMERS",
        MEDHEVAL_ROOT
        / "code/baselines/Med-LVLMs/llava-med-1.5/transformers-4.37.2/src",
    )
)
MITIGATION_BACKEND = ROOT / "corrected_sgta/rule_mitigation_backend.py"
MITIGATION_GENERATION_UTILS = (
    MITIGATION_TRANSFORMERS / "transformers/generation/utils.py"
)
OPERA_ROOT = REPO_ROOT / "third_party/baselines/OPERA"
OPERA_COMMIT = "e1bb7632508bdfaa5cdaf5b69b64419688107da8"
OPERA_GENERATION_UTILS = (
    OPERA_ROOT / "transformers-4.29.2/src/transformers/generation/utils.py"
)
PAI_ROOT = REPO_ROOT / "third_party/baselines/PAI"
PAI_COMMIT = "9bbd8bd57a0b0923f996197e4bd3e02cc10b8d58"
PAI_PORT_FILES = {
    name: MITIGATION_EVAL_ROOT / "PAI_files" / name
    for name in ("CFG.py", "attention.py", "model_loader.py")
}
PAI_OFFICIAL_FILES = {
    name: PAI_ROOT / name for name in ("CFG.py", "attention.py")
}
MEDHEVAL_COMMIT = "7f7d980e676890ed99513f4629fd4d5af19bc8b9"
M3ID_PORT_FILES = {
    "avisc_sample.py": MITIGATION_EVAL_ROOT / "avisc_utils/avisc_sample.py",
    "llava_mistral.py": (
        MITIGATION_ROOT / "llava/model/language_model/llava_mistral.py"
    ),
    "readme.md": MITIGATION_ROOT / "readme.md",
}
AVISC_ROOT = REPO_ROOT / "third_party/baselines/AvisC"
AVISC_COMMIT = "772eba499dcbda0eaaf844fda2fdf5057dcd0175"
AVISC_OFFICIAL_SAMPLE = AVISC_ROOT / "avisc_utils/avisc_sample.py"
AVISC_DYNAMIC_SAMPLE = ROOT / "corrected_sgta/avisc_sample_dynamic.py"
PROTOCOL_VERSION = "rule-vqa-mitigation-runner-v8-leading-native-eos"


def mitigation_max_new_tokens(method: str) -> int:
    # MedHEval's OPERA VQA branch uses a 128-token budget; other mitigation
    # branches keep the local 1024-token yes/no budget used in prior runs.
    return 128 if method == "OPERA" else 1024


class RuleRunnerError(RuntimeError):
    """Raised when protocol identity or an artifact fails closed."""


@dataclass(frozen=True)
class DatasetSpec:
    annotation: str
    image_root: str
    entrypoint: str


@dataclass(frozen=True)
class MethodSpec:
    enabled: bool
    backend: str
    implementation: str
    upstream: str
    temperature: float
    num_beams: int
    paper_protocol_exact: bool
    note: str


DATASETS = {
    "iuxray": DatasetSpec(
        annotation=str(RULE_DATA_ROOT / "test/iuxray_test.jsonl"),
        image_root=str(MEDHEVAL_ROOT / "images/IU-Xray"),
        entrypoint=str(RULE_ROOT / "llava/eval/model_vqa_iuxray.py"),
    ),
    "harvard": DatasetSpec(
        annotation=str(RULE_DATA_ROOT / "test/harvard_test.jsonl"),
        image_root=os.environ.get(
            "ANCHOR_HARVARD_IMAGE_ROOT",
            "/root/autodl-tmp/source_data/FairVLMed/extracted/Test",
        ),
        entrypoint=str(RULE_ROOT / "llava/eval/model_vqa_harvard.py"),
    ),
    "mimic": DatasetSpec(
        annotation=str(RULE_DATA_ROOT / "test/mimic_test.jsonl"),
        image_root=str(MEDHEVAL_ROOT / "images"),
        entrypoint=str(RULE_ROOT / "llava/eval/model_vqa_mimic.py"),
    ),
}

METHODS = {
    "greedy": MethodSpec(
        enabled=True,
        backend="rule_native",
        implementation="RULE dataset-specific model_vqa_*.py",
        upstream="https://github.com/richard-peng-xia/RULE",
        temperature=0.0,
        num_beams=1,
        paper_protocol_exact=False,
        note="Frozen deterministic local reconstruction; RULE did not publish its table evaluator.",
    ),
    "beam": MethodSpec(
        enabled=True,
        backend="rule_native",
        implementation="RULE dataset-specific model_vqa_*.py",
        upstream="https://github.com/richard-peng-xia/RULE",
        temperature=0.0,
        num_beams=5,
        paper_protocol_exact=False,
        note="Beam width 5 is frozen locally; the RULE paper/repository does not specify its table beam width.",
    ),
    "DoLa": MethodSpec(
        enabled=True,
        backend="rule_protocol_medheval_port",
        implementation=str(MITIGATION_BACKEND),
        upstream="https://github.com/voidism/DoLa",
        temperature=0.9,
        num_beams=1,
        paper_protocol_exact=False,
        note=(
            "RULE-exact prompt frontend with the MedHEval LLaVA-Med DoLa "
            "generation port; requires activation-sidecar validation."
        ),
    ),
    "VCD": MethodSpec(
        enabled=True,
        backend="rule_protocol_medheval_port",
        implementation=str(MITIGATION_BACKEND),
        upstream="https://github.com/DAMO-NLP-SG/VCD",
        temperature=1.0,
        num_beams=1,
        paper_protocol_exact=False,
        note=(
            "RULE-exact prompt frontend with the MedHEval LLaVA-Med VCD "
            "generation port; requires activation-sidecar validation."
        ),
    ),
    "OPERA": MethodSpec(
        enabled=True,
        backend="rule_protocol_medheval_port",
        implementation=str(MITIGATION_BACKEND),
        upstream="https://github.com/shikiw/OPERA",
        temperature=0.0,
        num_beams=5,
        paper_protocol_exact=False,
        note=(
            "RULE-exact prompt frontend with the OPERA/MedHEval generation "
            "port, official decoding defaults, dynamic LLaVA image-token "
            "positions, and activation-sidecar validation."
        ),
    ),
    "AVISC": MethodSpec(
        enabled=True,
        backend="rule_protocol_medheval_port",
        implementation=str(MITIGATION_BACKEND),
        upstream="https://github.com/sangminwoo/AvisC",
        temperature=1.0,
        num_beams=1,
        paper_protocol_exact=False,
        note=(
            "RULE-exact prompt frontend with official AVISC sampling, the "
            "MedHEval Mistral masking port, dynamic image-token coordinates, "
            "and activation-sidecar validation."
        ),
    ),
    "M3ID": MethodSpec(
        enabled=True,
        backend="rule_protocol_medheval_port",
        implementation=str(MITIGATION_BACKEND),
        upstream="https://openaccess.thecvf.com/content/CVPR2024/papers/Favero_Multi-Modal_Hallucination_Control_by_Visual_Information_Grounding_CVPR_2024_paper.pdf",
        temperature=1.0,
        num_beams=1,
        paper_protocol_exact=False,
        note=(
            "RULE-exact prompt frontend with MedHEval's paper-based M3ID "
            "reimplementation; no official code was released. Uses dynamic "
            "image removal and activation-sidecar validation."
        ),
    ),
    "DAMRO": MethodSpec(
        enabled=False,
        backend="medalign_third_party_reimplementation",
        implementation=str(MEDALIGN_BACKEND),
        upstream="paper only; no official implementation released",
        temperature=0.0,
        num_beams=1,
        paper_protocol_exact=False,
        note="Reserved; current port has a CUDA device-side assertion.",
    ),
    "PAI": MethodSpec(
        enabled=True,
        backend="rule_protocol_medheval_port",
        implementation=str(MITIGATION_BACKEND),
        upstream="https://github.com/LALBJ/PAI",
        temperature=0.0,
        num_beams=1,
        paper_protocol_exact=False,
        note=(
            "RULE-exact prompt frontend with the official PAI attention/CFG "
            "logic adapted by MedHEval for Mistral, dynamic image-token "
            "ranges, and activation-sidecar validation."
        ),
    ),
    "PAIControl": MethodSpec(
        enabled=True,
        backend="rule_protocol_medheval_port",
        implementation=str(MITIGATION_BACKEND),
        upstream="identity control for https://github.com/LALBJ/PAI",
        temperature=0.0,
        num_beams=1,
        paper_protocol_exact=False,
        note=(
            "No-intervention control using the exact PAI MedHEval runtime, "
            "model loader, RULE prompt frontend, and generation budget."
        ),
    ),
    "VISTA": MethodSpec(
        enabled=False,
        backend="not_integrated",
        implementation="none",
        upstream="https://github.com/LzVv123456/VISTA",
        temperature=0.0,
        num_beams=1,
        paper_protocol_exact=False,
        note="Reserved; requires a target-architecture visual steering vector.",
    ),
}


def stable_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash every model file; intentionally expensive once, never heuristic."""
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RuleRunnerError(f"model directory has no files: {path}")
    entries = [
        {
            "path": str(item.relative_to(path)),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in files
    ]
    return hashlib.sha256(stable_json(entries).encode()).hexdigest(), entries


def fingerprint(payload: dict[str, Any]) -> str:
    envelope = {"protocol_version": PROTOCOL_VERSION, **payload}
    return hashlib.sha256(stable_json(envelope).encode()).hexdigest()


def git_identity(path: Path) -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=path,
            text=True,
            capture_output=True,
            check=False,
        )

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    if not head.returncode and not status.returncode:
        if status.stdout.strip():
            raise RuleRunnerError(
                f"tracked files are dirty in required source repo: {path}"
            )
        return {"path": str(path), "commit": head.stdout.strip(), "tracked_clean": True}
    if not path.is_dir():
        raise RuleRunnerError(f"required source directory is missing: {path}")
    tree_sha, files = sha256_tree(path)
    return {
        "path": str(path),
        "commit": None,
        "tracked_clean": None,
        "vendored_tree_sha256": tree_sha,
        "vendored_file_count": len(files),
        "identity_note": "source directory is vendored without .git metadata",
    }


def image_identity(rows: Iterable[dict[str, Any]], image_root: Path) -> dict[str, Any]:
    relative_paths = sorted({str(row["image"]) for row in rows})
    entries = []
    missing = []
    for relative in relative_paths:
        path = image_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            raise RuleRunnerError(f"invalid image {path}: {error}") from error
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if missing:
        raise RuleRunnerError(
            f"{len(missing)} selected images are missing under {image_root}; "
            f"first={missing[:3]}"
        )
    return {
        "root": str(image_root.resolve()),
        "unique_images": len(entries),
        "sha256": hashlib.sha256(stable_json(entries).encode()).hexdigest(),
    }


def assert_enabled(method: str) -> MethodSpec:
    spec = METHODS[method]
    if not spec.enabled:
        raise RuleRunnerError(
            f"{method} is provenance-registered but not enabled: {spec.note}"
        )
    return spec


def build_command(
    *,
    python: Path,
    entrypoint: Path,
    model_path: Path,
    image_root: Path,
    questions: Path,
    answers: Path,
    method: str,
    dataset: str | None = None,
    activation_file: Path | None = None,
) -> list[str]:
    spec = assert_enabled(method)
    if spec.backend == "rule_protocol_medheval_port":
        if dataset not in DATASETS:
            raise RuleRunnerError("ported mitigation command requires a RULE dataset")
        if activation_file is None:
            raise RuleRunnerError(
                "ported mitigation command requires an activation file"
            )
        return [
            str(python),
            str(MITIGATION_BACKEND),
            "--dataset",
            dataset,
            "--method",
            method,
            "--model-path",
            str(model_path),
            "--image-folder",
            str(image_root),
            "--question-file",
            str(questions),
            "--answers-file",
            str(answers),
            "--activation-file",
            str(activation_file),
            "--conv-mode",
            CONV_MODE,
            "--max-new-tokens",
            str(mitigation_max_new_tokens(method)),
            "--seed",
            "0",
        ]
    if spec.backend != "rule_native":
        raise RuleRunnerError(f"unsupported enabled backend: {spec.backend}")
    return [
        str(python),
        str(entrypoint),
        "--model-path",
        str(model_path),
        "--image-folder",
        str(image_root),
        "--question-file",
        str(questions),
        "--answers-file",
        str(answers),
        "--conv-mode",
        CONV_MODE,
        "--temperature",
        str(spec.temperature),
        "--num_beams",
        str(spec.num_beams),
    ]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def artifact_hashes(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuleRunnerError(f"completed artifact is missing: {path}")
        result[name] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def completed_job_valid(
    meta: dict[str, Any],
    expected_fingerprint: str,
    paths: dict[str, Path],
    expected_qids: list[str],
) -> bool:
    if meta.get("status") != "complete":
        return False
    if meta.get("fingerprint") != expected_fingerprint:
        raise RuleRunnerError("existing job fingerprint differs; refusing reuse")
    current = artifact_hashes(paths)
    if current != meta.get("artifacts"):
        raise RuleRunnerError(
            "completed artifact hashes differ; refusing reuse "
            f"current={current} meta={meta.get('artifacts')}"
        )
    answer_qids = [str(row["question_id"]) for row in load_jsonl(paths["answers"])]
    record_qids = [str(row["question_id"]) for row in load_jsonl(paths["records"])]
    if answer_qids != expected_qids or record_qids != expected_qids:
        raise RuleRunnerError("completed artifact qids/order differ; refusing reuse")
    metrics = json.loads(paths["metrics"].read_text())
    if metrics.get("n") != len(expected_qids):
        raise RuleRunnerError("completed metrics sample count differs; refusing reuse")
    return True


def split_chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def parse_mapping(values: list[str], base: dict[str, str]) -> dict[str, Path]:
    result = {key: Path(value) for key, value in base.items()}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or key not in result:
            raise RuleRunnerError(f"expected one of {sorted(result)}=PATH, got {value!r}")
        result[key] = Path(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "corrected_runs/rule_mitigation_v2",
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["iuxray"])
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=["greedy", "beam"])
    parser.add_argument("--image-root", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Smoke-only deterministic prefix before chunking")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def runtime_identity(
    python: Path, *, pythonpath: str | None = None
) -> dict[str, Any]:
    code = (
        "import importlib.util,inspect,json,sys,torch,transformers,PIL;"
        "from transformers.generation.utils import GenerationMixin;"
        "print(json.dumps({\"python\":sys.version,\"torch\":torch.__version__,"
        "\"transformers\":transformers.__version__,\"pillow\":PIL.__version__,"
        "\"generation_utils\":inspect.getfile(GenerationMixin),"
        "\"opera_available\":hasattr(GenerationMixin,\"opera_beam_search\"),"
        "\"pai_available\":importlib.util.find_spec(\"PAI_files\") is not None,"
        "\"m3id_available\":importlib.util.find_spec(\"avisc_utils\") is not None,"
        "\"avisc_available\":importlib.util.find_spec(\"corrected_sgta.avisc_sample_dynamic\") is not None},"
        "sort_keys=True))"
    )
    environment = os.environ.copy()
    if pythonpath is not None:
        environment["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [str(python), "-c", code],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if result.returncode:
        raise RuleRunnerError("cannot fingerprint runtime: " + result.stderr.strip())
    return json.loads(result.stdout)


def _validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise RuleRunnerError("--chunk-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise RuleRunnerError("--limit must be positive")
    for method in args.methods:
        assert_enabled(method)
    if not args.python.is_file():
        raise RuleRunnerError(f"Python executable is missing: {args.python}")
    if not args.model_path.is_dir():
        raise RuleRunnerError(f"model directory is missing: {args.model_path}")
    if any(
        METHODS[method].backend == "rule_protocol_medheval_port"
        for method in args.methods
    ):
        for required in (
            MITIGATION_BACKEND,
            MITIGATION_ROOT,
            MITIGATION_TRANSFORMERS,
            MITIGATION_GENERATION_UTILS,
            *PAI_PORT_FILES.values(),
            *M3ID_PORT_FILES.values(),
            AVISC_OFFICIAL_SAMPLE,
            AVISC_DYNAMIC_SAMPLE,
        ):
            if not required.exists():
                raise RuleRunnerError(
                    f"required mitigation backend path is missing: {required}"
                )


def execution_context(
    method: str, *, gpu: str, base_environment: dict[str, str]
) -> tuple[Path, dict[str, str]]:
    """Return method-specific cwd/environment; never share ambiguous imports."""
    spec = assert_enabled(method)
    environment = dict(base_environment)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    if spec.backend == "rule_native":
        environment["PYTHONPATH"] = str(RULE_ROOT)
        return RULE_ROOT, environment
    if spec.backend == "rule_protocol_medheval_port":
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(MITIGATION_EVAL_ROOT),
                str(MITIGATION_ROOT),
                str(MITIGATION_TRANSFORMERS),
                str(ROOT),
            )
        )
        return ROOT, environment
    raise RuleRunnerError(f"unsupported enabled backend: {spec.backend}")


def main() -> int:
    args = parse_args()
    _validate_args(args)
    args.out = args.out.resolve()
    image_roots = parse_mapping(
        args.image_root, {key: spec.image_root for key, spec in DATASETS.items()}
    )
    rule_git = git_identity(RULE_ROOT)
    model_sha, model_files = sha256_tree(args.model_path)
    native_runtime = runtime_identity(args.python)
    mitigation_methods = [
        method
        for method in args.methods
        if METHODS[method].backend == "rule_protocol_medheval_port"
    ]
    mitigation_runtime = None
    mitigation_backend_audit = None
    if mitigation_methods:
        mitigation_pythonpath = os.pathsep.join(
            (
                str(MITIGATION_EVAL_ROOT),
                str(MITIGATION_ROOT),
                str(MITIGATION_TRANSFORMERS),
                str(ROOT),
            )
        )
        mitigation_runtime = runtime_identity(
            args.python, pythonpath=mitigation_pythonpath
        )
        if "OPERA" in args.methods and not mitigation_runtime.get("opera_available"):
            raise RuleRunnerError("mitigation runtime does not expose OPERA decoding")
        if (
            {"PAI", "PAIControl"} & set(args.methods)
        ) and not mitigation_runtime.get("pai_available"):
            raise RuleRunnerError("mitigation runtime does not expose PAI modules")
        if "M3ID" in args.methods and not mitigation_runtime.get("m3id_available"):
            raise RuleRunnerError("mitigation runtime does not expose M3ID modules")
        if "AVISC" in args.methods and not mitigation_runtime.get("avisc_available"):
            raise RuleRunnerError("mitigation runtime does not expose AVISC modules")
        mitigation_backend_audit = {
            "generation_utils": str(MITIGATION_GENERATION_UTILS),
            "generation_utils_sha256": sha256_file(MITIGATION_GENERATION_UTILS),
            "opera_source": {
                "path": str(OPERA_ROOT),
                "commit": OPERA_COMMIT,
                "official_generation_utils_sha256": (
                    sha256_file(OPERA_GENERATION_UTILS)
                    if OPERA_GENERATION_UTILS.is_file()
                    else None
                ),
                "license": "MIT",
            },
            "pai_source": {
                "path": str(PAI_ROOT),
                "commit": PAI_COMMIT,
                "port_files": {
                    name: sha256_file(path)
                    for name, path in PAI_PORT_FILES.items()
                },
                "official_files": {
                    name: sha256_file(path)
                    for name, path in PAI_OFFICIAL_FILES.items()
                    if path.is_file()
                },
                "license": "MIT",
            },
            "m3id_source": {
                "path": str(MITIGATION_ROOT),
                "medheval_commit": MEDHEVAL_COMMIT,
                "port_files": {
                    name: sha256_file(path)
                    for name, path in M3ID_PORT_FILES.items()
                },
                "official_code_released": False,
                "implementation_status": "MedHEval paper-based reimplementation",
            },
            "avisc_source": {
                "path": str(AVISC_ROOT),
                "commit": AVISC_COMMIT,
                "official_sample_sha256": sha256_file(AVISC_OFFICIAL_SAMPLE),
                "dynamic_sample_sha256": sha256_file(AVISC_DYNAMIC_SAMPLE),
                "medheval_mistral_sha256": sha256_file(
                    M3ID_PORT_FILES["llava_mistral.py"]
                ),
                "license": "MIT",
            },
        }
    runner_sha = sha256_file(Path(__file__))
    evaluator_path = Path(sys.modules[evaluate_rule_rows.__module__].__file__)
    evaluator_sha = sha256_file(evaluator_path)
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "RULE binary VQA only; not report generation",
        "paper_exact_reproduction": False,
        "rule_source": rule_git,
        "model": {
            "path": str(args.model_path.resolve()),
            "tree_sha256": model_sha,
            "files": model_files,
        },
        "python": str(args.python.resolve()),
        "runtime_versions": {
            "rule_native": native_runtime,
            "mitigation_port": mitigation_runtime,
        },
        "runner_sha256": runner_sha,
        "evaluator_sha256": evaluator_sha,
        "mitigation_backend_audit": mitigation_backend_audit,
        "reserved_backend_audit": {
            "medalign_path": str(MEDALIGN_ROOT),
            "medalign_commit": MEDALIGN_COMMIT,
            "medalign_backend_sha256": (
                sha256_file(MEDALIGN_BACKEND) if MEDALIGN_BACKEND.is_file() else None
            ),
            "license": "No LICENSE file found; local audit/execution only, do not copy or redistribute.",
            "report_status": (
                "Not exact: MEDALIGN shell commands reference report_eval files "
                "outside the repository."
            ),
        },
        "methods": {method: asdict(METHODS[method]) for method in args.methods},
        "datasets": {},
        "chunk_size": args.chunk_size,
        "limit": args.limit,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    state: list[dict[str, Any]] = []
    base_environment = os.environ.copy()

    for dataset in args.datasets:
        spec = DATASETS[dataset]
        annotation = Path(spec.annotation)
        entrypoint = Path(spec.entrypoint)
        image_root = image_roots[dataset]
        for required in (annotation, entrypoint):
            if not required.is_file():
                raise RuleRunnerError(f"required RULE file is missing: {required}")
        rows = load_jsonl(annotation)
        if args.limit is not None:
            rows = rows[: args.limit]
        qids = [str(row["question_id"]) for row in rows]
        if len(qids) != len(set(qids)):
            raise RuleRunnerError(f"{dataset}: duplicate question ids")
        dataset_images = image_identity(rows, image_root)
        manifest["datasets"][dataset] = {
            "annotation": str(annotation),
            "annotation_sha256": sha256_file(annotation),
            "entrypoint": str(entrypoint),
            "entrypoint_sha256": sha256_file(entrypoint),
            "selected_questions": len(rows),
            "images": dataset_images,
        }
        atomic_json(args.out / "manifest.json", manifest)

        for method in args.methods:
            method_spec = METHODS[method]
            execution_entrypoint = (
                entrypoint
                if method_spec.backend == "rule_native"
                else MITIGATION_BACKEND
            )
            for chunk_id, chunk in enumerate(split_chunks(rows, args.chunk_size)):
                job = args.out / dataset / method / f"chunk_{chunk_id:04d}"
                questions = job / "questions.jsonl"
                answers = job / "answers.raw.jsonl"
                records = job / "records.jsonl"
                metrics = job / "metrics.json"
                activation = job / "activation.json"
                meta_path = job / "meta.json"
                log_path = job / "run.log"
                job.mkdir(parents=True, exist_ok=True)
                expected_qids = [str(row["question_id"]) for row in chunk]
                if questions.exists():
                    existing_qids = [
                        str(row["question_id"]) for row in load_jsonl(questions)
                    ]
                    if existing_qids != expected_qids:
                        raise RuleRunnerError(
                            f"{job}: existing questions differ; refusing overwrite"
                        )
                else:
                    write_jsonl(questions, chunk)
                payload = {
                    "dataset": dataset,
                    "method": method,
                    "method_provenance": asdict(method_spec),
                    "decoding": {
                        "conv_mode": CONV_MODE,
                        **(
                            generation_config(method, mitigation_max_new_tokens(method))
                            if method_spec.backend
                            == "rule_protocol_medheval_port"
                            else {
                                "do_sample": method_spec.temperature > 0,
                                "temperature": method_spec.temperature,
                                "num_beams": method_spec.num_beams,
                                "max_new_tokens": 1024,
                            }
                        ),
                        "configuration_source": (
                            "frozen RULE-compatible mitigation backend"
                            if method_spec.backend
                            == "rule_protocol_medheval_port"
                            else "fixed in RULE entrypoint"
                        ),
                    },
                    "chunk_id": chunk_id,
                    "expected_qids": expected_qids,
                    "questions_sha256": sha256_file(questions),
                    "annotation_sha256": sha256_file(annotation),
                    "image_identity": image_identity(chunk, image_root),
                    "entrypoint": str(execution_entrypoint),
                    "entrypoint_sha256": sha256_file(execution_entrypoint),
                    "mitigation_backend_protocol": (
                        MITIGATION_BACKEND_PROTOCOL
                        if method_spec.backend == "rule_protocol_medheval_port"
                        else None
                    ),
                    "mitigation_generation_utils_sha256": (
                        sha256_file(MITIGATION_GENERATION_UTILS)
                        if method_spec.backend == "rule_protocol_medheval_port"
                        else None
                    ),
                    "method_external_source": (
                        {
                            "commit": PAI_COMMIT,
                            "port_files": {
                                name: sha256_file(path)
                                for name, path in PAI_PORT_FILES.items()
                            },
                            "official_files": {
                                name: sha256_file(path)
                                for name, path in PAI_OFFICIAL_FILES.items()
                                if path.is_file()
                            },
                        }
                        if method in {"PAI", "PAIControl"}
                        else (
                            {
                                "commit": AVISC_COMMIT,
                                "official_sample_sha256": sha256_file(
                                    AVISC_OFFICIAL_SAMPLE
                                ),
                                "dynamic_sample_sha256": sha256_file(
                                    AVISC_DYNAMIC_SAMPLE
                                ),
                                "medheval_mistral_sha256": sha256_file(
                                    M3ID_PORT_FILES["llava_mistral.py"]
                                ),
                            }
                            if method == "AVISC"
                            else (
                                {
                                    "medheval_commit": MEDHEVAL_COMMIT,
                                    "port_files": {
                                        name: sha256_file(path)
                                        for name, path in M3ID_PORT_FILES.items()
                                    },
                                    "official_code_released": False,
                                }
                                if method == "M3ID"
                                else (
                                {
                                    "commit": OPERA_COMMIT,
                                    "official_generation_utils_sha256": (
                                        sha256_file(OPERA_GENERATION_UTILS)
                                        if OPERA_GENERATION_UTILS.is_file()
                                        else None
                                    ),
                                }
                                    if method == "OPERA"
                                    else None
                                )
                            )
                        )
                    ),
                    "ordered_prompt_sha256": prompt_manifest(
                        dataset, chunk
                    )["ordered_prompt_sha256"],
                    "rule_commit": rule_git["commit"],
                    "model_tree_sha256": model_sha,
                    "runtime_versions": (
                        mitigation_runtime
                        if method_spec.backend == "rule_protocol_medheval_port"
                        else native_runtime
                    ),
                    "runner_sha256": runner_sha,
                    "evaluator_sha256": evaluator_sha,
                }
                job_fingerprint = fingerprint(payload)
                paths = {
                    "answers": answers,
                    "records": records,
                    "metrics": metrics,
                }
                if method_spec.backend == "rule_protocol_medheval_port":
                    paths["activation"] = activation
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    if completed_job_valid(
                        meta, job_fingerprint, paths, expected_qids
                    ):
                        state.append(
                            {
                                "dataset": dataset,
                                "method": method,
                                "chunk": chunk_id,
                                "status": "reused_complete",
                                "fingerprint": job_fingerprint,
                            }
                        )
                        continue
                elif any(path.exists() for path in paths.values()):
                    raise RuleRunnerError(
                        f"{job}: artifacts exist without meta; refusing overwrite"
                    )

                running_meta = {
                    "protocol_version": PROTOCOL_VERSION,
                    "fingerprint": job_fingerprint,
                    "status": "planned" if args.dry_run else "running",
                    "payload": payload,
                    "command": build_command(
                        python=args.python,
                        entrypoint=entrypoint,
                        model_path=args.model_path,
                        image_root=image_root,
                        questions=questions,
                        answers=answers,
                        method=method,
                        dataset=dataset,
                        activation_file=activation,
                    ),
                }
                atomic_json(meta_path, running_meta)
                if args.dry_run:
                    state.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "chunk": chunk_id,
                            "status": "planned",
                            "fingerprint": job_fingerprint,
                        }
                    )
                    atomic_json(args.out / "queue_state.json", state)
                    continue

                command = running_meta["command"]
                cwd, environment = execution_context(
                    method, gpu=args.gpu, base_environment=base_environment
                )
                started = time.time()
                with log_path.open("a") as log:
                    log.write("$ " + " ".join(command) + "\n")
                    log.flush()
                    result = subprocess.run(
                        command,
                        cwd=cwd,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if result.returncode:
                    running_meta.update(
                        {
                            "status": "failed",
                            "returncode": result.returncode,
                            "elapsed_seconds": time.time() - started,
                        }
                    )
                    atomic_json(meta_path, running_meta)
                    state.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "chunk": chunk_id,
                            "status": f"failed:{result.returncode}",
                            "fingerprint": job_fingerprint,
                        }
                    )
                    atomic_json(args.out / "queue_state.json", state)
                    if not args.continue_on_error:
                        return result.returncode
                    continue
                try:
                    answer_rows = load_jsonl(answers)
                    if method_spec.backend == "rule_protocol_medheval_port":
                        if not activation.is_file():
                            raise RuleRunnerError(
                                "mitigation activation sidecar is missing"
                            )
                        activation_report = json.loads(activation.read_text())
                        validate_activation_report(
                            activation_report,
                            method=method,
                            expected_samples=len(chunk),
                            expected_prompt_sha256=running_meta["payload"][
                                "ordered_prompt_sha256"
                            ],
                            expected_max_new_tokens=mitigation_max_new_tokens(method),
                        )
                    report, normalized = evaluate_rule_rows(chunk, answer_rows)
                except (
                    OSError,
                    json.JSONDecodeError,
                    RuleEvaluationError,
                    RuleMitigationBackendError,
                    RuleRunnerError,
                ) as error:
                    running_meta.update(
                        {
                            "status": "failed_validation",
                            "error": str(error),
                            "elapsed_seconds": time.time() - started,
                        }
                    )
                    atomic_json(meta_path, running_meta)
                    if not args.continue_on_error:
                        raise RuleRunnerError(str(error)) from error
                    continue
                atomic_json(metrics, report)
                write_jsonl(records, normalized)
                running_meta.update(
                    {
                        "status": "complete",
                        "elapsed_seconds": time.time() - started,
                        "finished_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "artifacts": artifact_hashes(paths),
                    }
                )
                atomic_json(meta_path, running_meta)
                state.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "chunk": chunk_id,
                        "status": "complete",
                        "fingerprint": job_fingerprint,
                    }
                )
                atomic_json(args.out / "queue_state.json", state)
    atomic_json(args.out / "manifest.json", manifest)
    atomic_json(args.out / "queue_state.json", state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuleRunnerError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
