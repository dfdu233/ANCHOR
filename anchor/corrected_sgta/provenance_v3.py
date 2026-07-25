"""Content-complete identity for checkpoints, custom model code, and SGTA code."""

from __future__ import annotations

import importlib.metadata
from functools import lru_cache
from pathlib import Path

from corrected_sgta.models import HULU_PATH, LLAVA_PATH, LLAVA_REPO
from corrected_sgta.source_bank_v2 import sha256_file


MODEL_ROOTS = {"hulu": HULU_PATH, "llava": LLAVA_PATH}


def hash_paths(root: Path, paths: list[Path]) -> dict[str, dict]:
    output = {}
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        output[key] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return output


@lru_cache(maxsize=2)
def model_identity(model: str) -> dict:
    root = MODEL_ROOTS[model].resolve()
    small = []
    for pattern in ("*.json", "*.py", "*.model", "*.txt"):
        small.extend(root.glob(pattern))
    weights = list(root.glob("*.safetensors"))
    artifacts = hash_paths(root, small + weights)
    external_code = {}
    if model == "llava":
        llava_root = (LLAVA_REPO / "llava").resolve()
        external_code = hash_paths(llava_root, list(llava_root.rglob("*.py")))
    return {
        "identity_version": "model-content-identity-v3",
        "model": model,
        "checkpoint_root": str(root),
        "checkpoint_revision": root.name,
        "checkpoint_artifacts": artifacts,
        "external_model_code": external_code,
        "runtime_packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "pillow")
        },
    }


def code_identity(project_root: Path) -> dict:
    names = (
        "corrected_sgta/source_bank_v2.py",
        "corrected_sgta/provenance_v3.py",
        "corrected_sgta/frequency_alignment_v2.py",
        "corrected_sgta/models.py",
        "corrected_sgta/models_surface.py",
        "corrected_sgta/models_alignment.py",
        "corrected_sgta/infer_ce.py",
        "corrected_sgta/cache.py",
        "corrected_sgta/protocol.py",
        "corrected_sgta/protocol_v2.py",
        "corrected_sgta/methods.py",
        "corrected_sgta/build_visual_centers_v2.py",
        "corrected_sgta/build_visual_centers_v3.py",
        "corrected_sgta/infer_alignment_v2.py",
        "corrected_sgta/infer_alignment_v3.py",
        "corrected_sgta/analyze_alignment_v2.py",
        "corrected_sgta/freeze_alignment_report_v3.py",
        "corrected_sgta/structure_audit_v2.py",
    )
    return {
        name: sha256_file(project_root / name)
        for name in names
        if (project_root / name).is_file()
    }
