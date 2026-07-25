"""Small deterministic provenance helpers for SGTA alignment caches."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.models import HULU_PATH, LLAVA_PATH
from corrected_sgta.source_bank_v2 import sha256_file


MODEL_ROOTS = {"hulu": HULU_PATH, "llava": LLAVA_PATH}


def model_identity(model: str) -> dict:
    root = MODEL_ROOTS[model].resolve()
    artifacts = {}
    for name in (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ):
        path = root / name
        if path.is_file():
            artifacts[name] = sha256_file(path)
    return {
        "model": model,
        "checkpoint_root": str(root),
        "checkpoint_revision": root.name,
        "configuration_artifacts": artifacts,
    }


def code_identity(project_root: Path) -> dict:
    names = (
        "corrected_sgta/source_bank_v2.py",
        "corrected_sgta/provenance_v2.py",
        "corrected_sgta/models_alignment.py",
        "corrected_sgta/build_visual_centers_v2.py",
        "corrected_sgta/infer_alignment_v2.py",
        "corrected_sgta/analyze_alignment_v2.py",
        "corrected_sgta/methods.py",
        "corrected_sgta/protocol_v2.py",
    )
    return {
        name: sha256_file(project_root / name)
        for name in names
        if (project_root / name).is_file()
    }
