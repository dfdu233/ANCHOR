"""Final Wave-A code identity layered over content-complete model identity."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.provenance_v3 import code_identity as base_code_identity
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import sha256_file


def code_identity(project_root: Path) -> dict:
    identity = base_code_identity(project_root)
    for name in (
        "corrected_sgta/source_bank_v3.py",
        "corrected_sgta/provenance_v4.py",
        "corrected_sgta/build_visual_centers_v4.py",
        "corrected_sgta/infer_alignment_v4.py",
        "corrected_sgta/run_alignment_cxr_v4.sh",
    ):
        path = project_root / name
        if path.is_file():
            identity[name] = sha256_file(path)
    return identity
