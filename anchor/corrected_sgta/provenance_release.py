"""Content identity for the GPU-released Wave-A entry points."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.provenance_wave_a import code_identity as base_code_identity
from corrected_sgta.source_bank_v2 import sha256_file


def code_identity(project_root: Path) -> dict:
    identity = base_code_identity(project_root)
    for name in (
        "corrected_sgta/provenance_v2.py",
        "corrected_sgta/provenance_release.py",
        "corrected_sgta/build_visual_centers_release.py",
        "corrected_sgta/infer_alignment_release.py",
        "corrected_sgta/structure_audit_wave_a.py",
        "corrected_sgta/merge_alignment_gate_wave_a.py",
        "corrected_sgta/run_alignment_cxr_release.sh",
    ):
        path = project_root / name
        if not path.is_file():
            raise RuntimeError(f"missing behavior-identity artifact: {path}")
        identity[name] = sha256_file(path)
    return identity
