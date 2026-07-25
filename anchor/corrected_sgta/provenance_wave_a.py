"""Final code/checkpoint identity for the preregistered Wave-A pipeline."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.provenance_v3 import code_identity as base_code_identity
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import sha256_file


def code_identity(project_root: Path) -> dict:
    identity = base_code_identity(project_root)
    for name in (
        "corrected_sgta/source_bank_v3.py",
        "corrected_sgta/provenance_wave_a.py",
        "corrected_sgta/build_visual_centers_wave_a.py",
        "corrected_sgta/infer_alignment_wave_a.py",
        "corrected_sgta/analyze_alignment_v2.py",
        "corrected_sgta/freeze_alignment_report_v3.py",
        "corrected_sgta/structure_audit_v2.py",
        "corrected_sgta/run_alignment_cxr_wave_a.sh",
        "refine-logs/EXPERIMENT_PLAN_AMENDMENT_V2.md",
    ):
        path = project_root / name
        if path.is_file():
            identity[name] = sha256_file(path)
    return identity
