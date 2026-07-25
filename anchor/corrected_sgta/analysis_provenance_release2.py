"""Identity of post-cache Wave-A release-2 adjudication code."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.source_bank_v2 import sha256_file


def analysis_code_identity(project_root: Path) -> dict:
    names = (
        "corrected_sgta/analyze_alignment_v2.py",
        "corrected_sgta/freeze_alignment_report_v3.py",
        "corrected_sgta/freeze_alignment_report_release2.py",
        "corrected_sgta/structure_audit_v2.py",
        "corrected_sgta/structure_audit_wave_a.py",
        "corrected_sgta/structure_audit_release2.py",
        "corrected_sgta/merge_alignment_gate_wave_a.py",
        "corrected_sgta/merge_alignment_gate_release2.py",
        "corrected_sgta/analysis_provenance_release2.py",
    )
    return {name: sha256_file(project_root / name) for name in names}
