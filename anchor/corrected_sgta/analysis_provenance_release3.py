"""Post-cache adjudication identity for Wave-A release 3."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.analysis_provenance_release2 import analysis_code_identity as base_identity
from corrected_sgta.source_bank_v2 import sha256_file


def analysis_code_identity(project_root: Path) -> dict:
    identity = base_identity(project_root)
    for name in (
        "corrected_sgta/frequency_alignment_release3.py",
        "corrected_sgta/structure_audit_release3.py",
        "corrected_sgta/analysis_provenance_release3.py",
        "corrected_sgta/merge_alignment_gate_release3.py",
    ):
        identity[name] = sha256_file(project_root / name)
    return identity
