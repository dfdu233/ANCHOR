"""Transitive code identity for the frozen feature-transport report."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.source_bank_v2 import sha256_file
from corrected_sgta.transport_provenance_final import final_transport_identity


def complete_transport_identity(project_root: Path) -> dict:
    """Hash every module that can change the reported metrics or gate decision."""
    identity = final_transport_identity(project_root)
    for name in (
        "corrected_sgta/methods.py",
        "corrected_sgta/analyze_alignment_v2.py",
        "corrected_sgta/freeze_alignment_report_release2.py",
        "corrected_sgta/freeze_alignment_report_v3.py",
        "corrected_sgta/merge_alignment_gate_wave_a.py",
        "corrected_sgta/transport_provenance_complete.py",
        "corrected_sgta/merge_feature_transport_gate_complete.py",
        "corrected_sgta/run_feature_transport_cxr_complete.sh",
    ):
        identity[name] = sha256_file(project_root / name)
    return identity
