"""Transitive identity after removing the unintended closure gate."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.provenance_release2 import center_code_identity
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import sha256_file


def inference_code_identity(project_root: Path) -> dict:
    files = (
        "refine-logs/EXPERIMENT_PLAN_AMENDMENT_V4.md",
        "refine-logs/EXPERIMENT_PLAN_AMENDMENT_V4_REVIEW_FIXES.md",
        "refine-logs/EXPERIMENT_PLAN_AMENDMENT_V4_EXECUTION_FIX.md",
        "corrected_sgta/frequency_alignment_source_spectrum_release2.py",
        "corrected_sgta/provenance_source_spectrum_release4.py",
        "corrected_sgta/infer_alignment_source_spectrum_release2.py",
        "corrected_sgta/infer_alignment_source_spectrum_release4.py",
        "corrected_sgta/analyze_alignment_source_spectrum_release3.py",
        "corrected_sgta/structure_audit_source_spectrum_release2.py",
        "corrected_sgta/adjudicate_source_spectrum_release2.py",
        "corrected_sgta/run_source_spectrum_cxr_release4.sh",
        "corrected_sgta/infer_alignment_v2.py",
        "corrected_sgta/infer_alignment_release2.py",
        "corrected_sgta/frequency_alignment_release3.py",
        "corrected_sgta/analyze_alignment_v2.py",
        "corrected_sgta/structure_audit_v2.py",
        "corrected_sgta/structure_audit_wave_a.py",
        "corrected_sgta/protocol_v2.py",
        "corrected_sgta/cache.py",
        "corrected_sgta/infer_ce.py",
        "corrected_sgta/methods.py",
        "corrected_sgta/models_alignment.py",
        "corrected_sgta/source_bank_v2.py",
        "corrected_sgta/source_bank_v3.py",
    )
    identity = {name: sha256_file(project_root / name) for name in files}
    identity["center_code_identity"] = center_code_identity(project_root)
    return identity

