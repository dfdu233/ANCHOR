"""Transitive identity after the V5 integrity review."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.provenance_release2 import center_code_identity
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import sha256_file


def model_source_residual_identity_release2(project_root: Path) -> dict:
    files = (
        "refine-logs/EXPERIMENT_PLAN_AMENDMENT_V5.md",
        "refine-logs/EXPERIMENT_PLAN_AMENDMENT_V5_REVIEW_FIXES.md",
        "corrected_sgta/model_source_residual_provenance_release2.py",
        "corrected_sgta/infer_model_source_residual.py",
        "corrected_sgta/infer_model_source_residual_release2.py",
        "corrected_sgta/analyze_model_source_residual_release2.py",
        "corrected_sgta/audit_model_source_residual_release2.py",
        "corrected_sgta/adjudicate_model_source_residual_release2.py",
        "corrected_sgta/run_model_source_residual_release2.sh",
        "corrected_sgta/models_transport.py",
        "corrected_sgta/models_alignment.py",
        "corrected_sgta/models_surface.py",
        "corrected_sgta/analyze_alignment_v2.py",
        "corrected_sgta/protocol.py",
        "corrected_sgta/protocol_v2.py",
        "corrected_sgta/cache.py",
        "corrected_sgta/infer_ce.py",
        "corrected_sgta/methods.py",
        "corrected_sgta/source_bank_v2.py",
        "corrected_sgta/source_bank_v3.py",
    )
    identity = {name: sha256_file(project_root / name) for name in files}
    identity["center_code_identity"] = center_code_identity(project_root)
    return identity

