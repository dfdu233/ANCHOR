"""Code identity for the preregistered source-spectrum experiment."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.provenance_release2 import center_code_identity
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import sha256_file


def inference_code_identity(project_root: Path) -> dict:
    files = (
        "refine-logs/EXPERIMENT_PLAN_AMENDMENT_V4.md",
        "corrected_sgta/frequency_alignment_source_spectrum.py",
        "corrected_sgta/provenance_source_spectrum.py",
        "corrected_sgta/infer_alignment_source_spectrum.py",
        "corrected_sgta/analyze_alignment_source_spectrum.py",
        "corrected_sgta/structure_audit_source_spectrum.py",
        "corrected_sgta/run_source_spectrum_cxr.sh",
    )
    identity = {name: sha256_file(project_root / name) for name in files}
    identity["center_code_identity"] = center_code_identity(project_root)
    return identity

