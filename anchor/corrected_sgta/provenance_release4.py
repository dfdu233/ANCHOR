"""Final Wave-A release-4 identity."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.provenance_release2 import center_code_identity
from corrected_sgta.provenance_release3 import inference_code_identity as base_identity
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import sha256_file


def inference_code_identity(project_root: Path) -> dict:
    identity = base_identity(project_root)
    for name in (
        "corrected_sgta/provenance_release4.py",
        "corrected_sgta/infer_alignment_release4.py",
        "corrected_sgta/analyze_alignment_release4.py",
        "corrected_sgta/freeze_alignment_report_release4.py",
        "corrected_sgta/structure_audit_release4.py",
        "corrected_sgta/analysis_provenance_release4.py",
        "corrected_sgta/merge_alignment_gate_release4.py",
        "corrected_sgta/run_alignment_cxr_release4.sh",
    ):
        path = project_root / name
        if not path.is_file():
            raise RuntimeError(f"missing release-4 identity artifact: {path}")
        identity[name] = sha256_file(path)
    return identity
