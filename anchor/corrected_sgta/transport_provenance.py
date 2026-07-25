"""Identity for the minimal visual-token source transport experiment."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.provenance_release2 import center_code_identity
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import sha256_file


def transport_code_identity(project_root: Path) -> dict:
    names = (
        "corrected_sgta/source_bank_v2.py",
        "corrected_sgta/source_bank_v3.py",
        "corrected_sgta/models.py",
        "corrected_sgta/models_surface.py",
        "corrected_sgta/models_alignment.py",
        "corrected_sgta/models_transport.py",
        "corrected_sgta/cache.py",
        "corrected_sgta/protocol.py",
        "corrected_sgta/protocol_v2.py",
        "corrected_sgta/infer_ce.py",
        "corrected_sgta/infer_feature_transport.py",
        "corrected_sgta/analyze_alignment_release4.py",
        "corrected_sgta/freeze_alignment_report_release4.py",
        "corrected_sgta/feature_transport_structure_audit.py",
        "corrected_sgta/merge_alignment_gate_release4.py",
        "corrected_sgta/transport_provenance.py",
        "corrected_sgta/run_feature_transport_cxr.sh",
        "refine-logs/EXPERIMENT_PLAN_AMENDMENT_V2.md",
    )
    return {name: sha256_file(project_root / name) for name in names}
