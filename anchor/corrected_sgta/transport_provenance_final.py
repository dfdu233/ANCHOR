"""Final report identity for the feature-transport smoke."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.source_bank_v2 import sha256_file
from corrected_sgta.transport_provenance import transport_code_identity


def final_transport_identity(project_root: Path) -> dict:
    identity = transport_code_identity(project_root)
    for name in (
        "corrected_sgta/analyze_feature_transport.py",
        "corrected_sgta/transport_provenance_final.py",
        "corrected_sgta/merge_feature_transport_gate.py",
        "corrected_sgta/run_feature_transport_cxr_final.sh",
    ):
        identity[name] = sha256_file(project_root / name)
    return identity
