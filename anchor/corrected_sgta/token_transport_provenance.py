"""Complete code identity for source-guided token alignment."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.source_bank_v2 import sha256_file
from corrected_sgta.transport_provenance_complete import complete_transport_identity


def token_transport_identity(project_root: Path) -> dict:
    identity = complete_transport_identity(project_root)
    for name in (
        "corrected_sgta/models_token_transport.py",
        "corrected_sgta/token_transport_provenance.py",
        "corrected_sgta/infer_token_transport.py",
        "corrected_sgta/analyze_token_transport.py",
        "corrected_sgta/merge_token_transport_gate.py",
        "corrected_sgta/run_token_transport_cxr.sh",
        "refine-logs/EXPERIMENT_PLAN_AMENDMENT_V3.md",
    ):
        identity[name] = sha256_file(project_root / name)
    return identity
