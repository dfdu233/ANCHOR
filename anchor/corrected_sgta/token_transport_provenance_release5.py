"""Complete identity for robust capped-simplex token alignment."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.source_bank_v2 import sha256_file
from corrected_sgta.token_transport_provenance_release4 import (
    token_transport_identity_release4,
)


def token_transport_identity_release5(project_root: Path) -> dict:
    identity = token_transport_identity_release4(project_root)
    for name in (
        "corrected_sgta/models_token_transport_release3.py",
        "corrected_sgta/infer_token_transport_release4.py",
        "corrected_sgta/token_transport_structure_audit_release3.py",
        "corrected_sgta/token_transport_provenance_release5.py",
        "corrected_sgta/merge_token_transport_gate_release5.py",
        "corrected_sgta/run_token_transport_cxr_release5.sh",
    ):
        identity[name] = sha256_file(project_root / name)
    return identity
