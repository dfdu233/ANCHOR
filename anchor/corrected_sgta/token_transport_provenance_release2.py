"""Final identity for token alignment with explicit inference parameters."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.source_bank_v2 import sha256_file
from corrected_sgta.token_transport_provenance import token_transport_identity


def token_transport_identity_release2(project_root: Path) -> dict:
    identity = token_transport_identity(project_root)
    for name in (
        "corrected_sgta/infer_token_transport_release2.py",
        "corrected_sgta/token_transport_provenance_release2.py",
        "corrected_sgta/merge_token_transport_gate_release2.py",
        "corrected_sgta/run_token_transport_cxr_release2.sh",
    ):
        identity[name] = sha256_file(project_root / name)
    return identity
