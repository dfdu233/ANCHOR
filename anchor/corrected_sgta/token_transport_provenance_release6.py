"""Final identity for the V3-preregistered token-alignment report."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.source_bank_v2 import sha256_file
from corrected_sgta.token_transport_provenance_release5 import (
    token_transport_identity_release5,
)


def token_transport_identity_release6(project_root: Path) -> dict:
    identity = token_transport_identity_release5(project_root)
    for name in (
        "corrected_sgta/token_transport_provenance_release6.py",
        "corrected_sgta/merge_token_transport_gate_release6.py",
        "corrected_sgta/run_token_transport_cxr_release6.sh",
    ):
        identity[name] = sha256_file(project_root / name)
    return identity
